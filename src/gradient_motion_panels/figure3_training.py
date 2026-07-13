"""Memory-bounded training for checkpoint-compatible Figure 3 models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any, Callable, Iterator
from uuid import uuid4

import numpy as np

from .figure3_models import make_dense_network, make_minimal_network


SOURCE_COMMIT = "659d0c3c34a8ab0f05abd76b38756debd4ea9214"
MODEL_KINDS = ("minimal", "dense")
PLUMES = ("smooth", "complex")
PROFILES = ("archived-source", "paper-methods")


@dataclass(frozen=True)
class TrainingDefaults:
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    model_kind: str
    plume: str
    profile: str
    epochs: int
    batch_size: int
    learning_rate: float
    threshold: float = 5.0
    seed: int = 0
    repeat: int = 1
    shuffle: bool = False
    device: str = "cpu"
    torch_threads: int | None = None
    max_train_samples: int | None = None
    max_test_samples: int | None = None

    def __post_init__(self) -> None:
        if self.model_kind not in MODEL_KINDS:
            raise ValueError(f"model_kind must be one of {MODEL_KINDS}.")
        if self.plume not in PLUMES:
            raise ValueError(f"plume must be one of {PLUMES}.")
        if self.profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}.")
        if (
            self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0
            or not np.isfinite(self.learning_rate)
        ):
            raise ValueError("epochs, batch_size, and learning_rate must be positive.")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite.")
        if self.repeat < 1 or self.seed < 0:
            raise ValueError("repeat must be >= 1 and seed must be non-negative.")
        if self.torch_threads is not None and self.torch_threads < 1:
            raise ValueError("torch_threads must be >= 1 when supplied.")
        for name in ("max_train_samples", "max_test_samples"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 when supplied.")


def training_defaults(
    model_kind: str, plume: str, *, profile: str = "archived-source"
) -> TrainingDefaults:
    """Resolve the archived executable or paper-Methods hyperparameters."""

    if model_kind not in MODEL_KINDS or plume not in PLUMES:
        raise ValueError(f"Expected model in {MODEL_KINDS} and plume in {PLUMES}.")
    if profile == "paper-methods":
        return TrainingDefaults(epochs=500, batch_size=500, learning_rate=1e-4)
    if profile != "archived-source":
        raise ValueError(f"profile must be one of {PROFILES}.")
    if model_kind == "minimal":
        return TrainingDefaults(epochs=300, batch_size=100, learning_rate=1e-5)
    learning_rate = 1e-5 if plume == "smooth" else 1e-4
    return TrainingDefaults(epochs=500, batch_size=500, learning_rate=learning_rate)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_legacy_array(path: str | Path) -> np.memmap:
    """Open a legacy NPY without loading it into memory."""

    path = Path(path)
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Legacy training inputs must be .npy files; got {path}.")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap):
        raise TypeError(f"Expected a memory-mapped NPY array for {path}.")
    return array


def preprocess_batch(
    samples: np.ndarray,
    *,
    model_kind: str,
    threshold: float,
    sanitize: bool = True,
) -> np.ndarray:
    """Apply the archived log-threshold transform and model-specific layout."""

    values = np.array(samples, copy=True)
    if values.ndim != 4 or values.shape[1:] != (30, 1, 2):
        raise ValueError(
            f"Figure 3 samples must have shape (N, 30, 1, 2); got {values.shape}."
        )
    if sanitize:
        np.nan_to_num(values, copy=False)
    # Preserve the source expression: np.log1p differs in its last-bit rounding.
    values = np.log(np.maximum(values - threshold, 0.0) + 1.0).astype(
        np.float32, copy=False
    )
    if model_kind == "minimal":
        return np.ascontiguousarray(values)
    if model_kind != "dense":
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}.")
    return np.ascontiguousarray(
        np.concatenate((values[:, :, 0, 0], values[:, :, 0, 1]), axis=1)
    )


def _effective_count(array: np.ndarray, limit: int | None) -> int:
    return len(array) if limit is None else min(len(array), limit)


def _validate_pair(
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    role: str,
    limit: int | None,
) -> tuple[int, dict[str, int]]:
    if samples.ndim != 4 or samples.shape[1:] != (30, 1, 2):
        raise ValueError(
            f"{role} samples must have shape (N, 30, 1, 2); got {samples.shape}."
        )
    if labels.ndim not in (1, 2) or (labels.ndim == 2 and labels.shape[1:] != (1,)):
        raise ValueError(
            f"{role} labels must have shape (N,) or (N, 1); got {labels.shape}."
        )
    if len(samples) != len(labels):
        raise ValueError(
            f"{role} sample/label counts differ: {len(samples)} versus {len(labels)}."
        )
    count = _effective_count(samples, limit)
    if count < 1:
        raise ValueError(f"{role} arrays are empty.")
    counts = {"0": 0, "1": 0}
    for start in range(0, count, 1_000_000):
        values = np.asarray(labels[start : min(start + 1_000_000, count)]).reshape(-1)
        valid = (values == 0) | (values == 1)
        if not bool(np.all(valid)):
            invalid = np.unique(values[~valid])[:5].tolist()
            raise ValueError(f"{role} labels must be binary 0/1; found {invalid}.")
        counts["0"] += int(np.count_nonzero(values == 0))
        counts["1"] += int(np.count_nonzero(values == 1))
    if role == "training" and min(counts.values()) == 0:
        raise ValueError("Training labels must contain both classes.")
    return count, counts


def iter_batches(
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    count: int,
    batch_size: int,
    model_kind: str,
    threshold: float,
    sanitize: bool = True,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield bounded NumPy batches, using contiguous mmap slices when unshuffled."""

    order = rng.permutation(count) if shuffle else None
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        selection: slice | np.ndarray = (
            slice(start, stop) if order is None else order[start:stop]
        )
        batch_samples = preprocess_batch(
            samples[selection],
            model_kind=model_kind,
            threshold=threshold,
            sanitize=sanitize,
        )
        batch_labels = (
            np.asarray(labels[selection]).reshape(-1).astype(np.int64, copy=True)
        )
        yield batch_samples, batch_labels


def reset_parameters_like_archived_source(model: Any) -> None:
    """Repeat the source's construct-then-reset initialization pass."""

    def reset_children(module: Any) -> None:
        for child in module.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()

    model.apply(reset_children)


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")


def _torch_save_temporary(payload: Any, path: Path) -> Path:
    import torch

    temporary = _temporary_sibling(path)
    try:
        torch.save(payload, temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _json_save_temporary(payload: dict[str, Any], path: Path) -> Path:
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = _torch_save_temporary(payload, path)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_output_lock(output_dir: Path) -> Iterator[None]:
    """Hold a non-blocking, crash-released lock for one training directory."""

    lock_path = output_dir / ".figure3_training.lock"
    handle = lock_path.open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - exercised on POSIX CI
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                f"Another Figure 3 trainer is already using {output_dir}."
            ) from error
        locked = True
        yield
    finally:
        try:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - exercised on POSIX CI
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _lock_training_output(
    function: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def locked_call(
        train_data_path: str | Path,
        train_labels_path: str | Path,
        output_dir: str | Path,
        config: TrainingConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        resolved_output = Path(output_dir).resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)
        with _exclusive_output_lock(resolved_output):
            return function(
                train_data_path,
                train_labels_path,
                resolved_output,
                config,
                **kwargs,
            )

    return locked_call


def _move_optimizer_state(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if hasattr(value, "to"):
                state[name] = value.to(device)


def _resolve_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if torch.cuda.is_available():
            requested = "cuda"
        elif mps_available:
            requested = "mps"
        else:
            requested = "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is unavailable.")
    return torch.device(requested)


def _set_seed(seed: int, torch_threads: int | None) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _input_record(path: Path, array: np.ndarray, *, hash_input: bool) -> dict[str, Any]:
    return {
        **_array_identity(path, array),
        "sha256": sha256_file(path) if hash_input else None,
    }


def evaluate_model(
    model: Any,
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    count: int,
    config: TrainingConfig,
    device: Any,
) -> dict[str, Any]:
    """Compute final CE, accuracy, and class-1 ROC AUC in bounded batches."""

    import torch
    from sklearn.metrics import roc_auc_score

    criterion = torch.nn.CrossEntropyLoss()
    probabilities = np.empty(count, dtype=np.float64)
    targets_all = np.empty(count, dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    loss_sum = 0.0
    correct = 0
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch_samples, batch_labels in iter_batches(
            samples,
            labels,
            count=count,
            batch_size=config.batch_size,
            model_kind=config.model_kind,
            threshold=config.threshold,
            sanitize=True,
            shuffle=False,
            rng=rng,
        ):
            inputs = torch.from_numpy(batch_samples).to(device)
            targets = torch.from_numpy(batch_labels).to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_count = len(batch_labels)
            loss_sum += float(loss.item()) * batch_count
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            probabilities[offset : offset + batch_count] = (
                torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            )
            targets_all[offset : offset + batch_count] = batch_labels
            offset += batch_count
    auc = (
        float(roc_auc_score(targets_all, probabilities))
        if len(np.unique(targets_all)) == 2
        else None
    )
    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "roc_auc": auc,
        "samples": count,
    }


def _array_identity(path: Path, array: np.ndarray) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _resume_fingerprint(
    config: TrainingConfig,
    *,
    resolved_device: str,
    training_inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    import torch

    values = asdict(config)
    values.pop("epochs")
    values.pop("max_test_samples")
    values["resume_environment"] = {
        "resolved_device": resolved_device,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    values["training_inputs"] = training_inputs
    return values


def _suggested_dense_units(state_dict: dict[str, Any]) -> list[int]:
    weights = state_dict["dense_layers.0.weight"].detach().cpu().numpy()
    norms = np.linalg.norm(weights, axis=1)
    return [int(value) for value in np.argsort(norms)[-2:][::-1]]


@_lock_training_output
def train_figure3_model(
    train_data_path: str | Path,
    train_labels_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    *,
    test_data_path: str | Path | None = None,
    test_labels_path: str | Path | None = None,
    hash_inputs: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train one model and write a raw state dict plus auditable run metadata."""

    import torch

    invocation_started = time.perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    train_data_path = Path(train_data_path).resolve()
    train_labels_path = Path(train_labels_path).resolve()
    output_dir = Path(output_dir).resolve()
    if (test_data_path is None) != (test_labels_path is None):
        raise ValueError("Pass both test_data_path and test_labels_path, or neither.")
    resolved_test_data = Path(test_data_path).resolve() if test_data_path else None
    resolved_test_labels = (
        Path(test_labels_path).resolve() if test_labels_path else None
    )

    train_samples = open_legacy_array(train_data_path)
    train_labels = open_legacy_array(train_labels_path)
    train_count, train_class_counts = _validate_pair(
        train_samples,
        train_labels,
        role="training",
        limit=config.max_train_samples,
    )
    test_samples = test_labels = None
    test_count = None
    test_class_counts = None
    if resolved_test_data and resolved_test_labels:
        test_samples = open_legacy_array(resolved_test_data)
        test_labels = open_legacy_array(resolved_test_labels)
        test_count, test_class_counts = _validate_pair(
            test_samples,
            test_labels,
            role="test",
            limit=config.max_test_samples,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_init": output_dir / "model_init.pth",
        "model": output_dir / "model.pth",
        "loss": output_dir / "train_loss.pth",
        "state": output_dir / "training_state.pth",
        "metadata": output_dir / "training_metadata.json",
    }
    if resume:
        missing_resume = [
            str(paths[name])
            for name in ("state", "model_init")
            if not paths[name].exists()
        ]
        if missing_resume:
            raise FileNotFoundError(
                "Cannot resume without the saved state and initialization: "
                + ", ".join(missing_resume)
            )
    if not resume and not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Training outputs already exist; pass overwrite=True or resume=True: "
                + ", ".join(existing)
            )
    if not resume and overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)

    input_arrays = {
        "train_data": (train_data_path, train_samples),
        "train_labels": (train_labels_path, train_labels),
    }
    if (
        resolved_test_data is not None
        and resolved_test_labels is not None
        and test_samples is not None
        and test_labels is not None
    ):
        input_arrays["test_data"] = (resolved_test_data, test_samples)
        input_arrays["test_labels"] = (resolved_test_labels, test_labels)
    input_identities = {
        role: _array_identity(path, array)
        for role, (path, array) in input_arrays.items()
    }
    input_records = {
        role: _input_record(path, array, hash_input=hash_inputs)
        for role, (path, array) in input_arrays.items()
    }
    device = _resolve_device(config.device)
    resume_fingerprint = _resume_fingerprint(
        config,
        resolved_device=str(device),
        training_inputs={
            "data": input_records["train_data"],
            "labels": input_records["train_labels"],
        },
    )

    _set_seed(config.seed, config.torch_threads)
    model = (
        make_minimal_network()
        if config.model_kind == "minimal"
        else make_dense_network()
    )
    reset_parameters_like_archived_source(model)
    initial_state = _cpu_state_dict(model)
    if not resume:
        _atomic_torch_save(initial_state, paths["model_init"])
    initial_checkpoint_sha256 = sha256_file(paths["model_init"])

    start_epoch = 0
    train_losses: list[float] = []
    rng = np.random.default_rng(config.seed)
    if resume:
        try:
            training_state = torch.load(
                paths["state"], map_location="cpu", weights_only=False
            )
        except TypeError:  # pragma: no cover - old torch fallback
            training_state = torch.load(paths["state"], map_location="cpu")
        if training_state.get("fingerprint") != resume_fingerprint:
            raise ValueError("Resume configuration does not match training_state.pth.")
        if training_state.get("model_init_sha256") != initial_checkpoint_sha256:
            raise ValueError(
                "model_init.pth does not match the resumable training state."
            )
        model.load_state_dict(training_state["model_state_dict"], strict=True)
        start_epoch = int(training_state["completed_epochs"])
        if start_epoch > config.epochs:
            raise ValueError(
                "Resume target epochs cannot be less than completed epochs "
                f"({config.epochs} < {start_epoch})."
            )
        train_losses = [float(value) for value in training_state["train_losses"]]
        if len(train_losses) != start_epoch:
            raise ValueError("Resume loss history does not match completed_epochs.")
        rng.bit_generator.state = training_state["numpy_rng_state"]
        torch.set_rng_state(training_state["torch_rng_state"])
        if torch.cuda.is_available() and training_state.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(training_state["cuda_rng_states"])

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    if resume:
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        epoch_started = time.perf_counter()
        for batch_samples, batch_labels in iter_batches(
            train_samples,
            train_labels,
            count=train_count,
            batch_size=config.batch_size,
            model_kind=config.model_kind,
            threshold=config.threshold,
            sanitize=config.model_kind == "minimal",
            shuffle=config.shuffle,
            rng=rng,
        ):
            inputs = torch.from_numpy(batch_samples).to(device)
            targets = torch.from_numpy(batch_labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch + 1}, batch {batches + 1}."
                )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        epoch_loss /= batches
        train_losses.append(epoch_loss)
        state_payload = {
            "completed_epochs": epoch + 1,
            "model_state_dict": _cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "model_init_sha256": initial_checkpoint_sha256,
            "fingerprint": resume_fingerprint,
        }
        _atomic_torch_save(state_payload, paths["state"])
        if progress:
            progress(
                {
                    "model": config.model_kind,
                    "plume": config.plume,
                    "repeat": config.repeat,
                    "epoch": epoch + 1,
                    "epochs": config.epochs,
                    "loss": epoch_loss,
                    "seconds": time.perf_counter() - epoch_started,
                }
            )

    final_state = _cpu_state_dict(model)
    test_metrics = None
    if test_samples is not None and test_labels is not None and test_count is not None:
        test_metrics = evaluate_model(
            model,
            test_samples,
            test_labels,
            count=test_count,
            config=config,
            device=device,
        )

    changed_inputs = []
    for role, (path, array) in input_arrays.items():
        identity_changed = _array_identity(path, array) != input_identities[role]
        expected_sha256 = input_records[role]["sha256"]
        hash_changed = (
            expected_sha256 is not None and sha256_file(path) != expected_sha256
        )
        if identity_changed or hash_changed:
            changed_inputs.append(role)
    if changed_inputs:
        raise RuntimeError(
            "Input files changed while training: " + ", ".join(changed_inputs)
        )
    metadata: dict[str, Any] = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "trainer": "src/gradient_motion_panels/figure3_training.py",
        "config": asdict(config),
        "resolved_device": str(device),
        "optimizer": {
            "name": "Adam",
            "learning_rate": config.learning_rate,
            "weight_decay": 0.0,
            "other_parameters": "PyTorch defaults",
        },
        "loss": "CrossEntropyLoss(mean, unweighted)",
        "training": {
            "completed_epochs": len(train_losses),
            "resumed_from_epoch": start_epoch,
            "shuffle": config.shuffle,
            "drop_last": False,
            "workers": 0,
            "scheduler": None,
            "checkpoint_selection": "final epoch",
            "evaluation": (
                "final-only; archived dense per-epoch validation did not affect "
                "optimization or checkpoint selection"
            ),
        },
        "input_contract": {
            "sample_axes": ["sample", "time", "singleton", "antenna"],
            "sample_shape": [30, 1, 2],
            "antenna_channels": ["right", "left"],
            "dense_layout": "30 right samples followed by 30 left samples",
            "preprocessing": "log(max(I - threshold, 0) + 1)",
            "training_nan_handling": (
                "nan_to_num" if config.model_kind == "minimal" else "none"
            ),
            "evaluation_nan_handling": "nan_to_num",
        },
        "inputs": input_records,
        "effective_samples": {
            "training": train_count,
            "test": test_count,
        },
        "class_counts": {
            "training": train_class_counts,
            "test": test_class_counts,
        },
        "train_loss": train_losses,
        "test_metrics": test_metrics,
        "checkpoint": {
            "path": str(paths["model"]),
            "sha256": None,
            "format": "raw CPU PyTorch state_dict",
        },
        "model_init": {
            "path": str(paths["model_init"]),
            "sha256": initial_checkpoint_sha256,
            "initialization": "constructed, then reset once like archived source",
        },
        "suggested_dense_units_by_weight_norm": (
            _suggested_dense_units(final_state)
            if config.model_kind == "dense"
            else None
        ),
        "runtime": {
            "seconds_this_invocation": None,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "torch_threads": torch.get_num_threads(),
            "deterministic_seed": config.seed,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_deterministic": (
                bool(torch.backends.cudnn.deterministic)
                if hasattr(torch.backends, "cudnn")
                else None
            ),
        },
        "provenance_note": (
            "This is a seeded, checkpoint-compatible reimplementation trained from "
            "supplied arrays; it is not a recovered publication initialization."
        ),
    }
    model_temporary = loss_temporary = metadata_temporary = None
    try:
        model_temporary = _torch_save_temporary(final_state, paths["model"])
        loss_temporary = _torch_save_temporary(train_losses, paths["loss"])
        metadata["checkpoint"]["sha256"] = sha256_file(model_temporary)
        metadata["runtime"]["seconds_this_invocation"] = (
            time.perf_counter() - invocation_started
        )
        metadata_temporary = _json_save_temporary(metadata, paths["metadata"])

        # training_metadata.json is the commit marker for a coherent final
        # artifact set. Remove the old marker, replace data files, then publish
        # the new marker last so a failed publication cannot look complete.
        paths["metadata"].unlink(missing_ok=True)
        os.replace(model_temporary, paths["model"])
        os.replace(loss_temporary, paths["loss"])
        os.replace(metadata_temporary, paths["metadata"])
    finally:
        for temporary in (model_temporary, loss_temporary, metadata_temporary):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return metadata
