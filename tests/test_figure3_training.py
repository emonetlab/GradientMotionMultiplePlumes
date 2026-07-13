from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.figure3_models import (  # noqa: E402
    make_dense_network,
    make_minimal_network,
)
from gradient_motion_panels.figure3_training import (  # noqa: E402
    TrainingConfig,
    _validate_pair,
    preprocess_batch,
    train_figure3_model,
    training_defaults,
)
import gradient_motion_panels.figure3_training as figure3_training  # noqa: E402


def _legacy_arrays(root: Path) -> dict[str, Path]:
    """Write small, balanced left-right-paired fixtures in the legacy layout."""

    rng = np.random.default_rng(2048)
    paths: dict[str, Path] = {}
    for role, half_count in (("train", 8), ("test", 4)):
        right = rng.uniform(5.5, 14.0, size=(half_count, 30)).astype(np.float32)
        left = rng.uniform(0.0, 4.5, size=(half_count, 30)).astype(np.float32)
        first = np.zeros((half_count, 30, 1, 2), dtype=np.float32)
        first[:, :, 0, 0] = right
        first[:, :, 0, 1] = left
        samples = np.concatenate((first, first[..., ::-1]), axis=0)
        labels = np.concatenate(
            (
                np.zeros(half_count, dtype=np.float32),
                np.ones(half_count, dtype=np.float32),
            )
        )
        data_path = root / f"{role}_data_R1.npy"
        labels_path = root / f"{role}_labels_R1.npy"
        np.save(data_path, samples)
        np.save(labels_path, labels)
        paths[f"{role}_data"] = data_path
        paths[f"{role}_labels"] = labels_path
    return paths


def _trainer_command(
    arrays: dict[str, Path],
    output_root: Path,
    *,
    models: tuple[str, ...],
    minimal_epochs: int = 1,
    dense_epochs: int = 1,
    resume: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_figure3_models.py"),
        "--train-data",
        str(arrays["train_data"]),
        "--train-labels",
        str(arrays["train_labels"]),
        "--test-data",
        str(arrays["test_data"]),
        "--test-labels",
        str(arrays["test_labels"]),
        "--plume",
        "smooth",
        "--models",
        *models,
        "--output-root",
        str(output_root),
        "--seed",
        "17",
        "--minimal-epochs",
        str(minimal_epochs),
        "--minimal-batch-size",
        "4",
        "--minimal-learning-rate",
        "0.001",
        "--dense-epochs",
        str(dense_epochs),
        "--dense-batch-size",
        "4",
        "--dense-learning-rate",
        "0.001",
        "--torch-threads",
        "1",
    ]
    command.append("--resume" if resume else "--overwrite")
    return command


def _run_trainer(
    arrays: dict[str, Path],
    output_root: Path,
    *,
    models: tuple[str, ...],
    minimal_epochs: int = 1,
    dense_epochs: int = 1,
    resume: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _trainer_command(
            arrays,
            output_root,
            models=models,
            minimal_epochs=minimal_epochs,
            dense_epochs=dense_epochs,
            resume=resume,
        ),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _load_raw_state(torch: Any, path: Path) -> dict[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        state = torch.load(path, map_location="cpu")
    assert isinstance(state, dict)
    assert "state_dict" not in state
    return state


def _assert_states_equal(
    torch: Any, left: dict[str, Any], right: dict[str, Any]
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_figure3_training_defaults_distinguish_source_and_methods() -> None:
    minimal = training_defaults("minimal", "smooth")
    assert (minimal.epochs, minimal.batch_size, minimal.learning_rate) == (
        300,
        100,
        1e-5,
    )
    dense_smooth = training_defaults("dense", "smooth")
    assert (
        dense_smooth.epochs,
        dense_smooth.batch_size,
        dense_smooth.learning_rate,
    ) == (500, 500, 1e-5)
    dense_complex = training_defaults("dense", "complex")
    assert (
        dense_complex.epochs,
        dense_complex.batch_size,
        dense_complex.learning_rate,
    ) == (500, 500, 1e-4)
    methods = training_defaults("minimal", "complex", profile="paper-methods")
    assert (methods.epochs, methods.batch_size, methods.learning_rate) == (
        500,
        500,
        1e-4,
    )


def test_figure3_training_preprocessing_and_input_validation() -> None:
    samples = np.zeros((2, 30, 1, 2), dtype=np.float32)
    samples[:, :, 0, 0] = 6.0
    samples[:, :, 0, 1] = 8.0
    samples[0, 0, 0, 0] = np.nan
    samples[0, 1, 0, 0] = 4.0
    samples[0, 2, 0, 0] = 5.0

    minimal = preprocess_batch(samples, model_kind="minimal", threshold=5.0)
    assert minimal.dtype == np.float32
    assert np.allclose(minimal[0, :3, 0, 0], 0.0)
    assert minimal[0, 3, 0, 0] == pytest.approx(np.log(2.0))
    assert minimal[0, 0, 0, 1] == pytest.approx(np.log(4.0))

    dense = preprocess_batch(samples, model_kind="dense", threshold=5.0)
    assert dense.shape == (2, 60)
    assert np.array_equal(dense[:, :30], minimal[:, :, 0, 0])
    assert np.array_equal(dense[:, 30:], minimal[:, :, 0, 1])

    rng = np.random.default_rng(9)
    source_like = rng.uniform(0.0, 20.0, size=(3, 30, 1, 2)).astype(np.float32)
    literal_archive = np.log(np.maximum(source_like - 5.0, 0.0) + 1.0).astype(
        np.float32
    )
    transformed = preprocess_batch(source_like, model_kind="minimal", threshold=5.0)
    assert np.array_equal(transformed, literal_archive)

    with np.errstate(invalid="ignore"):
        dense_without_sanitation = preprocess_batch(
            samples, model_kind="dense", threshold=5.0, sanitize=False
        )
    assert np.isnan(dense_without_sanitation[0, 0])

    with pytest.raises(ValueError, match="shape"):
        preprocess_batch(
            np.zeros((2, 30, 2), dtype=np.float32),
            model_kind="minimal",
            threshold=5.0,
        )
    with pytest.raises(ValueError, match="binary"):
        _validate_pair(
            samples,
            np.asarray([0.0, 0.5], dtype=np.float32),
            role="training",
            limit=None,
        )


def test_figure3_training_cli_is_deterministic_resumable_and_compatible(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    arrays = _legacy_arrays(tmp_path)
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"

    first = _run_trainer(arrays, run_a, models=("minimal", "dense"))
    second = _run_trainer(arrays, run_b, models=("minimal", "dense"))
    assert '"event": "training_complete"' in first.stdout
    assert '"event": "training_complete"' in second.stdout

    for model_kind, factory in (
        ("minimal", make_minimal_network),
        ("dense", make_dense_network),
    ):
        directory_a = run_a / model_kind / "smooth" / "R1"
        directory_b = run_b / model_kind / "smooth" / "R1"
        expected_names = {
            "model_init.pth",
            "model.pth",
            "train_loss.pth",
            "training_state.pth",
            "training_metadata.json",
        }
        assert expected_names.issubset(path.name for path in directory_a.iterdir())

        state_a = _load_raw_state(torch, directory_a / "model.pth")
        state_b = _load_raw_state(torch, directory_b / "model.pth")
        _assert_states_equal(torch, state_a, state_b)
        factory().load_state_dict(state_a, strict=True)

        metadata = json.loads((directory_a / "training_metadata.json").read_text())
        assert metadata["status"] == "complete"
        assert metadata["config"]["profile"] == "archived-source"
        assert metadata["config"]["seed"] == 17
        assert metadata["config"]["epochs"] == 1
        assert metadata["config"]["batch_size"] == 4
        assert metadata["effective_samples"] == {"training": 16, "test": 8}
        assert metadata["class_counts"]["training"] == {"0": 8, "1": 8}
        assert metadata["checkpoint"]["format"] == "raw CPU PyTorch state_dict"
        assert metadata["inputs"]["train_data"]["sha256"] is not None
        assert metadata["runtime"]["deterministic_algorithms"] is True
        assert metadata["test_metrics"]["roc_auc"] is not None
        if model_kind == "dense":
            units = metadata["suggested_dense_units_by_weight_norm"]
            assert len(units) == 2
            assert len(set(units)) == 2

        training_state = torch.load(
            directory_a / "training_state.pth",
            map_location="cpu",
            weights_only=False,
        )
        environment = training_state["fingerprint"]["resume_environment"]
        assert environment["resolved_device"] == "cpu"
        assert environment["torch"] == str(torch.__version__)
        assert training_state["model_init_sha256"] == metadata["model_init"]["sha256"]

    # A resumable state is not enough if its original initialization artifact
    # has disappeared; reject this before doing more training.
    missing_init = run_b / "minimal" / "smooth" / "R1" / "model_init.pth"
    missing_init.unlink()
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_trainer(
            arrays,
            run_b,
            models=("minimal",),
            minimal_epochs=2,
            resume=True,
        )
    assert (
        "Cannot resume without the saved state and initialization" in error.value.stderr
    )

    # Extending a one-epoch run must match an uninterrupted two-epoch run.
    shutil.rmtree(run_a / "dense")
    _run_trainer(
        arrays,
        run_a,
        models=("minimal", "dense"),
        minimal_epochs=2,
        resume=True,
    )
    assert (run_a / "dense" / "smooth" / "R1" / "model.pth").exists()
    uninterrupted = tmp_path / "uninterrupted"
    _run_trainer(
        arrays,
        uninterrupted,
        models=("minimal",),
        minimal_epochs=2,
    )
    resumed_dir = run_a / "minimal" / "smooth" / "R1"
    uninterrupted_dir = uninterrupted / "minimal" / "smooth" / "R1"
    resumed = _load_raw_state(torch, resumed_dir / "model.pth")
    direct = _load_raw_state(torch, uninterrupted_dir / "model.pth")
    _assert_states_equal(torch, resumed, direct)
    resumed_metadata = json.loads((resumed_dir / "training_metadata.json").read_text())
    direct_metadata = json.loads(
        (uninterrupted_dir / "training_metadata.json").read_text()
    )
    assert resumed_metadata["config"]["epochs"] == 2
    assert resumed_metadata["train_loss"] == pytest.approx(
        direct_metadata["train_loss"], rel=0, abs=0
    )

    # The new checkpoints can feed the existing summary generator, including
    # non-publication dense-unit selections appropriate for newly trained models.
    minimal_path = resumed_dir / "model.pth"
    dense_path = run_a / "dense" / "smooth" / "R1" / "model.pth"
    summary = tmp_path / "figure3_summary.npz"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_figure3_summary.py"),
            "--minimal-smooth",
            str(minimal_path),
            "--minimal-complex",
            str(minimal_path),
            "--minimal-smooth-probe",
            str(minimal_path),
            "--minimal-complex-probe",
            str(minimal_path),
            "--dense-smooth",
            str(dense_path),
            "--dense-complex",
            str(dense_path),
            "--dense-smooth-units",
            "0",
            "2",
            "--dense-complex-units",
            "1",
            "3",
            "--motion-samples",
            "2",
            "--batch-size",
            "2",
            "--output",
            str(summary),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dense_state = _load_raw_state(torch, dense_path)
    expected_filters = (
        dense_state["dense_layers.0.weight"][[0, 2]].numpy().reshape(2, 2, 30)
    )
    with np.load(summary, allow_pickle=False) as bundle:
        assert np.allclose(bundle["dense_smooth_filters"], expected_filters)
        metadata = json.loads(str(bundle["metadata_json"]))
        assert metadata["dense_units"] == {
            "smooth": [0, 2],
            "complex": [1, 3],
        }


def test_failed_resume_keeps_prior_complete_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    arrays = _legacy_arrays(tmp_path)
    output_root = tmp_path / "failure_run"
    _run_trainer(arrays, output_root, models=("minimal",))
    output_dir = output_root / "minimal" / "smooth" / "R1"
    model_before = (output_dir / "model.pth").read_bytes()
    loss_before = (output_dir / "train_loss.pth").read_bytes()
    metadata_before = (output_dir / "training_metadata.json").read_text()

    def fail_evaluation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("forced evaluation failure")

    original_evaluate = figure3_training.evaluate_model
    monkeypatch.setattr(figure3_training, "evaluate_model", fail_evaluation)
    config = TrainingConfig(
        model_kind="minimal",
        plume="smooth",
        profile="archived-source",
        epochs=2,
        batch_size=4,
        learning_rate=0.001,
        seed=17,
        repeat=1,
        device="cpu",
        torch_threads=1,
    )
    with pytest.raises(RuntimeError, match="forced evaluation failure"):
        train_figure3_model(
            arrays["train_data"],
            arrays["train_labels"],
            output_dir,
            config,
            test_data_path=arrays["test_data"],
            test_labels_path=arrays["test_labels"],
            hash_inputs=True,
            resume=True,
        )

    assert (output_dir / "model.pth").read_bytes() == model_before
    assert (output_dir / "train_loss.pth").read_bytes() == loss_before
    assert (output_dir / "training_metadata.json").read_text() == metadata_before
    training_state = torch.load(
        output_dir / "training_state.pth", map_location="cpu", weights_only=False
    )
    assert training_state["completed_epochs"] == 2
    assert not list(output_dir.glob(".*.tmp"))

    # If publication fails after replacing model.pth, the old completion marker
    # must already be gone; a same-epoch resume can then republish coherently.
    monkeypatch.setattr(figure3_training, "evaluate_model", original_evaluate)
    original_replace = figure3_training.os.replace

    def fail_loss_publication(source: Any, destination: Any) -> None:
        if Path(destination).name == "train_loss.pth":
            raise OSError("forced publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(figure3_training.os, "replace", fail_loss_publication)
    with pytest.raises(OSError, match="forced publication failure"):
        train_figure3_model(
            arrays["train_data"],
            arrays["train_labels"],
            output_dir,
            config,
            test_data_path=arrays["test_data"],
            test_labels_path=arrays["test_labels"],
            hash_inputs=True,
            resume=True,
        )
    assert not (output_dir / "training_metadata.json").exists()
    assert not list(output_dir.glob(".*.tmp"))

    monkeypatch.setattr(figure3_training.os, "replace", original_replace)
    repaired = train_figure3_model(
        arrays["train_data"],
        arrays["train_labels"],
        output_dir,
        config,
        test_data_path=arrays["test_data"],
        test_labels_path=arrays["test_labels"],
        hash_inputs=True,
        resume=True,
    )
    assert repaired["checkpoint"]["sha256"] == figure3_training.sha256_file(
        output_dir / "model.pth"
    )


def test_concurrent_trainers_have_one_writer(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    arrays = _legacy_arrays(tmp_path)
    output_root = tmp_path / "concurrent_run"
    command = _trainer_command(
        arrays,
        output_root,
        models=("minimal",),
        minimal_epochs=20,
    )
    processes = [
        subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=60) for process in processes]
    return_codes = [process.returncode for process in processes]
    assert sum(code == 0 for code in return_codes) == 1
    loser_index = next(index for index, code in enumerate(return_codes) if code != 0)
    assert "Another Figure 3 trainer is already using" in results[loser_index][1]

    output_dir = output_root / "minimal" / "smooth" / "R1"
    metadata = json.loads((output_dir / "training_metadata.json").read_text())
    state = _load_raw_state(torch, output_dir / "model.pth")
    make_minimal_network().load_state_dict(state, strict=True)
    assert metadata["training"]["completed_epochs"] == 20
    assert not list(output_dir.glob(".*.tmp"))

    # The winner released its lock and left a coherent resumable state.
    _run_trainer(
        arrays,
        output_root,
        models=("minimal",),
        minimal_epochs=21,
        resume=True,
    )
