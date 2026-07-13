#!/usr/bin/env python3
"""Extract Figure 3 filters and probe responses from compatible checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.figure3_models import (  # noqa: E402
    class_one_probability,
    dense_input,
    gradient_probe,
    iter_motion_probes,
    make_dense_network,
    make_minimal_network,
)


def checkpoint_path(value: Path) -> Path:
    return value / "model.pth" if value.is_dir() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minimal-smooth",
        type=Path,
        required=True,
        help="User-supplied R1 filter checkpoint.",
    )
    parser.add_argument(
        "--minimal-complex",
        type=Path,
        required=True,
        help="User-supplied R1 filter checkpoint.",
    )
    parser.add_argument(
        "--minimal-smooth-probe",
        type=Path,
        required=True,
        help="User-supplied R2 probe checkpoint.",
    )
    parser.add_argument(
        "--minimal-complex-probe",
        type=Path,
        required=True,
        help="User-supplied R2 probe checkpoint.",
    )
    parser.add_argument(
        "--dense-smooth", type=Path, required=True, help="R1 dense checkpoint."
    )
    parser.add_argument(
        "--dense-complex", type=Path, required=True, help="R1 dense checkpoint."
    )
    parser.add_argument(
        "--dense-smooth-units",
        type=int,
        nargs=2,
        metavar=("UNIT_A", "UNIT_B"),
        default=(3, 11),
        help="First-layer units to display (publication-checkpoint default: 3 11).",
    )
    parser.add_argument(
        "--dense-complex-units",
        type=int,
        nargs=2,
        metavar=("UNIT_A", "UNIT_B"),
        default=(1, 12),
        help="First-layer units to display (publication-checkpoint default: 1 12).",
    )
    parser.add_argument("--motion-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    import torch

    path = checkpoint_path(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch fallback
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} does not contain a state dictionary.")
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_model(path: Path, kind: str) -> tuple[Any, dict[str, Any]]:
    model = make_minimal_network() if kind == "minimal" else make_dense_network()
    state = load_state(path)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, state


def predict(
    model: Any, samples: np.ndarray, *, dense: bool, batch_size: int
) -> np.ndarray:
    import torch

    values = dense_input(samples) if dense else np.asarray(samples, dtype=np.float32)
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            logits = model(
                torch.as_tensor(values[start : start + batch_size], dtype=torch.float32)
            )
            probabilities.append(class_one_probability(logits))
    return np.concatenate(probabilities)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.motion_samples < 1:
        raise ValueError("--motion-samples must be >= 1.")
    for name, units in (
        ("--dense-smooth-units", args.dense_smooth_units),
        ("--dense-complex-units", args.dense_complex_units),
    ):
        if len(set(units)) != 2 or any(unit < 0 or unit >= 20 for unit in units):
            raise ValueError(f"{name} requires two distinct indices from 0 through 19.")
    paths = {
        "minimal_smooth": checkpoint_path(args.minimal_smooth),
        "minimal_complex": checkpoint_path(args.minimal_complex),
        "minimal_smooth_probe": checkpoint_path(args.minimal_smooth_probe),
        "minimal_complex_probe": checkpoint_path(args.minimal_complex_probe),
        "dense_smooth": checkpoint_path(args.dense_smooth),
        "dense_complex": checkpoint_path(args.dense_complex),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 3 checkpoint(s): " + ", ".join(missing))

    minimal_smooth, minimal_smooth_state = load_model(
        paths["minimal_smooth"], "minimal"
    )
    minimal_complex, minimal_complex_state = load_model(
        paths["minimal_complex"], "minimal"
    )
    minimal_smooth_probe, _ = load_model(paths["minimal_smooth_probe"], "minimal")
    minimal_complex_probe, _ = load_model(paths["minimal_complex_probe"], "minimal")
    dense_smooth, dense_smooth_state = load_model(paths["dense_smooth"], "dense")
    dense_complex, dense_complex_state = load_model(paths["dense_complex"], "dense")

    differences, gradient_samples = gradient_probe()
    arrays: dict[str, np.ndarray] = {
        "time_seconds": np.linspace(-0.5, 0.0, 30, dtype=np.float32),
        "minimal_smooth_filters": minimal_smooth_state["first_layer.weight"][0, :, 0, :]
        .detach()
        .cpu()
        .numpy()
        .T,
        "minimal_complex_filters": minimal_complex_state["first_layer.weight"][
            0, :, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
        .T,
        "dense_smooth_filters": dense_smooth_state["dense_layers.0.weight"][
            list(args.dense_smooth_units)
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(2, 2, 30),
        "dense_complex_filters": dense_complex_state["dense_layers.0.weight"][
            list(args.dense_complex_units)
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(2, 2, 30),
        "gradient_difference": differences,
        "minimal_gradient_smooth": predict(
            minimal_smooth_probe,
            gradient_samples,
            dense=False,
            batch_size=args.batch_size,
        ),
        "minimal_gradient_complex": predict(
            minimal_complex_probe,
            gradient_samples,
            dense=False,
            batch_size=args.batch_size,
        ),
        "dense_gradient_smooth": predict(
            dense_smooth, gradient_samples, dense=True, batch_size=args.batch_size
        ),
        "dense_gradient_complex": predict(
            dense_complex, gradient_samples, dense=True, batch_size=args.batch_size
        ),
    }

    shifts: list[float] = []
    motion_values = {
        "minimal_motion_smooth": [],
        "minimal_motion_complex": [],
        "dense_motion_smooth": [],
        "dense_motion_complex": [],
    }
    for shift, samples in iter_motion_probes(
        sample_count=args.motion_samples, seed=args.seed
    ):
        shifts.append(shift)
        motion_values["minimal_motion_smooth"].append(
            float(
                predict(
                    minimal_smooth_probe,
                    samples,
                    dense=False,
                    batch_size=args.batch_size,
                ).mean()
            )
        )
        motion_values["minimal_motion_complex"].append(
            float(
                predict(
                    minimal_complex_probe,
                    samples,
                    dense=False,
                    batch_size=args.batch_size,
                ).mean()
            )
        )
        motion_values["dense_motion_smooth"].append(
            float(
                predict(
                    dense_smooth, samples, dense=True, batch_size=args.batch_size
                ).mean()
            )
        )
        motion_values["dense_motion_complex"].append(
            float(
                predict(
                    dense_complex, samples, dense=True, batch_size=args.batch_size
                ).mean()
            )
        )
    arrays["motion_shift_seconds"] = np.asarray(shifts, dtype=np.float32)
    arrays.update(
        {
            name: np.asarray(values, dtype=np.float32)
            for name, values in motion_values.items()
        }
    )

    metadata = {
        "generator": "scripts/generate_figure3_summary.py",
        "seed": args.seed,
        "motion_samples_per_shift": args.motion_samples,
        "batch_size": args.batch_size,
        "dense_units": {
            "smooth": list(args.dense_smooth_units),
            "complex": list(args.dense_complex_units),
        },
        "checkpoint_note": "MNM filter panels use R1, while the preprint's MNM probe panels used R2.",
        "checkpoint_authentication": "SHA-256 values identify supplied files but do not authenticate their publication role.",
        "checkpoints": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output.resolve()), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
