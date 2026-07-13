#!/usr/bin/env python3
"""Train checkpoint-compatible Figure 3 minimal and dense models."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.figure3_training import (  # noqa: E402
    MODEL_KINDS,
    PLUMES,
    PROFILES,
    TrainingConfig,
    train_figure3_model,
    training_defaults,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--plume", choices=PLUMES, required=True)
    parser.add_argument(
        "--models",
        choices=MODEL_KINDS,
        nargs="+",
        default=list(MODEL_KINDS),
        help="Model kinds to train (default: minimal dense).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root for <model>/<plume>/R<repeat>/ training artifacts.",
    )
    parser.add_argument("--profile", choices=PROFILES, default="archived-source")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        help="Initialization seed (default: repeat - 1).",
    )
    parser.add_argument("--minimal-epochs", type=int)
    parser.add_argument("--minimal-batch-size", type=int)
    parser.add_argument("--minimal-learning-rate", type=float)
    parser.add_argument("--dense-epochs", type=int)
    parser.add_argument("--dense-batch-size", type=int)
    parser.add_argument("--dense-learning-rate", type=float)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle each epoch; the archived source left samples in file order.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device such as cpu, cuda, cuda:0, mps, or auto.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads per model (default: 1 for repeatability).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Use the first N training samples (intended for smoke tests).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        help="Use the first N test samples (intended for smoke tests).",
    )
    hash_mode = parser.add_mutually_exclusive_group()
    hash_mode.add_argument(
        "--hash-inputs",
        dest="hash_inputs",
        action="store_true",
        help="Record SHA-256 for every input NPY (default).",
    )
    hash_mode.add_argument(
        "--no-hash-inputs",
        dest="hash_inputs",
        action="store_false",
        help="Skip input SHA-256 calculation for a faster startup.",
    )
    parser.set_defaults(hash_inputs=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing states and start requested models not yet begun.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifacts and start from initialization.",
    )
    return parser.parse_args()


def _override(args: argparse.Namespace, model_kind: str, name: str) -> Any:
    return getattr(args, f"{model_kind}_{name}")


def resolve_config(args: argparse.Namespace, model_kind: str) -> TrainingConfig:
    defaults = training_defaults(model_kind, args.plume, profile=args.profile)
    config = TrainingConfig(
        model_kind=model_kind,
        plume=args.plume,
        profile=args.profile,
        epochs=defaults.epochs,
        batch_size=defaults.batch_size,
        learning_rate=defaults.learning_rate,
        threshold=args.threshold,
        seed=args.repeat - 1 if args.seed is None else args.seed,
        repeat=args.repeat,
        shuffle=args.shuffle,
        device=args.device,
        torch_threads=args.torch_threads,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    changes = {
        field: value
        for field in ("epochs", "batch_size", "learning_rate")
        if (value := _override(args, model_kind, field)) is not None
    }
    return replace(config, **changes)


def _progress(event: dict[str, Any]) -> None:
    print(json.dumps({"event": "epoch_complete", **event}), flush=True)


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1.")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be non-negative.")

    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else REPO_ROOT / args.output_root
    )
    summaries = []
    # Resetting every model to the same resolved seed makes a dense-only run match
    # a combined minimal+dense invocation.
    for model_kind in dict.fromkeys(args.models):
        config = resolve_config(args, model_kind)
        output_dir = output_root / model_kind / args.plume / f"R{args.repeat}"
        resume_model = args.resume and (output_dir / "training_state.pth").exists()
        print(
            json.dumps(
                {
                    "event": "training_start",
                    "output_dir": str(output_dir.resolve()),
                    "config": {
                        "model": config.model_kind,
                        "plume": config.plume,
                        "profile": config.profile,
                        "repeat": config.repeat,
                        "seed": config.seed,
                        "epochs": config.epochs,
                        "batch_size": config.batch_size,
                        "learning_rate": config.learning_rate,
                        "mode": "resume" if resume_model else "start",
                    },
                }
            ),
            flush=True,
        )
        metadata = train_figure3_model(
            args.train_data,
            args.train_labels,
            output_dir,
            config,
            test_data_path=args.test_data,
            test_labels_path=args.test_labels,
            hash_inputs=args.hash_inputs,
            overwrite=args.overwrite,
            resume=resume_model,
            progress=_progress,
        )
        summaries.append(
            {
                "model": model_kind,
                "output_dir": str(output_dir.resolve()),
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                "test_metrics": metadata["test_metrics"],
                "suggested_dense_units_by_weight_norm": metadata[
                    "suggested_dense_units_by_weight_norm"
                ],
            }
        )
    print(json.dumps({"event": "training_complete", "runs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
