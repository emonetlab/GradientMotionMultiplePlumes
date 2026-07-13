#!/usr/bin/env python3
"""Refit Figure 2 GLMs from one or more legacy train/test array repeats."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.centerline_models import (  # noqa: E402
    aggregate_figure2_fits,
    fit_figure2_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plume", choices=("smooth", "complex"), required=True)
    parser.add_argument("--train-data", type=Path, action="append", required=True)
    parser.add_argument("--train-labels", type=Path, action="append", required=True)
    parser.add_argument("--test-data", type=Path, action="append", required=True)
    parser.add_argument("--test-labels", type=Path, action="append", required=True)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Samples per feature-extraction chunk (keeps legacy memmaps bounded).",
    )
    parser.add_argument("--already-transformed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = (args.train_data, args.train_labels, args.test_data, args.test_labels)
    counts = {len(group) for group in paths}
    if len(counts) != 1:
        raise ValueError("Pass the same number of train/test data and label paths.")
    fits = []
    for train_data, train_labels, test_data, test_labels in zip(*paths):
        fits.append(
            fit_figure2_models(
                np.load(train_data, mmap_mode="r", allow_pickle=False),
                np.load(train_labels, mmap_mode="r", allow_pickle=False),
                np.load(test_data, mmap_mode="r", allow_pickle=False),
                np.load(test_labels, mmap_mode="r", allow_pickle=False),
                delay=args.delay,
                threshold=args.threshold,
                apply_log_transform=not args.already_transformed,
                chunk_size=args.chunk_size,
            )
        )
    rows = aggregate_figure2_fits(fits)
    for row in rows:
        row["plume"] = args.plume
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "plume",
        "feature",
        "auc_mean",
        "auc_sd",
        "weight_mean",
        "weight_sd",
        "repeats",
    )
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "plume": args.plume,
                "repeats": len(fits),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
