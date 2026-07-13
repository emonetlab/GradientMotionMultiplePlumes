#!/usr/bin/env python3
"""Render any of Figures 1--3, enforcing their explicit input contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "published_figures_1_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("1", "2", "3"),
        action="append",
        help="May be repeated; defaults to 1,2,3.",
    )
    parser.add_argument("--smooth-summary", type=Path, help="Required for Figure 1.")
    parser.add_argument("--complex-summary", type=Path, help="Required for Figure 1.")
    parser.add_argument("--figure3-summary", type=Path, help="Required for Figure 3.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require(value: Path | None, flag: str, figure: str) -> Path:
    if value is None:
        raise ValueError(
            f"Figure {figure} requires {flag}. See REPRODUCE_FIGURES_1_3.md."
        )
    resolved = value if value.is_absolute() else REPO_ROOT / value
    if not resolved.exists():
        raise FileNotFoundError(f"{flag} does not exist: {resolved}")
    return resolved


def main() -> int:
    args = parse_args()
    figures = args.figure or ["1", "2", "3"]
    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPO_ROOT / args.output_dir
    )
    for figure in figures:
        if figure == "1":
            smooth = require(args.smooth_summary, "--smooth-summary", figure)
            complex_ = require(args.complex_summary, "--complex-summary", figure)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "render_published_figure1.py"),
                "--smooth-summary",
                str(smooth),
                "--complex-summary",
                str(complex_),
            ]
        elif figure == "2":
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "render_published_figure2.py"),
            ]
        else:
            summary = require(args.figure3_summary, "--figure3-summary", figure)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "render_published_figure3.py"),
                "--summary",
                str(summary),
            ]
        command.extend(("--output-dir", str(output)))
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
