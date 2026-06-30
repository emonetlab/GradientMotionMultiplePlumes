#!/usr/bin/env python3
"""Render the published behavior regression panels from fixed parameters."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = REPO_ROOT / "metadata" / "published_panel_params.json"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "published_behavior_panels"

COMMON_ANALYSIS_ARGS = {
    "timescale_ms": "--timescale-ms",
    "x_min": "--x-min",
    "x_max": "--x-max",
    "y_min": "--y-window-min",
    "y_max": "--y-window-max",
}

PANEL_ARG_MAP = {
    "fig5e_cue_beta": {
        "smooth_table": "--smooth-table",
        "complex_table": "--complex-table",
        "smooth_label": "--smooth-label",
        "complex_label": "--complex-label",
        "smooth_title": "--smooth-title",
        "complex_title": "--complex-title",
        "fig_width": "--fig-width",
        "fig_height": "--fig-height",
        "y_label": "--y-label",
        "output_stem": "--output-stem",
    },
    "fig5f_cue_dominance": {
        "smooth_table": "--smooth-table",
        "complex_table": "--complex-table",
        "smooth_label": "--smooth-label",
        "complex_label": "--complex-label",
        "smooth_display": "--smooth-display",
        "complex_display": "--complex-display",
        "metric": "--metric",
        "fig_width": "--fig-width",
        "fig_height": "--fig-height",
        "y_min": "--y-min",
        "y_max": "--y-max",
        "title": "--title",
        "y_label": "--y-label",
        "output_stem": "--output-stem",
    },
    "figs5_total_differential": {
        "smooth_table": "--smooth-table",
        "complex_table": "--complex-table",
        "smooth_label": "--smooth-label",
        "complex_label": "--complex-label",
        "smooth_title": "--smooth-title",
        "complex_title": "--complex-title",
        "metric": "--metric",
        "fig_width": "--fig-width",
        "fig_height": "--fig-height",
        "y_label": "--y-label",
        "output_stem": "--output-stem",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--panel",
        choices=sorted(PANEL_ARG_MAP),
        action="append",
        help="Render only the named panel. May be repeated. Defaults to all panels.",
    )
    return parser.parse_args()


def _as_repo_path(value: str | int | float) -> str:
    if isinstance(value, str) and (value.startswith("data/") or value.startswith("scripts/")):
        return str(REPO_ROOT / value)
    return str(value)


def _append_args(cmd: list[str], values: dict, arg_map: dict[str, str]) -> None:
    for key, flag in arg_map.items():
        if key in values and values[key] is not None:
            cmd.extend([flag, _as_repo_path(values[key])])


def main() -> int:
    args = _parse_args()
    params_path = args.params if args.params.is_absolute() else REPO_ROOT / args.params
    params = json.loads(params_path.read_text())
    analysis_filter = params["analysis_filter"]
    panel_names = args.panel or list(params["panels"])

    for panel_name in panel_names:
        panel = params["panels"][panel_name]
        script_path = REPO_ROOT / panel["script"]
        panel_output = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir

        cmd = [sys.executable, str(script_path)]
        _append_args(cmd, analysis_filter, COMMON_ANALYSIS_ARGS)
        _append_args(cmd, panel, PANEL_ARG_MAP[panel_name])
        cmd.extend(["--output-dir", str(panel_output)])

        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
