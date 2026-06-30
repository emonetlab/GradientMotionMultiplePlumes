#!/usr/bin/env python3
"""Render a compact cue-dominance panel (gradient-motion differential/total) by plume."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.style import (
    apply_style_from_config,
    get_plot_config_value,
    get_plume_palette,
    load_plot_config,
)

METRIC_ALIASES = {
    "ll": "log_likelihood_mean",
    "log_likelihood": "log_likelihood_mean",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-table", required=True, type=Path)
    parser.add_argument("--complex-table", required=True, type=Path)
    parser.add_argument("--smooth-label", default="smooth")
    parser.add_argument("--complex-label", default="complex")
    parser.add_argument("--smooth-display", default="Smooth")
    parser.add_argument("--complex-display", default="Complex")
    parser.add_argument("--metric", default="log_likelihood_mean")
    parser.add_argument("--fig-width", type=float, default=2.4)
    parser.add_argument("--fig-height", type=float, default=2.0)
    parser.add_argument("--y-min", type=float, default=-1.0)
    parser.add_argument("--y-max", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("figs") / "paper_panels")
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--title", default="Cue dominance per plume")
    parser.add_argument("--y-label", default="Gradient-Motion dominance")
    parser.add_argument("--timescale-ms", type=float, default=None)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-window-min", type=float, default=None)
    parser.add_argument("--y-window-max", type=float, default=None)
    parser.add_argument("--plot-config", type=Path, default=Path("config") / "published_plot_config.yml")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _ratio_from_table(path: Path, metric: str) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison table: {path}")
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise KeyError(f"Missing metric '{metric}' in {path}")
    if "variant" not in df.columns:
        raise KeyError(f"Missing 'variant' column in {path}")
    pivot = df.set_index("variant")[metric]
    required = ["mother", "vel_only", "grad_only", "joint_drop"]
    missing = [name for name in required if name not in pivot.index]
    if missing:
        raise KeyError(f"Missing variants in {path}: {missing}")
    denom = float(pivot["mother"] - pivot["joint_drop"])
    if denom == 0:
        raise ValueError(f"Zero denominator for dominance ratio in {path}")
    return float((pivot["grad_only"] - pivot["vel_only"]) / denom)


def _resolve_plume_color(plume_label: str, plume_palette: dict[str, Any], fallback: str) -> Any:
    if plume_label in plume_palette:
        return plume_palette[plume_label]
    if plume_label.startswith("complex") and "complex" in plume_palette:
        return plume_palette["complex"]
    if plume_label.startswith("smooth") and "smooth" in plume_palette:
        return plume_palette["smooth"]
    return fallback


def main() -> int:
    args = _parse_args()
    metric = METRIC_ALIASES.get(args.metric, args.metric)

    plot_config = load_plot_config(args.plot_config)
    apply_style_from_config(plot_config)
    plt.rcParams["figure.constrained_layout.use"] = False

    tick_cfg = get_plot_config_value(plot_config, "ticks", {}) or {}
    tick_dir = tick_cfg.get("direction", "in")
    tick_len = float(tick_cfg.get("length", 3))
    tick_w = float(tick_cfg.get("width", 1.5))
    tick_color = tick_cfg.get("color", "black")

    plume_palette = get_plume_palette(plot_config) or {}
    smooth_color = _resolve_plume_color(args.smooth_label, plume_palette, "#9467bd")
    complex_color = _resolve_plume_color(args.complex_label, plume_palette, "#bcbd22")

    smooth_ratio = _ratio_from_table(args.smooth_table, metric)
    complex_ratio = _ratio_from_table(args.complex_table, metric)

    plot_df = pd.DataFrame(
        {
            "plume_label": [args.smooth_label, args.complex_label],
            "display": [args.smooth_display, args.complex_display],
            "ratio": [smooth_ratio, complex_ratio],
            "color": [smooth_color, complex_color],
        }
    )

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    x = np.arange(len(plot_df), dtype=float)
    ax.bar(
        x,
        plot_df["ratio"].to_numpy(dtype=float),
        color=plot_df["color"].tolist(),
        width=0.62,
        linewidth=0,
        zorder=3,
    )
    ax.axhline(0.0, color="black", linestyle=(0, (1.2, 1.2)), linewidth=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["display"].tolist())
    ax.set_ylim(float(args.y_min), float(args.y_max))
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_title(args.title)
    ax.set_ylabel(args.y_label)
    ax.set_xlabel("")
    ax.grid(False, axis="x")

    ax.tick_params(axis="x", direction=tick_dir, width=tick_w, length=tick_len, colors=tick_color)
    ax.tick_params(axis="y", direction=tick_dir, width=tick_w, length=tick_len, colors=tick_color)

    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#2a2a2a")
        ax.spines[side].set_linewidth(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.23, top=0.86)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.output_stem or f"cue_dominance_panel_ts{ts}"
    out_dir = args.output_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(
        pdf_path,
        metadata={
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)

    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "style": "cue_dominance_single_panel",
        "figure_size_inches": [args.fig_width, args.fig_height],
        "metric": metric,
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "output_hashes": {
            "png_sha256": _sha256(png_path),
            "pdf_sha256": _sha256(pdf_path),
        },
        "inputs": {
            "smooth_table": str(args.smooth_table),
            "complex_table": str(args.complex_table),
        },
        "labels": {
            "smooth_label": args.smooth_label,
            "complex_label": args.complex_label,
            "smooth_display": args.smooth_display,
            "complex_display": args.complex_display,
            "title": args.title,
            "y_label": args.y_label,
        },
        "params": {
            "timescale_ms": args.timescale_ms,
            "x_min": args.x_min,
            "x_max": args.x_max,
            "y_min": args.y_window_min,
            "y_max": args.y_window_max,
        },
        "display": {
            "y_axis_min": float(args.y_min),
            "y_axis_max": float(args.y_max),
        },
        "dominance_values": plot_df.to_dict(orient="records"),
    }
    meta_path = out_dir / "panel_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(str(png_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
