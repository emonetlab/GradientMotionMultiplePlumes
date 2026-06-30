#!/usr/bin/env python3
"""Render per-plume total/differential decomposition panels from comparison tables."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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

VARIANT_ORDER = ["mother", "joint_drop", "grad_only", "vel_only"]
VARIANT_LABELS = {
    "mother": "Full",
    "joint_drop": "Base",
    "grad_only": "Gradient",
    "vel_only": "Motion",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-table", required=True, type=Path)
    parser.add_argument("--complex-table", required=True, type=Path)
    parser.add_argument("--smooth-label", default="smooth")
    parser.add_argument("--complex-label", default="complex")
    parser.add_argument("--smooth-title", default="Smooth plume")
    parser.add_argument("--complex-title", default="Complex plume")
    parser.add_argument("--metric", default="log_likelihood_mean")
    parser.add_argument("--fig-width", type=float, default=2.7)
    parser.add_argument("--fig-height", type=float, default=1.75)
    parser.add_argument("--y-label", default="Mean log-likelihood")
    parser.add_argument("--output-dir", type=Path, default=Path("figs") / "paper_panels")
    parser.add_argument("--output-stem", default="")
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


def _resolve_plume_color(plume_label: str, plume_palette: dict[str, Any], fallback: str) -> Any:
    if plume_label in plume_palette:
        return plume_palette[plume_label]
    if plume_label.startswith("complex") and "complex" in plume_palette:
        return plume_palette["complex"]
    if plume_label.startswith("smooth") and "smooth" in plume_palette:
        return plume_palette["smooth"]
    return fallback


def _metric_axis_label(metric: str) -> str:
    if metric == "log_likelihood_mean":
        return "Mean log-likelihood"
    if metric == "auroc":
        return "AUC ROC"
    return metric


def _panel_df(table_path: Path, metric: str) -> pd.DataFrame:
    if not table_path.exists():
        raise FileNotFoundError(f"Missing comparison table: {table_path}")
    df = pd.read_csv(table_path)
    if "variant" not in df.columns:
        raise KeyError(f"Missing 'variant' column in {table_path}")
    if metric not in df.columns:
        raise KeyError(f"Missing metric '{metric}' in {table_path}")
    out = (
        df.loc[df["variant"].isin(VARIANT_ORDER), ["variant", metric]]
        .assign(variant=lambda d: pd.Categorical(d["variant"], categories=VARIANT_ORDER, ordered=True))
        .dropna(subset=["variant"])
        .sort_values("variant")
        .assign(variant_label=lambda d: d["variant"].map(VARIANT_LABELS))
        .rename(columns={metric: "value"})
    )
    missing = [v for v in VARIANT_ORDER if v not in set(out["variant"].astype(str))]
    if missing:
        raise ValueError(f"Missing variants {missing} in {table_path}")
    return out


def _render_one_panel(
    *,
    table_path: Path,
    plume_label: str,
    plume_title: str,
    metric: str,
    fig_width: float,
    fig_height: float,
    y_label: str,
    output_dir: Path,
    output_stem: str,
    tick_cfg: dict[str, Any],
    plume_palette: dict[str, Any],
    context_params: dict[str, Any],
) -> dict[str, Any]:
    panel_df = _panel_df(table_path, metric)
    value_min = float(panel_df["value"].min())
    value_max = float(panel_df["value"].max())
    span = value_max - value_min
    pad = max(span * 0.30, 0.001) if span > 0 else max(abs(value_max) * 0.02, 0.001)
    y_min = value_min - pad
    y_max = value_max + pad

    plume_color = _resolve_plume_color(plume_label, plume_palette, "#4C78A8")

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    x_labels = [VARIANT_LABELS[v] for v in VARIANT_ORDER]
    x_pos = list(range(len(x_labels)))

    # Draw bars from the lower panel bound so they visually rise "from below".
    ax.bar(
        x_pos,
        (panel_df["value"] - y_min).to_list(),
        bottom=y_min,
        color=plume_color,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.95,
        width=0.70,
        zorder=3,
    )
    ax.axhline(0.0, color="black", linestyle=(0, (1.2, 1.2)), linewidth=1.0, zorder=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_title(plume_title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    ax.set_ylim(y_min, y_max)
    ax.grid(False, axis="x")

    tick_dir = tick_cfg.get("direction", "in")
    tick_len = float(tick_cfg.get("length", 3))
    tick_w = float(tick_cfg.get("width", 1.5))
    tick_color = tick_cfg.get("color", "black")
    ax.tick_params(axis="x", direction=tick_dir, width=tick_w, length=tick_len, colors=tick_color)
    ax.tick_params(axis="y", direction=tick_dir, width=tick_w, length=tick_len, colors=tick_color)

    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#2a2a2a")
        ax.spines[side].set_linewidth(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    value_map = dict(zip(panel_df["variant_label"], panel_df["value"]))
    patch_map = dict(zip(x_labels, ax.patches))

    def add_vertical_arrow(x_label: str, y_start: float, y_end: float, text: str) -> None:
        x = x_labels.index(x_label)
        ax.annotate(
            "",
            xy=(x, y_end),
            xytext=(x, y_start),
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.0},
        )
        x_offset = 8 if x <= 1.5 else -8
        align = "left" if x_offset > 0 else "right"
        ax.annotate(
            text,
            xy=(x, (y_start + y_end) / 2),
            xytext=(x_offset, 0),
            textcoords="offset points",
            ha=align,
            va="center",
            rotation=90,
        )

    def add_connector(start_label: str, end_label: str, y_end: float) -> None:
        start_patch = patch_map[start_label]
        start_x = start_patch.get_x() + start_patch.get_width()
        start_y = start_patch.get_y() + start_patch.get_height()
        end_x = x_labels.index(end_label)
        ax.plot(
            [start_x, end_x],
            [start_y, y_end],
            linestyle=(0, (1, 1)),
            linewidth=1.0,
            color="black",
            zorder=4,
        )

    if {"Base", "Full"}.issubset(value_map):
        total_top = float(value_map["Full"])
        add_vertical_arrow("Base", float(value_map["Base"]), total_top, "total")
        add_connector("Full", "Base", total_top)

    if {"Motion", "Gradient"}.issubset(value_map):
        diff_top = float(value_map["Gradient"])
        add_vertical_arrow("Motion", float(value_map["Motion"]), diff_top, "differential")
        add_connector("Gradient", "Motion", diff_top)

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.24, top=0.86)

    stem = f"{output_stem}_{plume_label}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    return {
        "plume_label": plume_label,
        "plume_title": plume_title,
        "metric": metric,
        "figure_size_inches": [fig_width, fig_height],
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "output_hashes": {
            "png_sha256": _sha256(png_path),
            "pdf_sha256": _sha256(pdf_path),
        },
        "table_path": str(table_path),
        "values": panel_df.to_dict(orient="records"),
        "y_limits": [y_min, y_max],
        "context": context_params,
    }


def main() -> int:
    args = _parse_args()
    metric = METRIC_ALIASES.get(args.metric, args.metric)
    if args.y_label == "Mean log-likelihood":
        y_label = _metric_axis_label(metric)
    else:
        y_label = args.y_label

    plot_config = load_plot_config(args.plot_config)
    apply_style_from_config(plot_config)
    plt.rcParams["figure.constrained_layout.use"] = False
    tick_cfg = get_plot_config_value(plot_config, "ticks", {}) or {}
    plume_palette = get_plume_palette(plot_config) or {}

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.output_stem or f"total_differential_ts{ts}"
    out_dir = args.output_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "timescale_ms": args.timescale_ms,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_window_min,
        "y_max": args.y_window_max,
    }

    results = []
    results.append(
        _render_one_panel(
            table_path=args.smooth_table,
            plume_label=args.smooth_label,
            plume_title=args.smooth_title,
            metric=metric,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            y_label=y_label,
            output_dir=out_dir,
            output_stem=stem,
            tick_cfg=tick_cfg,
            plume_palette=plume_palette,
            context_params=context,
        )
    )
    results.append(
        _render_one_panel(
            table_path=args.complex_table,
            plume_label=args.complex_label,
            plume_title=args.complex_title,
            metric=metric,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            y_label=y_label,
            output_dir=out_dir,
            output_stem=stem,
            tick_cfg=tick_cfg,
            plume_palette=plume_palette,
            context_params=context,
        )
    )

    meta_path = out_dir / "panel_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "created_at": dt.datetime.now().isoformat(),
                "style": "total_differential_per_plume",
                "metric": metric,
                "labels": {
                    "smooth_label": args.smooth_label,
                    "complex_label": args.complex_label,
                    "smooth_title": args.smooth_title,
                    "complex_title": args.complex_title,
                    "y_label": y_label,
                },
                "context": context,
                "panels": results,
            },
            indent=2,
        )
    )

    for row in results:
        print(row["output_png"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
