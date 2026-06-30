#!/usr/bin/env python3
"""Render a compact two-panel cue-beta figure (Smooth vs Complex) for one parameter set.

This recreates the manuscript-style panel with:
- y label "Cue β"
- x labels "M" (motion) and "G" (gradient)
- gradient sign-flipped for display
- significance annotations (n.s., *, **, ***)
"""

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
    get_palette,
    get_plot_config_value,
    get_plume_palette,
    load_plot_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-table", required=True, type=Path)
    parser.add_argument("--complex-table", required=True, type=Path)
    parser.add_argument("--smooth-label", default="smooth")
    parser.add_argument("--complex-label", default="complex")
    parser.add_argument("--smooth-title", default="VR\nSmooth")
    parser.add_argument("--complex-title", default="VR\nComplex")
    parser.add_argument("--fig-width", type=float, default=1.7)
    parser.add_argument("--fig-height", type=float, default=1.6)
    parser.add_argument("--output-dir", type=Path, default=Path("figs") / "paper_panels")
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--y-label", default="Cue β")
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--timescale-ms", type=float, default=None)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-window-min", type=float, default=None)
    parser.add_argument("--y-window-max", type=float, default=None)
    parser.add_argument("--plot-config", type=Path, default=Path("config") / "published_plot_config.yml")
    parser.add_argument(
        "--no-flip-gradient-sign",
        action="store_true",
        help="Disable display sign flip for gradient beta.",
    )
    return parser.parse_args()


def _first_present(row: pd.Series, candidates: list[str]) -> float:
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    return float("nan")


def _stderr_from_row(row: pd.Series, prefix: str) -> float:
    ci_low = _first_present(row, [f"{prefix}_ci_lower"])
    ci_high = _first_present(row, [f"{prefix}_ci_upper"])
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        return float(abs(ci_high - ci_low) / 3.92)
    return _first_present(row, [f"{prefix}_se"])


def _extract_mother_row(table_path: Path) -> pd.Series:
    if not table_path.exists():
        raise FileNotFoundError(f"Missing table: {table_path}")
    df = pd.read_csv(table_path)
    if "variant" in df.columns:
        mother = df.loc[df["variant"] == "mother"]
        if mother.empty:
            raise ValueError(f"No 'mother' row found in {table_path}")
        return mother.iloc[0]
    if len(df) == 1:
        return df.iloc[0]
    raise ValueError(f"Cannot infer mother row from {table_path}; add a 'variant' column.")


def _extract_beta_table(table_path: Path, plume_label: str, flip_gradient_sign: bool) -> pd.DataFrame:
    row = _extract_mother_row(table_path)

    motion_beta = _first_present(row, ["beta_motion", "ov_coef"])
    motion_p = _first_present(row, ["p_motion", "ov_p"])
    motion_se = _stderr_from_row(row, "beta_motion")

    grad_beta = _first_present(row, ["beta_gradient", "sg_coef"])
    grad_p = _first_present(row, ["p_gradient", "sg_p"])
    grad_se = _stderr_from_row(row, "beta_gradient")

    if flip_gradient_sign and np.isfinite(grad_beta):
        grad_beta = -float(grad_beta)

    rows = [
        {
            "plume": plume_label,
            "cue": "motion",
            "x_label": "M",
            "beta": motion_beta,
            "stderr": motion_se if np.isfinite(motion_se) else 0.0,
            "pvalue": motion_p,
        },
        {
            "plume": plume_label,
            "cue": "gradient",
            "x_label": "G",
            "beta": grad_beta,
            "stderr": grad_se if np.isfinite(grad_se) else 0.0,
            "pvalue": grad_p,
        },
    ]
    out = pd.DataFrame(rows)
    if out["beta"].isna().all():
        raise ValueError(f"No beta values found in {table_path}")
    return out


def _sig_text(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n.s."
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."


def _title_color(plume_label: str, plume_palette: dict[str, Any]) -> Any:
    if plume_label in plume_palette:
        return plume_palette[plume_label]
    if plume_label.startswith("complex") and "complex" in plume_palette:
        return plume_palette["complex"]
    if plume_label.startswith("smooth") and "smooth" in plume_palette:
        return plume_palette["smooth"]
    return "black"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = _parse_args()
    flip_gradient_sign = not args.no_flip_gradient_sign

    plot_config = load_plot_config(args.plot_config)
    apply_style_from_config(plot_config)
    plt.rcParams["figure.constrained_layout.use"] = False

    cue_palette = get_palette(plot_config, "cue") or {}
    plume_palette = get_plume_palette(plot_config) or {}
    tick_cfg = get_plot_config_value(plot_config, "ticks", {}) or {}
    tick_dir = tick_cfg.get("direction", "in")
    tick_len = float(tick_cfg.get("length", 3))
    tick_w = float(tick_cfg.get("width", 1.5))
    tick_color = tick_cfg.get("color", "black")
    color_motion = cue_palette.get("motion", "#2ca02c")
    color_gradient = cue_palette.get("gradient", "#ff7f0e")

    smooth_df = _extract_beta_table(args.smooth_table, args.smooth_label, flip_gradient_sign)
    complex_df = _extract_beta_table(args.complex_table, args.complex_label, flip_gradient_sign)

    all_df = pd.concat([smooth_df, complex_df], ignore_index=True)
    ymin_data = float((all_df["beta"] - all_df["stderr"]).min())
    ymax_data = float((all_df["beta"] + all_df["stderr"]).max())

    y_min = args.y_min if args.y_min is not None else min(-0.035, ymin_data - 0.01, 0.0)
    y_max = args.y_max if args.y_max is not None else max(0.18, ymax_data + 0.02, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width, args.fig_height), sharey=True)
    panels = [
        (axes[0], smooth_df, args.smooth_label, args.smooth_title),
        (axes[1], complex_df, args.complex_label, args.complex_title),
    ]

    for idx, (ax, panel_df, plume_label, plume_title) in enumerate(panels):
        panel_df = panel_df.set_index("cue").loc[["gradient", "motion"]].reset_index()
        x = np.array([0, 1], dtype=float)
        y = panel_df["beta"].to_numpy(dtype=float)
        err = panel_df["stderr"].to_numpy(dtype=float)
        pvals = panel_df["pvalue"].to_numpy(dtype=float)
        bar_colors = [
            color_gradient if cue == "gradient" else color_motion
            for cue in panel_df["cue"].tolist()
        ]

        ax.bar(
            x,
            y,
            color=bar_colors,
            width=0.62,
            linewidth=0,
            zorder=3,
        )
        ax.errorbar(
            x,
            y,
            yerr=err,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=0,
            zorder=4,
        )
        ax.axhline(0.0, color="black", linestyle=(0, (1.2, 1.2)), linewidth=1.0, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(panel_df["x_label"].tolist())
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(y_min, y_max)
        ax.grid(False)
        ax.tick_params(axis="x", direction=tick_dir, length=0, colors=tick_color)
        ax.tick_params(axis="y", direction=tick_dir, width=tick_w, length=tick_len, colors=tick_color)

        for side in ("left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("#2a2a2a")
            ax.spines[side].set_linewidth(1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        title_color = _title_color(plume_label, plume_palette)
        title_x = 0.60 if idx == 0 else 0.40
        ax.text(
            title_x,
            0.90,
            plume_title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            rotation=33,
            color=title_color,
            fontsize=10,
            fontweight="normal",
        )

        for xpos, beta, se, pval in zip(x, y, err, pvals, strict=False):
            txt = _sig_text(pval)
            text_y = beta + se + 0.008
            if beta < 0:
                text_y = max(0.008, se + 0.006)
            if txt.startswith("*"):
                fw = "bold"
                text_y = max(text_y, beta + se + 0.012)
            else:
                fw = "normal"
            ax.text(xpos, text_y, txt, ha="center", va="bottom", fontsize=10, fontweight=fw)

        if idx == 0:
            ax.set_ylabel(args.y_label)
            ax.set_yticks([0.0, 0.1])
        else:
            ax.tick_params(axis="y", labelleft=False, direction=tick_dir, width=tick_w, length=tick_len, colors=tick_color)

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.90, wspace=0.28)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.output_stem or f"cue_beta_panel_ts{ts}"
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
        "style": "cue_beta_two_panel_compact",
        "figure_size_inches": [args.fig_width, args.fig_height],
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
            "smooth_title": args.smooth_title,
            "complex_title": args.complex_title,
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
            "flip_gradient_sign": flip_gradient_sign,
            "y_min": y_min,
            "y_max": y_max,
            "sig_alpha": 0.05,
        },
        "values": all_df.to_dict(orient="records"),
    }
    meta_path = out_dir / "panel_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(str(png_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
