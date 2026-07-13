#!/usr/bin/env python3
"""Render Figure 3 from a checkpoint-derived compact summary bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.style import apply_style_from_config, load_plot_config  # noqa: E402


DEFAULT_METADATA = REPO_ROOT / "metadata" / "figures_1_3.json"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "published_figures_1_3"
REQUIRED_ARRAYS = {
    "time_seconds",
    "minimal_smooth_filters",
    "minimal_complex_filters",
    "dense_smooth_filters",
    "dense_complex_filters",
    "gradient_difference",
    "minimal_gradient_smooth",
    "minimal_gradient_complex",
    "dense_gradient_smooth",
    "dense_gradient_complex",
    "motion_shift_seconds",
    "minimal_motion_smooth",
    "minimal_motion_complex",
    "dense_motion_smooth",
    "dense_motion_complex",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def panel_label(ax: plt.Axes, label: str) -> None:
    x = -0.23 if label not in {"A", "F"} else -0.05
    y = 1.13 if label not in {"A", "F"} else 1.00
    ax.text(
        x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom"
    )


def strip_axis(ax: plt.Axes) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.tick_params(direction="in", length=3, width=1.2, labelsize=7)


def minimal_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(0.2, 3.7, r"$(f_1 * R)(t)+(f_2 * L)(t)$", fontsize=9)
    ax.text(0.2, 1.2, r"$(f_1 * L)(t)+(f_2 * R)(t)$", fontsize=9)
    for y in (1.35, 3.85):
        ax.add_patch(Circle((5.0, y), 0.28, fill=False, edgecolor="black", lw=1.2))
        ax.plot([3.6, 4.72], [y, y], color="black", lw=1.1)
    ax.add_patch(
        Rectangle((5.9, 2.2), 0.65, 0.65, fill=False, edgecolor="black", lw=1.2)
    )
    ax.text(6.22, 2.51, r"$-$", ha="center", va="center", fontsize=12)
    ax.plot([5.28, 5.9], [3.85, 2.75], color="black")
    ax.plot([5.28, 5.9], [1.35, 2.3], color="black")
    ax.add_patch(Circle((7.25, 2.52), 0.28, fill=False, edgecolor="black", lw=1.2))
    ax.text(7.25, 2.52, "S", ha="center", va="center", fontsize=8)
    ax.plot([6.55, 6.97], [2.52, 2.52], color="black")
    ax.annotate(
        "",
        (8.15, 2.52),
        (7.53, 2.52),
        arrowprops={"arrowstyle": "-|>", "color": "black"},
    )
    ax.text(8.25, 2.52, "P(centerline side)", va="center", fontsize=8)
    ax.text(3.4, 4.6, "Minimal network model", ha="center", fontsize=10)


def dense_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    xs = (1.0, 3.3, 5.5, 7.4)
    heights = (4.1, 2.4, 2.4, 0.8)
    labels = ("60", "20", "20", "1")
    for x, height, label in zip(xs, heights, labels):
        ax.add_patch(
            Rectangle(
                (x, 2.5 - height / 2),
                0.55,
                height,
                fill=False,
                edgecolor="black",
                lw=1.2,
            )
        )
        ax.text(x + 0.27, 2.5 + height / 2 + 0.18, label, ha="center", fontsize=8)
    for left, right in zip(xs[:-1], xs[1:]):
        ax.plot([left + 0.55, right], [2.5, 2.5], color="0.4", lw=1.0)
    ax.add_patch(Circle((8.5, 2.5), 0.28, fill=False, edgecolor="black", lw=1.2))
    ax.text(8.5, 2.5, "S", ha="center", va="center", fontsize=8)
    ax.plot([7.95, 8.22], [2.5, 2.5], color="black")
    ax.annotate(
        "",
        (9.35, 2.5),
        (8.78, 2.5),
        arrowprops={"arrowstyle": "-|>", "color": "black"},
    )
    ax.text(9.45, 2.5, "P(centerline side)", va="center", fontsize=8)
    ax.text(4.5, 4.75, "Dense network model", ha="center", fontsize=10)
    ax.text(0.55, 3.7, "R", fontsize=8)
    ax.text(0.55, 1.2, "L", fontsize=8)


def filter_panel(
    ax: plt.Axes,
    time: np.ndarray,
    filters: np.ndarray,
    title: str,
    title_color: tuple[float, ...],
) -> None:
    filters = np.asarray(filters)
    if filters.ndim == 2:
        ax.plot(time, filters[0], color="0.6", lw=1.8)
        ax.plot(time, filters[1], color="black", lw=1.8)
        ax.axhline(0, color="0.65", ls=":", lw=0.8)
        ax.set_title(title, color=title_color, fontsize=8, pad=3)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Trained filters", fontsize=8)
        ax.set_xticks([-0.5, -0.25, 0.0])
        strip_axis(ax)
    elif filters.ndim == 3:
        ax.axis("off")
        ax.set_title(title, color=title_color, fontsize=8, pad=3)
        ax.text(
            0.01,
            0.46,
            "Trained filters",
            rotation=90,
            transform=ax.transAxes,
            fontsize=8,
            va="center",
        )
        for unit in range(filters.shape[0]):
            inset = ax.inset_axes([0.32, 0.54 - 0.48 * unit, 0.66, 0.40])
            inset.plot(
                time, filters[unit, 0], color="0.6", lw=1.2, alpha=1.0 - 0.25 * unit
            )
            inset.plot(
                time, filters[unit, 1], color="black", lw=1.2, alpha=1.0 - 0.25 * unit
            )
            inset.axhline(0, color="0.65", ls=":", lw=0.8)
            inset.set_xticks([-0.5, -0.25, 0.0])
            if unit == 0:
                inset.set_xticklabels([])
            else:
                inset.set_xlabel("Time (s)", fontsize=8)
            strip_axis(inset)
    else:
        raise ValueError(f"Unexpected filter array shape: {filters.shape}")


def response_panel(
    ax: plt.Axes,
    x: np.ndarray,
    smooth: np.ndarray,
    complex_: np.ndarray,
    *,
    title: str,
    xlabel: str,
    colors: dict[str, tuple[float, ...]],
) -> None:
    ax.axhline(0.5, color="0.65", ls=":", lw=0.8)
    ax.axvline(0, color="0.65", ls=":", lw=0.8)
    ax.plot(x, smooth, color=colors["smooth"], marker="o", ms=3, lw=1.7, label="smooth")
    ax.plot(
        x, complex_, color=colors["complex"], marker="o", ms=3, lw=1.7, label="complex"
    )
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Model response\n(averaged)", fontsize=8)
    if "Antenna" in xlabel:
        ax.set_xticks([-0.5, 0.0, 0.5])
    else:
        ax.set_xticks([-0.1, 0.0, 0.1])
    strip_axis(ax)
    ax.legend(frameon=False, fontsize=6, loc="best", handlelength=1.4)


def main() -> int:
    args = parse_args()
    summary_path = (
        args.summary if args.summary.is_absolute() else REPO_ROOT / args.summary
    )
    metadata_path = (
        args.metadata if args.metadata.is_absolute() else REPO_ROOT / args.metadata
    )
    metadata = json.loads(metadata_path.read_text())
    colors = {name: tuple(value) for name, value in metadata["palette"].items()}
    with np.load(summary_path, allow_pickle=False) as bundle:
        missing = sorted(REQUIRED_ARRAYS - set(bundle.files))
        if missing:
            raise KeyError(f"Figure 3 summary is missing arrays: {missing}")
        arrays = {name: np.asarray(bundle[name]) for name in REQUIRED_ARRAYS}
        source_metadata = (
            json.loads(str(bundle["metadata_json"].item()))
            if "metadata_json" in bundle
            else {}
        )
    apply_style_from_config(load_plot_config())
    plt.rcParams["figure.constrained_layout.use"] = False

    fig = plt.figure(figsize=(10.0, 6.5))
    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=(0.72, 1, 1),
        left=0.07,
        right=0.98,
        bottom=0.10,
        top=0.96,
        hspace=0.78,
        wspace=0.72,
    )
    ax_a = fig.add_subplot(grid[0, :2])
    minimal_schematic(ax_a)
    panel_label(ax_a, "A")
    ax_f = fig.add_subplot(grid[0, 2:])
    dense_schematic(ax_f)
    panel_label(ax_f, "F")

    filter_specs = [
        (
            "B",
            arrays["minimal_smooth_filters"],
            "Minimal model trained\non smooth plume",
            colors["smooth"],
        ),
        (
            "C",
            arrays["minimal_complex_filters"],
            "Minimal model trained\non complex plume",
            colors["complex"],
        ),
        (
            "G",
            arrays["dense_smooth_filters"],
            "Dense model trained\non smooth plume",
            colors["smooth"],
        ),
        (
            "H",
            arrays["dense_complex_filters"],
            "Dense model trained\non complex plume",
            colors["complex"],
        ),
    ]
    for column, (label, values, title, color) in enumerate(filter_specs):
        ax = fig.add_subplot(grid[1, column])
        filter_panel(ax, arrays["time_seconds"], values, title, color)
        panel_label(ax, label)

    response_specs = [
        (
            "D",
            arrays["gradient_difference"],
            arrays["minimal_gradient_smooth"],
            arrays["minimal_gradient_complex"],
            "Response on\ngradient-only signal",
            "Antenna difference (a.u.)",
            colors["gradient"],
        ),
        (
            "E",
            arrays["motion_shift_seconds"],
            arrays["minimal_motion_smooth"],
            arrays["minimal_motion_complex"],
            "Response on\nmotion-only signal",
            "Temporal shift (s)",
            colors["motion"],
        ),
        (
            "I",
            arrays["gradient_difference"],
            arrays["dense_gradient_smooth"],
            arrays["dense_gradient_complex"],
            "Response on\ngradient-only signal",
            "Antenna difference (a.u.)",
            colors["gradient"],
        ),
        (
            "J",
            arrays["motion_shift_seconds"],
            arrays["dense_motion_smooth"],
            arrays["dense_motion_complex"],
            "Response on\nmotion-only signal",
            "Temporal shift (s)",
            colors["motion"],
        ),
    ]
    for column, (label, x, smooth, complex_, title, xlabel, title_color) in enumerate(
        response_specs
    ):
        ax = fig.add_subplot(grid[2, column])
        response_panel(
            ax, x, smooth, complex_, title=title, xlabel=xlabel, colors=colors
        )
        ax.title.set_color(title_color)
        panel_label(ax, label)

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPO_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure3.pdf"
    png_path = output_dir / "figure3.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    (output_dir / "figure3_metadata.json").write_text(
        json.dumps(
            {
                "figure": 3,
                "renderer": "scripts/render_published_figure3.py",
                "summary": str(summary_path),
                "summary_metadata": source_metadata,
                "outputs": [str(pdf_path), str(png_path)],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
