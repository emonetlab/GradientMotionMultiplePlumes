#!/usr/bin/env python3
"""Render the complete Figure 1 layout from two compact plume summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.plume_fields import load_plume_summary  # noqa: E402
from gradient_motion_panels.style import apply_style_from_config, load_plot_config  # noqa: E402


DEFAULT_METADATA = REPO_ROOT / "metadata" / "figures_1_3.json"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "published_figures_1_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-summary", type=Path, required=True)
    parser.add_argument("--complex-summary", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )


def clean_image_axis(ax: plt.Axes, color: tuple[float, ...]) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(1.0)


def navigation_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.scatter([0.8], [2.5], marker="*", s=140, color="black", zorder=4)
    ax.text(0.8, 1.65, "Odor\nsource", ha="center", va="top", fontsize=8)
    for x, width, alpha in [
        (2.0, 2.4, 0.18),
        (3.2, 3.0, 0.13),
        (4.8, 4.2, 0.09),
        (6.3, 5.2, 0.06),
    ]:
        ax.add_patch(
            Ellipse(
                (x + 1.0, 2.5), width, 1.2 + 0.18 * x, color="0.35", alpha=alpha, lw=0
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            (0.8, 4.25),
            (3.5, 4.25),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.5,
            color="black",
        )
    )
    ax.text(0.8, 4.48, "Wind direction", ha="left", fontsize=8)
    ax.scatter(
        [6.2, 8.2],
        [2.55, 1.4],
        s=80,
        color=[(0.85, 0.35, 0.12)],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.add_patch(
        FancyArrowPatch((6.05, 2.75), (5.15, 3.45), arrowstyle="->", color="black")
    )
    ax.text(4.55, 3.58, "Upwind", fontsize=8)
    ax.add_patch(
        FancyArrowPatch((8.05, 1.55), (7.15, 2.25), arrowstyle="->", color="black")
    )
    ax.text(6.65, 2.38, "Downwind", fontsize=8, ha="right")
    ax.add_patch(
        FancyArrowPatch((8.2, 1.2), (8.2, 0.45), arrowstyle="->", color="black")
    )
    ax.text(8.2, 0.20, "Crosswind", ha="center", fontsize=8)


def antenna_schematic(ax: plt.Axes, colors: dict[str, tuple[float, ...]]) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.add_patch(Ellipse((2.2, 3.1), 1.8, 1.25, color=(0.95, 0.55, 0.10), ec="0.3"))
    ax.scatter(
        [1.55, 1.55],
        [3.45, 2.75],
        s=35,
        color=[colors["right"], colors["left"]],
        zorder=3,
    )
    ax.annotate(
        "Spatial\ninformation",
        (1.55, 3.1),
        (0.3, 3.1),
        arrowprops={"arrowstyle": "<->", "color": "black"},
        ha="center",
        va="center",
        fontsize=8,
    )
    t = np.linspace(0, 1, 80)
    trace = (
        0.15
        + 0.9 * np.exp(-(((t - 0.58) / 0.14) ** 2))
        + 0.35 * np.exp(-(((t - 0.82) / 0.12) ** 2))
    )
    ax.plot(5.0 + 4.2 * t, 1.3 + 2.0 * trace, color=colors["left"], lw=2)
    ax.plot(5.0 + 4.2 * t, 1.45 + 1.8 * np.roll(trace, 4), color=colors["right"], lw=2)
    ax.annotate(
        "Temporal information",
        (8.8, 4.6),
        (5.3, 4.6),
        arrowprops={"arrowstyle": "<->", "color": "black"},
        ha="center",
        fontsize=8,
    )
    ax.annotate(
        "",
        (9.35, 1.0),
        (4.8, 1.0),
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "black"},
    )
    ax.annotate(
        "",
        (4.8, 5.0),
        (4.8, 0.9),
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "black"},
    )
    ax.text(7.1, 0.55, "Time", ha="center", fontsize=8)
    ax.text(4.35, 3.0, "Odor intensity", rotation=90, va="center", fontsize=8)


def image_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    array: np.ndarray,
    *,
    color: tuple[float, ...],
    title: str,
    cmap: str,
    limit: float | None = None,
    colorbar_label: str | None = None,
) -> None:
    kwargs = {
        "origin": "lower",
        "aspect": "auto",
        "cmap": cmap,
        "interpolation": "nearest",
    }
    ticks = None
    if limit is not None:
        kwargs.update(vmin=-limit, vmax=limit)
        ticks = [-limit, 0, limit]
    image = ax.imshow(array, **kwargs)
    clean_image_axis(ax, color)
    ax.set_title(title, fontsize=8, color=color, pad=3)
    if colorbar_label:
        colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025, ticks=ticks)
        colorbar.ax.tick_params(labelsize=6, length=2)
        colorbar.ax.set_title(colorbar_label, fontsize=6, pad=2)


def main() -> int:
    args = parse_args()
    metadata_path = (
        args.metadata if args.metadata.is_absolute() else REPO_ROOT / args.metadata
    )
    smooth_path = (
        args.smooth_summary
        if args.smooth_summary.is_absolute()
        else REPO_ROOT / args.smooth_summary
    )
    complex_path = (
        args.complex_summary
        if args.complex_summary.is_absolute()
        else REPO_ROOT / args.complex_summary
    )
    metadata = json.loads(metadata_path.read_text())
    smooth = load_plume_summary(smooth_path)
    complex_ = load_plume_summary(complex_path)
    for expected, summary in (("smooth", smooth), ("complex", complex_)):
        actual = summary.metadata.get("plume")
        if actual != expected:
            raise ValueError(
                f"Expected the {expected} summary role, but its metadata says {actual!r}."
            )
        if summary.metadata.get("snapshot_role") == "substitute":
            warnings.warn(
                f"Rendering the {expected} plume with a substitute snapshot: "
                f"{summary.metadata.get('snapshot_note') or 'no note supplied'}",
                RuntimeWarning,
                stacklevel=1,
            )
    colors = {name: tuple(value) for name, value in metadata["palette"].items()}
    limits = metadata["figure1"]["display_limits"]
    apply_style_from_config(load_plot_config())

    fig = plt.figure(figsize=(11.0, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(
        4, 4, width_ratios=(1, 1, 1.18, 1.18), hspace=0.28, wspace=0.22
    )

    ax_a = fig.add_subplot(grid[0, 0:2])
    navigation_schematic(ax_a)
    panel_label(ax_a, "A")

    ax_b = fig.add_subplot(grid[1, 0])
    image_panel(
        fig,
        ax_b,
        smooth.snapshot,
        color=colors["smooth"],
        title="Smooth odor plume",
        cmap="gray_r",
    )
    panel_label(ax_b, "B")
    ax_c = fig.add_subplot(grid[1, 1])
    image_panel(
        fig,
        ax_c,
        complex_.snapshot,
        color=colors["complex"],
        title="Complex odor plume",
        cmap="gray_r",
    )
    panel_label(ax_c, "C")

    ax_d = fig.add_subplot(grid[2:4, 0:2])
    antenna_schematic(ax_d, colors)
    panel_label(ax_d, "D")

    specs = [
        (
            "E",
            smooth.gradient_snapshot,
            colors["smooth"],
            r"Smooth: $\partial I/\partial y$",
            limits["smooth_gradient_snapshot"],
            "a.u.",
        ),
        (
            "F",
            complex_.gradient_snapshot,
            colors["complex"],
            r"Complex: $\partial I/\partial y$",
            limits["complex_gradient_snapshot"],
            "a.u.",
        ),
        (
            "G",
            smooth.motion_snapshot,
            colors["smooth"],
            r"$-(\partial I/\partial y)(\partial I/\partial t)$",
            limits["smooth_motion_snapshot"],
            "a.u.",
        ),
        (
            "H",
            complex_.motion_snapshot,
            colors["complex"],
            r"$-(\partial I/\partial y)(\partial I/\partial t)$",
            limits["complex_motion_snapshot"],
            "a.u.",
        ),
        (
            "I",
            smooth.gradient_zscore,
            colors["smooth"],
            r"$\langle\partial I/\partial y\rangle_t$",
            limits["gradient_zscore"],
            "z-score",
        ),
        (
            "J",
            complex_.gradient_zscore,
            colors["complex"],
            r"$\langle\partial I/\partial y\rangle_t$",
            limits["gradient_zscore"],
            "z-score",
        ),
        (
            "K",
            smooth.motion_zscore,
            colors["smooth"],
            r"$\langle-(\partial I/\partial y)(\partial I/\partial t)\rangle_t$",
            limits["motion_zscore"],
            "z-score",
        ),
        (
            "L",
            complex_.motion_zscore,
            colors["complex"],
            r"$\langle-(\partial I/\partial y)(\partial I/\partial t)\rangle_t$",
            limits["motion_zscore"],
            "z-score",
        ),
    ]
    for index, (label, array, color, title, limit, cbar) in enumerate(specs):
        row, col = divmod(index, 2)
        ax = fig.add_subplot(grid[row, col + 2])
        image_panel(
            fig,
            ax,
            array,
            color=color,
            title=title,
            cmap="bwr",
            limit=float(limit),
            colorbar_label=cbar,
        )
        panel_label(ax, label)

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPO_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure1.pdf"
    png_path = output_dir / "figure1.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    (output_dir / "figure1_metadata.json").write_text(
        json.dumps(
            {
                "figure": 1,
                "renderer": "scripts/render_published_figure1.py",
                "smooth_summary": str(smooth_path),
                "complex_summary": str(complex_path),
                "smooth_input_metadata": smooth.metadata,
                "complex_input_metadata": complex_.metadata,
                "outputs": [str(pdf_path), str(png_path)],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
