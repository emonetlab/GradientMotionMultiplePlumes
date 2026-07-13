#!/usr/bin/env python3
"""Render Figure 2 from the checked-in published GLM summary values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.style import apply_style_from_config, load_plot_config  # noqa: E402


DEFAULT_TABLE = REPO_ROOT / "data" / "published_panel_tables" / "fig2_glm_summary.csv"
DEFAULT_METADATA = REPO_ROOT / "metadata" / "figures_1_3.json"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "published_figures_1_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        action="append",
        help="Summary CSV; repeat to combine separately refit smooth/complex tables.",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def panel_label(ax: plt.Axes, label: str) -> None:
    x = -0.25 if label != "A" else -0.04
    y = 1.18 if label != "A" else 1.00
    ax.text(
        x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom"
    )


def schematic(ax: plt.Axes, colors: dict[str, tuple[float, ...]]) -> None:
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.add_patch(Rectangle((0.4, 2.2), 4.0, 3.0, fill=False, edgecolor="black", lw=1.2))
    for x, y, alpha in [
        (1.1, 3.7, 0.12),
        (2.0, 3.2, 0.14),
        (3.1, 4.1, 0.10),
        (3.8, 2.8, 0.12),
    ]:
        ax.scatter([x], [y], s=500, color="0.4", alpha=alpha, linewidth=0)
    ax.plot([0.4, 4.4], [3.7, 3.7], color="0.2", lw=1, ls=":")
    ax.scatter(
        [2.45, 2.45],
        [3.48, 3.92],
        color=[colors["right"], colors["left"]],
        s=24,
        zorder=3,
    )
    t = np.linspace(-0.5, 0, 30)
    right = (
        0.15
        + 1.4 * np.exp(-(((t + 0.25) / 0.045) ** 2))
        + 0.9 * np.exp(-(((t + 0.08) / 0.05) ** 2))
    )
    left = (
        0.12
        + 1.15 * np.exp(-(((t + 0.28) / 0.05) ** 2))
        + 1.1 * np.exp(-(((t + 0.11) / 0.045) ** 2))
    )
    ax.plot(4.9 + 3.0 * (t + 0.5) / 0.5, 2.3 + right, color=colors["right"], lw=1.8)
    ax.plot(4.9 + 3.0 * (t + 0.5) / 0.5, 2.3 + left, color=colors["left"], lw=1.8)
    ax.text(5.0, 4.25, r"$R(t)$", color=colors["right"], fontsize=9)
    ax.text(7.25, 4.25, r"$L(t)$", color=colors["left"], fontsize=9)
    ax.annotate(
        "",
        (8.5, 3.5),
        (7.9, 3.5),
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "black"},
    )
    ax.text(8.75, 4.5, r"Sum  $\langle L+R\rangle_t$", fontsize=9)
    ax.text(
        8.75,
        3.75,
        r"Gradient  $\langle L-R\rangle_t$",
        fontsize=9,
        color=colors["gradient"],
    )
    ax.text(
        8.75,
        3.0,
        r"Motion  $\langle L_{t-1}R_t-L_tR_{t-1}\rangle_t$",
        fontsize=9,
        color=colors["motion"],
    )
    x = np.linspace(-4, 4, 120)
    y = 1 / (1 + np.exp(-x))
    ax.plot(12.2 + 1.3 * (x + 4) / 8, 0.45 + 1.35 * y, color="black", lw=1.5)
    ax.text(11.7, 0.12, "Logistic regression", fontsize=9)
    ax.annotate(
        "",
        (12.15, 1.1),
        (11.25, 2.55),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "black"},
    )


def bar_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    metric: str,
    error: str,
    title: str,
    title_color: tuple[float, ...],
    colors: dict[str, tuple[float, ...]],
    ylim: tuple[float, float],
    ylabel: str,
) -> None:
    ordered = table.set_index("feature").loc[["sum", "gradient", "motion"]]
    positions = np.arange(3)
    values = ordered[metric].to_numpy(float)
    errors = ordered[error].to_numpy(float)
    bar_colors = ("black", colors["gradient"], colors["motion"])
    ax.bar(positions, values, color=bar_colors, width=0.72)
    ax.errorbar(positions, values, yerr=errors, fmt="none", ecolor="black", lw=1.2)
    ax.set_xticks(
        positions, ["Sum only", "Gradient only", "Motion only"], rotation=48, ha="right"
    )
    for tick, color in zip(ax.get_xticklabels(), bar_colors):
        tick.set_color(color)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, color=title_color, fontsize=9, pad=4)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.tick_params(direction="in", length=3, width=1.2, labelsize=8)


def main() -> int:
    args = parse_args()
    requested_tables = args.table or [DEFAULT_TABLE]
    table_paths = [
        path if path.is_absolute() else REPO_ROOT / path for path in requested_tables
    ]
    metadata_path = (
        args.metadata if args.metadata.is_absolute() else REPO_ROOT / args.metadata
    )
    table = pd.concat([pd.read_csv(path) for path in table_paths], ignore_index=True)
    required = {"plume", "feature", "auc_mean", "auc_sd", "weight_mean", "weight_sd"}
    if missing := sorted(required - set(table.columns)):
        raise KeyError(f"Figure 2 table is missing columns: {missing}")
    expected_rows = {
        (plume, feature)
        for plume in ("smooth", "complex")
        for feature in ("sum", "gradient", "motion")
    }
    actual_rows = set(zip(table["plume"], table["feature"]))
    if len(table) != len(expected_rows) or actual_rows != expected_rows:
        raise ValueError(
            "Figure 2 input must contain exactly one row for each smooth/complex "
            "and sum/gradient/motion combination."
        )
    metadata = json.loads(metadata_path.read_text())
    colors = {name: tuple(value) for name, value in metadata["palette"].items()}
    apply_style_from_config(load_plot_config())
    plt.rcParams["figure.constrained_layout.use"] = False

    fig = plt.figure(figsize=(6.5, 7.0))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.0, 1, 1),
        left=0.12,
        right=0.97,
        bottom=0.08,
        top=0.97,
        hspace=0.95,
        wspace=0.65,
    )
    ax_a = fig.add_subplot(grid[0, :])
    schematic(ax_a, colors)
    panel_label(ax_a, "A")

    panels = [
        (
            "B",
            "smooth",
            "auc_mean",
            "auc_sd",
            "Smooth odor plume\nModels with single features",
            colors["smooth"],
            (0.5, 1.0),
            "Model performance (AUC)",
        ),
        (
            "C",
            "complex",
            "auc_mean",
            "auc_sd",
            "Complex odor plume\nModels with single features",
            colors["complex"],
            (0.5, 1.0),
            "Model performance (AUC)",
        ),
        (
            "D",
            "smooth",
            "weight_mean",
            "weight_sd",
            "Smooth odor plume\nModel with all three features",
            colors["smooth"],
            (0.0, 3.1),
            "Weight value",
        ),
        (
            "E",
            "complex",
            "weight_mean",
            "weight_sd",
            "Complex odor plume\nModel with all three features",
            colors["complex"],
            (0.0, 0.65),
            "Weight value",
        ),
    ]
    for index, (label, plume, metric, error, title, color, ylim, ylabel) in enumerate(
        panels
    ):
        row = 1 + index // 2
        col = index % 2
        ax = fig.add_subplot(grid[row, col])
        bar_panel(
            ax,
            table.loc[table["plume"] == plume],
            metric=metric,
            error=error,
            title=title,
            title_color=color,
            colors=colors,
            ylim=ylim,
            ylabel=ylabel,
        )
        panel_label(ax, label)

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPO_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure2.pdf"
    png_path = output_dir / "figure2.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    (output_dir / "figure2_metadata.json").write_text(
        json.dumps(
            {
                "figure": 2,
                "renderer": "scripts/render_published_figure2.py",
                "summary_tables": [str(path) for path in table_paths],
                "values": table.to_dict(orient="records"),
                "note": (
                    "Values and error bars are the explicit hardcodes used by the preprint notebook."
                    if table_paths == [DEFAULT_TABLE]
                    else "Values were supplied by external refit summary tables."
                ),
                "outputs": [str(pdf_path), str(png_path)],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
