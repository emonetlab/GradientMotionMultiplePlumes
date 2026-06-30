"""Minimal publication plotting style helpers for the panel scripts.

This module intentionally contains only the style/palette helpers needed by the
published panel renderers.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import yaml

DEFAULT_PLOT_CONFIG: dict[str, Any] = {
    "colors": {
        "cue": {
            "gradient": [1.0, 0.498, 0.055],
            "motion": [0.173, 0.627, 0.173],
        },
        "plume": {
            "complex": [0.737, 0.741, 0.133],
            "smooth": [0.580, 0.404, 0.741],
            "complex_1a": [0.737, 0.741, 0.133],
        },
    },
    "fonts": {
        "family": "Arial",
        "title_size": 12,
        "axis_label_size": 10,
        "tick_label_size": 10,
        "legend_size": 10,
        "annotation_size": 10,
    },
    "lines": {
        "linewidth": 1,
        "markersize": 4,
        "markeredgewidth": 0.8,
    },
    "ticks": {
        "direction": "in",
        "length": 3,
        "width": 1.5,
        "color": "black",
    },
    "paper": {
        "pdf_fonttype": 42,
        "transparent_background": True,
        "despine": True,
        "grid": False,
    },
    "save": {
        "dpi": 300,
    },
}


def load_plot_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML plotting config, or return the built-in paper defaults."""
    if config_path is None:
        return deepcopy(DEFAULT_PLOT_CONFIG)

    path = Path(config_path)
    if not path.exists():
        return deepcopy(DEFAULT_PLOT_CONFIG)

    with path.open("r") as f:
        loaded = yaml.safe_load(f) or {}
    config = deepcopy(DEFAULT_PLOT_CONFIG)
    _deep_update(config, loaded)
    return config


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def get_plot_config_value(
    config: dict[str, Any],
    key_path: str | list[str],
    default: Any = None,
) -> Any:
    """Get a nested config value using dotted path notation."""
    keys = key_path.split(".") if isinstance(key_path, str) else key_path
    current: Any = config
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def get_palette(config: dict[str, Any], palette_type: str) -> dict[str, Any]:
    return get_plot_config_value(config, f"colors.{palette_type}", {})


def get_plume_palette(config: dict[str, Any]) -> dict[str, Any]:
    return get_palette(config, "plume")


def apply_style_from_config(config: dict[str, Any], context: str = "paper") -> None:
    """Apply the exact compact paper plotting conventions used for these panels."""
    fonts = get_plot_config_value(config, "fonts", {}) or {}
    ticks = get_plot_config_value(config, "ticks", {}) or {}
    paper = get_plot_config_value(config, "paper", {}) or {}
    lines = get_plot_config_value(config, "lines", {}) or {}

    sns.set_theme(
        style="ticks",
        context=context,
        font_scale=float(fonts.get("scale", 1.0)),
        rc={
            "font.family": fonts.get("family", "Arial"),
            "axes.titlesize": fonts.get("title_size", 12),
            "axes.labelsize": fonts.get("axis_label_size", 10),
            "xtick.labelsize": fonts.get("tick_label_size", 10),
            "ytick.labelsize": fonts.get("tick_label_size", 10),
        },
    )

    plt.rcParams["pdf.fonttype"] = int(paper.get("pdf_fonttype", 42))
    plt.rcParams["ps.fonttype"] = int(paper.get("pdf_fonttype", 42))
    plt.rcParams["axes.spines.top"] = not bool(paper.get("despine", True))
    plt.rcParams["axes.spines.right"] = not bool(paper.get("despine", True))
    plt.rcParams["axes.grid"] = bool(paper.get("grid", False))
    plt.rcParams["xtick.direction"] = ticks.get("direction", "in")
    plt.rcParams["ytick.direction"] = ticks.get("direction", "in")
    plt.rcParams["xtick.major.size"] = float(ticks.get("length", 3))
    plt.rcParams["ytick.major.size"] = float(ticks.get("length", 3))
    plt.rcParams["xtick.major.width"] = float(ticks.get("width", 1.5))
    plt.rcParams["ytick.major.width"] = float(ticks.get("width", 1.5))
    plt.rcParams["xtick.color"] = ticks.get("color", "black")
    plt.rcParams["ytick.color"] = ticks.get("color", "black")
    plt.rcParams["lines.linewidth"] = float(lines.get("linewidth", 1))
    plt.rcParams["lines.markersize"] = float(lines.get("markersize", 4))
    plt.rcParams["lines.markeredgewidth"] = float(lines.get("markeredgewidth", 0.8))
    plt.rcParams["figure.constrained_layout.use"] = True
