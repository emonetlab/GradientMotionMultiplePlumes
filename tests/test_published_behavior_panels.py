from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO_ROOT / "metadata" / "published_panel_params.json"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_published_panel_params_reference_existing_files() -> None:
    params = json.loads(PARAMS_PATH.read_text())
    for panel in params["panels"].values():
        assert (REPO_ROOT / panel["script"]).exists()
        assert (REPO_ROOT / panel["smooth_table"]).exists()
        assert (REPO_ROOT / panel["complex_table"]).exists()


def test_render_all_published_behavior_panels(tmp_path: Path) -> None:
    out_dir = tmp_path / "published_behavior_panels"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_published_behavior_panels.py"),
            "--output-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    expected_dirs = [
        out_dir / "fig5e_cue_beta",
        out_dir / "fig5f_cue_dominance",
        out_dir / "figs5_total_differential",
    ]
    for panel_dir in expected_dirs:
        assert (panel_dir / "panel_metadata.json").exists()
        assert list(panel_dir.glob("*.png")), panel_dir
        assert list(panel_dir.glob("*.pdf")), panel_dir

    total_panel_dir = out_dir / "figs5_total_differential"
    assert (total_panel_dir / "figs5_total_differential_smooth.pdf").exists()
    assert (total_panel_dir / "figs5_total_differential_complex.pdf").exists()


def test_generated_pdfs_use_editable_truetype_fonts(tmp_path: Path) -> None:
    pdffonts_bin = shutil.which("pdffonts")
    if pdffonts_bin is None:
        return

    out_dir = tmp_path / "published_behavior_panels"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_published_behavior_panels.py"),
            "--output-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    for pdf_path in out_dir.glob("*/*.pdf"):
        font_output = subprocess.check_output([pdffonts_bin, str(pdf_path)], text=True)
        rows = [line for line in font_output.splitlines() if line.strip()][2:]
        assert rows, f"No embedded fonts detected in {pdf_path}\n{font_output}"
        assert all("TrueType" in row for row in rows), font_output


def _synthetic_turn_table(seed: int, *, motion_weight: float, gradient_weight: float):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for cluster_idx in range(12):
        for row_idx in range(28):
            motion = rng.normal()
            gradient = rng.normal()
            signal = rng.normal()
            eta = motion_weight * motion + gradient_weight * gradient + 0.15 * signal + rng.normal(scale=0.7)
            turn_direction = 1.0 if eta > 0 else -1.0
            rows.append(
                {
                    "cluster_id": f"cluster-{cluster_idx}",
                    "turn_direction": turn_direction,
                    "turn_start_t": float(row_idx),
                    "turn_end_t": float(row_idx) + 0.1,
                    "turn_x": 120.0 + rng.normal(scale=10.0),
                    "turn_y": 82.0 + rng.normal(scale=5.0),
                    "turn_theta": 180.0,
                    "near_margin": False,
                    "facing_upwind": True,
                    "walking_upwind": True,
                    "odor_velocity_esmooth_200ms": motion,
                    "spatial_gradient_esmooth_200ms": gradient,
                    "signal_esmooth_200ms": signal,
                }
            )
    return pd.DataFrame(rows)


def test_generate_summary_tables_from_prepared_turn_tables(tmp_path: Path) -> None:
    import pandas as pd

    smooth = _synthetic_turn_table(1, motion_weight=0.1, gradient_weight=-1.0)
    complex_ = _synthetic_turn_table(2, motion_weight=1.0, gradient_weight=0.1)
    smooth_path = tmp_path / "smooth_turns.parquet"
    complex_path = tmp_path / "complex_turns.parquet"
    smooth.to_parquet(smooth_path, index=False)
    complex_.to_parquet(complex_path, index=False)
    out_dir = tmp_path / "summary_tables"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_published_summary_tables.py"),
            "--smooth-turn-table",
            str(smooth_path),
            "--complex-turn-table",
            str(complex_path),
            "--output-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    expected = [
        "fig5e_cue_beta_smooth_plume.csv",
        "fig5e_cue_beta_complex_plume.csv",
        "fig5f_s5_model_comparison_smooth_plume.csv",
        "fig5f_s5_model_comparison_complex_plume.csv",
        "summary_generation_manifest.json",
    ]
    for name in expected:
        assert (out_dir / name).exists(), name

    beta = pd.read_csv(out_dir / "fig5e_cue_beta_smooth_plume.csv")
    assert list(beta["variant"]) == ["mother"]
    assert {"beta_motion", "p_motion", "beta_gradient", "p_gradient"}.issubset(beta.columns)

    comparison = pd.read_csv(out_dir / "fig5f_s5_model_comparison_complex_plume.csv")
    assert set(comparison["variant"]) == {"mother", "vel_only", "grad_only", "joint_drop"}
    assert comparison["log_likelihood_mean"].notna().all()


def test_prepare_turn_table_from_timeseries_reconstructs_predictors() -> None:
    import numpy as np
    import pandas as pd

    from gradient_motion_panels.summary_tables import (
        DEFAULT_PLUMES,
        PublishedParams,
        prepare_turn_table_from_timeseries,
    )

    t = np.arange(0, 2.0, 1 / 60.0)
    timeseries = pd.DataFrame(
        {
            "cluster_id": "s1:1",
            "session_id": "s1",
            "full_trjn": "1",
            "t": t,
            "x": 100.0 + t,
            "y": 82.0 + 0.1 * t,
            "theta": 180.0,
            "vx": -np.ones_like(t),
            "odor_velocity": np.sin(t),
            "spatial_gradient": np.cos(t),
            "signal": np.ones_like(t),
        }
    )
    turns = pd.DataFrame(
        {
            "cluster_id": ["s1:1", "s1:1"],
            "turn_start_t": [0.5, 1.0],
            "turn_end_t": [0.6, 1.1],
            "turn_direction": [1.0, -1.0],
            "turn_x": [100.5, 101.0],
            "turn_y": [82.0, 82.1],
            "turn_theta": [180.0, 180.0],
        }
    )
    params = PublishedParams(timescale_ms=200, arena_bounds_quantile=0.0)
    out = prepare_turn_table_from_timeseries(timeseries, turns, plume_spec=DEFAULT_PLUMES["smooth"], params=params)
    assert len(out) == 2
    assert out["facing_upwind"].all()
    assert out["walking_upwind"].all()
    assert "odor_velocity_esmooth_200ms" in out.columns
    assert "spatial_gradient_esmooth_200ms" in out.columns
    assert "signal_esmooth_200ms" in out.columns
