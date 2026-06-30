from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO_ROOT / "metadata" / "published_panel_params.json"


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
