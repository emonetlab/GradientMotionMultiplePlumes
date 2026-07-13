from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.centerline_models import (
    centerline_features,
    extract_centerline_features,
    log_threshold_transform,
)
from gradient_motion_panels.figure3_models import (
    class_one_probability,
    dense_input,
    gradient_probe,
    make_dense_network,
    make_minimal_network,
    motion_probe,
)
from gradient_motion_panels.plume_fields import (
    FrameAxisView,
    PlumeSummary,
    compute_plume_summary,
    finite_difference_y,
    gradient_and_motion,
    load_plume_summary,
    safe_standardized_mean,
    save_plume_summary,
)
from gradient_motion_panels.plume_sources import crop_and_pad_smooth_frame


def test_figure_metadata_and_published_table_are_complete() -> None:
    metadata = json.loads((REPO_ROOT / "metadata" / "figures_1_3.json").read_text())
    assert (
        metadata["source_code"]["preprint_commit"]
        == "659d0c3c34a8ab0f05abd76b38756debd4ea9214"
    )
    assert metadata["figure1"]["map_window"]["start_inclusive"] == 300
    assert metadata["figure1"]["map_window"]["stop_exclusive"] == 1800
    assert metadata["figure1"]["map_window"]["gradient_samples"] == 1500
    assert metadata["figure1"]["map_window"]["motion_samples"] == 1498
    table = pd.read_csv(
        REPO_ROOT / "data" / "published_panel_tables" / "fig2_glm_summary.csv"
    )
    assert len(table) == 6
    complex_motion = table.query("plume == 'complex' and feature == 'motion'").iloc[0]
    smooth_gradient = table.query("plume == 'smooth' and feature == 'gradient'").iloc[0]
    assert complex_motion["auc_mean"] == pytest.approx(0.618)
    assert smooth_gradient["weight_mean"] == pytest.approx(2.614)


def test_crosswind_difference_and_motion_match_published_equations() -> None:
    ramp = np.tile(np.arange(4, dtype=float)[:, None], (1, 3))
    assert np.allclose(finite_difference_y(ramp), 1.0)
    previous = ramp
    current = ramp + 2.0
    following = ramp + 6.0
    gradient, motion = gradient_and_motion(previous, current, following)
    assert np.allclose(gradient, 1.0)
    assert np.allclose(motion, -3.0)


def test_plume_summary_streaming_and_zero_variance() -> None:
    base = np.tile(np.arange(5, dtype=float)[:, None], (1, 6))
    frames = np.stack([base + index for index in range(7)])
    summary = compute_plume_summary(
        frames,
        snapshot_index=3,
        map_sigma=0,
        cue_snapshot_sigma=0,
        snapshot_sigma=0,
    )
    assert summary.metadata["gradient_samples"] == 7
    assert summary.metadata["motion_samples"] == 5
    assert summary.metadata["cue_snapshot_center_index"] == 4
    assert np.allclose(summary.gradient_mean, 1.0)
    assert np.allclose(summary.gradient_std, 0.0)
    assert np.allclose(summary.motion_mean, -1.0)
    assert np.allclose(
        safe_standardized_mean(summary.gradient_mean, summary.gradient_std), 1.0
    )


def test_plume_snapshot_and_cue_center_match_legacy_frame_alignment() -> None:
    y = np.tile(np.arange(5, dtype=float)[:, None], (1, 4))
    frames = np.stack([(index + 1) * y for index in range(6)])
    summary = compute_plume_summary(
        frames,
        snapshot_index=1,
        map_sigma=0,
        cue_snapshot_sigma=0,
        snapshot_sigma=0,
        frame_index_offset=300,
    )
    assert np.array_equal(summary.snapshot, frames[1])
    assert np.allclose(summary.gradient_snapshot, 3.0)
    assert summary.metadata["snapshot_full_video_frame"] == 301
    assert summary.metadata["cue_snapshot_center_full_video_frame"] == 302


def test_plume_summary_rejects_clamped_frame_windows() -> None:
    frames = np.zeros((5, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="outside a source"):
        compute_plume_summary(frames, snapshot_index=1, start=0, stop=6)


def test_frame_axis_view_rejects_invalid_axis() -> None:
    with pytest.raises(ValueError, match="invalid"):
        FrameAxisView(np.zeros((3, 4, 5)), time_axis=3)


def test_generate_figure1_summary_records_frame_coordinates(tmp_path: Path) -> None:
    movie = np.stack(
        [np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 4)) + i for i in range(6)]
    )
    source = tmp_path / "movie.npy"
    output = tmp_path / "summary.npz"
    np.save(source, movie)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_figure1_summary.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--plume",
            "complex",
            "--snapshot-index",
            "1",
            "--start",
            "0",
            "--stop",
            "6",
            "--frame-index-offset",
            "300",
            "--snapshot-role",
            "substitute",
        ],
        cwd=tmp_path,
        check=True,
    )
    summary = load_plume_summary(output)
    assert summary.metadata["frame_adapter"]["normalized_time_axis"] == 0
    assert summary.metadata["window_full_video_start"] == 300
    assert summary.metadata["window_full_video_stop"] == 306
    assert summary.metadata["snapshot_full_video_frame"] == 301
    assert summary.metadata["cue_snapshot_center_full_video_frame"] == 302
    assert summary.metadata["gradient_samples"] == 6
    assert summary.metadata["motion_samples"] == 4


def test_smooth_crop_pad_geometry() -> None:
    source = np.ones((406, 216), dtype=np.float32)
    result = crop_and_pad_smooth_frame(source)
    assert result.shape == (224, 351)
    assert np.all(result[:4] == 0)
    assert np.all(result[-4:] == 0)
    assert np.all(result[4:-4] == 1)


def test_figure2_features_and_sensor_swap_signs() -> None:
    traces = np.asarray([[[1, 4], [2, 5], [3, 6]]], dtype=float)
    features = centerline_features(traces, delay=1)
    assert np.allclose(features[0], [3.5, -3.0, -3.0])
    swapped = centerline_features(traces[..., ::-1], delay=1)
    assert swapped[0, 0] == pytest.approx(features[0, 0])
    assert np.allclose(swapped[0, 1:], -features[0, 1:])
    manuscript = centerline_features(traces, delay=1, convention="manuscript")
    assert manuscript[0, 0] == pytest.approx(features[0, 0])
    assert np.allclose(manuscript[0, 1:], -features[0, 1:])


def test_figure2_chunked_features_apply_legacy_nan_sanitation() -> None:
    samples = np.arange(36, dtype=np.float32).reshape(2, 9, 2)
    samples[0, 0, 0] = np.nan
    samples[1, 1, 1] = np.inf
    sanitized = np.nan_to_num(samples.copy())
    expected = centerline_features(log_threshold_transform(sanitized))
    actual = extract_centerline_features(samples, chunk_size=1)
    assert np.all(np.isfinite(actual))
    assert np.allclose(actual, expected)


def test_figure3_probe_shapes_and_seed() -> None:
    differences, gradient = gradient_probe()
    assert differences.shape == (11,)
    assert gradient.shape == (11, 30, 1, 2)
    shifts1, motion1 = motion_probe(sample_count=8, max_shift=2, seed=11)
    shifts2, motion2 = motion_probe(sample_count=8, max_shift=2, seed=11)
    assert np.allclose(shifts1, np.arange(-2, 3) / 60)
    assert all(np.array_equal(left, right) for left, right in zip(motion1, motion2))
    assert set(np.unique(motion1[0])).issubset({0.0, 2.0})


def test_checkpoint_compatible_models_are_antisymmetric_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    bilateral = rng.normal(size=(5, 30, 1, 2)).astype(np.float32)
    minimal = make_minimal_network()
    with torch.no_grad():
        direct = minimal(torch.as_tensor(bilateral))
        swapped = minimal(torch.as_tensor(bilateral[..., ::-1].copy()))
    assert torch.allclose(direct, -swapped, atol=1e-6)
    assert np.allclose(
        class_one_probability(direct) + class_one_probability(swapped), 1.0, atol=1e-6
    )

    dense = make_dense_network()
    flat = dense_input(bilateral)
    swapped_flat = np.concatenate((flat[:, 30:], flat[:, :30]), axis=1)
    with torch.no_grad():
        direct = dense(torch.as_tensor(flat))
        swapped = dense(torch.as_tensor(swapped_flat))
    assert torch.allclose(direct, -swapped, atol=1e-6)


def test_generate_figure3_summary_from_checkpoint_contract(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    minimal_path = tmp_path / "minimal.pth"
    dense_path = tmp_path / "dense.pth"
    torch.save(make_minimal_network().state_dict(), minimal_path)
    torch.save(make_dense_network().state_dict(), dense_path)
    output = tmp_path / "summary.npz"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_figure3_summary.py"),
            "--minimal-smooth",
            str(minimal_path),
            "--minimal-complex",
            str(minimal_path),
            "--minimal-smooth-probe",
            str(minimal_path),
            "--minimal-complex-probe",
            str(minimal_path),
            "--dense-smooth",
            str(dense_path),
            "--dense-complex",
            str(dense_path),
            "--motion-samples",
            "8",
            "--batch-size",
            "4",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    with np.load(output, allow_pickle=False) as bundle:
        assert bundle["minimal_smooth_filters"].shape == (2, 30)
        assert bundle["dense_complex_filters"].shape == (2, 2, 30)
        assert bundle["motion_shift_seconds"].shape == (13,)
        assert bundle["dense_motion_smooth"].shape == (13,)


def _synthetic_plume_summary(seed: int, plume: str) -> PlumeSummary:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:36j, -1:1:54j]
    snapshot = np.exp(-((y / (0.22 + 0.1 * (x + 1))) ** 2)) * (
        1 + 0.15 * rng.normal(size=x.shape)
    )
    gradient = np.gradient(snapshot, axis=0)
    motion = gradient * (0.2 * np.sin(5 * x))
    std = np.full_like(snapshot, 0.2)
    return PlumeSummary(
        snapshot=snapshot.astype(np.float32),
        gradient_snapshot=gradient.astype(np.float32),
        motion_snapshot=motion.astype(np.float32),
        gradient_mean=(0.08 * gradient).astype(np.float32),
        gradient_std=std.astype(np.float32),
        motion_mean=(0.08 * motion).astype(np.float32),
        motion_std=std.astype(np.float32),
        metadata={"plume": plume, "synthetic_test_fixture": True},
    )


def test_render_figures_1_2_3_smoke(tmp_path: Path) -> None:
    smooth = save_plume_summary(
        _synthetic_plume_summary(1, "smooth"), tmp_path / "smooth.npz"
    )
    complex_ = save_plume_summary(
        _synthetic_plume_summary(2, "complex"), tmp_path / "complex.npz"
    )
    out = tmp_path / "figures"
    swapped = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_published_figure1.py"),
            "--smooth-summary",
            str(complex_),
            "--complex-summary",
            str(smooth),
            "--output-dir",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert swapped.returncode != 0
    assert "Expected the smooth summary role" in swapped.stderr
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_published_figure1.py"),
            "--smooth-summary",
            str(smooth),
            "--complex-summary",
            str(complex_),
            "--output-dir",
            str(out),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_published_figure2.py"),
            "--output-dir",
            str(out),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    time = np.linspace(-0.5, 0, 30)
    difference = np.linspace(-0.5, 0.5, 11)
    shifts = np.arange(-6, 7) / 60
    figure3 = tmp_path / "figure3.npz"
    np.savez_compressed(
        figure3,
        time_seconds=time,
        minimal_smooth_filters=np.vstack((-np.exp(8 * time), np.exp(8 * time))),
        minimal_complex_filters=np.vstack((-np.exp(18 * time), np.exp(18 * time))) / 5,
        dense_smooth_filters=np.zeros((2, 2, 30)),
        dense_complex_filters=np.zeros((2, 2, 30)),
        gradient_difference=difference,
        minimal_gradient_smooth=1 / (1 + np.exp(-8 * difference)),
        minimal_gradient_complex=np.full(11, 0.5),
        dense_gradient_smooth=1 / (1 + np.exp(-5 * difference)),
        dense_gradient_complex=np.full(11, 0.5),
        motion_shift_seconds=shifts,
        minimal_motion_smooth=np.full(13, 0.5),
        minimal_motion_complex=0.5
        + 0.04 * np.exp(-((shifts / 0.025) ** 2)) * np.sign(shifts),
        dense_motion_smooth=np.full(13, 0.5),
        dense_motion_complex=0.5
        + 0.12 * np.exp(-((shifts / 0.025) ** 2)) * np.sign(shifts),
        metadata_json=np.asarray(json.dumps({"synthetic_test_fixture": True})),
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_published_figure3.py"),
            "--summary",
            str(figure3),
            "--output-dir",
            str(out),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    for figure in (1, 2, 3):
        assert (out / f"figure{figure}.pdf").exists()
        png = out / f"figure{figure}.png"
        assert png.exists()
        assert png.stat().st_size > 50_000
        from matplotlib.image import imread

        pixels = imread(png)
        assert pixels.shape[0] >= 1_000
        assert pixels.shape[1] >= 1_000
        assert float(np.std(pixels[..., :3])) > 0.05
        assert (out / f"figure{figure}_metadata.json").exists()
