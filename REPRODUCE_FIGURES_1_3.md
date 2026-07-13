# Reproduce Figures 1--3

The publication renderers separate large/raw inputs from compact plotting summaries. Figure 2 is self-contained. Figures 1 and 3 first generate an NPZ summary, then render from it.

All fixed values and provenance notes live in [`metadata/figures_1_3.json`](metadata/figures_1_3.json). The source reference is preprint commit [`659d0c3`](https://github.com/ClarkLabCode/OdorMotionMLdev/commit/659d0c3c34a8ab0f05abd76b38756debd4ea9214).

## Environment

```bash
conda env create -f environment.yml
conda activate gradient_motion_multiple_plumes
pip install -e ".[figures123,test]"
```

PyTorch is needed only to extract Figure 3 summaries from checkpoints. Rendering existing summaries does not import PyTorch.

## Figure 1

The implementation computes the crosswind centered difference `dI/dy`, the motion cue `-(dI/dy)(dI/dt)`, and per-pixel temporal mean/std maps. In the 1500-frame analysis window, gradient statistics use all 1500 frames and motion statistics use the 1498 frames with temporal neighbors. Whole-window maps use no spatial smoothing. Snapshot panels use Gaussian sigma 3 (plume) and 1.5 (cue maps). As in the notebook, a plume snapshot at frame `n` is paired with cue maps centered at frame `n+1` from the triplet `n:n+3`; both indices are recorded.

### Smooth plume

Download `10302017_10cms_bounded_2.h5` from [Dryad dataset 10.5061/dryad.g27mq71](https://doi.org/10.5061/dryad.g27mq71). The input dataset is `/dataset2`, shaped `(3600, 406, 216)` at 15 Hz.

```bash
python scripts/generate_figure1_summary.py \
  --input data/raw/10302017_10cms_bounded_2.h5 \
  --dataset /dataset2 \
  --plume smooth \
  --smooth-profile notebook_legacy \
  --snapshot-index 601 \
  --start 300 --stop 1800 \
  --hash-input \
  --output data/generated/figure1_smooth.npz \
  --source-url https://doi.org/10.5061/dryad.g27mq71
```

`notebook_legacy` reproduces the committed conversion notebook's duplicated first frame and omitted final frame. `corrected` uses linear timestamps through both endpoints. The configured published AVI has a different filename from the notebook output, so the repository records this timing uncertainty instead of treating either profile as proven exact. Dryad reports MD5 `cf3cadd8a9b53c2a3f6d3aca37f71a62` for file id 11167; `--hash-input` additionally records SHA-256 in the generated summary.

### Complex plume

For the published complex data panels, the required input is the full 3600-frame, background-subtracted `smoke_1a_orig_backgroundsubtracted` sequence. Its original AVI-derived intensity range is approximately 0--120, so the archived pipeline applies `255/120 = 2.125`:

```bash
python scripts/generate_figure1_summary.py \
  --input /path/to/full_complex_movie.npy \
  --plume complex \
  --snapshot-index 2900 \
  --start 300 --stop 1800 \
  --intensity-scale 2.125 \
  --hash-input \
  --source-url /record/the/source/or/archive/URL \
  --output data/generated/figure1_complex.npz
```

If the input has already been scaled to approximately 0--255, use `--intensity-scale 1` and record that preprocessing in the source URL or adjacent provenance notes.

The public [DANDI 001871 Figure S1 NWB asset](https://api.dandiarchive.org/api/assets/5e60d590-b1c1-4830-bb38-a36f6d967e75/) contains exactly the raw frames 300--1799 needed for the temporal maps, under `/scratch/smoke_1a_orig_figs1_background_subtracted_window_frames`. Download and summarize it as follows. Loaded row 0 is full-video frame 300, so the correct local analysis window is `0:1500`, not `300:1800`. The example deliberately substitutes full-video frame 601 for unavailable frame 2900 and records that fact.

```bash
mkdir -p data/raw
curl -L \
  https://api.dandiarchive.org/api/assets/5e60d590-b1c1-4830-bb38-a36f6d967e75/download/ \
  -o data/raw/sub-figs1-complex-smoke-1a_ses-figure-s1_image.nwb

python scripts/generate_figure1_summary.py \
  --input data/raw/sub-figs1-complex-smoke-1a_ses-figure-s1_image.nwb \
  --dataset /scratch/smoke_1a_orig_figs1_background_subtracted_window_frames \
  --plume complex \
  --snapshot-index 301 \
  --start 0 --stop 1500 \
  --frame-index-offset 300 \
  --intensity-scale 1 \
  --snapshot-role substitute \
  --snapshot-note "DANDI row 301 / full-video frame 601; published frame 2900 unavailable" \
  --hash-input \
  --source-url https://doi.org/10.48324/dandi.001871/0.260630.1657 \
  --output data/generated/figure1_complex_dandi_validation.npz
```

This public summary validates I--L. Its C/F/H panels are labeled as a substitute in summary metadata and are not the published snapshot.

The DANDI scratch frames are already scaled to a legacy-like 0--255 range. Keep `--intensity-scale 1`. The old `255/120` multiplier applies only to the original low-range AVI and must not be applied twice.

### Render

```bash
python scripts/render_published_figure1.py \
  --smooth-summary data/generated/figure1_smooth.npz \
  --complex-summary data/generated/figure1_complex.npz
```

## Figure 2

The exact values and error bars hardcoded in the preprint notebook are checked into `data/published_panel_tables/fig2_glm_summary.csv`.

```bash
python scripts/render_published_figure2.py
```

To refit from legacy arrays, repeat each path option once per independent dataset repeat. Generate the two plume tables separately, then pass both to the renderer:

```bash
python scripts/generate_figure2_summary.py \
  --plume smooth \
  --train-data /path/train_data_R1.npy \
  --train-labels /path/train_labels_R1.npy \
  --test-data /path/test_data_R1.npy \
  --test-labels /path/test_labels_R1.npy \
  --output data/generated/figure2_smooth.csv

python scripts/generate_figure2_summary.py \
  --plume complex \
  --train-data /path/train_data_R1.npy \
  --train-labels /path/train_labels_R1.npy \
  --test-data /path/test_data_R1.npy \
  --test-labels /path/test_labels_R1.npy \
  --output data/generated/figure2_complex.csv

python scripts/render_published_figure2.py \
  --table data/generated/figure2_smooth.csv \
  --table data/generated/figure2_complex.csv
```

Legacy arrays are `(sample, 30, 1, 2)`, with channel 0 = right and channel 1 = left. The code applies the notebook's `nan_to_num` sanitation, computes features in 100,000-sample chunks, and then standardizes using training-set statistics. Override memory use with `--chunk-size`. The executable source computes `0.5*mean(R+L)`, `mean(R-L)`, and `mean(R(t)L(t+1)-R(t+1)L(t))`; the manuscript omits the factor 0.5 in the sum and shows the opposite gradient/motion signs if L/R labels are read literally. Standardization makes AUC and fitted standardized weights invariant to the sum scaling, while coefficient signs still depend on convention.

The published complex error bars were explicit notebook hardcodes and differ from standard deviations printed from the loaded result arrays. The checked-in table preserves the values actually plotted.

## Figure 3

Generate a compact summary from the six checkpoint roles used by the notebook:

```bash
python scripts/generate_figure3_summary.py \
  --minimal-smooth /path/minimal_smooth_R1/model.pth \
  --minimal-complex /path/minimal_complex_R1/model.pth \
  --minimal-smooth-probe /path/minimal_smooth_R2/model.pth \
  --minimal-complex-probe /path/minimal_complex_R2/model.pth \
  --dense-smooth /path/dense_smooth_R1/model.pth \
  --dense-complex /path/dense_complex_R1/model.pth \
  --seed 0 \
  --output data/generated/figure3_summary.npz

python scripts/render_published_figure3.py \
  --summary data/generated/figure3_summary.npz
```

The original synthetic probes did not set a random seed; this implementation defaults to seed 0 and records it in the summary metadata. The minimal filter/AUC panels used initialization R1, while its synthetic-response panels used R2. Dense panels used R1.

The minimal checkpoints and center-shifted complex dense checkpoint were not found in source commit `659d0c3` or the Dryad/DANDI deposits inspected on 2026-07-13. A locally discovered older sparse dense checkpoint produces AUC 0.610 rather than the published 0.543 and is not a valid substitute. The generator requires all six roles and fails on missing or shape-incompatible checkpoints rather than fabricating curves. Recorded SHA-256 values identify user-supplied inputs; they do not authenticate a checkpoint as the one used for publication.

## Render several figures

```bash
python scripts/render_published_figures_1_3.py \
  --smooth-summary data/generated/figure1_smooth.npz \
  --complex-summary data/generated/figure1_complex.npz \
  --figure3-summary data/generated/figure3_summary.npz
```

Use `--figure 2` to render only self-contained Figure 2.

## Verification

```bash
python -m pytest -q
```

Tests cover the spatial/temporal finite differences, motion sign, temporal statistics, smooth-plume crop/pad geometry, bilateral feature signs, checkpoint antisymmetry when PyTorch is installed, exact published Figure 2 values, and smoke rendering of all three complete layouts.
