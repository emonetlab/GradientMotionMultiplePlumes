# Reproduce Figures 1--3

The publication renderers separate large/raw inputs from compact plotting summaries. Figure 2 is self-contained. Figures 1 and 3 first generate an NPZ summary, then render from it.

All fixed values and provenance notes live in [`metadata/figures_1_3.json`](metadata/figures_1_3.json). The source reference is preprint commit [`659d0c3`](https://github.com/ClarkLabCode/OdorMotionMLdev/commit/659d0c3c34a8ab0f05abd76b38756debd4ea9214).

## Environment

```bash
conda env create -f environment.yml
conda activate gradient_motion_multiple_plumes
pip install -e ".[figures123,test]"
```

PyTorch is needed to train Figure 3 models or extract summaries from checkpoints. Rendering existing summaries does not import PyTorch.

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

### Legacy-array contract

Training starts from the precomputed NumPy arrays used by the archived source. For each plume, supply all four files:

| File | Full-data shape | Role |
|---|---:|---|
| `train_data_R1.npy` | `(2_400_000, 30, 1, 2)` | Bilateral 0.5-second training histories |
| `train_labels_R1.npy` | `(2_400_000,)` | Binary centerline-side labels |
| `test_data_R1.npy` | `(600_000, 30, 1, 2)` | Bilateral test histories |
| `test_labels_R1.npy` | `(600_000,)` | Binary test labels |

Labels may instead have shape `(N, 1)`, but every value must be 0 or 1 and the training set must contain both classes. Sample channel 0 is the right antenna and channel 1 is the left antenna. The literal archived transform is `log(max(I - 5, 0) + 1)`. Minimal-model training and final evaluation first apply `nan_to_num`; archived dense-model training did not, so the trainer preserves that asymmetry and fails clearly if it produces a non-finite loss. Dense inputs concatenate all 30 right-channel values followed by all 30 left-channel values; a plain reshape would interleave the channels and is not equivalent.

The model repeat and the data repeat are different concepts. Every archived model initialization, including model R2, loaded the files ending in `R1.npy`. R2 is a second model initialization, not a request for `train_data_R2.npy`.

The trainer memory-maps the NPY files and preprocesses one batch at a time. `--max-train-samples` and `--max-test-samples` are useful for smoke tests, but a checkpoint trained with either limit is not a full-data Figure 3 model.

### Train compatible models

The following four commands create all six checkpoint roles used by Figure 3. The two R1 commands train both model types; the two R2 commands train only the minimal model. The explicit seeds match the CLI defaults of `repeat - 1`.

```bash
# Smooth R1: minimal and dense
python scripts/train_figure3_models.py \
  --train-data data/raw/figure3/smooth/train_data_R1.npy \
  --train-labels data/raw/figure3/smooth/train_labels_R1.npy \
  --test-data data/raw/figure3/smooth/test_data_R1.npy \
  --test-labels data/raw/figure3/smooth/test_labels_R1.npy \
  --plume smooth \
  --models minimal dense \
  --profile archived-source \
  --repeat 1 --seed 0 \
  --hash-inputs \
  --output-root data/generated/figure3_models

# Complex R1: minimal and dense
python scripts/train_figure3_models.py \
  --train-data data/raw/figure3/complex/train_data_R1.npy \
  --train-labels data/raw/figure3/complex/train_labels_R1.npy \
  --test-data data/raw/figure3/complex/test_data_R1.npy \
  --test-labels data/raw/figure3/complex/test_labels_R1.npy \
  --plume complex \
  --models minimal dense \
  --profile archived-source \
  --repeat 1 --seed 0 \
  --hash-inputs \
  --output-root data/generated/figure3_models

# Smooth R2: minimal probe model only
python scripts/train_figure3_models.py \
  --train-data data/raw/figure3/smooth/train_data_R1.npy \
  --train-labels data/raw/figure3/smooth/train_labels_R1.npy \
  --test-data data/raw/figure3/smooth/test_data_R1.npy \
  --test-labels data/raw/figure3/smooth/test_labels_R1.npy \
  --plume smooth \
  --models minimal \
  --profile archived-source \
  --repeat 2 --seed 1 \
  --hash-inputs \
  --output-root data/generated/figure3_models

# Complex R2: minimal probe model only
python scripts/train_figure3_models.py \
  --train-data data/raw/figure3/complex/train_data_R1.npy \
  --train-labels data/raw/figure3/complex/train_labels_R1.npy \
  --test-data data/raw/figure3/complex/test_data_R1.npy \
  --test-labels data/raw/figure3/complex/test_labels_R1.npy \
  --plume complex \
  --models minimal \
  --profile archived-source \
  --repeat 2 --seed 1 \
  --hash-inputs \
  --output-root data/generated/figure3_models
```

These commands produce:

```text
data/generated/figure3_models/
  minimal/smooth/R1/model.pth
  minimal/smooth/R2/model.pth
  minimal/complex/R1/model.pth
  minimal/complex/R2/model.pth
  dense/smooth/R1/model.pth
  dense/complex/R1/model.pth
```

The default `archived-source` profile follows the values actually executed by the preprint-era source:

| Model | Plume | Epochs | Batch size | Adam learning rate |
|---|---|---:|---:|---:|
| Minimal | Smooth or complex | 300 | 100 | `1e-5` |
| Dense | Smooth | 500 | 500 | `1e-5` |
| Dense | Complex | 500 | 500 | `1e-4` |

The paper Methods instead state 500 epochs, batch size 500, and learning rate `1e-4` for every model and plume. Select those values with `--profile paper-methods`. The differences are retained as named profiles because neither should be silently presented as the other. Individual values can be overridden with `--minimal-epochs`, `--minimal-batch-size`, `--minimal-learning-rate`, and the corresponding `--dense-*` options; all resolved values are recorded in the run metadata.

For source-like ordering, the default leaves samples unshuffled. `--shuffle` is an intentional departure. The default device is CPU with one Torch thread. `--device auto`, `--device cuda`, or another supported PyTorch device can accelerate training, but numerical results are not guaranteed to be bit-identical across devices or PyTorch versions.

Each run directory contains:

- `model.pth`: final raw CPU `state_dict`, directly accepted by `generate_figure3_summary.py`;
- `model_init.pth`: seeded initial state after the extra parameter reset used by the archived trainer;
- `train_loss.pth`: one mean loss per completed epoch;
- `training_state.pth`: model, optimizer, loss history, and NumPy/Torch/CUDA RNG state for resuming; and
- `training_metadata.json`: input shapes and dtypes, class counts, resolved configuration, effective sample counts, final test metrics, runtime versions, hashes, and provenance notes.

`training_state.pth` is updated atomically after every epoch. To continue an interrupted run, reissue the same command with `--resume`; the target epoch count may be increased, but the training inputs, hash mode, resolved device, runtime versions, and other training settings must match. In a combined minimal/dense invocation, `--resume` continues model directories that have state and starts requested model directories that have not yet been created; an inconsistent partial directory still fails safely. Use `--overwrite` to discard existing artifacts and restart. Without either option, existing output files cause a failure instead of being replaced.

Only one trainer may write a run directory at a time; a concurrent invocation fails before touching training artifacts, and the operating system releases the lock if the process exits. Final evaluation and input-integrity checks occur before replacing `model.pth`. The metadata file is published last as the completion marker, so an interrupted final publication cannot present a new checkpoint with stale `status: complete` metadata.

Resume granularity is one epoch. An interruption within an epoch repeats that epoch from its beginning; with the full arrays, an archived-source epoch contains 24,000 minimal-model batches or 4,800 dense-model batches. The original dense trainer also evaluated test cross-entropy after every epoch and saved validation/prediction artifacts. Those observations never affected optimization or checkpoint selection, so this checkpoint-focused routine evaluates once at the final epoch and does not claim full legacy artifact parity.

The initial and final checkpoint SHA-256 values are always recorded. Input hashing is also enabled by default; the four commands use `--hash-inputs` to make that archival choice explicit. Input identity and hashes are captured before training and checked again afterward. Use `--no-hash-inputs` when the additional full-file I/O over the large NPY arrays is undesirable; their paths, shapes, dtypes, sizes, and modification times are still checked, but their SHA-256 fields are null. The selected seed, initialization reset, CPU-thread count, deterministic-algorithm flag, cuDNN setting, and CUDA workspace configuration are also recorded. The archived source did not seed NumPy or PyTorch, so a new seeded R1 or R2 run is a deterministic reimplementation, not the recovered publication initialization.

### Generate and render the summary

Generate a compact summary from the six checkpoint roles used by the notebook:

```bash
python scripts/generate_figure3_summary.py \
  --minimal-smooth data/generated/figure3_models/minimal/smooth/R1 \
  --minimal-complex data/generated/figure3_models/minimal/complex/R1 \
  --minimal-smooth-probe data/generated/figure3_models/minimal/smooth/R2 \
  --minimal-complex-probe data/generated/figure3_models/minimal/complex/R2 \
  --dense-smooth data/generated/figure3_models/dense/smooth/R1 \
  --dense-complex data/generated/figure3_models/dense/complex/R1 \
  --seed 0 \
  --output data/generated/figure3_summary.npz

python scripts/render_published_figure3.py \
  --summary data/generated/figure3_summary.npz
```

The original synthetic probes did not set a random seed; this implementation defaults to seed 0 and records it in the summary metadata. The minimal filter/AUC panels used initialization R1, while its synthetic-response panels used R2. Dense panels used R1.

The summary generator defaults to dense first-layer units 3 and 11 for smooth and 1 and 12 for complex because those units were selected in the publication checkpoints. Hidden units are permutation-equivalent, so these fixed indices are not meaningful for a newly trained dense model. Each dense run's `training_metadata.json` contains `suggested_dense_units_by_weight_norm`. Inspect the suggested filters, then pass the chosen indices with `--dense-smooth-units UNIT_A UNIT_B` and `--dense-complex-units UNIT_A UNIT_B`. The summary metadata records the selected indices. The norm-ranked suggestions are an auditable starting point, not a claim that they reproduce the publication's hand-selected filters.

The legacy train/test arrays, threshold masks, minimal checkpoints, and center-shifted complex dense checkpoint were not found in source commit `659d0c3` or the Dryad/DANDI deposits inspected on 2026-07-13. The public complex DANDI movie begins at full-video frame 300, whereas archived neural-network training clips begin at frame 240. The committed data generator also has execution errors and an effective Gaussian sigma of 1.0 despite the configured and Methods value of 1.5. Consequently, neither the public movies nor the committed generator provide an authenticated route to the missing arrays. A locally discovered older sparse dense checkpoint produces AUC 0.610 rather than the published 0.543 and is not a valid substitute.

The trainer can generate checkpoint-compatible models when the legacy arrays are supplied, but it cannot recover the missing publication initialization, guarantee the published AUCs, or make a newly selected pair of dense filters identical to the plotted pair. The summary generator requires all six roles and fails on missing or shape-incompatible checkpoints rather than fabricating curves. Recorded SHA-256 values identify supplied files; hashes do not authenticate their publication role.

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
