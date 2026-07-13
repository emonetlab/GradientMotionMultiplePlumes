# GradientMotionMultiplePlumes

Publication code for *Fly navigational responses to odor motion and gradient cues are tuned to plume statistics* ([bioRxiv v1](https://doi.org/10.1101/2025.03.31.646361)).

This repository now contains scripted, tested pipelines for Figures 1, 2, and 3 in addition to the published behavior-regression panels. The Figure 1 and 3 pipelines deliberately keep large movies and model checkpoints outside Git; their renderers consume compact, provenance-carrying NPZ summaries.

## Reproduction status

| Figure | Included code | External requirement |
|---|---|---|
| 1 | Plume conversion, finite-difference gradient/motion maps, streaming temporal statistics, and a programmatic all-panel layout | Public Dryad/DANDI movies; the published complex frame-2900 snapshot needs the full original background-subtracted movie |
| 2 | Published B--E values, a programmatic A schematic, and memory-bounded GLM refitting | Nothing for the published-value renderer; legacy train/test arrays only for refitting |
| 3 | Checkpoint-compatible MNM/DNM definitions, seeded probes, summary extraction, and a programmatic all-panel layout | The MNM and center-shifted complex DNM checkpoints used by the notebook were not found in the inspected sources |

Panels that were originally assembled as artwork are represented by new programmatic schematics. Quantitative panels implement the archived equations and checked-in values where those values are available. Figure 3 probes are seeded for repeatability, unlike the unseeded notebook, and exact rendering remains checkpoint-dependent. The discrepancies and missing artifacts established during reconstruction are recorded in [`metadata/figures_1_3.json`](metadata/figures_1_3.json).

## Quick start

```bash
conda env create -f environment.yml
conda activate gradient_motion_multiple_plumes
pip install -e ".[figures123,test]"  # adds PyTorch for Figure 3 extraction
python scripts/render_published_figure2.py
python -m pytest -q
```

Figure 2 writes editable PDF and PNG output to `figures/published_figures_1_3/`. See [`REPRODUCE_FIGURES_1_3.md`](REPRODUCE_FIGURES_1_3.md) for Figure 1 public-data commands and the Figure 3 checkpoint contract.

## Behavior-regression panels

The existing Figure 5E/F and Figure S5 workflow remains available:

```bash
python scripts/render_published_behavior_panels.py
```

The behavior source is [DANDI dandiset 001871, version 0.260630.1657](https://dandiarchive.org/dandiset/001871/0.260630.1657). Compact published tables are checked in, and `scripts/generate_published_summary_tables.py` can regenerate them from the archived behavior NWB files. See [`REPRODUCE.md`](REPRODUCE.md).

## Provenance

The Figure 1--3 implementation was reconstructed against the preprint and the preprint-era source revision [`659d0c3`](https://github.com/ClarkLabCode/OdorMotionMLdev/commit/659d0c3c34a8ab0f05abd76b38756debd4ea9214). The original repository has no declared software license, so this repository uses a focused reimplementation of the published equations and parameters rather than copying the notebooks wholesale. Missing-file statements are scoped to that revision and the Dryad/DANDI deposits inspected on 2026-07-13.

The DANDI inputs are attributed as: Brudner, Samuel (2026), *Data for fly navigational responses exploiting plume-specific odor motion and gradient cues*, version 0.260630.1657, DANDI Archive, [doi:10.48324/dandi.001871/0.260630.1657](https://doi.org/10.48324/dandi.001871/0.260630.1657), CC BY 4.0. Smooth-plume data come from [Dryad doi:10.5061/dryad.g27mq71](https://doi.org/10.5061/dryad.g27mq71), CC0.
