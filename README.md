# GradientMotionMultiplePlumes

Publication code for *Fly navigational responses to odor motion and gradient cues are tuned to plume statistics* ([bioRxiv v1](https://doi.org/10.1101/2025.03.31.646361)).

This repository now contains scripted, tested pipelines for Figures 1, 2, and 3 in addition to the published behavior-regression panels. The Figure 1 and 3 pipelines deliberately keep large movies and model checkpoints outside Git; their renderers consume compact, provenance-carrying NPZ summaries.

## Quick start

```bash
conda env create -f environment.yml
conda activate gradient_motion_multiple_plumes
pip install -e ".[figures123,test]"  # adds PyTorch for Figure 3 training/extraction
python scripts/render_published_figure2.py
python -m pytest -q
```

Figure 2 writes editable PDF and PNG output to `figures/published_figures_1_3/`. See [`REPRODUCE_FIGURES_1_3.md`](REPRODUCE_FIGURES_1_3.md) for Figure 1 public-data commands and the Figure 3 training and checkpoint contracts.

## Behavior-regression panels

The existing Figure 5E/F and Figure S5 workflow remains available:

```bash
python scripts/render_published_behavior_panels.py
```

The behavior source is [DANDI dandiset 001871, version 0.260630.1657](https://dandiarchive.org/dandiset/001871/0.260630.1657). Compact published tables are checked in, and `scripts/generate_published_summary_tables.py` can regenerate them from the archived behavior NWB files. See [`REPRODUCE.md`](REPRODUCE.md).

## Provenance

The DANDI inputs are attributed as: Brudner, Samuel (2026), *Data for fly navigational responses exploiting plume-specific odor motion and gradient cues*, version 0.260630.1657, DANDI Archive, [doi:10.48324/dandi.001871/0.260630.1657](https://doi.org/10.48324/dandi.001871/0.260630.1657), CC BY 4.0. Smooth-plume data come from [Dryad doi:10.5061/dryad.g27mq71](https://doi.org/10.5061/dryad.g27mq71), CC0.
