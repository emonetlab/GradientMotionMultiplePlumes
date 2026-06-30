# GradientMotionMultiplePlumes

Code to produce published figure panels and quantitative visualizations for the paper on fly navigational responses to odor motion and gradient cues across multiple plume environments.

This repository is intended to house the paper-facing reproduction code for multiple classes of visualizations, including plume statistics, model performance, navigation simulations, and fly behavior analyses. The currently implemented workflow reproduces the behavior-regression panels listed below.

## Data Reference

The source dataset is DANDI dandiset `001871`, version `0.260630.1657`:

https://dandiarchive.org/dandiset/001871/0.260630.1657

This repository includes the small published summary tables used to render the currently implemented behavior-regression panels. It also includes code to regenerate those summary tables from the archived DANDI behavior NWB files.

## Implemented Panels

The fixed panel parameters are recorded in `metadata/published_panel_params.json`.

The included behavior-regression scripts produce:

- `fig5e_cue_beta`: cue beta panel for smooth and complex plumes.
- `fig5f_cue_dominance`: cue dominance panel for smooth and complex plumes.
- `figs5_total_differential`: total/differential model-comparison panels for each plume.

## Quick Start

Render the checked-in published summary tables:

```bash
conda env create -f environment.yml
conda run -n gradient_motion_multiple_plumes python scripts/render_published_behavior_panels.py
```

Regenerate the summary tables from DANDI behavior NWB files:

```bash
conda run -n gradient_motion_multiple_plumes python scripts/generate_published_summary_tables.py \
  --download-dandi \
  --dandi-cache data/dandi_cache/001871 \
  --output-dir data/generated_summary_tables
```

Render panels from regenerated summary tables by passing those CSVs to the panel scripts, or replace the checked-in `data/published_panel_tables/*.csv` after verifying the generated values.

Outputs are written to `figures/published_behavior_panels/` by default.

## Panel Parameters

The published behavior panels use:

- Time scale: `200 ms`
- Predictor lookup: `50 ms` before turn start
- Horizontal position: `30 <= x <= 220 mm`
- Vertical position: `67 <= y <= 97 mm`
- Behavioral filter: `facing_upwind and walking_upwind and not near_margin`
- Predictors: `spatial_gradient`, `odor_velocity`, and `signal`
- Smooth plume facing-upwind window: `160 <= theta <= 200 degrees`
- Complex plume facing-upwind window: `150 <= theta <= 210 degrees`

For the cue beta display, the gradient beta is multiplied by `-1`, as recorded in `metadata/published_panel_params.json`.
