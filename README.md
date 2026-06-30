# GradientMotionMultiplePlumes

Code to produce the published behavior-regression figure panels for the paper on fly navigational responses to odor motion and gradient cues across multiple plume environments.

## Data Reference

The source dataset is DANDI dandiset `001871`:

https://dandiarchive.org/dandiset/001871

This repository includes the small published summary tables needed to render the behavior-regression panels. The full archived dataset is referenced through DANDI.

## Published Panels

The fixed panel parameters are recorded in `metadata/published_panel_params.json`.

The included scripts produce:

- `fig5e_cue_beta`: cue beta panel for smooth and complex plumes.
- `fig5f_cue_dominance`: cue dominance panel for smooth and complex plumes.
- `figs5_total_differential`: total/differential model-comparison panels for each plume.

## Quick Start

```bash
conda env create -f environment.yml
conda run -n gradient_motion_multiple_plumes python scripts/render_published_behavior_panels.py
```

Outputs are written to `figures/published_behavior_panels/` by default.

## Panel Parameters

The published behavior panels use:

- Time scale: `200 ms`
- Horizontal position: `30 <= x <= 220 mm`
- Vertical position: `67 <= y <= 97 mm`
- Behavioral filter: `facing_upwind and walking_upwind and not near_margin`
- Predictors: `spatial_gradient`, `odor_velocity`, and `signal`

For the cue beta display, the gradient beta is multiplied by `-1`, as recorded in `metadata/published_panel_params.json`.
