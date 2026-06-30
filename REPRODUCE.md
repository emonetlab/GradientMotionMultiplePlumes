# Reproduce Published Behavior Panels

This repository contains the code and compact summary tables required to produce the published behavior-regression panels.

## Data Source

Full dataset: DANDI dandiset `001871`

https://dandiarchive.org/dandiset/001871

## Render All Panels

```bash
python scripts/render_published_behavior_panels.py
```

The renderer reads `metadata/published_panel_params.json` and writes PNG/PDF outputs plus per-panel metadata to `figures/published_behavior_panels/`.

## Render One Panel

```bash
python scripts/render_published_behavior_panels.py --panel fig5e_cue_beta
python scripts/render_published_behavior_panels.py --panel fig5f_cue_dominance
python scripts/render_published_behavior_panels.py --panel figs5_total_differential
```

## Inputs

Published summary tables are in `data/published_panel_tables/`:

- `fig5e_cue_beta_smooth_plume.csv`
- `fig5e_cue_beta_complex_plume.csv`
- `fig5f_s5_model_comparison_smooth_plume.csv`
- `fig5f_s5_model_comparison_complex_plume.csv`

## Fixed Published Parameters

The parameters used by all render commands are stored in `metadata/published_panel_params.json`:

- `timescale_ms = 200`
- `x_min = 30`, `x_max = 220`
- `y_min = 67`, `y_max = 97`
- `behavioral_filter = facing_upwind and walking_upwind and not near_margin`
