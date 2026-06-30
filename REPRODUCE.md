# Reproduce Published Figure Panels

This repository contains paper-facing code to produce published figure panels and quantitative visualizations. The currently implemented workflow reproduces the behavior-regression panels and their compact summary tables.

## Data Source

Full dataset: DANDI dandiset `001871`, version `0.260630.1657`

https://dandiarchive.org/dandiset/001871/0.260630.1657

## Behavior Panels: Regenerate Summary Tables From DANDI NWB Files

Download the behavior NWB files for the smooth and complex plume experiments, reconstruct the turn-level predictors, and fit the fixed published models:

```bash
python scripts/generate_published_summary_tables.py \
  --download-dandi \
  --dandi-cache data/dandi_cache/001871 \
  --output-dir data/generated_summary_tables
```

If the DANDI files are already local, use:

```bash
python scripts/generate_published_summary_tables.py \
  --input-root /path/to/dandiset-001871 \
  --output-dir data/generated_summary_tables
```

The generator writes:

- `fig5e_cue_beta_smooth_plume.csv`
- `fig5e_cue_beta_complex_plume.csv`
- `fig5f_s5_model_comparison_smooth_plume.csv`
- `fig5f_s5_model_comparison_complex_plume.csv`
- `summary_generation_manifest.json`

## Behavior Panels: Render All Implemented Panels

```bash
python scripts/render_published_behavior_panels.py
```

The renderer reads `metadata/published_panel_params.json` and writes PNG/PDF outputs plus per-panel metadata to `figures/published_behavior_panels/`.

## Behavior Panels: Render One Panel

```bash
python scripts/render_published_behavior_panels.py --panel fig5e_cue_beta
python scripts/render_published_behavior_panels.py --panel fig5f_cue_dominance
python scripts/render_published_behavior_panels.py --panel figs5_total_differential
```

## Behavior Panel Inputs

Published summary tables are in `data/published_panel_tables/`:

- `fig5e_cue_beta_smooth_plume.csv`
- `fig5e_cue_beta_complex_plume.csv`
- `fig5f_s5_model_comparison_smooth_plume.csv`
- `fig5f_s5_model_comparison_complex_plume.csv`

## Behavior Panel Parameters

The parameters used by the summary-table generator and render commands are stored in `metadata/published_panel_params.json`:

- `timescale_ms = 200`
- `response_offset_s = 0.05`
- `x_min = 30`, `x_max = 220`
- `y_min = 67`, `y_max = 97`
- `behavioral_filter = facing_upwind and walking_upwind and not near_margin`
