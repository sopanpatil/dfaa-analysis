# figures/

Figure-generation scripts for the manuscript. All read directly from the
main pipeline's output directories (Stages 2-7 in the top-level README) and
compute displayed values live rather than hardcoding them, so they will
correctly reflect the results of a full pipeline re-run.

## Contents

- `build_figure1_transition_metrics.py` — Figure 1 (transition gap and
  depletion/recovery rate by RCP, both directions). Reads
  `../aggregated_output/` and `../d2f_aggregated_output/`.
- `build_figure2_cluster_map.py` — Figure 2 (FTD and DTF catchment typology
  maps, both k=2). Reads `../typology_output/`, `../d2f_typology_output/`,
  `../great_britain_boundary.geojson`, and `./catchment_boundaries/`.
- `build_figure3_morans_i.py` — Figure 3 (Moran's I heatmap, both
  directions, all four RCPs). Reads `../spatial_output/` and
  `../d2f_spatial_output/`.
- `build_figure4_snow_relationships.py` — Figure 4 (static/dynamic snow
  correlation strength and incremental R² gain). Reads `../spatial_output/`,
  `../d2f_spatial_output/`, and `../d2f_regression_output/`.

## External data dependencies (not bundled in this repository)

- **CAMELS-GB v2 catchment boundaries** (`catchment_boundaries/camels_gb_v2_catchment_boundaries.shp`,
  plus its `.shx`/`.dbf`/`.prj` companions): catchment boundary polygons,
  keyed on `ID_STRING`. Part of the same CAMELS-GB v2 dataset cited in the
  top-level README (Coxon et al., 2026). Place inside `figures/catchment_boundaries/`
  — these are only needed here, not by the main pipeline.
- **ONS Countries boundary** (`great_britain_boundary.geojson`): same file
  used by `spatial_pattern_f2d.py`/`spatial_pattern_d2f.py` in the main
  pipeline (see top-level README) — place it at the repository root, not
  inside `figures/`.

## Running

Run from within this folder, after the main pipeline (Stages 2-7) has
produced the output directories listed above:

```bash
cd figures
python build_figure1_transition_metrics.py
python build_figure2_cluster_map.py
python build_figure3_morans_i.py
python build_figure4_snow_relationships.py
```

Each writes its PNG directly into this folder.

## Requirements

In addition to the top-level pipeline's requirements: `matplotlib`,
`geopandas` (Figure 2 only).
