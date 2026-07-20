# dfaa-analysis

Analysis pipeline for flood-to-drought (FTD) and drought-to-flood (DTF) transition dynamics across Great Britain under future climate scenarios. This repository contains the full pipeline from HBV-simulated discharge through to the country-level statistics reported in the paper's results.

This repository depends on the separate [`hbv-model`](https://github.com/sopanpatil/hbv-model) repository for the HBV model itself and its calibrated parameters — see Stage 1 below.

## Data bundled in this repository

- **CAMELS-GB v2 catchment attribute tables** (`camels_gb_v2_{climatic,hydrologic,hydrogeology,topographic}_attributes.csv`): these four small attribute tables are bundled directly in this repository so the pipeline runs without a separate download. They are © UK Centre for Ecology & Hydrology and contributing organisations, released under the [Open Government Licence](https://eidc.ac.uk/licences/ogl/plain), and redistributed here under its terms. Per the licence, any reuse must include this attribution: *"Contains data supplied by UK Centre for Ecology & Hydrology, British Geological Survey, Environment Agency, Natural Resources Wales and Scottish Environmental Protection Agency."* See Coxon et al. (2026) for the full dataset description and citation. This applies only to these four attribute CSVs — the repository's own code is licensed separately under the MIT License in `LICENSE`.

## External data dependencies (not bundled in this repository)

- **CAMELS-GB v2 observed discharge and hydrometeorological time series**, and the **catchment boundaries shapefile** (`camels_gb_v2_catchment_boundaries.shp`, needed only by `figures/` — see `figures/README.md`): these are substantially larger than the attribute tables above and are not bundled here. Publicly available from the NERC Environmental Data Service; see Coxon et al. (2026).
- **CHESS-SCAPE**: climate projection forcing (precipitation, temperature, PET). Publicly available from the Centre for Environmental Data Analysis; see Robinson et al. (2023).
- **ONS Countries boundary** (`great_britain_boundary.geojson`): used for country-level spatial analysis (Scotland/England/Wales assignment), by both the main pipeline (Stage 4) and `figures/`.

## Pipeline stages

### Stage 1 — Hydrological simulation (Methods 2.1/2.3)
- `run_hbv_chess_scape.py` — runs the calibrated HBV model (from `hbv-model`) across CHESS-SCAPE forcing for a given RCP/ensemble member, producing discharge and internal states (SM, UZ, LZ, SP/snowpack).

### Stage 2 — Event and transition extraction (Methods 2.2)

**Design note — fixed baseline reference:** every event/threshold definition in this stage is fixed from the baseline period only, then applied identically to classify or standardise *both* the baseline and future periods. This is deliberate: it keeps the reference stationary, so that any change measured between periods reflects a genuine climate signal rather than a shifting yardstick. Concretely, Q5/Q80 discharge thresholds and the SSI-1 empirical CDF are each fit once, from baseline data only, and that same fixed threshold/CDF is then used to detect events and compute standardised values in both periods.

**FTD (flood-to-drought):**
- `compute_thresholds.py` — fits fixed Q5 (flood) and Q80 (drought) discharge thresholds per catchment from the baseline period only (one threshold set per catchment, not computed separately for the future period).
- `extract_transitions.py` — applies those baseline-fixed thresholds to detect FTD events in *both* the baseline and future periods, and attaches SM/UZ/LZ/SP states at key transition points.

**DTF (drought-to-flood):**
- `compute_ssi.py` — fits an empirical CDF (per calendar month) to baseline monthly flows only, then uses that same baseline-fitted CDF to standardise *both* periods, producing separate `ssi_baseline_*.parquet` and `ssi_future_*.parquet` outputs (unlike thresholds, SSI values themselves are computed for both periods — but always referenced against the fixed baseline distribution).
- `explore_drought_to_flood.py` — applies the same baseline-fixed thresholds as `compute_thresholds.py` to detect DTF events in both periods, and attaches SM/SP states (recovery-direction equivalent of `extract_transitions.py`).

**Shared utility:**
- `catchment_filter.py` — filters catchments by validation KGE ≥ 0.5 (621 of 671 retained), reading directly from `hbv-model`'s `calibrated_parameters.csv` (no separate KGE file is duplicated here). Used throughout the rest of the pipeline. **Note**: its default path (`../hbv-model/calibrated_parameters.csv`) assumes `hbv-model` is cloned as a sibling folder alongside this repository; if not, pass `kge_path` explicitly when calling `get_valid_catchments()`.

**Pairing-rule diagnostic:**
- `quantify_pairing_supersession.py` — standalone diagnostic quantifying how often the one-to-one pairing rule in `extract_transitions.py`/`explore_drought_to_flood.py` *supersedes* (drops) a precursor flood/drought event versus *consumes* it into a registered transition. Re-derives flood/drought events independently, with the identical detection logic, so it has no import dependency on either pipeline script and never modifies their outputs. Backs the pairing-rule statistics reported in Methods 2.2 and Discussion (75.1% of flood events and 59.5% of drought events superseded, pooled across all catchments/RCPs/ensembles/periods).

### Stage 3 — Ensemble aggregation (Methods 2.3)
- `aggregate_ensemble.py` — aggregates FTD transitions across the 4 ensemble members per RCP, computes baseline-vs-future delta metrics, and merges in whiplash frequency (from SSI).
- `aggregate_ensemble_d2f.py` — DTF equivalent.

### Stage 4 — Spatial analysis (Methods 2.4)
- `spatial_pattern_f2d.py` / `spatial_pattern_d2f.py` — Moran's I spatial autocorrelation and mapping for each delta metric, across all four RCPs.

### Stage 5 — Catchment typology (Methods 2.5)
- `catchment_typology.py` — k-means (primary) and Ward hierarchical (cross-check) clustering of FTD catchment response, at RCP8.5.
- `catchment_typology_d2f.py` — DTF equivalent.
- `catchment_typology_rcp26_check.py` — robustness check re-deriving the FTD typology at RCP2.6 instead of RCP8.5.
- `check_kmeans_ward_disagreement.py` / `check_kmeans_ward_disagreement_d2f.py` — diagnoses which catchments the two clustering methods disagree on, and whether disagreement is driven by extreme-value outliers or genuine borderline cases.

### Stage 6 — Catchment attribute regression (Methods 2.6)
- `catchment_attribute_regression.py` — Spearman correlation, OLS, and random forest importance, regressing FTD delta metrics against CAMELS-GB v2 attributes.
- `catchment_attribute_regression_d2f.py` — DTF equivalent.

### Stage 7 — Snow validation (Methods 2.7)
- `test_snowmelt_hypothesis.py` / `test_snowmelt_hypothesis_d2f.py` — tests whether static `frac_snow` (historical snow fraction) correlates significantly with delta metrics, and differs significantly between response clusters/countries.
- `test_frac_snow_incremental.py` / `test_frac_snow_incremental_d2f.py` — tests whether `frac_snow` adds independent explanatory power beyond the other 13 catchment attributes (redundancy check, VIF, nested F-test), restricted to the RCP8.5 analysis subsample (n=576 FTD / 611 DTF) rather than the full unfiltered 671-catchment attribute table. **Note**: one catchment (31023) is missing `slope_fdc` and is dropped by the `.dropna()` step in Tests 1-2 and by the RCP8.5 row of Test 3, so the VIF/redundancy outputs and the RCP8.5 nested F-test row report n=575 FTD / 610 DTF, one less than the 576/611 typology sample. **Run with `--all`** (not the single-RCP default) to regenerate the full four-RCP outputs that `figures/build_figure4_snow_relationships.py` reads — the committed CSVs must contain all four RCPs, not just RCP8.5, or Figure 4b will be built from incomplete data.
- `analyze_dynamic_snow.py` — tests whether *dynamically simulated* snowpack decline (from HBV's SP state, comparing baseline to future) correlates with transition-gap change, as an independent, mechanistic complement to the static `frac_snow` attribute above.
- `validate_snowpack_fracsnow.py` — standalone validation test backing the Results 3.4 / Discussion claim that simulated baseline snow storage corroborates the independent `frac_snow` attribute. Correlates ensemble-median baseline mean/peak simulated snowpack (HBV's SP state) against `frac_snow` across all 671 catchments. Requires the raw `chess_scape_output/*_hbv_sp.csv` time series (Stage 1 output, not bundled — see "Derived data" below); run on JASMIN, confirming n=671 and rho=0.706 (mean_sp), the figure reported in Results 3.4 as rho=0.71. Output archived at `spatial_output/snowpack_fracsnow_validation.csv`.

### Country-level statistics (Table 2)
- `country_summary_f2d.py` / `country_summary_d2f_allrcp.py` — per-country (England/Scotland/Wales) medians, Kruskal-Wallis tests, and pairwise Mann-Whitney U tests for each delta metric, across all four RCPs.

### Figures
- `figures/` — all manuscript figure-generation scripts (Figures 1-4, plus a supplementary DTF typology figure). Reads directly from the output directories above rather than hardcoding values. See `figures/README.md` for details and its own external data dependencies.

## Requirements

`pip install -r requirements.txt`. This covers the main pipeline (Stages 1-7); `figures/` has one additional dependency (`geopandas`, for Figure 2 only) already included in the same file — see `figures/README.md` for which script needs it. `hbv-model` (Stage 1) is not on PyPI and must be cloned separately as a sibling directory — see "External data dependencies" above.

## Running the pipeline

Each script can be run independently once its required inputs exist (see each script's own docstring for exact input/output paths). Most scripts default to processing all four RCPs; pass `--rcp <rcp>` (and `--ensemble <ens>` where relevant) to run a single combination.

Example, starting from scratch for one RCP/ensemble combination:

```bash
python run_hbv_chess_scape.py --rcp rcp85 --ensemble 01 --params-csv /path/to/hbv-model/calibrated_parameters.csv
python compute_thresholds.py --rcp rcp85 --ensemble 01
python compute_ssi.py --rcp rcp85 --ensemble 01
python extract_transitions.py --rcp rcp85 --ensemble 01 --period baseline
python extract_transitions.py --rcp rcp85 --ensemble 01 --period future
python explore_drought_to_flood.py --rcp rcp85 --ensemble 01 --max-gap 90
python quantify_pairing_supersession.py --rcp rcp85 --ensemble 01
# ... repeat the above across all RCPs and ensemble members, then:
python quantify_pairing_supersession.py --summarise-only
python aggregate_ensemble.py
python aggregate_ensemble_d2f.py
python catchment_typology.py
python catchment_typology_d2f.py
# etc.
```

## Derived data

The following small, final output directories are included in this repository, since they directly underpin every figure, table, and reported statistic in the manuscript:

- `aggregated_output/` / `d2f_aggregated_output/` — per-catchment, per-RCP delta metrics (Stage 3 output)
- `typology_output/` / `d2f_typology_output/` — cluster assignments, cluster profiles, and k-selection diagnostics (Stage 5 output)
- `regression_output/` / `d2f_regression_output/` — Spearman correlations, OLS regressions, and random forest importance, per RCP (Stage 6 output)
- `spatial_output/` / `d2f_spatial_output/` — Moran's I summaries, snow-relationship test results, and country-level statistics (Stages 4 and 7 output; excludes the PNG maps these stages also produce, which are regenerable from this data). Includes `catchment_n_by_rcp.csv` (per-RCP catchment counts underlying Methods 2.3 / Table S6 — the 576/611 figures quoted in the main text are the RCP8.5 values specifically; other RCPs range 570–576 for FTD and 610–613 for DTF), `frac_snow_vif.csv` / `frac_snow_vif_d2f.csv` (VIF for `frac_snow`, RCP8.5 analysis subsample), and `snowpack_fracsnow_validation.csv` (baseline simulated snowpack vs. `frac_snow`, all 671 catchments — backs the rho=0.71 figure in Results 3.4).
- `pairing_diagnostics_output/` — per-catchment, per-RCP/ensemble/period pairing-rule diagnostics (superseded/consumed/gap-excluded event counts) from `quantify_pairing_supersession.py`, plus the pooled `pairing_supersession_summary.csv` backing the Methods 2.2 / Discussion supersession statistics. Included in full despite being an intermediate-style output, since it is small (~1 MB total) and directly supports specific reported numbers.

The larger, intermediate outputs upstream of these (`thresholds/`, `ssi_output/`, `transitions_output/`, `drought_to_flood_exploratory/`, and the HBV discharge/state time series from Stage 1) are not included, since they are large and fully regenerable from the public CHESS-SCAPE forcing, CAMELS-GB v2 attributes, and this repository's code.

## Citation

If you use this code, please cite the archived release (see `CITATION.cff`).
