# Model scope and evaluation

The published source contains the current four-mountain deployment and earlier experiments. The deployed choices are recorded in `outputs/nwp-weather/forecast-portfolio/deployment_selection.json` after installing the runtime bundle. Public configuration and checksum metadata are in `artifacts/runtime-files.json`.

## Forecast products

- Alta 1h uses the selected compact MRMS/NowcastNet adaptation. Other radar comparisons use native NEXRAD DPR rate sequences, persistence/advection, and validation-selected HRRR combinations. Vail uses HRRR where quality-qualified radar verification is insufficient.
- Native HRRR snowfall is differenced from cumulative ASNOW. Only apparent decreases within each field's own GRIB quantization bounds are reconciled; raw values and adjustments are retained. Missing intervals and larger resets are rejected.
- Alta's trained 12h forecast applies to its supported reporting window. A current HRRR fallback is used when that trained window is unavailable.
- The 24h forecast combines HRRR and the corresponding neural accumulation estimate using frozen validation-selected weights.
- The 48h neural model uses weather and, for selected mountains, radar context. Its supervised targets are reporting-day accumulations. It is not independently validated hourly snowfall prediction.

## Evaluation discipline

Models and combinations were selected using validation data. Historical held-out periods have already been inspected during this project and are identified as such in reports. A future, untouched period is needed before claiming out-of-sample operational gains.

Dry/wet errors must be evaluated on the same observed windows for all candidates, with dry defined by observed zero and wet by observed positive accumulation. Compare sample counts and measurement definitions alongside RMSE. Liquid-equivalent short-horizon errors cannot be compared numerically with snow-depth errors without an additional, independently verified conversion.

The app's live Open-Meteo comparison is distinct from archived provider benchmarks. Current provider differences are forecasts, not known errors. Model intervals are descriptive or uncalibrated unless a report explicitly establishes coverage on independent observations.

## Failure behavior

All refreshes share a bounded batch. Each source retains its original issue and valid times when it fails. The app does not relabel an old forecast as newly issued or extend HRRR beyond its native endpoint. New radar caches are checked for station, issue, grid, units, distinct scans, and causal product times before inference.

Daily radar listings remain refreshable until a completed archive day has actually been fetched after an ingestion allowance. Midnight and year rollover tests protect against freezing incomplete live listings.

## Reproducing research

The source includes acquisition and training scripts for Copper/Dillon experiments as well as the four deployed mountains. Runtime installation does not reproduce training. Inspect the relevant script's CLI and data paths before downloading full archives; acquisition can require tens of GB and considerable compute. Preserve dataset clocks, quality flags, frozen splits, model-selection records, and source attribution.
