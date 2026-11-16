# Data sources and model artifacts

Snowline combines public weather forecasts, radar products and station observations. Its source code, source datasets and third-party model references are distinct materials; a source-code license does not replace the terms attached to those other materials.

## What the optional runtime bundle contains

`artifacts/runtime-files.json` is the exact allowlist for the separately distributed `snowline-runtime-2026-11-15.zip`. It records each relative installation path, byte count and SHA-256 digest. The bundle includes frozen fitted models, preprocessing and selection files, the small reference tables needed for inference, and dated scientific reports. It excludes raw radar/HRRR archives, training arrays, Python environments and active forecast caches.

The runtime assets are approximately 154 MiB unpacked and 45 MiB compressed. The largest individual file is about 40 MiB. Install the verified bundle into the repository root with the supplied installer; do not commit the extracted `data/` and `outputs/` trees to Git.

The bundled reports are historical experiment records. Their displayed forecasts, observation dates and figures remain the original dated snapshots. The app's **Refresh all** obtains new inputs and writes separate current forecasts; it does not retrain the models or change historical test results. The full historical training archive is not included, so the bundle supports live inference and reading the preserved reports, rather than reproducing every training experiment from raw data.

Serialized `.pkl`, `.joblib` and model checkpoint files must come from this release or another source you trust. Do not replace them with arbitrary uploaded model files.

## Weather and radar

| Source | Use in Snowline | Attribution and scope |
| --- | --- | --- |
| [NOAA HRRR](https://rapidrefresh.noaa.gov/hrrr/) and its [public archive](https://registry.opendata.aws/noaa-hrrr-pds/) | Issued temperature, humidity, wind, liquid precipitation, snow water and native snowfall forecasts | NOAA/NCEP. Point weather fields also use the University of Utah HRRR Zarr mirror. Initialization, native accumulation intervals and forecast availability remain explicit. |
| [NOAA NEXRAD](https://www.ncei.noaa.gov/products/radar/next-generation-weather-radar), distributed through the Unidata Level III archive | Three causal DPR radar frames for 1–3-hour liquid precipitation forecasts | NOAA/NWS radar observations. DPR product 176 is a liquid precipitation rate, not a measurement of fresh snow depth. |
| [NOAA MRMS](https://www.nssl.noaa.gov/projects/mrms/) | Radar sequence inputs, quality screening and short-horizon reference targets | NOAA/NSSL. MRMS and NEXRAD share observational ancestry; agreement with MRMS is not independent snowfall validation. |
| [Open-Meteo](https://open-meteo.com/en/docs) and its [previous-runs API](https://open-meteo.com/en/docs/previous-runs-api) | External comparison forecast and forecast-conditioned neural predictors | Retain Open-Meteo and underlying-provider attribution. The saved integration identifies the data as CC BY 4.0; hosted API usage conditions and access tiers are separate from data attribution. |
| [Iowa Environmental Mesonet MOS archive](https://mesonet.agron.iastate.edu/mos/) | Issued NOAA GFS MOS forecasts near the four stations | NOAA/NWS forecast guidance, archived by Iowa State University. Native 3/6-hour forecast fields are not hourly observed snowfall. |
| [NOAA Climate Prediction Center](https://www.cpc.ncep.noaa.gov/) | Lagged RONI climate-index reference | The reference file contains revised historical values, not archived issue-date releases. Its use is a pseudo-real-time convention and does not prove an ENSO causal effect. |

## Snowfall and mountain observations

**Alta–Collins.** Michael L. Wasserstein and Jim Steenburgh's University of Utah Hive datasets supply manual 12-hour new snow and liquid-equivalent measurements:

- [Alta–Collins observations 2000–2023](https://doi.org/10.7278/S50d-nsy5-8bje), record `280qy-4ee96`.
- [Alta–Collins observations 2023–2024](https://doi.org/10.7278/S5d-dx7x-d8ay), record `6hdjr-stw17`.

The preserved source metadata specifies **Creative Commons Attribution–NonCommercial 3.0 Unported**. The runtime daily-history table includes derived Alta rows; the original authors' readme notices travel with the runtime bundle. These data are not relicensed as unrestricted commercial data by this repository. The 2023/24 patrol observations also have different quality-control status from the earlier cleaned archive.

**Vail.** Historical CAVMM station records come from the [Colorado Avalanche Information Center station display](https://stations.avalanche.state.co.us/precip.php), with Vail Resort identified as the station provider. Snowline's target is the reported `Sno24` sensor endpoint: its reset protocol and equivalence to manual daily fresh snowfall are not fully established. The source audit does not establish a blanket redistribution license for the underlying station archive. It is not included wholesale; the small derived reference and evaluation tables retain this source qualification. Preserve the original attribution and verify source permissions before broader data redistribution or commercial use.

**Stowe and Jay Peak.** The 24/48-hour neural models use NOAA GHCN-Daily / CoCoRaHS observers [US1VTLM0001, Stowe 0.2 SW](https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/US1VTLM0001.csv.gz) and [US1VTOL0001, Westfield 0.7 WNW](https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/US1VTOL0001.csv.gz). These are lower-elevation town sites, not resort summits. Earlier daily-model reports use different NOAA station targets; they are documented separately rather than silently substituted. Observation flags, missing values and reporting clocks are retained.

**Temperature observations.** Vail temperature refinement uses [NRCS SNOTEL station 842](https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=842&state=CO). Stowe experiments use [Forest Ecosystem Monitoring Cooperative Mount Mansfield weather observations](https://www.uvm.edu/femc/data/archive/project/mt-mansfield-west-slope-mid-elevation/dataset/mansfield-west-met-site/metadata). Alta recent weather comes from MesoWest's CLN display. Snow depth or SWE changes are not relabeled as fresh snowfall. Source-specific notices and original links remain in the archived report metadata; no blanket license is asserted for third-party station data.

## Model implementation provenance

The radar architectures are compact local research adaptations, not the published authors' pretrained systems or verified reproductions of their reported performance.

- **ConvLSTM:** based on the mechanisms in [Shi et al. (2015)](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf).
- **DGMR adaptation:** based on [Ravuri et al. (2021)](https://doi.org/10.1038/s41586-021-03854-z) and DeepMind's [published pseudocode](https://storage.googleapis.com/dm-nowcasting-example-data/pseudocode.zip), copyright DeepMind, **CC BY 4.0**. Network widths and implementation choices differ. The original notice is in `licenses/DGMR-pseudocode.txt`, with detailed adaptation notes in `docs/RADAR_IMPLEMENTATION_NOTES.txt`; both also travel with the runtime assets.
- **NowcastNet adaptation:** an independent compact implementation of the mechanisms in [Zhang et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10356617/); no original author checkpoint is distributed.
- **48-hour and Alta 12-hour models:** locally trained temporal convolutional networks and selected ensembles/portfolios. Forecast accumulation targets do not supply independent hourly fresh-snow supervision.

The legacy research source also contains optional TabPFN experiments. TabPFN pretrained weights and the local model cache are excluded from this bundle; their separate Prior Labs license remains in `licenses/TabPFN-v2.txt` for anyone enabling those older experiments.

## Evaluation limits

Training, validation and test winters are separated in the saved experiments. Per-mountain model and blend selection use validation results. The 2023/24 season had already been inspected in earlier project work, so it is held out from the current fitting process rather than a wholly untouched project test set. Most live rolling windows differ from the fixed station reporting intervals used for training. Reported intervals and model rankings should be interpreted with those limits and the target-specific definitions above.
