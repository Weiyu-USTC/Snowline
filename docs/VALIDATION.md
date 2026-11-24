# Release validation — 15 November 2026

Validated locally on macOS arm64 with Python 3.14 and the direct dependency versions recorded in `requirements.txt`.

- Source suite: **128 tests run, 126 passed, 2 optional archived-fixture tests skipped**. External network connections are blocked in this suite; local HTTP checks are allowed.
- Mac widget: native build and ad-hoc signature verification passed. The built app was not launched over the existing installation.
- Hourly scheduling policy: **28 checks passed**.
- Runtime installer: all **126 bundled files** matched their recorded SHA-256 hashes. Tests also cover altered, missing, extra, and unsafe archive entries and preservation of modified local files.
- Isolated checkout: installed only the companion bundle, then performed a first refresh with empty weather/radar caches. **All nine update stages succeeded**, including report and chart generation. The refresh completed on 15 November at 10:08 PM EST (16 November 03:08 UTC).
- Forecast responses: all four mountains supplied six usable horizon comparisons; Alta 12h correctly used its documented HRRR fallback outside the trained reporting window. JSON serialization passed without non-finite values.
- Packaging: source files contain no author-specific home-directory paths or recognized embedded credential patterns. Git ignore rules exclude runtime data, environments, app binaries, and credential files. This automated scan is not a security audit.

The isolated checkout reused the already-installed dependency packages through a separate test environment; it did not download a new Python distribution or prove fresh installation on every platform. GitHub-hosted Linux and macOS workflows have not run yet because this package has not been pushed to a repository. Test those workflows after upload, especially before claiming Windows or Intel Mac support.

No new model training or validation-based selection was performed as part of this packaging work. Live source availability can change after the successful refresh above.


## Mammoth hourly forecasts

Snowline fitted 14 candidates on the 2020/21 and 2021/22 winters, selected a model on 2022/23 and scored the 2023/24 and 2024/25 winters separately from selection. The Jeff Dozier optical station measures frozen precipitation intensity on an instrument-specific water-equivalent scale. The model combines issued HRRR fields and quality-screened MRMS radar. Native HRRR ends 48 hours after initialization. The model continues to the full rolling window using features fitted with this missing tail represented during training.

The model evaluation includes separate wet and dry RMSE, identical-window HRRR and ECMWF comparisons, actual observed hourly storm curves and a sensitivity check for the minute frequently missing from otherwise complete hours in 2024/25. At Main Lodge, the independent fresh-depth calibration is scored by complete Pacific calendar month. Supporting assets and fitted weights ship with the source. [Mammoth model evaluation](../artifacts/mammoth-hourly/report.html)

## Seasonal view — 18 November 2026

- Updated source suite: **159 tests run, 157 passed, 2 optional archived-fixture tests skipped**, using the existing complete Python 3.14 environment. No retraining or model selection changes.
- Frontend: **31 JavaScriptCore checks passed**, including unit conversion, exact observation/forecast endpoint alignment, missing-data behavior, leap years, off-season zoom, stale request handling, and imported-snapshot separation.
- Local page checks: all four seasonal charts loaded in an isolated in-app browser; metric conversion and mountain selection worked. Current source coverage and unavailable forecast joins remained visible.
- Seasonal acquisition: Alta current totals plus five matching official historical seasons were cached; NOAA station and Vail proxy inputs refreshed or reused valid recent caches. Public GET endpoints only read local data; acquisition runs during Refresh all.
- Current records do not establish a valid measured baseline and matching forecast cutoff at any of the four mountains. Forecast-line and uncertainty-band rendering is tested with controlled inputs; no unsupported live continuation is claimed.
