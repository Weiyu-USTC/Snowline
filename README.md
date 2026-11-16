# Snowline

Mountain weather forecasts, trained snowfall models, and a floating macOS widget for **Alta, Vail, Stowe, and Jay Peak**.

Snowline compares the selected forecast, labeled **Our model**, with Open-Meteo over the same location and accumulation window. Choose 1, 3, 6, 12, 24, or 48 hours; switch between metric and US units; refresh all four mountains together. The Mac widget can also refresh at the top of each hour while it is open and Wi-Fi is available.

This is a research application. The local model adaptations are not the full published DGMR or NowcastNet systems, and their results do not establish operational forecast superiority.

## What is included

- Python acquisition, inference, evaluation, and training source.
- Local HTML dashboard and native SwiftUI macOS widget.
- Frozen model-selection manifest and a checksum-verified runtime installer.
- Tests for forecast windows, missing data, model comparisons, radar caches, midnight rollover, and hourly scheduling.
- GitHub Actions configuration for source tests and a Mac build.

The source repository deliberately excludes downloaded archives, current forecasts, generated app binaries, and Python environments. Saved models and the reference tables needed for inference are distributed in a **separate companion runtime ZIP**. A first refresh downloads current inputs; it does not retrain models.

## Run locally

The released models were tested with **Python 3.14** and the direct dependency versions in `requirements.txt`. The web dashboard runs as a local Python service; the floating widget requires macOS 13+ and Apple's command-line developer tools. macOS is the tested desktop platform. [GitHub Actions passed the Linux Python checks and the macOS widget build](https://github.com/Weiyu-USTC/Snowline/actions/runs/43799532516).

Download the **[v0.1.0 model bundle (46.6 MB)](https://github.com/Weiyu-USTC/Snowline/releases/download/v0.1.0/snowline-runtime-2026-11-15.zip)** from the **[Snowline v0.1.0 release](https://github.com/Weiyu-USTC/Snowline/releases/tag/v0.1.0)**. The release also provides [SHA-256 checksums](https://github.com/Weiyu-USTC/Snowline/releases/download/v0.1.0/SHA256SUMS.txt).

These links require the `v0.1.0` release and its assets to be published. Pushing the source through GitHub Desktop does not upload the model bundle; maintainers can follow the [manual release steps](docs/PUBLISHING.md).

With Python 3.14 installed, start from a new folder in Terminal:

```sh
git clone --branch v0.1.0 https://github.com/Weiyu-USTC/Snowline.git
cd Snowline
bash scripts/setup.sh
curl --fail --location --retry 3 \
  --output artifacts/snowline-runtime-2026-11-15.zip \
  https://github.com/Weiyu-USTC/Snowline/releases/download/v0.1.0/snowline-runtime-2026-11-15.zip
.venv/bin/python scripts/install_runtime.py artifacts/snowline-runtime-2026-11-15.zip
.venv/bin/python scripts/doctor.py
.venv/bin/python src/serve_snow_dashboard.py --open
```

If you already have this release's source, run the commands beginning with `bash scripts/setup.sh` from its root folder. A tagged clone starts in detached-HEAD mode, which is fine for running the release; create a branch before developing changes. Obtain the source and runtime bundle from the same release. The installer verifies all 126 files against `artifacts/runtime-files.json` before installing them. The model ZIP belongs under the Git-ignored `artifacts/` directory; downloaded inputs and installed models stay in Git-ignored `data/` and `outputs/` directories.

Open **http://127.0.0.1:8765/** and click **Refresh all**. The first update can take several minutes and download substantial weather/radar data. Missing or late feeds are reported separately; a missing forecast is never silently presented as zero. If a service is temporarily unavailable, a later refresh can retry it.

### Floating Mac widget

After the setup above:

```sh
zsh desktop/build.sh
open "desktop/Snowline Widget.app"
```

Or double-click **Open Snowline Widget.command**. The widget starts the local service, lets you select mountains and horizons, and opens the full dashboard using **Full forecast**. Rebuild the app after moving the checkout so its fallback paths are updated. See [desktop setup](desktop/README.md).

**Auto-update** uses the Mac's clock at HH:00:00 and requires usable Wi-Fi. Sleeping, closed, missed, or already-busy hours are skipped. It does not install a login item, prevent sleep, or run optimizers.

## What the forecasts mean

| Window | Selected approach |
| --- | --- |
| 1–3 hours | Radar nowcasting, radar/HRRR combinations, or HRRR where radar verification is insufficient |
| 6–12 hours | Native HRRR guidance; Alta additionally has a trained 12-hour combination for its supported reporting window |
| 24 hours | Validation-selected HRRR/neural combinations |
| 48 hours | Weather/radar temporal neural models; Stowe uses the selected weather-only model |

**The 1-hour and 3-hour comparisons measure liquid-equivalent precipitation, not fresh-snow depth.** Their units are `mm water` or `in water`. Longer comparisons use snowfall depth in cm/inches. The hourly weather chart is Open-Meteo guidance; it is separate from the trained accumulation forecasts.

Alta uses Collins observations. Vail's target is a snow-sensor proxy with an unverified reset convention. The current Stowe and Jay Peak neural targets are lower-elevation nearby town observers, not summit snowfall. Mountain temperature corrections do not turn those observations into resort snowfall measurements.

The 24/48-hour neural models were trained on reporting-day totals. Rolling live windows and any hourly allocation remain experimental. Their uncertainty ranges are uncalibrated. Saved model selections use validation errors; inspected historical test results are not a wholly unseen future evaluation. Read [model scope and evaluation](docs/MODELS.md) and [data sources and terms](docs/DATA_SOURCES.md).

## Tests

After installing dependencies:

```sh
.venv/bin/python scripts/test_source.py
```

This suite uses temporary fixtures, blocks external network connections, and requires no runtime bundle or training archive. Two optional tests skip when their archived fixtures are absent. Some other historical research tests require separately downloaded datasets; the source-only suite is the portable release check.

On a Mac, test the hourly scheduler separately:

```sh
swiftc -parse-as-library desktop/HourlyRefreshPolicy.swift desktop/tests/HourlyRefreshPolicyTests.swift -o /tmp/snowline-policy-tests
/tmp/snowline-policy-tests
```

## Repository layout

```text
dashboard/     Browser UI
desktop/       SwiftUI widget and scheduler
src/           Acquisition, inference, training and evaluation modules
tests/         Regression tests
scripts/       Setup, verified runtime installation, checks
artifacts/     Runtime file manifest (weights distributed separately)
docs/          Data attribution, model scope, publishing instructions
data/          Created locally; ignored by Git
outputs/       Installed artifacts and generated results; ignored by Git
```

Refresh uses frozen weights. Training scripts in `src/` are retained for research, but reproducing historical experiments requires their acquisition/preparation pipelines and substantial disk space. The companion bundle is sufficient for live inference; it is not the complete training dataset.

## Publishing this project

See [GitHub publishing instructions](docs/PUBLISHING.md). Upload the source tree to your repository and attach the runtime ZIP to a GitHub Release. Do not upload the original working folder or force-add `data/`, `outputs/`, `.venv*`, or compiled `.app` files.

No new software license has been assigned by this export. Retain your repository's existing applicable license, or choose one before describing the source as open source. Third-party data and adapted material retain their own terms; see [attribution notices](docs/DATA_SOURCES.md) and `licenses/`. In particular, the Alta research data includes a noncommercial restriction. The runtime bundle is not a blanket grant of commercial redistribution rights.
