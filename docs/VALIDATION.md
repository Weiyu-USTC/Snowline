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
