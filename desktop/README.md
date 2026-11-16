# Snowline Mac widget

The floating widget is a native SwiftUI/AppKit client for the local Snowline
forecast service. It supports Alta, Vail, Stowe, and Jay Peak, horizons from
1 to 48 hours, US/metric units, and a button opening the full dashboard.

## Setup

1. Complete the Python environment and runtime-asset setup in the
   [project README](../README.md). Keep the main environment at `.venv` in the
   project directory.
2. Use macOS 13 or later and Apple's Xcode Command Line Tools. If needed,
   install the tools with `xcode-select --install`.
3. From the project directory, run `./desktop/build.sh`.
4. Open `Open Snowline Widget.command`, or open
   `desktop/Snowline Widget.app` after building.

The build targets the current Mac's architecture (`arm64` or `x86_64`) and
signs the application locally. A generated app is a local build artifact;
it is not a notarized installer. This repository contains source, not a
prebuilt application.

To compile for a different supported Mac architecture, set `SNOWLINE_ARCH`:

```sh
SNOWLINE_ARCH=x86_64 ./desktop/build.sh
```

The widget prefers the project's `.venv/bin/python`. If that is absent, it
tries the interpreter recorded at build time, then `python3` on the launch
environment's path. To choose the build-time interpreter explicitly:

```sh
SNOWLINE_PYTHON=/absolute/path/to/python3 ./desktop/build.sh
```

Keep the app in the project's `desktop` directory. Recreate the Python
environment after moving the project to a different computer; virtual
environments are not portable. If a folder-access dialog appears, macOS
needs permission to read this project. Running the project outside
Documents avoids requiring access to that protected folder.

## Use

Click the floating mountain icon to expand or collapse the panel. Drag the
icon to move it. Choose a mountain and horizon, then select **Refresh all**
to update all four mountains. The app downloads new public inputs and runs
the saved models; it does not retrain them. **Our model** denotes the
selected model or its documented fallback for the requested horizon.

**Full forecast** opens the dashboard in the default browser. Both clients
use `http://127.0.0.1:8765`; the service is accessible only from this Mac.
Quitting the widget leaves the service available to open dashboard pages.

**Auto-update** checks at each local `HH:00:00` while the widget is open and
Wi-Fi is available. Sleeping, offline, late, or busy hours are skipped.
It does not wake the Mac, install a login item, or catch up missed hours.

Short-range radar comparisons measure liquid precipitation. Longer
comparisons use snowfall accumulation over the displayed window; the
dashboard describes station targets, fallback methods, and validation
limits. A failed source retains its original timestamps rather than
appearing newly issued. See the project README for model and data details.

## Troubleshooting

If the service cannot start, confirm the Python and runtime assets were
installed, then read `outputs/dashboard/widget-server.log`. The separate
`Open Snow Dashboard.command` launcher also starts the service and shows
errors in Terminal. Rebuild the widget after changing its Swift sources.

The widget uses port 8765. Stop any other Snowline checkout's service before
running this copy, so the widget connects to the intended project's data.
Building the widget does not launch it or alter an already-running app.

## Auto-update policy checks

These checks exercise timing and Wi-Fi policy without changing the Mac's
clock, connecting to weather services, or changing saved preferences:

```sh
test_dir="$(mktemp -d)"
xcrun swiftc -parse-as-library -module-cache-path "$test_dir/modules" \
  desktop/HourlyRefreshPolicy.swift desktop/tests/HourlyRefreshPolicyTests.swift \
  -o "$test_dir/hourly-policy-tests"
"$test_dir/hourly-policy-tests"
rm -rf "$test_dir"
```
