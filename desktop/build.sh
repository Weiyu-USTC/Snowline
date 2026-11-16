#!/bin/zsh
set -eu
cd "${0:A:h}"
project_root="${PWD:h}"
app_path="$PWD/Snowline Widget.app"
build_arch="${SNOWLINE_ARCH:-$(uname -m)}"
case "$build_arch" in
  arm64|x86_64) ;;
  *) print -u2 "Unsupported Mac architecture: $build_arch"; exit 1 ;;
esac
if [[ -n "${SNOWLINE_PYTHON:-}" ]]; then
    python_path="$SNOWLINE_PYTHON"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
    python_path="$project_root/.venv/bin/python"
else
    python_path="$(command -v python3)"
fi
if [[ ! -x "$python_path" ]]; then
    print -u2 "Python is unavailable. Complete the Python setup in the project README."
    exit 1
fi
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/snowline-build.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT
mkdir -p "$app_path/Contents/MacOS" "$app_path/Contents/Resources" "$build_dir/modules"
xcrun swiftc -swift-version 5 -parse-as-library -O -target "$build_arch-apple-macosx13.0" \
  -module-cache-path "$build_dir/modules" \
  -framework AppKit -framework SwiftUI -framework Network SnowlineWidget.swift HourlyAutoUpdater.swift HourlyRefreshPolicy.swift \
  -o "$app_path/Contents/MacOS/SnowlineWidget"
"$python_path" - "$app_path" "$project_root" "$python_path" <<'PY'
import plistlib,sys
from pathlib import Path
app,root,python=sys.argv[1:]
info={
 'CFBundleExecutable':'SnowlineWidget',
 'CFBundleIdentifier':'local.snowline.widget',
 'CFBundleName':'Snowline Widget',
 'CFBundleDisplayName':'Snowline Widget',
 'CFBundlePackageType':'APPL',
 'CFBundleShortVersionString':'1.0',
 'CFBundleVersion':'1',
 'LSMinimumSystemVersion':'13.0',
 'LSUIElement':True,
 'NSHighResolutionCapable':True,
 'NSAppTransportSecurity':{'NSAllowsLocalNetworking':True},
 'SnowlineProjectRoot':root,
 'SnowlinePython':python,
}
with open(Path(app)/'Contents/Info.plist','wb') as f: plistlib.dump(info,f)
PY
codesign --force --sign - "$app_path"
print "Built $app_path ($build_arch)"
