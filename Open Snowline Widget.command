#!/bin/zsh
set -eu
cd "${0:A:h}"
widget_binary="desktop/Snowline Widget.app/Contents/MacOS/SnowlineWidget"
rebuild=0
if [[ ! -x "$widget_binary" ]]; then
    rebuild=1
else
    for source in desktop/*.swift desktop/build.sh; do
        [[ "$source" -nt "$widget_binary" ]] && rebuild=1
    done
fi
if (( rebuild )); then
    /bin/zsh desktop/build.sh
fi
exec open "desktop/Snowline Widget.app"
