#!/usr/bin/env python3
"""Download/parse independent NOAA daily snowfall, with explicit missingness.

    python src/fetch_ghcn.py --offline --output-dir data
    python src/fetch_ghcn.py --download --output-dir data
    python src/fetch_ghcn.py --self-test

Only complete October and November--April seasons enter primary training.
The raw station files contain all elements; exported daily CSVs contain SNOW.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
STATION_IDS = ("USC00051660", "USC00052281", "USC00058575")
REFERENCE_ID = "USS0006K24S"


def parse_snow(text: str, expected_station: str) -> dict[date, dict]:
    """Retain raw values/flags; SNOW units are mm, not tenths of mm."""
    records = {}
    for line in text.splitlines():
        if not line:
            continue
        if len(line) < 269:
            raise ValueError("Truncated GHCN .dly record")
        if line[:11] != expected_station:
            raise ValueError("Unexpected station in .dly file")
        if line[17:21] != "SNOW":
            continue
        year, month = int(line[11:15]), int(line[15:17])
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            dt = date(year, month, day)
            if dt in records:
                raise ValueError(f"Duplicate date: {dt}")
            block = line[21 + 8 * (day - 1):29 + 8 * (day - 1)]
            value = int(block[:5])
            mflag, qflag, sflag = (v.strip() for v in block[5:8])
            usable = value >= 0 and not qflag and mflag != "P"
            records[dt] = {
                "station_id": expected_station,
                "date": dt.isoformat(),
                "snowfall_mm": value if value != -9999 else "",
                "snowfall_inches": round(value / 25.4, 6) if value != -9999 else "",
                "measurement_flag": mflag,
                "quality_flag": qflag,
                "source_flag": sflag,
                "usable": int(usable),
            }
    if not records:
        raise ValueError(f"No SNOW records found for {expected_station}")
    return records


def date_range(start: date, end: date):
    while start <= end:
        yield start
        start += timedelta(days=1)


def aggregate(records: dict, station: dict, start_year: int, end_year: int) -> list[dict]:
    rows = []
    for year in range(start_year, end_year + 1):
        oct_dates = list(date_range(date(year, 10, 1), date(year, 10, 31)))
        winter_dates = list(date_range(date(year, 11, 1), date(year + 1, 4, 30)))

        def summarize(dates):
            valid = [records[d] for d in dates if d in records and records[d]["usable"]]
            return len(valid), sum(r["snowfall_mm"] for r in valid) / 25.4

        oct_n, oct_sum = summarize(oct_dates)
        winter_n, winter_sum = summarize(winter_dates)
        rows.append({
            "station_id": station["id"], "station_name": station["name"],
            "season_start": year, "season": f"{year}/{year + 1}",
            "october_snowfall_inches": round(oct_sum, 6) if oct_n == 31 else "",
            "nov_apr_snowfall_inches": round(winter_sum, 6) if winter_n == len(winter_dates) else "",
            "october_observed_sum_inches": round(oct_sum, 6),
            "nov_apr_observed_sum_inches": round(winter_sum, 6),
            "october_usable_days": oct_n, "october_expected_days": 31,
            "nov_apr_usable_days": winter_n, "nov_apr_expected_days": len(winter_dates),
            "complete_oct_nov_apr": int(oct_n == 31 and winter_n == len(winter_dates)),
        })
    return rows


def parse_metadata(text):
    stations = {}
    for line in text.splitlines():
        if line[:11] in (*STATION_IDS, REFERENCE_ID):
            stations[line[:11]] = {
                "id": line[:11], "latitude": float(line[12:20]),
                "longitude": float(line[21:30]), "elevation_m": float(line[31:37]),
                "name": line[41:71].strip(), "original_metadata_line": line,
            }
    if set(stations) != set((*STATION_IDS, REFERENCE_ID)):
        raise ValueError("Required stations are missing from metadata")
    return stations


def distance_km(a, b):
    la, lb, loa, lob = map(math.radians, [a["latitude"], b["latitude"], a["longitude"], b["longitude"]])
    h = math.sin((lb - la) / 2) ** 2 + math.cos(la) * math.cos(lb) * math.sin((lob - loa) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fetch(url, path):
    # Do not replace a valid cached file with an interrupted or empty response.
    data = urlopen(url, timeout=60).read()
    if not data:
        raise ValueError(f"Empty response from {url}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)


def self_test():
    station = "USC00051660"
    cells = [f"{0:5d}  0"] * 31
    cells[0] = f"{254:5d}  0"  # Exactly ten inches, not one inch.
    cells[1] = f"{-9999:5d}   "
    cells[2] = f"{0:5d}P 0"
    cells[3] = f"{51:5d} X0"
    cells[4] = f"{0:5d}T 0"
    records = parse_snow(station + "202002SNOW" + "".join(cells), station)
    assert len(records) == 29
    assert records[date(2020, 2, 1)]["snowfall_inches"] == 10
    assert not records[date(2020, 2, 2)]["usable"]
    assert not records[date(2020, 2, 3)]["usable"]
    assert not records[date(2020, 2, 4)]["usable"]
    assert records[date(2020, 2, 5)]["usable"]  # Trace remains a recorded zero.
    assert aggregate(records, {"id": station, "name": "TEST"}, 2019, 2019)[0]["nov_apr_expected_days"] == 182
    row = aggregate(records, {"id": station, "name": "TEST"}, 2019, 2019)[0]
    assert row["october_snowfall_inches"] == "" and row["nov_apr_snowfall_inches"] == ""
    assert row["october_usable_days"] == 0  # An absent month is not zero snowfall.
    assert not row["complete_oct_nov_apr"]
    print("GHCN checks passed: units, missingness, quality flags, trace, leap year, absent months.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Parse existing cached files (default).")
    mode.add_argument("--download", action="store_true", help="Refresh three public NOAA files and metadata before parsing.")
    parser.add_argument("--first-season", type=int, default=1950)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument("--retrieved-date", default="2026-11-03", help="Date of the cached/downloaded snapshot; update when refreshing.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.first_season > args.last_season:
        parser.error("first-season must not exceed last-season")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if args.download:
        fetch(BASE_URL + "ghcnd-stations.txt", root / "ghcnd-stations.txt")
        fetch(BASE_URL + "readme.txt", root / "ghcnd-readme.txt")
        for sid in STATION_IDS:
            fetch(BASE_URL + f"all/{sid}.dly", root / f"ghcn_{sid}.dly")
    stations = parse_metadata((root / "ghcnd-stations.txt").read_text())
    meta = []
    for sid in STATION_IDS:
        source = root / f"ghcn_{sid}.dly"
        records = parse_snow(source.read_text(), sid)
        daily_rows = [records[d] for d in sorted(records)]
        seasonal = aggregate(records, stations[sid], args.first_season, args.last_season)
        write_csv(root / f"ghcn_{sid}_snow_daily.csv", daily_rows)
        write_csv(root / f"ghcn_{sid}_snow_seasonal.csv", seasonal)
        good = [r for r in seasonal if r["complete_oct_nov_apr"]]
        presumed_zero_sensitivity = 0
        for y in range(args.first_season, args.last_season + 1):
            ds = list(date_range(date(y, 10, 1), date(y + 1, 4, 30)))
            if all(d in records and records[d]["snowfall_mm"] != "" and
                   records[d]["snowfall_mm"] >= 0 and not records[d]["quality_flag"] for d in ds):
                presumed_zero_sensitivity += 1
        meta.append({
            **stations[sid],
            "distance_from_copper_snotel_km": round(distance_km(stations[REFERENCE_ID], stations[sid]), 2),
            "source_url": BASE_URL + f"all/{sid}.dly", "retrieved_date": args.retrieved_date,
            "raw_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "first_snow_record_date": min(records).isoformat(),
            "last_snow_record_date": max(records).isoformat(),
            "usable_snow_days": sum(r["usable"] for r in daily_rows),
            "complete_seasons_1950_2025": len(good) if (args.first_season, args.last_season) == (1950, 2025) else None,
            "first_season_start": args.first_season, "last_season_start": args.last_season,
            "complete_seasons": len(good), "complete_season_starts": [r["season_start"] for r in good],
            "complete_seasons_including_NOAA_presumed_zero": presumed_zero_sensitivity,
        })
        print(f"{sid} {stations[sid]['name']}: {len(good)} complete seasons")
    metadata = {
        "stations": meta, "distance_reference": stations[REFERENCE_ID],
        "selection": "Dillon 1 E is the primary independent dataset (41 strict complete seasons in the supplied snapshot). Climax is a higher-elevation secondary station (22). Vail has only five. Selection is geographic/completeness based, not based on hypothesis correlations.",
        "format_source_url": BASE_URL + "readme.txt",
        "units": "SNOW is millimetres; inches = mm / 25.4. PRCP uses a different scale.",
        "quality_policy": "Accept nonnegative SNOW with blank QFLAG, not -9999, MFLAG not P (missing presumed zero). Trace T retained as recorded zero. All original values and flags are preserved. Absent months remain missing.",
        "season_policy": "October requires 31/31 usable days. November-April requires every calendar day including leap days. Incomplete totals are blank. Observed partial sums are diagnostic columns, not training totals.",
        "policy_sensitivity": "Counts admitting NOAA MFLAG=P presumed zeros are recorded separately; primary CSVs exclude them.",
        "caveats": "Independent NOAA stations, not the resort Cache Site. Shared regional weather means nearby stations in a year are not independent climate realizations. Do not pool them as independent seasons. Station moves and observing practices need further homogeneity assessment.",
    }
    (root / "ghcn_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
