#!/usr/bin/env python3
"""Fetch versioned NOAA ENSO inputs using only the Python standard library.

Live refresh (overwrites the selected output directory's ENSO files)::

    python src/fetch_enso.py --output-dir data

Offline reproduction from the bundled raw snapshots::

    python src/fetch_enso.py --offline-dir data --output-dir /tmp/enso-check

The CSVs preserve NOAA's published one-decimal table values, including their
rounding, and leave unavailable periods blank. The full current record is
retained intentionally: the modeling pipeline must select the season and
information cutoff. A later DJF observation must never become a November 1
predictor. Historical revisions also prevent these snapshots from being a
vintage-perfect simulation of what forecasters knew at each historical date.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


SEASONS = ("DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ")
COLUMNS = ("year",) + SEASONS
SOURCES = {
    "oni_noaa_v6_raw.html": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/",
    "roni_noaa_v6_raw.html": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/",
    "oni_noaa_raw.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
}


class _TableRows(HTMLParser):
    """Extract cells without depending on a browser or optional HTML packages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] = []
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row:
            self.rows.append(self.row)
            self.row = []


def parse_index_html(html: str) -> list[dict[str, str]]:
    """Parse a NOAA v6 table; reject missing headers, conflicts, and bad values.

    Rows can be repeated in NOAA's presentation markup, but conflicting
    duplicates are an error. Missing cells are allowed only at the end of the
    latest year. Returned values remain strings to preserve source precision.
    """
    if not re.search(r"ERSST\.?v6", html, re.IGNORECASE):
        raise ValueError("Expected an ERSSTv6 source; inspect an upstream version change.")
    parser = _TableRows()
    parser.feed(html)
    parser.close()
    expected = ["Year", *SEASONS]
    if not any([cell.split()[0] if cell else "" for cell in row] == expected for row in parser.rows):
        raise ValueError("NOAA seasonal table header is missing or has changed order.")
    by_year: dict[int, list[str]] = {}
    for row in parser.rows:
        if not row or not re.fullmatch(r"\d{4}", row[0]):
            continue
        year = int(row[0])
        if year < 1950:
            continue
        if not 2 <= len(row) <= len(COLUMNS):
            raise ValueError(f"Unexpected number of cells in year {year}: {len(row)}")
        padded = row + [""] * (len(COLUMNS) - len(row))
        missing_started = False
        for value in padded[1:]:
            if value == "":
                missing_started = True
                continue
            if missing_started:
                raise ValueError(f"Interior missing value in year {year}")
            if not re.fullmatch(r"[+-]?\d+\.\d", value):
                raise ValueError(f"Unexpected table value {value!r} in year {year}")
            if not -6.0 < float(value) < 6.0:
                raise ValueError(f"Implausible ENSO anomaly in year {year}: {value}")
        if year in by_year and by_year[year] != padded:
            raise ValueError(f"Conflicting duplicate table rows for {year}")
        by_year[year] = padded
    if not by_year or min(by_year) != 1950:
        raise ValueError("Expected a historical table beginning in 1950.")
    if sorted(by_year) != list(range(1950, max(by_year) + 1)):
        raise ValueError("Historical table contains missing years.")
    for year, row in by_year.items():
        if year != max(by_year) and "" in row[1:]:
            raise ValueError(f"Incomplete historical year {year}")
    return [dict(zip(COLUMNS, by_year[year])) for year in sorted(by_year)]


def parse_oni_ascii(text: str) -> dict[tuple[int, str], float]:
    """Parse the two-decimal raw ONI anomaly for provenance cross-checks."""
    lines = text.splitlines()
    if not lines or lines[0].split() != ["SEAS", "YR", "TOTAL", "ANOM"]:
        raise ValueError("Unexpected ONI ASCII header.")
    result: dict[tuple[int, str], float] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split()
        if len(cells) != 4 or cells[0] not in SEASONS:
            raise ValueError(f"Unexpected ONI ASCII row: {line!r}")
        season, year, _total, anomaly = cells
        key, value = (int(year), season), float(anomaly)
        if not math.isfinite(value) or not -6 < value < 6 or key in result:
            raise ValueError(f"Invalid or duplicate ONI ASCII entry: {line!r}")
        result[key] = value
    return result


def build_snapshot(payloads: dict[str, bytes], retrieval: dict) -> dict[str, bytes]:
    """Validate all inputs before preparing any files for publication."""
    outputs = dict(payloads)
    tables = {}
    for name in ("oni", "roni"):
        tables[name] = parse_index_html(payloads[f"{name}_noaa_v6_raw.html"].decode("utf-8"))
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(tables[name])
        outputs[f"{name}_noaa.csv"] = buffer.getvalue().encode("utf-8")
    raw = parse_oni_ascii(payloads["oni_noaa_raw.txt"].decode("ascii"))
    for row in tables["oni"]:
        for season in SEASONS:
            if row[season] == "":
                continue
            key = (int(row["year"]), season)
            if key not in raw or abs(float(row[season]) - raw[key]) > 0.050001:
                raise ValueError(f"ONI HTML/ASCII sources disagree at {key}; retry or inspect versions.")
    provenance = {
        "retrieval": retrieval,
        "parsed_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "ERSSTv6",
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                   **({"source_url": SOURCES[name]} if name in SOURCES else {})}
            for name, data in outputs.items()
        },
        "oni": {"table_url": SOURCES["oni_noaa_v6_raw.html"], "csv": "oni_noaa.csv",
                "base_period": "Centered 30-year periods updated every 5 years."},
        "roni": {"table_url": SOURCES["roni_noaa_v6_raw.html"], "csv": "roni_noaa.csv",
                 "base_period": "1991-2020; tropical-mean anomaly removed; variance rescaled."},
        "schema": {"year": "Year of central month: DJF 2016 is Dec 2015-Feb 2016.",
                   "unit": "degrees Celsius", "precision": "Published one-decimal HTML values.",
                   "missing": "Blank means unavailable, never zero."},
        "forecast_cutoff": {
            "recommended_feature": "For November 1 in year y, use JAS[y]; publication of ASO is not assured.",
            "forbidden_predictor": "DJF[y+1] or final event strength at a November 1 issue date.",
            "warning": "Full record is downloaded. The modeling pipeline must enforce season and feature cutoffs.",
            "revision_caveat": "Latest revised historical indices support pseudo-real-time, not vintage-perfect, hindcasts."},
        "classification_caveat": "Rounded DJF ONI >=1.5 is a demo proxy, not full ENSO event identification. Threshold results depend on rounding, season and index.",
        "operational_change": {"date": "2026-02", "note": "CPC adopted RONI for official monitoring and prediction.",
                               "url": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/announcement.php"},
    }
    outputs["enso_provenance.json"] = (json.dumps(provenance, indent=2) + "\n").encode("utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--offline-dir", type=Path, help="Read existing raw snapshots instead of using the network.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Per-request timeout in seconds (default: 45).")
    args = parser.parse_args()
    if args.offline_dir:
        payloads = {name: (args.offline_dir / name).read_bytes() for name in SOURCES}
        retrieval = {"mode": "offline_snapshot", "downloaded_at_utc": None,
                     "note": "Reparsed stored bytes; parsing time is not their original download time."}
        prior = args.offline_dir / "enso_provenance.json"
        if prior.is_file():
            metadata = json.loads(prior.read_text())
            retrieval["original_retrieval"] = metadata.get("retrieval", {
                "retrieved_date_local": metadata.get("retrieved_date_local"),
                "timezone": metadata.get("timezone")})
    else:
        payloads = {}
        started = datetime.now(timezone.utc).isoformat()
        for name, url in SOURCES.items():
            request = Request(url, headers={"User-Agent": "copper-snowfall-research/0.1"})
            with urlopen(request, timeout=args.timeout) as response:
                payloads[name] = response.read()
        retrieval = {"mode": "live_download", "started_at_utc": started,
                     "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    outputs = build_snapshot(payloads, retrieval)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        destination = args.output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
    print(f"Validated and wrote {len(outputs)} ENSO files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
