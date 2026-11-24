"""Collect NOAA MRMS liquid precipitation and radar quality at Mammoth Main Lodge."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile
import zlib

import numpy as np
import requests

from fetch_alta_radar import decode_crop
from snowline_sites import SITES

POINT = (SITES["mammoth"]["latitude"], SITES["mammoth"]["longitude"])
BASE = "https://mrms.ncep.noaa.gov/2D"
PUBLIC_ARCHIVE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS"
PRODUCT_IDS = {"PrecipRate": [209, 6, 1], "RadarQualityIndex": [209, 8, 0]}
FOLDER = Path("data/mammoth/radar")
# Same minimum RQI used for radar inputs in the existing Snowline pipeline.
MIN_RQI = 0.1
MAX_DOWNLOAD_AGE = timedelta(minutes=20)
MAX_DISPLAY_AGE = timedelta(minutes=90)


def _empty():
    return dict(source="NOAA MRMS", product="PrecipRate", status="unavailable",
                timestamp=None, url=None, quality_timestamp=None, quality_url=None,
                coverage="unavailable", quality_index=None, precip_rate_mm_h=None,
                units="mm/h liquid equivalent", archive_path=None,
                message="No Mammoth radar sample has been collected yet.")


def _utc(now):
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Radar time must include a timezone.")
    return now.astimezone(timezone.utc)


def read_current_radar(root, now=None):
    """Read the most recent complete crop, including its original observation time."""
    now = _utc(now)
    folder = Path(root) / FOLDER
    for day in sorted(folder.glob("[0-9]" * 8), reverse=True):
        for path in sorted(day.glob("*.npz"), reverse=True):
            try:
                with np.load(path, allow_pickle=False) as saved:
                    record = json.loads(saved["metadata"].item())
                stamp = datetime.fromisoformat(record["timestamp"])
                if stamp.tzinfo is None:
                    continue
            except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile, zlib.error):
                continue
            if stamp > now:
                continue
            if now - stamp > MAX_DISPLAY_AGE:
                record.update(status="partial", coverage="stale", precip_rate_mm_h=None,
                              message="The latest Mammoth radar sample is over 90 minutes old.")
            return record
    return _empty()


def _source_urls(product, stamp):
    day = stamp.strftime("%Y%m%d")
    fname = f"MRMS_{product}_00.00_{stamp:%Y%m%d-%H%M%S}.grib2.gz"
    return f"{BASE}/{product}/{fname}", f"{PUBLIC_ARCHIVE}/{product}_00.00/{day}/{fname}"


def _field(url, product):
    response = requests.get(url, timeout=(5, 15))
    response.raise_for_status()
    return decode_field(response.content, product)


def decode_field(payload, product):
    grid, _, lat, lon, metadata = decode_crop(
        payload, native_size=64, pool=1, center=POINT, point=POINT)
    if metadata["grib2_product"] != PRODUCT_IDS[product]:
        raise ValueError(f"Unexpected GRIB2 product for {product}")
    metadata["sha256"] = hashlib.sha256(payload).hexdigest()
    return grid, lat, lon, metadata


def _point_coverage(rate, quality):
    if not np.isfinite(rate) or rate < 0:
        return "missing", None, "The Mammoth radar precipitation cell is missing."
    if not np.isfinite(quality) or not 0 <= quality <= 1:
        return "missing", None, "The Mammoth radar quality cell is missing."
    if quality < MIN_RQI:
        return "poor_quality", None, "Radar quality at Mammoth is too low to estimate precipitation."
    return "usable", float(rate), "Radar liquid precipitation rate at the Main Lodge grid cell."


def archive_pair(root, stamp, rate_field, quality_field, collection):
    """Persist two co-located fields observed at the same expected UTC time."""
    rate, lat, lon, info = rate_field
    quality, qlat, qlon, qinfo = quality_field
    if datetime.fromisoformat(info["valid_time"]) != stamp:
        raise ValueError("Precipitation timestamp differs from requested scan")
    if qinfo["valid_time"] != info["valid_time"]:
        raise ValueError("Quality and precipitation timestamps differ")
    if not np.array_equal(lat, qlat) or not np.array_equal(lon, qlon):
        raise ValueError("Quality and precipitation coordinates differ")

    quality_value = qinfo["point_native_value"]
    coverage, point_rate, message = _point_coverage(info["point_native_value"], quality_value)
    relative = FOLDER / stamp.strftime("%Y%m%d/%Y%m%d-%H%M%S.npz")
    record = dict(_empty(), status="succeeded" if coverage == "usable" else "partial",
                  coverage=coverage, precip_rate_mm_h=point_rate, timestamp=info["valid_time"],
                  quality_timestamp=qinfo["valid_time"], url=_source_urls("PrecipRate", stamp)[1],
                  quality_url=_source_urls("RadarQualityIndex", stamp)[1],
                  quality_index=quality_value if 0 <= quality_value <= 1 else None,
                  minimum_quality_index=MIN_RQI, latitude=POINT[0], longitude=POINT[1],
                  cell_latitude=info["station_native_latitude"],
                  cell_longitude=info["station_native_longitude"],
                  archive_path=str(relative), message=message, selection=collection,
                  rate_sha256=info["sha256"], quality_sha256=qinfo["sha256"])
    temporary = None
    try:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as f:
            temporary = Path(f.name)
            np.savez_compressed(f, precip_rate_mm_h=rate, radar_quality_index=quality,
                                latitude=lat, longitude=lon, metadata=json.dumps(record, allow_nan=False))
        temporary.replace(target)
        return record
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_current_radar(root, now=None):
    """Archive a matched precipitation/quality pair. A failed refresh preserves prior crops."""
    now = _utc(now)
    root = Path(root)
    try:
        latest = f"{BASE}/PrecipRate/MRMS_PrecipRate.latest.grib2.gz"
        try:
            rate_field = _field(latest, "PrecipRate")
            stamp = datetime.fromisoformat(rate_field[3]["valid_time"])
            if not timedelta(0) <= now - stamp <= MAX_DOWNLOAD_AGE:
                raise ValueError("Latest precipitation grid is stale")
            qurl, qarchive = _source_urls("RadarQualityIndex", stamp)
            try:
                quality_field = _field(qurl, "RadarQualityIndex")
            except (requests.RequestException, OSError, EOFError, ValueError):
                quality_field = _field(qarchive, "RadarQualityIndex")
        except (requests.RequestException, OSError, EOFError, ValueError):
            # NOAA also mirrors time-stamped grids to AWS. Allow time for both fields to arrive.
            stamp = now.replace(second=0, microsecond=0) - timedelta(minutes=6 + now.minute % 2)
            rate_field = _field(_source_urls("PrecipRate", stamp)[1], "PrecipRate")
            quality_field = _field(_source_urls("RadarQualityIndex", stamp)[1], "RadarQualityIndex")
        return archive_pair(root, stamp, rate_field, quality_field, "Live refresh")
    except (requests.RequestException, OSError, EOFError, ValueError, KeyError) as exc:
        previous = read_current_radar(root, now)
        return dict(previous, status="failed", message=f"Mammoth radar refresh failed. {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = fetch_current_radar(args.root)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result["status"] == "failed" else 0)
