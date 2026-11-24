"""Extract Mammoth cells and crops from retained, time-matched national MRMS scans."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from mammoth_radar import FOLDER, archive_pair, decode_field


def write_point_index(root, records):
    """Write a chronological table of the exact fields archived for the backfill."""
    target = root / FOLDER / "historical_points.csv"
    fields = ["timestamp", "precip_rate_mm_h", "quality_index", "coverage",
              "cell_latitude", "cell_longitude", "archive_path", "url", "quality_url"]
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", newline="", dir=target.parent, delete=False) as f:
            temporary = Path(f.name)
            writer = csv.DictWriter(f, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target.relative_to(root)


def backfill(root):
    source = root / "data/alta_nowcast/radar/raw"
    files = {}
    for product in ("PrecipRate", "RadarQualityIndex"):
        files[product] = {
            p.name.split("_")[-1].removesuffix(".grib2.gz"): p
            for p in (source / product).glob("*/*.grib2.gz")
        }
    stamps = sorted(files["PrecipRate"].keys() & files["RadarQualityIndex"].keys())
    if not stamps:
        raise ValueError(f"No retained matched radar pairs under {source}")

    def extract(key):
        stamp = datetime.strptime(key, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
        rate = decode_field(files["PrecipRate"][key].read_bytes(), "PrecipRate")
        quality = decode_field(files["RadarQualityIndex"][key].read_bytes(), "RadarQualityIndex")
        return archive_pair(root, stamp, rate, quality, "Retained Alta study timestamps")

    counts = Counter()
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(extract, stamps):
            records.append(record)
            counts[record["coverage"]] += 1
            rate = record["precip_rate_mm_h"]
            if rate is not None and rate > 0:
                counts["positive_rate"] += 1
    return dict(scans=len(stamps), days=len({stamp[:8] for stamp in stamps}),
                first=stamps[0], last=stamps[-1], counts=dict(counts),
                index=str(write_point_index(root, records)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(backfill(args.root), indent=2))
