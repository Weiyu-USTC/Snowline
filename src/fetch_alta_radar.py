"""Download and exactly decode public MRMS GRIB2-PNG fields around Alta.

This is numeric GRIB data, not rendered weather-map imagery. Template 5.41
embeds integer samples in PNG; the GRIB reference and decimal/binary scales
convert them to physical values. Unsupported templates fail explicitly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import requests

BASE = "https://noaa-mrms-pds.s3.amazonaws.com/"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/alta_nowcast/radar"
LAT, LON = 40.57607, -111.63889  # Alta-Collins manual study plot, Hive readme.
REGION_LAT, REGION_LON = 40.583, -111.638  # Fixed regional grid anchor, not station coordinates.
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
_local = threading.local()


def session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = "Alta-public-radar-research/1.0"
    return _local.session


def uint(b):
    return int.from_bytes(b, "big")


def signed_grib(b):
    """GRIB integers use sign magnitude, not two's complement."""
    n = uint(b)
    sign = 1 << (8 * len(b) - 1)
    return -(n & (sign - 1)) if n & sign else n


def decode_crop(payload: bytes, native_size=256, pool=2, *, point=None, center=None):
    point_lat, point_lon = (LAT, LON) if point is None else point
    center_lat, center_lon = (REGION_LAT, REGION_LON) if center is None else center
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    if raw[:4] != b"GRIB" or raw[7] != 2 or raw[-4:] != b"7777":
        raise ValueError("Not a complete GRIB2 message")
    if uint(raw[8:16]) != len(raw):
        raise ValueError("Unexpected multiple/truncated GRIB messages")
    sections, pos = {}, 16
    while pos < len(raw) - 4:
        length = uint(raw[pos:pos+4])
        if length < 5:
            raise ValueError("Invalid GRIB section")
        number = raw[pos + 4]
        if number in sections:
            raise ValueError("Multiple GRIB fields unsupported")
        sections[number] = raw[pos:pos+length]
        pos += length
    s1, s3, s5, s6 = (sections[i] for i in (1, 3, 5, 6))
    if uint(s3[12:14]) != 0 or uint(s5[9:11]) != 41 or s6[5] != 255:
        raise ValueError("Expected regular-latlon grid, PNG packing, no bitmap")
    if s3[71] != 0:
        raise ValueError("Unsupported scan direction")
    nx, ny = uint(s3[30:34]), uint(s3[34:38])
    basic, subdivisions = uint(s3[38:42]), uint(s3[42:46])
    unit = basic / subdivisions if basic and subdivisions != 0xFFFFFFFF else 1e-6
    lat0 = signed_grib(s3[46:50]) * unit
    lon0 = signed_grib(s3[50:54]) * unit
    dx, dy = uint(s3[63:67]) * unit, uint(s3[67:71]) * unit
    r = round((lat0 - center_lat) / dy)
    c = round(((center_lon % 360) - lon0) / dx)
    station_r = round((lat0-point_lat)/dy)
    station_c = round(((point_lon % 360)-lon0)/dx)
    r0, c0 = r - native_size // 2, c - native_size // 2
    if native_size % pool or min(r0, c0) < 0 or r0+native_size>ny or c0+native_size>nx:
        raise ValueError("Invalid crop bounds or pooling")
    if not (r0 <= station_r < r0 + native_size and c0 <= station_c < c0 + native_size):
        raise ValueError("Radar point outside crop")
    pic = Image.open(io.BytesIO(sections[7][5:]))
    if pic.size != (nx, ny):
        raise ValueError("PNG and GRIB grid differ")
    integers = np.asarray(pic.crop((c0, r0, c0+native_size, r0+native_size)))
    reference = struct.unpack(">f", s5[11:15])[0]
    binary, decimal = signed_grib(s5[15:17]), signed_grib(s5[17:19])
    values = (reference + integers.astype(np.float32) * 2.0**binary) * 10.0**(-decimal)
    valid = np.isfinite(values) & (values >= 0)
    # Preserve missing fractions. Missing data never become physical zero.
    shape = (native_size//pool, pool, native_size//pool, pool)
    count = valid.reshape(shape).sum((1, 3))
    total = np.where(valid, values, 0).reshape(shape).sum((1, 3))
    pooled = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
    validity = count.astype(np.float32) / pool**2
    lats = lat0 - (r0 + np.arange(native_size//pool)*pool + (pool-1)/2) * dy
    lons = lon0 + (c0 + np.arange(native_size//pool)*pool + (pool-1)/2) * dx
    lons = (lons + 180) % 360 - 180
    valid_time = datetime(uint(s1[12:14]), *s1[14:19], tzinfo=timezone.utc)
    metadata = {
        "valid_time": valid_time.isoformat(), "native_grid": [ny, nx],
        "native_degrees": [dy, dx], "crop_native_origin": [r0, c0],
        "crop_native_shape": [native_size, native_size], "pool": pool,
        "reference": reference, "binary_scale": binary, "decimal_scale": decimal,
        "packing_template": 41, "grid_template": 0,
        "grib2_product": [raw[6], sections[4][9], sections[4][10]],
        "point_native_value": float(values[station_r-r0, station_c-c0]),
        "alta_native_value": float(values[station_r-r0, station_c-c0]),
        "point_extraction_version": 2,
        "station_latitude": point_lat, "station_longitude": point_lon,
        "station_native_row": station_r, "station_native_column": station_c,
        "station_native_latitude": lat0-station_r*dy,
        "station_native_longitude": (lon0+station_c*dx+180)%360-180,
        "negative_native_fraction": float((~valid).mean()),
        "missing_policy": "Negative physical values retained as missing, not zero; pooling ignores missing and preserves valid fraction.",
    }
    return pooled.astype(np.float32), validity, lats, lons, metadata


def get_bytes(url, retries=4):
    error = None
    for attempt in range(retries):
        try:
            r = session().get(url, timeout=(20, 100))
            r.raise_for_status()
            return r.content, dict(r.headers)
        except requests.RequestException as exc:
            error = exc
            if getattr(exc.response, "status_code", None) == 404:
                break
            time.sleep(0.5 * 2**attempt)
    raise error


def inventory(product, day):
    folder = OUT / "inventories"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{product}_{day}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows, token = [], None
    while True:
        params = {"list-type": "2", "prefix": f"CONUS/{product}_00.00/{day}/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session().get(BASE, params=params, timeout=40)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.findall("s:Contents", NS):
            row = {c.tag.rsplit("}", 1)[-1]: c.text for c in item}
            row["Size"] = int(row["Size"])
            rows.append(row)
        if root.findtext("s:IsTruncated", default="false", namespaces=NS) != "true":
            break
        token = root.findtext("s:NextContinuationToken", namespaces=NS)
    path.write_text(json.dumps(rows, indent=2))
    return rows


def fetch_one(product, stamp, native_size=256, pool=2):
    date, ts = stamp.strftime("%Y%m%d"), stamp.strftime("%Y%m%d-%H%M%S")
    name = f"MRMS_{product}_00.00_{ts}.grib2.gz"
    url = f"{BASE}CONUS/{product}_00.00/{date}/{name}"
    folder = OUT / "raw" / product / date
    path = folder / name
    derived = OUT / "frames" / product / date / f"{ts}.npz"
    meta = derived.with_suffix(".json")
    previous_meta = {}
    if derived.exists() and meta.exists():
        previous_meta = json.loads(meta.read_text())
        if previous_meta.get("point_extraction_version") == 2:
            return previous_meta
    folder.mkdir(parents=True, exist_ok=True)
    headers = {}
    if path.exists():
        payload = path.read_bytes()
    else:
        payload, headers = get_bytes(url)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
    values, valid, lat, lon, attrs = decode_crop(payload, native_size, pool)
    if attrs["valid_time"] != stamp.isoformat():
        raise ValueError(f"Radar time differs: {attrs['valid_time']} vs {stamp}")
    derived.parent.mkdir(parents=True, exist_ok=True)
    coarse_shape = (values.shape[0]//2, 2, values.shape[1]//2, 2)
    coarse_count = valid.reshape(coarse_shape).sum((1, 3))
    coarse_sum = np.where(np.isfinite(values), values*valid, 0).reshape(coarse_shape).sum((1, 3))
    coarse = np.divide(coarse_sum, coarse_count, out=np.full(coarse_sum.shape, np.nan), where=coarse_count > 0)
    np.savez_compressed(derived, values=values, valid_fraction=valid,
                        values_64=coarse.astype(np.float32), valid_fraction_64=coarse_count/4,
                        latitude=lat, longitude=lon,
                        alta_native_value=attrs["alta_native_value"],
                        valid_time=np.datetime64(stamp.replace(tzinfo=None), "s"))
    attrs.update({"product": product, "url": url, "sha256": hashlib.sha256(payload).hexdigest(),
                  "bytes": len(payload), "raw_path": str(path.relative_to(ROOT)),
                  "derived_path": str(derived.relative_to(ROOT)),
                  "downloaded_utc": previous_meta.get("downloaded_utc", datetime.now(timezone.utc).isoformat()),
                  "derived_utc": datetime.now(timezone.utc).isoformat(),
                  "http_last_modified": headers.get("Last-Modified",previous_meta.get("http_last_modified")),
                  "http_etag": headers.get("ETag",previous_meta.get("http_etag")),
                  "units": {"PrecipRate": "mm h-1 liquid-equivalent", "RadarQualityIndex": "unitless", "PrecipFlag": "categorical"}.get(product, "See NOAA product table")})
    meta.write_text(json.dumps(attrs, indent=2))
    return attrs


def parse_time(x):
    return datetime.fromisoformat(x.replace("Z", "+00:00")).astimezone(timezone.utc)


def assemble(schedule_path, output_path=None):
    """Stack a frozen cohort, keeping absent/invalid radar separate from zero."""
    plan = json.loads(Path(schedule_path).read_text())
    cases = plan["cases"] if isinstance(plan, dict) else plan
    count = len(cases)
    rate = np.full((count, 29, 64, 64), np.nan, np.float32)
    valid = np.zeros(rate.shape, np.float32)
    rqi = np.full((count, 6, 64, 64), np.nan, np.float32)
    rate_alta = np.full((count, 29), np.nan, np.float32)
    rqi_alta = np.full((count, 6), np.nan, np.float32)
    present = np.zeros((count, 29), bool)
    qpresent = np.zeros((count, 6), bool)
    point_version = np.zeros((count,29),np.uint8)
    qpoint_version = np.zeros((count,6),np.uint8)
    radar_time = np.empty((count, 29), "datetime64[s]")
    quality_time = np.empty((count, 6), "datetime64[s]")
    lat = lon = None
    summary = []
    for i, case in enumerate(cases):
        start = parse_time(case["window_start_utc"])
        for j in range(29):
            dt = start + timedelta(minutes=10*j)
            radar_time[i, j] = dt.replace(tzinfo=None)
            file = OUT / "frames/PrecipRate" / dt.strftime("%Y%m%d") / (dt.strftime("%Y%m%d-%H%M%S") + ".npz")
            if file.exists():
                with np.load(file) as z:
                    rate[i,j], valid[i,j] = z["values_64"], z["valid_fraction_64"]
                    point_version[i,j]=json.loads(file.with_suffix(".json").read_text()).get("point_extraction_version",1)
                    if point_version[i,j]==2:
                        point=float(z["alta_native_value"])
                        rate_alta[i,j] = point if point>=0 else np.nan
                    present[i,j] = True
                    this_lat, this_lon = z["latitude"].reshape(64,2).mean(1), z["longitude"].reshape(64,2).mean(1)
                    if lat is None:
                        lat, lon = this_lat, this_lon
                    elif not (np.allclose(lat, this_lat) and np.allclose(lon, this_lon)):
                        raise ValueError("Inconsistent grid geometry")
        for j in range(6):
            dt = start.replace(minute=0) + timedelta(hours=j)
            quality_time[i,j] = dt.replace(tzinfo=None)
            file = OUT / "frames/RadarQualityIndex" / dt.strftime("%Y%m%d") / (dt.strftime("%Y%m%d-%H%M%S") + ".npz")
            if file.exists():
                with np.load(file) as z:
                    rqi[i,j] = z["values_64"]
                    qpoint_version[i,j]=json.loads(file.with_suffix(".json").read_text()).get("point_extraction_version",1)
                    if qpoint_version[i,j]==2:
                        point=float(z["alta_native_value"])
                        rqi_alta[i,j] = point if point>=0 else np.nan
                    qpresent[i,j] = True
        summary.append({"case_id":case["case_id"],"split":case["split"],
                        "rate_frames_present":int(present[i].sum()),"quality_frames_present":int(qpresent[i].sum()),
                        "rate_valid_fraction":float(valid[i].mean()),
                        "alta_rate_max":float(np.nanmax(rate_alta[i])) if np.isfinite(rate_alta[i]).any() else None,
                        "alta_rqi_median":float(np.nanmedian(rqi_alta[i])) if np.isfinite(rqi_alta[i]).any() else None})
    path = Path(output_path) if output_path else OUT / "radar_sequences.npz"
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, rate=rate, valid_fraction=valid, rqi_hourly=rqi,
                        rate_alta_native=rate_alta, rqi_alta_native=rqi_alta,
                        frame_present=present, quality_present=qpresent,
                        point_extraction_version=point_version,quality_point_extraction_version=qpoint_version,
                        radar_time=radar_time, quality_time=quality_time,
                        case_id=np.array([c["case_id"] for c in cases]),
                        split=np.array([c["split"] for c in cases]),
                        winter=np.array([c["winter"] for c in cases]),
                        latitude=lat, longitude=lon)
    temporary.replace(path)
    audit={"cases":count,"complete_rate_cases":int(present.all(1).sum()),
           "complete_quality_cases":int(qpresent.all(1).sum()),
           "rate_shape":list(rate.shape),"rqi_shape":list(rqi.shape),
           "latitude":lat.tolist(),"longitude":lon.tolist(),
           "min_rate":float(np.nanmin(rate)),"max_rate":float(np.nanmax(rate)),
           "min_rqi":float(np.nanmin(rqi)),"max_rqi":float(np.nanmax(rqi)),
           "station_rqi_quantiles":np.nanquantile(rqi_alta,[0,.1,.25,.5,.75,.9,1]).tolist(),
           "regional_rqi_quantiles":np.nanquantile(rqi,[0,.1,.25,.5,.75,.9,1]).tolist(),
           "station_point_correction_complete":bool(np.all(point_version[present]==2) and np.all(qpoint_version[qpresent]==2)),
           "regional_tensor_sha256":{
               "rate":hashlib.sha256(rate.tobytes()).hexdigest(),
               "valid_fraction":hashlib.sha256(valid.tobytes()).hexdigest(),
               "rqi_hourly":hashlib.sha256(rqi.tobytes()).hexdigest()},
           "units":"rate mm/h liquid equivalent; RQI dimensionless0–1; fresh snow NOT observed at10min cadence",
           "quality_causality":"Only09Z/10Zqualityfieldsareavailableasinput at10:50UTC.11–14Zqualityisverification-only.",
           "case_summary":summary}
    path.with_suffix(".json").write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in ("latitude","longitude","case_summary")}),flush=True)
    return path


def fetch_current(issue_utc, workers=8):
    """Fetch only nine observations available before the requested issue.

    A conservative ten-minute buffer is applied to MRMS valid timestamps.
    No future radar or future RQI is included in the current model inputs.
    """
    issue = parse_time(issue_utc) if isinstance(issue_utc, str) else issue_utc
    if issue.minute % 10 or issue.second or issue.microsecond:
        raise ValueError("Issue must be on a ten-minute boundary")
    if issue > datetime.now(timezone.utc):
        raise ValueError("Cannot issue an actual forecast in the future")
    last = issue - timedelta(minutes=10)
    times = [last-timedelta(minutes=10*j) for j in range(8,-1,-1)]
    qtimes = sorted({t.replace(minute=0,second=0) for t in times})
    jobs = [("PrecipRate",t) for t in times]+[("RadarQualityIndex",t) for t in qtimes]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        metas=list(pool.map(lambda pair:fetch_one(*pair),jobs))
    rate,valid,rqi,alta=[],[],[],[]
    for t in times:
        p=OUT/"frames/PrecipRate"/t.strftime("%Y%m%d")/(t.strftime("%Y%m%d-%H%M%S")+".npz")
        q=t.replace(minute=0,second=0)
        qp=OUT/"frames/RadarQualityIndex"/q.strftime("%Y%m%d")/(q.strftime("%Y%m%d-%H%M%S")+".npz")
        with np.load(p) as z:
            rate.append(z["values_64"]);valid.append(z["valid_fraction_64"]);alta.append(z["alta_native_value"])
            lat=z["latitude"].reshape(64,2).mean(1);lon=z["longitude"].reshape(64,2).mean(1)
        with np.load(qp) as z:
            rqi.append(z["values_64"])
    folder=OUT/"current"/issue.strftime("%Y%m%d-%H%MZ")
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"context.npz",rate=np.asarray(rate)[None],valid_fraction=np.asarray(valid)[None],
                        rqi=np.asarray(rqi)[None],rate_alta_native=np.asarray(alta)[None],latitude=lat,longitude=lon,
                        radar_time=np.array([t.replace(tzinfo=None) for t in times],dtype="datetime64[s]"),
                        issue_time=np.datetime64(issue.replace(tzinfo=None),"s"))
    (folder/"manifest.json").write_text(json.dumps({"issue_utc":issue.isoformat(),"last_radar_valid_utc":last.isoformat(),
                        "availability_buffer_minutes":10,"radar_observations":9,"future_observations_used":False,
                        "domain_note":"Training issues were fixed10:50UTC; arbitrary-hour inference is an out-of-training-hour application.",
                        "sources":metas},indent=2))
    print(str(folder/"context.npz"),flush=True)
    return folder/"context.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", type=Path, help="JSON list or {cases: list}; each has window_start_utc/window_end_utc")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--step-minutes", type=int, default=10)
    ap.add_argument("--native-size", type=int, default=256)
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--quality-hours", type=int, default=1)
    ap.add_argument("--manifest-name", default="download_manifest.json")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--current-issue", help="UTC issue on a ten-minute boundary; nine past radar frames only")
    args = ap.parse_args()
    if args.current_issue:
        fetch_current(args.current_issue,args.workers)
        return
    if args.assemble_only:
        assemble(args.schedule)
        return
    if args.schedule:
        cases = json.loads(args.schedule.read_text())
        if isinstance(cases, dict):
            cases = cases["cases"]
    else:
        cases = [{"window_start_utc": args.start, "window_end_utc": args.end}]
    jobs = set()
    for case in cases:
        t, end = parse_time(case["window_start_utc"]), parse_time(case["window_end_utc"])
        qt = t.replace(minute=0, second=0)
        while qt <= end:
            if qt.hour % args.quality_hours == 0:
                jobs.add(("RadarQualityIndex", qt))
            qt += timedelta(hours=1)
        while t <= end:
            jobs.add(("PrecipRate", t))
            if t.minute == 0 and t.hour % args.quality_hours == 0:
                jobs.add(("RadarQualityIndex", t))
            t += timedelta(minutes=args.step_minutes)
    jobs = sorted(jobs, key=lambda x: (x[1], x[0]))
    OUT.mkdir(parents=True, exist_ok=True)
    results, failures, start = [], [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, prod, t, args.native_size, args.pool): (prod, t) for prod, t in jobs}
        for i, f in enumerate(as_completed(futures), 1):
            prod, stamp = futures[f]
            try:
                results.append(f.result())
            except Exception as exc:
                failures.append({"product": prod, "time": stamp.isoformat(), "error": str(exc)})
            if i % 100 == 0 or i == len(jobs):
                print(json.dumps({"completed": i, "total": len(jobs), "failures": len(failures),
                                  "elapsed_s": round(time.time()-start), "downloaded_bytes": sum(r["bytes"] for r in results)}), flush=True)
                (OUT / args.manifest_name).write_text(json.dumps({"source": BASE, "results": results, "failures": failures}, indent=2))
    print("Finished", OUT / args.manifest_name, flush=True)


if __name__ == "__main__":
    main()
