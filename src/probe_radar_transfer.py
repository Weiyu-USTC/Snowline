"""Small, calendar-selected feasibility probe; does not train or alter Alta data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

import fetch_alta_radar as radar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/radar-transfer'
MOUNTAINS = {
    'vail': {'name': 'Vail', 'latitude': 39.6176, 'longitude': -106.38,
             'target': 'CAVMM station proxy; fresh-snow/reset protocol remains unverified'},
    'stowe': {'name': 'Stowe', 'latitude': 44.5247, 'longitude': -72.8153,
              'target': 'NOAA Mount Mansfield daily station, not resort-wide or hourly ground truth'},
    'jay': {'name': 'Jay Peak', 'latitude': 44.9381, 'longitude': -72.5022,
            'target': 'NOAA Jay Peak daily station at 561 m, not summit or hourly ground truth'},
}


def crop_probe():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = [f'{year}1215' for year in range(2020, 2024)]
    dates += [f'{year}{month:02d}15' for year in range(2021, 2025) for month in (2, 4)]
    rows, missing, provenance = [], [], []
    for day in sorted(dates):
        stamp = day + '-100000'
        source = {}
        for product in ('PrecipRate', 'RadarQualityIndex'):
            p = ROOT / f'data/alta_nowcast/radar/raw/{product}/{day}/MRMS_{product}_00.00_{stamp}.grib2.gz'
            if not p.exists():
                missing.append(str(p.relative_to(ROOT)))
                break
            source[product] = p.read_bytes()
            provenance.append({'path': str(p.relative_to(ROOT)), 'bytes': len(source[product]),
                               'sha256': hashlib.sha256(source[product]).hexdigest()})
        if len(source) != 2:
            continue
        for mountain, loc in MOUNTAINS.items():
            radar.LAT = radar.REGION_LAT = loc['latitude']
            radar.LON = radar.REGION_LON = loc['longitude']
            rate, valid, lat, lon, rm = radar.decode_crop(source['PrecipRate'], native_size=256, pool=4)
            q, qvalid, _, _, qm = radar.decode_crop(source['RadarQualityIndex'], native_size=256, pool=4)
            input_mask = (valid >= .9) & (q >= .1)
            target_mask = (valid >= .9) & (q >= .5)
            interior = np.s_[4:-4, 4:-4]
            point_rate, point_rqi = rm['alta_native_value'], qm['alta_native_value']
            rows.append(dict(mountain=mountain, name=loc['name'], timestamp_utc=rm['valid_time'],
                             native_station_rate_mm_hr=point_rate,
                             native_station_rqi=point_rqi,
                             native_station_meets_input_rqi=point_rqi >= .1,
                             native_station_meets_target_rqi=point_rqi >= .5,
                             native_station_rate_valid=point_rate >= 0,
                             region_finite_rate_fraction=float(np.isfinite(rate).mean()),
                             region_rate_valid_fraction_ge_09=float((valid >= .9).mean()),
                             region_input_eligible_fraction=float(input_mask.mean()),
                             region_target_eligible_fraction=float(target_mask[interior].mean()),
                             region_mean_rqi=float(np.nanmean(q)),
                             region_mean_rate_mm_hr=float(np.nanmean(rate)),
                             region_max_rate_mm_hr=float(np.nanmax(rate))))
            np.savez_compressed(OUT / f'{mountain}_{stamp}.npz', rate_mm_hr=rate,
                                valid_fraction=valid, radar_quality_index=q, quality_valid_fraction=qvalid,
                                input_mask=input_mask, target_mask=target_mask,
                                latitude=lat, longitude=lon, station_latitude=loc['latitude'],
                                station_longitude=loc['longitude'], native_station_rate_mm_hr=point_rate,
                                native_station_rqi=point_rqi, timestamp_utc=rm['valid_time'])
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'crop_quality_probe.csv', index=False)
    summary = []
    for mountain, sub in frame.groupby('mountain'):
        summary.append(dict(mountain=mountain, cases=len(sub),
                            station_rqi_min=float(sub.native_station_rqi.min()),
                            station_rqi_median=float(sub.native_station_rqi.median()),
                            station_rqi_max=float(sub.native_station_rqi.max()),
                            station_rqi_ge_05_count=int(sub.native_station_meets_target_rqi.sum()),
                            station_rqi_ge_01_count=int(sub.native_station_meets_input_rqi.sum()),
                            mean_region_target_fraction=float(sub.region_target_eligible_fraction.mean()),
                            min_region_target_fraction=float(sub.region_target_eligible_fraction.min()),
                            max_region_target_fraction=float(sub.region_target_eligible_fraction.max())))
    result = {'as_of_date': '2026-11-07', 'sample_rule': 'December 15, February 15, April 15 at 10:00 UTC in each winter 2020/21 through 2023/24; calendar selected without storm outcomes',
              'limits': 'Twelve snapshots per mountain only. This verifies decodability and exposes quality limitations; it is not a complete archive audit, climatology, storm evaluation, or independent surface snowfall validation.',
              'quality_rules': {'input': 'valid_fraction >= 0.9 and RQI >= 0.1', 'target': 'valid_fraction >= 0.9 and RQI >= 0.5; four outer cells excluded'},
              'decoder': 'Existing fetch_alta_radar.decode_crop, same physical GRIB scaling and missing-value policy; per-process location override only',
              'shape': [64, 64], 'native_crop_shape': [256, 256], 'grid_spacing_degrees': .04,
              'mountains': MOUNTAINS, 'summary': summary, 'missing_sources': missing, 'source_files': provenance}
    (OUT / 'crop_quality_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'summary': summary, 'missing_sources': missing}, indent=2))


def online_probe():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = [
        ('mrms_first_preciprate', 'https://noaa-mrms-pds.s3.amazonaws.com/', 'CONUS/PrecipRate_00.00/'),
        ('mrms_first_rqi', 'https://noaa-mrms-pds.s3.amazonaws.com/', 'CONUS/RadarQualityIndex_00.00/'),
        ('mrms_recent_preciprate', 'https://noaa-mrms-pds.s3.amazonaws.com/', 'CONUS/PrecipRate_00.00/20261106/'),
    ]
    for site in ('KGJX', 'KFTG', 'KCXX', 'KTYX'):
        for prefix in (f'2024/01/15/{site}/', f'2026/11/06/{site}/'):
            checks.append((f'nexrad_{site}_{prefix[:10]}', 'https://unidata-nexrad-level2.s3.amazonaws.com/', prefix))
    results = []
    for name, url, prefix in checks:
        r = requests.get(url, params={'list-type': '2', 'prefix': prefix, 'max-keys': '1'}, timeout=30)
        row = {'name': name, 'url': r.url, 'http_status': r.status_code}
        if r.ok:
            root = ET.fromstring(r.content)
            row['keys'] = [e.text for e in root.findall('.//{*}Key')]
            row['sizes'] = [int(e.text) for e in root.findall('.//{*}Size')]
        results.append(row)
    # Metadata is small; inspect documented forecast arrays and public chunks.
    from pyproj import Proj
    proj = Proj(a=6371229, b=6371229, proj='lcc', lon_0=262.5, lat_0=38.5, lat_1=38.5, lat_2=38.5)
    for day in ('20240115', '20261106'):
        spec = 'surface/APCP_1hr_acc_fcst'
        base = f'https://hrrrzarr.s3.amazonaws.com/sfc/{day}/{day}_06z_fcst.zarr/{spec}/'
        r = requests.get(base + '.zmetadata', timeout=30)
        row = {'name': f'hrrr_metadata_{day}', 'url': r.url, 'http_status': r.status_code}
        if r.ok:
            metadata = r.json()['metadata']
            row['array_metadata'] = metadata[spec + '/.zarray']
            row['attributes'] = metadata[spec + '/.zattrs']
        results.append(row)
        for mountain, loc in MOUNTAINS.items():
            x, y = proj(loc['longitude'], loc['latitude'])
            ix, iy = round((x + 2697520.1425219304) / 3000), round((y + 1587306.1525566636) / 3000)
            url = base + spec + f'/0.{iy // 150}.{ix // 150}'
            r = requests.head(url, timeout=30)
            results.append({'name': f'hrrr_{mountain}_{day}', 'url': url, 'http_status': r.status_code,
                            'bytes': r.headers.get('Content-Length'), 'grid_x': ix, 'grid_y': iy,
                            'verification': 'Public object HEAD only; numeric content not downloaded or decoded'})
    artifact = {'as_of_date': '2026-11-07', 'method': 'Anonymous HTTPS; S3 max-keys=1 listings, small HRRR metadata GET and chunk HEAD only. No bulk download.', 'results': results}
    (OUT / 'public_archive_probe.json').write_text(json.dumps(artifact, indent=2))
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()
    if args.online:
        online_probe()
    else:
        crop_probe()
