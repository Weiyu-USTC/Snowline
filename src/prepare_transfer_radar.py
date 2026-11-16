"""Recrop retained CONUS radar into Vail/shared-Vermont morning datasets.

One process decodes each national PNG once. Original Alta files are read-only;
no network access, fitting, interpolation of absent targets or synthetic data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import time

import numpy as np
import pandas as pd
from PIL import Image

from fetch_alta_radar import signed_grib, uint

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/alta_nowcast/radar'
OUT = ROOT / 'outputs/radar-transfer/prepared'
REGIONS = {
    'vail': {'center': [39.6176, -106.38], 'stations': {'vail': [39.6176, -106.38]}},
    'vermont': {'center': [44.7314, -72.65875],
                'stations': {'stowe': [44.5247, -72.8153], 'jay': [44.9381, -72.5022]}},
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode_regions(payload):
    raw = gzip.decompress(payload) if payload[:2] == b'\x1f\x8b' else payload
    if raw[:4] != b'GRIB' or raw[7] != 2 or raw[-4:] != b'7777' or uint(raw[8:16]) != len(raw):
        raise ValueError('Expected one complete GRIB2 message')
    sections, pos = {}, 16
    while pos < len(raw)-4:
        length = uint(raw[pos:pos+4]); number = raw[pos+4]
        if length < 5 or number in sections:
            raise ValueError('Invalid/multiple GRIB fields')
        sections[number] = raw[pos:pos+length]; pos += length
    s1, s3, s5, s6 = (sections[k] for k in (1, 3, 5, 6))
    if uint(s3[12:14]) != 0 or uint(s5[9:11]) != 41 or s6[5] != 255 or s3[71] != 0:
        raise ValueError('Unsupported grid, packing, bitmap or scan direction')
    nx, ny = uint(s3[30:34]), uint(s3[34:38])
    basic, subdivisions = uint(s3[38:42]), uint(s3[42:46])
    unit = basic/subdivisions if basic and subdivisions != 0xffffffff else 1e-6
    lat0, lon0 = signed_grib(s3[46:50])*unit, signed_grib(s3[50:54])*unit
    dx, dy = uint(s3[63:67])*unit, uint(s3[67:71])*unit
    reference = struct.unpack('>f', s5[11:15])[0]
    binary, decimal = signed_grib(s5[15:17]), signed_grib(s5[17:19])
    valid_time = datetime(uint(s1[12:14]), *s1[14:19], tzinfo=timezone.utc)
    result = {}
    with Image.open(io.BytesIO(sections[7][5:])) as pic:
        if pic.size != (nx, ny):
            raise ValueError('PNG/grid geometry mismatch')
        # The first crop loads the national PNG; second reuses the decoded image.
        for name, config in REGIONS.items():
            lat, lon = config['center']
            r0 = round((lat0-lat)/dy)-128
            c0 = round(((lon % 360)-lon0)/dx)-128
            if min(r0, c0) < 0 or r0+256 > ny or c0+256 > nx:
                raise ValueError('Crop beyond source grid')
            integers = np.asarray(pic.crop((c0, r0, c0+256, r0+256)))
            values = (reference+integers.astype(np.float32)*2.**binary)*10.**(-decimal)
            valid = np.isfinite(values) & (values >= 0)
            counts = valid.reshape(64, 4, 64, 4).sum((1, 3))
            sums = np.where(valid, values, 0).reshape(64, 4, 64, 4).sum((1, 3))
            pooled = np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0).astype(np.float32)
            points = []
            for slat, slon in config['stations'].values():
                r = round((lat0-slat)/dy)-r0
                c = round(((slon % 360)-lon0)/dx)-c0
                points.append(float(values[r, c]) if valid[r, c] else np.nan)
            result[name] = {'values': pooled, 'valid_fraction': counts.astype(np.float32)/16,
                            'latitude': lat0-(r0+np.arange(64)*4+1.5)*dy,
                            'longitude': (lon0+(c0+np.arange(64)*4+1.5)*dx+180) % 360-180,
                            'point': np.array(points, np.float32), 'valid_time': valid_time,
                            'native_grid_shape': [ny, nx], 'native_spacing_degrees': [dy, dx]}
    return result


def case_arrays(case):
    start = datetime.fromisoformat(case['window_start_utc'])
    rt = [start+timedelta(minutes=10*j) for j in range(29)]
    qt = [start.replace(hour=9, minute=0)+timedelta(hours=j) for j in range(6)]
    arrays = {}
    for name, config in REGIONS.items():
        arrays[name] = {'rate': np.full((29, 64, 64), np.nan, np.float32),
                        'valid': np.zeros((29, 64, 64), np.float32),
                        'quality': np.full((6, 64, 64), np.nan, np.float32),
                        'quality_valid': np.zeros((6, 64, 64), np.float32),
                        'point_rate': np.full((len(config['stations']), 29), np.nan, np.float32),
                        'point_quality': np.full((len(config['stations']), 6), np.nan, np.float32)}
    present, qpresent = np.zeros(29, bool), np.zeros(6, bool)
    sources, missing = [], []
    for product, stamps, rate_key, valid_key, point_key, flags in (
        ('PrecipRate', rt, 'rate', 'valid', 'point_rate', present),
        ('RadarQualityIndex', qt, 'quality', 'quality_valid', 'point_quality', qpresent)):
        for j, stamp in enumerate(stamps):
            day, ts = stamp.strftime('%Y%m%d'), stamp.strftime('%Y%m%d-%H%M%S')
            path = SOURCE / f'raw/{product}/{day}/MRMS_{product}_00.00_{ts}.grib2.gz'
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            payload = path.read_bytes()
            decoded = decode_regions(payload)
            for name, dec in decoded.items():
                if dec['valid_time'] != stamp:
                    raise ValueError('Source timestamp mismatch')
                a = arrays[name]
                a[rate_key][j], a[valid_key][j], a[point_key][:, j] = dec['values'], dec['valid_fraction'], dec['point']
                a['latitude'], a['longitude'] = dec['latitude'], dec['longitude']
            flags[j] = True
            sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(payload), 'bytes': len(payload)})
    hidx = np.array([(t.hour-9) for t in rt])
    assert all(qt[i] <= t for t, i in zip(rt, hidx))
    rows = []
    for name, a in arrays.items():
        quality = a['quality'][hidx]
        imask = (a['valid'][:9] >= .9) & np.isfinite(quality[:9]) & (quality[:9] >= .1)
        tmask = (a['valid'][9:] >= .9) & np.isfinite(quality[9:]) & (quality[9:] >= .5)
        tmask[:, :4] = False; tmask[:, -4:] = False
        tmask[:, :, :4] = False; tmask[:, :, -4:] = False
        inp = a['rate'][:9].copy(); iq = quality[:9].copy()
        inp[~imask] = np.nan; iq[~imask] = 0
        x = np.stack([inp, iq], axis=1)
        target = a['rate'][9:].copy(); target[~tmask] = np.nan
        pquality = a['point_quality'][:, hidx]
        pointmask = np.isfinite(a['point_rate'][:, 9:]) & np.isfinite(pquality[:, 9:]) & (pquality[:, 9:] >= .5)
        ptarget = a['point_rate'][:, 9:].copy(); ptarget[~pointmask] = np.nan
        complete = bool(present.all() and qpresent.all())
        eligible = complete and float(tmask.mean()) >= .1
        row = {'region': name, 'case_id': case['case_id'], 'winter': case['winter'], 'split': case['split'],
               'complete_rate_frames': bool(present.all()), 'complete_quality_frames': bool(qpresent.all()),
               'complete_sequence': complete, 'eligible': eligible,
               'input_valid_fraction': float(imask.mean()), 'target_valid_fraction': float(tmask.mean()),
               'target_interior_valid_fraction': float(tmask[:, 4:-4, 4:-4].mean())}
        for i, station in enumerate(REGIONS[name]['stations']):
            row[station+'_point_target_valid_fraction'] = float(pointmask[i].mean())
            row[station+'_point_all20_valid'] = bool(pointmask[i].all())
            row[station+'_point_1h_11to12_valid'] = bool(pointmask[i, 1:8].all())
            row[station+'_point_3h_11to14_valid'] = bool(pointmask[i, 1:20].all())
            row[station+'_forecast_1h_eligible'] = bool(eligible and pointmask[i, 1:8].all())
            row[station+'_forecast_3h_eligible'] = bool(eligible and pointmask[i, 1:20].all())
        folder = OUT/name/'cases'; folder.mkdir(parents=True, exist_ok=True)
        path = folder/(case['case_id']+'.npz')
        np.savez_compressed(path, x=x, y=target, input_mask=imask, target_mask=tmask,
                            rate_mm_hr=a['rate'], valid_fraction=a['valid'], rqi_hourly=a['quality'],
                            quality_valid_fraction=a['quality_valid'], point_rate_mm_hr=a['point_rate'],
                            point_rqi_hourly=a['point_quality'], point_target_mm_hr=ptarget, point_target_mask=pointmask,
                            point_ids=np.array(list(REGIONS[name]['stations'])), latitude=a['latitude'], longitude=a['longitude'],
                            frame_present=present, quality_present=qpresent,
                            radar_time=np.array([t.replace(tzinfo=None) for t in rt], dtype='datetime64[s]'),
                            quality_time=np.array([t.replace(tzinfo=None) for t in qt], dtype='datetime64[s]'),
                            forecast_issue_utc=case['forecast_issue_utc'], case_id=case['case_id'],
                            winter=case['winter'], split=case['split'], eligible=eligible)
        row['output_path'] = str(path.relative_to(ROOT)); row['output_bytes'] = path.stat().st_size
        rows.append(row)
    return {'case_id': case['case_id'], 'rows': rows, 'source_files': sources, 'missing_source_files': missing}


def main(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    schedule = SOURCE/'frozen_schedule.json'
    full = json.loads(schedule.read_text())['cases']
    cases = full if limit is None else full[:limit]
    source_hash = sha(Path(__file__).read_bytes())
    plan = {'source_schedule': str(schedule.relative_to(ROOT)), 'source_schedule_sha256': sha(schedule.read_bytes()),
            'source_script_sha256': source_hash, 'case_count': len(full), 'regions': REGIONS,
            'units': 'mm/hour liquid-equivalent precipitation; not snowfall depth',
            'cadence': '29 frames every10min09:20–14:00UTC;9 inputs ending10:40;20 targets10:50–14:00;issue10:50UTC',
            'quality': 'Input native-valid fraction>=0.9 and RQI>=0.1; target native-valid fraction>=0.9 and RQI>=0.5 plus4cell border exclusion. Hourly quality asof each timestamp; futurequalityneverinput.',
            'scope': 'Morning-only four-winter radar dataset; no other-mountain models or independent hourly fresh-snow observations',
            'missing_policy': 'Negative or absent rate/quality remainsNaN; no interpolated/imputed targets',
            'full_dataset_requested': len(cases) == len(full)}
    existing = OUT/'plan.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        assert previous['source_script_sha256'] == source_hash
    existing.write_text(json.dumps(plan, indent=2))
    started = time.monotonic()
    folder = OUT/'case_audits'; folder.mkdir(exist_ok=True)
    audits = []
    for n, case in enumerate(cases, 1):
        apath = folder/(case['case_id']+'.json')
        if apath.exists():
            audit = json.loads(apath.read_text())
            assert all((ROOT/row['output_path']).exists() for row in audit['rows'])
        else:
            audit = case_arrays(case)
            apath.write_text(json.dumps(audit, indent=2, allow_nan=False))
        audits.append(audit)
        if n % 10 == 0 or n == len(cases):
            status = {'processed_cases': n, 'requested_cases': len(cases), 'full_cohort_cases': len(full),
                      'elapsed_seconds': time.monotonic()-started, 'last_case': case['case_id']}
            (OUT/'progress.json').write_text(json.dumps(status, indent=2)); print(json.dumps(status), flush=True)
    frame = pd.DataFrame([row for audit in audits for row in audit['rows']])
    frame.to_csv(OUT/'case_inventory.csv', index=False)
    count_fields = ['complete_rate_frames', 'complete_quality_frames', 'complete_sequence', 'eligible']
    count_fields += [c for c in frame if c.endswith('_valid') or c.endswith('_eligible')]
    summary = []
    for (region, winter, split), sub in frame.groupby(['region', 'winter', 'split']):
        row = {'region': region, 'winter': int(winter), 'split': split, 'scheduled': len(sub)}
        applicable = [c for c in count_fields if not sub[c].isna().all()]
        row.update({c: int(sub[c].fillna(False).sum()) for c in applicable})
        row['mean_target_valid_fraction'] = float(sub.target_valid_fraction.mean())
        row['mean_target_interior_valid_fraction'] = float(sub.target_interior_valid_fraction.mean())
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'winter_quality_summary.csv', index=False)
    final = {**plan, 'finished_cases': len(cases), 'complete_full_cohort': len(cases) == len(full),
             'elapsed_seconds_this_invocation': time.monotonic()-started,
             'output_npz_bytes': int(frame.output_bytes.sum()), 'winter_summary': summary,
             'missing_source_files': sorted(set(p for a in audits for p in a['missing_source_files'])),
             'source_field_files_used': sum(len(a['source_files']) for a in audits)}
    (OUT/'audit.json').write_text(json.dumps(final, indent=2))
    print(json.dumps({'complete': final['complete_full_cohort'], 'elapsed_seconds': final['elapsed_seconds_this_invocation'],
                      'output_npz_bytes': final['output_npz_bytes'], 'missing_source_files': len(final['missing_source_files'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(); main(args.limit)
