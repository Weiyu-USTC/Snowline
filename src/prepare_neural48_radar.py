"""Prepare issue-known radar inputs for a weather-conditioned 48-hour model.

This adapter reads cached NOAA MRMS observations only. Future precipitation and
quality fields in the source containers never enter any exported predictor.
Existing source files, model inputs and checkpoints remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALTA = ROOT / 'data/alta_nowcast/radar'
TRANSFER = ROOT / 'outputs/radar-transfer/prepared'
OUT = ROOT / 'data/neural48_training/radar'
REGIONS = {
    'alta': {'alta': (40.57607, -111.63889)},
    'vail': {'vail': (39.6176, -106.38)},
    'vermont': {'mansfield': (44.5247, -72.8153), 'jay': (44.9381, -72.5022)},
}
CHANNELS = ['log1p_quality_screened_rate_mm_h', 'mean_raw_rqi',
            'qualified_input_fraction', 'mean_native_rate_valid_fraction',
            'quality_available_fraction']
STATS = ['local_qualified_rate_mean_mm_h', 'local_qualified_rate_max_mm_h',
         'local_qualified_wet_fraction_gt_0_1_mm_h', 'local_mean_raw_rqi',
         'local_qualified_input_fraction', 'local_mean_native_rate_valid_fraction',
         'native_point_qualified_rate_mm_h', 'native_point_rqi']


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8_388_608), b''):
            h.update(chunk)
    return h.hexdigest()


def mean_finite(array, axes):
    valid = np.isfinite(array)
    count = valid.sum(axis=axes)
    total = np.where(valid, array, 0).sum(axis=axes)
    return np.divide(total, count, out=np.full(total.shape, np.nan, dtype=np.float64), where=count > 0)


def pool_mean(array, factor=4):
    """Pool only finite cells, preserving all-missing pools as NaN."""
    a = np.asarray(array, dtype=np.float32)
    if a.ndim != 3 or a.shape[-2:] != (64, 64) or 64 % factor:
        raise ValueError('Expected time × 64 × 64 source maps')
    size = 64 // factor
    return mean_finite(a.reshape(len(a), size, factor, size, factor), (2, 4)).astype(np.float32)


def select_context(rate, valid, quality, radar_times, quality_times, issue,
                   frame_present, quality_present, point_rate, point_quality):
    """Strict causal selection; source arrays may contain later verification data."""
    rt = np.asarray(radar_times, dtype='datetime64[s]')
    qt = np.asarray(quality_times, dtype='datetime64[s]')
    issued = np.datetime64(pd.Timestamp(issue).tz_convert('UTC').tz_localize(None), 's')
    if len(rt) != len(rate) or len(qt) != len(quality):
        raise ValueError('Timestamp and field counts differ')
    if np.any(np.diff(rt) <= np.timedelta64(0, 's')) or np.any(np.diff(qt) <= np.timedelta64(0, 's')):
        raise ValueError('Source timestamps must be unique and increasing')
    ix = np.flatnonzero((rt >= issued - np.timedelta64(90, 'm')) &
                        (rt <= issued - np.timedelta64(10, 'm')))
    if len(ix) != 9 or not np.all(np.diff(rt[ix]) == np.timedelta64(10, 'm')):
        raise ValueError('Expected nine context frames from issue−90 to issue−10 minutes')
    qix = np.searchsorted(qt, rt[ix], side='right') - 1
    if (qix < 0).any() or np.any(qt[qix] > rt[ix]):
        raise ValueError('No causal radar quality field for a context frame')
    if np.any(rt[ix] + np.timedelta64(10, 'm') > issued):
        raise ValueError('Radar latency assumption violated')
    return {
        'rate': np.asarray(rate)[ix].astype(np.float32),
        'valid': np.asarray(valid)[ix].astype(np.float32),
        'quality': np.asarray(quality)[qix].astype(np.float32),
        'radar_time': rt[ix], 'quality_time': qt[qix],
        'frame_present': np.asarray(frame_present, bool)[ix],
        'quality_present': np.asarray(quality_present, bool)[qix],
        'point_rate': np.asarray(point_rate, np.float32)[:, ix],
        'point_quality': np.asarray(point_quality, np.float32)[:, qix],
    }


def encode_context(context, latitude, longitude, sites):
    """Finite CNN encoding with explicit masks plus unfilled physical arrays."""
    rate, valid, quality = (context[k] for k in ['rate', 'valid', 'quality'])
    if rate.shape != (9, 64, 64) or valid.shape != rate.shape or quality.shape != rate.shape:
        raise ValueError('Context geometry differs from nine 64×64 frames')
    native_valid = np.isfinite(rate) & (rate >= 0) & np.isfinite(valid)
    quality_valid = np.isfinite(quality) & (quality >= 0)
    qualified = native_valid & (valid >= .9) & quality_valid & (quality >= .1)
    screened = np.where(qualified, rate, np.nan)
    raw_quality = np.where(quality_valid, quality, np.nan)
    raw_valid = np.where(np.isfinite(valid), np.clip(valid, 0, 1), 0)
    pooled_rate = pool_mean(screened)
    pooled_rqi = pool_mean(raw_quality)
    qualified_fraction = pool_mean(qualified.astype(np.float32))
    source_valid_fraction = pool_mean(raw_valid)
    quality_fraction = pool_mean(quality_valid.astype(np.float32))
    cnn = np.stack([
        np.nan_to_num(np.log1p(pooled_rate), nan=0.),
        np.nan_to_num(pooled_rqi, nan=0.),
        qualified_fraction, source_valid_fraction, quality_fraction,
    ], axis=1).astype(np.float32)
    if not np.isfinite(cnn).all():
        raise ValueError('CNN predictors must be finite with explicit missingness channels')
    local, positions = [], []
    for point, (name, (slat, slon)) in enumerate(sites.items()):
        iy, ix = int(np.argmin(abs(latitude - slat))), int(np.argmin(abs(longitude - slon)))
        if iy < 1 or ix < 1 or iy >= 63 or ix >= 63:
            raise ValueError('Local neighborhood extends outside cached source crop')
        sl = np.s_[:, iy-1:iy+2, ix-1:ix+2]
        lr, lq, lv, lm = screened[sl], raw_quality[sl], raw_valid[sl], qualified[sl]
        count = np.isfinite(lr).sum((1, 2))
        maximum = np.max(np.where(np.isfinite(lr), lr, -np.inf), axis=(1, 2))
        maximum[count == 0] = np.nan
        wet = np.divide(((lr > .1) & np.isfinite(lr)).sum((1, 2)), count,
                        out=np.full(9, np.nan), where=count > 0)
        pr, pq = context['point_rate'][point], context['point_quality'][point]
        pq = np.where(np.isfinite(pq) & (pq >= 0), pq, np.nan)
        pr = np.where(np.isfinite(pr) & (pr >= 0) & np.isfinite(pq) & (pq >= .1), pr, np.nan)
        stats = np.stack([mean_finite(lr, (1, 2)), maximum, wet,
                          mean_finite(lq, (1, 2)), lm.mean((1, 2)), lv.mean((1, 2)), pr, pq], axis=-1)
        local.append(stats.astype(np.float32))
        positions.append([iy, ix])
    return {'x': cnn, 'pooled_rate_mm_h': pooled_rate, 'pooled_rqi': pooled_rqi,
            'local_stats': np.stack(local), 'local_stats_missing': ~np.isfinite(local),
            'site_grid64_indices': np.array(positions, np.int16),
            'input_qualified_fraction': float(qualified.mean()),
            'local_input_qualified_fraction': np.asarray([a[:, 4].mean() for a in local]),
            'local_mean_rqi': np.asarray([mean_finite(a[:, 3], 0) for a in local])}


def alta_sources():
    path = ALTA / 'radar_sequences.npz'
    z = np.load(path, allow_pickle=False)
    # Decompress the physical arrays once; only nine radar frames and causal
    # quality fields are subsequently admitted by select_context.
    keys = ['rate', 'valid_fraction', 'rqi_hourly', 'rate_alta_native', 'rqi_alta_native',
            'frame_present', 'quality_present', 'radar_time', 'quality_time',
            'case_id', 'latitude', 'longitude']
    arrays = {k: z[k] for k in keys}
    z.close()
    return arrays, {'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)}


def main(output=OUT):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    schedule_path = ALTA / 'frozen_schedule.json'
    schedule = json.loads(schedule_path.read_text())['cases']
    all_rows, products = [], {}
    alta, alta_provenance = alta_sources()
    alta_lookup = {str(c): i for i, c in enumerate(alta['case_id'])}
    for region, sites in REGIONS.items():
        records, sources, rows = [], [], []
        for case in schedule:
            if region == 'alta':
                i = alta_lookup[case['case_id']]
                rate, valid, quality = alta['rate'][i], alta['valid_fraction'][i], alta['rqi_hourly'][i]
                rt, qt = alta['radar_time'][i], alta['quality_time'][i]
                present, qpresent = alta['frame_present'][i], alta['quality_present'][i]
                pr, pq = alta['rate_alta_native'][i][None], alta['rqi_alta_native'][i][None]
                lat, lon = alta['latitude'], alta['longitude']
            else:
                path = TRANSFER / region / 'cases' / (case['case_id'] + '.npz')
                with np.load(path, allow_pickle=False) as z:
                    rate, valid, quality = z['rate_mm_hr'], z['valid_fraction'], z['rqi_hourly']
                    rt, qt = z['radar_time'], z['quality_time']
                    present, qpresent = z['frame_present'], z['quality_present']
                    pr, pq = z['point_rate_mm_hr'], z['point_rqi_hourly']
                    lat, lon = z['latitude'], z['longitude']
                    original_site_ids = ['mansfield' if s == 'stowe' else str(s) for s in z['point_ids']]
                    if original_site_ids != list(sites):
                        raise ValueError('Native point order does not match named stations')
                    if str(z['forecast_issue_utc']) != case['forecast_issue_utc']:
                        raise ValueError('Transfer forecast issue disagrees with frozen schedule')
                sources.append({'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)})
            context = select_context(rate, valid, quality, rt, qt, case['forecast_issue_utc'],
                                     present, qpresent, pr, pq)
            encoded = encode_context(context, lat, lon, sites)
            complete = bool(context['frame_present'].all() and context['quality_present'].all())
            row = {'region': region, 'case_id': case['case_id'], 'date': case['date'],
                   'winter': case['winter'], 'original_split': case['split'],
                   'forecast_issue_utc': case['forecast_issue_utc'], 'input_complete': complete,
                   'input_qualified_fraction': encoded['input_qualified_fraction'],
                   'input_usable_at_10pct': complete and encoded['input_qualified_fraction'] >= .1}
            for s, name in enumerate(sites):
                row[name + '_local_qualified_fraction'] = float(encoded['local_input_qualified_fraction'][s])
                value = float(encoded['local_mean_rqi'][s])
                row[name + '_local_mean_rqi'] = value if np.isfinite(value) else None
            rows.append(row); records.append({**encoded, **{
                k: context[k] for k in ['radar_time', 'quality_time', 'frame_present', 'quality_present']}})
        arrays = {key: np.stack([r[key] for r in records]) for key in [
            'x', 'pooled_rate_mm_h', 'pooled_rqi', 'local_stats', 'local_stats_missing',
            'radar_time', 'quality_time', 'frame_present', 'quality_present']}
        arrays.update(case_id=np.array([c['case_id'] for c in schedule]),
            date=np.array([c['date'] for c in schedule]), winter=np.array([c['winter'] for c in schedule]),
            original_split=np.array([c['split'] for c in schedule]),
            forecast_issue_utc=np.array([c['forecast_issue_utc'] for c in schedule]),
            input_complete=np.array([r['input_complete'] for r in rows]),
            input_qualified_fraction=np.array([r['input_qualified_fraction'] for r in rows], np.float32),
            input_usable_at_10pct=np.array([r['input_usable_at_10pct'] for r in rows]),
            channel_names=np.array(CHANNELS), local_stat_names=np.array(STATS), site_ids=np.array(list(sites)),
            site_grid64_indices=records[0]['site_grid64_indices'],
            latitude=lat.reshape(16, 4).mean(1), longitude=lon.reshape(16, 4).mean(1))
        path = output / (region + '_contexts.npz')
        np.savez_compressed(path, **arrays)
        if region == 'alta': sources = [alta_provenance]
        (output / (region + '_source_files.json')).write_text(json.dumps(sources, indent=2))
        products[region] = {'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path),
            'shape': list(arrays['x'].shape), 'input_complete': int(arrays['input_complete'].sum()),
            'input_usable_at_10pct': int(arrays['input_usable_at_10pct'].sum()),
            'site_ids': list(sites), 'source_files': len(sources), 'bytes': path.stat().st_size}
        all_rows.extend(rows)
        print(json.dumps({'region': region, **products[region]}), flush=True)
    frame = pd.DataFrame(all_rows); frame.to_csv(output / 'case_inventory.csv', index=False)
    summary = frame.groupby(['region', 'winter']).agg(scheduled=('case_id', 'size'),
        complete_contexts=('input_complete', 'sum'), usable_at_10pct=('input_usable_at_10pct', 'sum'),
        mean_qualified_fraction=('input_qualified_fraction', 'mean')).reset_index()
    summary.to_csv(output / 'winter_summary.csv', index=False)
    audit = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_schedule': str(schedule_path.relative_to(ROOT)), 'source_schedule_sha256': file_hash(schedule_path),
        'source_code_sha256': file_hash(Path(__file__)), 'regions': products, 'channel_names': CHANNELS,
        'local_stat_names': STATS, 'raw_units': 'MRMS liquid-equivalent precipitation rate, mm/hour, not snow depth',
        'inputs_only': True, 'future_target_fields_exported': False,
        'input_window': 'Nine ten-minute frames 09:20–10:40 UTC; issue 10:50 UTC',
        'availability_assumption': 'Radar available ten minutes after observation; original receipt latency is not archived',
        'quality': 'Rate requires native valid fraction >=0.9 and causal RQI >=0.1; raw low RQI retained in quality channels',
        'missing': 'All-missing pooled physical rates remain NaN. CNN rate is zero-filled only with explicit qualified-fraction and source-availability channels. Never infer dry weather from missing rates.',
        'pooling': 'Cached64×64 maps pooled in4×4 blocks to16×16; each CNN pixel represents16×16 native0.01-degree cells (~0.16 degrees)',
        'local': 'Site-specific3×3 neighborhoods of cached64×64 maps plus native-point rate/RQI; local stat NaNs retained with explicit mask',
        'split_note': 'Original split labels retained as provenance only. Root benchmark must assign whole-winter splits. Stowe and Jay share one Vermont sequence and date and must stay together.',
        'eligibility': 'No target-based exclusion. All725 scheduled origins retained. input_complete and >=10% input qualified coverage are diagnostics, not automatically imposed filters.',
        'existing_inputs_modified': False}
    (output / 'manifest.json').write_text(json.dumps(audit, indent=2))
    (output / 'README.txt').write_text('''RADAR PREDICTORS FOR THE 48-HOUR WEATHER-CONDITIONED MODEL

Each REGION_contexts.npz contains all725 scheduled morning origins from
2020/21–2023/24. Load with numpy.load(..., allow_pickle=False).
x is [origin,9time,5channel,16row,16column], finite float32. Channel names are
stored in channel_names. The first channel is log1p(mm/hour); masks distinguish
missing from genuinely dry. pooled_rate_mm_h retains NaN physical rates.
local_stats is [origin,site,9time,8stat], with matching local_stats_missing.
case_id and forecast_issue_utc join to the NWP and observed snow targets.
Stowe uses site_id mansfield, and shares the Vermont regional array with Jay.
No future verification fields or snow targets are included. Do not use the
original split labels as a substitute for a newly frozen evaluation protocol.
All observations precede issuance, under an explicit10-minute latency assumption.
Inputs contain radar-estimated liquid-equivalent rates, not hourly fresh snow.
Vail's poor local radar quality is preserved; it is not hidden by zero filling.
Source manifests contain input-container hashes; original files are unchanged.
''')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=OUT)
    main(parser.parse_args().output)
