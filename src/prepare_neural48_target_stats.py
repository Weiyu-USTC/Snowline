"""Issue-known local radar covariates at the modern Vermont snow stations.

These statistics use the cached 64×64 pooled regional grid, not new native MRMS
point extraction. The source mountain-centered statistics remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prepare_neural48_radar import (ROOT, ALTA, TRANSFER, OUT, STATS, file_hash,
                                    select_context, encode_context)

SITES = {'mansfield': (44.464, -72.6907), 'jay': (44.8916, -72.4435)}
TARGET_NAMES = {'mansfield': 'Stowe 0.2 SW daily observation station',
                'jay': 'Westfield 0.7 WNW daily observation station'}
TARGET_STATS = STATS[:6] + ['target_grid_cell_qualified_rate_mm_h', 'target_grid_cell_rqi']


def main():
    schedule = json.loads((ALTA / 'frozen_schedule.json').read_text())['cases']
    rows, missing, times, qtimes, source_records, points = [], [], [], [], [], []
    for case in schedule:
        path = TRANSFER / 'vermont/cases' / (case['case_id'] + '.npz')
        with np.load(path, allow_pickle=False) as z:
            rate, valid, quality = z['rate_mm_hr'], z['valid_fraction'], z['rqi_hourly']
            rt, qt, lat, lon = z['radar_time'], z['quality_time'], z['latitude'], z['longitude']
            indices = [(int(np.argmin(abs(lat - slat))), int(np.argmin(abs(lon - slon))))
                       for slat, slon in SITES.values()]
            point_rate = np.stack([np.where(valid[:, iy, ix] >= .9, rate[:, iy, ix], np.nan)
                                   for iy, ix in indices])
            point_quality = np.stack([quality[:, iy, ix] for iy, ix in indices])
            context = select_context(rate, valid, quality, rt, qt, case['forecast_issue_utc'],
                                     z['frame_present'], z['quality_present'], point_rate, point_quality)
        encoded = encode_context(context, lat, lon, SITES)
        rows.append(encoded['local_stats']); missing.append(encoded['local_stats_missing'])
        times.append(context['radar_time']); qtimes.append(context['quality_time'])
        source_records.append({'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)})
        if not points:
            for (site, coordinates), (iy, ix) in zip(SITES.items(), indices):
                points.append({'site_id': site, 'target_name': TARGET_NAMES[site],
                    'station_latitude': coordinates[0], 'station_longitude': coordinates[1],
                    'grid64_row': iy, 'grid64_column': ix,
                    'grid64_latitude': float(lat[iy]), 'grid64_longitude': float(lon[ix])})
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'vermont_target_local_stats.npz'
    np.savez_compressed(target, local_stats=np.stack(rows), local_stats_missing=np.stack(missing),
        case_id=np.array([c['case_id'] for c in schedule]),
        forecast_issue_utc=np.array([c['forecast_issue_utc'] for c in schedule]),
        radar_time=np.stack(times), quality_time=np.stack(qtimes),
        site_ids=np.array(list(SITES)), site_latitude=np.array([c[0] for c in SITES.values()]),
        site_longitude=np.array([c[1] for c in SITES.values()]), local_stat_names=np.array(TARGET_STATS),
        grid64_indices=np.array([[p['grid64_row'], p['grid64_column']] for p in points]),
        grid64_latitude=np.array([p['grid64_latitude'] for p in points]),
        grid64_longitude=np.array([p['grid64_longitude'] for p in points]))
    (OUT / 'vermont_target_local_stats.sources.json').write_text(json.dumps(source_records, indent=2))
    audit = {'status': 'complete', 'output_path': str(target.relative_to(ROOT)),
        'output_sha256': file_hash(target), 'source_code_sha256': file_hash(Path(__file__)),
        'shared_helper_sha256': file_hash(ROOT / 'src/prepare_neural48_radar.py'),
        'shape': [len(schedule), 2, 9, 8], 'sites': points, 'local_stat_names': TARGET_STATS,
        'spatial_resolution': '0.04-degree pooled radar grid; local neighborhood is3×3 grid cells',
        'point_semantics': 'Last two statistics are pooled64-grid target-cell values, not native1km MRMS extraction',
        'units': 'Radar-estimated liquid-equivalent precipitation mm/hour, not observed snowfall',
        'future_inputs': False, 'input_window': '09:20–10:40UTC; causal09/10UTCquality only; forecast issue10:50UTC',
        'target_identity': 'Modern lower-elevation Stowe-town/Westfield observations; not Mount Mansfield/Jay resort summit snowfall',
        'original_mountain_covariates_unchanged': True, 'source_files': len(source_records)}
    (OUT / 'vermont_target_local_stats.audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__': main()
