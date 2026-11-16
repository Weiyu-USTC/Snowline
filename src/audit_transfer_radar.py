"""Verify prepared transfer data and summarize regional/point availability."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_transfer_radar import REGIONS, ROOT, OUT


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    prep = json.loads((OUT/'audit.json').read_text())
    if not prep['complete_full_cohort']:
        raise RuntimeError('Wait for full preparation before auditing')
    plan = json.loads((OUT/'plan.json').read_text())
    schedule = json.loads((ROOT/plan['source_schedule']).read_text())['cases']
    assert len(schedule) == 725 == prep['finished_cases']
    source_rows = {}
    source_manifest_complete = True
    for case in schedule:
        audit = json.loads((OUT/'case_audits'/(case['case_id']+'.json')).read_text())
        assert audit['case_id'] == case['case_id']
        assert len(audit['source_files'])+len(audit['missing_source_files']) == 35
        for row in audit['source_files']:
            source_manifest_complete &= len(row.get('sha256', '')) == 64 and row['bytes'] > 0
            if row['path'] in source_rows:
                assert source_rows[row['path']] == row
            source_rows[row['path']] = row
    mismatches = []
    for name, source in source_rows.items():
        path = ROOT/name
        if path.stat().st_size != source['bytes'] or file_sha(path) != source['sha256']:
            mismatches.append(name)
    assert source_manifest_complete and not mismatches
    # Independently recover native point centers from the original audited
    # decoder, rather than treating the 4x4 pooled grid as station coordinates.
    import fetch_alta_radar as original_decoder
    first_rate = next(name for name in source_rows if '/PrecipRate/' in name)
    sample = (ROOT/first_rate).read_bytes()
    point_coordinates = {}
    for region, config in REGIONS.items():
        original_decoder.REGION_LAT, original_decoder.REGION_LON = config['center']
        for station, location in config['stations'].items():
            original_decoder.LAT, original_decoder.LON = location
            _, _, _, _, meta = original_decoder.decode_crop(sample, 256, 4)
            point_coordinates[station] = {
                'target_latitude': location[0], 'target_longitude': location[1],
                'native_pixel_latitude': meta['station_native_latitude'],
                'native_pixel_longitude': meta['station_native_longitude'],
            }
    records, maps = [], {}
    for region, config in REGIONS.items():
        full, one, total = np.zeros((64, 64), np.int32), np.zeros((64, 64), np.int32), 0
        reference_coordinates = None
        for case in schedule:
            path = OUT/region/'cases'/(case['case_id']+'.npz')
            with np.load(path) as z:
                x, y = z['x'], z['y']
                im, tm = z['input_mask'], z['target_mask']
                assert x.shape == (9, 2, 64, 64) and y.shape == (20, 64, 64)
                assert np.array_equal(np.isfinite(x[:, 0]), im)
                assert np.array_equal(np.isfinite(y), tm)
                assert not tm[:, :4].any() and not tm[:, -4:].any()
                assert not tm[:, :, :4].any() and not tm[:, :, -4:].any()
                assert str(z['case_id']) == case['case_id'] and str(z['split']) == case['split']
                assert z['radar_time'][8] < z['radar_time'][9]
                assert z['quality_time'][1] <= z['radar_time'][8] < z['quality_time'][2]
                if reference_coordinates is None:
                    reference_coordinates = (z['latitude'], z['longitude'])
                np.testing.assert_array_equal(z['latitude'], reference_coordinates[0])
                np.testing.assert_array_equal(z['longitude'], reference_coordinates[1])
                inp_complete = bool(z['frame_present'][:9].all() and z['quality_present'][:2].all())
                seq_complete = bool(z['frame_present'].all() and z['quality_present'].all())
                expected_eligible = seq_complete and tm.mean() >= .1
                assert bool(z['eligible']) == expected_eligible
                row = {'region': region, 'winter': int(z['winter']), 'split': str(z['split']), 'case_id': str(z['case_id']),
                       'input_complete': inp_complete,
                       'input_complete_any_usable_pixels': bool(inp_complete and im.any()),
                       'input_complete_coverage_ge_10pct': bool(inp_complete and im.mean() >= .1),
                       'input_valid_fraction': float(im.mean()),
                       'sequence_complete': seq_complete, 'original_protocol_eligible': expected_eligible,
                       'target_valid_fraction': float(tm.mean()), 'case_npz_bytes': path.stat().st_size}
                pq = z['point_target_mask']; pt = z['point_target_mm_hr']
                assert np.array_equal(np.isfinite(pt), pq)
                hourly_index = ((z['radar_time']-z['quality_time'][0])/np.timedelta64(1, 'h')).astype(int)
                point_q = z['point_rqi_hourly'][:, hourly_index[9:]]
                expected_pq = np.isfinite(z['point_rate_mm_hr'][:, 9:]) & np.isfinite(point_q) & (point_q >= .5)
                np.testing.assert_array_equal(pq, expected_pq)
                for i, station in enumerate(z['point_ids'].astype(str)):
                    row[station+'_truth_all20_usable'] = bool(pq[i].all())
                    row[station+'_truth_1h_usable'] = bool(pq[i, 1:8].all())
                    row[station+'_truth_3h_usable'] = bool(pq[i, 1:20].all())
                    row[station+'_forecast_1h_eligible'] = bool(expected_eligible and pq[i, 1:8].all())
                    row[station+'_forecast_3h_eligible'] = bool(expected_eligible and pq[i, 1:20].all())
                records.append(row)
                full += tm.all(axis=0); one += tm[1:8].all(axis=0); total += 1
                lat, lon = z['latitude'], z['longitude']
        maps[region] = {'all20_percent': full/total*100, 'one_hour_percent': one/total*100,
                        'latitude': lat, 'longitude': lon}
        np.savez_compressed(OUT/region/'availability_maps.npz', **maps[region], scheduled_cases=total)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'verified_case_inventory.csv', index=False)
    summary = []
    count_keys = ['input_complete', 'input_complete_any_usable_pixels', 'input_complete_coverage_ge_10pct',
                  'sequence_complete', 'original_protocol_eligible']
    count_keys += [k for k in frame if '_truth_' in k or '_forecast_' in k]
    for (region, winter, split), sub in frame.groupby(['region', 'winter', 'split']):
        row = {'region': region, 'winter': int(winter), 'split': split, 'scheduled_cases': len(sub)}
        row.update({k: int(sub[k].sum()) for k in count_keys if not sub[k].isna().all()})
        row['mean_input_valid_fraction'] = float(sub.input_valid_fraction.mean())
        row['mean_target_valid_fraction'] = float(sub.target_valid_fraction.mean())
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'verified_winter_summary.csv', index=False)
    result = {'as_of_date': '2026-11-07', 'status': 'verified_complete', 'scheduled_cases_per_region': 725,
              'regions': REGIONS, 'native_point_coordinates': point_coordinates,
              'case_files': len(records), 'case_npz_bytes': int(frame.case_npz_bytes.sum()),
              'consumed_unique_source_files': len(source_rows), 'source_files_have_hashes': source_manifest_complete,
              'source_files_rehashed_and_matched': len(source_rows)-len(mismatches), 'source_hash_mismatches': mismatches,
              'missing_source_files': prep['missing_source_files'],
              'source_bytes_read': sum(v['bytes'] for v in source_rows.values()),
              'winter_summary': summary,
              'input_completeness_rule': 'All 9 rate input frames and both causal 09/10 UTC quality frames present',
              'input_coverage_ge_10pct_rule': 'Additional descriptive count, not a change to original eligibility: complete inputs and at least 10% of 9x64x64 input pixels pass mask',
              'eligibility_rule': 'Original Alta: all 29 rate + 6 quality frames present and at least 10% of 20x64x64 target pixels pass mask',
              'one_hour_point_rule': 'All 7 native point target rates 11:00–12:00 UTC valid and native point RQI >= 0.5; forecast eligibility additionally requires original regional eligibility',
              'map_rule': 'Percent of all 725 scheduled cases with all 20 target frames passing masks at cell; missing target data count as unavailable. Input completeness is reported separately.',
              'target_units': 'liquid-equivalent radar precipitation mm/hour, not independent surface snowfall',
              'scope': 'Morning only 09:20–14:00 UTC; no HRRR features added or models trained here'}
    (OUT/'verification_audit.json').write_text(json.dumps(result, indent=2))
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white'})
    for key, filename, subtitle in (
        ('all20_percent', 'target_availability.png', 'All 20 target frames 10:50–14:00 UTC usable'),
        ('one_hour_percent', 'one_hour_target_availability.png', 'All 7 target frames 11:00–12:00 UTC usable')):
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.3), constrained_layout=True)
        for ax, (region, config) in zip(axs, REGIONS.items()):
            m = maps[region]
            mesh = ax.pcolormesh(m['longitude'], m['latitude'], m[key], shading='nearest', cmap='viridis', vmin=0, vmax=100)
            for station, (slat, slon) in config['stations'].items():
                ax.plot(slon, slat, '*', color='white', markeredgecolor='#152f35', markersize=14)
                ax.annotate({'vail': 'Vail', 'stowe': 'Stowe', 'jay': 'Jay Peak'}[station], (slon, slat),
                            xytext=(7, 6), textcoords='offset points', color='black', fontweight='bold',
                            bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none', 'pad': 2})
            ax.set_title('Vail region' if region == 'vail' else 'Shared northern Vermont region')
            ax.set_xlabel('Longitude (degrees)'); ax.set_ylabel('Latitude (degrees)')
            ax.set_aspect(1/np.cos(np.deg2rad(config['center'][0])))
        fig.colorbar(mesh, ax=axs, label='Scheduled morning cases with usable radar targets (%)', shrink=.82)
        fig.suptitle('Radar target availability across 725 scheduled mornings, winters 2020/21–2023/24\n'+subtitle, fontsize=13)
        fig.get_layout_engine().set(rect=(0, .09, 1, .91))
        fig.text(.5, .025, 'Same Alta quality rules: valid fraction ≥ 0.9, RQI ≥ 0.5; four-cell border excluded. Missing target data count as unavailable.\nThis measures radar availability, not fresh-snow accuracy; markers show target station locations.', ha='center', fontsize=8)
        fig.savefig(OUT/filename, dpi=180); plt.close(fig)
    result['directory_total_bytes_at_verification'] = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    (OUT/'verification_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': 'verified_complete', 'case_npz_bytes': result['case_npz_bytes'],
                      'directory_total_bytes_at_verification': result['directory_total_bytes_at_verification'],
                      'source_files_rehashed_and_matched': result['source_files_rehashed_and_matched'],
                      'winter_summary': summary}, indent=2))


if __name__ == '__main__':
    main()
