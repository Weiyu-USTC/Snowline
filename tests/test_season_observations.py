"""Season-chart observations: missing values, provenance and safe cache writes."""
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import season_observations as obs


def noaa(*rows, station='US1VTLM0001'):
    return gzip.compress(('\n'.join(station + ',' + row for row in rows)).encode())


def alta(rows):
    data = dict(weather=dict(snowfall=dict(history=rows)))
    return ('<script>window.Alta = ' + json.dumps(data) + ';</script>').encode()


def alta_history(code, totals=(5, 8)):
    start = int(str(code)[:4])
    return json.dumps(dict(snowfallHistory=[dict(reported_at=f'{start}-10-{day + 11:02d}',
        season=int(code), ytd_snow=total, new_snow=None) for day, total in enumerate(totals)])).encode()


class SeasonObservationTests(unittest.TestCase):
    def test_noaa_new_snow_units_missing_flags_and_trace(self):
        rows = obs.parse_noaa(noaa('20261001,SNOW,254,,,N,0730',
            '20261002,SNOW,0,P,,N,0730', '20261003,SNOW,0,T,,N,0730',
            '20261004,SNOW,20,A,,N,0730', '20261005,SNOW,30,,X,N,0730'), 'US1VTLM0001')
        self.assertEqual([r['snowfall_inches'] for r in rows], [10, None, 0, None, None])
        self.assertTrue(rows[2]['trace'])
        self.assertEqual(rows[0]['end'], '2026-10-01T11:30:00+00:00')

    def test_unknown_clock_is_not_replaced_with_nominal(self):
        rows = obs.parse_noaa(noaa('20261101,SNOW,0,,,N,', '20261103,SNOW,0,,,N,2500'), 'US1VTLM0001')
        self.assertEqual([r['end'] for r in rows], [None, None])
        self.assertEqual(len(rows), 2)  # no invented November 2 report

    def test_station_mismatch_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            obs.parse_noaa(noaa('20261001,SNOW,0,,,N,0730'), 'US1VTOL0001')
        with self.assertRaises(ValueError):
            obs.parse_noaa(noaa('20261001,SNOW,0,,,N,0730', '20261001,SNOW,0,,,N,0730'), 'US1VTLM0001')

    def test_alta_direct_totals_are_separate_and_clock_unknown(self):
        body = alta([dict(reported_at='2026-10-11', season=20262027, new_snow=2, ytd_snow=5),
                     dict(reported_at='2026-10-13', season=20262027, new_snow=3, ytd_snow=8)])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = obs._cache(root) / 'alta_weather.html'
            path.parent.mkdir(parents=True); path.write_bytes(body)
            with patch.object(obs, '_download', side_effect=AssertionError('GET must be offline')):
                result = obs.read_observations('alta', root)
            self.assertEqual(result['records'], [])
            self.assertEqual([r['total_inches'] for r in result['reported_totals']], [5, 8])
            self.assertTrue(all(r['end'] is None for r in result['reported_totals']))
            self.assertIsNone(result['reported_total_source']['station_id'])

    def test_release_combined_targets_fallback_filters_site(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = root / 'data/neural48_training/targets/daily_targets.csv'
            path.parent.mkdir(parents=True)
            path.write_text('site,reporting_date,snowfall_inches,valid_snowfall,period_end_utc\n'
                            'alta,2024-01-01,3,True,2024-01-01T11:00:00Z\n'
                            'vail,2024-01-01,9,True,2024-01-01T11:00:00Z\n')
            self.assertEqual(obs.read_observations('alta', root)['records'][0]['snowfall_inches'], 3)
            vail = obs.read_observations('vail', root)
            self.assertIsNone(vail['records'][0]['end'])
            self.assertFalse(vail['target']['verified_fresh_snow'])

    def test_new_noaa_cache_invalidates_old_value_and_preserves_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); target = root / 'data/neural48_training/targets/mansfield_daily.csv'
            target.parent.mkdir(parents=True)
            target.write_text('reporting_date,snowfall_inches,valid_snowfall,period_end_utc\n'
                              '2026-10-01,9,True,2026-10-01T11:30:00Z\n')
            path = obs._cache(root) / 'US1VTLM0001.csv.gz'; path.parent.mkdir(parents=True)
            path.write_bytes(noaa('20261001,SNOW,0,P,,N,0730', '20261003,SNOW,254,,,N,0730'))
            result = obs.read_observations('mansfield', root)
            self.assertEqual([r['snowfall_inches'] for r in result['records']], [None, 10])
            self.assertEqual(result['latest_observation_date'], '2026-10-03')

    def test_vail_quarantine_does_not_restore_bad_historical_values(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); target = root / 'data/neural48_training/targets/vail_daily.csv'
            target.parent.mkdir(parents=True)
            target.write_text('reporting_date,snowfall_inches,valid_snowfall,period_end_utc\n'
                              '2023-11-20,9,True,2023-11-20T11:00:00Z\n')
            self.assertIsNone(obs.read_observations('vail', root)['records'][0]['snowfall_inches'])

    def test_refresh_failure_preserves_prior_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = obs._cache(root) / 'alta_weather.html'; path.parent.mkdir(parents=True)
            previous = alta([dict(reported_at='2026-11-01', ytd_snow=8)])
            path.write_bytes(previous)
            with patch.object(obs, '_download', return_value=b'not the public data'):
                result = obs.refresh_observations(root, now=datetime(2030, 1, 1, tzinfo=timezone.utc))
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(len(result['errors']), 4)

    def test_refresh_reuses_recent_noaa_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = obs._cache(root) / 'US1VTLM0001.csv.gz'; path.parent.mkdir(parents=True)
            path.write_bytes(noaa('20261001,SNOW,0,,,N,0730'))
            calls = []
            def download(url):
                calls.append(url)
                raise ValueError('offline')
            with patch.object(obs, '_download', side_effect=download):
                result = obs.refresh_observations(root)
            self.assertEqual(result['sites']['mansfield']['status'], 'cached')
            self.assertFalse(any('US1VTLM0001' in url for url in calls))

    def test_alta_history_requires_matching_season_and_dates(self):
        with self.assertRaises(ValueError):
            obs.parse_alta_history(alta_history('20252026'), '20242025')
        body = json.dumps(dict(snowfallHistory=[dict(reported_at='2026-12-01',
            season=20252026, ytd_snow=1)])).encode()
        with self.assertRaises(ValueError):
            obs.parse_alta_history(body, '20252026')

    def test_alta_history_preserves_reported_corrections_without_smoothing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = obs._cache(root) / 'alta_history_20252026.json'
            path.parent.mkdir(parents=True); path.write_bytes(alta_history('20252026', (9, 8)))
            result = obs.read_observations('alta', root)
            history = result['reported_total_history'][0]
            self.assertEqual(history['season'], '2025/26')
            self.assertEqual([r['total_inches'] for r in history['reported_totals']], [9, 8])
            self.assertEqual(history['correction_dates'], ['2025-10-12'])
            self.assertTrue(any('downward source revisions' in n for n in result['notes']))
            self.assertEqual(result['records'], [])  # never substituted as manual Collins labels

    def test_five_saved_alta_seasons_are_never_redownloaded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = obs._cache(root); raw.mkdir(parents=True)
            for year in range(2021, 2026):
                code = f'{year}{year + 1}'
                (raw / f'alta_history_{code}.json').write_bytes(alta_history(code))
            with patch.object(obs, 'build_opener', side_effect=AssertionError('archive must stay cached')):
                result = obs._refresh_alta_history(root, datetime(2026, 11, 18, tzinfo=timezone.utc), time.monotonic() + 2)
            self.assertEqual(result['status'], 'cached')
            self.assertEqual(len(result['seasons']), 5)

    def test_optional_history_deadline_does_not_damage_current_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = obs._cache(root); raw.mkdir(parents=True)
            current = alta([dict(reported_at='2026-11-17', ytd_snow=49)])
            (raw / 'alta_weather.html').write_bytes(current)
            with patch.object(obs, 'build_opener') as opener:
                result = obs._refresh_alta_history(root, datetime(2026, 11, 18, tzinfo=timezone.utc), time.monotonic() - 1)
                opener.return_value.open.assert_not_called()
            self.assertEqual(result['status'], 'failed')
            self.assertEqual((raw / 'alta_weather.html').read_bytes(), current)
            self.assertTrue(any('could not be downloaded' in n for n in obs.read_observations('alta', root)['notes']))


if __name__ == '__main__':
    unittest.main()
