"""Season plotting must not turn missing snow or time into observed zeros."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import dashboard_season as season

UTC = timezone.utc


def observation(day, amount=0, end=None):
    return {'date': day, 'snowfall_inches': amount, 'end': end}


def source(records, station='CLN', **extra):
    return {'records': records, 'target': {'station_id': station, 'target_name': 'Test station'},
            'sources': [], 'notes': [], **extra}


def forecast(start, station='CLN', site='alta', **overrides):
    row = dict(hours=48, status='ready', quantity='snowfall', unit='inches',
               start=start.isoformat(), end=(start + timedelta(hours=48)).isoformat(),
               issued_at=(start - timedelta(minutes=30)).isoformat(),
               mean=5., lower=2., upper=9., target={'station_id': station},
               interval_label='Uncalibrated 80% range')
    row.update(overrides)
    return {'site_id': site, 'target': {'station_id': station}, 'horizons': [row]}


class SeasonTests(unittest.TestCase):
    def build(self, records, now='2026-10-03T12:00:00+00:00', site='alta', **kwargs):
        with patch.object(season, '_observations', return_value=records):
            return season.get_season(site, now=now, forecast=kwargs.pop('forecast', {}), **kwargs)

    def test_zero_is_observed_but_missing_is_not(self):
        data = source([observation('2026-10-01', 0), observation('2026-10-03', 2)])
        result = self.build(data)
        self.assertEqual([p['cumulative_inches'] for p in result['current']['points']], [0., None, 2.])
        self.assertEqual(result['current']['recorded_total_inches'], 2)
        self.assertIsNone(result['total_inches'])
        self.assertTrue(result['current']['total_is_lower_bound'])
        self.assertEqual(result['forecast']['status'], 'unavailable')

    def test_observed_prefix_keeps_its_own_date_without_advancing_to_today(self):
        result = self.build(source([observation('2026-10-01', 2), observation('2026-10-02', 3)]))
        self.assertEqual(result['total_inches'], 5)
        self.assertEqual(result['observed_through'], '2026-10-02')
        self.assertIsNone(result['current']['points'][-1]['cumulative_inches'])
        self.assertFalse(result['current']['complete'])

    def test_february_29_is_preserved(self):
        result = self.build(source([observation('2024-02-28', 1), observation('2024-02-29', 2),
                                    observation('2024-03-01', 3)]))
        old = result['history'][0]
        leap = next(p for p in old['points'] if p['date'] == '2024-02-29')
        march = next(p for p in old['points'] if p['date'] == '2024-03-01')
        self.assertEqual(old['season'], '2023/24')
        self.assertEqual(leap['cumulative_inches'], 3)
        self.assertEqual(march['cumulative_inches'], 6)
        self.assertFalse(old['complete'])

    def test_local_today_and_future_endpoints_are_enforced(self):
        rows = [observation('2026-10-01', 2), observation('2026-10-02', 9),
                observation('2026-10-01', 99, '2026-10-02T12:00:00Z')]
        result = self.build(source(rows), now='2026-10-02T03:00:00Z')
        self.assertEqual(result['today'], '2026-10-01')
        self.assertEqual(result['total_inches'], 2)

    def test_conflicting_duplicate_is_missing(self):
        result = self.build(source([observation('2026-10-01', 2), observation('2026-10-01', 4)]))
        self.assertEqual(result['current']['valid_days'], 0)
        self.assertIsNone(result['total_inches'])

    def test_band_adds_baseline_once_and_ignores_other_horizons(self):
        start = datetime(2026, 10, 3, 12, tzinfo=UTC)
        data = source([observation(f'2026-10-0{i}', i, start.isoformat() if i == 3 else None)
                       for i in (1, 2, 3)])
        prediction = forecast(start)
        prediction['horizons'].append(dict(prediction['horizons'][0], hours=24, mean=99, lower=98, upper=100))
        result = self.build(data, forecast=prediction)
        points = result['forecast']['points']
        self.assertEqual(result['forecast']['status'], 'ready')
        self.assertEqual([p['cumulative_inches'] for p in points], [6, 11])
        self.assertEqual([p['lower_inches'] for p in points], [6, 8])
        self.assertEqual([p['upper_inches'] for p in points], [6, 15])
        self.assertEqual(result['current']['points'][-1]['time'], points[0]['time'])
        self.assertEqual(result['current']['points'][-1]['date'], '2026-10-03')

    def test_current_keeps_exact_cutoff_but_history_aligns_by_calendar_date(self):
        rows = [observation('2025-10-01', 2, '2025-10-01T11:30:00Z'),
                observation('2026-10-01', 3, '2026-10-01T11:30:00Z')]
        result = self.build(source(rows, station='US1VTLM0001'), site='mansfield')
        self.assertEqual(result['current']['points'][0]['time'], '2026-10-01T11:30:00+00:00')
        self.assertNotIn('time', result['current']['points'][1])  # Missing reports retain gaps.
        self.assertNotIn('time', result['history'][0]['points'][0])

    def test_gap_target_units_and_staleness_prevent_false_continuation(self):
        start = datetime(2026, 10, 1, 12, tzinfo=UTC)
        data = source([observation('2026-10-01', 3, start.isoformat())])
        cases = [forecast(start + timedelta(hours=1)), forecast(start, station='OTHER'),
                 forecast(start, quantity='liquid_precipitation', unit='mm'),
                 forecast(start, issued_at=(start - timedelta(hours=4)).isoformat()),
                 forecast(start, stale=True), forecast(start, site='jay'),
                 forecast(start, lower=10, upper=1)]
        for prediction in cases:
            with self.subTest(prediction=prediction):
                result = self.build(data, now=start.isoformat(), forecast=prediction)
                self.assertEqual(result['forecast']['status'], 'unavailable')
                self.assertEqual(result['forecast']['points'], [])

    def test_direct_season_snapshot_never_invents_unknown_daily_amounts(self):
        data = source([], reported_totals=[{'date': '2026-10-02', 'total_inches': 12}],
                      reported_total_source={'name': 'Official totals', 'url': 'https://example.org',
                                             'target_name': 'Resort total'})
        result = self.build(data)
        self.assertEqual(result['total_inches'], 12)
        self.assertEqual([p['cumulative_inches'] for p in result['current']['points']], [None, 12, None])
        self.assertEqual(result['current']['kind'], 'reported_season_total')
        self.assertEqual(result['forecast']['status'], 'unavailable')

    def test_unverified_direct_total_source_cannot_be_mixed_with_station_model(self):
        start = datetime(2026, 10, 1, 12, tzinfo=UTC)
        data = source([], reported_totals=[{'date': '2026-10-01', 'total_inches': 12, 'end': start.isoformat()}],
                      reported_total_source={'target_name': 'Resort total'})
        result = self.build(data, now=start.isoformat(), forecast=forecast(start))
        self.assertIn('targets', result['forecast']['note'])
        self.assertEqual(result['forecast']['points'], [])

    def test_matched_official_alta_history_replaces_manual_archive_curves(self):
        official = {'name': 'Alta official', 'url': 'https://www.alta.com/weather',
                    'target_name': 'Alta official snowfall report', 'station_id': None}
        data = source([observation('2024-10-01', 100)],
                      reported_totals=[{'date': '2026-10-02', 'total_inches': 12}],
                      reported_total_source=official,
                      reported_total_history=[dict(season='2025/26', source=official,
                          reported_totals=[{'date': '2025-10-01', 'total_inches': 5},
                                           {'date': '2025-10-03', 'total_inches': 9}])])
        result = self.build(data)
        self.assertEqual(len(result['history']), 1)
        old = result['history'][0]
        self.assertEqual(old['season'], '2025/26')
        self.assertEqual(old['kind'], 'reported_season_total')
        self.assertEqual([p['cumulative_inches'] for p in old['points'][:3]], [5, None, 9])
        self.assertFalse(old['total_is_lower_bound'])
        self.assertEqual(result['target_label'], official['target_name'])
        self.assertTrue(any('same official' in note for note in result['notes']))

    def test_manual_history_remains_explicit_fallback_without_official_history(self):
        data = source([observation('2024-10-01', 100)],
                      reported_totals=[{'date': '2026-10-02', 'total_inches': 12}],
                      reported_total_source={'target_name': 'Resort total'})
        result = self.build(data)
        self.assertEqual(result['history'][0]['kind'], 'daily_sum')
        self.assertTrue(any('different sources' in note for note in result['notes']))

    def saved_snapshot(self, directory, start, family='weather_radar'):
        root = Path(directory)
        selection = root / 'outputs/nwp-weather/forecast-portfolio/deployment_selection.json'
        snapshot = root / 'outputs/nwp-weather/neural48-trained/current_forecast.json'
        selection.parent.mkdir(parents=True)
        snapshot.parent.mkdir(parents=True)
        selection.write_text(json.dumps({'sites': {'alta': {'48': dict(
            quantity='snowfall', unit='inches', family='weather_radar')}}}))
        payload = dict(site_id='alta', family=family, status='experimental',
            start=start.isoformat(), issued_at=(start - timedelta(minutes=30)).isoformat(),
            target={'station_id': 'CLN'}, deployed_48=dict(hours=48, status='ready', family=family,
                end=(start + timedelta(hours=48)).isoformat(), mean_inches=5, q10_inches=2, q90_inches=9))
        snapshot.write_text(json.dumps({'sites': {'alta': payload}}))
        return root

    def test_saved_started_forecast_is_preserved_as_prediction_not_observation(self):
        start = datetime(2026, 10, 1, 12, tzinfo=UTC)
        data = source([observation('2026-10-01', 3, start.isoformat())])
        with tempfile.TemporaryDirectory() as directory:
            root = self.saved_snapshot(directory, start)
            with patch('dashboard_portfolio.read_deployed', return_value={'site_id': 'alta', 'horizons': []}):
                result = self.build(data, now='2026-10-02T12:00:00Z', root=root, forecast=None)
        curve = result['forecast']
        self.assertEqual(curve['status'], 'ready')
        self.assertTrue(curve['retained_issued_forecast'])
        self.assertEqual(curve['points'][0]['time'], start.isoformat())
        self.assertEqual(curve['points'][-1]['cumulative_inches'], 8)
        self.assertEqual(result['observed_through'], '2026-10-01')
        self.assertIsNone(result['current']['points'][-1]['cumulative_inches'])
        self.assertIn('not an observation', curve['note'])

    def test_saved_expired_or_wrong_family_forecast_is_not_recovered(self):
        start = datetime(2026, 10, 1, 12, tzinfo=UTC)
        data = source([observation('2026-10-01', 3, start.isoformat())])
        for family, now in [('weather_only', '2026-10-02T12:00:00Z'),
                            ('weather_radar', '2026-10-03T12:00:00Z')]:
            with self.subTest(family=family, now=now), tempfile.TemporaryDirectory() as directory:
                root = self.saved_snapshot(directory, start, family)
                with patch('dashboard_portfolio.read_deployed', return_value={'site_id': 'alta', 'horizons': []}):
                    result = self.build(data, now=now, root=root, forecast=None)
                self.assertEqual(result['forecast']['status'], 'unavailable')
                self.assertEqual(result['forecast']['points'], [])

    def test_history_is_bounded_and_never_contains_current_season(self):
        rows = [observation(f'{year}-10-01', 1) for year in range(1990, 2027)]
        result = self.build(source(rows))
        self.assertEqual(len(result['history']), season.MAX_HISTORY)
        self.assertEqual(result['history'][-1]['season'], '2025/26')

    def test_source_error_still_produces_renderable_empty_payload(self):
        with patch.object(season, '_observations', side_effect=OSError('missing')):
            result = season.get_season('jay', now='2026-11-18T12:00:00Z', forecast={})
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['history'], [])
        self.assertIsNone(result['total_inches'])

    def test_vail_is_explicitly_proxy_and_bad_clock_rejected(self):
        self.assertIn('proxy', self.build(source([]), site='vail')['quantity_label'])
        with self.assertRaises(ValueError):
            season.get_season('alta', now=datetime(2026, 11, 18))
        with self.assertRaises(ValueError):
            season.get_season('unknown')


if __name__ == '__main__':
    unittest.main()
