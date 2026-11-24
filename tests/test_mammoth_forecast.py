"""Live model quantity, issue-time and weather coverage contracts, without downloads."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mammoth_forecast as forecast


RUN = datetime(2026, 11, 24, 12, tzinfo=timezone.utc)
NOW = RUN + timedelta(hours=6, minutes=30)


def guidance(hours_available=48):
    hours = []
    for i in range(1, hours_available + 1):
        end = RUN + i * forecast.HOUR
        hours.append(dict(start_utc=(end - forecast.HOUR).isoformat(), end_utc=end.isoformat(),
                          lead=i, precip_mm=3., snow_water_mm=1., temperature_c=-2.))
    return dict(status='ready', site=forecast.TARGET.copy(), run_utc=RUN.isoformat(),
                available_utc=(RUN+3*forecast.HOUR).isoformat(), retrieved_utc=NOW.isoformat(), hours=hours)


class MammothForecastTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        # An installed public checkout can obtain the full forecast from its bundled artifact.
        path = self.root/'artifacts/mammoth-hourly/model.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(version=1, family='test-climatology', uses_radar=False, estimator={'kind': 'climatology'},
                              monthly_means_mm=[.5]*12, calibration={'3': {'lower_offset_mm': -2., 'upper_offset_mm': 2.}})))

    def build(self):
        with patch('mammoth_weather.read_current', return_value=guidance()):
            return forecast.refresh_mammoth_model(self.root, NOW)

    def test_frozen_model_hourly_sums_and_exact_baseline(self):
        result = self.build()
        self.assertEqual(result['status'], 'succeeded', result)
        data = forecast.read_deployed_mammoth(self.root, NOW)
        self.assertEqual(data['quantity'], 'snow_water_equivalent')
        self.assertEqual(data['start'], (RUN + 7*forecast.HOUR).isoformat())
        self.assertFalse(data['window_started'])
        self.assertEqual(data['sources']['radar']['used_as_input'], False)
        for row in data['horizons']:
            h = row['hours']
            self.assertEqual(row['mean'], sum(v['mean_mm'] for v in data['hourly'][:h]))
            self.assertEqual(row['provider_mean'], float(h) if h <= 41 else None)
        self.assertEqual(data['horizons'][1]['upper'], 3.5)
        self.assertEqual(data['horizons'][1]['lower'], 0.)
        self.assertEqual(data['horizons'][-1]['provider_hours'], 41)
        self.assertTrue(all(r['provider_mm'] is None for r in data['hourly'][41:]))

    def test_available_radar_is_not_a_predictor_for_weather_only_selection(self):
        features = {'timestamp': (NOW - timedelta(minutes=5)).isoformat(), 'point': 2.}
        with patch('mammoth_weather.read_current', return_value=guidance()), \
             patch.object(forecast, '_radar_input', return_value=(features, {'available': True})):
            self.assertEqual(forecast.refresh_mammoth_model(self.root, NOW)['status'], 'succeeded')
        source = forecast.read_deployed_mammoth(self.root, NOW)['sources']['radar']
        self.assertTrue(source['available'])
        self.assertFalse(source['used_as_input'])

    def test_ecmwf_comparison_requires_same_quantity_and_complete_windows(self):
        external = guidance(72)
        # Keep one native 60-minute total missing. Total precipitation remains positive.
        external['hours'][16]['snow_water_mm'] = None
        with patch('mammoth_weather.read_current', return_value=guidance()), \
             patch('mammoth_weather.read_provider', return_value=external, create=True):
            self.assertEqual(forecast.refresh_mammoth_model(self.root, NOW)['status'], 'succeeded')
        data = forecast.read_deployed_mammoth(self.root, NOW)
        self.assertEqual(data['sources']['ecmwf']['status'], 'partial')
        self.assertEqual(data['horizons'][2]['ecmwf_mean'], 6.)
        self.assertEqual(data['horizons'][-1]['ecmwf_hours'], 47)
        self.assertIsNone(data['horizons'][-1]['ecmwf_mean'])
        self.assertEqual(data['hourly'][9]['ecmwf_mm'], None)
        external['site']['longitude'] = -119.5
        with patch('mammoth_weather.read_current', return_value=guidance()), \
             patch('mammoth_weather.read_provider', return_value=external, create=True):
            self.assertEqual(forecast.refresh_mammoth_model(self.root, NOW)['status'], 'succeeded')
        data = forecast.read_deployed_mammoth(self.root, NOW)
        self.assertEqual(data['sources']['ecmwf']['status'], 'unavailable')
        self.assertEqual(data['horizons'][2]['mean'], 3.)

    def test_stale_future_and_incomplete_new_runs_preserve_original(self):
        self.assertEqual(self.build()['status'], 'succeeded')
        prior = (self.root/forecast.SNAPSHOT).read_bytes()
        for case in ('expired', 'future', 'gap', 'negative', 'wrong site', 'future download'):
            with self.subTest(case=case):
                weather = guidance()
                if case == 'expired':
                    weather['status'] = 'stale'
                if case == 'future':
                    weather['available_utc'] = (NOW+forecast.HOUR).isoformat()
                if case == 'gap':
                    weather['hours'].pop(18)
                if case == 'negative':
                    weather['hours'][10]['snow_water_mm'] = -1.
                if case == 'wrong site':
                    weather['site']['latitude'] = 0.
                if case == 'future download':
                    weather['retrieved_utc'] = (NOW+forecast.HOUR).isoformat()
                with patch('mammoth_weather.read_current', return_value=weather):
                    result = forecast.refresh_mammoth_model(self.root, NOW)
                self.assertEqual('failed', result['status'])
                self.assertEqual(prior, (self.root/forecast.SNAPSHOT).read_bytes())

    def test_inconsistent_saved_total_is_withheld(self):
        self.build()
        path = self.root / forecast.SNAPSHOT
        payload = json.loads(path.read_text())
        payload['horizons'][2]['mean'] += 1.
        path.write_text(json.dumps(payload))
        result = forecast.read_deployed_mammoth(self.root, NOW)
        self.assertEqual(result['status'], 'unavailable')
        self.assertIn('total disagrees', result['warnings'][0])

    def test_original_issue_and_windows_remain_dated(self):
        self.build()
        data = forecast.read_deployed_mammoth(self.root, NOW + 4*forecast.HOUR)
        self.assertEqual(data['status'], 'cached')
        self.assertTrue(data['horizons'][0]['expired'])
        self.assertTrue(data['horizons'][-1]['stale'])
        self.assertEqual(data['start'], (RUN + 7 * forecast.HOUR).isoformat())
        self.assertEqual(data['issued_at'], NOW.isoformat())
        expired = forecast.read_deployed_mammoth(self.root, NOW + 55*forecast.HOUR)
        self.assertEqual(expired['hourly'], [])
        self.assertTrue(all(row['mean'] is None for row in expired['horizons']))
        self.assertEqual(forecast.read_deployed_mammoth(self.root, NOW-forecast.HOUR)['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
