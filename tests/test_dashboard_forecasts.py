"""Timing, unit, identity and inference contracts for dashboard refresh."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import dashboard_forecasts as dashboard


class DashboardForecastTests(unittest.TestCase):
    def test_hourly_values_are_complete_future_intervals_and_inches(self):
        now = pd.Timestamp('2026-11-06T15:23Z')
        times = pd.date_range('2026-11-06T00:00Z', periods=96, freq='h')
        response = {'hourly_units': {'snowfall':'cm','precipitation':'mm','temperature_2m':'°C','wind_speed_10m':'km/h'},
                    'hourly': {'time':[t.isoformat() for t in times], 'snowfall':[2.54]*96,
                               'precipitation':[1]*96,'temperature_2m':[-3]*96,'wind_speed_10m':[18]*96}}
        rows, start = dashboard._hourly_from_response(response, now)
        self.assertEqual(len(rows), 48)
        self.assertGreaterEqual(pd.Timestamp(rows[0]['start']), now)
        self.assertEqual(rows[0]['time'], '2026-11-06T17:00:00+00:00')
        self.assertEqual(rows[0]['snowfall_inches'], 1.)
        self.assertEqual(rows[-1]['horizon_hours'], 48)
        self.assertEqual(pd.Timestamp(rows[-1]['time'])-start, pd.Timedelta(hours=48))
        response['hourly_units']['snowfall'] = 'inch'
        with self.assertRaises(ValueError): dashboard._hourly_from_response(response, now)

    @staticmethod
    def mos_rows(runs):
        rows = []
        for run in runs:
            run = pd.Timestamp(run)
            for h in range(6,73,3):
                rows.append(dict(runtime=run, ftime=run+pd.Timedelta(hours=h), station='KSLC', model='GFS',
                                 tmp=25,dpt=20,wdr=270,wsp=10,p06=50 if h%6==0 else np.nan,
                                 q06=2 if h%6==0 else np.nan))
        return pd.DataFrame(rows)

    def test_daily_windows_preserve_trained_clock_and_availability(self):
        raw = self.mos_rows(['2026-11-05T00:00Z','2026-11-06T00:00Z','2026-11-06T12:00Z'])
        future, run = dashboard._mos_future(raw, 'alta', pd.Timestamp('2026-11-06T15:23Z'))
        self.assertEqual(run, pd.Timestamp('2026-11-06T00:00Z'))
        self.assertEqual(future.lead_day.tolist(), [2])
        self.assertEqual(future.iloc[0].valid_start_utc, '2026-11-07T11:00:00+00:00')
        self.assertEqual(future.iloc[0].valid_end_utc, '2026-11-08T11:00:00+00:00')
        self.assertAlmostEqual(future.iloc[0].forecast_qpf_ordinal_weighted_mean, 2)
        early, chosen = dashboard._mos_future(raw, 'alta', pd.Timestamp('2026-11-06T05:00Z'))
        self.assertEqual(chosen, pd.Timestamp('2026-11-05T00:00Z'))
        self.assertEqual(early.lead_day.tolist(), [2])
        raw['station'] = 'KLXV'
        with self.assertRaises(ValueError): dashboard._mos_future(raw, 'alta', pd.Timestamp('2026-11-06T15:23Z'))

    def test_saved_model_replays_existing_predictions_without_refitting(self):
        base = dashboard.ROOT/'outputs/nwp-weather/mos/alta'
        if not all((base/name).is_file() for name in ('current_input.csv', 'current_predictions.csv', 'final_models.joblib')):
            self.skipTest('Optional saved MOS model and replay fixtures are not installed')
        future = pd.read_csv(base/'current_input.csv')
        observed = pd.read_csv(base/'current_predictions.csv')
        observed = observed[observed.model.eq(dashboard.MODEL_NAME)].sort_values('lead_day')
        with patch.object(dashboard.nwp.EmpiricalForest, 'fit', side_effect=AssertionError('Refresh must not train')):
            rows, digest = dashboard._daily_inference('alta', future)
        self.assertEqual(len(digest), 64)
        np.testing.assert_allclose([r['mean_inches'] for r in rows], observed.predicted_inches, atol=1e-12)
        np.testing.assert_allclose([r['p_powder'] for r in rows], observed.p_powder, atol=1e-12)
        self.assertTrue(all(0 <= r['p_powder'] <= 1 for r in rows))
        json.dumps(rows, allow_nan=False)

    @staticmethod
    def saved_fixture(now):
        start=now.ceil('h')
        return {'site':{'id':'alta'},'status':'cached','generated_at':now.isoformat(),
                'warnings':[], 'source':{'name':'Saved test forecast','status':'cached','retrieved_at':now.isoformat()},
                'daily':[], 'hourly':[{'start':start.isoformat(),'time':(start+pd.Timedelta(hours=1)).isoformat(),'snowfall_inches':0.}]}

    def test_offline_get_never_fetches_and_unknown_site_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard, 'CACHE', Path(directory)), \
             patch.object(dashboard, '_fetch', side_effect=AssertionError('No implicit download')), \
             patch.object(dashboard, '_saved_snapshot', side_effect=lambda site,now:self.saved_fixture(now)):
            result = dashboard.get_forecast('alta')
            self.assertEqual(result['status'], 'cached')
            self.assertEqual(result['site']['id'], 'alta')
            json.dumps(result, allow_nan=False)
            with self.assertRaises(ValueError): dashboard.get_forecast('../alta')

    def test_cached_periods_expire_without_changing_source_dates(self):
        now = pd.Timestamp('2026-11-06T15:23Z')
        result = {'generated_at':'2026-11-05T12:00:00Z','warnings':[],
                  'source':{'issued_at':'2026-11-05T00:00:00Z','retrieved_at':'2026-11-05T12:00:00Z'},
                  'hourly':[{'start':'2026-11-06T14:00:00Z','time':'2026-11-06T15:00:00Z'},
                            {'start':'2026-11-06T16:00:00Z','time':'2026-11-06T17:00:00Z'}],
                  'daily':[{'start':'2026-11-06T11:00:00Z','end':'2026-11-07T11:00:00Z'}]}
        served = dashboard._serve(result, now, cached=True)
        self.assertEqual(len(served['hourly']), 1)
        self.assertEqual(served['hourly'][0]['horizon_hours'], 1)
        self.assertEqual(served['daily'], [])
        self.assertEqual(served['generated_at'], result['generated_at'])
        self.assertTrue(served['source']['stale'])
        self.assertGreater(served['cache']['age_hours'], 24)
        self.assertEqual(served['hours_available'], 1)
        self.assertEqual(len(result['hourly']), 2)

    def test_source_failure_never_fabricates_daily_powder(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard, 'CACHE', Path(directory)), \
             patch.object(dashboard, '_fetch_hourly', side_effect=OSError('Offline')), \
             patch.object(dashboard, '_fetch_daily', side_effect=OSError('Offline')), \
             patch.object(dashboard, '_saved_snapshot', side_effect=lambda site,now:self.saved_fixture(now)):
            result = dashboard.get_forecast('alta', refresh=True)
            self.assertEqual(len(result['errors']), 2)
            self.assertEqual(result['daily'], [])
            self.assertNotEqual(result['status'], 'ready')
            self.assertEqual(result['source']['status'], 'cached')
            json.dumps(result, allow_nan=False)


if __name__ == '__main__': unittest.main()
