import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hrrr_portfolio import difference_asnow,reconcile_asnow_packing,schedule,location
from benchmark_hrrr_portfolio import cumulative_at
import pandas as pd
import tempfile,json
from unittest.mock import patch
import hrrr_portfolio as hrrr

class HourlySnowTests(unittest.TestCase):
    def test_cumulative_is_differenced_not_resummed(self):
        result=difference_asnow([0,.01,.01,.03])
        np.testing.assert_allclose(result,[0,.01,0,.02])
        self.assertAlmostEqual(result.sum(),.03)
    def test_missing_endpoint_does_not_create_zero(self):
        result=difference_asnow([.01,np.nan,.03,.04])
        self.assertTrue(np.isnan(result[1:3]).all())
        self.assertAlmostEqual(result[3],.01)
    def test_reset_is_not_negative_snow(self):
        result=difference_asnow([.02,.01,.03])
        self.assertTrue(np.isnan(result[1]))
    def test_actual_precision_transition_is_reconciled_within_grib_bounds(self):
        raw=np.array([.00262,.00260,.00270]);steps=np.array([.00002,.00010,.00010])
        adjusted,changes=reconcile_asnow_packing(raw,steps)
        self.assertEqual(len(changes),2)
        self.assertTrue((np.abs(adjusted-raw)<=steps/2+1e-12).all())
        hourly=difference_asnow(adjusted)
        self.assertTrue(np.isfinite(hourly).all());self.assertEqual(hourly[1],0.)
        self.assertAlmostEqual(hourly.sum(),raw[-1])
    def test_reconciliation_rejects_a_real_reset_instead_of_clipping(self):
        with self.assertRaisesRegex(ValueError,'beyond GRIB packing bounds'):
            reconcile_asnow_packing([.02,.01,.03],[.0001]*3)
    def test_missing_packing_or_values_cannot_fabricate_a_zero_increment(self):
        adjusted,_=reconcile_asnow_packing([.01,np.nan,.03,.04],[.0001]*4)
        self.assertTrue(np.isnan(difference_asnow(adjusted)[1:3]).all())
        adjusted,_=reconcile_asnow_packing([.00262,.00260],[.00002,np.nan])
        self.assertTrue(np.isnan(difference_asnow(adjusted)[1]))
    def test_reconciliation_keeps_valid_snow_and_cannot_hide_cumulative_drift(self):
        raw=[0,.01,.01,.03];adjusted,changes=reconcile_asnow_packing(raw,[.0001]*4)
        np.testing.assert_array_equal(adjusted,raw);self.assertEqual(changes,[])
        with self.assertRaisesRegex(ValueError,'beyond GRIB packing bounds'):
            reconcile_asnow_packing([.001,.00094,.00088],[.0001]*3)
    def test_calendar_selection_is_outcome_independent(self):
        runs=schedule()
        self.assertEqual(len(runs),189)
        self.assertEqual(len(set(runs)),189)
        self.assertTrue(all(r.hour==6 and r.day in [1,5,9,13,17,21,25,29] for r in runs))
    def test_gridpoint_stays_within_half_grid_diagonal(self):
        self.assertLess(location(40.57607,-111.63889)['distance_projected_m'],2200)
    def test_fractional_report_clock_forecast_only_interpolation(self):
        d=pd.DataFrame({'forecast_hour':[6,7,30,31],'snowfall_cumulative_inches':[1.,2.,4.,6.]})
        self.assertAlmostEqual(cumulative_at(d,30.5)-cumulative_at(d,6.5),3.5)
    def test_missing_adjacent_endpoint_cannot_be_interpolated(self):
        d=pd.DataFrame({'forecast_hour':[6,30],'snowfall_cumulative_inches':[1.,4.]})
        self.assertTrue(np.isnan(cumulative_at(d,6.5)))
    def test_failed_current_download_preserves_published_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);old='{"run_time_utc":"2026-11-11T18:00:00Z","retrieved_utc":"original"}'
            (out/'current.json').write_text(old);(out/'current_weather.csv').write_text('previous-weather')
            with patch.object(hrrr,'OUT',out),patch.object(hrrr,'fetch_run',side_effect=OSError('network unavailable')):
                with self.assertRaises(RuntimeError):hrrr.current()
            self.assertEqual((out/'current.json').read_text(),old)
            self.assertEqual((out/'current_weather.csv').read_text(),'previous-weather')
    def test_rechecking_immutable_run_retains_retrieval_and_publishes_paired_paths(self):
        run=(pd.Timestamp.now(tz='UTC')-pd.Timedelta(hours=3)).floor('6h');original=(run+pd.Timedelta(hours=3)).isoformat()
        hours=np.arange(1,49);frame=pd.DataFrame({'site':'alta','forecast_hour':hours,'run_time_utc':run.isoformat(),
            'valid_time_utc':(run+pd.to_timedelta(hours,unit='h')).astype(str)})
        for name in hrrr.VARIABLES:frame[name]=0.
        snow=frame[['site','forecast_hour','run_time_utc','valid_time_utc']].copy();snow['snowfall_inches']=0.
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'current.json').write_text(json.dumps({'run_time_utc':run.isoformat(),'retrieved_utc':original}))
            with patch.object(hrrr,'OUT',out),patch.object(hrrr,'ROOT',out),patch.object(hrrr,'SITES',{'alta':(40,-111,2900)}),patch.object(hrrr,'fetch_run',return_value=frame),patch.object(hrrr,'fetch_asnow',return_value=snow):
                result=hrrr.current()
            self.assertEqual(result['retrieved_utc'],original)
            self.assertTrue((out/result['weather_path']).exists());self.assertTrue((out/result['snowfall_path']).exists())
            self.assertEqual(pd.Timestamp(result['run_time_utc']),run)
            self.assertGreaterEqual(pd.Timestamp(result['checked_at']),pd.Timestamp(original))

if __name__=='__main__':unittest.main()
