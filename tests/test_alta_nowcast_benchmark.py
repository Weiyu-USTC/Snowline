"""Unit/time checks that can otherwise silently invalidate the comparison."""
from pathlib import Path
import sys, unittest
import tempfile
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluate_alta_nowcast import integral,ensemble_crps,phase_flow,weather_features

class TimingAndScoring(unittest.TestCase):
    def test_rate_integral_uses_clock_correct_window(self):
        a=np.full((8,20,4,4),2.)
        a[:,0]=10000 #10:50UTC lies outside11UTC targetwindow
        for h in [1,2,3]:self.assertTrue(np.allclose(integral(a,h),2*h))
        a[:,1]=4 #11:00 endpoint gets half-weight, not a full10minbin
        self.assertTrue(np.allclose(integral(a,1),2+1/6))

    def test_crps_matches_direct_pairwise_definition(self):
        a=np.array([[0.,1.],[1.,2.],[4.,3.]])
        y=np.array([2.,0.])
        expected=abs(a-y).mean(0)-.5*abs(a[:,None]-a[None,:]).mean((0,1))
        self.assertTrue(np.allclose(ensemble_crps(a,y),expected))

    def test_motion_advects_in_observed_direction(self):
        x=np.zeros((9,2,64,64));x[:,1]=1
        for j in range(9):x[j,0,25:30,20+j:25+j]=2
        p=phase_flow(x,3)
        self.assertGreater(np.argmax(p[2].sum(0)),np.argmax(x[-1,0].sum(0)))

    def test_hrrr_daily_window_excludes_before11utc(self):
        periods=np.arange(1,49);a=np.ones(48);a[:5]=999
        d={'precip_mm':{'periods':periods,'mean3x3':a}}
        f=weather_features(d,'2023-01-01')
        self.assertEqual(f['precip_mm_sum_24h'],24)
        self.assertEqual(f['precip_mm_sum_3h'],3)

    def test_fractional_hour_resampling_has_consistent_timestamp_units(self):
        from forecast_alta_nowcast import resample_run
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'20260101T0000Z';folder.mkdir()
            names=['precip_mm','snow_water_mm','temperature_k','dewpoint_k','wind_u_ms','wind_v_ms','relative_humidity_pct']
            for name in names:np.savez(folder/(name+'.npz'),periods=np.arange(1,49),mean3x3=np.arange(1.,49.))
            d,_=resample_run(folder,pd.Timestamp('2026-01-01T05:20Z'))
            self.assertAlmostEqual(d['precip_mm']['mean3x3'].sum(),428.)
            self.assertAlmostEqual(d['precip_mm']['mean3x3'][0],6+1/3)

if __name__=='__main__':unittest.main()
