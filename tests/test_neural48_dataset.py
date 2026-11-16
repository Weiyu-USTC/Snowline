"""Checks that provider accumulation windows conserve amounts and missingness."""
from pathlib import Path
import sys, unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_neural48_dataset import provider_window, bounded_instant

class ProviderWindows(unittest.TestCase):
    def frame(self):
        t=pd.date_range('2024-01-01T10:00Z',periods=52,freq='h')
        return pd.DataFrame(dict(valid_time_utc=t,temperature_2m=0.,precipitation=1.,snowfall=2.54,wind_speed_10m=4.,wind_direction_10m=270.,relative_humidity_2m=90.,pressure_msl=1000.))
    def test_fractional_reporting_clock_conserves_amount(self):
        x,t=provider_window(self.frame(),pd.Timestamp('2024-01-01T11:30Z'))
        self.assertEqual(x.shape,(48,7));self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(x[:,3],1.)
        np.testing.assert_allclose(x[:,4],1.)
        self.assertEqual(t[-1],pd.Timestamp('2024-01-03T11:30Z'))
    def test_missing_snow_is_not_zero(self):
        f=self.frame();f.loc[5,'snowfall']=np.nan
        x,_=provider_window(f,pd.Timestamp('2024-01-01T11:30Z'))
        self.assertEqual(np.isnan(x[:,4]).sum(),2)
    def test_missing_interval_is_not_interpolated(self):
        f=self.frame().drop(index=5)
        x,_=provider_window(f,pd.Timestamp('2024-01-01T11:30Z'))
        self.assertGreaterEqual(np.isnan(x[:,4]).sum(),2)
    def test_instantaneous_gap_is_not_bridged(self):
        times=np.array([0,3600000000000,7200000000000],dtype=np.int64)
        self.assertTrue(np.isnan(bounded_instant(times,np.array([1.,np.nan,3.]),5400000000000)))
        self.assertTrue(np.isnan(bounded_instant(times[[0,2]],np.array([1.,3.]),3600000000000)))

if __name__=='__main__':unittest.main()
