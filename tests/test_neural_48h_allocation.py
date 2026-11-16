import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from report_neural_48h import day_weights, cumulative_distribution, uniform_weights


class TemporalAllocation(unittest.TestCase):
    def setUp(self):
        self.start=pd.Timestamp('2026-11-10T00:00Z')
        self.hours=pd.date_range(self.start+pd.Timedelta(hours=1),periods=72,freq='h')

    def test_full_day_conservation_and_crop_without_renormalization(self):
        snow=np.arange(1,73,dtype=float)
        full,reason=day_weights(self.start,self.start+pd.Timedelta(days=1),self.hours,snow,self.hours[:24])
        cropped,_=day_weights(self.start,self.start+pd.Timedelta(days=1),self.hours,snow,self.hours[12:24])
        self.assertIsNone(reason)
        self.assertAlmostEqual(full.sum(),1.)
        np.testing.assert_allclose(cropped,full[12:])
        self.assertLess(cropped.sum(),1.)

    def test_half_hour_boundaries_are_not_double_counted(self):
        lo=self.start+pd.Timedelta(hours=12,minutes=30)
        one,_=day_weights(lo,lo+pd.Timedelta(days=1),self.hours,np.ones(72),self.hours)
        two,_=day_weights(lo+pd.Timedelta(days=1),lo+pd.Timedelta(days=2),self.hours,np.ones(72),self.hours)
        self.assertAlmostEqual(one[12],.5/24)
        self.assertAlmostEqual(one[36],.5/24)
        self.assertAlmostEqual(one[36]+two[36],1/24)
        self.assertAlmostEqual(one.sum(),1.)

    def test_dry_or_missing_day_is_unavailable(self):
        for snow in [np.zeros(72),np.r_[np.nan,np.ones(71)]]:
            result,reason=day_weights(self.start,self.start+pd.Timedelta(days=1),self.hours,snow,self.hours[:24])
            self.assertIsNone(result)
            self.assertTrue(reason)

    def test_snow_outside_display_is_not_pulled_into_it(self):
        snow=np.zeros(72);snow[0]=4.
        result,reason=day_weights(self.start,self.start+pd.Timedelta(days=1),self.hours,snow,self.hours[12:24])
        self.assertIsNone(reason)
        self.assertEqual(result.sum(),0.)

    def test_joint_paths_are_summed_before_quantiles(self):
        paths=np.array([[0.,10.],[10.,0.]])
        low,high=cumulative_distribution(paths,np.eye(2))
        self.assertEqual(low[-1],10.)
        self.assertEqual(high[-1],10.)
        self.assertNotEqual(low[-1],np.quantile(paths,.1,axis=0).sum())

    def test_incomplete_and_negative_source(self):
        w,_=day_weights(self.start,self.start+pd.Timedelta(days=1),self.hours[1:],np.ones(71),self.hours[:24])
        self.assertIsNone(w)
        with self.assertRaises(ValueError):
            day_weights(self.start,self.start+pd.Timedelta(days=1),self.hours,-np.ones(72),self.hours[:24])

    def test_uniform_is_explicit_separate_and_conserves(self):
        weights=uniform_weights(self.start,self.start+pd.Timedelta(days=1),self.hours[:24])
        np.testing.assert_allclose(weights,np.full(24,1/24))
        self.assertAlmostEqual(weights.sum(),1.)


if __name__=='__main__':unittest.main()
