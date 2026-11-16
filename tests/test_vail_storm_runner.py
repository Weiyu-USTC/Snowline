"""Checks for origin cutoff, censored onset labels and benchmark scoring."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_vail_storm_models as runner


class EventBenchmarkTests(unittest.TestCase):
    def test_origin_history_cannot_read_future_observations(self):
        dates=pd.date_range('2025-09-01 04:00',periods=12)
        target=pd.Series([0.,0.,2.,2.,2.,0.,0.,0.,2.,2.,0.,3.],index=dates)
        origins=dates[[0,4,7]]
        before=runner.initial_history(target,origins)
        revised=target.copy();revised.loc[dates[8]:]=100.
        after=runner.initial_history(revised,origins)
        np.testing.assert_array_equal(before,after)
        np.testing.assert_array_equal(before,[[0.,0.,0.,1.],[2.,3.,1.,1.],[0.,3.,1.,1.]])

    def test_missing_issue_and_history_gap_are_censored_not_dry(self):
        dates=pd.date_range('2025-09-01 04:00',periods=7)
        target=pd.Series([0.,2.,np.nan,2.,2.,0.,0.],index=dates)
        initial=runner.initial_history(target,dates[2:])
        self.assertTrue(np.isnan(initial[0,0]))
        np.testing.assert_array_equal(initial[0,1:],[0.,0.,0.])
        np.testing.assert_array_equal(initial[1:],[[2.,0.,0.,1.],[2.,0.,0.,1.],[0.,1.,1.,1.],[0.,2.,1.,1.]])
        np.testing.assert_array_equal(np.isnan(runner.ages(initial)),[True,True,True,False,False])

    def test_run_age_is_open_tail_and_resets_at_winter_boundary(self):
        dates=pd.date_range('2025-08-30 04:00',periods=25)
        target=pd.Series(np.r_[0.,2.,np.ones(23)*2],index=dates)
        # A transition outside the fitting winter cannot establish a known age.
        np.testing.assert_array_equal(runner.initial_history(target,[dates[-1]]),[[2.,0.,0.,1.]])
        target.loc['2025-09-02 04:00']=0.
        np.testing.assert_array_equal(runner.initial_history(target,[dates[-1]]),[[2.,14.,1.,1.]])

    def test_onset_is_quiet_to_active_not_any_active_day(self):
        initial=np.array([[2.,3.,1.,1.],[0.,3.,1.,1.],[0.,0.,0.,1.]])
        y=np.array([[2.,2.,2.],[.3,1.,2.],[0.,.9,.2]])
        event,known=runner.window_onset_labels(y,initial,3)
        np.testing.assert_array_equal(event,[False,True,False])
        self.assertTrue(known.all())
        event,known=runner.window_onset_labels(y,initial,1)
        np.testing.assert_array_equal(event,[False,False,False])
        self.assertTrue(known.all())

    def test_onset_window_requires_observed_initial_and_window_states(self):
        initial=np.array([[np.nan,0.,0.,0.],[0.,1.,1.,1.],[0.,1.,1.,1.]])
        y=np.array([[2.,2.,2.],[2.,np.nan,0.],[2.,2.,np.nan]])
        _,known=runner.window_onset_labels(y,initial,1)
        np.testing.assert_array_equal(known,[False,True,True])
        _,known=runner.window_onset_labels(y,initial,3)
        # Conservative protocol omits even a confirmed positive if its full
        # window has gaps; it must not quietly treat a missing reading as dry.
        np.testing.assert_array_equal(known,[False,False,False])

    def test_empirical_crps_equals_direct_pairwise_definition(self):
        paths=np.array([[[0.,.2],[2.,1.],[4.,3.]],[[1.,0.],[1.,2.],[1.,4.]]])
        actual=np.array([[1.,2.],[2.,1.]])
        expected=np.abs(paths-actual[:,None,:]).mean(axis=1)
        expected-=.5*np.abs(paths[:,:,None,:]-paths[:,None,:,:]).mean(axis=(1,2))
        np.testing.assert_allclose(runner.crps_sample(paths,actual),expected,rtol=1e-14)

    def test_one_member_crps_is_absolute_error(self):
        paths=np.array([[[0.,2.,4.]]]);actual=np.array([[1.,2.,1.]])
        np.testing.assert_array_equal(runner.crps_sample(paths,actual),[[1.,0.,3.]])


if __name__=='__main__':unittest.main()
