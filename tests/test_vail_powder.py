"""Independent checks for inclusive powder-threshold probabilities."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vail_storm_statistical import EventModel
from build_vail_powder_forecast import neural_exceedance


def calendar(n,h):
    phase=np.linspace(.1,1.2,n*h).reshape(n,h)
    return np.stack([np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)],axis=-1)


class PowderProbabilityTests(unittest.TestCase):
    def model(self):
        rng=np.random.default_rng(905)
        summary=rng.normal(size=(120,2))
        current=rng.choice([0.,.5,1.,5.,9.],120)
        target=rng.choice([0.,.5,1.,5.,9.],120)
        model=EventModel(kind='semi_markov',regularization=.1)
        model.fit(summary,calendar(120,1)[:,0],current,rng.integers(0,15,120),target)
        model._all_hazards=lambda summary,cal:np.full((len(summary),cal.shape[1],30),.3)
        model._pool=lambda phase,state,age:np.array([1.,5.,9.] if state else [0.,.5])
        return model

    def test_statistical_threshold_is_inclusive_and_one_matches_activity(self):
        model=self.model();kwargs=dict(summary=np.zeros((2,2)),calendar_future=calendar(2,5),
                                      current_y=[0.,6.],state_age=[3,14],n_paths=256,seed=91)
        five=model.forecast(**kwargs,exceedance_threshold=5.)
        # Two of three active-state donors are >=5, including exact equality.
        np.testing.assert_allclose(five['p_exceedance'],.3*(2/3),atol=1e-12)
        one=model.forecast(**kwargs,exceedance_threshold=1.)
        np.testing.assert_allclose(one['p_exceedance'],one['p_active'],atol=1e-12)
        huge=model.forecast(**kwargs,exceedance_threshold=100.)
        np.testing.assert_array_equal(huge['p_exceedance'],0.)

    def test_statistical_new_output_cannot_change_existing_means_or_paths(self):
        model=self.model();kwargs=dict(summary=np.zeros((2,2)),calendar_future=calendar(2,5),
                                      current_y=[0.,6.],state_age=[3,14],n_paths=256,seed=92)
        before=model.forecast(**kwargs)
        after=model.forecast(**kwargs,exceedance_threshold=5.)
        for key in ['mean','p_active','p_onset','p_onset_within','paths','active_paths','onset_paths']:
            np.testing.assert_array_equal(before[key],after[key])

    def test_neural_tail_has_correct_truncation_and_mixture_weights(self):
        weights=np.array([[[.25,.75]]])
        params=np.zeros((1,1,2,6))
        probabilities=np.array([.2,.8])
        params[...,0]=np.log(probabilities/(1-probabilities))
        params[...,2:4]=1.
        params[...,5]=1.  # log amount is N(0,1), conditional on >=0.
        prediction=dict(component_weights=weights,component_params=params)
        active=.25*.2+.75*.8
        np.testing.assert_allclose(neural_exceedance(prediction,threshold=1.),active,atol=1e-12)
        # N(0,1)'s 75th percentile is the median after truncation below zero.
        threshold=np.exp(.6744897501960817)
        np.testing.assert_allclose(neural_exceedance(prediction,threshold=threshold),active*.5,atol=1e-12)

    def test_neural_five_inch_tail_matches_independent_truncated_normal_draws(self):
        rng=np.random.default_rng(93)
        weights=np.array([[[.35,.65]]]);params=np.zeros((1,1,2,6))
        p=np.array([.3,.75]);mu=np.array([.8,1.7]);sigma=np.array([.5,.8])
        params[...,0]=np.log(p/(1-p));params[...,2:4]=1.
        params[...,4]=mu;params[...,5]=sigma
        exact=float(neural_exceedance(dict(component_weights=weights,component_params=params),threshold=5.).item())
        # Rejection sampling is independent of the helper's Gaussian CDF ratio.
        estimates=[]
        for j in range(2):
            log_amount=rng.normal(mu[j],sigma[j],800000)
            log_amount=log_amount[log_amount>=0.]
            estimates.append(p[j]*(log_amount>=np.log(5.)).mean())
        simulated=float(np.dot(weights[0,0],estimates))
        self.assertLess(abs(exact-simulated),.0015)


if __name__=='__main__':unittest.main()
