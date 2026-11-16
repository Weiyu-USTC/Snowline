"""Independent checks of distribution scoring used in the event comparison."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import ndtr, ndtri

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_vail_storm_models import crps_sample, hurdle_lognormal_crps


class StormScoreTests(unittest.TestCase):
    def test_degenerate_distribution_scores_absolute_error(self):
        paths=np.full((1,8,3),2.)
        actual=np.array([[0.,2.,5.]])
        for fair in [False,True]:
            np.testing.assert_allclose(crps_sample(paths,actual,fair=fair),[[2.,0.,3.]])
        params=np.array([[[0.,.3,.8],[0.,-.2,1.1],[0.,2.,.3]]])
        np.testing.assert_allclose(hurdle_lognormal_crps(params,actual),actual)

    def test_hurdle_lognormal_crps_matches_dense_quantile_distribution(self):
        params=np.array([[[.2,-.3,.4],[.7,.5,.7],[1.,.4,1.1],[.45,.2,.9]]])
        actual=np.array([[0.,.5,5.,12.]])
        n=300000
        u=(np.arange(n)+.5)/n
        p,mu,sigma=params[0].T
        quantile=(u[:,None]-(1-p))/p
        samples=np.where(quantile<=0,0.,np.exp(mu+sigma*ndtri(np.clip(quantile,1e-12,1-1e-12))))[None]
        numerical=crps_sample(samples,actual)
        analytic=hurdle_lognormal_crps(params,actual)
        np.testing.assert_allclose(analytic,numerical,atol=.002,rtol=.001)

    def test_zero_observation_has_closed_form_atom_aware_score(self):
        params=np.array([[[.1,.3,.9],[.8,-.2,.5],[1.,.7,1.2]]])
        p,mu,sigma=params[0].T
        expected=2*p*p*np.exp(mu+.5*sigma*sigma)*ndtr(-sigma/np.sqrt(2))
        np.testing.assert_allclose(hurdle_lognormal_crps(params,np.zeros((1,3))),expected[None],rtol=1e-13)

    def test_fair_crps_estimates_gaussian_parent_score(self):
        rng=np.random.default_rng(8273)
        paths=rng.normal(size=(2000,128,1))
        actual=np.full((2000,1),.4)
        fair=crps_sample(paths,actual,fair=True)
        ordinary=crps_sample(paths,actual,fair=False)
        expected=.4*(2*ndtr(.4)-1)+2*np.exp(-.4**2/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi)
        self.assertLess(abs(fair.mean()-expected),.005)
        self.assertTrue((ordinary>=fair-1e-12).all())
        with self.assertRaises(ValueError):crps_sample(paths[:,:1],actual,fair=True)


if __name__=='__main__':unittest.main()
