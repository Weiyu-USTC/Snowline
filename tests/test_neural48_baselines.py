import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from neural48_baselines import fit_baselines,predict_baselines,CURRENT,ALL_WEATHER,CLIMATE


class JointBaselineTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(5);n=80
        nwp=rng.normal(size=(n,48,23));static=rng.normal(size=(n,4))
        history=rng.normal(size=(n,3));site=np.r_[np.zeros(40,int),np.ones(40,int)]
        y=np.c_[np.arange(n)%11,10-np.arange(n)%11].astype(float)
        doy=np.r_[np.arange(1,41),np.arange(1,41)]
        fit=np.r_[np.arange(30),np.arange(40,70)];test=np.r_[np.arange(30,40),np.arange(70,80)]
        return nwp,static,history,site,y,fit,test,doy

    def test_no_test_target_or_predictor_leakage(self):
        n,s,h,site,y,tr,te,doy=self.data()
        first=fit_baselines(n,s,h,site,y,tr,doy,n_paths=256)
        y2=y.copy();y2[te]=99999.
        n2=n.copy();n2[te]=99999.
        second=fit_baselines(n2,s,h,site,y2,tr,doy,n_paths=256)
        a=predict_baselines(first,n,s,h,site,te,doy)
        b=predict_baselines(second,n,s,h,site,te,doy)
        for key in a:
            np.testing.assert_array_equal(a[key]['daily_mean'],b[key]['daily_mean'])
            np.testing.assert_array_equal(a[key]['joint_paths'],b[key]['joint_paths'])
        self.assertEqual(first['train_target_sha256'],second['train_target_sha256'])

    def test_joint_paths_preserve_donor_pair_sum_and_site(self):
        n,s,h,site,y,tr,te,doy=self.data();y[site==1]+=20.
        bundle=fit_baselines(n,s,h,site,y,tr,doy,n_paths=256)
        result=predict_baselines(bundle,n,s,h,site,te,doy)
        for key,r in result.items():
            expected=np.where(site[te]==0,10.,50.)
            np.testing.assert_allclose(r['joint_paths'].sum(-1),np.repeat(expected[:,None],256,axis=1))
            np.testing.assert_allclose(r['daily_mean'].sum(-1),expected)

    def test_samples_are_batch_and_order_invariant(self):
        n,s,h,site,y,tr,te,doy=self.data()
        bundle=fit_baselines(n,s,h,site,y,tr,doy,n_paths=256)
        a=predict_baselines(bundle,n,s,h,site,te,doy)
        b=predict_baselines(bundle,n,s,h,site,te[::-1],doy)
        for key in a:np.testing.assert_array_equal(a[key]['joint_paths'],b[key]['joint_paths'][::-1])

    def test_missing_or_untrained_site_rejected(self):
        n,s,h,site,y,tr,te,doy=self.data()
        bundle=fit_baselines(n,s,h,site,y,tr[site[tr]==0],doy,n_paths=256)
        with self.assertRaises(ValueError):predict_baselines(bundle,n,s,h,site,te,doy)
        y[tr[0],1]=np.nan
        with self.assertRaises(ValueError):fit_baselines(n,s,h,site,y,tr,doy,n_paths=256)


if __name__=='__main__':unittest.main()
