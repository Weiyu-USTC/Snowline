from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from train_neural48 import fit_preprocessing, transform, clean_history, unavailable_history, metrics, predict_neural
from neural48_models import Neural48Config, Neural48Model

class TrainingContract(unittest.TestCase):
    def arrays(self,n=12):
        rng=np.random.default_rng(12)
        return dict(nwp=rng.normal(size=(n,48,23)),radar=rng.uniform(size=(n,9,5,16,16)),local=rng.normal(size=(n,9,8)),
            static=rng.normal(size=(n,8)),history=rng.uniform(size=(n,17)),site=np.arange(n)%4,y=rng.uniform(size=(n,2)))

    def test_preprocessing_ignores_all_holdout_values(self):
        a=self.arrays();idx=np.arange(8);p=fit_preprocessing(a,idx)
        changed={k:v.copy() for k,v in a.items()}
        for key in ['nwp','local','static','history','y']:changed[key][8:]=1e12
        self.assertEqual(p,fit_preprocessing(changed,idx))

    def test_missing_history_has_explicit_masks_and_no_stale_amount(self):
        a=self.arrays();p=fit_preprocessing(a,np.arange(8));a['history']=unavailable_history(a['history'])
        x=transform(a,p)
        self.assertTrue(np.isfinite(x['static']).all())
        np.testing.assert_allclose(x['static'],x['static_no_history'])
        h=a['history'].copy();h[:,-2]=500.;h[:,-1]=60.
        self.assertTrue(np.isnan(clean_history(h)[:,-2]).all())
        np.testing.assert_array_equal(x['static'][:,25+8+np.array([0,1,3,4,6,7,9,10,12,13,15])],1.)

    def test_inference_uses_site_scaling_and_hourly_fallback_contract(self):
        torch.set_num_threads(1);a=self.arrays();p=fit_preprocessing(a,np.arange(8))
        model=Neural48Model(Neural48Config(46,50,variant='weather_only')).eval()
        b={'preprocessing':p,'models':{'weather_only':[{'model':model,'rho':[0.]*4,'seed':7}]},
           'selection':{'selected_family_by_site':dict.fromkeys(['alta','vail','mansfield','jay'],'weather_only')}}
        pred=predict_neural(b,a,n_paths=100)
        self.assertEqual(pred['daily_mean'].shape,(12,2));self.assertEqual(pred['joint_paths'].shape,(12,100,2))
        self.assertTrue(np.isfinite(pred['daily_mean']).all())
        self.assertTrue((pred['joint_paths']>=0).all())
        self.assertIn('no independently supervised hourly',pred['target_definition'])

    def test_crps_degenerate_distribution_equals_mae(self):
        y=np.array([0.,2.,4.]);mean=np.array([1.,2.,2.]);paths=np.repeat(mean[:,None],300,axis=1)
        result=metrics(y,mean,paths)
        self.assertAlmostEqual(result['mae_inches'],result['crps_inches'],places=10)

if __name__=='__main__':unittest.main()
