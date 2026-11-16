from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from scipy.stats import gamma
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from neural48_models import (Neural48Config, Neural48Model, checkpoint_payload,
    restore_model, distribution_quantiles, sample_joint_paths,
    zero_inflated_gamma_nll, forecast_loss, parameter_count)


class Neural48Architecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def inputs(self, batch=3):
        torch.manual_seed(4)
        nwp=torch.randn(batch,48,6)
        radar=torch.rand(batch,9,5,16,16)
        local=torch.randn(batch,9,16)
        site=torch.arange(batch)%4
        static=torch.randn(batch,4)
        return nwp,radar,local,site,static

    def test_forward_finite_and_weather_only_ablation_ignores_radar(self):
        args=self.inputs()
        for variant in ['weather_only','weather_radar']:
            model=Neural48Model(Neural48Config(6,4,variant=variant)).eval()
            p=model(*args)
            self.assertEqual(tuple(p['mean'].shape),(3,2))
            self.assertTrue(torch.isfinite(p['mean']).all())
            self.assertTrue((p['mean']>=0).all())
            if variant=='weather_only':
                changed=list(args);changed[1]=torch.full_like(args[1],float('nan'));changed[2]=None
                torch.testing.assert_close(p['mean'],model(*changed)['mean'])

    def test_first_day_does_not_consume_second_day_forecast_hours(self):
        model=Neural48Model(Neural48Config(6,4,variant='weather_only')).eval()
        args=list(self.inputs());expected=model(*args)['mean'][:,0]
        args[0]=args[0].clone();args[0][:,24:]+=100
        torch.testing.assert_close(expected,model(*args)['mean'][:,0],rtol=0,atol=0)

    def test_masked_targets_have_zero_gradient_contribution(self):
        model=Neural48Model(Neural48Config(6,4,variant='weather_only')).eval()
        pred=model(*self.inputs())
        target=torch.tensor([[0.,float('nan')],[1.,2.],[3.,0.]])
        losses=forecast_loss(pred,target)
        self.assertEqual(int(losses['valid_targets']),5)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(float(zero_inflated_gamma_nll(pred,target)[0,1]),0.)

    def test_nll_matches_gamma_and_zero_mass(self):
        probability=torch.tensor([[.25,.6]],dtype=torch.float64)
        params={'probability':probability,'occurrence_logit':torch.logit(probability),
            'concentration':torch.tensor([[2.,3.]],dtype=torch.float64),
            'rate':torch.tensor([[1.,2.]],dtype=torch.float64),'mean':torch.tensor([[.5,.9]])}
        actual=zero_inflated_gamma_nll(params,torch.tensor([[0.,1.25]],dtype=torch.float64)).numpy()
        expected=[-np.log(.75),-np.log(.6)-gamma.logpdf(1.25,a=3.,scale=.5)]
        np.testing.assert_allclose(actual[0],expected)

    def test_quantiles_monotone_and_sample_mean_matches_exact_mean(self):
        params={'probability':np.array([[.3,.6]]),'concentration':np.array([[2.,3.]]),'rate':np.array([[1.,1.5]])}
        q=distribution_quantiles(params,[.1,.5,.9,.99])
        self.assertTrue(np.isfinite(q).all());self.assertTrue((q>=0).all())
        self.assertTrue((np.diff(q,axis=0)>=0).all())
        paths=sample_joint_paths(params,n_paths=100000,seed=52)
        exact=params['probability']*params['concentration']/params['rate']
        np.testing.assert_allclose(paths.mean(1),exact,rtol=.03,atol=.01)
        same=sample_joint_paths(params,n_paths=40,seed=9)
        np.testing.assert_array_equal(same,sample_joint_paths(params,n_paths=40,seed=9))

    def test_saved_model_replays_identically(self):
        model=Neural48Model(Neural48Config(6,4)).eval();args=self.inputs()
        original=model(*args)['mean']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights.pt';torch.save(checkpoint_payload(model),path)
            restored=restore_model(path)
        torch.testing.assert_close(original,restored(*args)['mean'],rtol=0,atol=0)
        self.assertGreater(parameter_count(model),50000)
        self.assertLess(parameter_count(model),150000)


if __name__=='__main__': unittest.main()
