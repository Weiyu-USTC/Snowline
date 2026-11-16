"""Independent event-model probability, censoring and support checks."""
import copy
import importlib.util
import io
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import vail_storm_statistical as statistical

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Neural event checks require optional torch dependency')
import torch
from torch import nn
import vail_storm_neural as neural
from vail_hourly_model import HourlyScaler


def calendar(n,h=1):
    phase=np.linspace(.1,1.2,n*h).reshape(n,h)
    return np.stack([np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)],axis=-1)


class StatisticalEventTests(unittest.TestCase):
    def fit_model(self):
        rng=np.random.default_rng(19)
        summary=rng.normal(size=(150,3));summary[:,2]=np.nan
        current=rng.choice([0.,.3,2.,5.],150)
        target=rng.choice([0.,.3,2.,5.],150)
        ages=rng.integers(0,20,150)
        model=statistical.EventModel(kind='semi_markov',regularization=.1)
        model.fit(summary,calendar(150)[:,0],current,ages,target)
        return model,summary,current,ages,target

    def test_unknown_and_tail_age_are_not_invented_or_forced_to_end(self):
        np.testing.assert_array_equal(statistical.advance_age([0,1,1,0],[0,14,14,0],[0,1,0,1]),[0,14,1,1])
        np.testing.assert_array_equal(statistical.age_index([np.nan,0,1,14,25]),[0,0,1,14,14])

    def test_forward_probabilities_and_means_match_analytic_constant_hazard(self):
        model,*_=self.fit_model()
        model._all_hazards=lambda summary,cal:np.full((len(summary),cal.shape[1],30),.3)
        model._pool=lambda phase,state,age:np.array([2. if state else .2])
        model._initial_probability=lambda phase:.6
        current=np.array([0.,2.,np.nan]);age=np.array([5,14,0])
        result=model.forecast(np.zeros((3,3)),calendar(3,8),current,age,n_paths=30000,seed=7)
        np.testing.assert_allclose(result['p_active'],.3,atol=1e-12)
        np.testing.assert_allclose(result['mean'],.3*2+.7*.2,atol=1e-12)
        np.testing.assert_allclose(result['p_onset'][:,0],[.3,0.,.12],atol=1e-12)
        np.testing.assert_allclose(result['p_onset'][:,1:],.21,atol=1e-12)
        np.testing.assert_allclose(result['paths'].mean(axis=1),result['mean'],atol=.025)
        np.testing.assert_allclose(result['onset_paths'].mean(axis=1),result['p_onset'],atol=.015)
        np.testing.assert_array_equal(result['paths']>=1,result['active_paths'])
        np.testing.assert_array_equal(result['onset_paths'][:,:,1:],(~result['active_paths'][:,:,:-1])&result['active_paths'][:,:,1:])
        self.assertTrue((result['p_onset']<=result['p_active']+1e-12).all())

    def test_sample_seed_cannot_change_exact_probabilities_or_means(self):
        model,*_=self.fit_model();summary=np.zeros((2,3));cal=calendar(2,7)
        a=model.forecast(summary,cal,[0.,2.],[0,14],n_paths=5,seed=7)
        b=model.forecast(summary,cal,[0.,2.],[0,14],n_paths=50,seed=9)
        for name in ['mean','p_active','p_onset']:np.testing.assert_array_equal(a[name],b[name])

    def test_missing_labels_do_not_change_fit_or_emissions(self):
        model,summary,current,ages,target=self.fit_model()
        altered=statistical.EventModel(kind='semi_markov',regularization=.1)
        altered.fit(np.vstack([summary,np.full((1,3),1e8)]),np.concatenate([calendar(150)[:,0],calendar(1)[:,0]]),
                    np.r_[current,1e8],np.r_[ages,100],np.r_[target,np.nan])
        np.testing.assert_array_equal(model.donor_y,altered.donor_y)
        np.testing.assert_array_equal(model.base_logit.coefficient,altered.base_logit.coefficient)
        np.testing.assert_array_equal(model.weather_logit.mean,altered.weather_logit.mean)
        self.assertEqual(model.training_transitions,altered.training_transitions)

    def test_fit_copies_donors_and_preserves_zero_and_subthreshold_amounts(self):
        model,summary,current,ages,target=self.fit_model()
        before=model.forecast(np.zeros((2,3)),calendar(2,5),[0.,2.],[0,14],n_paths=1000,seed=12)
        target[:]=999.;summary[:]=999.;current[:]=999.;ages[:]=999
        after=model.forecast(np.zeros((2,3)),calendar(2,5),[0.,2.],[0,14],n_paths=1000,seed=12)
        np.testing.assert_array_equal(before['paths'],after['paths'])
        self.assertTrue((after['paths']==0).any())
        self.assertTrue(((after['paths']>0)&(after['paths']<1)).any())
        self.assertTrue(set(np.unique(after['paths'])).issubset(set(np.float32([0.,.3,2.,5.]))))


class TinyStateDecoder(nn.Module):
    """Nonconstant state/age emissions make a sensitive forward-recursion fixture."""
    def encode(self,x):return torch.zeros((len(x),2))
    def decode(self,encoded,future,history):
        result=torch.zeros((len(history),6),dtype=history.dtype)
        result[:,0]=-.8+.7*history[:,0]-.1*history[:,1]
        result[:,1]=-.4
        result[:,2]=1.2
        result[:,3]=2.
        result[:,4]=.4+.1*history[:,0]
        result[:,5]=.5
        return result


class NeuralEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self,n=3,h=8):
        rng=np.random.default_rng(20)
        x=rng.normal(size=(n,168,2)).astype('float32')
        f=np.zeros((n,h,10),dtype='float32');f[:,:,:4]=calendar(n,h)
        f[:,:,4]=np.arange(1,h+1)/60;f[:,:,5]=f[:,:,4]**2
        scaler=HourlyScaler().fit(x).fit_future(f)
        scaler.state_prior_monthly=np.full(12,.4,dtype='float32')
        initial=np.array([[0.,5.,1.,1.],[2.,14.,1.,1.],[np.nan,0.,0.,0.]],dtype='float32')[:n]
        return x,f,scaler,initial

    def test_teacher_history_never_reads_current_or_future_target(self):
        y=np.array([[2.,0.,3.,.4,0.,2.]],dtype='float32')
        initial=np.array([[.3,4.,1.,1.]],dtype='float32')
        original=neural.teacher_history(y,initial)
        changed=y.copy();changed[:,3:]=100.
        revised=neural.teacher_history(changed,initial)
        np.testing.assert_array_equal(original[:,:4],revised[:,:4])
        self.assertFalse(np.array_equal(original[:,4:],revised[:,4:]))

    def test_missing_gap_censors_state_and_age_until_observed_change(self):
        y=np.array([[2.,np.nan,2.,2.,.3,0.]],dtype='float32')
        history=neural.teacher_history(y,np.array([[0.,5.,1.,1.]],dtype='float32'))[0]
        np.testing.assert_array_equal(history[2],[0.,0.,0.,0.])
        np.testing.assert_array_equal(history[3],[1.,0.,0.,1.])
        np.testing.assert_array_equal(history[4],[1.,0.,0.,1.])
        np.testing.assert_allclose(history[5],[0.,np.log(2),1.,1.],atol=1e-7)

    def test_masked_amount_cannot_change_loss_or_gradients(self):
        base=torch.tensor([-.8,-.4,1.2,2.,.5,.6]).repeat(2,5,1)
        y=torch.tensor([[0.,.3,1.,3.,.5],[2.,0.,.5,5.,.1]])
        mask=torch.ones_like(y);mask[:,2]=0.
        bad=y.clone();bad[:,2]=1e5
        a=base.clone().requires_grad_();b=base.clone().requires_grad_()
        loss_a=neural.sequence_loss(a,y,mask);loss_b=neural.sequence_loss(b,bad,mask)
        loss_a.backward();loss_b.backward()
        torch.testing.assert_close(loss_a,loss_b,rtol=0,atol=0)
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_low_and_high_emission_support_survives_float_rounding(self):
        low=np.tile([-30.,30.,.5,.3,-1.2,.2],(100000,1)).astype('float32')
        draws=neural._sample_params(low,np.random.default_rng(40))
        self.assertTrue((draws>=0).all());self.assertTrue((draws<1).all())
        high=low.copy();high[:,0]=30.
        draws=neural._sample_params(high,np.random.default_rng(41))
        self.assertTrue(np.isfinite(draws).all());self.assertTrue((draws>=1).all())

    def test_forward_means_and_probabilities_match_coherent_samples(self):
        x,f,scaler,initial=self.fixture();model=TinyStateDecoder().eval()
        exact=neural.forecast_neural(model,scaler,x,f,initial,return_components=True)
        paths=neural.simulate_neural(model,scaler,x,f,initial,n_paths=25000,seed=22)
        np.testing.assert_allclose(paths.mean(1),exact['mean'],atol=.035)
        np.testing.assert_allclose((paths>=1).mean(1),exact['p_active'],atol=.015)
        np.testing.assert_allclose(exact['component_weights'].sum(-1),1.,atol=2e-6)
        self.assertTrue((exact['p_onset']<=exact['p_active']+1e-8).all())
        self.assertEqual(exact['p_onset'][1,0],0.)
        self.assertTrue((np.diff(exact['p_onset_within'],axis=1)>=-1e-8).all())
        np.testing.assert_allclose(exact['p_onset_within'][:,0],exact['p_onset'][:,0],atol=1e-7)
        for i in [0,1]:
            prev=np.concatenate([np.full((paths.shape[1],1),initial[i,0]>=1),paths[i,:,:-1]>=1],axis=1)
            onset=(~prev)&(paths[i]>=1)
            np.testing.assert_allclose(np.maximum.accumulate(onset,axis=1).mean(0),exact['p_onset_within'][i],atol=.02)

    def test_tail_state_persists_without_forced_exit_and_unknown_age_stays_unknown(self):
        weights=np.zeros((1,2,15));weights[0,1,14]=1.
        result=neural._propagate(weights,np.ones_like(weights))
        self.assertEqual(result[0,1,14],1.)
        weights[:]=0;weights[0,1,0]=1.
        result=neural._propagate(weights,np.ones_like(weights))
        self.assertEqual(result[0,1,0],1.)

    def test_cumulative_onset_support_survives_rounded_float32_prior(self):
        class ZeroOnsetDecoder(TinyStateDecoder):
            def decode(self,encoded,future,history):
                params=super().decode(encoded,future,history)
                params[:,0]=-1000.  # Exact zero probability after sigmoid.
                return params
        x,f,scaler,initial=self.fixture()
        initial[:]=[np.nan,0.,0.,0.]
        scaler.state_prior_monthly[:]=np.float32(.4)
        weights=neural.initial_state_weights(scaler,f,initial)
        # Promoting the two independently rounded float32 masses can exceed1.
        self.assertTrue((weights.sum(axis=(1,2))>1.).all())
        exact=neural.forecast_neural(ZeroOnsetDecoder(),scaler,x,f,initial)
        np.testing.assert_array_equal(exact['p_onset'],0.)
        np.testing.assert_array_equal(exact['p_onset_within'],0.)

    def test_checkpoint_restores_future_guard_prior_and_exact_predictions(self):
        x,f,scaler,initial=self.fixture()
        f[:,:,6]=1.7;f[:,:,7]=1.
        model=neural.NeuralStormModel(6,width=4,dropout=.2).eval()
        before=neural.forecast_neural(model,scaler,x,f,initial)
        buffer=io.BytesIO();torch.save(neural.checkpoint_payload(model,scaler,{}),buffer);buffer.seek(0)
        restored,restored_scaler=neural.restore_neural(torch.load(buffer,weights_only=True))
        after=neural.forecast_neural(restored,restored_scaler,x,f,initial)
        for key in before:np.testing.assert_allclose(before[key],after[key],atol=1e-7,rtol=1e-6)
        np.testing.assert_array_equal(scaler.state_prior_monthly,restored_scaler.state_prior_monthly)
        self.assertTrue((restored_scaler.transform_future(f)[:,:,7]==0).all())


if __name__=='__main__':unittest.main()
