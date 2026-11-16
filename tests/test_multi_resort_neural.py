"""Daily preprocessing, information-boundary and restoration checks."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import multi_resort_neural as daily


def fixture():
    rng=np.random.default_rng(117)
    x=rng.normal(size=(6,60,3)).astype('float32')
    x[0,4,1]=np.nan
    h=8;phase=np.arange(6)[:,None]*.15+np.arange(h)[None,:]*.017
    future=np.zeros((6,h,10),dtype='float32')
    future[...,:4]=np.stack([np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)],axis=-1)
    future[...,4]=np.arange(1,h+1)/60;future[...,5]=future[...,4]**2
    future[...,6]=np.linspace(-.4,.4,6)[:,None]
    y=np.array([[0,0,.2,2,3,0,0,.4],[0,1,4,2,0,0,.1,0],
                [2,1,0,.1,0,3,4,2],[0,0,0,0,.2,1,2,0],
                [9,9,9,9,9,9,9,9],[8,8,8,8,8,8,8,8]],dtype='float32')
    initial=np.tile([np.nan,0,0,0],(6,1)).astype('float32')
    return x,future,y,initial


class DailyNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.future,cls.y,cls.initial=fixture()
        cls.config=dict(width=8,dropout=0.,batch_size=4,lr=.001,weight_decay=.02,weather_decay_days=3.)
        cls.model,cls.scaler,cls.logs,_,_=daily.train_neural(cls.x,cls.future,cls.y,cls.initial,
                                                        np.arange(4),config=cls.config,epochs=2,seed=919)

    def test_daily_scaler_has_recency_not_an_hour_cycle(self):
        scaler=daily.DailyScaler().fit(self.x[:4]).fit_future(self.future[:4])
        transformed=scaler.transform(self.x[:1])
        self.assertEqual(transformed.shape,(1,60,7))
        np.testing.assert_allclose(transformed[0,:,-1],np.linspace(-1,0,60),atol=1e-7)
        self.assertEqual(transformed[0,4,1],0.)
        self.assertEqual(transformed[0,4,4],1.)
        with self.assertRaises(ValueError):scaler.transform(self.x[:1,:59])

    def test_training_constant_weather_channel_is_inert(self):
        x=self.x[:4].copy();x[...,2]=5.
        scaler=daily.DailyScaler().fit(x)
        x[0,:,2]=10000.
        np.testing.assert_array_equal(scaler.transform(x)[...,2],0.)

    def test_future_covariates_use_only_fitted_support(self):
        future=self.future[:4].copy();future[...,7:]=0.
        scaler=daily.DailyScaler().fit(self.x[:4]).fit_future(future)
        test=future[:1].copy();test[...,6]=99.;test[...,7]=1.
        guarded=scaler.transform_future(test)
        np.testing.assert_array_equal(guarded[...,7],0.)
        np.testing.assert_allclose(guarded[...,6],future[...,6].max())

    def test_heldout_weather_and_labels_cannot_change_fit(self):
        x=self.x.copy();y=self.y.copy();future=self.future.copy()
        x[4:]=1e5;y[4:]=100.;future[4:,...,6]=-99.
        model,scaler,_,_,_=daily.train_neural(x,future,y,self.initial,np.arange(4),
                                            config=self.config,epochs=2,seed=919)
        np.testing.assert_array_equal(scaler.mean,self.scaler.mean)
        np.testing.assert_array_equal(scaler.state_prior_monthly,self.scaler.state_prior_monthly)
        for key,value in model.state_dict().items():
            np.testing.assert_array_equal(value.numpy(),self.model.state_dict()[key].numpy())

    def test_checkpoint_restores_exact_unknown_state_forecasts_and_paths(self):
        checkpoint=daily.checkpoint_payload(self.model,self.scaler,self.config)
        restored,scaler=daily.restore_neural(checkpoint)
        args=(self.x[4:],self.future[4:],self.initial[4:])
        before=daily.forecast_neural(self.model,self.scaler,*args,return_components=True)
        after=daily.forecast_neural(restored,scaler,*args,return_components=True)
        for key in before:np.testing.assert_array_equal(before[key],after[key])
        paths=daily.simulate_neural(self.model,self.scaler,*args,n_paths=128,seed=25)
        other=daily.simulate_neural(restored,scaler,*args,n_paths=128,seed=25)
        np.testing.assert_array_equal(paths,other)
        self.assertTrue(np.isfinite(paths).all() and (paths>=0).all())
        np.testing.assert_allclose(daily.exceedance_probability(before,1.),before['p_active'],atol=1e-7)
        self.assertTrue((daily.exceedance_probability(before,5.)<=before['p_active']+1e-7).all())


if __name__=='__main__':unittest.main()
