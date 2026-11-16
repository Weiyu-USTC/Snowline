"""Daily-weather neural duration model for independent mountain snowfall targets.

The event decoder and exact finite-state propagation are shared with the Vail
experiment. This adapter replaces hourly preprocessing with daily preprocessing
and records the new target/input scope. No Vail or Dillon model weights are used.

Inputs are origin-known regional weather [origin, context_day, variable], future
calendar/lagged ENSO [origin, lead, 10], and target histories used only by the
conditional fitting likelihood. Forecast propagation never receives future
observed snowfall or weather. A missing issue-time target state is integrated
against a prior estimated from training labels only.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.special import expit, log_ndtr
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deep_dillon import seed_everything
from vail_hourly_model import HourlyScaler
from vail_storm_neural import (
    AGE_CAP, THRESHOLD, NeuralStormModel, forecast_neural, simulate_neural,
    sequence_loss, teacher_history, validate_initial,
)


class DailyScaler(HourlyScaler):
    """Training-only scaling plus missingness and one daily recency coordinate.

Inherited future guards hold training-constant covariates fixed and clip RONI
to the fitted range. There are no artificial hourly cycles on these daily data.
Missing values receive the fitted context mean, never a future interpolation.
"""

    def fit(self, x):
        x=np.asarray(x,dtype='float32')
        if x.ndim!=3 or min(x.shape)<1:
            raise ValueError('Daily context must have shape [N,days,C]')
        self.context_days=int(x.shape[1])
        self.raw_features=int(x.shape[2])
        super().fit(x)
        # Keep an entirely unobserved/constant training weather channel inert.
        # Missingness itself remains explicit; its fitted support is recorded.
        finite=np.isfinite(x)
        self.observed_counts=finite.sum(axis=(0,1)).astype(int)
        flattened=np.where(finite,x,np.nan).reshape(-1,x.shape[-1])
        self.input_constant=np.asarray([
            (np.nanmax(col)-np.nanmin(col)<1e-7) if np.isfinite(col).any() else True
            for col in flattened.T],dtype=bool)
        return self

    def transform(self, x):
        x=np.asarray(x,dtype='float32')
        if x.ndim!=3 or x.shape[1:]!=(self.context_days,self.raw_features):
            raise ValueError('Daily context shape differs from fitted preprocessing')
        missing=~np.isfinite(x)
        scaled=(x-self.mean)/self.std
        scaled[missing]=0.
        scaled[...,self.input_constant]=0.
        # 0 is the most recent supplied weather day; -1 the oldest. The caller
        # applies its observation/publication lag before constructing this array.
        recency=np.broadcast_to(np.linspace(-1.,0.,self.context_days,dtype='float32')[None,:,None],
                                (len(x),self.context_days,1))
        return np.concatenate([scaled,missing.astype('float32'),recency],axis=-1).astype('float32')

    def as_dict(self):
        result=super().as_dict()
        result.update(kind='daily_weather',context_days=self.context_days,raw_features=self.raw_features,
                      observed_counts=self.observed_counts.tolist(),input_constant=self.input_constant.tolist(),
                      time_features='One context-recency coordinate from -1 to0; no within-day hour cycles',
                      policy='Training-context mean/std; mean imputation with missingness; constant weather channels inert; no future fill')
        return result


def train_neural(x_raw,future,y,initial,fit_idx,valid_idx=None,config=None,epochs=35,seed=20261104):
    """Train a fresh model; optional inner validation chooses stopping epoch.

This uses a masked conditional sequence likelihood across missing labels, not a
fully marginalized missing-state likelihood. Report forecasting errors using
forecast_neural's free-running recursion, never the teacher-forced fit means.
The caller supplies disjoint whole-winter groups and decides whether issue-time
target states are known; the current three-resort experiment makes them unknown.
"""
    config={} if config is None else dict(config)
    seed_everything(seed)
    x_raw=np.asarray(x_raw,dtype='float32');future=np.asarray(future,dtype='float32')
    y=np.asarray(y,dtype='float32');initial=validate_initial(initial)
    if len(x_raw)!=len(y) or len(future)!=len(y) or len(initial)!=len(y):
        raise ValueError('Origin counts differ')
    if future.ndim!=3 or future.shape[1]!=y.shape[1] or future.shape[2]!=10:
        raise ValueError('Future schema must be [N,H,10] with lead/60 in column4 and RONI in column6')
    fit_idx=np.asarray(fit_idx,dtype=int)
    valid_idx=None if valid_idx is None else np.asarray(valid_idx,dtype=int)
    if not len(fit_idx):raise ValueError('At least one fitting origin is required')
    if valid_idx is not None and not len(valid_idx):raise ValueError('Validation indices cannot be empty')
    if valid_idx is not None and set(fit_idx)&set(valid_idx):
        raise ValueError('Fit and validation indices overlap')
    good=np.isfinite(y[fit_idx])&(y[fit_idx]>=0)
    if not good.any():raise ValueError('No observed fitting labels')
    scaler=DailyScaler().fit(x_raw[fit_idx]).fit_future(future[fit_idx])
    phase=np.arctan2(future[fit_idx,...,0],future[fit_idx,...,1])%(2*np.pi)
    bins=np.floor(phase/(2*np.pi)*12).astype(int)
    global_prior=float((y[fit_idx][good]>=THRESHOLD).mean())
    scaler.state_prior_monthly=np.asarray([
        ((y[fit_idx][good&(bins==month)]>=THRESHOLD).sum()+5*global_prior)/
        (np.sum(good&(bins==month))+5) for month in range(12)],dtype='float32')
    histories=teacher_history(y,initial)

    def arrays(indices):
        return [scaler.transform(x_raw[indices]),scaler.transform_future(future[indices]),
                histories[indices],np.nan_to_num(y[indices],nan=0.),
                (np.isfinite(y[indices])&(y[indices]>=0)).astype('float32')]

    train_arrays=arrays(fit_idx)
    tensors=[torch.tensor(a,dtype=torch.float32) for a in train_arrays]
    loader=DataLoader(TensorDataset(*tensors),batch_size=config.get('batch_size',64),shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    model=NeuralStormModel(train_arrays[0].shape[-1],width=config.get('width',16),
                          dropout=config.get('dropout',.25),weather_decay_days=config.get('weather_decay_days',7.))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.001),weight_decay=config.get('weight_decay',.02))
    validation=None if valid_idx is None else [torch.tensor(a,dtype=torch.float32) for a in arrays(valid_idx)]
    best=float('inf');best_epoch=1;best_state=None;logs=[]
    for epoch in range(1,epochs+1):
        model.train();running=0.;count=0.
        for bx,bf,bh,by,bm in loader:
            optimizer.zero_grad();params=model(bx,bf,bh);loss=sequence_loss(params,by,bm,bh)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite daily event-model training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();observed=float(bm.sum());running+=float(loss.detach())*observed;count+=observed
        row={'epoch':epoch,'training_sequence_nll':running/max(count,1.)}
        if validation is not None:
            model.eval();total=0.;observed_total=0.
            # Bounded validation batches make the same adapter suitable for the
            # longer NOAA archives without retaining an entire computation graph.
            with torch.no_grad():
                for start in range(0,len(valid_idx),128):
                    bx,bf,bh,by,bm=[a[start:start+128] for a in validation]
                    observed=float(bm.sum())
                    total+=float(sequence_loss(model(bx,bf,bh),by,bm,bh))*observed
                    observed_total+=observed
            score=total/max(observed_total,1.)
            if not np.isfinite(score):raise FloatingPointError('Nonfinite validation loss')
            row['validation_sequence_nll']=score
            if score<best-1e-5:
                best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        else:best_epoch=epoch
        logs.append(row)
        if validation is not None and epoch>=config.get('min_epochs',12) and epoch-best_epoch>=config.get('patience',8):break
    if best_state is not None:model.load_state_dict(best_state)
    return model,scaler,logs,best_epoch,best


def exceedance_probability(prediction,threshold=5.):
    """Exact marginal probability for thresholds >=1 inch from saved components."""
    if not np.isfinite(threshold) or threshold<THRESHOLD:
        raise ValueError('This tail helper supports finite thresholds >=1 inch')
    params=np.asarray(prediction['component_params'],dtype=np.float64)
    weights=np.asarray(prediction['component_weights'],dtype=np.float64)
    mu,sigma=params[...,4],params[...,5]
    tail=np.exp(log_ndtr((mu-np.log(threshold))/sigma)-log_ndtr(mu/sigma))
    return np.clip((weights*expit(params[...,0])*tail).sum(axis=-1),0.,1.)


def checkpoint_payload(model,scaler,config):
    return {'state_dict':model.state_dict(),'scaler':scaler.as_dict(),'config':dict(config),
            'channels':model.channels,'width':model.width,'dropout':model.dropout,
            'weather_decay_days':model.weather_decay_days,'threshold_inches':THRESHOLD,
            'state_prior_monthly':scaler.state_prior_monthly.tolist(),
            'target':'Site-specific daily snowfall target; >=1-inch observed state with unknown initial target when requested',
            'architecture':'Daily-weather causal TCN with six residual blocks/twelve convolutions; autoregressive age-dependent event/amount decoder',
            'model_family':'daily_neural_duration','context_days':scaler.context_days}


def restore_neural(checkpoint):
    if checkpoint.get('model_family')!='daily_neural_duration':
        raise ValueError('Checkpoint is not a daily-weather neural duration model')
    model=NeuralStormModel(checkpoint['channels'],checkpoint['width'],checkpoint['dropout'],checkpoint['weather_decay_days'])
    model.load_state_dict(checkpoint['state_dict'])
    scaler=DailyScaler();record=checkpoint['scaler']
    for name in ['mean','std','future_min','future_max']:
        if name in record:setattr(scaler,name,np.asarray(record[name],dtype='float32'))
    scaler.context_days=int(record['context_days']);scaler.raw_features=int(record['raw_features'])
    scaler.observed_counts=np.asarray(record['observed_counts'],dtype=int)
    scaler.input_constant=np.asarray(record['input_constant'],dtype=bool)
    scaler.state_prior_monthly=np.asarray(checkpoint['state_prior_monthly'],dtype='float32')
    return model,scaler
