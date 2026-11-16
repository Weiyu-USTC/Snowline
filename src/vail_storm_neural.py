"""Weather-conditioned neural semi-Markov daily sensor-event model.

Hourly observations are encoded once at issuance. A shared event decoder then
models threshold-state transition, time in state, and conditional amount. The
training likelihood factors the daily sequence using only previous labels;
forecast paths recursively use sampled states, never future observations.

The >=1-inch state refers to CAVMM's reported daily endpoint proxy. It is not a
verified meteorological storm definition or an hourly fresh-snow increment.
"""
from __future__ import annotations
import copy
import math

import numpy as np
from scipy.special import ndtr, ndtri
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from advanced_snow_models import ChannelLayerNorm, TemporalResidualBlock
from deep_dillon import seed_everything
from vail_hourly_model import HourlyScaler


THRESHOLD = 1.0
AGE_CAP = 14
HISTORY_COLUMNS = ['previous_high_state', 'log_state_age',
                   'age_known', 'previous_amount_known']


def validate_initial(initial):
    initial=np.asarray(initial,dtype='float32')
    if initial.ndim!=2 or initial.shape[1]!=4:
        raise ValueError('initial must be [N,4]: amount, age, age_known, state_known')
    return initial


def history_features(amount, age, age_known, known):
    amount=np.asarray(amount); age=np.asarray(age)
    known=np.asarray(known,dtype=bool)&np.isfinite(amount)
    age_known=np.asarray(age_known,dtype=bool)&known
    return np.stack([np.where(known,amount>=THRESHOLD,0.),
                     np.where(age_known,np.log1p(np.clip(np.nan_to_num(age,nan=0.),0.,AGE_CAP)),0.),
                     np.asarray(age_known,dtype=bool)&known,known],axis=-1).astype('float32')


def teacher_history(y,initial):
    """Past-only teacher features; missing days censor state duration.

    A newly observed amount after a gap has a known threshold state but an
    unknown full age. The observed run count is a lower bound until an observed
    state change determines an exact new age. Unknown amounts never become
    observed zero-snow states.
    """
    y=np.asarray(y,dtype='float32'); initial=validate_initial(initial)
    if y.ndim!=2 or len(y)!=len(initial):raise ValueError('y must be [N,H]')
    amount=initial[:,0].copy();age=initial[:,1].copy()
    known=(initial[:,3]>0)&np.isfinite(amount)
    age_known=(initial[:,2]>0)&known
    age=np.where(age_known,np.clip(age,1,AGE_CAP),0.)
    output=[]
    for h in range(y.shape[1]):
        output.append(history_features(amount,age,age_known,known))
        nxt=y[:,h]; nxt_known=np.isfinite(nxt)&(nxt>=0)
        same=known&nxt_known&((amount>=THRESHOLD)==(nxt>=THRESHOLD))
        changed=known&nxt_known&~same
        age=np.where(changed,1.,np.where(same&age_known,np.minimum(age+1.,AGE_CAP),0.))
        age_known=np.where(same,age_known,changed)
        amount=nxt.copy();known=nxt_known
    return np.stack(output,axis=1)


class NeuralStormModel(nn.Module):
    """Six residual blocks / twelve causal convolutions plus event decoder."""
    def __init__(self,channels,width=16,dropout=.25,weather_decay_days=7.):
        super().__init__()
        self.channels=channels;self.width=width;self.dropout=dropout
        self.weather_decay_days=float(weather_decay_days)
        self.projection=nn.Conv1d(channels,width,1)
        self.blocks=nn.Sequential(*[TemporalResidualBlock(width,3,2**i,dropout) for i in range(6)])
        self.norm=ChannelLayerNorm(width)
        self.decoder=nn.Sequential(nn.Linear(2*width+10+4,64),nn.SiLU(),nn.Dropout(dropout),
                                   nn.Linear(64,32),nn.SiLU(),nn.Linear(32,7))
        with torch.no_grad():
            # Initial high-state rate ~25%, conditional below-threshold positive
            # rate ~40%, beta mean ~1/3, and moderate positive snowfall tail.
            self.decoder[-1].bias.copy_(torch.tensor([-1.1,-.4,.1,1.1,.3,-.5,0.]))

    def encode(self,x):
        z=self.norm(self.blocks(self.projection(x.transpose(1,2))))
        return torch.cat([z[:,:,-1],z.mean(-1)],-1)

    def decode(self,encoded,future,history):
        # future[...,4] is lead/60 in the parent experiment's fixed schema.
        lead=future[...,4:5]*60
        if encoded.ndim==2 and future.ndim==3:
            encoded=encoded[:,None,:].expand(-1,future.shape[1],-1)
        summary=encoded*torch.exp(-lead/self.weather_decay_days)
        raw=self.decoder(torch.cat([summary,future,history],-1))
        transition=raw[...,0]
        low_positive=raw[...,1]
        alpha=.3+nn.functional.softplus(raw[...,2])
        beta=.3+nn.functional.softplus(raw[...,3])
        mu=.5+1.7*torch.tanh(raw[...,4]/1.7)
        sigma=.2+.8*torch.sigmoid(raw[...,5])
        # The seventh output represents an explicit duration-state interaction:
        # longer events may increase or decrease continuation probability.
        # It is bounded and multiplies observed/predicted log-age, not labels.
        duration_adjustment=.5*torch.tanh(raw[...,6])*history[...,1]
        return torch.stack([transition+duration_adjustment,low_positive,alpha,beta,mu,sigma],-1)

    def forward(self,x,future,history):
        return self.decode(self.encode(x),future,history)


def sequence_loss(params,y,mask=None,history=None):
    """Proper hurdle likelihood: >=1 truncated lognormal; <1 zero + Beta."""
    if mask is None:mask=torch.isfinite(y)&(y>=0)
    mask=mask.to(params.dtype)
    value=torch.nan_to_num(y,nan=0.)
    high=(value>=THRESHOLD).to(params.dtype)
    positive=(value>0).to(params.dtype)
    state_logits,low_logits,alpha,beta,mu,sigma=params.unbind(-1)
    state_nll=nn.functional.binary_cross_entropy_with_logits(state_logits,high,reduction='none')
    low_occurrence=nn.functional.binary_cross_entropy_with_logits(low_logits,positive,reduction='none')
    fraction=torch.clamp(value,min=1e-5,max=1-1e-5)
    beta_nll=-(alpha-1)*torch.log(fraction)-(beta-1)*torch.log1p(-fraction)
    beta_nll+=torch.lgamma(alpha)+torch.lgamma(beta)-torch.lgamma(alpha+beta)
    log_amount=torch.log(torch.clamp(value,min=THRESHOLD))
    high_nll=.5*((log_amount-mu)/sigma)**2+torch.log(sigma)+log_amount+.5*math.log(2*math.pi)
    # Normalizing mass P(log Y >= 0) = Phi(mu/sigma).
    high_nll+=torch.special.log_ndtr(mu/sigma)
    state_known=torch.ones_like(mask) if history is None else history[...,-1]
    nll=state_nll*state_known+high*high_nll+(1-high)*(low_occurrence+positive*beta_nll)
    return (nll*mask).sum()/mask.sum().clamp(min=1.)


def conditional_mean(params):
    """Mean conditional on supplied previous state, not marginal lead mean."""
    state,low,alpha,beta,mu,sigma=params.unbind(-1)
    p=torch.sigmoid(state);q=torch.sigmoid(low)
    high_mean=torch.exp(mu+.5*sigma**2+torch.special.log_ndtr(mu/sigma+sigma)-torch.special.log_ndtr(mu/sigma))
    return p*high_mean+(1-p)*q*alpha/(alpha+beta)


def train_neural(x_raw,future,y,initial,fit_idx,valid_idx=None,config=None,epochs=35,seed=20261104):
    """Fit one configuration; epoch selection uses held-out sequence NLL.

    fit_idx/valid_idx must be disjoint winter groups, supplied by the caller.
    With complete labels, teacher forcing evaluates a factorized sequence
    likelihood. Across missing-label gaps this is a masked conditional score,
    not a fully marginalized missing-state likelihood. Report daily errors from
    exact forecast_neural propagation, never teacher-forced conditional means.
    """
    config={} if config is None else config
    seed_everything(seed)
    x_raw=np.asarray(x_raw,dtype='float32');future=np.asarray(future,dtype='float32')
    y=np.asarray(y,dtype='float32');initial=validate_initial(initial)
    fit_idx=np.asarray(fit_idx,dtype=int)
    valid_idx=None if valid_idx is None else np.asarray(valid_idx,dtype=int)
    if valid_idx is not None and set(fit_idx)&set(valid_idx):raise ValueError('Fit and validation indices overlap')
    scaler=HourlyScaler().fit(x_raw[fit_idx]).fit_future(future[fit_idx])
    # Train-only circular-calendar prior for a missing state at issue time.
    phase=np.arctan2(future[fit_idx,...,0],future[fit_idx,...,1])%(2*np.pi)
    bins=np.floor(phase/(2*np.pi)*12).astype(int)
    good=np.isfinite(y[fit_idx])&(y[fit_idx]>=0)
    global_prior=float((y[fit_idx][good]>=THRESHOLD).mean())
    scaler.state_prior_monthly=np.asarray([
        ((y[fit_idx][good&(bins==month)]>=THRESHOLD).sum()+5*global_prior)/
        (np.sum(good&(bins==month))+5) for month in range(12)],dtype='float32')
    histories=teacher_history(y,initial)
    arrays=[scaler.transform(x_raw[fit_idx]),scaler.transform_future(future[fit_idx]),
            histories[fit_idx],np.nan_to_num(y[fit_idx],nan=0.),
            (np.isfinite(y[fit_idx])&(y[fit_idx]>=0)).astype('float32')]
    tensors=[torch.tensor(a,dtype=torch.float32) for a in arrays]
    loader=DataLoader(TensorDataset(*tensors),batch_size=config.get('batch_size',48),shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    model=NeuralStormModel(arrays[0].shape[-1],width=config.get('width',16),
                           dropout=config.get('dropout',.25),weather_decay_days=config.get('weather_decay_days',7.))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.001),weight_decay=config.get('weight_decay',.02))
    valid=None
    if valid_idx is not None:
        valid=[torch.tensor(a,dtype=torch.float32) for a in
               [scaler.transform(x_raw[valid_idx]),scaler.transform_future(future[valid_idx]),histories[valid_idx],
                np.nan_to_num(y[valid_idx],nan=0.),
                (np.isfinite(y[valid_idx])&(y[valid_idx]>=0)).astype('float32')]]
    best=float('inf');best_epoch=1;best_state=None;logs=[]
    for epoch in range(1,epochs+1):
        model.train();running=0.;count=0.
        for bx,bf,bh,by,bm in loader:
            optimizer.zero_grad();params=model(bx,bf,bh);loss=sequence_loss(params,by,bm,bh)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();running+=float(loss.detach())*float(bm.sum());count+=float(bm.sum())
        row={'epoch':epoch,'training_sequence_nll':running/max(count,1.)}
        if valid is not None:
            model.eval()
            with torch.no_grad():score=float(sequence_loss(model(*valid[:3]),valid[3],valid[4],valid[2]))
            row['validation_sequence_nll']=score
            if score<best-1e-5:
                best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        else:best_epoch=epoch
        logs.append(row)
        if valid is not None and epoch>=12 and epoch-best_epoch>=8:break
    if best_state is not None:model.load_state_dict(best_state)
    return model,scaler,logs,best_epoch,best


def _sample_params(params,rng):
    """Sample exact model branches without clipping sampled snowfall amounts."""
    state_logit,low_logit,alpha,beta,mu,sigma=np.moveaxis(params.astype('float64'),-1,0)
    p=1/(1+np.exp(-np.clip(state_logit,-30,30)))
    q=1/(1+np.exp(-np.clip(low_logit,-30,30)))
    high=rng.random(p.shape)<p
    low_positive=rng.random(p.shape)<q
    # Survival-space inversion is stable when the untruncated normal lies
    # almost entirely below zero; float32 lower-CDF inversion can round to1.
    survival=ndtr(mu/sigma)*(1-rng.random(p.shape))
    z=-ndtri(np.clip(survival,1e-300,1-1e-15))
    high_amount=np.exp(mu+sigma*z)
    low_amount=rng.beta(alpha,beta)*low_positive
    # Keep branch support after floating-point conversion at the boundary.
    high_amount=np.maximum(high_amount,THRESHOLD)
    low_amount=np.minimum(low_amount,float(np.nextafter(np.float32(THRESHOLD),np.float32(0.))))
    return np.where(high,high_amount,low_amount).astype('float32')


def initial_state_weights(scaler,future,initial):
    """Initial [N,2,15] distribution, with unknown duration represented by age0."""
    initial=validate_initial(initial)
    weights=np.zeros((len(initial),2,AGE_CAP+1),dtype='float64')
    phase=np.arctan2(future[:,0,0],future[:,0,1])%(2*np.pi)
    month=np.floor(phase/(2*np.pi)*12).astype(int)
    prior=getattr(scaler,'state_prior_monthly',np.full(12,.3))[month]
    for i,(amount,age,age_known,state_known) in enumerate(initial):
        if state_known>0 and np.isfinite(amount):
            state=int(amount>=THRESHOLD)
            a=int(np.clip(age,1,AGE_CAP)) if age_known>0 else 0
            weights[i,state,a]=1.
        else:
            weights[i,0,0]=1-prior[i];weights[i,1,0]=prior[i]
    return weights


def _propagate(weights,probability_high,suppress_onset=False):
    result=np.zeros_like(weights)
    for state in range(2):
        for age in range(AGE_CAP+1):
            p=probability_high[:,state,age];w=weights[:,state,age]
            same=p if state else 1-p
            next_age=0 if age==0 else min(AGE_CAP,age+1)
            result[:,state,next_age]+=w*same
            if not(suppress_onset and state==0):
                result[:,1-state,1]+=w*(1-same)
    return result


def forecast_neural(model,scaler,x_raw,future,initial,return_components=False,batch_origins=32):
    """Exact finite-state forecasts; no Monte Carlo noise in predictive means.

    Age14 pools all durations >=14 days without forcing termination. Age0
    remains unknown until the state changes. Optional mixture components are
    the previous-state weights and conditional emission parameters at each lead.
    """
    x_raw=np.asarray(x_raw,dtype='float32');future=np.asarray(future,dtype='float32')
    initial=validate_initial(initial);model.eval()
    keys=['mean','p_active','p_onset','p_onset_within']
    results={key:[] for key in keys}
    if return_components:results.update(component_weights=[],component_params=[])
    grid_state=np.repeat(np.arange(2),AGE_CAP+1)
    grid_age=np.tile(np.arange(AGE_CAP+1),2)
    grid_history=history_features(grid_state.astype(float),grid_age,grid_age>0,np.ones(len(grid_age),dtype=bool))
    # Binary representatives 0/1 provide only the previous threshold state.
    with torch.no_grad():
        for start in range(0,len(x_raw),batch_origins):
            stop=min(start+batch_origins,len(x_raw));n=stop-start;k=len(grid_age)
            f=scaler.transform_future(future[start:stop])
            encoded=model.encode(torch.tensor(scaler.transform(x_raw[start:stop]))).repeat_interleave(k,dim=0)
            hist=torch.tensor(np.tile(grid_history,(n,1)))
            weights=initial_state_weights(scaler,future[start:stop],initial[start:stop])
            no_onset=weights.copy();block={key:[] for key in keys}
            if return_components:block.update(component_weights=[],component_params=[])
            for h in range(f.shape[1]):
                ff=np.repeat(f[:,h,:],k,axis=0)
                params=model.decode(encoded,torch.tensor(ff),hist)
                cond_mean=conditional_mean(params).numpy().reshape(n,2,AGE_CAP+1)
                probability=torch.sigmoid(params[:,0]).numpy().reshape(n,2,AGE_CAP+1)
                block['mean'].append((weights*cond_mean).sum(axis=(1,2)))
                block['p_active'].append((weights*probability).sum(axis=(1,2)))
                block['p_onset'].append((weights[:,0,:]*probability[:,0,:]).sum(axis=1))
                if return_components:
                    block['component_weights'].append(weights.reshape(n,k).copy())
                    block['component_params'].append(params.numpy().reshape(n,k,6))
                weights=_propagate(weights,probability)
                no_onset=_propagate(no_onset,probability,suppress_onset=True)
                # Float32 priors can sum to 1+epsilon after promotion; keep
                # the reported probability in its exact mathematical support.
                block['p_onset_within'].append(np.clip(1-no_onset.sum(axis=(1,2)),0.,1.))
            for key in results:results[key].append(np.stack(block[key],axis=1))
    return {key:np.concatenate(blocks,axis=0) for key,blocks in results.items()}


def simulate_neural(model,scaler,x_raw,future,initial,n_paths=256,seed=20261104,batch_origins=32):
    """Return [origin,path,lead] coherent samples using only issue-time inputs."""
    x_raw=np.asarray(x_raw,dtype='float32');future=np.asarray(future,dtype='float32')
    initial=validate_initial(initial)
    if len(x_raw)!=len(initial) or len(future)!=len(initial):raise ValueError('Origin counts differ')
    rng=np.random.default_rng(seed);model.eval();blocks=[]
    with torch.no_grad():
        for start in range(0,len(x_raw),batch_origins):
            stop=min(start+batch_origins,len(x_raw));n=stop-start
            encoded=model.encode(torch.tensor(scaler.transform(x_raw[start:stop])))
            encoded=encoded.repeat_interleave(n_paths,dim=0)
            f=scaler.transform_future(future[start:stop])
            init=np.repeat(initial[start:stop],n_paths,axis=0)
            # Draw a missing initial state from the same train-only prior used
            # by exact propagation; preserve unknown age until a transition.
            initial_weights=initial_state_weights(scaler,future[start:stop],initial[start:stop]).reshape(n,-1)
            cdf=np.cumsum(initial_weights,axis=1)
            choices=np.sum(rng.random((n,n_paths,1))>cdf[:,None,:],axis=-1).reshape(-1)
            state=choices//(AGE_CAP+1);age=(choices%(AGE_CAP+1)).astype(float)
            amount=state.astype(float);known=np.ones(len(state),dtype=bool);age_known=age>0
            path=np.empty((n*n_paths,f.shape[1]),dtype='float32')
            for h in range(f.shape[1]):
                ff=np.repeat(f[:,h,:],n_paths,axis=0)
                history=history_features(amount,age,age_known,known)
                params=model.decode(encoded,torch.tensor(ff),torch.tensor(history)).numpy()
                nxt=_sample_params(params,rng)
                same=known&((amount>=THRESHOLD)==(nxt>=THRESHOLD))
                changed=known&~same
                age=np.where(changed,1.,np.where(same&age_known,np.minimum(age+1.,AGE_CAP),0.))
                age_known=np.where(same,age_known,changed)
                amount=nxt;known=np.ones_like(known,dtype=bool);path[:,h]=nxt
            blocks.append(path.reshape(n,n_paths,f.shape[1]))
    return np.concatenate(blocks,axis=0)


def checkpoint_payload(model,scaler,config):
    return {'state_dict':model.state_dict(),'scaler':scaler.as_dict(),'config':config,
            'channels':model.channels,'width':model.width,'dropout':model.dropout,
            'weather_decay_days':model.weather_decay_days,'threshold_inches':THRESHOLD,
            'state_prior_monthly':getattr(scaler,'state_prior_monthly',np.full(12,.3)).tolist(),
            'target':'CAVMM daily04 sensor threshold state; not verified storm labels',
            'architecture':'168-hour causal TCN; autoregressive age-dependent threshold-event and conditional amount decoder'}


def restore_neural(checkpoint):
    model=NeuralStormModel(checkpoint['channels'],checkpoint['width'],checkpoint['dropout'],checkpoint['weather_decay_days'])
    model.load_state_dict(checkpoint['state_dict'])
    scaler=HourlyScaler()
    for name in ['mean','std','future_min','future_max']:
        if name in checkpoint['scaler']:setattr(scaler,name,np.asarray(checkpoint['scaler'][name],dtype='float32'))
    scaler.state_prior_monthly=np.asarray(checkpoint.get('state_prior_monthly',np.full(12,.3)),dtype='float32')
    return model,scaler
