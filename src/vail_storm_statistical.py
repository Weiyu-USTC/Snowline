"""Observed-state event models for the Vail daily sensor-endpoint experiment.

These models represent threshold-defined episodes of CAIC-reported Sno24, not
verified hourly fresh-snow storms. A state is active when the daily reading is
at least one inch. Unknown spell age remains unknown until an observed or
simulated transition establishes a new spell. Age14 is a persistent tail bin.

All fitting, standardization, emission donors and initialization frequencies
use training rows only. Forecast weather uses the issue-time summary; its logit
correction decays toward the seasonal event model rather than using future
observed weather. Exact finite-state forward means accompany sampled paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

THRESHOLD = 1.0
MAX_AGE = 14
N_STATES = 2*(MAX_AGE+1)


def calendar_phase(calendar):
    """Annual phase in radians from the first annual sine/cosine pair."""
    calendar=np.asarray(calendar,dtype=float)
    return np.mod(np.arctan2(calendar[...,0],calendar[...,1]),2*np.pi)


def age_index(age):
    age=np.asarray(age,dtype=float)
    return np.where(np.isfinite(age)&(age>=1),np.clip(np.nan_to_num(age),1,MAX_AGE),0).astype(int)


def advance_age(state,age,next_state):
    """Age0 is left-censored; age14 is a tail bin, not forced storm ending."""
    state,age,next_state=np.broadcast_arrays(state,age,next_state)
    return np.where(state!=next_state,1,np.where(age==0,0,np.minimum(age+1,MAX_AGE))).astype(int)


@dataclass
class _FittedLogit:
    mean: np.ndarray
    scale: np.ndarray
    constant: np.ndarray
    coefficient: np.ndarray
    intercept: float

    @classmethod
    def fit(cls,x,y,C):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=int)
        x=np.where(np.isfinite(x),x,np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            mean=np.nanmean(x,axis=0)
            scale=np.nanstd(x,axis=0)
        mean=np.nan_to_num(mean,nan=0.)
        constant=~np.isfinite(scale)|(scale<1e-7)
        scale=np.where(constant,1.,scale)
        standardized=(np.where(np.isfinite(x),x,mean)-mean)/scale
        standardized[:,constant]=0.
        if len(np.unique(y))==1:
            probability=(y.sum()+.5)/(len(y)+1.)
            coefficient=np.zeros(x.shape[1]);intercept=float(logit(probability))
        else:
            estimator=LogisticRegression(C=C,max_iter=1500,solver='lbfgs')
            with threadpool_limits(limits=2):
                estimator.fit(standardized,y)
            coefficient=estimator.coef_[0].copy();intercept=float(estimator.intercept_[0])
        coefficient[constant]=0.
        return cls(mean,scale,constant,coefficient,intercept)

    def decision(self,x):
        x=np.asarray(x,dtype=float)
        filled=np.where(np.isfinite(x),x,self.mean)
        standardized=(filled-self.mean)/self.scale
        standardized[...,self.constant]=0.
        return standardized@self.coefficient+self.intercept

    def partial_decision(self,x,start,stop):
        """Linear contribution of a feature block, excluding the intercept."""
        x=np.asarray(x,dtype=float)
        mean=self.mean[start:stop];scale=self.scale[start:stop]
        z=(np.where(np.isfinite(x),x,mean)-mean)/scale
        z[...,self.constant[start:stop]]=0.
        return z@self.coefficient[start:stop]


class EventModel:
    """Daily Markov or weather-conditioned duration-aware event model.

    ``regularization`` is sklearn's inverse L2 penalty C. No hyperparameters
    are selected internally. The caller supplies only its fit-winter rows.

    fit(summary[N,S], calendar_next[N,4], current_y[N], state_age[N], next_y[N])
    forecast(summary[N,S], calendar_future[N,H,4], current_y[N], state_age[N])

    Unknown current_y is supported for forecasting via a training-season state
    mixture. Training transition rows require current_y and next_y; missing
    targets never become zeros. NaN or0 state_age denotes left-censored age.
    """
    def __init__(self,kind='semi_markov',regularization=.1,weather_decay_days=3.,seed=20261104,use_weather=True,
                 emission_window_days=45,min_emission_donors=20):
        if kind not in ('markov','semi_markov'):
            raise ValueError('kind must be markov or semi_markov')
        if regularization<=0 or weather_decay_days<=0:
            raise ValueError('Positive regularization and weather_decay_days required')
        self.kind=kind;self.regularization=float(regularization)
        self.use_weather=bool(use_weather) and kind=='semi_markov'
        self.weather_decay_days=float(weather_decay_days);self.seed=int(seed)
        self.emission_window_days=float(emission_window_days)
        self.min_emission_donors=int(min_emission_donors)

    def _core(self,calendar,state,age):
        calendar=np.asarray(calendar,dtype=float)
        state=np.asarray(state,dtype=float).reshape(-1)
        age=np.asarray(age,dtype=float).reshape(-1)
        base=[state[:,None],calendar,state[:,None]*calendar]
        if self.kind=='semi_markov':
            known=(age>0).astype(float);log_age=np.log1p(age)
            duration=np.column_stack([known,log_age,log_age**2])
            base.extend([duration,state[:,None]*duration])
        return np.concatenate(base,axis=1)

    @staticmethod
    def _check_fit_shapes(summary,calendar,current_y,state_age,next_y):
        n=len(current_y)
        if summary.ndim!=2 or calendar.ndim!=2 or calendar.shape[0]!=n or calendar.shape[1]<4 or summary.shape[0]!=n:
            raise ValueError('Expected summary[N,S], calendar_next[N,C] with C>=4')
        if state_age.shape!=(n,) or next_y.shape!=(n,):
            raise ValueError('Expected current_y/state_age/next_y vectors with equal length')
        if not np.isfinite(calendar).all():
            raise ValueError('Known calendar features must be finite')

    def fit(self,summary,calendar_next,current_y,state_age,next_y):
        summary=np.asarray(summary,dtype=float);calendar=np.asarray(calendar_next,dtype=float)
        current=np.asarray(current_y,dtype=float).reshape(-1)
        age=age_index(np.asarray(state_age,dtype=float).reshape(-1))
        y=np.asarray(next_y,dtype=float).reshape(-1)
        self._check_fit_shapes(summary,calendar,current,age,y)
        labels=np.isfinite(y)&(y>=0)
        transition=labels&np.isfinite(current)&(current>=0)
        if transition.sum()<5:raise ValueError('At least five observed training transitions required')
        self.summary_features=summary.shape[1]
        self.calendar_features=calendar.shape[1]
        state=(current[transition]>=THRESHOLD).astype(int)
        next_state=(y[transition]>=THRESHOLD).astype(int)
        core=self._core(calendar[transition],state,age[transition])
        self.core_features=core.shape[1]
        self.base_logit=_FittedLogit.fit(core,next_state,self.regularization)
        if self.use_weather:
            full=np.concatenate([core,summary[transition]],axis=1)
            self.weather_logit=_FittedLogit.fit(full,next_state,self.regularization)
        else:self.weather_logit=None

        # Donor phase is determined using only the current and next observed day.
        donor_age=np.zeros(len(y),dtype=int)
        donor_age[transition]=advance_age(state,age[transition],next_state)
        self.donor_y=y[labels].copy()
        self.donor_state=(self.donor_y>=THRESHOLD).astype(int)
        self.donor_age=donor_age[labels]
        self.donor_phase=calendar_phase(calendar[labels])
        self._pool_cache={}
        self._pool_mean_cache={}
        self._pool_fallback={}
        self.training_rows=int(len(y));self.training_transitions=int(transition.sum())
        self.training_emissions=int(labels.sum())
        self.training_positive_states=int(next_state.sum())
        return self

    def _group(self,state,age):
        if state==0 or self.kind=='markov':return -1
        return 0 if age==0 else 1 if age==1 else 2 if age==2 else 3

    def _pool(self,phase,state,age):
        if not hasattr(self,'_pool_fallback'):self._pool_fallback={}
        group=self._group(state,age)
        key=(round(float(phase),10),int(state),group)
        if key in self._pool_cache:return self._pool_cache[key]
        delta=np.abs(self.donor_phase-phase)
        local=np.minimum(delta,2*np.pi-delta)<=2*np.pi*self.emission_window_days/365.2425
        regime=self.donor_state==state
        if group==-1:phase_mask=np.ones(len(regime),dtype=bool)
        elif group==3:phase_mask=self.donor_age>=3
        else:phase_mask=self.donor_age==group
        candidates=self.donor_y[local&regime&phase_mask]
        level='state_calendar_age' if group!=-1 else 'state_calendar'
        if len(candidates)<self.min_emission_donors:
            candidates=self.donor_y[local&regime]
            level='state_calendar'
        if len(candidates)<self.min_emission_donors:
            candidates=self.donor_y[regime]
            level='state_all_calendar'
        # Degenerate training cases get support-respecting fallback emissions.
        if len(candidates)==0:
            candidates=np.array([0. if state==0 else THRESHOLD])
            level='support_fallback_no_training_examples'
        self._pool_cache[key]=candidates
        self._pool_mean_cache[key]=float(np.mean(candidates))
        self._pool_fallback[key]=level
        return candidates

    def _initial_probability(self,phase):
        delta=np.abs(self.donor_phase-phase)
        local=np.minimum(delta,2*np.pi-delta)<=2*np.pi*self.emission_window_days/365.2425
        states=self.donor_state[local]
        if len(states)<self.min_emission_donors:states=self.donor_state
        return float(np.mean(states))

    def _all_hazards(self,summary,calendar_future):
        n,h,_=calendar_future.shape
        states=np.repeat([0,1],MAX_AGE+1)
        ages=np.tile(np.arange(MAX_AGE+1),2)
        weather_offset=None
        if self.weather_logit is not None:
            weather_offset=self.weather_logit.partial_decision(summary,self.core_features,self.core_features+self.summary_features)
        result=np.empty((n,h,N_STATES),dtype=float)
        for lead in range(h):
            calendar=np.repeat(calendar_future[:,lead],N_STATES,axis=0)
            core=self._core(calendar,np.tile(states,n),np.tile(ages,n))
            baseline=self.base_logit.decision(core).reshape(n,N_STATES)
            if weather_offset is not None:
                full=(self.weather_logit.partial_decision(core,0,self.core_features)
                      +self.weather_logit.intercept).reshape(n,N_STATES)+weather_offset[:,None]
                decay=np.exp(-lead/self.weather_decay_days)
                latent=baseline+decay*(full-baseline)
            else:latent=baseline
            result[:,lead]=expit(np.clip(latent,-20,20))
        return result

    def one_step_probability(self,summary,calendar_next,current_y,state_age):
        """P(next endpoint>=1), suitable for held-out Bernoulli log loss."""
        summary=np.asarray(summary,dtype=float);calendar=np.asarray(calendar_next,dtype=float)
        current=np.asarray(current_y,dtype=float).reshape(-1);age=age_index(state_age)
        hazards=self._all_hazards(summary,calendar[:,None,:])[:,0]
        phases=calendar_phase(calendar)-2*np.pi/365.2425
        result=np.empty(len(current))
        for i,value in enumerate(current):
            if np.isfinite(value) and value>=0:
                ix=int(value>=THRESHOLD)*(MAX_AGE+1)+age[i]
                result[i]=hazards[i,ix]
            else:
                p=self._initial_probability(phases[i]%(2*np.pi))
                result[i]=(1-p)*hazards[i,0]+p*hazards[i,MAX_AGE+1]
        return result

    def forecast(self,summary,calendar_future,current_y,state_age,n_paths=256,seed=None,exceedance_threshold=None):
        """Return exact means/probabilities and coherent Monte Carlo paths.

        Outputs: mean, p_active, p_onset [N,H]; paths [N,P,H]. p_onset counts
        inactive-to-active transitions, including reactivation after a quiet
        interval. It is distinct from probability of an active reading.
        An optional threshold adds exact p_exceedance=P(reading>=threshold).
        """
        summary=np.asarray(summary,dtype=float);calendar=np.asarray(calendar_future,dtype=float)
        current=np.asarray(current_y,dtype=float).reshape(-1);age=age_index(state_age)
        if calendar.ndim!=3 or calendar.shape[2]!=getattr(self,'calendar_features',4):
            raise ValueError('Future calendar/covariate dimension must match the fitted model')
        n,h,_=calendar.shape
        if summary.shape!=(n,self.summary_features) or len(current)!=n or len(age)!=n:
            raise ValueError('Inconsistent forecast issue dimensions')
        if not np.isfinite(calendar).all() or n_paths<1:raise ValueError('Finite calendars and positive n_paths required')
        if exceedance_threshold is not None and (not np.isfinite(exceedance_threshold) or exceedance_threshold<0):
            raise ValueError('Exceedance threshold must be finite and nonnegative')
        rng=np.random.default_rng(self.seed if seed is None else seed)
        hazards=self._all_hazards(summary,calendar)
        phases=calendar_phase(calendar)
        dist=np.zeros((n,N_STATES),dtype=float)
        sampled_state=np.zeros((n,n_paths),dtype=int)
        sampled_age=np.broadcast_to(age[:,None],(n,n_paths)).copy()
        for i,value in enumerate(current):
            if np.isfinite(value) and value>=0:
                state=int(value>=THRESHOLD)
                dist[i,state*(MAX_AGE+1)+age[i]]=1.
                sampled_state[i]=state
            else:
                p=self._initial_probability((phases[i,0]-2*np.pi/365.2425)%(2*np.pi))
                dist[i,0]=1-p;dist[i,MAX_AGE+1]=p
                sampled_state[i]=rng.random(n_paths)<p;sampled_age[i]=0
        mean=np.zeros((n,h));p_active=np.zeros((n,h));p_onset=np.zeros((n,h));p_onset_within=np.zeros((n,h))
        p_exceedance=None if exceedance_threshold is None else np.zeros((n,h))
        no_onset=dist.copy()
        paths=np.zeros((n,n_paths,h),dtype=np.float32)
        state_paths=np.zeros((n,n_paths,h),dtype=bool)
        onset_paths=np.zeros((n,n_paths,h),dtype=bool)
        origin_index=np.arange(n)[:,None]
        for lead in range(h):
            probability=hazards[:,lead]
            p_onset[:,lead]=(dist[:,:MAX_AGE+1]*probability[:,:MAX_AGE+1]).sum(1)
            nxt=np.zeros_like(dist);nxt_no_onset=np.zeros_like(dist)
            for state in (0,1):
                for old_age in range(MAX_AGE+1):
                    j=state*(MAX_AGE+1)+old_age
                    for new_state in (0,1):
                        new_age=int(advance_age(state,old_age,new_state))
                        dest=new_state*(MAX_AGE+1)+new_age
                        weight=probability[:,j] if new_state else 1-probability[:,j]
                        nxt[:,dest]+=dist[:,j]*weight
                        if not (state==0 and new_state==1):
                            nxt_no_onset[:,dest]+=no_onset[:,j]*weight
            if not np.allclose(nxt.sum(1),1.,atol=1e-10):raise ArithmeticError('State probability mass lost')
            dist=nxt;p_active[:,lead]=dist[:,MAX_AGE+1:].sum(1)
            no_onset=nxt_no_onset;p_onset_within[:,lead]=1-no_onset.sum(1)
            emission_means=np.empty((n,N_STATES))
            emission_tails=None if p_exceedance is None else np.empty((n,N_STATES))
            for i in range(n):
                for state in (0,1):
                    for next_age in range(MAX_AGE+1):
                        pool=self._pool(phases[i,lead],state,next_age)
                        j=state*(MAX_AGE+1)+next_age
                        emission_means[i,j]=pool.mean()
                        if emission_tails is not None:emission_tails[i,j]=np.mean(pool>=exceedance_threshold)
            mean[:,lead]=(dist*emission_means).sum(1)
            if p_exceedance is not None:p_exceedance[:,lead]=(dist*emission_tails).sum(1)

            path_probability=probability[origin_index,sampled_state*(MAX_AGE+1)+sampled_age]
            new_state=(rng.random((n,n_paths))<path_probability).astype(int)
            onset_paths[:,:,lead]=(sampled_state==0)&(new_state==1)
            new_age=advance_age(sampled_state,sampled_age,new_state)
            for i in range(n):
                for state in (0,1):
                    age_groups=[-1] if state==0 or self.kind=='markov' else [0,1,2,3]
                    for group in age_groups:
                        mask=new_state[i]==state
                        if group!=-1:
                            mask&=(new_age[i]>=3) if group==3 else (new_age[i]==group)
                        count=int(mask.sum())
                        if count:
                            donor_age=0 if group==-1 else group
                            pool=self._pool(phases[i,lead],state,donor_age)
                            paths[i,mask,lead]=pool[rng.integers(len(pool),size=count)]
            sampled_state,sampled_age=new_state,new_age
            state_paths[:,:,lead]=sampled_state.astype(bool)
        result=dict(mean=mean,p_active=p_active,p_onset=p_onset,p_onset_within=p_onset_within,paths=paths,
                    active_paths=state_paths,onset_paths=onset_paths,metadata=self.metadata())
        if p_exceedance is not None:result['p_exceedance']=np.clip(p_exceedance,0.,1.)
        return result

    def predict_paths(self,summary,calendar_future,current_y,state_age,n_paths=256,seed=None):
        return self.forecast(summary,calendar_future,current_y,state_age,n_paths=n_paths,seed=seed)['paths']

    def metadata(self):
        parameter_count=len(self.base_logit.coefficient)+1
        if self.weather_logit is not None:parameter_count+=len(self.weather_logit.coefficient)+1
        fallback=getattr(self,'_pool_fallback',{})
        return dict(kind=self.kind,use_weather=self.use_weather,threshold_inches=THRESHOLD,max_age_days=MAX_AGE,
                    age_zero='unknown/left-censored; remains unknown until state changes',
                    age_cap='persistent14+ tail bin, not forced exit',
                    logistic_C=self.regularization,weather_decay_days=self.weather_decay_days,
                    transition_training_rows=self.training_transitions,emission_training_rows=self.training_emissions,
                    summary_features=self.summary_features,core_features=self.core_features,
                    fitted_logistic_coefficients=parameter_count,
                    emission_policy=f'training-only empirical state/calendar±{self.emission_window_days:g}day/age-group donors; fallback below {self.min_emission_donors} donors',
                    unique_emission_pools_by_level={level:list(fallback.values()).count(level)
                                                   for level in sorted(set(fallback.values()))},
                    amount_support='inactive [0,1), active [1,infinity); raw positive offsets retained',
                    prediction='exact finite-state recursion for means/onset/active probabilities; sampled emission trajectories',
                    weather_policy='frozen issue-time summary; weather logit correction decays to seasonal duration hazard; no future observed weather')

