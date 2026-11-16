"""Synthetic illustration only: storm pulses and their ensemble mean.

No Vail observations, fitted parameters, or forecast results enter this figure.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/vail-storm-review'


def pulse_cdf(u):
    u=np.clip(u,0.,1.)
    return 6*u**2-8*u**3+3*u**4


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261104)
    n,days=4000,30
    boundaries=np.arange(days+1)
    paths=[]
    for _ in range(n):
        # Prehistory allows ongoing events at the illustration's left boundary.
        starts=rng.uniform(-3,days,size=rng.poisson(.13*(days+3)))
        duration=rng.uniform(1.,3.,size=len(starts))
        total=rng.gamma(2.,3.,size=len(starts))
        curve=np.zeros(days)
        for t,d,a in zip(starts,duration,total):
            curve+=a*np.diff(pulse_cdf((boundaries-t)/d))
        paths.append(curve)
    paths=np.asarray(paths)
    assert np.isfinite(paths).all() and paths.min()>=-1e-12
    # Analytical expectation for a stationary pulse process: rate * mean mark.
    expected_mean=.13*6
    assert abs(paths.mean()-expected_mean)<.06
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#b5c4cc','text.color':'#213e4a',
                         'axes.labelcolor':'#213e4a','xtick.color':'#425e69','ytick.color':'#425e69'})
    fig,axes=plt.subplots(4,1,figsize=(11.2,8),sharex=True,
                          gridspec_kw={'height_ratios':[1,1,1,1.35]})
    colors=['#007c91','#a76536','#77588d']
    for i in range(3):
        ax=axes[i]
        ax.bar(np.arange(1,days+1),paths[i],color=colors[i],alpha=.8,width=.86)
        ax.set_ylabel('in / day')
        ax.set_title(f'Possible storm sequence {i+1}',loc='left',fontsize=10,pad=5)
        ax.set_ylim(0,max(7.,paths[:3].max()*1.08))
        ax.grid(axis='y',color='#e6ecef');ax.set_axisbelow(True)
    mean=paths.mean(0); low,high=np.quantile(paths,[.1,.9],axis=0)
    ax=axes[-1]; x=np.arange(1,days+1)
    ax.fill_between(x,low,high,step='mid',color='#007c91',alpha=.17,label='Pointwise 10–90% range')
    ax.plot(x,mean,'o-',ms=3,color='#007c91',lw=2,label='Mean of 4,000 event sequences')
    ax.axhline(expected_mean,ls='--',lw=1,color='#677780',label='Exact expected value')
    ax.set_ylabel('in / day');ax.set_ylim(bottom=0)
    ax.set_title('The same event model, summarized as an average',loc='left',fontsize=10,pad=5)
    ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper right')
    ax.grid(axis='y',color='#e6ecef');ax.set_xlim(.5,30.5)
    ax.set_xlabel('Illustrative future day — no calendar dates or Vail forecast')
    fig.suptitle('Storm events can produce a smooth average forecast',x=.09,y=.98,ha='left',fontsize=19,weight='bold')
    fig.text(.09,.925,'Synthetic marked Poisson arrivals + smooth, finite-duration pulses; daily amounts integrate each pulse.',fontsize=10,color='#526c78')
    fig.text(.09,.025,'Illustration only. Parameters are invented to explain the distinction; this is not fitted to Vail.\n'
             'Forecast skill requires predicting when risk increases, not merely drawing plausible spikes.',fontsize=9,color='#755540')
    fig.subplots_adjust(left=.09,right=.98,top=.86,bottom=.13,hspace=.52)
    for ext in ['png','svg']:fig.savefig(OUT/f'storm_events_vs_mean.{ext}',dpi=180)
    plt.close(fig)
    np.savez_compressed(OUT/'illustration_paths.npz',daily_amounts=paths)
    (OUT/'illustration_manifest.json').write_text(json.dumps({
        'purpose':'Synthetic explanatory figure, NOT a fitted model or weather forecast',
        'seed':20261104,'paths':n,'days':days,'arrival_rate_per_day':.13,
        'event_total_distribution':'Gamma(shape=2,scale=3 inches)',
        'duration_distribution':'Uniform(1,3 days)',
        'pulse':'Beta(2,3) density on [0,1], divided by duration; normalized total equals event mark',
        'expected_daily_amount':expected_mean,'simulated_daily_mean':float(paths.mean())},indent=2))
    print(OUT/'storm_events_vs_mean.png')


if __name__=='__main__':main()
