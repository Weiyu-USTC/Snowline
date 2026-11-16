"""Publication-style figure contrasting historical means and the saved outlook."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

os.environ.setdefault('MPLCONFIGDIR','/tmp/copper-outlook-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
MONTHS=['nov','dec','jan','feb','mar','apr']


def main():
    saved=json.loads((OUT/'forecast_2026_27.json').read_text())
    history=pd.read_csv(ROOT/'data/copper_monthly_user.csv')
    history=history.loc[history.start_year.between(2005,2024),MONTHS]
    assert len(history)==20 and history.notna().all().all()
    average=history.mean().to_numpy()
    half=t.ppf(.975,len(history)-1)*history.std(ddof=1).to_numpy()/np.sqrt(len(history))
    forecast=saved['forecasts']['experimental_enso_only']
    predicted=np.array([forecast['point_inches'][m] for m in MONTHS])
    low=np.array([forecast['monthly_lower'][m] for m in MONTHS])
    high=np.array([forecast['monthly_upper'][m] for m in MONTHS])
    np.testing.assert_allclose(average,[saved['forecasts']['historical_monthly_mean']['point_inches'][m] for m in MONTHS])

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.spines.left':False,'axes.spines.bottom':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(12.8,8),facecolor='white')
    fig.subplots_adjust(left=.095,right=.965,bottom=.205,top=.705)
    ink='#163349';blue='#087e9d';gray='#626c76'
    fig.text(.095,.955,'COPPER MOUNTAIN',fontsize=11,weight='bold',color=blue,va='top')
    fig.text(.095,.913,'Monthly snowfall outlook · 2026/27',fontsize=25,weight='bold',color=ink,va='top')
    fig.text(.095,.853,'Upcoming months compared with 20 historical seasons (2005/06–2024/25)',
             fontsize=12,color='#566774',va='top')
    x=np.arange(6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True,color='#e5ebef',linewidth=.8)
    ax.fill_between(x,low,high,color=blue,alpha=.12,linewidth=0,zorder=1)
    # Both historical layers have the requested alpha=0.5.
    ax.fill_between(x,average-half,average+half,color=gray,alpha=.5,linewidth=0,zorder=2)
    ax.plot(x,average,color=gray,alpha=.5,linewidth=2.5,linestyle=(0,(5,3)),zorder=3)
    ax.plot(x,predicted,color=blue,linewidth=3,marker='o',markersize=7,
            markeredgecolor='white',markeredgewidth=1.5,zorder=4)
    for month,value in zip(x,predicted):
        ax.annotate(f'{value:.1f}',(month,value),xytext=(0,13),textcoords='offset points',
                    ha='center',va='bottom',fontsize=11,weight='bold',color=blue,
                    bbox={'boxstyle':'round,pad=.16','facecolor':'white','edgecolor':'none','alpha':.88})
    ax.set(xticks=x,xticklabels=['Nov 2026','Dec','Jan 2027','Feb','Mar','Apr'],
           xlim=(-.17,5.17),ylim=(0,100),yticks=np.arange(0,101,20),ylabel='Monthly new snowfall (inches)')
    ax.tick_params(axis='both',length=0,pad=10,labelcolor='#3f5667')
    ax.yaxis.label.set_color(ink)
    ax.yaxis.labelpad=12
    handles=[Line2D([0],[0],color=blue,lw=3,marker='o',markersize=6,label='Experimental ENSO-only forecast'),
             Line2D([0],[0],color=gray,alpha=.5,lw=2.5,ls='--',label='Historical average · α = 0.5'),
             Patch(facecolor=blue,alpha=.12,label='Forecast range · nominal 90%'),
             Patch(facecolor=gray,alpha=.5,label='Historical mean · 95% confidence band')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.09,.807),ncol=2,
               frameon=False,columnspacing=3.0,handlelength=2.5,labelspacing=.75,fontsize=10.5)
    fig.text(.095,.132,f'November–April total: forecast {predicted.sum():.0f} in  ·  historical average {average.sum():.0f} in',
             fontsize=12,weight='bold',color=ink)
    fig.text(.095,.087,'As of November 3, 2026. Copper’s October total is unverified; this forecast uses ENSO without October.',
             fontsize=9.5,color='#566774')
    fig.text(.095,.059,'Gray: uncertainty in the historical mean. Blue: exploratory uncertainty for a future month, not simultaneous coverage.',
             fontsize=9.5,color='#566774')
    fig.text(.095,.031,'Predictor-based models have not demonstrated better accuracy than historical averages in held-out tests.',
             fontsize=9.5,color='#566774')
    for extension in ['png','svg']:
        fig.savefig(OUT/f'copper_2026_27_monthly_outlook.{extension}',dpi=220,facecolor='white')
    plt.close(fig)
    pd.DataFrame({'month':MONTHS,'historical_mean_inches':average,
                  'historical_mean_ci95_low':average-half,'historical_mean_ci95_high':average+half,
                  'forecast_inches':predicted,'forecast_range_low':low,'forecast_range_high':high}).to_csv(
                      OUT/'copper_2026_27_monthly_outlook_data.csv',index=False)
    print(OUT/'copper_2026_27_monthly_outlook.png')


if __name__=='__main__':main()
