"""Standalone scientific report for the native NEXRAD pilot."""
import json,html,shutil
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nexrad_acquire import DATA,SITES
from nexrad_train import OUT

NAMES={'alta':'Alta Collins','mansfield':'Stowe town','jay':'Westfield / Jay vicinity'}
COLORS={'NEXRAD_HRRR_portfolio':'#007b83','HRRR':'#d58f32','NEXRAD_selected':'#70b7b2','DPR_persistence':'#9296a0','DPR_advection':'#bbc0c7','train_mean':'#ccd4d8','previous_MRMS_nowcastnet':'#655baa','previous_MRMS_convlstm':'#876dba','previous_MRMS_dgmr':'#b39dcc'}
LABELS={'NEXRAD_HRRR_portfolio':'Radar + HRRR portfolio','NEXRAD_selected':'Selected native NEXRAD','HRRR':'HRRR','DPR_persistence':'DPR persistence','DPR_advection':'DPR advection','train_mean':'Training mean','previous_MRMS_nowcastnet':'Previous MRMS NowcastNet','previous_MRMS_convlstm':'Previous MRMS ConvLSTM','previous_MRMS_dgmr':'Previous MRMS DGMR'}

def main():
 metrics=pd.read_csv(OUT/'paired_metrics.csv');pred=pd.read_csv(OUT/'paired_predictions.csv',dtype={'case_id':str});choice=json.loads((OUT/'portfolio_selection.json').read_text());current=json.loads((OUT/'current_forecast.json').read_text())
 coverage=json.loads((OUT/'coverage.json').read_text())
 for key in ['vail_ftg','vail_gjx']:
  if not any(c['site']==key for c in coverage):
   files=sorted((DATA/'crops'/key).glob('20*.npz'))
   with np.load(files[0]) as z:m=json.loads(str(z['metadata_json']))[-1]
   coverage.append(dict(site=key,scheduled=189,radar_crops=len(files),qualified_train=0,qualified_validation=0,qualified_test=0,radar_geometry=m,reason='No qualified MRMS reference at frozen RQI>=0.5; retain HRRR fallback'))
 (OUT/'coverage.json').write_text(json.dumps(coverage,indent=2))
 report=json.loads((OUT/'report.json').read_text());report['coverage']=coverage;report['portfolio_selection']=choice
 report['current_alignment']='Historical issue 10:50 UTC predicts11:00–14:00UTC. Live issue may have a different gap to the next complete clock hour; advection uses actual leads, but calibrator has not been independently validated for every gap.'
 report['deployment_caution']='Limited pilot with4–8 wet test cases per mountain; liquid benchmark cannot establish snowfall skill. Test not used to change validation-selected models or weights.'
 (OUT/'report.json').write_text(json.dumps(report,indent=2))
 plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','font.size':10})
 fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
 for col,(site,name) in enumerate(NAMES.items()):
  d=metrics[(metrics.site==site)&(metrics.split=='test')&metrics.model.isin(LABELS)].copy();d['order']=d.model.map({k:i for i,k in enumerate(LABELS)});d=d.sort_values('order')
  for row,h in enumerate([1,3]):
   ax=axes[row,col];v=d[f'RMSE_{h}h_mm'].to_numpy();bars=ax.barh(np.arange(len(d)),v,color=[COLORS[n] for n in d.model]);ax.set_yticks(np.arange(len(d)),[LABELS[n] for n in d.model]);ax.invert_yaxis();ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9);ax.set_xlim(0,max(v)*1.22);ax.set_xlabel(f'{h}-hour accumulation RMSE (liquid mm)');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
   ax.set_title(name.replace('Westfield / Jay vicinity','Jay vicinity')+f' · n={int(d.n.min())}, wet={int(d[f"wet_{h}h_cases"].max())}')
 fig.suptitle('Frozen 2023/24 test winter · Native NEXRAD and HRRR\nQuality-screened MRMS liquid reference; shared radar ancestry, not snowfall truth',fontsize=15)
 fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 for ax,(site,name) in zip(axes,NAMES.items()):
  r=current['sites'][site];native=r.get('native_nexrad_hourly',r['hourly']);hourly=r['hourly'];x=np.arange(3)
  for i,(label,values,color) in enumerate([('Native NEXRAD',[h['liquid_mm'] for h in native],'#70b7b2'),('HRRR',[h.get('hrrr_liquid_mm',np.nan) for h in hourly],'#d58f32'),('Portfolio',[h['liquid_mm'] for h in hourly],'#007b83')]):
   ax.bar(x+(i-1)*.23,values,width=.22,label=label,color=color)
  ax.axhline(0,color='#007b83',lw=1.5,zorder=4);ax.set_xticks(x,[h['end'][11:16] for h in hourly]);ax.set_xlabel('Hour ending (UTC)');ax.set_title(name);ax.set_ylabel('Estimated liquid precipitation (mm)');ax.set_ylim(0,max(.1,max(h['liquid_mm'] for h in hourly)*1.3));ax.grid(axis='y',alpha=.2)
 axes[-1].legend(loc='upper right');fig.suptitle('Current three complete clock hours · issue '+current['issued_at']+'\nLiquid forecast only; Vail radar unavailable for validated use',fontsize=13)
 fig.savefig(OUT/'current_liquid.png',dpi=160);plt.close(fig)
 # Show a clearly labelled observed-wet case from the untouched test set for illustration.
 fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 examples=[]
 for ax,(site,name) in zip(axes,NAMES.items()):
  d=pred[(pred.site==site)&(pred.split=='test')];unique=d[d.model=='NEXRAD_HRRR_portfolio'].copy();unique['total']=unique[['truth1','truth2','truth3']].sum(axis=1);r=unique.sort_values('total',ascending=False).iloc[0];cid=r.case_id;examples.append({'site':site,'case_id':cid,'selection':'Largest observed 3 h total among qualified held-out cases, solely to illustrate a storm; not used for model selection.'})
  case=d[d.case_id==cid];ax.plot([1,2,3],r[['truth1','truth2','truth3']].to_numpy(float),'o-',color='#193b45',lw=2.5,label='MRMS reference')
  for model in ['NEXRAD_HRRR_portfolio','HRRR','NEXRAD_selected','previous_MRMS_nowcastnet']:
   q=case[case.model==model]
   if len(q):ax.plot([1,2,3],q.iloc[0][['pred1','pred2','pred3']].to_numpy(float),'o--',label=LABELS[model],color=COLORS[model])
  ax.axhline(0,color='#888',lw=.7);ax.set_xticks([1,2,3],['11–12','12–13','13–14']);ax.set_xlabel('UTC hour');ax.set_ylabel('Liquid accumulation (mm)');ax.set_title(name+' · '+cid);ax.grid(alpha=.2)
 axes[-1].legend(fontsize=8);fig.suptitle('Illustrative held-out storms: largest observed 3h case at each site\nExamples chosen after evaluation for display only',fontsize=13);fig.savefig(OUT/'storm_examples.png',dpi=160);plt.close(fig)
 (OUT/'storm_examples.json').write_text(json.dumps(examples,indent=2))
 show=metrics[(metrics.split=='test')&metrics.model.isin(['NEXRAD_HRRR_portfolio','HRRR','NEXRAD_selected','train_mean','previous_MRMS_convlstm','previous_MRMS_nowcastnet','previous_MRMS_dgmr'])].copy();show['site']=show.site.map(NAMES);show['model']=show.model.map(LABELS)
 cols=['site','model','n','MAE_1h_mm','RMSE_1h_mm','MAE_3h_mm','RMSE_3h_mm','wet_3h_cases'];table=show[cols].to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
 covrows=[]
 for r in coverage:
  g=r['radar_geometry'];covrows.append({'Target':r['site'],'Downloaded cases':r['radar_crops'],'Qualified train / validation / test':f"{r['qualified_train']} / {r['qualified_validation']} / {r['qualified_test']}",'Radar range km':round(g['distance_km'],1),'Nominal beam above target m':round(g['beam_above_target_m'])})
 cov=pd.DataFrame(covrows).to_html(index=False,border=0)
 weights=''.join(f'<li>{html.escape(NAMES[k])}: radar {v["radar_weight"]:.0%} + HRRR {v["hrrr_weight"]:.0%}; native method {html.escape(v["native_model"])}.</li>' for k,v in choice.items())
 doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Native NEXRAD short-horizon pilot</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f3f6f4;color:#18383f;max-width:1280px;margin:32px auto;padding:0 24px}}section{{background:white;border-radius:14px;padding:24px;margin:22px 0}}h1,h2{{line-height:1.25}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;border-bottom:1px solid #dce5df;padding:9px}}.scroll{{overflow:auto}}a{{color:#007b83}}.note{{background:#eef5f2;padding:16px;border-left:4px solid #007b83}}</style><h1>Native NEXRAD: a measured 1–3 hour pilot</h1><p>Real KMTX, KCXX, KFTG and KGJX Level III DPR radar products, paired with NOAA HRRR. Generated {datetime.now(timezone.utc).isoformat()}.</p><section><h2>What the test supports</h2><p>This is a small, frozen calendar cohort of 189 mornings across four winters. Models train on 2020/21 and 2021/22, select on 2022/23, and test on 2023/24. Every score below is liquid precipitation, not fresh-snow depth.</p><p>The portfolio weights are selected on validation data only. They are retained even when another model has lower test error. The 2023/24 winter has been inspected in earlier project work; it is held out from this fit and weight selection, not a wholly unseen project dataset.</p><ul>{weights}</ul><p class="note">MRMS verification shares NEXRAD observations, so this measures agreement with a radar analysis rather than independent snowfall accuracy. Only 4–8 of the test mornings at each site exceed 0.3 mm over 3 h; broad superiority is not established.</p><img src="comparison.png" alt="Held-out liquid accumulation errors"><div class="scroll">{table}</div></section><section><h2>Current next three full hours</h2><img src="current_liquid.png" alt="Current native radar and HRRR liquid forecast"><p>Historical forecasts issue 10:50 UTC and cover 11:00–14:00 UTC. The live forecast begins at the next complete clock hour after its issue time. Advection uses the actual lead, but calibration has not been separately validated for every possible issue-to-hour gap. Zero is drawn explicitly. Snowfall conversion requires forecast phase and snow-to-liquid ratio and has extra uncertainty; it is deliberately absent from this liquid-only chart.</p></section><section><h2>What a storm comparison looks like</h2><img src="storm_examples.png" alt="Three hourly accumulation curves compared to MRMS"><p>These are the largest observed 3 h totals among qualified held-out cases, chosen only to make storm behaviour visible. This display choice does not change models or portfolio weights.</p></section><section><h2>Mountain coverage and data limits</h2><div class="scroll">{cov}</div><p>Vail's local MRMS quality averages about0.20 and produces no verification windows at the frozen RQI ≥ 0.5 threshold. Genuine native radar files exist, but we retain the HRRR fallback instead of claiming a validated radar model there. Nominal 0.5° beam heights use standard 4/3-Earth geometry; they do not resolve intervening terrain blockage. DPR is a hybrid product and can use higher elevations.</p><p>Stowe and Jay use the same lower-elevation NOAA station locations as the existing station-calibrated models: Stowe town and Westfield near Jay. These are not summit observations or resort-wide snowfall measurements.</p></section><section><h2>Reproducibility</h2><p>DPR product 176 contains instantaneous rate in inches/hour, decoded with MetPy and converted to mm/hour. Zero in this product is a valid estimated zero, although mountain under-detection remains possible. Three scans have volume starts at or before issue−70,−40 and−10minutes; each product must be generated at least 2 minutes before issue. File hashes and generation times are retained. MRMS target integration uses 10-minute endpoints, qualifying source validity≥0.9 and RQI≥0.5 in at least 5 of 9 local gridcells at every endpoint.</p><p><a href="paired_metrics.csv">Paired test and validation metrics</a> · <a href="paired_predictions.csv">Case predictions</a> · <a href="current_forecast.json">Current machine-readable forecast</a> · <a href="report.json">Design and coverage audit</a> · <a href="portfolio_selection.json">Frozen portfolio weights</a></p><p>Primary sources: <a href="https://registry.opendata.aws/noaa-nexrad/">NOAA NEXRAD public archive</a>, <a href="https://www.ncei.noaa.gov/products/radar/next-generation-weather-radar">NCEI product documentation</a>, <a href="https://www.roc.noaa.gov/public-documents/icds/2620003AE.pdf">Radar Operations Center Level III specification</a>.</p></section></html>'''
 (OUT/'report.html').write_text(doc)
 public=OUT.parents[0]/'nwp-weather/forecast-portfolio/nexrad'
 public.mkdir(parents=True,exist_ok=True)
 for source in OUT.iterdir():
  if source.suffix in {'.html','.png','.pdf','.csv','.json'}:shutil.copy2(source,public/source.name)
 print(OUT/'report.html')
 print('Public report copy:',public/'report.html')
if __name__=='__main__':main()
