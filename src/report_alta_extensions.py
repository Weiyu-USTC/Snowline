"""New experiment, observed accumulation and transfer-data report sections."""
from pathlib import Path
import base64
import html
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/alta-nowcast'
OUT = BASE/'learning-review'
TUNE = BASE/'tuning-20261107'
NAMES = ['convlstm','dgmr','nowcastnet']
LABELS = {'convlstm':'ConvLSTM','dgmr':'DGMR adaptation','nowcastnet':'NowcastNet adaptation'}
COLORS = {'convlstm':'#257db0','dgmr':'#c17a32','nowcastnet':'#8055a6'}


def image(path, alt):
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'


def table(frame):
    formats = {c:lambda v:f'{v:.3g}' for c in frame.columns if 'LR' in str(c)}
    return '<div class="table-wrap">'+frame.to_html(index=False,border=0,float_format=lambda v:f'{v:.5f}',formatters=formats)+'</div>'


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=175,bbox_inches='tight')
    plt.close(fig)
    return image(OUT/f'{name}.png',name.replace('_',' '))


def accumulation_section():
    tuned = TUNE/'accumulation'
    directory = tuned if (tuned/'observed_1h_audit.json').exists() else OUT
    if not (directory/'observed_1h_audit.json').exists():return ''
    audit = json.loads((directory/'observed_1h_audit.json').read_text())
    title = 'New tuned checkpoints' if directory==tuned else 'Original pilot checkpoints'
    event = audit['illustrative_event']
    figures = ''.join('<figure>'+image(directory/f'{name}.png',alt)+'</figure>' for name,alt in [
        ('one_hour_event_curves','Observed and predicted cumulative precipitation within one hour'),
        ('observed_1h_alta','Observed versus forecast one-hour precipitation at the Alta grid cell')])
    more = ''.join(image(directory/f'{name}.png',alt) for name,alt in [
        ('one_hour_event_maps','Observed precipitation map and six forecast maps for the same one-hour storm'),
        ('observed_1h_regional','Regional mean observed versus predicted one-hour accumulation')])
    amounts = pd.read_csv(directory/'observed_1h_amount_metrics.csv')
    point = amounts[amounts.scope=='alta'][['model','cases','mae_mm','rmse_mm','bias_mm']].rename(columns={
        'model':'Method','cases':'Dates','mae_mm':'Alta-cell MAE (mm)','rmse_mm':'Alta-cell RMSE (mm)','bias_mm':'Bias (mm)'})
    export_names = ['observed_1h_accumulation.csv','observed_1h_amount_metrics.csv','one_hour_event_curves.csv','observed_1h_audit.json']
    for name in export_names:
        if directory!=OUT:shutil.copy2(directory/name,OUT/('tuned_'+name))
    prefix = 'tuned_' if directory!=OUT else ''
    links = ' · '.join(f'<a href="{prefix}{n}">{n.replace("_"," ")}</a>' for n in export_names)
    return f'''<h3 id="observed-accumulation">What did the one-hour accumulation actually look like?</h3>
    <p><strong>{title} compared with observed radar amounts.</strong> One-hour accumulation is the integral of precipitation rate over 60 minutes: a constant 2 mm/hour produces 2 mm of water. Here, seven radar rate snapshots at 10-minute spacing are integrated from <strong>11:00 to 12:00 UTC</strong>, following the 10:50 UTC forecast issue. Thus the window ends 70 minutes after issue. These are liquid-equivalent water amounts, not inches of fresh snow; snow depth also depends on precipitation phase and snow density.</p>
    <p>The example below is January 14, 2024, selected solely because it has the largest observed Alta-cell one-hour amount in this test cohort: <strong>{event['alta_observed_mm']:.3f} mm</strong>. HRRR provides an hourly total, so it appears only at the 60-minute endpoint. This storm is illustrative; the full 180-date comparison below is the basis for the reported errors.</p>{figures}
    <p class="small">Each point on the seasonal chart is one morning hour on that date. Lines connecting dates are visual guides; these data do not provide a continuous hourly test record. The Alta point is a roughly 3.4 × 4.4 km pooled radar cell and is not an independent snow-board observation. No forecast uncertainty band is inferred from a single deterministic observation.</p>{table(point)}
    <details><summary>View spatial storm maps and the regional-average comparison</summary>{more}<p class="small">Error of the regional average can cancel opposite errors at different pixels. It is distinct from the average pixel-level error in the benchmark tables. The valid radar area varies by date but is identical across methods within each date.</p></details><p class="downloads">{links}</p>'''


def tuning_section():
    if not (TUNE/'plan.json').exists():return ''
    plan = json.loads((TUNE/'plan.json').read_text())
    histories, results, progress = {}, [], []
    for name in NAMES:
        for path in sorted((TUNE/name).glob('*/history.json')):
            d = pd.DataFrame(json.loads(path.read_text()))
            if d.empty:continue
            histories[(name,path.parent.name)] = d
            result_path = path.parent/'result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            if result:results.append(result)
            selectable=d[d.validation_eligible_for_selection]
            progress.append({'Model':LABELS[name],'Run':path.parent.name,
                'Updates':int(d.step.iloc[-1]),'Status':'Complete' if result else 'Running',
                'Best validation MSE':float(selectable.validation_mse_mm_h_squared.min()) if len(selectable) else np.nan,
                'Latest validation MSE':float(d.validation_mse_mm_h_squared.iloc[-1])})
    frozen = TUNE/'final_selection.json'
    selection = json.loads(frozen.read_text()) if frozen.exists() else None
    if histories:
        pd.concat([d.assign(model_id=name,run_id=run) for (name,run),d in histories.items()],ignore_index=True).to_csv(OUT/'tuning_training_histories.csv',index=False)
    status = 'Training completed' if selection else f'{len(results)} of 15 runs completed'
    body = f'''<section id="tuning"><div class="eyebrow">New experiment · November 7, 2026</div><h2>Learning-rate search and repeated training</h2><p><strong>{status}.</strong> Three starting learning rates (0.0001, 0.0003, 0.001) are tested for each architecture. The selected rate is repeated with two additional random seeds. Every run has a 3,000-update ceiling, a fixed 32-case naturally sampled training probe, and the same 42-case validation set evaluated every 100 updates.</p>
    <p>Both optimizer learning rates halve after sustained validation stalls; early stopping requires at least 1,500 updates and ten checks without a meaningful improvement. NowcastNet’s evolution stage stays fixed at 500 updates. The best checkpoint and final checkpoint, optimizer states, interval-average losses and gradient norms are retained separately.</p>
    <p class="small">An “update” counts one main-model training iteration; adversarial generator iterations also update a discriminator. The fixed training probe happened to draw only one wet-defined case out of 32, compared with 56 of 352 in the full training split. It tracks the same cases through training, but its gap from validation is not the full training–validation gap. Full-training checkpoint scores are evaluated after selection.</p>
    <p class="notice"><strong>Scope of “best”.</strong> This is a bounded search among three compact local implementations and three initial rates. Rate selection uses one screening seed; architecture comparison uses the mean of three seeds at that selected rate. Repeated selection on 42 cases can be optimistic. The 2023/24 test winter has already been inspected in earlier work; it is a reused benchmark, not a new blind test. No test metric enters the new selection rule.</p>
    <p class="small">Within each architecture’s learning-rate screen, initialization and batch order are held fixed. Comparison with the original pilot changes the training protocol—including an independently seeded batch sampler, adaptive scheduling and checkpoint frequency—so any difference cannot be attributed to learning rate alone.</p>'''
    if histories:
        fig,axs = plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
        palette = ['#7ba6aa','#28788a','#bc7745']
        for ax,name in zip(axs,NAMES):
            for color,rate in zip(palette,[1e-4,3e-4,1e-3]):
                key=(name,f'lr{rate:.0e}_seed20261105')
                if key not in histories:continue
                d=histories[key];d=d[d.validation_eligible_for_selection]
                ax.plot(d.step,d.validation_mse_mm_h_squared,'o-',ms=3,lw=1.4,color=color,label=f'Initial LR {rate:g}')
            ax.set(title=LABELS[name],xlabel='Optimizer update',ylabel='Validation rate MSE · (mm/h)²')
            ax.grid(alpha=.15);ax.set_yscale('log');ax.legend(frameon=False,fontsize=8)
        fig.suptitle('Learning-rate screening · same validation metric and screening seed',fontsize=15,fontweight='bold')
        body += save(fig,'tuning_rate_screen')
    if not selection:
        body += '<p>Training is in progress. The curves above show only measurements already saved; no winner is declared before replication finishes.</p>'
        finished=[]
        for name in NAMES:
            fpath=TUNE/name/'family_result.json'
            if fpath.exists():
                f=json.loads(fpath.read_text())
                finished.append({'Completed architecture':LABELS[name],
                    'Selected initial LR':f['selected_initial_learning_rate'],
                    'Seeds':f['replicate_count'],'Mean best validation MSE':f['mean_best_validation_mse'],
                    'Seed SD':f['sd_best_validation_mse']})
        if finished:body+=table(pd.DataFrame(finished))
        if progress:body += table(pd.DataFrame(progress))
        return body+'</section>'
    families = selection['models']
    winner = selection.get('winner_by_mean_validation',selection.get('selected_family'))
    rows=[]
    representative={}
    for name in NAMES:
        f=families[name]
        assert f.get('replicate_count',len(f['runs']))==3, 'Three successful seeds are required for final family comparison'
        rate=f.get('selected_initial_lr',f.get('selected_initial_learning_rate'))
        sd=f.get('std_best_validation_mse',f.get('sd_best_validation_mse'))
        checkpoint=Path(f['representative_checkpoint'])
        if not checkpoint.is_absolute():checkpoint=ROOT/checkpoint
        representative[name]=pd.DataFrame(json.loads((checkpoint.parent/'history.json').read_text()))
        r=json.loads((checkpoint.parent/'result.json').read_text())
        chosen_row=representative[name].set_index('step').loc[r['best_step']]
        rows.append({'Model':LABELS[name],'Selected initial LR':rate,
            'Mean best validation MSE':f['mean_best_validation_mse'],'Seed SD':sd,
            'Representative seed':r['seed'],'Selected update':r['best_step'],
            'LR at selected update':chosen_row.learning_rate_used,
            'Final update':r['completed_updates'],'Final LR':r['final_learning_rate'],
            'Stop reason':r['stop_reason']})
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'tuning_model_comparison.csv',index=False)
    pd.DataFrame(results).drop(columns=['training_draw_counts'],errors='ignore').to_csv(OUT/'tuning_all_runs.csv',index=False)
    body += f'<h3 id="tuning-results">Validation-selected architecture: {LABELS[winner]}</h3><p>The representative model for each architecture is the median-scoring seed, rather than its luckiest run. Seed standard deviation describes these three runs; it is not a population confidence interval.</p>'+table(summary[['Model','Selected initial LR','Mean best validation MSE','Seed SD']])
    body += '<details><summary>Selected checkpoints, final learning rates and stopping decisions</summary>'+table(summary.drop(columns=['Mean best validation MSE','Seed SD']))+'</details>'
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for j,name in enumerate(NAMES):
        values=np.array([r['best_validation_mse'] for r in families[name]['runs']])
        ax.scatter(j+np.linspace(-.12,.12,len(values)),values,s=55,color=COLORS[name],label='Individual training seeds' if j==0 else None,zorder=4)
        ax.errorbar(j,values.mean(),yerr=values.std(ddof=1),fmt='_',markersize=24,capsize=8,color='#233f45',lw=1.8,label='Mean ± one seed SD' if j==0 else None,zorder=5)
    ax.set_xticks(range(3),[LABELS[n] for n in NAMES]);ax.set(ylabel='Best validation rate MSE · (mm/hour)²',title='How much does retraining change the result?')
    ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=9)
    body += save(fig,'tuning_seed_comparison')
    ordered=summary.sort_values('Mean best validation MSE')
    gap=ordered.iloc[1]['Mean best validation MSE']-ordered.iloc[0]['Mean best validation MSE']
    if gap<max(ordered.iloc[0]['Seed SD'],ordered.iloc[1]['Seed SD']):
        body += '<p>The gap between the two lowest mean scores is smaller than at least one of their seed standard deviations. The numerical winner is provisional; three seeds and one selection winter do not establish a stable architecture ranking.</p>'
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for j,name in enumerate(NAMES):
        d=representative[name];d=d[d.validation_eligible_for_selection]
        ax=axs[0,j]
        ax.plot(d.step,d.fixed_train_probe_mse,color=COLORS[name],label='32-case train probe · 1 wet case')
        ax.plot(d.step,d.validation_mse_mm_h_squared,'--',color='#263e45',label='Fixed validation')
        beststep=int(summary.iloc[j]['Selected update'])
        ax.axvline(beststep,color=COLORS[name],alpha=.4,ls=':')
        ax.set(title=LABELS[name],ylabel='Rate MSE · (mm/hour)²',xlabel='Optimizer update');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
        ax=axs[1,j]
        ax.step(d.step,d.next_learning_rate,where='post',color=COLORS[name])
        ax.set(xlabel='Optimizer update',ylabel='Learning rate after evaluation');ax.set_yscale('log');ax.grid(alpha=.15)
    fig.suptitle('Representative runs · comparable train/validation losses and learning rates',fontsize=15,fontweight='bold')
    body += save(fig,'tuning_fixed_losses')
    fig,axs=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    for ax,name in zip(axs,NAMES):
        d=representative[name]
        for phase,g in d.groupby('phase',sort=False):
            ax.plot(g.step,g.last_minibatch_objective,color='#a1aaae',alpha=.45,lw=.8,label='Last minibatch' if phase=='generator' else None)
            ax.plot(g.step,g.interval_mean_objective,color=COLORS[name],alpha=.55 if phase=='evolution' else 1,lw=2,label=f'{phase.title()} interval mean')
        ax.set(title=LABELS[name],xlabel='Optimizer update',ylabel='Model-specific training objective');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Averaging batches makes the optimization trend easier to assess',fontsize=15,fontweight='bold')
    body += save(fig,'tuning_averaged_objectives')
    gradients=[]
    for name,d in representative.items():
        last=d.iloc[-1]
        gradients.append({'Model':LABELS[name],'Final update':int(last.step),
            'Last gradient norm before clipping':last.last_gradient_norm_before_clipping,
            'Last interval mean gradient norm':last.interval_mean_gradient_norm_before_clipping,
            'Last interval clipping fraction':last.interval_generator_clipping_fraction})
    body += '<details><summary>Final gradient and clipping diagnostics</summary><p>Norms are recorded before clipping; the update uses a clipping threshold of 5. A nonzero minibatch gradient is expected with stochastic training, so these norms alone are not a convergence test.</p>'+table(pd.DataFrame(gradients))+'</details>'
    if (TUNE/'evaluation_audit.json').exists():
        selected_losses=pd.read_csv(TUNE/'checkpoint_losses.csv')
        final_losses=pd.read_csv(TUNE/'final_checkpoint_losses.csv')
        endrows=[]
        for name in NAMES:
            b=selected_losses[selected_losses.model_id==name].set_index('split')
            f=final_losses[final_losses.model_id==name].set_index('split')
            endrows.append({'Model':LABELS[name],
                'Best full-train MSE':b.loc['train','rate_mse_mm_h_squared'],
                'Final full-train MSE':f.loc['train','rate_mse_mm_h_squared'],
                'Best validation MSE':b.loc['selection','rate_mse_mm_h_squared'],
                'Final validation MSE':f.loc['selection','rate_mse_mm_h_squared'],
                'Remaining development MSE':b.loc['development_remainder','rate_mse_mm_h_squared']})
        body += '<h3>End-of-run score using the full training split</h3><p>These inference-only scores use all 352 training cases, with natural case frequency. The remaining 125 development cases were excluded from this tuning selection and are a diagnostic check, although they have appeared in earlier project analyses. All columns below use rate MSE in (mm/hour)².</p>'+table(pd.DataFrame(endrows))
        new=pd.read_csv(TUNE/'short_summary.csv');old=pd.read_csv(BASE/'short_summary.csv')
        comparison=[]
        for name in NAMES:
            for h in [1,3]:
                a=new[(new.model==LABELS[name])&(new.hours==h)].iloc[0]
                b=old[(old.model==LABELS[name])&(old.hours==h)].iloc[0]
                comparison.append({'Model':LABELS[name],'Hours':h,'Original MAE (mm)':b.mae_mm,
                    'Tuned MAE (mm)':a.mae_mm,'MAE change (%)':f'{100*(a.mae_mm/b.mae_mm-1):+.1f}',
                    'Original RMSE (mm)':b.rmse_mm,'Tuned RMSE (mm)':a.rmse_mm})
        body += '<h3>Did tuning improve the reused winter benchmark?</h3>'+table(pd.DataFrame(comparison))
        body += '<p class="small">Negative MAE change is an improvement. The new weights were frozen before these evaluations. Eight-member forecasts and identical dates, verification masks and inference noise protocol are used for the original-versus-tuned comparison. Traditional controls retain the original benchmark timing: 10:50 UTC radar issue and the archived 06 UTC HRRR cycle.</p>'
        body += '<p><strong>No universal replacement emerged.</strong> ConvLSTM and NowcastNet reduce mean absolute error relative to their original checkpoints; DGMR changes little. The original DGMR checkpoint still has lower 1-hour and 3-hour spatial MSE than any newly selected checkpoint. ConvLSTM remains the winner of the frozen validation rule; within the new benchmark results, NowcastNet has the lowest 1-hour MSE and DGMR the lowest 3-hour MSE. These differing rankings show why the metric and forecast window must be specified.</p>'
        controls=old[~old.model.isin(LABELS.values())]
        benchmark=pd.concat([new,controls],ignore_index=True)
        benchmark=benchmark[benchmark.hours.isin([1,3])][['model','hours','mae_mm','rmse_mm','crps_mm']]
        body += '<details><summary>New checkpoints compared with traditional controls</summary><p>Errors are averaged over valid pixels and then over cases, in millimetres of accumulated water. This spatial metric differs from first averaging the entire region and then scoring that single mean.</p>'+table(benchmark.rename(columns={'model':'Method','hours':'Hours','mae_mm':'MAE (mm)','rmse_mm':'RMSE (mm)','crps_mm':'CRPS (mm)'}))+'</details>'
        probabilistic=[]
        for name in NAMES:
            for h in [1,3]:
                a=new[(new.model==LABELS[name])&(new.hours==h)].iloc[0]
                b=old[(old.model==LABELS[name])&(old.hours==h)].iloc[0]
                probabilistic.append({'Model':LABELS[name],'Hours':h,
                    'Original CRPS (mm)':b.crps_mm,'Tuned CRPS (mm)':a.crps_mm,
                    'Original event CSI':b.csi,'Tuned event CSI':a.csi})
        body += '<details><summary>Ensemble and precipitation-event performance</summary><p>CRPS measures error of the forecast ensemble (lower is better). CSI measures detection of pixels accumulating at least 1 mm of liquid precipitation over the stated window (higher is better); this is not a powder-day or snow-depth definition. These measures were not used to select the tuned checkpoints.</p>'+table(pd.DataFrame(probabilistic))+'</details>'
        paired=pd.read_csv(TUNE/'paired_comparisons.csv')
        paired=paired[(paired.reference=='original checkpoint')&(paired.metric.isin(['mae','mse']))&(paired.hours.isin([1,3]))]
        body += '<details><summary>Paired uncertainty for the error change</summary><p>Seven-day block-bootstrap 95% intervals within the reused winter; an interval crossing zero does not distinguish improvement from zero at this level. MAE differences use mm; MSE differences use mm². These intervals do not measure uncertainty across future winters.</p>'+table(paired[['model','hours','metric','difference','ci95_low','ci95_high']])+'</details>'
    else:body += '<p>Choices are frozen. Post-selection benchmark evaluation is still running.</p>'
    for name in ['plan.json','final_selection.json','evaluation_audit.json','integrity_audit.json','orchestration_amendment.json','checkpoint_losses.csv','final_checkpoint_losses.csv','short_summary.csv','paired_comparisons.csv']:
        p=TUNE/name
        if p.exists():shutil.copy2(p,OUT/('tuning_'+name))
    body += '<p class="downloads"><a href="tuning_plan.json">Frozen experiment plan</a> · <a href="tuning_final_selection.json">Frozen model selection</a> · <a href="tuning_all_runs.csv">All training runs</a> · <a href="tuning_training_histories.csv">Complete new training histories</a> · <a href="tuning_model_comparison.csv">Model comparison</a></p>'
    if (TUNE/'evaluation_audit.json').exists():
        body += '<p class="downloads"><a href="tuning_short_summary.csv">New benchmark scores</a> · <a href="tuning_paired_comparisons.csv">All paired error intervals</a> · <a href="tuning_evaluation_audit.json">Post-selection evaluation audit</a></p>'
    if (TUNE/'integrity_audit.json').exists():
        body+='<p class="downloads"><a href="tuning_integrity_audit.json">Checkpoint and training integrity audit</a> · <a href="tuning_orchestration_amendment.json">Recorded scheduling adjustment</a></p>'
    return body+'</section>'


def transfer_section():
    folder=ROOT/'outputs/radar-transfer'
    audit_path=folder/'crop_quality_audit.json'
    if not audit_path.exists():return ''
    audit=json.loads(audit_path.read_text())
    rows=[]
    for r in audit['summary']:
        rows.append({'Mountain':{'stowe':'Stowe','jay':'Jay Peak','vail':'Vail'}[r['mountain']],
            'Station RQI range':f"{r['station_rqi_min']:.2f}–{r['station_rqi_max']:.2f}",
            'Station snapshots passing RQI ≥ 0.5':f"{r['station_rqi_ge_05_count']} / {r['cases']}",
            'Mean regional target-valid area':f"{100*r['mean_region_target_fraction']:.1f}%"})
    for name in ['crop_quality_probe.csv','crop_quality_audit.json','public_archive_probe.json']:
        if (folder/name).exists():shutil.copy2(folder/name,OUT/('transfer_'+name))
    complete=''
    prepared=folder/'prepared'
    verification=prepared/'verification_audit.json'
    if verification.exists():
        v=json.loads(verification.read_text())
        assert v['status']=='verified_complete' and not v['source_hash_mismatches']
        w=pd.DataFrame(v['winter_summary'])
        totals=[]
        for station,region in [('stowe','vermont'),('jay','vermont'),('vail','vail')]:
            d=w[w.region==region]
            key=station+'_forecast_1h_eligible'
            counts={split:int(d.loc[d.split==split,key].sum()) for split in ['train','validation','test']}
            totals.append({'Mountain':{'stowe':'Stowe','jay':'Jay Peak','vail':'Vail'}[station],
                'Regional eligible windows':int(d.original_protocol_eligible.sum()),
                'Local 1h radar targets':int(d[key].sum()),'Training targets':counts['train'],
                'Development targets':counts['validation'],'Historical test targets':counts['test']})
        complete='<h3>Full four-winter datasets prepared and checked</h3><p>Both regional datasets now contain all 725 scheduled morning cases, with unavailable observations retained as missing. Each case stores nine input frames and twenty target frames, coordinates, quality masks and provenance. These are ready for model preparation; no new mountain models have been trained in this step.</p>'+table(pd.DataFrame(totals))
        complete+=f'<p class="small">Local targets require complete forecast inputs, eligible regional verification, and all seven native-point radar rates passing the quality mask during 11:00–12:00 UTC. Stowe and Jay share the same Vermont sequences and must remain together in storm/winter splits. The files total {v["case_npz_bytes"]/1024**2:.1f} MiB; {v["source_files_rehashed_and_matched"]:,} source files passed hash verification.</p>'
        png=prepared/'one_hour_target_availability.png'
        if png.exists():complete+=image(png,'Radar target availability across the Vail and Vermont regions over all 725 scheduled mornings')
        for name in ['verification_audit.json','verified_winter_summary.csv','verified_case_inventory.csv']:
            shutil.copy2(prepared/name,OUT/('transfer_full_'+name))
        complete+='<p class="downloads"><a href="transfer_full_verified_winter_summary.csv">Full counts by winter</a> · <a href="transfer_full_verified_case_inventory.csv">Every case and quality flag</a> · <a href="transfer_full_verification_audit.json">Full preparation audit</a></p>'
    return '''<section id="radar-transfer"><div class="eyebrow">Public data · Other mountains</div><h2>Vermont is the stronger next transfer candidate</h2>
    <p>The retained MRMS raw files are continental-US grids, so the existing four winters of morning sequences can be cropped for Vail, Stowe and Jay Peak without another bulk radar download. Public <a href="https://registry.opendata.aws/noaa-mrms-pds/">MRMS</a> and <a href="https://registry.opendata.aws/noaa-hrrr-pds/">HRRR</a> archives also provide a path to additional dates and issue times. These archives contain precipitation and weather fields; they do not establish independent hourly snowfall-depth observations at each resort.</p>'''+complete+'<details><summary>Initial calendar-sampled radar quality check</summary>'+table(pd.DataFrame(rows))+'''
    <p><strong>Verified sample:</strong> twelve calendar-selected snapshots per mountain, December 15, February 15 and April 15 in each of four winters. The same decoding and quality thresholds as Alta were used. This is a feasibility probe, not a full archive-completeness audit or an accuracy test. Its area percentages use separate mountain-centered crops; the completed Vermont dataset uses one shared domain.</p>
    </details>
    <p><strong>Stowe and Jay Peak:</strong> start with a shared Vermont radar model and separate station calibration. Both locations pass the sampled target-quality threshold; their overlapping domains should share storm information. Stowe uses the NOAA Mount Mansfield station, while Jay uses a NOAA station at approximately 561 m elevation; these are station targets, not resort-wide or summit totals. Their daily records can evaluate daily snowfall. Hourly fresh-snowfall depth still requires independent measurements.</p>
    <p><strong>Vail:</strong> the location has low radar quality, consistent with the difficulty of observing mountain precipitation. Zero local targets means reliable radar verification is unavailable under the unchanged quality threshold; it says nothing about whether snow fell. Retain the quality mask and combine surrounding radar with HRRR and validated surface observations.</p>
    <p>Raw <a href="https://registry.opendata.aws/noaa-nexrad/">NEXRAD Level II</a> is another public option: KCXX for northern Vermont, with KTYX as a supplementary view; KGJX and KFTG around Vail. Station availability does not guarantee visibility of shallow snowfall at the resort. The current archive bucket is <code>unidata-nexrad-level2</code>. Our saved data remain one morning sequence per day; all-hour training needs additional windows.</p>
    <p class="downloads"><a href="transfer_crop_quality_probe.csv">Quality samples</a> · <a href="transfer_crop_quality_audit.json">Crop audit</a> · <a href="transfer_public_archive_probe.json">Verified public archive endpoints</a></p></section>'''
