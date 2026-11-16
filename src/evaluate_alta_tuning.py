"""Post-selection evaluation of the isolated November 7 tuning experiment.

Never changes model selection. The 2023/24 benchmark has already been inspected
in earlier work: it is a reused historical benchmark, not a fresh blind test.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from alta_radar_models import load_bundle, predict_ensemble
from diagnose_alta_checkpoint_loss import rate_errors
from evaluate_alta_nowcast import integral, field_scores, load_hrrr_run, block_difference

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/alta-nowcast'
OUT = BASE / 'tuning-20261107'
NAMES = ['convlstm', 'dgmr', 'nowcastnet']
LABELS = {'convlstm':'ConvLSTM', 'dgmr':'DGMR adaptation', 'nowcastnet':'NowcastNet adaptation'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def assert_identical_mask(actual, expected, context='verification'):
    """Counts alone cannot detect equal-size masks covering different pixels."""
    actual, expected = np.asarray(actual, bool), np.asarray(expected, bool)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise AssertionError(f'{context}: verification mask geometry differs')


def aligned_prediction_batches(reference_index, target_index, batch_size=4):
    """Retain original batch positions/seeds, including mixed dev/test boundary."""
    reference_index = np.asarray(reference_index, int)
    target_index = np.asarray(target_index, int)
    if len(set(reference_index)) != len(reference_index) or len(set(target_index)) != len(target_index):
        raise ValueError('Prediction indices must be unique')
    target_positions = {int(value): i for i, value in enumerate(target_index)}
    if not set(target_positions).issubset(reference_index):
        raise ValueError('Target case missing from original prediction schedule')
    for start in range(0, len(reference_index), batch_size):
        batch = reference_index[start:start+batch_size]
        destinations = [(offset, target_positions[int(value)]) for offset, value in enumerate(batch)
                        if int(value) in target_positions]
        if destinations:
            yield start, batch, destinations


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    selection_path = OUT / 'final_selection.json'
    selected = json.loads(selection_path.read_text())
    assert all(name in selected['models'] for name in NAMES)
    assert all(len(selected['models'][name]['runs'])==3 for name in NAMES), 'Complete all three seeds before evaluation'
    frozen_hash = digest(selection_path)
    audit = {'evaluation_started_utc':datetime.now(timezone.utc).isoformat(),
             'selection_sha256':frozen_hash,
             'selection_file':str(selection_path.relative_to(ROOT)),
             'test_status':'Previously inspected 2023/24 benchmark, reused after freezing new validation-only choices',
             'test_used_by_this_selection':False, 'members':8,
             'prediction_seed_scheme':'20261105 + 100000 + original development-plus-test batch start, batch size 4',
             'prediction_noise_alignment':'Original full prediction positions preserved; only batches intersecting test are evaluated',
             'models':{}, 'source_sha256':digest(__file__)}
    with np.load(BASE / 'learning_arrays.npz') as z:
        data = {key:z[key] for key in z.files}
    x,y = data['x'],data['y']
    ids,split = data['case_id'].astype(str),data['split'].astype(str)
    eligible = data['eligible'].astype(bool)
    dev = np.flatnonzero(eligible & (split=='validation'))
    train = np.flatnonzero(eligible & (split=='train'))
    test = np.flatnonzero(eligible & (split=='test'))
    groups = {'train':train, 'selection':dev[::4],
              'development_remainder':np.setdiff1d(dev,dev[::4]), 'reused_test':test}
    assert [len(groups[k]) for k in groups] == [352,42,125,180]
    assert not set(train) & set(test)
    assert not set(dev) & set(test)
    reference_index = np.flatnonzero(eligible & np.isin(split,['validation','test']))
    for name in NAMES:
        with np.load(BASE/'radar_predictions'/f'{name}_index.npz') as idx:
            assert np.array_equal(idx['source_index'],reference_index)
            assert np.array_equal(idx['case_id'].astype(str),ids[reference_index])
    schedule = list(aligned_prediction_batches(reference_index,test))
    cases = pd.read_csv(BASE/'case_inventory.csv',dtype={'case_id':str}).set_index('case_id')
    old = pd.read_csv(BASE/'short_case_metrics.csv',dtype={'case_id':str})
    assert not old.duplicated(['case_id','model','hours']).any()
    for (_,h),group in old.groupby(['model','hours']):
        assert set(group.case_id)==set(ids[test])
    baseline = old[old.model=='HRRR'].set_index(['case_id','hours'])
    hrrr = {str(ids[i]):load_hrrr_run(str(ids[i])) for i in test}
    assert all(v is not None for v in hrrr.values())
    iy = int(np.argmin(abs(data['latitude']-40.57607)))
    ix = int(np.argmin(abs(data['longitude']+111.63889)))
    pred_dir = OUT/'radar_predictions'
    pred_dir.mkdir(exist_ok=True)
    split_rows, final_rows, rows = [], [], []
    for name in NAMES:
        info = selected['models'][name]
        checkpoint = Path(info['representative_checkpoint'])
        if not checkpoint.is_absolute():checkpoint = ROOT/checkpoint
        checksum = digest(checkpoint)
        assert checksum==info['representative_checkpoint_sha256'], 'Frozen checkpoint hash mismatch'
        bundle = load_bundle(checkpoint)
        audit['models'][name] = {'checkpoint':str(checkpoint.relative_to(ROOT)),
            'checkpoint_sha256':checksum, 'best_step':bundle['best_step'],
            'seed':bundle['config'].seed}
        for group,index in groups.items():
            metrics = rate_errors(bundle['model'],x,y,index)
            if group=='selection':
                assert np.isclose(metrics['rate_mse_mm_h_squared'],bundle['best_validation_mse'],rtol=2e-5,atol=1e-7), 'Selected validation score failed replay'
            split_rows.append({'model_id':name,'model':LABELS[name], 'split':group,
                'cases':len(index), 'selected_step':bundle['best_step'], **metrics})
            print(json.dumps({'model':name,'split':group,**metrics}),flush=True)
        shape = (len(test),8,y.shape[1],*y.shape[-2:])
        pfile = pred_dir/f'{name}.npy'
        predictions = np.lib.format.open_memmap(pfile,mode='w+',dtype=np.float32,shape=shape)
        completed = 0
        written = set()
        for start,batch,destinations in schedule:
            p = predict_ensemble(bundle,x[batch],members=8,batch_size=4,seed=20361105+start)
            assert p.shape == (len(batch),*shape[1:]) and np.isfinite(p).all() and p.min()>=0
            for source_position,destination in destinations:
                predictions[destination] = p[source_position]
                assert destination not in written
                written.add(destination)
                completed += 1
            if start%40==0:
                predictions.flush()
                print(json.dumps({'model':name,'prediction_cases':completed,'total':len(test)}),flush=True)
        assert written==set(range(len(test)))
        predictions.flush()
        np.savez_compressed(pred_dir/f'{name}_index.npz',case_id=ids[test],
                            source_index=test,split=split[test],winter=data['winter'][test])
        for j,i in enumerate(test):
            cid = str(ids[i]); e = np.asarray(predictions[j],float)
            for h in [1,2,3]:
                truth = integral(y[i],h)
                forecast = integral(e,h)
                weather = hrrr[cid]['precip_mm']
                hr = weather['mrms64'][(weather['periods']>=6)&(weather['periods']<6+h)].sum(0)
                expected_mask = np.isfinite(y[i,1:6*h+2]).all(0)&np.isfinite(hr)
                mask = expected_mask & np.isfinite(forecast).all(0)
                assert_identical_mask(mask,expected_mask,f'{cid} {h}h {name}')
                scores = field_scores(forecast,truth,mask)
                assert scores is not None
                assert scores['pixels']==int(baseline.loc[(cid,h),'pixels']), (cid,h,'mask mismatch')
                actual = float(truth[iy,ix]) if mask[iy,ix] else np.nan
                point = float(forecast[:,iy,ix].mean())
                rows.append({'case_id':cid,'date':cases.loc[cid,'date'],'model_id':name,
                    'model':LABELS[name], 'hours':h,**scores, 'alta_truth_mm':actual,
                    'alta_predicted_mm':point,'alta_mae_mm':abs(actual-point),
                    'alta_mse':(actual-point)**2})
        audit['models'][name]['checkpoint_unchanged'] = digest(checkpoint)==checksum
        assert audit['models'][name]['checkpoint_unchanged']
        audit['models'][name]['prediction_sha256'] = digest(pfile)
        del bundle,predictions
        final_path = checkpoint.parent/'final.pt'
        final_hash = digest(final_path)
        final_bundle = load_bundle(final_path)
        for group in ['train','selection']:
            metrics = rate_errors(final_bundle['model'],x,y,groups[group])
            final_rows.append({'model_id':name,'model':LABELS[name],'split':group,
                'cases':len(groups[group]),'final_step':final_bundle['step'],**metrics})
        assert digest(final_path)==final_hash
        audit['models'][name]['final_checkpoint_sha256'] = final_hash
        del final_bundle
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'short_case_metrics.csv',index=False)
    controls = old[old.model.isin(['Persistence','Motion extrapolation','HRRR'])].copy()
    controls['model_id'] = controls.model.map({'Persistence':'persistence','Motion extrapolation':'motion','HRRR':'hrrr'})
    combined = pd.concat([frame,controls],ignore_index=True).sort_values(['case_id','hours','model'])
    assert not combined.duplicated(['case_id','hours','model']).any()
    assert len(combined)==180*3*6
    combined.to_csv(OUT/'combined_short_case_metrics.csv',index=False)
    pd.DataFrame(split_rows).to_csv(OUT/'checkpoint_losses.csv',index=False)
    pd.DataFrame(final_rows).to_csv(OUT/'final_checkpoint_losses.csv',index=False)
    summaries = []
    for (model,h),d in frame.groupby(['model','hours']):
        tp,fp,fn = d[['tp','fp','fn']].sum()
        summaries.append({'model':model,'hours':h,'cases':len(d),'mae_mm':d.mae.mean(),
            'mse_mm2':d.mse.mean(),'rmse_mm':np.sqrt(d.mse.mean()),'crps_mm':d.crps.mean(),
            'bias_mm':d.bias.mean(),'csi':tp/(tp+fp+fn) if tp+fp+fn else np.nan,
            'alta_cases':d.alta_truth_mm.notna().sum(), 'alta_mae_mm':d.alta_mae_mm.mean(),
            'alta_rmse_mm':np.sqrt(d.alta_mse.mean())})
    pd.DataFrame(summaries).to_csv(OUT/'short_summary.csv',index=False)
    comparisons = []
    for (model,h),d in frame.groupby(['model','hours']):
        d = d.set_index('case_id')
        for reference,refmodel in [('original checkpoint',model),('HRRR','HRRR'),('Persistence','Persistence')]:
            ref = old[(old.model==refmodel)&(old.hours==h)].set_index('case_id').loc[d.index]
            for metric in ['mae','mse','crps']:
                comparisons.append({'model':model,'hours':h,'reference':reference,'metric':metric,
                    **block_difference(d[metric].to_numpy()-ref[metric].to_numpy(),d.date)})
    pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv',index=False)
    assert digest(selection_path)==frozen_hash
    audit.update({'status':'complete','elapsed_seconds':time.monotonic()-started,
        'selection_unchanged':True,'test_cases':len(test),
        'evaluation_finished_utc':datetime.now(timezone.utc).isoformat()})
    (OUT/'evaluation_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'status':'complete','elapsed_seconds':audit['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
