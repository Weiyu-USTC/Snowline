"""Reconstruct the saved Alta training sampler; never fit or mutate a model.

The original code used one seeded NumPy Generator only for minibatch choice.
This diagnostic recovers the selected cases at logged steps and describes their
target difficulty. It cannot isolate the causal contribution of batch difficulty
to a loss change because weights, stochastic forecasts and adversaries also change.
"""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/alta-nowcast'
OUT = BASE / 'learning-review'
NAMES = ['convlstm', 'dgmr', 'nowcastnet']
LABELS = {'convlstm':'ConvLSTM','dgmr':'DGMR adaptation','nowcastnet':'NowcastNet adaptation'}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = {r['model']:r for r in json.loads((BASE/'models/selected/recipe_selection.json').read_text())}
    bundles = {n:json.loads((BASE/f'models/selected/{n}.pt.json').read_text()) for n in NAMES}
    checkpoints = {n:BASE/f'models/selected/{n}.pt' for n in NAMES}
    checkpoint_hashes = {n:sha256(p) for n,p in checkpoints.items()}
    archived_numpy = json.loads((BASE/'environment.json').read_text())['packages']['numpy']
    assert np.__version__ == archived_numpy, 'NumPy version differs from the saved training environment.'
    with np.load(BASE/'learning_arrays.npz') as z:
        ids = z['case_id'].astype(str)
        train_index = np.flatnonzero(z['eligible'].astype(bool) & (z['split'].astype(str)=='train'))
        targets = z['y']
    sources = {}
    rows = []
    model_audits = {}
    sampled_sequences = {}
    for n in NAMES:
        bundle = bundles[n]
        config = bundle['config']
        saved = bundle['sampling_audit']
        recipe = selection[n]['selected']['recipe']
        executed = BASE/f'models/recipe_{recipe}/alta_radar_models.executed.py'
        orchestration = BASE/f'models/recipe_{recipe}/train_alta_radar.executed.py'
        source = executed.read_text()
        assert 'rng=np.random.default_rng(seed)' in source
        assert 'ix=rng.choice(len(xx),min(batch_size,len(xx)),replace=False,p=probabilities)' in source
        assert source.count('rng.choice(') == 1
        assert 'train_index=np.flatnonzero(eligible&(split=="train"))' in orchestration.read_text()
        split_audit = json.loads((BASE/f'models/selected/{n}_split_audit.json').read_text())
        assert split_audit['training_cases'] == ids[train_index].tolist()
        assert len(train_index) == saved['training_cases'] == 352
        probability = np.asarray(saved['sampling_probability'], dtype=np.float64)
        assert len(probability) == len(train_index) and np.isclose(probability.sum(),1)
        wet_case = np.asarray(saved['wet_pixel_fraction']) >= .1
        assert wet_case.sum() == saved['wet_case_count'] == 56
        rng = np.random.default_rng(config['seed'])
        records = {r['step']:r for r in bundle['training_log']}
        counts = np.zeros(len(train_index), dtype=int)
        sequence = []
        for step in range(1, config['steps']+1):
            ix = rng.choice(len(train_index), min(config['batch_size'],len(train_index)),
                            replace=False, p=probability)
            counts[ix] += 1
            sequence.append(ix.tolist())
            if step not in records:
                continue
            target = targets[train_index[ix]]
            valid = np.isfinite(target) & (target >= 0)
            values = np.clip(target[valid],0,128).astype(np.float64)
            row = records[step]
            rows.append({'model_id':n, 'model':LABELS[n], 'step':step, 'phase':row['phase'],
                'case_id_1':ids[train_index[ix[0]]], 'case_id_2':ids[train_index[ix[1]]],
                'training_local_index_1':int(ix[0]),'training_local_index_2':int(ix[1]),
                'target_mean_rate_mm_h':float(values.mean()),
                'target_dry_baseline_mse_mm_h_squared':float(np.mean(values**2)),
                'target_wet_fraction_above_0_1_mm_h':float(np.mean(values>.1)),
                'target_max_rate_mm_h':float(values.max()),'valid_target_pixels':len(values),
                'raw_training_objective':row['loss'],'discriminator_objective':row['discriminator_loss'],
                'validation_mse_mm_h_squared':row.get('validation_mse_mm_h_squared','')})
        sampled_sequences[n] = sequence
        model_audits[n] = {'seed':config['seed'],'optimizer_updates':config['steps'],
            'batch_size':config['batch_size'],'case_draw_slots':int(counts.sum()),
            'nominal_dataset_passes':float(counts.sum()/len(train_index)),
            'unique_training_cases_seen':int(np.count_nonzero(counts)),
            'training_cases_never_seen':ids[train_index[counts==0]].tolist(),
            'wet_training_case_count':int(wet_case.sum()),'wet_case_draw_slots':int(counts[wet_case].sum()),
            'draw_count_min':int(counts.min()),'draw_count_median':float(np.median(counts)),
            'draw_count_max':int(counts.max()),'case_draw_counts':dict(zip(ids[train_index].tolist(),counts.tolist())),
            'recipe':recipe,'selected_checkpoint_step':bundle['best_step']}
        sources[str(executed.relative_to(ROOT))] = sha256(executed)
        sources[str(orchestration.relative_to(ROOT))] = sha256(orchestration)
        sources[str((BASE/f'models/selected/{n}.pt.json').relative_to(ROOT))] = sha256(BASE/f'models/selected/{n}.pt.json')
    same_sequence = all(sampled_sequences[n] == sampled_sequences[NAMES[0]] for n in NAMES)
    assert same_sequence
    unchanged = all(sha256(p) == checkpoint_hashes[n] for n,p in checkpoints.items())
    assert unchanged
    with (OUT/'batch_difficulty.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    a = next(r for r in rows if r['model_id']=='convlstm' and r['step']==1350)
    b = next(r for r in rows if r['model_id']=='convlstm' and r['step']==1500)
    audit = {'generated_utc':datetime.now(timezone.utc).isoformat(),'retrained':False,
        'diagnostic':'Reconstructed deterministic minibatch sampling, not direct per-update inference.',
        'numpy_version':np.__version__,'archived_training_numpy_version':archived_numpy,
        'sampler_verified_against_archived_executed_source':True,
        'all_selected_models_share_identical_minibatch_sequence':same_sequence,
        'checkpoint_hashes_unchanged':unchanged,'checkpoint_sha256':checkpoint_hashes,
        'case_order_matches_saved_split_audits':True,'models':model_audits,'source_sha256':sources,
        'target_difficulty_definition':'Mean clipped physical target rate squared over finite nonnegative target pixels (error of an always-zero forecast). It is not the fitted model loss.',
        'step_1350':a,'step_1500':b,
        'last_batch_mean_rate_ratio':b['target_mean_rate_mm_h']/a['target_mean_rate_mm_h'],
        'last_batch_dry_mse_ratio':b['target_dry_baseline_mse_mm_h_squared']/a['target_dry_baseline_mse_mm_h_squared'],
        'cautions':['Target difficulty is a confounder, not a causal decomposition of observed loss changes.',
                    'Weights, stochastic model draws, and discriminator states change between logged steps.',
                    'Raw loss is one minibatch and different model families use differently scaled objectives.',
                    'Nominal dataset passes are draw slots divided by case count; weighted sampling is not full epochs.']}
    (OUT/'batch_difficulty_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:audit[k] for k in ['all_selected_models_share_identical_minibatch_sequence','step_1350','step_1500','last_batch_mean_rate_ratio','last_batch_dry_mse_ratio']},indent=2))


if __name__ == '__main__':
    main()
