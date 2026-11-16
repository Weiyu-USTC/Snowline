"""Describe held-out Vail errors for Nov-Dec target dates without refitting.

This is a calendar sensitivity analysis of existing out-of-fold forecasts,
not a new model-selection step or an independent test set. Multiple origins
can predict the same observation; inferential resampling remains by winter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vail_hourly_model import OUT, metrics, paired_bootstrap


KEYS = ['winter', 'origin', 'lead']


def comparison_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Require identical issue/lead/observation pairs across every model."""
    if rows.empty:
        raise ValueError('No held-out prediction rows')
    required = set(KEYS + ['model', 'date', 'actual_inches', 'predicted_inches'])
    if required - set(rows.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(rows.columns))}')
    if rows.duplicated(['model']+KEYS).any():
        raise ValueError('Repeated model/issue/lead predictions')
    result = rows.sort_values(KEYS+['model']).reset_index(drop=True).copy()
    reference = None
    for name, part in result.groupby('model'):
        part = part.set_index(KEYS)
        if not np.isfinite(part[['actual_inches', 'predicted_inches']].to_numpy()).all():
            raise ValueError(f'Nonfinite observation or prediction for {name}')
        if reference is None:
            reference = part[['date', 'actual_inches']]
        elif not part.index.equals(reference.index):
            raise ValueError(f'Unequal valid target pairs for {name}')
        elif not part[['date', 'actual_inches']].equals(reference):
            raise ValueError(f'Inconsistent target values or dates for {name}')
    return result


def november_december_targets(rows: pd.DataFrame) -> pd.DataFrame:
    """Filter by target/reporting date, not forecast issue date."""
    target_dates = pd.to_datetime(rows['date'])
    return comparison_rows(rows.loc[target_dates.dt.month.isin([11, 12])])


def by_winter_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for winter, subset in rows.groupby('winter'):
        result = metrics(subset)
        result.insert(0, 'winter', int(winter))
        output.append(result)
    return pd.concat(output, ignore_index=True)


def summarize(path: Path, output_directory: Path) -> dict:
    rows = comparison_rows(pd.read_csv(path))
    subset = november_december_targets(rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics(subset).to_csv(output_directory/'nov_dec_target_metrics.csv', index=False)
    by_winter_metrics(rows).to_csv(output_directory/'by_winter_metrics.csv', index=False)
    by_winter_metrics(subset).to_csv(output_directory/'nov_dec_by_winter_metrics.csv', index=False)
    if 'Hourly TCN' not in set(subset.model):
        raise ValueError('Hourly TCN rows required for the paired comparison')
    paired = paired_bootstrap(subset)
    (output_directory/'nov_dec_paired_error_bootstrap.json').write_text(json.dumps(paired, indent=2))
    # Each model has the same target pairs. Count one copy, not all models.
    one = subset[subset.model == 'Hourly TCN']
    winter_inventory = []
    for winter, part in one.groupby('winter'):
        observations = part[['date', 'actual_inches']].drop_duplicates()
        winter_inventory.append(dict(winter=int(winter), pairs=len(part),
                                     origins=int(part.origin.nunique()),
                                     unique_target_days=int(part.date.nunique()),
                                     first_target=str(part.date.min()), last_target=str(part.date.max()),
                                     first_origin=str(part.origin.min()), last_origin=str(part.origin.max()),
                                     distinct_days_at_least_one_inch=int((observations.actual_inches >= 1).sum())))
    summary = dict(
        source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        analysis='Descriptive Nov-Dec target-date sensitivity of existing out-of-fold predictions; no retraining or model selection',
        target='CAVMM reported 04:00 Sno24 proxy; not verified daily fresh snowfall',
        models=sorted(subset.model.unique().tolist()), eligible_winters=sorted(one.winter.unique().astype(int).tolist()),
        winters=int(one.winter.nunique()), pairs_per_model=len(one),
        origins=int(one.origin.nunique()), unique_target_days=int(one.date.nunique()),
        first_target=str(one.date.min()), last_target=str(one.date.max()),
        by_winter=winter_inventory,
        interpretation='Dates are reporting endpoints; missing/quarantined targets excluded equally. Multiple issue/lead pairs can reuse one observed day. Paired confidence ranges resample whole winters, not rows. No probability event scores computed because per-pair occurrence distributions were not saved.',
        quarantine_note='Autumn2023 is masked through Nov28; later Nov-Dec2023 targets can still appear. Absence of a whole winter is determined from retained predictions, not assumed from the quarantine.',
    )
    (output_directory/'nov_dec_evaluation_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=OUT/'crossvalidation_predictions.csv')
    parser.add_argument('--output-directory', type=Path, default=OUT)
    args = parser.parse_args()
    summary = summarize(args.input, args.output_directory)
    print(json.dumps(summary, indent=2))
    result = pd.read_csv(args.output_directory/'nov_dec_target_metrics.csv')
    print(result[result.leads == '1–60'][['model', 'pairs', 'winters', 'mse', 'rmse', 'mae', 'coverage80']].to_string(index=False))


if __name__ == '__main__':
    main()
