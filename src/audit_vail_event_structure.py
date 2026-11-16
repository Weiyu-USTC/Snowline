"""Descriptive event clustering in screened Vail endpoint readings; no fitting.

The binary indicator here is a >=1-inch reported Sno24 endpoint, NOT a verified
fresh-snow day or identified meteorological storm. Missing dates split runs.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/vail-storm-review'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT/'outputs/vail-hourly/daily_target_0400.csv'
    frame = pd.read_csv(source, parse_dates=[0]).set_index('reporting_endpoint_source_clock')
    snow = frame.reported_sno24_inches
    assert snow.loc['2020-03-20':'2020-05-31'].isna().all()
    assert snow.loc['2023-10-01':'2023-11-28'].isna().all()
    rows, runs = [], []
    for winter in [2019, 2020, 2022, 2023, 2024, 2025]:
        dates = pd.date_range(f'{winter}-10-01 04:00', f'{winter+1}-04-30 04:00')
        values = snow.reindex(dates).to_numpy(float)
        observed = np.isfinite(values)
        states = np.where(observed, (values >= 1).astype(float), np.nan)
        pair_mask = observed[:-1] & observed[1:]
        prior, next_ = states[:-1][pair_mask], states[1:][pair_mask]
        counts = {f'n_{a}{b}':int(((prior==a)&(next_==b)).sum()) for a in [0,1] for b in [0,1]}
        row = dict(winter=f'{winter}/{winter+1}', observed_days=int(observed.sum()),
                   days_ge1=int(np.nansum(states)), adjacent_pairs=int(pair_mask.sum()), **counts)
        row['p_ge1_next_given_ge1_today'] = counts['n_11']/(counts['n_10']+counts['n_11'])
        row['p_ge1_next_given_below1_today'] = counts['n_01']/(counts['n_00']+counts['n_01'])
        rows.append(row)
        i = 0
        while i < len(states):
            if not np.isfinite(states[i]):
                i += 1
                continue
            j = i+1
            while j<len(states) and np.isfinite(states[j]) and states[j]==states[i]:
                j += 1
            left_bounded = i>0 and np.isfinite(states[i-1])
            right_bounded = j<len(states) and np.isfinite(states[j])
            runs.append(dict(winter=row['winter'], state='at_least_1_inch' if states[i] else 'below_1_inch',
                             start=str(dates[i].date()), end=str(dates[j-1].date()),
                             observed_length_days=j-i, left_censored=not left_bounded,
                             right_censored=not right_bounded,
                             complete=bool(left_bounded and right_bounded)))
            i = j
    table = pd.DataFrame(rows)
    run_table = pd.DataFrame(runs)
    counts = table[['n_00','n_01','n_10','n_11']].sum().to_dict()
    pooled = dict(observed_days=int(table.observed_days.sum()), days_ge1=int(table.days_ge1.sum()),
                  adjacent_pairs=int(table.adjacent_pairs.sum()), **{k:int(v) for k,v in counts.items()},
                  p_ge1_next_given_ge1_today=float(counts['n_11']/(counts['n_10']+counts['n_11'])),
                  p_ge1_next_given_below1_today=float(counts['n_01']/(counts['n_00']+counts['n_01'])))
    run_summary = []
    for state, group in run_table.groupby('state'):
        complete = group[group.complete]
        run_summary.append(dict(state=state, observed_runs=len(group), complete_runs=len(complete),
                                censored_runs=int((~group.complete).sum()),
                                complete_length_mean=float(complete.observed_length_days.mean()),
                                complete_length_median=float(complete.observed_length_days.median()),
                                complete_length_max=int(complete.observed_length_days.max())))
    summary = dict(target='Screened CAVMM reported 04:00 Sno24 proxy; >=1 inch descriptive threshold only',
                   seasonal_domain='Oct–Apr, six partial winters; current 2026/27 excluded',
                   pair_policy='Only consecutive calendar days, both observed, in the same winter; missing/quarantined values never become zero',
                   run_policy='Missing dates and winter edges censor runs. Complete-run statistics require an observed opposite-state day at both ends; they underrepresent long censored spells.',
                   inference_limit='Descriptive association, not causal front evidence, proof of Poisson/Hawkes structure, true storm identification or predictive skill. Sensor resets and offsets can create persistence.',
                   pooled=pooled, complete_run_summary=run_summary)
    table.to_csv(OUT/'proxy_event_diagnostics.csv', index=False)
    run_table.to_csv(OUT/'proxy_event_runs.csv', index=False)
    (OUT/'proxy_event_diagnostics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__=='__main__':
    main()
