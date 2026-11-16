"""Aggregate the frozen Dillon observations without imputing missing snowfall."""
from pathlib import Path
import calendar
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'dillon-season-table'
MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'outputs/dillon_deep_daily_input.csv'
    daily = list(csv.DictReader(source.open()))
    grouped = {}
    for r in daily:
        grouped.setdefault(r['date'][:7], []).append(r)
    records = []
    for start in range(2005, 2027):
        for month in (MONTHS if start < 2026 else [10]):
            year = start if month >= 10 else start + 1
            key = f'{year:04d}-{month:02d}'
            usable = [r for r in grouped[key] if r['snowfall_usable'] == '1']
            raw = sum(int(float(r['snowfall_raw_mm'])) for r in usable)
            expected = calendar.monthrange(year, month)[1]
            records.append(dict(season=f'{start}/{str(start+1)[-2:]}', start_year=start,
                                date=f'{key}-01', month=month,
                                reported_mm=raw if usable else None,
                                usable_days=len(usable), expected_days=expected,
                                missing_days=expected-len(usable),
                                reported_inches=raw/25.4 if usable else None))
    historical = records[:-1]
    means = {m: sum(r['reported_inches'] for r in historical if r['month']==m and not r['missing_days']) /
             sum(r['month']==m and not r['missing_days'] for r in historical) for m in MONTHS}
    forecast_file = ROOT / 'outputs/shuffled_forecast_monthly.csv'
    forecast = [{k: v if k in ('model', 'month') else float(v) for k,v in row.items()}
                for row in csv.DictReader(forecast_file.open())]
    selected = json.loads((ROOT/'outputs/shuffled_snow_results.json').read_text())['production_selected_family']
    current = [records[-1]['reported_inches']]
    current += [next(r['predicted_full_month_inches'] for r in forecast if r['model']==selected and r['month']==f'2026-{m}') for m in (11,12)]
    current += [means[m] for m in MONTHS[3:]]
    wide = []
    for start in range(2005,2026):
        rs = [r for r in historical if r['start_year']==start]
        row = {'season':rs[0]['season']}
        for r in rs:
            name=calendar.month_abbr[r['month']]
            row[name+'_inches']=r['reported_inches']
            row[name+'_status']='missing' if r['usable_days']==0 else 'partial' if r['missing_days'] else 'observed'
        row['Oct_May_inches']=sum(r['reported_inches'] or 0 for r in rs)
        row['total_status']='partial_observed_subtotal' if any(r['missing_days'] for r in rs) else 'observed_total'
        wide.append(row)
    for name, vals, statuses, status in [('Historical mean',[means[m] for m in MONTHS],['complete_month_mean']*8,'sum_of_monthly_means'),
                                       ('2026/27 outlook',current,['observed','GRU_forecast','GRU_forecast']+['historical_mean_baseline']*5,'hybrid_outlook')]:
        row={'season':name}
        for m,v,s in zip(MONTHS,vals,statuses):
            row[calendar.month_abbr[m]+'_inches']=v
            row[calendar.month_abbr[m]+'_status']=s
        row['Oct_May_inches']=sum(vals)
        row['total_status']=status
        wide.append(row)
    with (OUT/'dillon_season_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(wide[0]))
        writer.writeheader(); writer.writerows(wide)
    result=dict(station='Dillon 1 E',station_id='USC00052281',prepared='2026-11-04',observation_cutoff='2026-11-01',
                source_url='https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00052281.dly',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                forecast_sha256=hashlib.sha256(forecast_file.read_bytes()).hexdigest(),
                months=MONTHS,records=records,means=means,forecast=forecast,selected_model=selected,wide=wide,
                hybrid_total=sum(current),missing_days=sum(r['missing_days'] for r in historical),
                notes=['Historical partial months are reported subtotals, not estimates. Missing May 2009 is unavailable.',
                       'Historical monthly means exclude incomplete months. Mean total is the sum of those means.',
                       '2026/27 combines observed October, selected GRU November–December, and historical-mean January–May.',
                       'Forecasts reuse frozen inputs through November 1; no weather refresh or new model training.',
                       'Totals cover October–May only, excluding June–September. Trace snowfall retained at reported zero.'])
    assert result['missing_days']==38
    assert sum(r['missing_days']>0 for r in historical)==7
    assert round(result['hybrid_total'],1)==105.8
    assert records[31]['date']=='2009-05-01' and records[31]['reported_inches'] is None
    (OUT/'table_data.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'historical_seasons':21,'missing_days':result['missing_days'],'hybrid_total':result['hybrid_total'],'selected_model':selected}))


if __name__ == '__main__':
    main()
