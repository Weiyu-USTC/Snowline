"""Calibrate hourly HRRR snow water against monthly Main Lodge fresh-snow depth."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PATH = Path('outputs/mammoth-hourly/depth.json')
PACIFIC = ZoneInfo('America/Los_Angeles')
COLD_THRESHOLD_C = -5.
MM_PER_INCH = 25.4
SOURCE = 'https://www.mammothmountain.com/on-the-mountain/snowfall-history'
TARGET = 'Mammoth Mountain Main Lodge'


def _seasons(root):
    # Use the newest downloaded table, including any corrections to prior months.
    directory = root / 'data/dashboard/seasonal/raw/mammoth'
    dbpath = directory / 'main_lodge.sqlite3'
    if dbpath.is_file():
        with sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro', uri=True) as db:
            row = db.execute('SELECT payload FROM latest WHERE id=1').fetchone()
        if row:
            return json.loads(row[0])['seasons']
    return json.loads((directory / 'main_lodge.json').read_text())['seasons']


def _months(root):
    import pandas as pd
    df = pd.read_csv(root/'data/mammoth/weather/history.csv.gz')
    timekeys = ['start_utc', 'end_utc', 'run_utc', 'available_utc']
    for key in timekeys:
        df[key] = pd.to_datetime(df[key], utc=True)
    if not (df.end_utc - df.start_utc).eq(pd.Timedelta(hours=1)).all():
        raise ValueError('A weather row has an incorrect hourly interval.')
    if not (df.end_utc - df.run_utc).eq(pd.to_timedelta(df.lead, unit='h')).all():
        raise ValueError('HRRR lead and valid time differ.')
    df = df[(df.available_utc <= df.start_utc) & (df.run_utc <= df.available_utc)
            & df.snow_water_mm.ge(0) & df.temperature_c.notna()]
    selected = df.sort_values(['start_utc', 'run_utc']).drop_duplicates('start_utc', keep='last')
    local = selected.set_index('start_utc').tz_convert(PACIFIC).sort_index()
    history = _seasons(root)
    climatology = defaultdict(list)
    for season, row in history.items():
        if int(season) > 2021:
            continue
        for i, amount in enumerate(row['months']):
            if amount is not None:
                climatology[(9+i) % 12 + 1].append(amount)
    results = []
    for (year, month), values in local.groupby([local.index.year, local.index.month]):
        season = year if month >= 10 else year - 1
        if season < 2020 or season > 2024 or month not in (11, 12, 1, 2, 3, 4):
            continue
        start = pd.Timestamp(year=year, month=month, day=1, tz=PACIFIC)
        end = start + pd.offsets.MonthBegin(1)
        expected = pd.date_range(start, end, inclusive='left', freq='h')
        if not expected.equals(values.index):
            continue
        amount = history.get(str(season), {}).get('months', [None]*10)[(month - 10) % 12]
        if amount is None:
            continue
        cold = values.temperature_c <= COLD_THRESHOLD_C
        climo = climatology[month]
        if not climo:
            raise ValueError('No earlier resort archive for the monthly climatology.')
        results.append(dict(month=f'{year:04}-{month:02}', season=int(season), hours=len(expected),
                            split='train' if season <= 2021 else 'validation' if season == 2022 else 'test',
                            observed_inches=amount, cold_swe_mm=float(values.loc[cold, 'snow_water_mm'].sum()),
                            warm_swe_mm=float(values.loc[~cold, 'snow_water_mm'].sum()),
                            climatology_inches=sum(climo)/len(climo), climatology_years=len(climo)))
    return results


def _score(observed, predicted):
    errors = [p-o for p, o in zip(predicted, observed)]
    return dict(months=len(errors), rmse_inches=math.sqrt(sum(e*e for e in errors)/len(errors)),
                mae_inches=sum(abs(e) for e in errors)/len(errors), bias_inches=sum(errors)/len(errors))


def train(root=ROOT):
    import numpy as np
    from scipy.optimize import nnls
    root = Path(root)
    rows = _months(root)
    training = [r for r in rows if r['split'] == 'train']
    if any(sum(r['split'] == s for r in rows) < 3 for s in ('train', 'validation', 'test')):
        raise ValueError('Insufficient complete months to fit and validate the depth estimate.')
    x = np.array([[r['cold_swe_mm'], r['warm_swe_mm']] for r in training])
    y = np.array([r['observed_inches']*MM_PER_INCH for r in training])
    total = x.sum(axis=1)
    if not np.dot(total, total):
        raise ValueError('No training snowfall water.')
    single = float(max(0., np.dot(total,y)/np.dot(total,total)))
    rat = nnls(x,y)[0].tolist()
    ratios = {'single_ratio': [single,single], 'temperature_ratios': rat, 'fixed_10_to_1': [10.,10.]}
    scores = {}
    for row in rows:
        for name,(cold,warm) in ratios.items():
            row[name] = (cold*row['cold_swe_mm'] + warm*row['warm_swe_mm'])/MM_PER_INCH
    for group in ('train','validation','test'):
        sample = [r for r in rows if r['split']==group]
        scores[group] = {name: _score([r['observed_inches'] for r in sample], [r[name] for r in sample])
                         for name in [*ratios,'climatology_inches']}
    # Freeze the selection using the intervening winter, before reporting holdout errors.
    # Keep the physical 10:1 conversion if it beats both fitted candidates on validation.
    winner = min(ratios, key=lambda name: scores['validation'][name]['rmse_inches'])
    for row in rows:
        row['estimate_inches'] = row[winner]
    model = dict(version=1, target=TARGET, quantity='fresh_snow_depth', unit='inches',
                 method_id=winner, ratios_cold_warm=ratios[winner], candidates=ratios,
                 threshold_c=COLD_THRESHOLD_C, scores=scores, months=rows,
                 trained_seasons=['2020/21','2021/22'], validation_season='2022/23', test_seasons=['2023/24','2024/25'],
                 timing='Hourly timing follows issued HRRR snowfall water near Main Lodge. Main Lodge validation is monthly.',
                 methodology='Only complete Pacific calendar months, including clock changes. Each hour uses the newest archived HRRR run available at its start, with a three-hour release allowance. Snow ratios are constrained nonnegative; a fixed 10:1 ratio can win validation. Climatology uses the resort archive through 2021/22.',
                 caveat='Depth is a proxy estimated from forecast snow water at a different elevation. '
                        'No hourly Main Lodge fresh-snow depth accuracy or interval has been established.',
                 resort_url=SOURCE, generated_at=datetime.now(timezone.utc).isoformat())
    path = root/PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(model, f, indent=2, allow_nan=False)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)
    return model


def read_model(root=ROOT):
    root = Path(root)
    for path in (root/PATH, root/'artifacts/mammoth-hourly/depth.json'):
        if path.is_file():
            model = json.loads(path.read_text())
            if model['version'] != 1 or model['target'] != TARGET:
                raise ValueError('Invalid Main Lodge depth calibration.')
            return model
    raise FileNotFoundError('Main Lodge depth calibration has not been installed.')


def predict_depth(model, weather):
    """Estimate inches of Main Lodge fresh-snow depth for each weather forecast hour."""
    ratios = [float(v) for v in model['ratios_cold_warm']]
    if len(ratios) != 2 or any(v < 0 or not math.isfinite(v) for v in ratios):
        raise ValueError('Invalid fitted snow ratios.')
    snow, temp, result = [], [], []
    for i,row in enumerate(weather):
        amount, t = row['snow_water_mm'], row['temperature_c']
        if amount is not None and t is not None:
            if i != len(snow) or amount < 0 or not math.isfinite(amount) or not math.isfinite(t):
                raise ValueError('Invalid native HRRR sequence.')
            snow.append(amount)
            temp.append(t)
        elif len(snow) >= 36 and amount is None and t is None:
            # Match the SWE model’s native-run tail (not a measured HRRR hour).
            gap = i - len(snow) + 1
            amount = sum(snow[-3:])/3 * math.exp(-gap/6.)
            t = sum(temp[-3:])/3
        else:
            raise ValueError('Hourly depth estimate lacks the required snowfall water or temperature.')
        result.append(amount * ratios[0 if t <= model['threshold_c'] else 1] / MM_PER_INCH)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    model = train(parser.parse_args().root)
    print(json.dumps({k:model[k] for k in ('method_id','ratios_cold_warm','scores')}, indent=2))
