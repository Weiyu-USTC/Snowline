"""Package the frozen Vail event-model experiment and verify every member."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/vail-storm-models'
DEST=ROOT.parent/'vail-storm-models-demo.zip'


def main():
    environment={'python':platform.python_version(),'platform':platform.platform(),
                 'packages':{name:importlib.metadata.version(name) for name in
                             ['numpy','pandas','scipy','scikit-learn','matplotlib','torch','joblib']}}
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2))
    names=['run_vail_storm_models.py','vail_storm_neural.py','vail_storm_statistical.py',
           'build_vail_storm_report.py','build_vail_powder_forecast.py','validate_vail_storm_artifacts.py','package_vail_storm_demo.py',
           'fetch_vail_caic.py','vail_hourly_model.py','vail_baselines.py','advanced_snow_models.py',
           'deep_dillon.py','build_vail_hourly_report.py','summarize_vail_evaluation.py',
           'validate_vail_artifacts.py']
    paths=[ROOT/'src'/name for name in names]
    for folder in ['data/vail_hourly','outputs/vail-hourly','outputs/vail-storm-models','outputs/vail-storm-review']:
        paths+=list((ROOT/folder).rglob('*'))
    paths+=[ROOT/'data/roni_noaa.csv',ROOT/'requirements.txt',ROOT/'requirements-deep.txt']
    paths+=list((ROOT/'research').glob('vail*.txt'))
    paths+=[ROOT/'research/storm_point_process_review.txt',ROOT/'research/front_conditioned_storm_review.txt']
    paths+=list((ROOT/'tests').glob('test_vail*.py'))
    paths=sorted(set(p for p in paths if p.is_file() and '__pycache__' not in p.parts))
    snapshot=OUT/'code_snapshot.json'
    snapshot.write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in paths if p.suffix=='.py'},indent=2))
    if snapshot not in paths:paths.append(snapshot)
    instructions='''VAIL EVENT-MODEL RESEARCH DEMO — NOVEMBER 4, 2026

Open outputs/vail-storm-models/storm_model_report.html.
All figures and reports work offline. forecast_daily.csv contains 57 dates,
November 5–December 31, 2026; forecast_paths.npz contains 1,024 paths/model.

MEASUREMENT AND MODEL SCOPE
The target is CAIC CAVMM's reported Sno24 sensor endpoint at 04:00 on the source
display clock. It is NOT established as daily fresh resort snowfall. Active
means this reading is at least one inch, not a verified meteorological event.
Do not sum endpoints into monthly resort snowfall. Reset timing, offsets,
publication lag and clock offset are unresolved. Explicit artifact windows
are masked in both predictors and targets; missing observations are not zeros.

The archive spans seven calendar years with gaps and six usable partial winters.
Four new daily state/duration models use known calendar variables and, for the
weather variants, 168 hours of observed weather. No future observed weather or
issued atmospheric forecasts enter. This first event experiment does not fit
continuous-time Hawkes events or normalized smooth pulses with storm totals.
Revised historical RONI is not release-vintage ENSO data.

RESULT
Overall neural-event RMSE is 2.470 versus climatology 2.484 and previous TCN
2.588 inches of the proxy. The overall gain against climatology is inconclusive
under six-winter bootstrap; the chronological check is essentially tied.
Short-range duration models improve errors modestly. Neural onset risks are
overpredicted. Neural nominal 80% interval coverage is 89.2% shuffled and 74.7%
chronological; ranges are not calibrated. These are comparisons to statistical
baselines, not archived operational meteorological forecasts.

REPRODUCE FROM THE EXTRACTED ROOT
  python3 -m pip install -r requirements.txt -r requirements-deep.txt
  python3 src/run_vail_storm_models.py --mode run
  python3 src/run_vail_storm_models.py --mode summarize
  python3 src/build_vail_powder_forecast.py
  python3 src/build_vail_storm_report.py
  python3 src/validate_vail_storm_artifacts.py
  python3 -m unittest discover -s tests -p 'test_vail_storm*.py'
  python3 -m unittest discover -s tests -p 'test_vail_powder.py'

The saved public snapshot needs no network. Selection is nested within whole-
winter folds; a latest-two-winter chronological evaluation is also included.
All six approaches share 342 test origins and 19,632 valid pairs. Statistical
models fit eligible one-step transitions (942 across all history); neural fits
use multi-horizon origins. Training objectives and eligible examples differ.

--mode rescore refits frozen selected configurations, without searching again.
--mode summarize updates tables from saved predictions without any fitting.
Saved checkpoints, source hashes, selection logs, tests and restoration checks
are included. environment.json records dependencies; code_snapshot.json records
source hashes. Earlier outputs/vail-hourly files are the unchanged comparator.
The inherited Dillon modules provide generic utilities; no Dillon model weights
are used. outputs/vail-storm-review records the proposal before these fits.
'''
    prefix='vail-storm-models-demo/'
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in paths:archive.write(path,prefix+str(path.relative_to(ROOT)))
        archive.writestr(prefix+'README.txt',instructions)
    with zipfile.ZipFile(DEST) as archive:
        assert archive.testzip() is None
        for path in paths:
            assert hashlib.sha256(archive.read(prefix+str(path.relative_to(ROOT)))).digest()==hashlib.sha256(path.read_bytes()).digest()
        count=len(archive.namelist())
    print(json.dumps({'archive':str(DEST),'files':count,'bytes':DEST.stat().st_size,
                      'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
