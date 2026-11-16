"""Package the reproducible Vail-only experiment and verify the archive."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/vail-hourly'
DEST = ROOT.parent/'vail-hourly-model-demo.zip'


def main():
    environment = {'python':platform.python_version(), 'platform':platform.platform(),
                   'packages':{name:importlib.metadata.version(name) for name in
                               ['numpy','pandas','scipy','scikit-learn','matplotlib','torch']}}
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2))
    sources = ['fetch_vail_caic.py','vail_hourly_model.py','vail_baselines.py',
               'build_vail_hourly_report.py','summarize_vail_evaluation.py',
               'advanced_snow_models.py','deep_dillon.py','package_vail_demo.py','validate_vail_artifacts.py']
    paths = [ROOT/'src'/name for name in sources]
    paths += list((ROOT/'data/vail_hourly').rglob('*'))
    paths += list(OUT.rglob('*'))
    paths += [ROOT/'data/roni_noaa.csv',ROOT/'requirements.txt',ROOT/'requirements-deep.txt']
    paths += [ROOT/'research/vail_data_audit.txt',ROOT/'research/vail_model_design_audit.txt']
    paths += list((ROOT/'tests').glob('test_vail*.py'))
    paths = sorted(set(p for p in paths if p.is_file() and '__pycache__' not in p.parts))
    snapshot = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.suffix=='.py'}
    code_snapshot = OUT/'code_snapshot.json'
    code_snapshot.write_text(json.dumps(snapshot,indent=2))
    if code_snapshot not in paths: paths.append(code_snapshot)
    instructions = '''VAIL HOURLY-INPUT / DAILY-ENDPOINT RESEARCH DEMO

Start with outputs/vail-hourly/vail_hourly_report.html.
The report and PNG figures work offline. Daily values are in forecast_daily.csv.

Important target qualification
This predicts the CAIC CAVMM Sno24 sensor reading at a fixed 04:00 display-clock
endpoint. It is NOT verified 24-hour fresh snowfall or an official Vail report.
Reset timing, sensor offsets and publication latency remain unresolved.
Do not sum these endpoint readings into monthly resort snowfall totals.

Data and model
Four Vail resort stations; 168 hours of observed weather; 12 causal convolutions.
Six partial historical winters; current input through November 4, 2026 04:00.
Forecast reporting endpoints November 5–December 31, 2026.
Whole-winter outer validation and inner model selection prevent overlapping
hourly windows crossing training/test boundaries. The comparisons are against
statistical baselines, not issued NWP or human weather forecasts.
The neural model did not establish an advantage over calendar climatology.
Current lagged RONI exceeds the historical fit range; its numerical feature is
clipped to that range and an unseen strong-El-Nino indicator is not activated.

Reproduce from the extracted root with Python and the listed dependencies:
  python3 -m pip install -r requirements.txt -r requirements-deep.txt
  python3 src/vail_hourly_model.py --mode run
  python3 src/summarize_vail_evaluation.py
  python3 src/build_vail_hourly_report.py
  python3 src/validate_vail_artifacts.py
  python3 -m unittest discover -s tests -p 'test_vail*.py'

Training consumes the saved public snapshot; no new network access is needed.
The fetcher can retrieve source tables, but use the frozen snapshot to reproduce
this result. See data/vail_hourly/README.txt for fields and source provenance.
Raw HTML, source URLs, checksums, quality masks, the fitted checkpoint, out-of-fold
predictions and training logs are included. environment.json records versions.

The inherited deep_dillon.py and advanced_snow_models.py supply generic neural
utilities only. Vail is trained independently; no Dillon model weights are used.
'''
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in paths:
            z.write(path,'vail-hourly-demo/'+str(path.relative_to(ROOT)))
        z.writestr('vail-hourly-demo/README.txt',instructions)
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        count=len(z.namelist())
    print(json.dumps({'archive':str(DEST),'files':count,'bytes':DEST.stat().st_size,
                      'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
