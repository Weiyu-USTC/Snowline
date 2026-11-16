"""Issue station-calibrated forecasts from the saved November4 GFS MOS run."""
from pathlib import Path
import argparse,json
import pandas as pd
from threadpoolctl import threadpool_limits
import nwp_benchmark as nwp
ROOT=Path(__file__).resolve().parents[1]
def main(site):
    raw=pd.read_csv(ROOT/'data/nwp_archive_probe/mos_daily_features.csv')
    raw=raw[raw.site==site].copy()
    # Use exactly the same matched training cohort as the completed benchmark.
    train=pd.read_csv(ROOT/f'outputs/nwp-weather/mos/{site}/matched_input.csv')
    future=raw[pd.to_datetime(raw.forecast_reference_utc,utc=True)==pd.Timestamp('2026-11-04',tz='UTC')].copy()
    assert len(future)==2 and set(future.target_date)=={'2026-11-05','2026-11-06'}
    history=nwp.read_targets(ROOT,site)
    with threadpool_limits(limits=2):
        result=nwp.forecast_current(train,future,ROOT/f'outputs/nwp-weather/mos/{site}',history)
    print(result[result.model==nwp.NAMES[-1]].to_string(index=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',choices=['alta','vail','mansfield','jay'],required=True);a=p.parse_args();main(a.site)
