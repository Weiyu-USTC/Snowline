"""Map audited fixed-offset JMA forecast aggregates to the benchmark schema."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def prepare():
    path=ROOT/'data/nwp_recent/jma_gsm_daily_conservative.csv'
    d=pd.read_csv(path);d=d[d.complete_core_hours==24].copy()
    d=d.rename(columns={'target_endpoint_utc':'valid_end_utc','target_start_utc':'valid_start_utc','daily_lead_days':'lead_day',
        'temperature_mean_c':'forecast_temp_mean_c','temperature_min_c':'forecast_temp_min_c','temperature_max_c':'forecast_temp_max_c',
        'precipitation_mm':'forecast_precip_mm','cold_precipitation_mm':'forecast_temp_cold_precip_mm',
        'wind_u_mean_ms':'forecast_wind_u_ms','wind_v_mean_ms':'forecast_wind_v_ms','wind_speed_mean_ms':'forecast_wind_speed_mean_ms','wind_speed_max_ms':'forecast_wind_speed_max_ms',
        'humidity_mean_pct':'forecast_humidity_mean_pct'})
    # Pressure stays in the source table for provenance but does not enter the
    # wind-increment contrast, whose predictor names must isolate wind fields.
    d['archive_type']='operational_forecast';d['model_id']='JMA_GSM_fixed_offset_mosaic'
    d['provenance_note']='Archived fixed-offset hourly forecast values, not one coherent run; latest implied nominal source bound>=24h before issue; actual posting times unavailable'
    for alignment,hour in [('main',11),('minus1h',10),('plus1h',12)]:
        part=d[(d.window_end_hour_utc==hour)&d.lead_day.isin([1,3])].copy()
        part.to_csv(path.parent/f'jma_benchmark_{alignment}.csv',index=False)
    # Longer-lead comparison: require complete forecast covariates for every
    # lead on the same site/date, so lead6 is not compared on a different era.
    primary=d[d.window_end_hour_utc==11];counts=primary.groupby(['site','target_date']).lead_day.nunique();dates=counts[counts==3].index
    common=primary.set_index(['site','target_date']).loc[dates].reset_index()
    common.to_csv(path.parent/'jma_benchmark_lead6_common.csv',index=False)
    info=dict(source=str(path.relative_to(ROOT)),main_rows=int(len(d[(d.window_end_hour_utc==11)&d.lead_day.isin([1,3])])),common_lead6_rows=len(common),scope='Main1/3-day forecast targets;6-day supplementary on matched dates',physical_baseline='raw_model_snowfall_inches is the provider-derived7:1/temperature-phase snow conversion, not native JMA fresh-snow depth',ignored_predictors=['pressure_mean_hpa','pressure_change_hpa'],forecast_provenance='fixed-offset forecast mosaic; no true run-id/posting-vintage claim')
    (path.parent/'benchmark_adapter.json').write_text(json.dumps(info,indent=2));print(info)

if __name__=='__main__':prepare()
