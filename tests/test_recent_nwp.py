"""Forecast information-boundary and amount/window checks for the JMA adapter."""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import fetch_recent_nwp as reader

class DailyForecastTests(unittest.TestCase):
    def fixture(self,offset=2):
        t=pd.date_range('2023-11-01T12:00Z',periods=24,freq='h')
        d=pd.DataFrame({'site':'alta','valid_time_utc':t,'nominal_offset_days':offset,'temperature_2m':-2.,'precipitation':1.,'wind_speed_10m':4.,'wind_direction_10m':270.,'relative_humidity_2m':80.,'pressure_msl':1000.,'snowfall':.7})
        d['implied_source_time_upper_bound_utc']=t-pd.Timedelta(days=offset)
        return d
    def derive(self,d):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(reader,'OUT',Path(folder)):reader.derive_daily(d)
            return pd.read_csv(Path(folder)/'jma_gsm_daily_conservative.csv').query('window_end_hour_utc == 11').iloc[0]
    def test_precip_window_and_units(self):
        r=self.derive(self.fixture());self.assertEqual(r.target_date,'2023-11-02');self.assertEqual(r.hours,24);self.assertEqual(r.precipitation_mm,24);self.assertAlmostEqual(r.raw_model_snowfall_inches,24*.7/2.54);self.assertEqual(r.temperature_mean_c,-2)
    def test_every_hour_was_forecast_before_issue(self):
        for offset,lead in reader.LEADS.items():
            r=self.derive(self.fixture(offset));self.assertEqual(r.daily_lead_days,lead);self.assertEqual(r.source_age_buffer_hours,24);self.assertEqual(pd.Timestamp(r.issue_time_utc)-pd.Timestamp(r.latest_implied_source_time_utc),pd.Timedelta(days=1))
    def test_missing_precip_is_not_zero(self):
        d=self.fixture();d.loc[5,'precipitation']=np.nan;r=self.derive(d);self.assertTrue(pd.isna(r.precipitation_mm));self.assertEqual(r.complete_core_hours,23)
    def test_wind_direction_to_vector(self):
        r=self.derive(self.fixture());self.assertAlmostEqual(r.wind_u_mean_ms,4.);self.assertAlmostEqual(r.wind_v_mean_ms,0.,places=10)

if __name__=='__main__':unittest.main()
