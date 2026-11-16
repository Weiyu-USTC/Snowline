"""Information-boundary and interval checks for issued MOS inputs."""
from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from build_mos_features import daily_features,instant_features


def fixture():
 h=np.r_[np.arange(6,61,3),66,72].astype(float)
 d=pd.DataFrame({'lead_hours':h,'temperature_c':h,'dewpoint_c':h-3,'wind_speed_mps':np.ones(len(h))*2,'wind_u_mps':np.ones(len(h))*-2,'wind_v_mps':np.zeros(len(h)), 'precip_probability_6h':np.where((h%6==0)&(h>=12),.4,np.nan),'qpf_category_6h':np.where((h%6==0)&(h>=12),2.,np.nan)})
 return d

class MOSIntervalTests(unittest.TestCase):
 def test_bounded_linear_mean_and_intervals(self):
  d=fixture();r,why=daily_features(d,11,35);self.assertIsNone(why)
  self.assertAlmostEqual(r['forecast_temp_mean_c'],23.)
  self.assertAlmostEqual(r['forecast_temp_change_c'],24.)
  self.assertAlmostEqual(r['forecast_pop_6h_weighted_mean'],.4)
  self.assertAlmostEqual(r['forecast_qpf_category_2_fraction'],1.)
  self.assertAlmostEqual(sum(v for k,v in r.items() if k.startswith('forecast_qpf_category_')),1.)
  self.assertAlmostEqual(r['forecast_qpf_ordinal_weighted_mean'],2.)
 def test_missing_precipitation_not_zero(self):
  d=fixture();d.loc[d.lead_hours.eq(18),'precip_probability_6h']=np.nan
  r,why=daily_features(d,11,35);self.assertIsNone(r);self.assertEqual(why,'incomplete_precipitation_windows')
 def test_missing_temperature_not_interpolated_across(self):
  d=fixture();d.loc[d.lead_hours.eq(18),'temperature_c']=np.nan
  r,why=daily_features(d,11,35);self.assertIsNone(r);self.assertEqual(why,'incomplete_instantaneous_forecast')
 def test_no_extrapolation(self):
  d=fixture();self.assertIsNone(instant_features(d.lead_hours.to_numpy(),d.temperature_c.to_numpy(),5,29,'temp'))
  self.assertIsNone(instant_features(d.lead_hours.to_numpy(),d.temperature_c.to_numpy(),51,75,'temp'))
 def test_fractional_report_clock(self):
  d=fixture();r,why=daily_features(d,12.5,36.5);self.assertIsNone(why);self.assertAlmostEqual(r['forecast_temp_mean_c'],24.5)
 def test_overlap_weighted_category_not_amount(self):
  d=fixture();d.loc[d.lead_hours.eq(12),'qpf_category_6h']=6
  r,why=daily_features(d,11,35);self.assertIsNone(why)
  self.assertAlmostEqual(r['forecast_qpf_category_6_fraction'],1/24)
  self.assertFalse(any('inches' in k or 'precip_mm' in k for k in r))

if __name__=='__main__':unittest.main()
