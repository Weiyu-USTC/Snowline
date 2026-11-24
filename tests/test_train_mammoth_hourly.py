import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from train_mammoth_hourly import forecast_interval, predict_hourly


class MammothTrainingContract(unittest.TestCase):
    def setUp(self):
        self.issue=datetime(2026,11,24,20,tzinfo=timezone.utc)
        self.weather=[{'precip_mm':1.,'snow_water_mm':.5,'temperature_c':-3.} for _ in range(42)]
        self.weather += [{'precip_mm':None,'snow_water_mm':None,'temperature_c':None} for _ in range(6)]
        self.model={'version':1,'estimator':{'kind':'radar_update','scale':2.,'gains':[.2,.1,.1]},
                    'monthly_means_mm':[.1]*12, 'calibration':{'3':{'lower_offset_mm':-2.,'upper_offset_mm':4.}}}

    def test_missing_tail_uses_only_issued_native_values(self):
        hourly=predict_hourly(self.model,self.issue,self.weather)
        self.assertEqual(len(hourly),48)
        np.testing.assert_allclose(hourly[:42],1.)
        np.testing.assert_allclose(hourly[42:],np.exp(-np.arange(1,7)/6.))
        self.weather[0]['snow_water_mm']=None
        with self.assertRaises(ValueError):
            predict_hourly(self.model,self.issue,self.weather)

    def test_future_and_old_radar_cannot_change_hourly_outlook(self):
        def run(minutes):
            radar=dict(timestamp=(self.issue+timedelta(minutes=minutes)).isoformat(),
                       point=2.,local=1.,regional=3.,west=1.,quality=.4)
            return predict_hourly(self.model,self.issue,self.weather,radar)
        base=predict_hourly(self.model,self.issue,self.weather)
        self.assertEqual(base,run(1))
        self.assertEqual(base,run(-91))
        valid=run(-34)
        self.assertGreater(valid[0],base[0])
        self.assertLess(valid[6]-base[6], valid[0]-base[0])
        self.assertEqual(forecast_interval(self.model,3,1.),(0.,5.))
        self.assertEqual(forecast_interval(self.model,8,1.),(None,None))


if __name__=='__main__':
    unittest.main()
