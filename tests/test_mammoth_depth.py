from pathlib import Path
import json
import math
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import mammoth_depth as depth


class MammothDepthTests(unittest.TestCase):
    def test_month_requires_all_local_hours_after_issue_including_dst(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lodge = root/'data/dashboard/seasonal/raw/mammoth'
            lodge.mkdir(parents=True)
            (lodge/'main_lodge.json').write_text(json.dumps({'seasons': {'2020': {'months': [20.]*10}}}))
            wx = root/'data/mammoth/weather'
            wx.mkdir(parents=True)
            local = ZoneInfo('America/Los_Angeles')
            # March has 743 local hours. February has a missing report so the entire month is excluded.
            times = pd.date_range('2021-02-01','2021-04-01',tz=local, inclusive='left',freq='h').tz_convert('UTC')
            times = times.delete(5)
            rows = []
            for a in times:
                for delta,amount in ((6,1.),(2,99.)):
                    run = a - pd.Timedelta(hours=delta)
                    rows.append(dict(start_utc=a, end_utc=a+pd.Timedelta(hours=1), run_utc=run,
                                     available_utc=run+pd.Timedelta(hours=3), lead=delta+1,
                                     temperature_c=-9., snow_water_mm=amount))
            pd.DataFrame(rows).to_csv(wx/'history.csv.gz',index=False)
            months = depth._months(root)
            self.assertEqual([(r['month'], r['hours']) for r in months], [('2021-03',743)])
            self.assertEqual(months[0]['cold_swe_mm'], 743.)
            self.assertEqual(months[0]['warm_swe_mm'], 0.)

    def test_fitted_units_and_native_tail(self):
        model = dict(ratios_cold_warm=[12.,6.],threshold_c=-5.)
        weather = [dict(snow_water_mm=2.54,temperature_c=-6.)] * 36
        weather += [dict(snow_water_mm=None,temperature_c=None)] * 12
        amounts = depth.predict_depth(model, weather)
        self.assertEqual(len(amounts), 48)
        self.assertAlmostEqual(amounts[0], 1.2)
        self.assertAlmostEqual(amounts[36], 1.2*math.exp(-1./6))
        self.assertAlmostEqual(amounts[-1], 1.2*math.exp(-2))
        with self.assertRaises(ValueError):
            depth.predict_depth(model, weather[:33]+weather[36:])


if __name__ == '__main__':
    unittest.main()
