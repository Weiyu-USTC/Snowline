from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import forecast_neural48 as live


class LiveNeuralForecastContract(unittest.TestCase):
    def test_refresh_cache_expires_when_forecast_window_begins(self):
        p={'generated_at':'2026-11-11T20:55Z','sites':{'alta':{'status':'experimental','start':'2026-11-11T21:00Z'}}}
        self.assertTrue(live.reusable_cache(p,pd.Timestamp('2026-11-11T20:59Z')))
        self.assertFalse(live.reusable_cache(p,pd.Timestamp('2026-11-11T21:10Z')))
        p['sites']['alta']['status']='unavailable'
        self.assertFalse(live.reusable_cache(p,pd.Timestamp('2026-11-11T20:59Z')))

    def test_history_never_uses_future_or_unavailable_report(self):
        issue=pd.Timestamp('2026-11-11T20:10Z')
        d=pd.DataFrame({'site':['alta']*3,'valid_snowfall':[True]*3,
            'period_end_utc':['2024-05-01T11:00Z','2026-11-10T21:00Z','2026-11-12T11:00Z'],
            'snowfall_inches':[2.,77.,99.]})
        with patch.object(live.pd,'read_csv',return_value=d):
            x,meta=live.history_inputs('alta',issue)
        self.assertTrue(np.isnan(x[-2]));self.assertEqual(x[-1],60.)
        self.assertFalse(meta['recent_history_available'])
        np.testing.assert_array_equal(x[[2,5,8,11,14]],0)

    def test_cumulative_quantiles_use_joint_paths_and_actual_family(self):
        # The two daily outcomes offset perfectly: total is always10, despite
        # both daily distributions spanning0to10. Adding daily quantiles fails.
        paths=np.tile(np.array([[0.,10.],[10.,0.]]),(384,1))[None]
        prediction={'daily_mean':np.array([[5.,5.]]),'joint_paths':paths,'methods':['Weather TCN']}
        bundle={'selection':{'frozen_utc':'2026-11-11T19:00Z','selected_family_by_site':{'alta':'weather_radar'}}}
        row={'site_id':'alta','status':'experimental','array_row':0,'radar_available':False,
            'start':'2026-11-11T21:00Z','midpoint':'2026-11-12T21:00Z','end':'2026-11-13T21:00Z',
            'provider_daily_inches':[0.,0.],'conservative_provider_daily_inches':[0.,0.]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(live,'OUT',Path(tmp)),patch('train_neural48.load_bundle',return_value=bundle),\
                 patch('train_neural48.predict_neural',return_value=prediction),patch.object(live,'plot_current'):
                payload=live.issue_from_prepared({'site':np.array([0])},[row],live.ROOT/'data/example')
        r=payload['sites']['alta']
        self.assertEqual(r['family'],'weather_only')
        self.assertEqual(r['model_name'],'Weather TCN')
        self.assertEqual(r['totals'][1]['q10_inches'],10.)
        self.assertEqual(r['totals'][1]['q90_inches'],10.)
        self.assertEqual(r['totals'][1]['mean_inches'],10.)

    def test_cached_forecast_is_aged_without_changing_issue_or_valid_window(self):
        payload={'sites':{'alta':{'issued_at':'2026-11-10T10:50Z','start':'2026-11-10T11:00Z','end':'2026-11-12T11:00Z'}}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'current_forecast.json';path.write_text(json.dumps(payload))
            with patch.object(live,'OUT',Path(tmp)):
                value=live.get_cached_forecasts()
        self.assertEqual(value['sites']['alta']['issued_at'],payload['sites']['alta']['issued_at'])
        self.assertEqual(value['sites']['alta']['start'],payload['sites']['alta']['start'])
        self.assertTrue(value['sites']['alta']['stale'])


if __name__=='__main__':unittest.main()
