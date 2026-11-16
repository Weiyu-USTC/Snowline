"""Saved model windows remain explicit and cannot become invented hourly snow."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dashboard_neural48 import read_neural48


class NeuralDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'snapshot.json'
        self.site = {'site_id':'alta', 'status':'experimental', 'model_name':'Weather neural', 'issued_at':'2026-11-11T10:00:00+00:00',
            'start':'2026-11-11T12:00:00+00:00', 'midpoint':'2026-11-12T12:00:00+00:00',
            'end':'2026-11-13T12:00:00+00:00',
            'totals':[{'hours':24,'end':'2026-11-12T12:00:00+00:00','mean_inches':1.,'q10_inches':0.,'q90_inches':3.,'provider_inches':2.},
                      {'hours':48,'end':'2026-11-13T12:00:00+00:00','mean_inches':2.,'q10_inches':0.,'q90_inches':5.,'provider_inches':3.}]}
    def tearDown(self): self.temp.cleanup()
    def read(self, hour=11):
        self.path.write_text(json.dumps({'generated_at':'2026-11-11T10:00:00+00:00','sites':{'alta':self.site}}))
        return read_neural48('alta', now=datetime(2026,11,11,hour,tzinfo=timezone.utc),path=self.path)
    def test_fully_future_totals_preserve_exact_windows_and_units(self):
        p=self.read()
        self.assertTrue(p['fully_future']);self.assertFalse(p['window_started'])
        self.assertFalse(p['hourly_validated']);self.assertFalse(p['stale'])
        self.assertEqual(p['totals'][1]['mean_inches'],2.)
    def test_begun_window_is_not_shifted_to_now(self):
        p=self.read(14)
        self.assertTrue(p['window_started']);self.assertFalse(p['fully_future'])
        self.assertEqual(p['start'],self.site['start'])
    def test_missing_data_is_unavailable_not_zero(self):
        self.assertEqual(read_neural48('alta',path=self.path)['status'],'unavailable')
        self.site['totals'][0]['mean_inches']=None
        self.assertEqual(self.read()['status'],'unavailable')
    def test_mismatched_site_clock_and_invalid_values_rejected(self):
        original=json.loads(json.dumps(self.site))
        for key,value in [('site_id','jay'),('midpoint','2026-11-12T13:00:00+00:00')]:
            self.site=json.loads(json.dumps(original));self.site[key]=value
            self.assertEqual(self.read()['status'],'unavailable')
        self.site=original;self.site['totals'][0]['provider_inches']=-1
        self.assertEqual(self.read()['status'],'unavailable')
    def test_missing_provider_is_not_fabricated(self):
        self.site['totals'][0]['provider_inches']=None
        self.assertEqual(self.read()['status'],'experimental')
        self.assertIsNone(self.read()['totals'][0]['provider_inches'])
    def test_regenerating_old_inputs_does_not_make_them_fresh(self):
        self.site['issued_at']='2026-11-11T07:00:00+00:00'
        p=self.read(11)
        self.assertTrue(p['stale'])
        self.assertEqual(p['issued_at'],'2026-11-11T07:00:00+00:00')
    def test_issue_after_start_is_rejected(self):
        self.site['issued_at']='2026-11-11T13:00:00+00:00'
        self.assertEqual(self.read()['status'],'unavailable')
    def test_stale_snapshot_retains_generation_date(self):
        p=self.read(20)
        self.assertTrue(p['stale']);self.assertEqual(p['generated_at'],'2026-11-11T10:00:00+00:00')

if __name__=='__main__': unittest.main()
