"""Time alignment, stale display and missing-value safety in the Alta adapter."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import dashboard_alta as alta

class AltaDashboardTests(unittest.TestCase):
    def test_fractional_hrrr_overlap(self):
        run=datetime(2026,1,1,tzinfo=timezone.utc)
        result=alta.integrate_hrrr([1,2,3,4],[1,2,3,4],run,run+timedelta(minutes=30),[0,1,2,3])
        self.assertEqual(result,[0,1.5,4.,7.5])
    def test_missing_hrrr_is_not_zero(self):
        run=datetime(2026,1,1,tzinfo=timezone.utc)
        with self.assertRaises(ValueError):alta.integrate_hrrr([1,2],[1,np.nan],run,run,[2])
    def test_uncovered_horizon_rejected(self):
        run=datetime(2026,1,1,tzinfo=timezone.utc)
        with self.assertRaises(ValueError):alta.integrate_hrrr([1],[1],run,run,[2])
    def test_age_preserves_issue(self):
        issue=datetime(2026,1,1,tzinfo=timezone.utc)
        payload={'status':'fresh','issued_at':issue.isoformat(),'end':(issue+timedelta(hours=3)).isoformat()}
        with patch.object(alta,'utc_now',return_value=issue+timedelta(hours=4)):
            result=alta._age(payload)
        self.assertTrue(result['stale']);self.assertTrue(result['expired'])
        self.assertEqual(result['issued_at'],payload['issued_at'])
        self.assertEqual(payload['status'],'fresh')

if __name__=='__main__':unittest.main()
