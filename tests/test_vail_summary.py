"""Checks that the autumn sensitivity compares the same held-out targets."""
import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Vail metrics import requires optional torch dependency')
from summarize_vail_evaluation import comparison_rows, november_december_targets


class VailCalendarSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame([
            dict(model=model, winter=2023, origin=origin, date=date, lead=lead,
                 actual_inches=2., predicted_inches=1.)
            for model in ['Hourly TCN', 'Climatology']
            for origin, date, lead in [('2023-10-15 04:00', '2023-11-01', 17),
                                       ('2023-11-30 04:00', '2024-01-01', 32)]])

    def test_filter_uses_target_month_not_issue_month(self):
        result = november_december_targets(self.rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result.date), {'2023-11-01'})
        self.assertEqual(set(result.origin), {'2023-10-15 04:00'})
        self.assertEqual(len(self.rows), 4)

    def test_missing_comparator_pair_cannot_silently_bias_comparison(self):
        with self.assertRaisesRegex(ValueError, 'Unequal valid target pairs'):
            comparison_rows(self.rows.iloc[:-1])

    def test_disagreement_on_observation_is_rejected(self):
        altered = self.rows.copy()
        altered.loc[0, 'actual_inches'] = 200.
        with self.assertRaisesRegex(ValueError, 'Inconsistent target'):
            comparison_rows(altered)


if __name__ == '__main__':
    unittest.main()
