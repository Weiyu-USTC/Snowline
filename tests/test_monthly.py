"""Independent checks for monthly data preparation and sequential forecasts."""
from __future__ import annotations

import calendar
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import prepare_station_monthly
import monthly


class MonthlyObservationTests(unittest.TestCase):
    station = {'id': 'TEST', 'name': 'Test station'}

    @staticmethod
    def records(year, month, mm=25.4):
        return {date(year, month, day): {'usable': 1, 'snowfall_mm': mm}
                for day in range(1, calendar.monthrange(year, month)[1] + 1)}

    def test_incomplete_month_does_not_erase_other_complete_month(self):
        records = self.records(2023, 10) | self.records(2023, 11)
        records[date(2023, 11, 12)]['usable'] = 0
        long_rows, wide_rows = prepare_station_monthly.monthly_rows(
            records, self.station, 2023, 2023, date(2024, 5, 1))
        self.assertEqual(wide_rows[0]['oct'], 31.)
        self.assertEqual(wide_rows[0]['nov'], '')
        nov = next(row for row in long_rows if row['month_label'] == 'nov')
        self.assertEqual(nov['usable_days'], 29)
        self.assertAlmostEqual(nov['observed_partial_sum_inches'], 29.)
        self.assertEqual(nov['snowfall_inches'], '')
        self.assertEqual(wide_rows[0]['dec'], '')
        self.assertEqual(wide_rows[0]['complete_oct_apr'], 0)

    def test_as_of_cutoff_cannot_use_present_but_future_records(self):
        records = self.records(2026, 10)
        long_rows, wide_rows = prepare_station_monthly.monthly_rows(
            records, self.station, 2026, 2026, date(2026, 10, 20))
        self.assertEqual(wide_rows[0]['oct'], '')
        self.assertEqual(wide_rows[0]['oct_usable_days'], 20)
        self.assertEqual(long_rows[0]['month_ended_by_as_of'], 0)
        self.assertEqual(long_rows[0]['observed_partial_sum_inches'], 20.)

    def test_leap_february_requires_29_days(self):
        records = self.records(2024, 2)
        long_rows, wide_rows = prepare_station_monthly.monthly_rows(
            records, self.station, 2023, 2023, date(2024, 5, 1))
        self.assertEqual(wide_rows[0]['feb'], 29.)
        self.assertEqual(wide_rows[0]['feb_expected_days'], 29)
        del records[date(2024, 2, 29)]
        _, incomplete = prepare_station_monthly.monthly_rows(
            records, self.station, 2023, 2023, date(2024, 5, 1))
        self.assertEqual(incomplete[0]['feb'], '')


class MonthlyForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = monthly.load_training('copper')
        cls.small_candidates = [
            {'family': 'climatology', 'enso': False},
            {'family': 'ridge', 'enso': True, 'alpha': 10.},
        ]

    def test_all_origin_enso_periods_are_available_before_issue(self):
        roni = pd.read_csv(ROOT / 'data/roni_noaa.csv').set_index('year')
        self.assertEqual(len(self.frame), 20)
        self.assertNotIn(2025, self.frame.start_year.tolist())
        for row in self.frame.itertuples():
            for origin, period in enumerate(['JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF']):
                expected_year = row.start_year + (origin == 5)
                self.assertEqual(getattr(row, f'enso_{origin}'), roni.loc[expected_year, period])

    def test_features_use_observed_prefix_only(self):
        row = self.frame.iloc[[3]].copy()
        for origin in range(6):
            a, names = monthly.make_features(row, origin, True)
            changed = row.copy()
            for column in monthly.MONTHS[origin + 1:]:
                changed[column] = 100000.
            for later_origin in range(origin + 1, 6):
                changed[f'enso_{later_origin}'] = 100000.
            b, _ = monthly.make_features(changed, origin, True)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(names[:origin + 1], monthly.MONTHS[:origin + 1])
            self.assertEqual(len(names), origin + 1 + 3)

    def test_unknown_october_is_not_silently_zero_filled(self):
        row = self.frame.iloc[[0]].copy()
        row['oct'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Missing/nonfinite'):
            monthly.make_features(row, 0, True, snow_context=True)
        X, names = monthly.make_features(row, 0, True, snow_context=False)
        self.assertEqual(X.shape, (1, 3))
        self.assertNotIn('oct', names)
        self.assertTrue(np.isfinite(X).all())
        config = {'family': 'ridge', 'enso': True, 'alpha': 100.}
        a = monthly.fit_predict(self.frame, row, 0, config, snow_context=False)
        row['oct'] = 999999.
        b = monthly.fit_predict(self.frame, row, 0, config, snow_context=False)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)

    def test_test_future_month_values_do_not_change_direct_forecasts(self):
        train, test = self.frame.iloc[:-1], self.frame.iloc[[-1]].copy()
        for config in [{'family': 'ridge', 'enso': True, 'alpha': 10.},
                       {'family': 'analog', 'enso': True, 'bandwidth': 1.5, 'uniform_mix': .2}]:
            original = monthly.fit_predict(train, test, 0, config)
            altered = test.copy()
            altered[monthly.MONTHS[1:]] = 100000.
            altered[[f'enso_{i}' for i in range(1, 6)]] = 100000.
            new = monthly.fit_predict(train, altered, 0, config)
            np.testing.assert_allclose(original, new, rtol=0, atol=0)
            self.assertEqual(original.shape, (1, 6))

    def test_inner_tuning_never_trains_on_its_validation_seasons(self):
        inner_input = self.frame.iloc[:-1].reset_index(drop=True)
        outer_held_year = self.frame.start_year.iloc[-1]
        calls = []

        def spy(train, test, origin, config, snow_context=True):
            train_years = set(train.start_year)
            test_years = set(test.start_year)
            self.assertFalse(train_years & test_years)
            self.assertNotIn(outer_held_year, train_years | test_years)
            calls.append((train_years, test_years))
            return np.tile(train[monthly.MONTHS[origin + 1:]].mean().to_numpy(), (len(test), 1))

        with patch.object(monthly, 'candidates', return_value=self.small_candidates), \
             patch.object(monthly, 'fit_predict', side_effect=spy):
            monthly.tune(inner_input, 0)
        self.assertEqual(len(calls), 8)  # Two candidates, four inner folds each.
        for offset in (0, 4):
            all_held = [year for _, held in calls[offset:offset + 4] for year in held]
            self.assertCountEqual(all_held, inner_input.start_year.tolist())

    def test_outer_selection_is_unchanged_when_heldout_answers_change(self):
        holdout = len(self.frame) - 1
        changed = self.frame.copy()
        changed.loc[holdout, monthly.MONTHS[1:]] = 100000.
        changed.loc[holdout, [f'enso_{i}' for i in range(1, 6)]] = 100000.
        with patch.object(monthly, 'candidates', return_value=self.small_candidates):
            before = monthly.outer_fold((self.frame, 0, holdout, True, None))
            after = monthly.outer_fold((changed, 0, holdout, True, None))
        self.assertEqual(before['selection'], after['selection'])
        np.testing.assert_array_equal(
            [row['prediction'] for row in before['predictions']],
            [row['prediction'] for row in after['predictions']])
        self.assertNotIn(int(self.frame.start_year.iloc[holdout]), before['selection']['training_years'])

    def test_jackknife_total_retains_correlated_month_residuals(self):
        rows = [{'prediction': [20., 30.], 'held_prediction': [20., 30.],
                 'held_actual': [20. + i, 30. - i]} for i in range(20)]
        result = monthly.jackknife_bounds(rows, ['nov', 'dec'])
        # Perfectly compensating month errors imply no total residual error.
        self.assertEqual(result['remaining_total_interval'], [50., 50.])
        self.assertLess(result['monthly_lower']['nov'], 20.)
        self.assertGreater(result['monthly_upper']['nov'], 20.)
        self.assertEqual(result['monthly_lower']['nov'], 2.)
        self.assertEqual(result['monthly_upper']['nov'], 38.)
        with self.assertRaises(ValueError):
            monthly.jackknife_bounds(rows[:8], ['nov', 'dec'])

    def test_saved_selections_exclude_their_whole_outer_season(self):
        result = json_load(ROOT / 'outputs/monthly_copper_results.json')
        self.assertEqual(result['n_seasons'], 20)
        self.assertEqual(len(result['selections']), 120)
        for selection in result['selections']:
            held_year = int(selection['held_out_season'].split('/')[0])
            self.assertEqual(len(selection['training_years']), 19)
            self.assertNotIn(held_year, selection['training_years'])
        predictions = pd.read_csv(ROOT / 'outputs/monthly_copper_predictions.csv')
        self.assertEqual(len(predictions.query("family == 'selected' and horizon == 1")), 120)
        self.assertEqual(len(predictions.query("family == 'selected' and origin == 0")), 120)
        metric = next(item for item in result['metrics']
                      if item['scope'] == 'sequential_next_month' and item['family'] == 'selected')
        rows = predictions.query("family == 'selected' and horizon == 1")
        self.assertAlmostEqual(metric['rmse_inches'], np.sqrt(np.mean((rows.prediction - rows.actual) ** 2)))


def json_load(path):
    import json
    return json.loads(path.read_text())


if __name__ == '__main__':
    unittest.main()
