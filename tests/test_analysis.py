"""Independent regression, leakage, data-quality and exported-model checks.

Run: python -m unittest discover -s tests -v
The saved-artifact checks expect src/analysis.py to have been run first.
"""
from __future__ import annotations

import calendar
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import analysis
import fetch_ghcn


class SeasonalAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = analysis.load_copper()
        cls.copper = cls.raw.loc[cls.raw.primary_included].copy()

    def test_common_target_and_unresolved_rows(self):
        self.assertEqual(len(self.raw), 21)
        self.assertEqual(len(self.copper), 20)
        self.assertEqual(self.copper.start_year.max(), 2024)
        np.testing.assert_array_equal(
            self.raw.nov_apr,
            self.raw[['nov', 'dec', 'jan', 'feb', 'mar', 'apr']].sum(axis=1),
        )
        np.testing.assert_array_equal(self.raw.oct_apr - self.raw.oct, self.raw.nov_apr)
        latest = self.raw.set_index('start_year').loc[2025]
        self.assertEqual(latest.nov_apr, 180)
        self.assertEqual(latest.observed_month_sum, 203)
        self.assertEqual(latest.reported_total, 263)
        self.assertEqual(latest.reported_minus_observed, 60)
        self.assertFalse(latest.primary_included)
        small_discrepancy = self.raw.set_index('start_year').loc[2019]
        self.assertEqual(small_discrepancy.reported_minus_observed, 1)
        self.assertTrue(small_discrepancy.primary_included)
        self.assertTrue(self.raw.loc[self.raw.start_year < 2022, 'may'].isna().all())
        self.assertEqual(self.raw.set_index('start_year').loc[2022, 'may'], 0)

    def test_enso_calendar_alignment_and_predictor_timing(self):
        oni = pd.read_csv(analysis.DATA / 'oni_noaa.csv').set_index('year')
        for row in self.copper.itertuples():
            self.assertEqual(row.oni_jas, oni.loc[row.start_year, 'JAS'])
            self.assertEqual(row.oni_djf, oni.loc[row.start_year + 1, 'DJF'])
        self.assertEqual(self.copper.set_index('start_year').loc[2015, 'oni_djf'], 2.5)
        for features in analysis.SPECS.values():
            self.assertTrue(set(features).issubset({'oct', 'oni_jas'}))

    def test_later_season_cannot_change_earlier_predictions(self):
        original, _, _ = analysis.backtest(self.copper, min_train=8)
        mutated = self.copper.copy()
        latest = mutated.start_year.idxmax()
        for column in ['oct', 'nov_apr', 'oni_jas', 'oni_djf', 'roni_jas', 'roni_djf']:
            mutated.loc[latest, column] += 10000
        changed, _, _ = analysis.backtest(mutated, min_train=8)
        keep = original.start_year < self.copper.start_year.max()
        np.testing.assert_allclose(original.loc[keep, 'prediction'], changed.loc[keep, 'prediction'], atol=0, rtol=0)
        self.assertTrue((original.train_last_year < original.start_year).all())

    def test_retrospective_fields_cannot_change_forecasts(self):
        original, _, _ = analysis.backtest(self.copper, min_train=8)
        mutated = self.copper.copy()
        for column in ['oni_djf', 'roni_djf', 'reported_total', 'user_el_nino']:
            mutated[column] = np.arange(len(mutated)) * 10000 + 999
        changed, _, _ = analysis.backtest(mutated, min_train=8)
        np.testing.assert_allclose(original.prediction, changed.prediction, atol=0, rtol=0)

    def test_independent_known_statistics_and_forecast_metrics(self):
        result = stats.pearsonr(self.copper.oct, self.copper.nov_apr)
        self.assertAlmostEqual(result.statistic, 0.24662387575591765, places=12)
        self.assertAlmostEqual(result.pvalue, 0.2945371723960756, places=12)
        _, metrics, _ = analysis.backtest(self.copper, min_train=8)
        metrics = {row['model']: row for row in metrics}
        expected = {
            'climatology': (36.7820821052729, 41.06392858230798),
            'october_linear': (36.57698891343609, 46.698920491733524),
            'enso_linear': (46.23505613203366, 53.52634300954841),
            'october_enso_ridge': (40.84235535940609, 47.125505858459775),
        }
        for name, (mae, rmse) in expected.items():
            self.assertEqual(metrics[name]['n_test'], 12)
            self.assertAlmostEqual(metrics[name]['mae_inches'], mae, places=9)
            self.assertAlmostEqual(metrics[name]['rmse_inches'], rmse, places=9)
        contrast = analysis.strong_contrast(self.copper)
        self.assertEqual(contrast['strong_seasons'], ['2009/2010', '2015/2016', '2023/2024'])
        self.assertAlmostEqual(contrast['difference_inches'], -16.019607843137265, places=10)
        self.assertEqual(analysis.strong_contrast(self.copper, drop_boundary=True)['n_strong'], 2)

    def test_invalid_copper_inputs_are_rejected_before_aggregation(self):
        source = pd.read_csv(analysis.DATA / 'copper_monthly_user.csv')
        cases = []
        missing = source.copy()
        missing.loc[0, 'nov'] = np.nan
        cases.append(missing)
        negative = source.copy()
        negative.loc[0, 'dec'] = -1
        cases.append(negative)
        cases.append(pd.concat([source, source.iloc[[0]]], ignore_index=True))
        with tempfile.TemporaryDirectory() as directory:
            for frame in cases:
                frame.to_csv(Path(directory) / 'copper_monthly_user.csv', index=False)
                with patch.object(analysis, 'DATA', Path(directory)):
                    with self.assertRaises(ValueError):
                        analysis.load_copper()

    def test_stations_are_independent_complete_cohorts(self):
        expected = {'USC00052281': 41, 'USC00051660': 22}
        for station, count in expected.items():
            loaded = analysis.load_station(station)
            self.assertEqual(len(loaded), count)
            self.assertEqual(set(loaded.station_id), {station})
            self.assertTrue((loaded.complete_oct_nov_apr == 1).all())
            self.assertTrue((loaded.october_usable_days == loaded.october_expected_days).all())
            self.assertTrue((loaded.nov_apr_usable_days == loaded.nov_apr_expected_days).all())

    def test_quality_failure_on_leap_day_invalidates_whole_target(self):
        station = 'USC00051660'
        lines = []
        for year, month in [(2019, 10), (2019, 11), (2019, 12), (2020, 1), (2020, 2), (2020, 3), (2020, 4)]:
            lines.append(station + f'{year}{month:02d}SNOW' + f'{254:5d}  0' * 31)
        parsed = fetch_ghcn.parse_snow('\n'.join(lines), station)
        complete = fetch_ghcn.aggregate(parsed, {'id': station, 'name': 'fixture'}, 2019, 2019)[0]
        self.assertEqual(complete['nov_apr_expected_days'], 182)
        self.assertEqual(complete['nov_apr_snowfall_inches'], 1820)
        self.assertEqual(complete['october_snowfall_inches'], 310)
        for unusable_block in [f'{51:5d} X0', f'{-9999:5d}   ', f'{0:5d}P 0']:
            changed_lines = list(lines)
            february = changed_lines[4]
            offset = 21 + 8 * 28
            changed_lines[4] = february[:offset] + unusable_block + february[offset + 8:]
            changed = fetch_ghcn.parse_snow('\n'.join(changed_lines), station)
            result = fetch_ghcn.aggregate(changed, {'id': station, 'name': 'fixture'}, 2019, 2019)[0]
            self.assertEqual(result['nov_apr_usable_days'], 181)
            self.assertEqual(result['nov_apr_snowfall_inches'], '')
            self.assertEqual(result['complete_oct_nov_apr'], 0)
        # Absence of an entire month must also remain missing, not become a dry month.
        absent = fetch_ghcn.parse_snow('\n'.join(lines[1:]), station)
        result = fetch_ghcn.aggregate(absent, {'id': station, 'name': 'fixture'}, 2019, 2019)[0]
        self.assertEqual(result['october_usable_days'], 0)
        self.assertEqual(result['october_snowfall_inches'], '')
        self.assertEqual(result['complete_oct_nov_apr'], 0)

    def test_saved_models_reproduce_independent_estimators(self):
        bundle = json.loads((ROOT / 'outputs/trained_models.json').read_text())
        self.assertEqual(bundle['default_model'], 'climatology')
        for dataset in ['copper', 'dillon', 'climax']:
            frame = pd.read_csv(ROOT / f'outputs/{dataset}_analysis_data.csv')
            for name, exported in bundle['datasets'][dataset]['models'].items():
                actual = analysis.predict(exported, frame)
                columns = exported['features']
                if not columns:
                    reference = np.repeat(frame.nov_apr.mean(), len(frame))
                else:
                    estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0)) if name.endswith('ridge') else LinearRegression()
                    estimator.fit(frame[columns], frame.nov_apr)
                    reference = estimator.predict(frame[columns])
                np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-10)

    def test_saved_fold_predictions_reproduce_training_only_fits(self):
        saved = pd.read_csv(ROOT / 'outputs/copper_backtest_predictions.csv')
        for row in saved.itertuples():
            train = self.copper.loc[self.copper.start_year < row.start_year]
            test = self.copper.loc[self.copper.start_year == row.start_year]
            forecast = analysis.predict(analysis.fit_model(train, row.model), test)[0]
            self.assertAlmostEqual(row.prediction, forecast, places=10)
            self.assertEqual(row.n_train, len(train))

    def test_cli_uses_exported_values_and_rejects_missing_inputs(self):
        bundle = json.loads((ROOT / 'outputs/trained_models.json').read_text())
        command = [sys.executable, str(ROOT / 'src/predict.py')]
        result = subprocess.run(command + ['--model', 'october_enso_ridge', '--october', '25', '--jas-oni', '0.5'], text=True, capture_output=True, check=True)
        output = json.loads(result.stdout)
        exported = bundle['datasets']['copper']['models']['october_enso_ridge']
        expected = analysis.predict(exported, pd.DataFrame({'oct': [25.0], 'oni_jas': [0.5]}))[0]
        self.assertEqual(output['nov_apr_inches'], round(expected, 1))
        self.assertNotIn('prediction_interval80_inches', output)
        for arguments in [['--model', 'enso_linear'], ['--october', '-1'], ['--jas-oni', 'nan']]:
            invalid = subprocess.run(command + arguments, text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == '__main__':
    unittest.main()
