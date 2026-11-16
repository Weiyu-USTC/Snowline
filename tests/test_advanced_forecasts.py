"""Check the saved forecast curves against actual checkpoints and coherent paths."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Optional neural forecast checks need requirements-deep.txt')
import torch
import deep_dillon as base
import issue_advanced_snow_forecast as forecast


@unittest.skipUnless((ROOT / 'outputs/advanced_forecast_manifest.json').exists(), 'Forecast comparison not generated yet')
class AdvancedForecastChecks(unittest.TestCase):
    PREFIX = 'advanced_forecast'

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.manifest = json.loads((ROOT / f'outputs/{cls.PREFIX}_manifest.json').read_text())
        cls.daily = pd.read_csv(ROOT / f'outputs/{cls.PREFIX}_daily.csv', parse_dates=['date'])
        cls.monthly = pd.read_csv(ROOT / f'outputs/{cls.PREFIX}_monthly.csv')
        cls.frame = pd.read_csv(ROOT / cls.manifest['input_path'], parse_dates=['date']).set_index('date')
        with np.load(ROOT / f'outputs/{cls.PREFIX}_paths.npz') as data:
            cls.models = data['models'].tolist()
            cls.dates = pd.DatetimeIndex(data['dates'])
            cls.paths = data['paths_inches']

    def test_all_checkpoint_predictions_reproduce_saved_daily_curves(self):
        self.assertEqual(forecast.sha256(ROOT / self.manifest['input_path']), self.manifest['input_sha256'])
        self.assertEqual(forecast.sha256(base.RONI_PATH), self.manifest['roni_sha256'])
        origin = pd.Timestamp(self.manifest['forecast_origin'])
        for family in self.models:
            metadata = self.manifest['models'][family]
            checkpoint = ROOT / metadata['checkpoint']
            self.assertEqual(forecast.sha256(checkpoint), metadata['checkpoint_sha256'])
            model, scaler, _ = forecast.load_model_checkpoint(checkpoint)
            params = base.predict(model, base.arrays(self.frame, [(2026, origin)], scaler))[0][:len(self.dates)]
            saved = self.daily.loc[self.daily.model.eq(family)]
            np.testing.assert_allclose(saved.predicted_inches, base.distribution_mean(params), rtol=0, atol=1e-8)
            for column, pos in [('p_snow', 0), ('positive_log_mu', 1), ('positive_log_sigma', 2)]:
                np.testing.assert_allclose(saved[column], params[:, pos], rtol=0, atol=1e-8)

    def test_shared_dates_preserve_unknown_bridge_and_current_day(self):
        self.assertEqual(self.frame.index.max(), pd.Timestamp('2026-11-01'))
        np.testing.assert_array_equal(self.dates, pd.date_range('2026-11-02', '2026-12-31'))
        self.assertEqual(self.paths.shape, (3, 4096, 60))
        for family in self.models:
            saved = self.daily.loc[self.daily.model.eq(family)]
            np.testing.assert_array_equal(saved.date, self.dates)
            bridge = saved.loc[saved.forecast_status.eq('unobserved_bridge')]
            self.assertEqual(bridge.date.dt.strftime('%Y-%m-%d').tolist(), ['2026-11-02', '2026-11-03'])
            self.assertEqual(saved.loc[saved.date.eq('2026-11-04'), 'forecast_status'].iloc[0], 'current_day_forecast')

    def test_monthly_means_sum_daily_means_and_intervals_sum_paths(self):
        for row in self.monthly.itertuples():
            model_index = self.models.index(row.model)
            in_month = self.dates.to_period('M').astype(str) == row.month
            daily = self.daily.loc[self.daily.model.eq(row.model)]
            expected = daily.loc[in_month, 'predicted_inches'].sum() + row.observed_prefix_inches
            self.assertAlmostEqual(row.predicted_full_month_inches, expected, delta=1e-5)
            totals = self.paths[model_index][:, in_month].sum(axis=1) + row.observed_prefix_inches
            np.testing.assert_allclose([row.q10_full_month_inches, row.q90_full_month_inches],
                                       np.quantile(totals, [.1, .9]), atol=1e-8, rtol=0)

    def test_display_cumulative_curves_exclude_the_unobserved_bridge(self):
        shown = self.dates >= pd.Timestamp('2026-11-04')
        for i, family in enumerate(self.models):
            daily = self.daily.loc[self.daily.model.eq(family)].reset_index(drop=True)
            self.assertTrue(daily.loc[~shown, 'display_cumulative_mean_inches'].isna().all())
            np.testing.assert_allclose(daily.loc[shown, 'display_cumulative_mean_inches'],
                                       daily.loc[shown, 'predicted_inches'].cumsum(), atol=1e-5, rtol=0)
            intervals = np.quantile(self.paths[i][:, shown].cumsum(axis=1), [.1, .9], axis=0)
            np.testing.assert_allclose(daily.loc[shown, ['display_cumulative_q10_inches', 'display_cumulative_q90_inches']].to_numpy().T,
                                       intervals, atol=1e-8, rtol=0)


@unittest.skipUnless((ROOT / 'outputs/shuffled_forecast_manifest.json').exists(), 'Shuffled-CV forecast comparison not generated yet')
class ShuffledForecastChecks(AdvancedForecastChecks):
    PREFIX = 'shuffled_forecast'


if __name__ == '__main__':
    unittest.main()
