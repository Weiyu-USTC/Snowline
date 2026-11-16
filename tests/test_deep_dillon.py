"""Independent checks of the daily model's information boundary and uncertainty."""
import json
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Optional daily neural tests require requirements-deep.txt')
import deep_dillon as model


class DeepDailyInformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / 'outputs/dillon_deep_daily_input.csv', parse_dates=['date']).set_index('date')
        cls.training = model.make_origins(cls.frame, model.TRAIN_YEARS, stride=14)
        cls.scaler = model.Scaler().fit(cls.frame, cls.training)

    def test_windows_from_different_splits_share_no_context_or_target_dates(self):
        date_sets = []
        for years in [model.TRAIN_YEARS, model.VALID_YEARS, model.TEST_YEARS]:
            origins = model.make_origins(self.frame, years, stride=7)
            dates = set()
            for year, origin in origins:
                self.assertIn(year, years)
                first = origin - pd.Timedelta(days=model.CONTEXT)
                last = origin + pd.Timedelta(days=model.HORIZON - 1)
                self.assertGreaterEqual(first, pd.Timestamp(f'{year}-08-01'))
                self.assertLessEqual(last, pd.Timestamp(f'{year + 1}-04-30'))
                dates.update(pd.date_range(first, last))
            date_sets.append(dates)
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            self.assertFalse(date_sets[a] & date_sets[b])

    def test_heldout_weather_cannot_change_training_scaler(self):
        changed = self.frame.copy()
        changed.loc['2017-08-01':, model.FEATURES] = 100000.
        other = model.Scaler().fit(changed, self.training)
        np.testing.assert_array_equal(self.scaler.mean, other.mean)
        np.testing.assert_array_equal(self.scaler.std, other.std)
        self.assertLess(pd.Timestamp(other.fit_max), pd.Timestamp('2017-08-01'))

    def test_future_weather_changes_labels_but_not_fixed_origin_model_inputs(self):
        origin = pd.Timestamp('2023-11-01')
        changed = self.frame.copy()
        changed.loc[origin:, model.FEATURES] = 12345.
        original = model.arrays(self.frame, [(2023, origin)], self.scaler)
        altered = model.arrays(changed, [(2023, origin)], self.scaler)
        np.testing.assert_array_equal(original[0].numpy(), altered[0].numpy())
        np.testing.assert_array_equal(original[1].numpy(), altered[1].numpy())
        self.assertFalse(np.array_equal(original[2].numpy(), altered[2].numpy()))
        model.seed_everything(1701)
        net = model.SnowGRU(hidden=32, dropout=0.)
        np.testing.assert_array_equal(model.predict(net, original), model.predict(net, altered))

    def test_test_observations_cannot_change_fit_only_dependence_paths(self):
        params = np.column_stack([np.linspace(.15, .45, model.HORIZON),
                                  np.zeros(model.HORIZON), np.full(model.HORIZON, .8)])
        changed = self.frame.copy()
        changed.loc['2021-08-01':, 'snowfall_inches'] = 9999.
        args = (model.TRAIN_YEARS + model.VALID_YEARS, pd.Timestamp('2023-11-01'), params, 123)
        paths, count = model.rank_template_paths(self.frame, *args, n_paths=256)
        altered, other_count = model.rank_template_paths(changed, *args, n_paths=256)
        self.assertEqual(count, other_count)
        np.testing.assert_array_equal(paths, altered)

    def test_future_enso_observations_do_not_enter_initial_forecast(self):
        origin = pd.Timestamp('2023-11-01')
        self.assertEqual(model.lagged_roni(origin), model.RONI.loc[2023, 'JAS'])
        self.assertEqual(model.lagged_roni(pd.Timestamp('2023-12-01')), model.RONI.loc[2023, 'ASO'])
        self.assertEqual(model.lagged_roni(pd.Timestamp('2024-01-01')), model.RONI.loc[2023, 'SON'])
        original = model.arrays(self.frame, [(2023, origin)], self.scaler)
        saved_roni = model.RONI
        try:
            model.RONI = saved_roni.copy()
            model.RONI.loc[2023, ['ASO', 'SON', 'OND', 'NDJ']] = 9999.
            model.RONI.loc[model.RONI.index >= 2024] = 9999.
            altered = model.arrays(self.frame, [(2023, origin)], self.scaler)
            np.testing.assert_array_equal(original[1].numpy(), altered[1].numpy())
        finally:
            model.RONI = saved_roni

    def test_reordering_preserves_each_leads_predictive_marginal(self):
        params = np.column_stack([np.linspace(.15, .45, model.HORIZON),
                                  np.linspace(-.2, .3, model.HORIZON), np.full(model.HORIZON, .8)])
        paths, _ = model.rank_template_paths(self.frame, model.TRAIN_YEARS,
                                              pd.Timestamp('2023-11-01'), params, 98, n_paths=512)
        quantiles = (np.arange(512) + .5) / 512
        expected = model.distribution_quantile(params[None, :, :], quantiles[:, None])
        np.testing.assert_allclose(np.sort(paths, axis=0), expected, rtol=0, atol=0)
        self.assertTrue(np.isfinite(paths).all())
        self.assertTrue((paths >= 0).all())

    def test_lognormal_expectation_includes_variance_term_and_zero_mass(self):
        params = np.array([[.25, np.log(2.), .8]])
        self.assertAlmostEqual(model.distribution_mean(params)[0], .25 * 2 * np.exp(.5 * .8 ** 2))
        self.assertEqual(model.distribution_quantile(params, .5)[0], 0.)
        self.assertAlmostEqual(model.distribution_quantile(params, .875)[0], 2.)


@unittest.skipUnless((ROOT / 'outputs/deep_dillon_results.json').exists(), 'Saved forecast not produced yet')
class DeepDailySavedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / 'outputs/deep_dillon_results.json').read_text())
        cls.daily = pd.read_csv(ROOT / 'outputs/deep_dillon_forecast_daily.csv', parse_dates=['date'])
        cls.monthly = pd.read_csv(ROOT / 'outputs/deep_dillon_forecast_monthly.csv')
        with np.load(ROOT / 'outputs/deep_dillon_forecast_paths.npz') as saved:
            cls.paths = saved['paths_inches']
            cls.path_dates = pd.to_datetime(saved['dates'])

    def test_current_forecast_respects_observation_cutoff_and_month_boundaries(self):
        cutoff = pd.Timestamp(self.result['observation_cutoff'])
        self.assertEqual(self.daily.date.min(), cutoff + pd.Timedelta(days=1))
        self.assertEqual(self.daily.date.max(), pd.Timestamp('2026-12-31'))
        self.assertEqual(len(self.daily), (pd.Timestamp('2026-12-31') - cutoff).days)
        np.testing.assert_array_equal(self.daily.date.to_numpy(), self.path_dates.to_numpy())
        np.testing.assert_allclose(self.daily.predicted_inches.cumsum(), self.daily.cumulative_predicted_inches,
                                   rtol=0, atol=1e-5)

    def test_monthly_predictions_are_daily_mean_sums_and_path_sum_quantiles(self):
        for row in self.monthly.itertuples():
            mask = self.daily.date.dt.to_period('M').astype(str).eq(row.month).to_numpy()
            expected_mean = self.daily.loc[mask, 'predicted_inches'].sum()
            self.assertAlmostEqual(row.predicted_remaining_inches, expected_mean, delta=1e-5)
            self.assertAlmostEqual(row.predicted_full_month_inches, expected_mean + row.observed_prefix_inches, delta=1e-5)
            quantiles = np.quantile(self.paths[:, mask].sum(axis=1) + row.observed_prefix_inches, [.1, .9])
            np.testing.assert_allclose([row.q10_full_month_inches, row.q90_full_month_inches], quantiles, rtol=0, atol=1e-6)

    def test_fixed_november_monthly_test_means_sum_matching_daily_forecasts(self):
        daily = pd.read_csv(ROOT / 'outputs/deep_dillon_test_daily.csv', parse_dates=['date'])
        monthly = pd.read_csv(ROOT / 'outputs/deep_dillon_test_monthly.csv')
        part = monthly.loc[monthly.fixed_nov1 & monthly.model.eq('multivariate_gru')]
        self.assertEqual(set(part.season_start), set(model.TEST_YEARS))
        self.assertLessEqual(len(part), 10)
        for row in part.itertuples():
            mask = (daily.season_start.eq(row.season_start) & daily.origin.eq(row.origin)
                    & daily.model.eq('multivariate_gru')
                    & daily.date.dt.to_period('M').astype(str).eq(row.month))
            self.assertAlmostEqual(row.predicted_inches, daily.loc[mask, 'predicted_inches'].sum(), places=5)

    def test_saved_forecast_checkpoint_reproduces_exported_distribution(self):
        saved = torch.load(ROOT / 'outputs/deep_models/dillon_gru_forecast.pt',
                           map_location='cpu', weights_only=True)
        scaler = model.Scaler()
        scaler.mean = np.asarray(saved['scaler']['mean'])
        scaler.std = np.asarray(saved['scaler']['std'])
        scaler.roni_mean = saved['scaler']['roni_mean']
        scaler.roni_std = saved['scaler']['roni_std']
        model.seed_everything(saved['seed'])
        net = model.SnowGRU(saved['config']['hidden'], saved['config']['dropout'])
        net.load_state_dict(saved['state_dict'], strict=True)
        frame = pd.read_csv(ROOT / self.result['input_path'], parse_dates=['date']).set_index('date')
        origin = frame.index.max() + pd.Timedelta(days=1)
        params = model.predict(net, model.arrays(frame, [(frame.index.max().year, origin)], scaler))[0]
        params = params[:len(self.daily)]
        expected = {'p_snow': params[:, 0], 'positive_log_mu': params[:, 1],
                    'positive_log_sigma': params[:, 2], 'predicted_inches': model.distribution_mean(params)}
        for column, quantile in [('q05_inches', .05), ('q10_inches', .1), ('q90_inches', .9), ('q95_inches', .95)]:
            expected[column] = model.distribution_quantile(params, quantile)
        for column, values in expected.items():
            np.testing.assert_allclose(values, self.daily[column], rtol=0, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
