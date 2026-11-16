"""Independent information-boundary and aggregation checks for deeper models."""
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
    raise unittest.SkipTest('Optional neural tests require requirements-deep.txt')
import torch
import deep_dillon as base
import advanced_snow_models as architectures
import benchmark_advanced_snow as benchmark


class AdvancedSnowInformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.seed_everything(20261104)
        cls.frame = pd.read_csv(ROOT / benchmark.PLAN['input'], parse_dates=['date']).set_index('date')
        cls.fit_origins = base.make_origins(cls.frame, list(range(2005, 2014)), stride=14)
        cls.scaler = base.Scaler().fit(cls.frame, cls.fit_origins)
        cls.configs = [next(c for c in benchmark.CONFIGS if c['family'] == family)
                       for family in benchmark.FAMILIES]

    def test_chronological_outer_and_inner_partitions_never_share_dates(self):
        all_test_years = []
        for fold in benchmark.FOLDS:
            fit, validation = benchmark.split_inner(fold)
            test = fold['test_years']
            self.assertEqual(fit + validation, fold['fit_years'])
            self.assertEqual(validation, fold['fit_years'][-3:])
            self.assertLess(max(fit), min(validation))
            self.assertLess(max(validation), min(test))
            sets = []
            for years in [fit, validation, test]:
                dates = set()
                for year, origin in base.make_origins(self.frame, years, stride=7):
                    start = origin - pd.Timedelta(days=base.CONTEXT)
                    end = origin + pd.Timedelta(days=base.HORIZON - 1)
                    self.assertGreaterEqual(start, pd.Timestamp(f'{year}-08-01'))
                    self.assertLessEqual(end, pd.Timestamp(f'{year+1}-04-30'))
                    dates.update(pd.date_range(start, end))
                sets.append(dates)
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[0] & sets[2])
            self.assertFalse(sets[1] & sets[2])
            all_test_years.extend(test)
        self.assertEqual(sorted(all_test_years), list(range(2017, 2026)))
        self.assertEqual(len(all_test_years), len(set(all_test_years)))

    def test_future_observed_weather_cannot_change_any_architecture_forecast(self):
        origin = pd.Timestamp('2023-11-01')
        changed = self.frame.copy()
        changed.loc[origin:, base.FEATURES] = 9999.
        original = base.arrays(self.frame, [(2023, origin)], self.scaler)
        altered = base.arrays(changed, [(2023, origin)], self.scaler)
        self.assertFalse(np.array_equal(original[2].numpy(), altered[2].numpy()))
        for config in self.configs:
            with self.subTest(family=config['family']):
                base.seed_everything(314)
                net = architectures.make_model(config).eval()
                np.testing.assert_array_equal(base.predict(net, original), base.predict(net, altered))

    def test_tcn_encoder_has_no_access_to_later_context_dates(self):
        config = next(c for c in self.configs if c['family'] == 'tcn')
        net = architectures.make_model(config).eval()
        x = torch.randn(2, base.CONTEXT, 14)
        changed = x.clone()
        changed[:, 31:, :] = 100 * torch.randn_like(changed[:, 31:, :])
        with torch.no_grad():
            left, right = net.encode(x), net.encode(changed)
        torch.testing.assert_close(left[:, :, :31], right[:, :, :31], rtol=0, atol=1e-6)
        self.assertFalse(torch.equal(left[:, :, 31:], right[:, :, 31:]))

    def test_forecast_queries_do_not_leak_later_query_features_to_earlier_leads(self):
        x = torch.randn(2, base.CONTEXT, 14)
        future = torch.randn(2, base.HORIZON, 10)
        changed = future.clone()
        changed[:, 30:, :] += 100
        for config in self.configs:
            with self.subTest(family=config['family']):
                net = architectures.make_model(config).eval()
                with torch.no_grad():
                    left, right = net(x, future), net(x, changed)
                for before, after in zip(left, right):
                    torch.testing.assert_close(before[:, :30], after[:, :30], rtol=0, atol=1e-6)

    def test_evaluation_prediction_is_independent_of_other_batch_examples(self):
        x = torch.randn(3, base.CONTEXT, 14)
        future = torch.randn(3, base.HORIZON, 10)
        changed = x.clone()
        changed[1:] += 1000
        for config in self.configs:
            with self.subTest(family=config['family']):
                net = architectures.make_model(config).eval()
                with torch.no_grad():
                    alone = net(x[:1], future[:1])
                    batch = net(changed, future)
                for a, b in zip(alone, batch):
                    torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-6)


@unittest.skipUnless((ROOT / 'outputs/advanced_snow_results.json').exists(), 'Benchmark not complete yet')
class AdvancedSnowSavedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / 'outputs/advanced_snow_results.json').read_text())
        cls.daily = pd.read_csv(ROOT / 'outputs/advanced_snow_daily_predictions.csv')
        cls.monthly = pd.read_csv(ROOT / 'outputs/advanced_snow_monthly_predictions.csv')

    def test_saved_choices_follow_inner_validation_not_outer_errors(self):
        for fold in self.result['folds']:
            for family in benchmark.FAMILIES:
                choices = [t for t in fold['trials'] if t['config']['family'] == family]
                expected = min(choices, key=lambda t: t['validation_nll'])
                self.assertEqual(fold['choices'][family], expected)
                self.assertEqual(fold['refits'][family]['best_epoch'], expected['best_epoch'])
                self.assertLess(pd.Timestamp(fold['refits'][family]['scaler']['fit_context_end']),
                                pd.Timestamp(f"{min(fold['test_years'])}-08-01"))
            expected_family = min(fold['choices'], key=lambda family: fold['choices'][family]['validation_nll'])
            self.assertEqual(fold['selected_family'], expected_family)

    def test_all_models_are_scored_on_identical_daily_targets(self):
        expected = None
        for name, rows in self.daily.groupby('model'):
            indexed = rows.set_index(['season_start', 'origin', 'date', 'lead_day']).sort_index()
            self.assertTrue(indexed.index.is_unique, name)
            if expected is None:
                expected = indexed.actual_inches
            else:
                pd.testing.assert_series_equal(indexed.actual_inches, expected)

    def test_monthly_point_means_are_sums_of_the_matching_fixed_origin_daily_means(self):
        daily = self.daily.copy()
        daily['month'] = pd.to_datetime(daily.date).dt.to_period('M').astype(str)
        grouped = daily.groupby(['model', 'season_start', 'origin', 'month']).predicted_inches.sum()
        rows = self.monthly.loc[self.monthly.model.ne('direct_monthly_climatology')]
        for row in rows.itertuples():
            actual = grouped.loc[(row.model, row.season_start, row.origin, row.month)]
            self.assertAlmostEqual(row.predicted_inches, actual, delta=2e-5)

    def test_fixed_ensemble_and_selected_rule_use_prespecified_construction(self):
        keys = ['season_start', 'origin', 'date', 'lead_day']
        wide = self.daily.pivot(index=keys, columns='model', values='predicted_inches')
        np.testing.assert_allclose(wide.equal_family_mean, wide[benchmark.FAMILIES].mean(axis=1),
                                   rtol=0, atol=1e-7)
        for fold in self.result['folds']:
            selected = fold['selected_family']
            mask = wide.index.get_level_values('season_start').isin(fold['test_years'])
            np.testing.assert_array_equal(wide.loc[mask, 'validation_selected_network'], wide.loc[mask, selected])

    def test_saved_architecture_checkpoints_reproduce_heldout_predictions(self):
        frame = pd.read_csv(ROOT / benchmark.PLAN['input'], parse_dates=['date']).set_index('date')
        fold = self.result['folds'][-1]
        year = fold['test_years'][0]
        origin = pd.Timestamp(f'{year}-11-01')
        for family in benchmark.FAMILIES:
            with self.subTest(family=family):
                saved = torch.load(ROOT / f'outputs/advanced_models/{family}_fold{fold["id"]}.pt',
                                   weights_only=True, map_location='cpu')
                self.assertEqual(saved['fit_season_starts'], fold['fit_years'])
                scaler = base.Scaler()
                scaler.mean, scaler.std = np.array(saved['scaler']['mean']), np.array(saved['scaler']['std'])
                scaler.roni_mean, scaler.roni_std = saved['scaler']['roni_mean'], saved['scaler']['roni_std']
                net = architectures.make_model(saved['config'])
                net.load_state_dict(saved['state_dict'], strict=True)
                params = base.predict(net, base.arrays(frame, [(year, origin)], scaler))[0]
                saved_rows = self.daily.loc[self.daily.model.eq(family) & self.daily.origin.eq(str(origin.date()))].sort_values('lead_day')
                self.assertEqual(len(saved_rows), base.HORIZON)
                np.testing.assert_allclose(base.distribution_mean(params), saved_rows.predicted_inches,
                                           rtol=0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
