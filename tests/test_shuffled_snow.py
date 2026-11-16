"""Independent grouping, selection and saved-output checks for shuffled CV."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Optional snowfall neural tests require requirements-deep.txt')
import torch
import deep_dillon as base
import crossvalidate_advanced_snow as cv
from advanced_snow_models import make_model


class ShuffledWinterDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.seed_everything(cv.SEED)
        cls.frame = pd.read_csv(ROOT / cv.PLAN['input'], parse_dates=['date']).set_index('date')

    def test_every_whole_winter_is_outer_test_once_and_inner_test_once(self):
        outer_test = []
        for fold in cv.FOLDS:
            fit, test = set(fold['fit_years']), set(fold['test_years'])
            self.assertFalse(fit & test)
            self.assertEqual(fit | test, set(cv.YEARS))
            outer_test.extend(fold['test_years'])
            inner_valid = []
            for inner in fold['inner_folds']:
                train, valid = set(inner['fit_years']), set(inner['validation_years'])
                self.assertFalse(train & valid)
                self.assertFalse((train | valid) & test)
                self.assertEqual(train | valid, fit)
                inner_valid.extend(inner['validation_years'])
            self.assertEqual(sorted(inner_valid), sorted(fit))
        self.assertEqual(sorted(outer_test), cv.YEARS)

    def test_shuffling_keeps_actual_context_and_target_dates_disjoint(self):
        winter_days = {}
        for year in cv.YEARS:
            days = set()
            for _, origin in base.make_origins(self.frame, [year], stride=7):
                days.update(pd.date_range(origin-pd.Timedelta(days=base.CONTEXT),
                                          origin+pd.Timedelta(days=base.HORIZON-1)))
            winter_days[year] = days
        for fold in cv.FOLDS:
            test_days = set().union(*(winter_days[y] for y in fold['test_years']))
            for inner in fold['inner_folds']:
                fit_days = set().union(*(winter_days[y] for y in inner['fit_years']))
                valid_days = set().union(*(winter_days[y] for y in inner['validation_years']))
                self.assertFalse(fit_days & valid_days)
                self.assertFalse(fit_days & test_days)
                self.assertFalse(valid_days & test_days)

    def test_fold_assignment_cannot_be_changed_by_snowfall_outcomes(self):
        original, _ = cv.balance_audit(self.frame)
        changed = self.frame.copy()
        changed['snowfall_inches'] = 1000.
        altered, _ = cv.balance_audit(changed)
        pd.testing.assert_series_equal(original.outer_fold, altered.outer_fold)
        self.assertFalse(original.nov_dec_total_inches.equals(altered.nov_dec_total_inches))
        self.assertEqual(cv.make_folds(), cv.FOLDS)

    def test_heldout_weather_cannot_change_inner_training_scaler(self):
        fold = cv.FOLDS[0]
        origins = base.make_origins(self.frame, fold['inner_folds'][0]['fit_years'], stride=7)
        original = base.Scaler().fit(self.frame, origins)
        changed = self.frame.copy()
        for year in fold['test_years']:
            changed.loc[f'{year}-08-01':f'{year+1}-04-30', base.FEATURES] = 9999.
        altered = base.Scaler().fit(changed, origins)
        np.testing.assert_array_equal(original.mean, altered.mean)
        np.testing.assert_array_equal(original.std, altered.std)
        self.assertEqual(original.roni_mean, altered.roni_mean)
        self.assertEqual(original.roni_std, altered.roni_std)

    def test_outer_scores_are_not_inputs_to_family_selection(self):
        trials = [{'config': {'family': family, 'width': width}, 'mean_validation_nll': 1+width/100.,
                   'outer_rmse': 100-width} for family in cv.FAMILIES for width in [16, 32]]
        choices, family = cv.choose_trials(trials)
        changed = json.loads(json.dumps(trials))
        for trial in changed:
            trial['outer_rmse'] *= -1000
        changed_choices, changed_family = cv.choose_trials(changed)
        self.assertEqual(family, changed_family)
        self.assertEqual({k: v['config'] for k, v in choices.items()},
                         {k: v['config'] for k, v in changed_choices.items()})


@unittest.skipUnless((ROOT / 'outputs/shuffled_snow_results.json').exists(), 'Shuffled outer CV not complete yet')
class ShuffledWinterSavedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / 'outputs/shuffled_snow_results.json').read_text())
        cls.daily = pd.read_csv(ROOT / 'outputs/shuffled_snow_daily_predictions.csv')
        cls.monthly = pd.read_csv(ROOT / 'outputs/shuffled_snow_monthly_predictions.csv')

    def test_saved_family_choices_and_epoch_counts_use_inner_folds_only(self):
        for fold in self.result['folds']:
            for trial in fold['trials']:
                self.assertEqual(len(trial['inner_trials']), 2)
                inner_nll = [t['trial']['validation_nll'] for t in trial['inner_trials']]
                inner_epochs = [t['trial']['best_epoch'] for t in trial['inner_trials']]
                self.assertAlmostEqual(trial['mean_validation_nll'], float(np.mean(inner_nll)))
                self.assertEqual(trial['refit_epochs'], int(math.floor(float(np.median(inner_epochs))+.5)))
                for inner in trial['inner_trials']:
                    self.assertFalse(set(inner['fit_years']) & set(fold['test_years']))
                    self.assertFalse(set(inner['validation_years']) & set(fold['test_years']))
            choices, selected = cv.choose_trials(fold['trials'])
            self.assertEqual(choices, fold['choices'])
            self.assertEqual(selected, fold['selected_family'])
            for family in cv.FAMILIES:
                self.assertEqual(fold['refits'][family]['best_epoch'], choices[family]['refit_epochs'])

    def test_one_outer_prediction_per_origin_lead_on_identical_model_cohorts(self):
        assignment = {year: fold['id'] for fold in cv.FOLDS for year in fold['test_years']}
        self.assertEqual(set(self.daily.season_start), set(cv.YEARS))
        self.assertTrue((self.daily.outer_fold == self.daily.season_start.map(assignment)).all())
        reference = None
        for model, rows in self.daily.groupby('model'):
            indexed = rows.set_index(['season_start', 'origin', 'date', 'lead_day']).sort_index()
            self.assertTrue(indexed.index.is_unique, model)
            if reference is None:
                reference = indexed.actual_inches
            else:
                pd.testing.assert_series_equal(indexed.actual_inches, reference)

    def test_monthly_forecasts_sum_daily_predictions_from_the_same_origin(self):
        daily = self.daily.copy()
        daily['month'] = pd.to_datetime(daily.date).dt.to_period('M').astype(str)
        grouped = daily.groupby(['model', 'season_start', 'origin', 'month']).predicted_inches.sum()
        for row in self.monthly.loc[self.monthly.model.ne('direct_monthly_climatology')].itertuples():
            self.assertAlmostEqual(row.predicted_inches,
                                   grouped.loc[(row.model, row.season_start, row.origin, row.month)], delta=2e-5)

    def test_ensemble_and_selected_family_do_not_use_outer_scores(self):
        wide = self.daily.pivot(index=['season_start', 'origin', 'date', 'lead_day'],
                                columns='model', values='predicted_inches')
        np.testing.assert_allclose(wide.equal_family_mean, wide[cv.FAMILIES].mean(axis=1), rtol=0, atol=1e-7)
        for fold in self.result['folds']:
            mask = wide.index.get_level_values('season_start').isin(fold['test_years'])
            np.testing.assert_array_equal(wide.loc[mask, 'validation_selected_network'],
                                          wide.loc[mask, fold['selected_family']])

    def test_safe_saved_checkpoint_reload_reproduces_heldout_forecast(self):
        frame = pd.read_csv(ROOT / cv.PLAN['input'], parse_dates=['date']).set_index('date')
        fold = self.result['folds'][1]
        year = fold['test_years'][0]
        origin = pd.Timestamp(f'{year}-11-01')
        for family in cv.FAMILIES:
            saved = torch.load(ROOT / fold['refits'][family]['checkpoint'], weights_only=True, map_location='cpu')
            self.assertEqual(saved['fit_season_starts'], fold['fit_years'])
            self.assertNotIn(year, saved['fit_season_starts'])
            scaler = base.Scaler()
            scaler.mean, scaler.std = np.asarray(saved['scaler']['mean']), np.asarray(saved['scaler']['std'])
            scaler.roni_mean, scaler.roni_std = saved['scaler']['roni_mean'], saved['scaler']['roni_std']
            net = make_model(saved['config'])
            net.load_state_dict(saved['state_dict'], strict=True)
            params = base.predict(net, base.arrays(frame, [(year, origin)], scaler))[0]
            rows = self.daily.loc[self.daily.model.eq(family) & self.daily.origin.eq(str(origin.date()))].sort_values('lead_day')
            self.assertEqual(len(rows), base.HORIZON)
            np.testing.assert_allclose(base.distribution_mean(params), rows.predicted_inches, rtol=0, atol=2e-6)

    def test_production_selection_uses_its_own_prespecified_crossvalidation(self):
        if 'production_models' not in self.result:
            self.skipTest('Production stage not complete yet')
        choices, selected = cv.choose_trials(self.result['production_trials'])
        self.assertEqual(selected, self.result['production_selected_family'])
        for family, metadata in self.result['production_models'].items():
            self.assertEqual(metadata['choice'], choices[family])
            saved = torch.load(ROOT / metadata['checkpoint'], weights_only=True, map_location='cpu')
            self.assertEqual(saved['fit_season_starts'], cv.YEARS)
            self.assertEqual(saved['config'], choices[family]['config'])
            self.assertEqual(saved['epochs'], choices[family]['refit_epochs'])


if __name__ == '__main__':
    unittest.main()
