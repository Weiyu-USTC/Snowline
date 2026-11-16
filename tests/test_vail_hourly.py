"""Information-boundary checks for the independent Vail hourly experiment."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Optional neural tests require requirements-deep.txt')
import torch
import vail_hourly_model as vail


class VailInformationBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(4104)
        clock = pd.date_range('2023-10-01', '2024-04-30 23:00', freq='h')
        cls.frame = pd.DataFrame({'temperature': np.arange(len(clock), dtype=float),
                                  'second_channel': np.sin(np.arange(len(clock))/24)},
                                 index=clock)
        endpoints = pd.date_range('2023-10-01 04:00', '2024-04-30 04:00')
        cls.target = pd.Series(np.arange(len(endpoints), dtype=float) % 7,
                               index=endpoints)

    def arrays(self, frame=None, target=None):
        # RONI is irrelevant to these boundaries; remove dependence on external
        # historical data so this fixture remains deterministic.
        with patch.object(vail, 'lagged_roni', return_value=0.):
            return vail.make_arrays(self.frame if frame is None else frame,
                                    self.target if target is None else target,
                                    [2023])

    def test_context_ends_at_issue_and_targets_start_next_daily_endpoint(self):
        meta, x, f, y = self.arrays()
        self.assertEqual(x.shape[1], 168)
        self.assertEqual(y.shape[1], 60)
        self.assertEqual(f.shape[1:], (60, 10))
        origin = pd.Timestamp(meta.iloc[0].origin)
        pd.testing.assert_index_equal(
            pd.date_range(end=origin, periods=168, freq='h'),
            self.frame.loc[origin-pd.Timedelta(hours=167):origin].index)
        np.testing.assert_array_equal(x[0], self.frame.loc[
            origin-pd.Timedelta(hours=167):origin].to_numpy().astype('float32'))
        np.testing.assert_array_equal(y[0], self.target.reindex(
            pd.date_range(origin+pd.Timedelta(days=1), periods=60)).to_numpy())
        for row in meta.itertuples():
            origin = pd.Timestamp(row.origin)
            self.assertGreaterEqual(origin-pd.Timedelta(hours=167),
                                    pd.Timestamp('2023-08-01'))
            self.assertLessEqual(origin+pd.Timedelta(days=60),
                                 pd.Timestamp('2024-04-30 23:59'))

    def test_future_observations_and_labels_cannot_change_current_inputs(self):
        meta, x, f, y = self.arrays()
        origin = pd.Timestamp(meta.iloc[0].origin)
        altered_frame, altered_target = self.frame.copy(), self.target.copy()
        altered_frame.loc[altered_frame.index > origin] = 987654.
        altered_target.loc[altered_target.index > origin] = 12345.
        _, changed_x, changed_f, changed_y = self.arrays(altered_frame, altered_target)
        np.testing.assert_array_equal(x[0], changed_x[0])
        np.testing.assert_array_equal(f[0], changed_f[0])
        self.assertFalse(np.array_equal(y[0], changed_y[0]))

    def test_missing_target_stays_missing_in_eligible_origin(self):
        target = self.target.copy()
        target.loc['2023-10-12 04:00'] = np.nan
        meta, _, _, y = self.arrays(target=target)
        origin = pd.Timestamp(meta.iloc[0].origin)
        lead_index = (pd.Timestamp('2023-10-12 04:00')-origin).days-1
        self.assertTrue(np.isnan(y[0, lead_index]))
        self.assertEqual(np.isfinite(y[0]).sum(), 59)

    def test_source_quality_quarantines_mask_context_and_target_not_raw_archive(self):
        dates = pd.to_datetime(['2020-03-19 04:00', '2020-03-20 04:00',
                                '2020-05-31 04:00', '2020-06-01 04:00',
                                '2023-09-30 04:00', '2023-10-01 04:00',
                                '2023-11-28 04:00', '2023-11-29 04:00',
                                '2024-01-01 04:00'])
        records = []
        for station in vail.STATIONS:
            for date in dates:
                records.append(dict(station_id=station, datetime_display=str(date),
                                    Temp=20., DewP=15., RH=80., Spd=5., Gst=10.,
                                    Dir=270., Sno24=-1. if date.year == 2024 else 8.,
                                    Pcp1=.1, MSLP=1000.))
        source = pd.DataFrame(records)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(vail, 'OUT', Path(directory)), patch.object(vail.pd, 'read_csv', return_value=source.copy()):
                raw, frame, target, _ = vail.load_data()
        masked = ['2020-03-20 04:00', '2020-05-31 04:00',
                  '2023-10-01 04:00', '2023-11-28 04:00', '2024-01-01 04:00']
        valid = ['2020-03-19 04:00', '2020-06-01 04:00',
                 '2023-09-30 04:00', '2023-11-29 04:00']
        self.assertTrue(target.loc[masked].isna().all())
        self.assertTrue(frame.loc[masked, 'CAVMM_Sno24'].isna().all())
        self.assertTrue((target.loc[valid] == 8.).all())
        np.testing.assert_allclose(frame.loc[valid, 'CAVMM_Sno24'], np.log1p(8.))
        self.assertTrue((frame.loc[masked, 'CAVMM_Temp'] == 20.).all())
        np.testing.assert_array_equal(raw.Sno24.to_numpy(), source.Sno24.to_numpy())

    def test_scaler_fits_only_supplied_training_context_and_marks_missing(self):
        training = np.zeros((2, 168, 2), dtype='float32')
        training[..., 0] = 3.
        training[..., 1] = np.nan
        scaler = vail.HourlyScaler().fit(training)
        mean_before, std_before = scaler.mean.copy(), scaler.std.copy()
        test = np.full((1, 168, 2), 1000000., dtype='float32')
        test[:, -1, 1] = np.nan
        transformed = scaler.transform(test)
        np.testing.assert_array_equal(scaler.mean, mean_before)
        np.testing.assert_array_equal(scaler.std, std_before)
        np.testing.assert_array_equal(scaler.mean, [3., 0.])
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.shape, (1, 168, 6))
        self.assertEqual(transformed[0, -1, 1], 0.)
        self.assertEqual(transformed[0, -1, 3], 1.)

    def test_unseen_constant_future_feature_cannot_inject_prediction_shift(self):
        x = np.zeros((2, 168, 2), dtype='float32')
        fit_future = np.zeros((2, 60, 10), dtype='float32')
        fit_future[0, :, 6], fit_future[1, :, 6] = -1.5, 1.4
        fit_future[:, :, 9] = 1.  # Also check a nonzero constant is preserved.
        scaler = vail.HourlyScaler().fit(x).fit_future(fit_future)
        supported = fit_future[1:2].copy()
        unseen = supported.copy()
        unseen[..., 7] = 1.  # Strong El Nino never appeared in fit samples.
        unseen[..., 9] = 0.
        model = vail.VailTCN(6, width=8, dropout=0.).eval()
        np.testing.assert_array_equal(vail.predict(model, scaler, x[:1], supported),
                                      vail.predict(model, scaler, x[:1], unseen))
        transformed = scaler.transform_future(unseen)
        self.assertTrue((transformed[..., 7] == 0.).all())
        self.assertTrue((transformed[..., 9] == 1.).all())

    def test_future_roni_clipping_uses_only_fit_range_and_does_not_mutate_inputs(self):
        fit_future = np.zeros((2, 60, 10), dtype='float32')
        fit_future[0, :, 6], fit_future[1, :, 6] = -1.5, 1.4
        scaler = vail.HourlyScaler().fit_future(fit_future)
        fit_min, fit_max = scaler.future_min.copy(), scaler.future_max.copy()
        heldout = fit_future.copy()
        heldout[0, :, 6], heldout[1, :, 6] = -9., 1.7
        saved = heldout.copy()
        transformed = scaler.transform_future(heldout)
        np.testing.assert_allclose(transformed[0, :, 6], -1.5)
        np.testing.assert_allclose(transformed[1, :, 6], 1.4)
        np.testing.assert_array_equal(scaler.future_min, fit_min)
        np.testing.assert_array_equal(scaler.future_max, fit_max)
        np.testing.assert_array_equal(heldout, saved)
        # No validation statistic can extend the fit range through transform.
        np.testing.assert_array_equal(scaler.transform_future(fit_future), fit_future)

    def test_causal_encoder_never_uses_later_context_hours(self):
        model = vail.VailTCN(6, width=8, dropout=0.).eval()
        x = torch.randn(2, 168, 6)
        altered = x.clone()
        altered[:, 83:, :] += 100.
        with torch.no_grad():
            a = model.norm(model.blocks(model.projection(x.transpose(1, 2))))
            b = model.norm(model.blocks(model.projection(altered.transpose(1, 2))))
        torch.testing.assert_close(a[:, :, :83], b[:, :, :83], rtol=0, atol=1e-6)
        self.assertFalse(torch.equal(a[:, :, 83:], b[:, :, 83:]))

    def test_masked_target_cannot_change_likelihood_or_gradients(self):
        base = [torch.randn(2, 60), torch.randn(2, 60), torch.ones(2, 60)]
        target = torch.ones(2, 60)
        mask = torch.ones_like(target)
        mask[:, 4:11] = 0.
        corrupted = target.clone()
        corrupted[:, 4:11] = 10000.
        a = [value.clone().requires_grad_() for value in base]
        b = [value.clone().requires_grad_() for value in base]
        loss_a, loss_b = vail.logloss(a, target, mask), vail.logloss(b, corrupted, mask)
        loss_a.backward(); loss_b.backward()
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
        for left, right in zip(a, b):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_inner_selection_and_refit_never_receive_outer_test_indices(self):
        meta = pd.DataFrame({'winter': np.repeat([2019, 2020, 2022, 2023, 2024, 2025], 2)})
        training = np.arange(8)
        outer_test = set(range(8, 12))
        x = np.zeros((12, 168, 1), dtype='float32')
        f = np.zeros((12, 60, 10), dtype='float32')
        y = np.zeros((12, 60), dtype='float32')
        calls = []

        def fake_train(x, f, y, fit, valid, config, **kwargs):
            calls.append((set(fit), None if valid is None else set(valid)))
            model = torch.nn.Linear(1, 1)
            return model, object(), [], 2, float(config['width'])

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(vail, 'OUT', Path(directory)), patch.object(vail, 'train_model', side_effect=fake_train):
                _, _, record = vail.select_and_fit(meta, x, f, y, training, 'check')
            self.assertTrue((Path(directory)/'check_training.json').exists())
        self.assertEqual(len(calls), len(vail.CONFIGS)+1)
        for fit, valid in calls[:-1]:
            self.assertFalse(fit & outer_test)
            self.assertFalse(valid & outer_test)
            self.assertFalse(fit & valid)
            self.assertEqual(fit | valid, set(training))
            self.assertFalse(set(meta.iloc[sorted(fit)].winter) & set(meta.iloc[sorted(valid)].winter))
        self.assertEqual(calls[-1], (set(training), None))
        self.assertEqual(record['training_winters'], [2019, 2020, 2022, 2023])

    def test_saved_model_and_scaler_restore_identical_60_day_distribution(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(2, 168, 2)).astype('float32')
        x[0, :10, 1] = np.nan
        fit_future = np.zeros((2, 60, 10), dtype='float32')
        fit_future[0, :, 6], fit_future[1, :, 6] = -1.5, 1.4
        scaler = vail.HourlyScaler().fit(x).fit_future(fit_future)
        model = vail.VailTCN(6, width=8, dropout=.25).eval()
        f = fit_future.copy()
        f[..., 6], f[..., 7] = 1.7, 1.  # Activates both future safeguards.
        before = vail.predict(model, scaler, x, f)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = vail.VailTCN(6, width=8, dropout=.25)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        saved_scaler = json.loads(json.dumps(scaler.as_dict()))
        restored_scaler = vail.HourlyScaler()
        restored_scaler.mean = np.array(saved_scaler['mean'])
        restored_scaler.std = np.array(saved_scaler['std'])
        restored_scaler.future_min = np.array(saved_scaler['future_min'], dtype='float32')
        restored_scaler.future_max = np.array(saved_scaler['future_max'], dtype='float32')
        after = vail.predict(restored, restored_scaler, x, f)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(after.shape, (2, 60, 3))
        self.assertTrue(np.isfinite(after).all())
        self.assertTrue(((after[..., 0] >= 0) & (after[..., 0] <= 1)).all())
        self.assertTrue((after[..., 2] > 0).all())

    def test_history_climate_and_path_dependence_exclude_nontraining_year(self):
        prior = self.target.rename(index=lambda d:d-pd.DateOffset(years=1))
        # Shifting leap day into a non-leap year duplicates Feb28 in this fixture.
        prior = prior[~prior.index.duplicated()]
        target = pd.concat([self.target, prior]).sort_index()
        modified = target.copy()
        modified.loc['2023-08-01':] = 50000.
        origins = [pd.Timestamp('2023-11-04 04:00')]
        np.testing.assert_array_equal(vail.climate(target, [2022], origins),
                                      vail.climate(modified, [2022], origins))
        params = np.tile([.6, 1., .5], (20, 1))
        first = pd.Timestamp('2023-11-05 04:00')
        original, count = vail.forecast_paths(target, [2022], first, params, n=64)
        changed, changed_count = vail.forecast_paths(modified, [2022], first, params, n=64)
        self.assertGreater(count, 0)
        self.assertEqual(count, changed_count)
        np.testing.assert_array_equal(original, changed)


if __name__ == '__main__':
    unittest.main()
