from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prepare_neural48_radar import encode_context, select_context


class IssueKnownRadar(unittest.TestCase):
    def fields(self):
        rt = np.datetime64('2020-11-01T09:20:00') + np.arange(29)*np.timedelta64(10, 'm')
        qt = np.datetime64('2020-11-01T09:00:00') + np.arange(6)*np.timedelta64(1, 'h')
        return [np.ones((29,64,64)), np.ones((29,64,64)), np.ones((6,64,64)), rt, qt,
                '2020-11-01T10:50:00+00:00', np.ones(29,bool), np.ones(6,bool),
                np.ones((1,29)), np.ones((1,6))]

    def test_future_poison_never_enters_inputs(self):
        a = self.fields(); expected = select_context(*a)
        a[0][9:] = 1e9; a[2][2:] = 1e9; a[8][:,9:] = 1e9; a[9][:,2:] = 1e9
        actual = select_context(*a)
        for key in expected: np.testing.assert_array_equal(expected[key], actual[key])

    def test_missing_and_dry_have_different_encodings(self):
        a = self.fields(); a[0][:9] = 0
        dry = select_context(*a)
        lat = np.arange(64); lon = np.arange(64)
        dry_x = encode_context(dry, lat, lon, {'site': (32,32)})
        a[0][:9] = np.nan; a[1][:9] = 0
        missing_x = encode_context(select_context(*a), lat, lon, {'site': (32,32)})
        self.assertTrue(np.all(dry_x['x'][:,0] == missing_x['x'][:,0]))
        self.assertTrue(np.all(dry_x['x'][:,2] == 1))
        self.assertTrue(np.all(missing_x['x'][:,2] == 0))
        self.assertTrue(np.isnan(missing_x['pooled_rate_mm_h']).all())

    def test_low_quality_remains_visible_but_invalid_rate_is_masked(self):
        a=self.fields(); a[2][:2]=.05
        out=encode_context(select_context(*a), np.arange(64), np.arange(64), {'site':(32,32)})
        self.assertTrue(np.all(out['x'][:,0] == 0))
        np.testing.assert_allclose(out['x'][:,1], .05)
        self.assertTrue(np.all(out['x'][:,2] == 0))
        self.assertTrue(np.all(out['x'][:,4] == 1))

    def test_causal_quality_changes_only_after_timestamp(self):
        a=self.fields(); a[2][0]=.2; a[2][1]=.7
        context=select_context(*a)
        np.testing.assert_allclose(context['quality'][:4],.2)
        np.testing.assert_allclose(context['quality'][4:],.7)


if __name__ == '__main__': unittest.main()
