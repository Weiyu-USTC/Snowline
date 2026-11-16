"""Scientific guards for accumulation units, missing values and report clocks."""
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import prepare_neural48_targets as target


def station_frame(rows):
    path = target.ROOT / 'data/neural48_training/source_probes/test_source.csv.gz'
    text = '\n'.join('US1VTLM0001,' + r for r in rows)
    with patch.object(target, 'fetch_station', return_value=path), \
            patch.object(target.gzip, 'open', return_value=StringIO(text)):
        return target.noaa_daily('mansfield')[0]


class TargetGuards(unittest.TestCase):
    def test_noaa_snow_units_and_presumed_zero(self):
        d = station_frame(['20240101,SNOW,254,,,N,0730',
                           '20240102,SNOW,0,P,,N,0730',
                           '20240103,SNOW,0,T,,N,0730'])
        self.assertEqual(d.snowfall_inches.iloc[0], 10.)
        self.assertTrue(pd.isna(d.snowfall_inches.iloc[1]))
        self.assertFalse(d.valid_snowfall.iloc[1])
        self.assertEqual(d.snowfall_inches.iloc[2], 0.)
        self.assertTrue(d.trace.iloc[2])
        self.assertEqual(len(target.make_pairs(d)), 0)

    def test_dst_never_becomes_exact_48h(self):
        d = station_frame(['20240309,SNOW,25,,,N,0730',
                           '20240310,SNOW,25,,,N,0730',
                           '20240311,SNOW,25,,,N,0730'])
        self.assertEqual(d.period_duration_hours.iloc[1], 23.)
        pair = target.make_pairs(d).iloc[-1]
        self.assertEqual(pair.total_duration_hours, 47.)
        self.assertFalse(pair.strict_48h_eligible)

    def test_actual_clock_jitter_is_not_24h_plus_24h(self):
        d = station_frame(['20240101,SNOW,25,,,N,0730',
                           '20240102,SNOW,25,,,N,0800',
                           '20240103,SNOW,25,,,N,0730'])
        pair = target.make_pairs(d).iloc[-1]
        self.assertEqual(pair.total_duration_hours, 48.)
        self.assertEqual(pair.day1_duration_hours, 24.5)
        self.assertEqual(pair.day2_duration_hours, 23.5)
        self.assertFalse(pair.exact_24h_plus_24h)
        self.assertFalse(pair.strict_48h_eligible)

    def test_missing_prior_clock_remains_unknown(self):
        d = station_frame(['20240101,SNOW,25,,,N,',
                           '20240102,SNOW,25,,,N,0730',
                           '20240103,SNOW,25,,,N,0730'])
        pair = target.make_pairs(d).iloc[-1]
        self.assertTrue(pd.isna(pair.target_start_utc))
        self.assertFalse(pair.strict_48h_eligible)

    def test_three_nominal_clocks_create_valid_pair(self):
        d = station_frame(['20240101,SNOW,0,,,N,0730',
                           '20240102,SNOW,254,,,N,0730',
                           '20240103,SNOW,127,,,N,0730'])
        pair = target.make_pairs(d).iloc[-1]
        self.assertTrue(pair.strict_48h_eligible)
        self.assertEqual(pair.snowfall_sum_inches, 15.)
        self.assertEqual(pair.nominal_forecast_start_utc, pd.Timestamp('2024-01-01T12:30Z'))

    def test_gap_not_backfilled(self):
        d = station_frame(['20240101,SNOW,25,,,N,0730',
                           '20240103,SNOW,25,,,N,0730'])
        self.assertTrue(pd.isna(d.period_start_utc.iloc[-1]))
        self.assertEqual(len(target.make_pairs(d)), 0)


if __name__ == '__main__':
    unittest.main()
