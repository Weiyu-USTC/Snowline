"""Keep airport gauge quantities and actual report clocks scientifically distinct."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import prepare_neural48_asos as asos


def raw(rows):
    return pd.DataFrame([dict(station='SLC', valid=when, p01i=amount, metar=metar,
        tmpf='32', dwpf='30', drct='270', sknt='10', wxcodes='-SN')
        for when, amount, metar in rows])


class AsosGuards(unittest.TestCase):
    def test_trace_is_flagged_and_missing_not_zero(self):
        d = asos.prepare(raw([('2024-01-01 00:54','0.0001','KSLC AUTO'),
                              ('2024-01-01 01:54','M','KSLC AUTO')]))
        self.assertTrue(d.trace_precipitation.iloc[0])
        self.assertEqual(d.gauge_precipitation_inches.iloc[0], 0.)
        self.assertTrue(pd.isna(d.gauge_precipitation_inches.iloc[1]))
        self.assertFalse(d.valid_gauge_amount.iloc[1])

    def test_gaps_do_not_become_hourly_zero_or_fake_timestamps(self):
        d = asos.prepare(raw([('2024-01-01 00:54','0.01','KSLC AUTO'),
                              ('2024-01-01 02:54','0.02','KSLC AUTO')]))
        self.assertEqual(len(d), 2)
        self.assertEqual(d.observation_time_utc.iloc[1].minute, 54)
        self.assertEqual(d.hours_since_previous_report.iloc[1], 2.)
        self.assertFalse(d.eligible_contiguous_hourly_report.iloc[1])

    def test_duplicate_routine_hour_is_quarantined(self):
        d = asos.prepare(raw([('2024-01-01 00:54','0.01','KSLC AUTO'),
                              ('2024-01-01 01:53','0.02','KSLC AUTO'),
                              ('2024-01-01 01:54','0.03','KSLC AUTO')]))
        self.assertEqual(d.multiple_routine_reports_in_hour.sum(), 2)
        self.assertTrue(d.gauge_precipitation_inches.iloc[1:].isna().all())

    def test_sensor_unavailability_and_unit_conversion(self):
        d = asos.prepare(raw([('2024-01-01 00:54','0.01','KSLC AUTO'),
                              ('2024-01-01 01:54','0.00','KSLC AUTO RMK PNO')]))
        self.assertAlmostEqual(d.gauge_precipitation_mm.iloc[0], .254)
        self.assertFalse(d.valid_gauge_amount.iloc[1])
        self.assertTrue(pd.isna(d.gauge_precipitation_mm.iloc[1]))
        self.assertFalse(d.gauge_reset_endpoints_independently_verified.any())


if __name__ == '__main__':
    unittest.main()
