from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZIP_STORED

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mammoth_radar as radar

NOW = datetime(2026, 11, 24, 18, 28, tzinfo=timezone.utc)


def field(value, when=NOW):
    info = dict(valid_time=when.isoformat(), point_native_value=value, sha256="example",
                station_native_latitude=37.655, station_native_longitude=-119.035)
    return np.full((2, 2), value), np.array([37.665, 37.655]), np.array([-119.045, -119.035]), info


class MammothRadar(unittest.TestCase):
    def test_missing_poor_quality_and_true_dry_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for rate, quality, expectation, point in (
                (0., .4, "usable", 0.), (2.5, .4, "usable", 2.5),
                (0., 0., "poor_quality", None), (-3., .9, "missing", None),
                (2., np.nan, "missing", None), (0., 1.5, "missing", None),
            ):
                with self.subTest(rate=rate, quality=quality):
                    rec = radar.archive_pair(root, NOW, field(rate), field(quality), "test")
                    self.assertEqual((expectation, point), (rec["coverage"], rec["precip_rate_mm_h"]))
                    self.assertEqual(rec, radar.read_current_radar(root, NOW + timedelta(seconds=1)))

    def test_foreign_timestamp_or_coordinates_rejected_before_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rate = field(0.)
            q = field(.4, NOW + timedelta(minutes=2))
            with self.assertRaisesRegex(ValueError, "timestamps differ"):
                radar.archive_pair(root, NOW, rate, q, "test")
            q = field(.4)
            q[1][0] += .01
            with self.assertRaisesRegex(ValueError, "coordinates differ"):
                radar.archive_pair(root, NOW, rate, q, "test")
            self.assertFalse(list(root.rglob("*.npz")))

    def test_refresh_failure_preserves_observation_time_and_expiry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(radar, "_field", side_effect=[field(2.), field(.3)]):
                rec = radar.fetch_current_radar(root, NOW)
            self.assertEqual("succeeded", rec["status"])
            with patch.object(radar, "_field", side_effect=requests.ConnectionError("offline")):
                failed = radar.fetch_current_radar(root, NOW + timedelta(minutes=10))
            self.assertEqual("failed", failed["status"])
            self.assertEqual((rec["timestamp"], 2.), (failed["timestamp"], failed["precip_rate_mm_h"]))
            self.assertEqual("unavailable", radar.read_current_radar(root, NOW - timedelta(minutes=1))["status"])
            stale = radar.read_current_radar(root, NOW + timedelta(hours=2))
            self.assertEqual(("stale", None, rec["timestamp"]),
                             (stale["coverage"], stale["precip_rate_mm_h"], stale["timestamp"]))

    def test_old_latest_file_cannot_be_reported_as_live(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(radar, "_field", return_value=field(0., NOW - timedelta(hours=1))):
                rec = radar.fetch_current_radar(Path(folder), NOW)
            self.assertEqual(("failed", None), (rec["status"], rec["precip_rate_mm_h"]))

    def test_immutable_archive_fallback_validates_both_observation_times(self):
        with tempfile.TemporaryDirectory() as folder:
            earlier = NOW - timedelta(minutes=6)
            replies = [requests.ConnectionError("offline"), field(1., earlier), field(.3, earlier)]
            with patch.object(radar, "_field", side_effect=replies) as download:
                rec = radar.fetch_current_radar(Path(folder), NOW)
            self.assertEqual(("succeeded", earlier.isoformat()), (rec["status"], rec["timestamp"]))
            for call in download.call_args_list[1:]:
                self.assertIn("noaa-mrms-pds.s3.amazonaws.com", call.args[0])
                self.assertIn("20261124-182200", call.args[0])

    def test_bad_crc_in_newest_scan_falls_back_to_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rec = radar.archive_pair(root, NOW, field(.5), field(.4), "test")
            valid_path = root / rec["archive_path"]
            broken_path = valid_path.with_name("20261124-183000.npz")
            with ZipFile(valid_path) as valid:
                metadata = valid.read("metadata.npy")
            with ZipFile(broken_path, "w", compression=ZIP_STORED) as broken:
                broken.writestr("metadata.npy", metadata)
            damaged = bytearray(broken_path.read_bytes())
            damaged[damaged.index(metadata) + 64] ^= 1
            broken_path.write_bytes(damaged)
            self.assertEqual(rec, radar.read_current_radar(root, NOW + timedelta(minutes=3)))


if __name__ == "__main__":
    unittest.main()
