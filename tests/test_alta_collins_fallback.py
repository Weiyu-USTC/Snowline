import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import alta_collins_fallback as feed


LOCATION = {'id': 2, 'slug': 'mid_mountain', 'elevation': 9662}


def reading(stamp, **overrides):
    return dict({'location': LOCATION, 'location_id': 2, 'dated_at': stamp,
                 'temperature': 23, 'temperature_as_celsius': -5,
                 'water': .01, 'water_as_cm': .03, 'wind_speed': 0}, **overrides)


def page(*rows):
    return ('<script>window.Alta = ' + json.dumps({'weather': {'observation': {
        'locations': [LOCATION], 'weather': list(rows)}}}) + ';</script>').encode()


class CollinsFallbackTests(unittest.TestCase):
    def test_exact_site_units_and_timestamp(self):
        now = pd.Timestamp('2026-11-23T04:07Z')
        body = page(reading('2026-11-22 20:00:00'),
                    reading('2026-11-22 20:00:00', location_id=3),
                    reading('2026-11-22 20:00:00', location=dict(LOCATION, elevation=8560)),
                    reading('2026-11-22 22:00:00'))
        data = feed._parse_report(body, now)
        self.assertEqual(len(data), 1)
        row = data.iloc[0]
        self.assertEqual(row.station_id, 'CLN')
        self.assertEqual(row.valid_time_utc, pd.Timestamp('2026-11-23T03:00Z'))
        self.assertEqual(row.temperature_c, -5)
        self.assertAlmostEqual(row.precipitation_mm, .254)
        self.assertTrue(pd.isna(row.wind_u_ms) and pd.isna(row.snow_depth_cm))
        self.assertFalse(row.hourly_fresh_snow_observed)

    def test_ambiguous_and_nonexistent_local_hours_excluded(self):
        for day,invalid,valid,utc in [('2026-11-01', '01:00', '02:00', '2026-11-01T09:30Z'),
                                     ('2026-03-08', '02:00', '03:00', '2026-03-08T09:30Z')]:
            with self.subTest(day=day):
                data = feed._parse_report(page(reading(f'{day} {invalid}:00'), reading(f'{day} {valid}:00')),
                                          pd.Timestamp(utc))
                self.assertEqual(len(data), 1)
                self.assertEqual(data.iloc[0].valid_time_utc, pd.Timestamp(utc) - pd.Timedelta(minutes=30))

    def test_stale_response_preserves_cache(self):
        old = page(reading('2026-11-22 10:00:00'))
        response = Mock()
        response.iter_content.return_value = [old]
        context = MagicMock()
        context.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as folder, patch.object(feed.requests, 'get', return_value=context):
            path = Path(folder) / feed.CACHE_NAME
            path.write_text('previous good snapshot')
            with self.assertRaisesRegex(ValueError, 'last three hours'):
                feed.fetch_collins_fallback(folder, now='2026-11-23T04:07Z')
            self.assertEqual(path.read_text(), 'previous good snapshot')


if __name__ == '__main__':
    unittest.main()
