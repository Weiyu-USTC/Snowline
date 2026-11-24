import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import mammoth_weather as weather


class MammothWeatherTests(unittest.TestCase):
    def test_native_ecmwf_swe_is_aligned_to_preceding_hour(self):
        run = datetime(2025, 2, 13, 12, tzinfo=timezone.utc)
        payload = dict(latitude=37.645, longitude=-119.08, elevation=2936, utc_offset_seconds=0,
                       hourly_units={'snowfall_water_equivalent':'mm'}, hourly={
                           'time': [(run + i*weather.HOUR).strftime('%Y-%m-%dT%H:%M') for i in range(73)],
                           'snowfall_water_equivalent':[None, 0, 2.4] + [0]*70})
        with tempfile.TemporaryDirectory() as root:
            with patch.object(weather,'urlopen', return_value=io.BytesIO(json.dumps(payload).encode())):
                data = weather.fetch_provider_run(root,run)
            self.assertEqual(data['hours'][0], dict(lead=1,start_utc=run.isoformat(),
                             end_utc=(run+weather.HOUR).isoformat(),snow_water_mm=0.))
            self.assertEqual(data['hours'][1]['snow_water_mm'],2.4)
            self.assertEqual(data['available_utc'],(run+8*weather.HOUR).isoformat())
            with patch.object(weather,'urlopen', side_effect=AssertionError('Run is cached')):
                self.assertEqual(weather.fetch_provider_run(root,run),data)
            self.assertIn('ecmwf_ifs',data['url'])
        with tempfile.TemporaryDirectory() as root:
            payload['hourly']['snowfall_water_equivalent'][1] = None
            with patch.object(weather,'urlopen', return_value=io.BytesIO(json.dumps(payload).encode())):
                with self.assertRaises(ValueError): weather.fetch_provider_run(root,run)
            self.assertFalse((Path(root)/weather.CACHE/'ecmwf_runs').exists())

    def test_readers_do_not_advance_old_or_future_forecasts(self):
        run = datetime(2025, 2, 13, 12, tzinfo=timezone.utc)
        for file, hours, reader in [('current.json',3,weather.read_current),('ecmwf_current.json',8,weather.read_provider)]:
            with self.subTest(file=file), tempfile.TemporaryDirectory() as root:
                data = dict(run_utc=run.isoformat(), available_utc=(run + hours*weather.HOUR).isoformat(),
                            hours=[dict(start_utc=run.isoformat(),end_utc=(run + weather.HOUR).isoformat())])
                weather.atomic_json(Path(root)/weather.CACHE/file, data)
                self.assertEqual(reader(root,run)['status'],'unavailable')
                active = reader(root,run + hours*weather.HOUR)
                self.assertEqual(active['status'],'ready')
                stale = reader(root,run + timedelta(days=2))
                self.assertEqual(stale['status'],'stale')
                self.assertEqual(stale['hours'],active['hours'])
