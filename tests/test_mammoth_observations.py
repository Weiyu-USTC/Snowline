from contextlib import closing
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mammoth_observations as obs


def table(headers, rows):
    head = '<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>'
    return '<table>' + head + ''.join(
        '<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</table>'


def resort(current=56):
    return b'Snow totals as reported from Main Lodge' + table(obs.HEADERS, [
        ['2026-27', current, 0, 5, 51] + [''] * 8,
        ['2000-1', 100, 10, 5, '', 85] + [0] * 7,
        ['1999-0', 70, 10, 5, 55] + [0] * 8,
    ]).encode()


def station(rows):
    return ('Sesame - 9,014ft | America/Los_Angeles' +
            table(['Date &amp; Time', *obs.FIELDS.values()], rows)).encode()


NOW = datetime(2026, 11, 24, 19, tzinfo=timezone.utc)
LIVE = station([['11/24/26 - 1015', '0.4', '28', '-1', '2', '8', 'S | 190'],
                ['11/24/26 - 1000', '-', '29', '23', '0', '-', '-']])


class MammothObservationTests(unittest.TestCase):
    def test_changing_responsive_copies_rejected(self):
        self.assertEqual(len(obs.parse_main_lodge(resort() + resort())), 3)
        with self.assertRaises(ValueError):
            obs.parse_main_lodge(resort() + resort(55))

    def test_month_ends_preseason_and_missing_are_kept_distinct(self):
        seasons = obs.parse_main_lodge(resort())
        points = obs._snapshots(2000, seasons['2000'], NOW.date())
        self.assertEqual(points, [dict(date='2000-10-31', total_inches=15, end=None)])
        points = obs._snapshots(1999, seasons['1999'], NOW.date())
        self.assertEqual(points[-1]['date'], '2000-07-31')
        self.assertEqual(points[-1]['total_inches'], 70)
        self.assertIn('2000-02-29', [p['date'] for p in points])

    def test_sesame_rejects_ambiguous_clocks_and_preserves_units(self):
        rows = obs.parse_sesame(station([
            ['11/01/26 - 0130', '1', '30', '12', '0', '3', 'N | 0'],
            ['11/01/26 - 0230', '1.2', '31', '-10', '0', '-', 'N | 0'],
        ]))
        self.assertEqual(list(rows), ['2026-11-01T10:30:00+00:00'])
        obs_row = next(iter(rows.values()))
        self.assertEqual((obs_row['snow_board_inches'], obs_row['snow_depth_inches']), (1.2, 31))
        self.assertEqual((obs_row['temperature_f'], obs_row['wind_mph'], obs_row['wind_direction_deg']),
                         (-10, 0, 0))
        self.assertIsNone(obs_row['wind_gust_mph'])

    def test_real_time_cache_and_independent_failures(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with patch.object(obs, '_download', side_effect=[resort(), LIVE]):
                self.assertEqual(obs.refresh_mammoth_observations(root, NOW)['status'], 'succeeded')
            payload = obs.read_mammoth_observations(root)
            self.assertEqual(payload['records'], [])
            self.assertEqual(payload['reported_totals'], [
                dict(date='2026-10-31', total_inches=5, end=None),
                dict(date='2026-11-24', total_inches=56, end=None)])
            self.assertEqual(payload['reported_total_source']['cadence'], 'monthly')
            self.assertEqual(obs.read_sesame_observation(root, NOW)['temperature_f'], -1)
            self.assertEqual(obs.read_sesame_observation(root, NOW + timedelta(hours=4))['status'], 'stale')
            with patch.object(obs, '_download', side_effect=AssertionError('recent source downloaded')):
                refreshed = obs.refresh_mammoth_observations(root, NOW + timedelta(minutes=20))
                self.assertEqual(refreshed['status'], 'succeeded')
                self.assertEqual({s['status'] for s in refreshed['sources'].values()}, {'cached'})
            with patch.object(obs, '_download', side_effect=[b'broken', LIVE]):
                refreshed = obs.refresh_mammoth_observations(root, NOW + timedelta(hours=1))
            self.assertEqual(refreshed['status'], 'partial')
            self.assertEqual(refreshed['sources']['main_lodge']['status'], 'failed')
            self.assertEqual(obs.read_mammoth_observations(root)['reported_totals'], payload['reported_totals'])
            with patch.object(obs, '_download', side_effect=[resort(), b'broken']):
                refreshed = obs.refresh_mammoth_observations(root, NOW + timedelta(hours=2))
            self.assertEqual(refreshed['status'], 'partial')
            self.assertEqual(refreshed['sources']['sesame']['status'], 'failed')
            self.assertEqual(obs.read_sesame_observation(root, NOW)['snow_board_inches'], 0.4)
            with closing(sqlite3.connect(obs._directory(root) / 'sesame.sqlite3')) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 2)

    def test_stale_html_keeps_original_report_time(self):
        with tempfile.TemporaryDirectory() as name:
            with patch.object(obs, '_download', side_effect=[resort(), LIVE]):
                stage = obs.refresh_mammoth_observations(name, NOW + timedelta(days=1))
            self.assertEqual(stage['status'], 'partial')
            self.assertEqual(obs.read_sesame_observation(name, NOW + timedelta(days=1))['observed_at_utc'],
                             '2026-11-24T18:15:00+00:00')

    def test_legacy_checks_survive_monthly_corrections_and_season_rollover(self):
        with tempfile.TemporaryDirectory() as name:
            directory = obs._directory(name)
            directory.mkdir(parents=True)
            old = dict(retrieved_at_utc=NOW.isoformat(), seasons=obs.parse_main_lodge(resort()))
            (directory / 'main_lodge.json').write_text(json.dumps(old))
            day26 = NOW + timedelta(days=2)
            rollover = datetime(2027, 10, 10, 21, tzinfo=timezone.utc)
            edited = b'Snow totals as reported from Main Lodge' + table(obs.HEADERS, [
                ['2026-27', 71, 0, 5, 66] + [0] * 8,
                ['2027-28', 12, 2, 10] + [''] * 9,
            ]).encode()
            with patch.object(obs, '_download', side_effect=[resort(60), resort(62), resort(69), edited]):
                for t in (day26, day26 + timedelta(hours=2), NOW + timedelta(days=6), rollover):
                    obs._refresh_lodge(directory, t)
            source = obs.read_mammoth_observations(name)
            self.assertEqual(source['reported_totals'], [dict(date='2027-10-10', total_inches=12, end=None)])
            season = next(s for s in source['reported_total_history'] if s['season'] == '2026/27')
            nov = [r for r in season['reported_totals'] if r['date'].startswith('2026-11')]
            self.assertEqual([(r['date'], r['total_inches'], r['end']) for r in nov], [
                ('2026-11-24', 56, None), ('2026-11-26', 62, None), ('2026-11-30', 71, None)])

    def test_incomplete_current_month_is_not_redated_daily(self):
        seasons = obs.parse_main_lodge(resort())
        as_of = datetime(2026, 12, 1).date()
        self.assertEqual(obs._snapshots(2026, seasons['2026'], as_of)[-1]['date'], '2026-11-30')


if __name__ == '__main__':
    unittest.main()
