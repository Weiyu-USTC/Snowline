"""Failure isolation and immutable source timestamps in deployed refresh."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import refresh_deployed_forecasts as refresh

class RefreshTests(unittest.TestCase):
    def test_missing_portfolio_component_is_reported_as_partial(self):
        from types import SimpleNamespace
        payload={'status':'partial','warnings':['Vail 24h: HRRR window incomplete.'],
                 'generated_at':'now','start':'start','end':'end',
                 'alta12':{'available':False,'reason':'Scheduled cycle not available.'}}
        with patch.dict(sys.modules,{'forecast_portfolio':SimpleNamespace(build=lambda:payload),
                                     'report_forecast_portfolio':SimpleNamespace(build=lambda:None)}):
            result=refresh.portfolio_stage()
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['errors'],payload['warnings'])

    def test_structured_failure_survives_nonzero_stage_exit(self):
        from unittest.mock import Mock
        expected='No complete HRRR run available; previous snapshot retained.'
        def launch(command,**kwargs):
            Path(command[-1]).write_text(json.dumps({'status':'failed','message':expected,'errors':[expected]}))
            kwargs['stdout'].write('{\n  "status": "failed"\n}\n')
            process=Mock();process.wait.return_value=1
            return process
        with tempfile.TemporaryDirectory() as tmp,patch.object(refresh.subprocess,'Popen',side_effect=launch):
            result=refresh.run_stage('hrrr',Path(tmp))
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['message'],expected)
        self.assertEqual(result['errors'],[expected])

    def test_crashed_stage_cannot_report_success(self):
        from unittest.mock import Mock
        def launch(command,**kwargs):
            Path(command[-1]).write_text(json.dumps({'status':'succeeded','message':'ready'}))
            process=Mock();process.wait.return_value=1
            return process
        with tempfile.TemporaryDirectory() as tmp,patch.object(refresh.subprocess,'Popen',side_effect=launch):
            result=refresh.run_stage('hrrr',Path(tmp))
        self.assertEqual(result['status'],'failed')
        self.assertIn('exited unexpectedly',result['message'])

    def test_failed_new_radar_retains_original_per_site_issue(self):
        old={'sites':{'alta':{'available':True,'issued_at':'2026-11-11T20:00Z','start':'2026-11-11T21:00Z','hourly':[{'liquid_mm':1.}]}}}
        new={'issued_at':'2026-11-11T22:00Z','sites':{'alta':{'available':False,'reason':'missing scan'},'vail':{'available':False,'reason':'coverage'}}}
        result,retained=refresh.merge_radar(old,new)
        self.assertEqual(retained,['alta']);self.assertEqual(result['sites']['alta']['issued_at'],'2026-11-11T20:00Z')
        self.assertEqual(result['sites']['alta']['hourly'][0]['liquid_mm'],1.)
        self.assertEqual(result['sites']['alta']['refresh_error'],'missing scan')
    def test_complete_new_radar_replaces_old_without_retained_flag(self):
        result,retained=refresh.merge_radar({'sites':{'alta':{'available':True,'issued_at':'old'}}},{'sites':{'alta':{'available':True,'issued_at':'new'}}})
        self.assertFalse(retained);self.assertEqual(result['sites']['alta']['issued_at'],'new')
    def test_one_failed_source_does_not_skip_other_sources(self):
        names=[]
        def fake_stage(name,folder,progress=None,deadline=None):
            names.append(name);return {'source':name,'status':'failed' if name=='hrrr' else 'succeeded','message':'source check'}
        with tempfile.TemporaryDirectory() as tmp,patch.object(refresh,'OUT',Path(tmp)),patch.object(refresh,'LATEST',Path(tmp)/'latest.json'),patch.object(refresh,'run_stage',side_effect=fake_stage):
            result=refresh.refresh_all()
        self.assertEqual(result['status'],'partial');self.assertEqual(set(names),set(refresh.STAGES))
        self.assertEqual(names.count('hrrr'),1);self.assertFalse(result['retrained']);self.assertEqual(result['sites'],list(refresh.SITES))
    def test_expired_budget_starts_no_process(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(refresh.subprocess,'Popen') as launch:
            result=refresh.run_stage('hrrr',Path(tmp),deadline=0)
        launch.assert_not_called();self.assertEqual(result['status'],'failed')
    def test_atomic_output_failure_preserves_old_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.json';p.write_text('{"issued_at":"old"}')
            with self.assertRaises(ValueError):refresh.atomic_json(p,{'amount':float('nan')})
            self.assertEqual(json.loads(p.read_text())['issued_at'],'old')
    def test_stage_entrypoints_contain_inference_not_optimizer_calls(self):
        import inspect
        source=inspect.getsource(refresh.neural48_stage)+inspect.getsource(refresh.nexrad_models_stage)+inspect.getsource(refresh.temperature_stage)
        self.assertNotIn('.fit(',source);self.assertNotIn('train()',source)
    def test_jay48_override_does_not_change_legacy24_portfolio_input(self):
        import numpy as np
        import forecast_neural48 as neural
        row={'site_id':'jay','status':'experimental','array_row':0,'radar_available':True,
             'issued_at':'2026-11-11T21:30:00Z','start':'2026-11-11T22:00:00Z',
             'midpoint':'2026-11-12T22:00:00Z','end':'2026-11-13T22:00:00Z',
             'provider_daily_inches':[0.,0.],'conservative_provider_daily_inches':[0.,0.]}
        bundle={'selection':{'frozen_utc':'2026-11-11T20:00Z','selected_family_by_site':{'jay':'weather_only'}}}
        def predict(bundle,arrays,family=None,n_paths=768):
            means=np.array([[10.,20.]]) if family=='weather_radar' else np.array([[1.,2.]])
            return {'daily_mean':means,'joint_paths':np.repeat(means[:,None,:],768,axis=1),'methods':[family or 'legacy']}
        with tempfile.TemporaryDirectory() as tmp,patch.object(neural,'ROOT',Path(tmp).resolve()),patch.object(neural,'OUT',Path(tmp)),patch.object(neural,'plot_current'),patch.object(neural,'get_cached_forecasts',return_value=None),patch('train_neural48.load_bundle',return_value=bundle),patch('train_neural48.predict_neural',side_effect=predict):
            result=neural.issue_from_prepared({'site':np.array([3])},[row],tmp,family_overrides={'jay':'weather_radar'})
        jay=result['sites']['jay']
        self.assertEqual(jay['totals'][0]['mean_inches'],1.)
        self.assertEqual(jay['totals'][1]['mean_inches'],3.)
        self.assertEqual(jay['family'],'weather_only')
        self.assertEqual(jay['deployed_48']['mean_inches'],30.)
        self.assertEqual(jay['deployed_48']['family'],'weather_radar')

class CollinsFeedTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        import prepare_mountain_hourly as mountain

        self.mountain = mountain
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.cache = self.raw / 'alta_cln_current.html'
        self.record_time = pd.Timestamp.now(tz='UTC').floor('h') - pd.Timedelta(hours=1)
        self.original = self.station_table(self.record_time - pd.Timedelta(hours=2)).encode()
        self.cache.write_bytes(self.original)
        self.enterContext(patch.object(mountain, 'ROOT', self.root))
        self.enterContext(patch.object(mountain, 'RAW', self.raw))
        self.primary = self.enterContext(patch.object(mountain.requests, 'get'))
        self.fallback = self.enterContext(patch('alta_collins_fallback.fetch_collins_fallback'))
        self.fallback.return_value = pd.DataFrame([{
            'station_id': 'CLN', 'valid_time_utc': self.record_time, 'temperature_c': -3.0,
        }])

    @staticmethod
    def station_table(time):
        stamp = time.strftime('%m/%d/%Y %H:%M')
        return (f'<a href="?stn=CLN">Collins</a><p>Tabular Listing of 1 Observations '
                f'from {stamp} GMT to {stamp} GMT</p><table>'
                '<tr><th>Time (GMT)</th><th>Temperature ° F</th></tr>'
                f'<tr><td>{time:%H:%M}</td><td>14</td></tr></table>')

    def set_response_html(self, html):
        self.primary.return_value.text = html
        self.primary.return_value.content = html.encode()

    def test_primary_route_failure_uses_alta_and_keeps_cached_station_table(self):
        self.primary.side_effect = self.mountain.requests.ConnectionError('No route to host')
        result = self.mountain.fetch_current_collins(timeout=2)
        self.assertEqual(result.attrs['feed'], 'Alta Ski Area')
        self.assertIn('No route to host', result.attrs['primary_error'])
        self.assertEqual(result.valid_time_utc.iloc[0], self.record_time)
        self.assertEqual(self.cache.read_bytes(), self.original)

    def test_unusable_or_stale_primary_table_is_rejected_before_cache_replacement(self):
        from datetime import timedelta
        for body in ('<p>stn=CLN: maintenance</p>', self.station_table(self.record_time - timedelta(days=2))):
            with self.subTest(body=body):
                self.set_response_html(body)
                result = self.mountain.fetch_current_collins(timeout=2)
                self.assertEqual(result.attrs['feed'], 'Alta Ski Area')
                self.assertTrue(result.attrs['primary_error'])
                self.assertEqual(self.cache.read_bytes(), self.original)
                self.assertEqual(list(self.raw.iterdir()), [self.cache])

    def test_both_feeds_fail_without_promoting_stale_data(self):
        self.primary.side_effect = self.mountain.requests.ConnectionError('Primary unavailable')
        self.fallback.side_effect = ValueError('Secondary unavailable')
        with self.assertRaises(ValueError) as failure:
            self.mountain.fetch_current_collins(timeout=2)
        self.assertIn('Primary unavailable', str(failure.exception))
        self.assertIn('Secondary unavailable', str(failure.exception))
        self.assertEqual(self.cache.read_bytes(), self.original)

    def test_valid_primary_refresh_preserves_source_time_and_avoids_fallback(self):
        body = self.station_table(self.record_time)
        self.set_response_html(body)
        result = self.mountain.fetch_current_collins(timeout=2)
        self.assertEqual(result.attrs['feed'], 'MesoWest')
        self.assertEqual(result.valid_time_utc.iloc[0], self.record_time)
        self.assertEqual(result.temperature_c.iloc[0], -10)
        self.assertEqual((self.root / result.source_file.iloc[0]).read_bytes(), body.encode())
        self.fallback.assert_not_called()


if __name__=='__main__':unittest.main()
