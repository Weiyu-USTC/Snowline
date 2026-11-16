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

if __name__=='__main__':unittest.main()
