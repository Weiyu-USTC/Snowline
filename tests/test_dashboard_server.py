"""Local HTTP boundaries and one all-mountain refresh, without downloads."""
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import serve_snow_dashboard as dashboard

class DashboardServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd=ThreadingHTTPServer(('127.0.0.1',0),dashboard.Handler)
        cls.thread=threading.Thread(target=cls.httpd.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown();cls.httpd.server_close();cls.thread.join(timeout=2)
    def setUp(self):
        with dashboard.LOCK:dashboard.JOBS.clear()
    def request(self,method,path,body=None,headers=None):
        conn=HTTPConnection('127.0.0.1',self.httpd.server_port,timeout=3)
        if body is not None and not isinstance(body,str):body=json.dumps(body)
        conn.request(method,path,body=body,headers=headers or {})
        response=conn.getresponse();data=response.read();status=response.status;conn.close()
        return status,json.loads(data)
    def test_health_is_local(self):
        status,data=self.request('GET','/api/health')
        self.assertEqual(status,200);self.assertTrue(data['local_only'])
    def test_cross_origin_and_rebinding_host_rejected(self):
        for headers in ({'Origin':'https://example.com'},{'Host':'attacker.example'}):
            self.assertEqual(self.request('GET','/api/health',headers=headers)[0],403)
    def test_unknown_site_and_path_traversal_rejected(self):
        for path,status in [('/api/forecast?site=unsupported',400),('/reports/alta/../../../../etc/passwd',404),('/reports/alta/%2e%2e/%2e%2e/README.md',404),('/src/serve_snow_dashboard.py',404)]:
            self.assertEqual(self.request('GET',path)[0],status)
    def test_refresh_shape_and_media_type_rejected(self):
        cases=[('/api/refresh',{'site':'alta'}, {},415),('/api/refresh',{'site':'unknown'},{'Content-Type':'application/json'},400),('/api/refresh',{'site':'alta','include_radar':'yes'},{'Content-Type':'application/json'},400),('/api/refresh',{'site':'alta','command':'bad'},{'Content-Type':'application/json'},400),('/api/refresh-all',[],{'Content-Type':'application/json'},400),('/api/refresh-all',{'command':'bad'},{'Content-Type':'application/json'},400)]
        with patch.object(dashboard,'WORKER') as worker:
            for path,body,headers,status in cases:self.assertEqual(self.request('POST',path,body,headers)[0],status)
            worker.submit.assert_not_called()
    def test_all_button_and_legacy_four_posts_join_one_job(self):
        with patch.object(dashboard,'WORKER') as worker:
            status,first=self.request('POST','/api/refresh-all',{}, {'Content-Type':'application/json'})
            self.assertEqual(status,202);self.assertEqual(first['site'],'all');self.assertTrue(first['include_radar'])
            self.assertEqual(set(first['sites']),dashboard.SITES)
            for site in dashboard.SITES:self.assertEqual(first['job_id'],dashboard.start_refresh(site,False)['job_id'])
            self.assertEqual(worker.submit.call_count,1)
            status,payload=self.request('GET','/api/jobs/'+first['job_id'])
            self.assertEqual(status,200);self.assertEqual(payload['status'],'queued')
    def test_partial_source_failure_keeps_four_site_results(self):
        outcome={'status':'partial','sources':{'hrrr':{'status':'failed'},'nexrad_models':{'status':'succeeded'}},'errors':[{'source':'hrrr','message':'retained dated forecast'}]}
        module=SimpleNamespace(refresh_all=Mock(return_value=outcome))
        read=lambda site:{'site':{'id':site},'status':'ready','deployed':{'status':'ready'}}
        with patch.dict(sys.modules,{'refresh_deployed_forecasts':module}),patch.object(dashboard,'WORKER'),patch.object(dashboard,'read_forecast',side_effect=read):
            job=dashboard.start_refresh_all();dashboard.run_refresh_all(job['job_id']);result=dashboard.JOBS[job['job_id']]
            self.assertEqual(result['status'],'partial');self.assertEqual(result['completed_sites'],4)
            self.assertEqual(set(result['results']),dashboard.SITES)
            self.assertFalse(result['retrained']);module.refresh_all.assert_called_once()
    def test_failed_batch_does_not_destroy_readable_cache(self):
        cached={'site':{'id':'vail'},'status':'cached','hourly':[{'time':'2026-11-06T16:00Z'}]}
        module=SimpleNamespace(refresh_all=Mock(side_effect=OSError('source unavailable')))
        with patch.dict(sys.modules,{'refresh_deployed_forecasts':module}),patch.object(dashboard,'WORKER'),patch.object(dashboard,'read_forecast',return_value=cached):
            job=dashboard.start_refresh_all();dashboard.run_refresh_all(job['job_id'])
            self.assertEqual(dashboard.JOBS[job['job_id']]['status'],'failed');self.assertEqual(dashboard.read_forecast('vail'),cached)
    def test_terminal_job_allows_a_new_explicit_refresh(self):
        with patch.object(dashboard,'WORKER'):
            first=dashboard.start_refresh_all();dashboard.set_job(first['job_id'],status='partial')
            self.assertNotEqual(first['job_id'],dashboard.start_refresh_all()['job_id'])

if __name__=='__main__':unittest.main()
