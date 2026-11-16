"""A successful cached radar acquisition must not become a refresh failure."""
import copy,json,os,sys,tempfile,unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import nexrad_acquire as radar


class RadarCacheTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.data=Path(self.temp.name)
  self.case={'case_id':'current_20261114T2200','forecast_issue_utc':'2026-11-14T22:00:00+00:00','split':'live'}
  self.site='alta';self.path=self.data/'crops'/self.site/(self.case['case_id']+'.npz')
  self.path.parent.mkdir(parents=True)
  issue=radar.utc(self.case['forecast_issue_utc'])
  self.metas=[]
  for lag in [70,40,10]:
   stamp=issue-timedelta(minutes=lag+1)
   self.metas.append({'key':'MTX_DPR_'+stamp.strftime('%Y_%m_%d_%H_%M_%S'),
    'volume_start_utc':radar.iso(stamp),'product_generated_utc':radar.iso(stamp+timedelta(minutes=5)),
    'units':'mm/hour','original_units':'inches/hour'})

 def save(self,case=None,site=None,metas=None,rates=None):
  np.savez_compressed(self.path,rate_mm_h=np.zeros((3,129,129)) if rates is None else rates,
    metadata_json=np.array(json.dumps(self.metas if metas is None else metas)),
    case_json=np.array(json.dumps(self.case if case is None else case)),
    site_json=np.array(json.dumps(radar.SITES[self.site] if site is None else site)),pixel_km=np.array(2.))

 def test_valid_cached_case_is_success_without_network_or_new_issue(self):
  self.save();original=self.path.read_bytes()
  with patch.object(radar,'DATA',self.data),patch.object(radar,'inventory') as inventory:
   result=radar.acquire_case(self.case,self.site)
  self.assertEqual(result['status'],'ok');self.assertTrue(result['cached'])
  self.assertEqual(result['forecast_issue_utc'],self.case['forecast_issue_utc'])
  self.assertEqual(self.path.read_bytes(),original);inventory.assert_not_called()

 def test_wrong_issue_site_units_shape_and_scan_times_are_rejected(self):
  mutations=[]
  c=dict(self.case,forecast_issue_utc='2026-11-14T21:50:00+00:00');mutations.append({'case':c})
  s=dict(radar.SITES['alta'],latitude=41);mutations.append({'site':s})
  m=copy.deepcopy(self.metas);m[-1]['units']='inches/hour';mutations.append({'metas':m})
  m=copy.deepcopy(self.metas);m[-1]['volume_start_utc']='2026-11-14T20:00:00+00:00';mutations.append({'metas':m})
  m=copy.deepcopy(self.metas);m[-1]['product_generated_utc']='2026-11-14T22:01:00+00:00';mutations.append({'metas':m})
  mutations.append({'rates':np.zeros((2,129,129))})
  mutations.append({'rates':np.full((3,129,129),np.nan)})
  for value in mutations:
   with self.subTest(value=list(value)):
    self.save(**value)
    with self.assertRaises(ValueError):radar.validate_crop(self.path,self.case,self.site)

 def test_bad_cache_retry_failure_preserves_existing_bytes_and_reports_error(self):
  self.path.write_bytes(b'broken radar cache')
  with patch.object(radar,'DATA',self.data),patch.object(radar,'inventory',side_effect=RuntimeError('Radar source offline')) as inventory:
   result=radar.acquire_case(self.case,self.site)
  self.assertEqual(result['status'],'error');self.assertIn('Radar source offline',result['error'])
  self.assertIn('existing cache rejected',result['error']);self.assertEqual(self.path.read_bytes(),b'broken radar cache')
  inventory.assert_called_once()

 def test_bad_cache_is_replaced_only_with_valid_reacquired_crop(self):
  self.path.write_bytes(b'broken radar cache')
  entries=[{'key':m['key'],'volume_start_utc':m['volume_start_utc']} for m in self.metas]
  decoded={m['key']:(np.zeros((129,129),np.float32),m) for m in self.metas}
  with patch.object(radar,'DATA',self.data),patch.object(radar,'inventory',return_value=entries),\
       patch.object(radar,'download',side_effect=lambda key:Path(key)),\
       patch.object(radar,'decode_crop',side_effect=lambda path,site:decoded[path.name]):
   first=radar.acquire_case(self.case,self.site)
   second=radar.acquire_case(self.case,self.site)
  self.assertEqual(first['status'],'ok');self.assertFalse(first['cached'])
  self.assertEqual(second['status'],'ok');self.assertTrue(second['cached'])
  self.assertFalse(list(self.path.parent.glob('*.tmp')))

 def test_acquisition_uses_causal_utc_days_at_midnight_and_year_rollover(self):
  for stamp in ['2026-11-15T00:00:00+00:00','2026-11-15T01:10:00+00:00',
                '2027-01-01T00:00:00+00:00','2027-01-01T01:10:00+00:00']:
   with self.subTest(issue=stamp):
    issue=radar.utc(stamp);case={'case_id':'current_'+issue.strftime('%Y%m%dT%H%M'),'forecast_issue_utc':stamp,'split':'live'}
    metas=[]
    for lag in radar.CONTEXT_LAGS_MINUTES:
     scan=issue-timedelta(minutes=lag+1)
     metas.append({'key':'MTX_DPR_'+scan.strftime('%Y_%m_%d_%H_%M_%S'),
       'volume_start_utc':radar.iso(scan),'product_generated_utc':radar.iso(scan+timedelta(minutes=5)),
       'units':'mm/hour','original_units':'inches/hour'})
    decoded={m['key']:(np.zeros((129,129),np.float32),m) for m in metas}
    def daily_inventory(site,date):
     return [{'key':m['key'],'volume_start_utc':m['volume_start_utc']} for m in metas
             if radar.utc(m['volume_start_utc']).date()==date.date()]
    with patch.object(radar,'DATA',self.data),patch.object(radar,'inventory',side_effect=daily_inventory) as inventory,\
         patch.object(radar,'download',side_effect=lambda key:Path(key)),\
         patch.object(radar,'decode_crop',side_effect=lambda path,site:decoded[path.name]):
     result=radar.acquire_case(case,'alta')
    self.assertEqual(result['status'],'ok',result)
    called=[call.args[1].date() for call in inventory.call_args_list]
    previous=(issue-timedelta(days=1)).date()
    self.assertEqual(called,[previous] if issue.hour==0 else [previous,issue.date()])
    self.assertEqual(result['forecast_issue_utc'],stamp)


class InventoryCompletionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.data=Path(self.temp.name);(self.data/'inventories').mkdir()
  self.now=radar.utc('2026-11-15T01:15:00+00:00')
  self.day=radar.utc('2026-11-14T00:00:00+00:00')
  self.path=self.data/'inventories/MTX_DPR_2026_11_14_.json'
  self.old={'key':'MTX_DPR_2026_11_14_22_14_39','volume_start_utc':'2026-11-14T22:14:39+00:00','bytes':20}
  self.new={'key':'MTX_DPR_2026_11_14_23_56_00','volume_start_utc':'2026-11-14T23:56:00+00:00','bytes':30}
  self.response=SimpleNamespace(content=(
    '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
    '<Contents><Key>'+self.new['key']+'</Key><Size>30</Size></Contents></ListBucketResult>').encode())

 def save(self,retrieved,entries=None):
  self.path.write_text(json.dumps({'schema_version':2,'prefix':'MTX_DPR_2026_11_14_',
   'retrieved_at':retrieved,'entries':[self.old] if entries is None else entries}))

 def invoke(self,now=None):
  with patch.object(radar,'DATA',self.data),patch.object(radar,'datetime',wraps=datetime) as clock,\
       patch.object(radar,'get',return_value=self.response) as get:
   clock.now.return_value=now or self.now
   result=radar.inventory('MTX',self.day)
  return result,get

 def test_legacy_partial_day_list_is_refetched_even_with_recent_mtime(self):
  self.path.write_text(json.dumps([self.old]))
  os.utime(self.path,(self.now.timestamp(),self.now.timestamp()))
  result,get=self.invoke()
  self.assertEqual(result,[self.new]);get.assert_called_once()
  saved=json.loads(self.path.read_text())
  self.assertEqual(saved['retrieved_at'],radar.iso(self.now));self.assertEqual(saved['schema_version'],2)
  self.assertEqual(saved['entries'],[self.new]);self.assertFalse(list(self.path.parent.glob('*.tmp')))

 def test_live_listing_expires_across_midnight_despite_touched_mtime(self):
  self.save('2026-11-14T22:26:34+00:00')
  os.utime(self.path,(self.now.timestamp(),self.now.timestamp()))
  result,get=self.invoke()
  self.assertEqual(result,[self.new]);get.assert_called_once()

 def test_previous_day_live_listing_reused_only_inside_ttl(self):
  self.save('2026-11-15T01:14:00+00:00')
  result,get=self.invoke();self.assertEqual(result,[self.old]);get.assert_not_called()
  result,get=self.invoke(now=self.now+timedelta(minutes=5))
  self.assertEqual(result,[self.new]);get.assert_called_once()

 def test_expired_live_listing_cannot_become_settled_just_by_waiting(self):
  self.save('2026-11-14T22:26:34+00:00')
  result,get=self.invoke(now=self.now+timedelta(days=30))
  self.assertEqual(result,[self.new]);get.assert_called_once()

 def test_settled_day_reuses_original_receipt_without_network(self):
  self.save('2026-11-16T00:01:00+00:00')
  before=self.path.read_bytes()
  result,get=self.invoke(now=radar.utc('2026-12-01T00:00:00+00:00'))
  self.assertEqual(result,[self.old]);get.assert_not_called();self.assertEqual(before,self.path.read_bytes())

 def test_future_receipt_is_not_reused(self):
  self.save('2026-11-16T00:01:00+00:00')
  result,get=self.invoke();self.assertEqual(result,[self.new]);get.assert_called_once()


if __name__=='__main__':unittest.main()
