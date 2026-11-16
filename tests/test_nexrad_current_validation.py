"""A rejected radar cache cannot re-enter the live forecast through inference."""
import copy,json,pickle,sys,tempfile,unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import nexrad_train as radar


class CurrentRadarValidationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name);self.data=self.root/'data';self.out=self.root/'out'
  self.data.mkdir();self.out.mkdir()
  self.case={'case_id':'current_20261115T0110','forecast_issue_utc':'2026-11-15T01:10:00+00:00','split':'live'}
  (self.data/'current_case.json').write_text(json.dumps(self.case))
  choice={'model':'DPR_persistence','validation_score':1.,'train_mean':[0.,0.,0.],
          'validation_absolute_error_p80_hourly_mm':[.1,.1,.1]}
  (self.out/'selection.json').write_text(json.dumps({'alta':choice,'mansfield':choice}))
  (self.out/'calibrators.pkl').write_bytes(pickle.dumps({}))
  self.save('alta');self.save('mansfield')

 def save(self,key,case=None,units='mm/hour',target=None):
  issue=radar.utc(self.case['forecast_issue_utc']);site=radar.SITES[key]
  metas=[]
  for lag in [70,40,10]:
   stamp=issue-timedelta(minutes=lag+1)
   metas.append({'key':site['radar']+'_DPR_'+stamp.strftime('%Y_%m_%d_%H_%M_%S'),
    'volume_start_utc':radar.iso(stamp),'product_generated_utc':radar.iso(stamp+timedelta(minutes=5)),
    'units':units,'original_units':'inches/hour','distance_km':20.,'beam_above_target_m':200.,'valid_grid_fraction':1.})
  path=self.data/'crops'/key/(self.case['case_id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(path,rate_mm_h=np.ones((3,129,129),np.float32),
    metadata_json=np.array(json.dumps(metas)),case_json=np.array(json.dumps(case or self.case)),
    site_json=np.array(json.dumps(target or site)),pixel_km=np.array(2.))
  return path

 def run_current(self):
  with patch.object(radar,'DATA',self.data),patch.object(radar,'OUT',self.out):radar.current()
  return json.loads((self.out/'current_forecast.json').read_text())

 def test_valid_crops_preserve_original_issue_and_predictions(self):
  result=self.run_current()
  for site in ['alta','mansfield']:
   row=result['sites'][site]
   self.assertTrue(row['available']);self.assertEqual(row['issued_at'],self.case['forecast_issue_utc'])
   self.assertEqual(row['start'],'2026-11-15T02:00:00+00:00')
   self.assertEqual([h['liquid_mm'] for h in row['hourly']],[1.,1.,1.])

 def test_wrong_issue_cache_is_unavailable_without_stopping_other_targets(self):
  old=dict(self.case,forecast_issue_utc='2026-11-14T23:50:00+00:00')
  path=self.save('alta',case=old);original=path.read_bytes()
  result=self.run_current()
  self.assertFalse(result['sites']['alta']['available'])
  self.assertIn('different forecast issue',result['sites']['alta']['reason'])
  self.assertTrue(result['sites']['mansfield']['available'])
  self.assertEqual(path.read_bytes(),original)

 def test_wrong_target_units_and_corrupt_cache_cannot_feed_inference(self):
  for kind in ['target','units','corrupt']:
   with self.subTest(kind=kind):
    if kind=='target':
     target=copy.deepcopy(radar.SITES['alta']);target['latitude']+=1
     self.save('alta',target=target)
    elif kind=='units':self.save('alta',units='inches/hour')
    else:self.save('alta').write_bytes(b'incomplete radar crop')
    result=self.run_current()
    self.assertFalse(result['sites']['alta']['available'])
    self.assertIn('Invalid NEXRAD scan sequence',result['sites']['alta']['reason'])
    self.assertTrue(result['sites']['mansfield']['available'])


if __name__=='__main__':unittest.main()
