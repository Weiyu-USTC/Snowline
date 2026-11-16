"""Safety-critical alignment/unit tests for native DPR nowcasting."""
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.ndimage import shift
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nexrad_acquire import geometry,decode_crop,SITES,DATA
from nexrad_train import motion,crop_features

class NexradTests(unittest.TestCase):
 def test_motion_direction(self):
  yy,xx=np.mgrid[:129,:129];old=5*np.exp(-((yy-60)**2+(xx-55)**2)/100)
  new=shift(old,(3,7),order=1,mode='constant');v,c=motion(old,new,.5)
  self.assertLess(np.linalg.norm(v-np.array([6,14])),3);self.assertGreater(c,.5)
 def test_spherical_geometry(self):
  d,a=geometry(40,-110,40,-110);self.assertAlmostEqual(float(d),0)
  d,a=geometry(40,-110,41,-110);self.assertAlmostEqual(float(a),0);self.assertAlmostEqual(float(d),111.195,places=2)
 def test_decoded_units_and_geometry(self):
  p=DATA/'MTX_DPR_2023_01_01_10_01_15'
  if not p.exists():self.skipTest('Public fixture not downloaded')
  g,m=decode_crop(p,SITES['alta']);self.assertEqual(g.shape,(129,129));self.assertEqual(m['units'],'mm/hour');self.assertAlmostEqual(m['distance_km'],102.254,places=2);self.assertGreater(float(np.nanmax(g)),1)
 def create(self,p,product_time):
  case=dict(forecast_issue_utc='2024-01-01T10:50:00+00:00')
  metas=[dict(volume_start_utc=f'2024-01-01T{t}:00+00:00',product_generated_utc=product_time) for t in ['09:40','10:10','10:40']]
  np.savez(p,rate_mm_h=np.zeros((3,129,129)),metadata_json=np.array(json.dumps(metas)),case_json=np.array(json.dumps(case)),site_json=np.array('{}'))
 def test_dry_is_zero_and_clock_hours(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'a.npz';self.create(p,'2024-01-01T10:45:00+00:00');x,b,a,m=crop_features(p)
   np.testing.assert_array_equal(a,np.zeros(3));np.testing.assert_array_equal(b,np.zeros(3));self.assertEqual(m['forecast_start_utc'],'2024-01-01T11:00:00+00:00')
 def test_future_product_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'a.npz';self.create(p,'2024-01-01T10:49:00+00:00')
   with self.assertRaises(ValueError):crop_features(p)
if __name__=='__main__':unittest.main()
