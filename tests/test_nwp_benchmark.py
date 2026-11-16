import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import nwp_benchmark as b


def fixture():
    rows=[]
    for year in [2003,2004,2005,2006,2007,2008]:
        for j in range(18):
            issue=pd.Timestamp(year,11,1,tz='UTC')+pd.Timedelta(days=j,hours=6)
            end=issue.normalize()+pd.Timedelta(days=2,hours=11)
            rows.append(dict(site='alta',issue_time_utc=issue,valid_start_utc=end-pd.Timedelta(days=1),valid_end_utc=end,target_date=end.tz_localize(None).normalize(),lead_day=1,target_inches=float((j+year)%9),archive_type='operational_forecast',forecast_temp_c=float(j-9),forecast_qpf_category=j%7,forecast_wind_u_ms=j/2,roni=(year-2005)/4))
    return pd.DataFrame(rows)


class TimingTests(unittest.TestCase):
    def test_timing_and_identity(self):
        d=b.validate_join(fixture());self.assertEqual(d.winter.nunique(),6);self.assertTrue(d.row_id.is_unique)
        bad=fixture();bad.loc[0,'valid_start_utc']=bad.loc[0,'issue_time_utc']-pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError,'before issue'):b.validate_join(bad)
    def test_reanalysis_rejected(self):
        d=fixture();d.loc[0,'archive_type']='reanalysis'
        with self.assertRaisesRegex(ValueError,'reanalysis'):b.validate_join(d)
    def test_observed_future_rejected(self):
        d=fixture();d['observed_temp']=4
        with self.assertRaisesRegex(ValueError,'history_end'):b.validate_join(d)
        d['history_end_utc']=pd.to_datetime(d.issue_time_utc)+pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError,'after issuance'):b.validate_join(d)
    def test_feature_ablation(self):
        frame=fixture();frame['forecast_temp_qpf_cold_interaction']=1
        groups=b.feature_groups(frame)
        self.assertEqual(groups[b.NAMES[1]],['roni'])
        self.assertIn('forecast_qpf_category',groups[b.NAMES[2]])
        self.assertNotIn('forecast_temp_c',groups[b.NAMES[2]])
        self.assertIn('forecast_temp_c',groups[b.NAMES[3]])
        self.assertNotIn('forecast_temp_qpf_cold_interaction',groups[b.NAMES[2]])
        self.assertIn('forecast_temp_qpf_cold_interaction',groups[b.NAMES[3]])
        self.assertNotIn('forecast_wind_u_ms',groups[b.NAMES[3]])
        self.assertIn('forecast_wind_u_ms',groups[b.NAMES[4]])


class DistributionTests(unittest.TestCase):
    def test_crps_matches_pairwise_formula(self):
        v=np.array([0.,1.,5.,12.]);w=np.array([[.2,.4,.3,.1],[.7,.1,.1,.1]]);y=np.array([3.,0.])
        p=b.weighted_distribution(v,w,y)
        manual=np.sum(w*abs(v[None]-y[:,None]),1)-.5*np.einsum('bi,bj,ij->b',w,w,abs(v[:,None]-v[None]))
        np.testing.assert_allclose(p['crps'],manual);np.testing.assert_allclose(p['p_powder'],[.4,.2]);np.testing.assert_allclose(p['mean'],w@v)
    def test_train_only_scaler_donors_and_bounds(self):
        x=np.array([[0,np.nan],[1,2],[2,3],[3,4],[4,5],[5,6]],float);y=np.array([0,0,1,5,8,12],float)
        model=b.EmpiricalForest({'n_estimators':8,'min_samples_leaf':2}).fit(x,y);median=model.median.copy()
        p=model.predict_distribution(np.array([[999,np.nan],[-999,1.]]),np.array([200,200]))
        np.testing.assert_equal(model.median,median);self.assertTrue(np.all(p['mean']<=12));self.assertTrue(np.all(p['mean']>=0));self.assertTrue(np.all((p['p_powder']>=0)&(p['p_powder']<=1)))
    def test_wholewinter_matched_scoring(self):
        d=fixture()
        with tempfile.TemporaryDirectory() as directory:
            cv,chrono=b.evaluate(d,directory,config={'n_estimators':8,'min_samples_leaf':3})
            plan=json.loads((Path(directory)/'benchmark_plan.json').read_text())
            for split in plan['splits']:self.assertFalse(set(split['train_winters'])&set(split['test_winters']))
            for p in [cv,chrono]:
                cohorts=[set(r.row_id) for _,r in p.groupby('model')];self.assertTrue(all(c==cohorts[0] for c in cohorts))
            paired=pd.read_csv(Path(directory)/'crossvalidation_incremental.csv');self.assertEqual(len(paired),9)
            scores=b.score_predictions(cv);self.assertTrue((scores.powder_logloss>=0).all())

if __name__=='__main__':unittest.main()
