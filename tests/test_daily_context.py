"""Independent checks for the richer monthly-context comparison."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import train_monthly_daily_context as model
from monthly import load_training,MONTHS


class DailyContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        daily=pd.read_csv(ROOT/'data/dillon_season_daily_features.csv')
        cls.frame=load_training().merge(daily.drop(columns=['season']),on='start_year')

    def test_future_feature_missingness_cannot_change_october_cohort(self):
        original=self.frame.dropna(subset=model.features(0,'daily_context')).start_year.tolist()
        changed=self.frame.copy()
        future=[c for c in changed if c.startswith(('nov_','dec_','jan_','feb_','mar_','apr_'))]
        changed[future]=np.nan
        actual=changed.dropna(subset=model.features(0,'daily_context')).start_year.tolist()
        self.assertEqual(actual,original)
        self.assertEqual(len(actual),18)

    def test_future_query_values_cannot_change_initial_prediction(self):
        frame=self.frame.dropna(subset=model.features(0,'daily_context')).reset_index(drop=True)
        train,query=frame.iloc[:-1],frame.iloc[-1:].copy()
        changed=query.copy()
        for c in changed:
            if c in MONTHS[1:] or c.startswith(('nov_','dec_','jan_','feb_','mar_','apr_')):
                changed[c]=100000.
        for config in [{'family':'ridge','inputs':'daily_context','alpha':100.},
                       {'family':'analog','inputs':'daily_context','bandwidth':1.},
                       {'family':'extra_trees','inputs':'daily_context','min_leaf':3}]:
            np.testing.assert_allclose(model.fit_predict(train,query,0,config),
                                       model.fit_predict(train,changed,0,config),rtol=0,atol=0)

    def test_saved_baselines_exclude_the_test_season_on_matched_cohorts(self):
        table=pd.read_csv(ROOT/'outputs/monthly_daily_context_predictions.csv')
        for origin,group in table.loc[table.comparison=='climatology'].groupby('origin'):
            cohort=self.frame.dropna(subset=model.features(int(origin),'daily_context'))
            for row in group.itertuples():
                train=cohort.loc[cohort.start_year!=row.start_year]
                self.assertAlmostEqual(row.prediction,train[row.target_month].mean(),places=10)
                self.assertEqual(row.n_train,len(train))


if __name__=='__main__':unittest.main()
