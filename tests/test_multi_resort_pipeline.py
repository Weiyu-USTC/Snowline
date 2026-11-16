"""Independent information-boundary checks for the three-mountain experiment."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import train_multi_resort as pipeline
from vail_storm_statistical import EventModel


class PreparedSourceBoundaryTests(unittest.TestCase):
    """Read frozen prepared artifacts; never overwrite an active training run."""
    def test_target_exports_preserve_site_identity_and_noaa_quality_policy(self):
        for site in pipeline.SITES:
            with self.subTest(site=site):
                export=pipeline.OUT/site/'target_daily.csv'
                if not export.exists():self.skipTest('Prepared data required')
                actual=pd.read_csv(export,parse_dates=['date']).set_index('date').snowfall_inches
                if site=='alta':
                    source=pd.read_csv(ROOT/'data/east_alta_research/alta_hive/alta_collins_daily_paired.csv')
                    expected=pd.Series(source.snowfall_24h_inches.to_numpy(),index=pd.to_datetime(source.reporting_endpoint_lst).dt.normalize())
                    self.assertTrue((pd.to_datetime(source.reporting_endpoint_lst).dt.hour==4).all())
                else:
                    key='mount_mansfield' if site=='mansfield' else 'jay_peak'
                    source=pd.read_csv(ROOT/f'data/east_alta_research/derived_noaa/{key}_daily.csv',keep_default_na=False,low_memory=False)
                    source.index=pd.to_datetime(source.date)
                    expected=pd.to_numeric(source.snowfall_inches,errors='coerce').where(source.snowfall_inches_quality_flag.eq(''))
                    presumed=source.snowfall_inches_measurement_flag.eq('P')&source.snowfall_inches_quality_flag.eq('')
                    np.testing.assert_array_equal(actual.reindex(source.index[presumed]),expected[presumed])
                self.assertTrue(actual.index.equals(expected.sort_index().index))
                np.testing.assert_allclose(actual,expected.sort_index(),rtol=0,atol=0,equal_nan=True)

    def test_all_origins_have_source_aligned_weather_targets_and_unknown_initial_state(self):
        for site,cfg in pipeline.SITES.items():
            with self.subTest(site=site):
                dest=pipeline.OUT/site
                if not (dest/'training_arrays.npz').exists():
                    self.skipTest('Run --mode prepare for the three sites first')
                data=np.load(dest/'training_arrays.npz')
                meta=pd.read_csv(dest/'origins.csv',parse_dates=['origin'])
                target=pd.read_csv(dest/'target_daily.csv',parse_dates=['date']).set_index('date').snowfall_inches
                weather=pipeline.read_weather(pipeline.INPUT/cfg['weather'])
                self.assertEqual(data['x'].shape,(len(meta),60,3))
                self.assertTrue(np.isnan(data['initial'][:,0]).all())
                np.testing.assert_array_equal(data['initial'][:,1:],0.)
                # Reconstruct independently by timestamp, rather than reusing
                # the stride/view implementation being audited.
                for i,row in meta.iterrows():
                    end=row.origin-pd.Timedelta(days=2)
                    dates=pd.date_range(end-pd.Timedelta(days=59),end)
                    np.testing.assert_allclose(data['x'][i],weather.reindex(dates).to_numpy(),rtol=1e-6,atol=1e-7,equal_nan=True)
                    labels=pd.date_range(row.origin+pd.Timedelta(days=1),periods=60)
                    np.testing.assert_allclose(data['y'][i],target.reindex(labels),rtol=1e-6,atol=1e-7,equal_nan=True)
                    self.assertLess(dates.max(),labels.min())
                    self.assertGreaterEqual(dates.min(),pd.Timestamp(f'{row.winter}-07-01'))
                    self.assertLessEqual(labels.max(),pd.Timestamp(f'{row.winter+1}-04-30'))
                    np.testing.assert_allclose(data['f'][i],pipeline.future(row.origin),rtol=0,atol=0)

    def test_whole_winter_outer_and_inner_partitions_have_no_date_overlap(self):
        for site in pipeline.SITES:
            with self.subTest(site=site):
                dest=pipeline.OUT/site
                if not (dest/'plan.json').exists():self.skipTest('Prepared data required')
                years=np.asarray(json.loads((dest/'plan.json').read_text())['years'])
                meta=pd.read_csv(dest/'origins.csv',parse_dates=['origin'])
                for tr,te in KFold(3,shuffle=True,random_state=pipeline.SEED).split(years):
                    training=sorted(years[tr]);test=set(years[te])
                    inner_fit=set(training[:-2]);inner_valid=set(training[-2:])
                    self.assertFalse(inner_fit&inner_valid or inner_fit&test or inner_valid&test)
                    def label_dates(which):
                        return set().union(*(set(pd.date_range(o+pd.Timedelta(days=1),periods=60))
                                             for o in meta.loc[meta.winter.isin(which),'origin']))
                    self.assertFalse(label_dates(training)&label_dates(test))
                self.assertLess(max(years[:-3]),min(years[-3:]))

    def test_current_weather_does_not_invent_unreported_november_second(self):
        expected={'alta':'2026-11-01','mansfield':'2026-10-31','jay':'2026-11-01'}
        for site,cfg in pipeline.SITES.items():
            with self.subTest(site=site):
                weather=pipeline.read_weather(pipeline.INPUT/cfg['weather'])
                self.assertTrue(weather.loc['2026-11-02':'2026-11-04'].isna().all().all())
                self.assertEqual(str(weather.precipitation_inches.dropna().index.max().date()),expected[site])
                self.assertEqual(str(weather.tmax_c.dropna().index.max().date()),'2026-11-01')
                context=weather.reindex(pd.date_range(end='2026-11-02',periods=60)).to_numpy()[None]
                self.assertTrue(np.isnan(context[0,-1]).all())
                self.assertTrue((pipeline.weather_summary(context)[0,-3:]>=1).all())


class PipelineFunctionBoundaryTests(unittest.TestCase):
    def test_noaa_quality_flags_and_missing_values_survive_weather_loading(self):
        rows=[]
        for i,(value,q,m) in enumerate([(25.4,'',''),(99.,'X',''),('','',''),(0.,'','P'),(0.,'','T')]):
            row={'date':str(pd.Timestamp('2000-01-01').date()+pd.Timedelta(days=i))}
            for name in ['tmax_c','tmin_c','precipitation_inches']:
                row.update({name:value,name+'_quality_flag':q,name+'_measurement_flag':m})
            rows.append(row)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'weather.csv';pd.DataFrame(rows).to_csv(path,index=False)
            result=pipeline.read_weather(path)
        self.assertEqual(result.iloc[0].tmax_c,25.4)
        self.assertAlmostEqual(result.iloc[0].precipitation_inches,np.log1p(25.4))
        self.assertTrue(result.iloc[1:3].isna().all().all())
        np.testing.assert_array_equal(result.iloc[3:].to_numpy(),0.)

    def test_training_history_censors_gaps_and_never_reads_future_target(self):
        dates=pd.date_range('2000-09-01',periods=8)
        target=pd.Series([0.,2.,2.,np.nan,2.,2.,0.,9.],index=dates)
        origins=dates[:7]
        before=pipeline.observed_history(target,origins)
        changed=target.copy();changed.iloc[-1]=999.
        np.testing.assert_array_equal(before,pipeline.observed_history(changed,origins))
        self.assertTrue(np.isnan(before[3]).all())
        self.assertTrue(np.isnan(before[4:6,1]).all())
        np.testing.assert_array_equal(before[6],[0.,1.])

    def test_statistical_fit_cannot_see_heldout_rows_or_future_support(self):
        d={'one_years':np.array([2000,2000,2001,2001]),'one_summary':np.arange(12).reshape(4,3),
           'one_f':np.arange(32).reshape(4,8),'one_history':np.arange(8).reshape(4,2),'one_y':np.array([0.,2.,99.,100.])}
        class FitRecorder:
            def __init__(self,**kwargs):pass
            def fit(self,*args):self.args=args
        with patch.object(pipeline,'EventModel',FitRecorder):
            model=pipeline.fit_stat(d,[2000],weather=True)
        np.testing.assert_array_equal(model.args[-1],[0.,2.])
        np.testing.assert_array_equal(model.args[0],d['one_summary'][:2])
        np.testing.assert_array_equal(model.future_max,d['one_f'][:2].max(0))

    def test_statistical_forecasts_withhold_observed_target_even_if_present(self):
        class ForecastRecorder:
            future_min=np.full(8,-10.);future_max=np.full(8,10.)
            def forecast(self,*args,**kwargs):self.args=args;return {'checked':True}
        model=ForecastRecorder()
        f=np.stack([pipeline.future(pd.Timestamp('2000-11-01'))]*2)
        d={'summary':np.zeros((2,6)),'f':f,'initial':np.array([[9.,5.,1.,1.]]*2),'y':np.full((2,60),999.)}
        pipeline.predict_stat(model,d,np.array([0,1]),n_paths=2)
        self.assertTrue(np.isnan(model.args[2]).all() and np.isnan(model.args[3]).all())
        self.assertEqual(model.args[1].shape,(2,60,8))

    def test_climatology_donors_ignore_all_heldout_targets(self):
        dates=pd.date_range('2000-10-01','2002-04-30')
        target=pd.Series(np.arange(len(dates))%6,index=dates,dtype=float)
        d={'target':target};origins=['2001-11-04']
        before=pipeline.climate(d,[2000],origins)
        altered=target.copy();altered.loc['2001-10-01':]=999.
        after=pipeline.climate({'target':altered},[2000],origins)
        for key in before:np.testing.assert_array_equal(before[key],after[key])

    def test_extended_calendar_covariates_support_fit_and_reject_mismatched_forecast(self):
        rng=np.random.default_rng(19);n=240
        phase=np.arange(n)*2*np.pi/366
        calendar=np.column_stack([np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase),rng.uniform(-2,2,n),np.zeros((n,3))])
        summary=rng.normal(size=(n,3));current=rng.choice([0.,2.],n);target=rng.choice([0.,.3,2.,6.],n)
        model=EventModel(kind='semi_markov',use_weather=True).fit(summary,calendar,current,np.ones(n),target)
        forecast=np.repeat(calendar[:2,None,:],4,axis=1)
        pred=model.forecast(summary[:2],forecast,[np.nan,np.nan],[np.nan,np.nan],n_paths=64,exceedance_threshold=5.)
        self.assertEqual(model.calendar_features,8)
        self.assertTrue(np.isfinite(pred['mean']).all())
        self.assertTrue((pred['p_exceedance']<=pred['p_active']+1e-10).all())
        with self.assertRaises(ValueError):model.forecast(summary[:2],forecast[...,:4],[np.nan,np.nan],[0,0])

    def test_prediction_rows_share_observation_mask_and_start_after_origin(self):
        d={'y':np.array([[2.,np.nan,0.]]),'meta':pd.DataFrame({'winter':[2000],'origin':['2000-11-04']})}
        prediction={k:np.zeros((1,3)) for k in ['mean','p_active','p_powder','q10','q90','crps']}
        result=pipeline.prediction_rows(d,np.array([0]),'test',prediction,'fold0')
        self.assertEqual(result.date.tolist(),['2000-11-05','2000-11-07'])
        self.assertEqual(result.lead.tolist(),[1,3])
        self.assertEqual(result.actual_inches.tolist(),[2.,0.])

    def test_current_issue_dates_states_and_monthly_sums_match_daily_forecasts(self):
        requested=[];stat_states=[];neural_states=[]
        def contexts(origins):
            requested.extend(origins)
            return np.zeros((len(origins),60,3),dtype='float32')
        def output(n=2048):
            return {'mean':np.full((1,60),2.),'p_active':np.full((1,60),.5),
                    'p_exceedance':np.full((1,60),.25),'paths':np.full((1,n,60),2.,dtype='float32')}
        class CurrentModel:
            future_min=np.full(8,-10.);future_max=np.full(8,10.)
            def forecast(self,summary,features,current,age,**kwargs):
                stat_states.append((current.copy(),age.copy()))
                return output(kwargs['n_paths'])
        def neural_forecast(model,scaler,x,f,initial,**kwargs):
            neural_states.append(initial.copy());return output()
        weather=pd.DataFrame(np.zeros((3,3)),index=pd.date_range('2026-10-30',periods=3),columns=['tmax_c','tmin_c','precipitation_inches'])
        climate={k:np.full((1,60),v) for k,v in [('mean',2.),('p_active',.5),('p_powder',.25),('q10',0.),('q90',6.),('crps',0.)]}
        with tempfile.TemporaryDirectory() as td:
            d={'site':'alta','dest':Path(td),'meta':pd.DataFrame({'winter':[2000]}),'contexts':contexts,'weather':weather,'years':[2000]}
            scaler=SimpleNamespace(future_min=np.full(10,-2.),future_max=np.full(10,2.))
            with patch.object(pipeline,'select_and_fit',return_value=({'Weather duration + ENSO':CurrentModel()},object(),scaler)), \
                 patch.object(pipeline.neural,'forecast_neural',side_effect=neural_forecast), \
                 patch.object(pipeline.neural,'exceedance_probability',return_value=np.full((1,60),.25)), \
                 patch.object(pipeline.neural,'simulate_neural',return_value=output()['paths']), \
                 patch.object(pipeline,'climate',return_value=climate):
                pipeline.issue(d)
            rows=pd.read_csv(Path(td)/'forecast_daily.csv')
            totals=pd.read_csv(Path(td)/'forecast_totals.csv')
            manifest=json.loads((Path(td)/'forecast_manifest.json').read_text())
        self.assertEqual(requested,[pd.Timestamp('2026-11-04')])
        self.assertEqual(rows.groupby('model').size().to_list(),[57,57,57])
        self.assertEqual(rows.date.min(),'2026-11-05');self.assertEqual(rows.date.max(),'2026-12-31')
        self.assertEqual(manifest['weather_nominal_last_date'],'2026-11-02')
        for state,age in stat_states:self.assertTrue(np.isnan(state).all() and np.isnan(age).all())
        for initial in neural_states:
            self.assertTrue(np.isnan(initial[:,0]).all());np.testing.assert_array_equal(initial[:,1:],0.)
        for _,row in totals.iterrows():
            expected_days={'Nov5–Dec31':57,'Nov5–30':26,'December':31}[row.period]
            self.assertEqual(row.expected_snowfall_inches,expected_days*2.)
            self.assertEqual(row.expected_powder_days,expected_days*.25)


if __name__=='__main__':unittest.main()
