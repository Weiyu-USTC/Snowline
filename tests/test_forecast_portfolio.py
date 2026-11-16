"""Forecast publication survives missing HRRR without inventing zero snowfall."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import forecast_portfolio as portfolio


class ForecastPortfolioTests(unittest.TestCase):
    selection={'native_hrrr_component':'hrrr_native','hrrr_weight':.5,
               'prior_neural_weight':.5,'validation_residual_p80_inches':1.}

    def test_missing_native_component_keeps_amount_and_bounds_missing(self):
        values=[0.]*24;values[7]=np.nan
        p=portfolio.portfolio24_result(values,2.,self.selection,'end')
        self.assertFalse(p['available'])
        self.assertEqual([p[k] for k in ['mean_inches','lower_inches','upper_inches']],[None,None,None])
        self.assertIn('complete 24-hour',p['reason'])

    def test_true_zero_is_available_and_not_a_missing_value(self):
        p=portfolio.portfolio24_result([0.]*24,0.,self.selection,'end')
        self.assertTrue(p['available']);self.assertEqual(p['mean_inches'],0.)
        self.assertEqual((p['lower_inches'],p['upper_inches']),(0.,1.))

    def test_stale_native_component_is_not_blended(self):
        p=portfolio.portfolio24_result([1.]*24,2.,self.selection,'end','Cached HRRR initialization is too old')
        self.assertFalse(p['available']);self.assertIsNone(p['mean_inches'])

    def test_missing_portfolio_can_still_publish_and_plot_neural_forecasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'outputs/nwp-weather/forecast-portfolio'
            out.mkdir(parents=True)
            def save(path,data):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
            now=pd.Timestamp.now(tz='UTC');start=now.ceil('h')
            run=now.floor('6h');ends=pd.date_range(run+pd.Timedelta(hours=1),periods=48,freq='h')
            provider_ends=pd.date_range(start+pd.Timedelta(hours=1),periods=48,freq='h')
            native=dict(run_time_utc=run.isoformat(),assumed_available_utc=(run+pd.Timedelta(hours=3)).isoformat(),
                forecast_end_utc=(run+pd.Timedelta(hours=48)).isoformat(),
                weather_path='paired/weather.csv',snowfall_path='paired/snow.csv')
            # If a run is only just initialized, use the earlier cycle to keep
            # the fixture's publication time causal at any wall-clock hour.
            if pd.Timestamp(native['assumed_available_utc'])>now:
                run-=pd.Timedelta(hours=6)
                ends=pd.date_range(run+pd.Timedelta(hours=1),periods=48,freq='h')
                native.update(run_time_utc=run.isoformat(),assumed_available_utc=(run+pd.Timedelta(hours=3)).isoformat(),forecast_end_utc=(run+pd.Timedelta(hours=48)).isoformat())
            save('data/hrrr_portfolio/current.json',native)
            weather=[];snow=[]
            for site in portfolio.SITES:
                for hour,end in enumerate(ends,1):
                    common=dict(site=site,forecast_hour=hour,run_time_utc=run.isoformat(),valid_time_utc=end.isoformat())
                    weather.append(dict(**common,temperature_k=270.,precip_mm=0.))
                    snow.append(dict(**common,snowfall_inches=np.nan))
            (root/'paired').mkdir();pd.DataFrame(weather).to_csv(root/'paired/weather.csv',index=False);pd.DataFrame(snow).to_csv(root/'paired/snow.csv',index=False)
            # No legacy flat files: successful build also verifies paired-pointer reads.
            provider=[dict(hourly_units={'snowfall':'cm'},hourly={'time':[e.isoformat() for e in provider_ends],'snowfall':[2.54]*48}) for _ in portfolio.SITES]
            save('provider.json',provider)
            sites={site:dict(start=start.isoformat(),issued_at=now.isoformat(),end=provider_ends[-1].isoformat(),
                target={'name':site},sources={'provider':{'best_match':{'path':'provider.json'}}},
                totals=[{'mean_inches':1.},{'mean_inches':3.}]) for site in portfolio.SITES}
            save('outputs/nwp-weather/neural48-trained/current_forecast.json',{'sites':sites})
            save('outputs/nexrad-short/current_forecast.json',{'sites':{}})
            save('outputs/nwp-weather/forecast-portfolio/hrrr24_selection.json',{s:self.selection for s in portfolio.SITES})
            with patch.object(portfolio,'ROOT',root),patch.object(portfolio,'OUT',out),patch.object(portfolio,'alta12',side_effect=ValueError('Missing Alta fixed cycle')):
                result=portfolio.build()
            self.assertEqual(result['status'],'partial');self.assertNotIn('plot_error',result)
            self.assertEqual(len(result['warnings']),4)
            for data in result['sites'].values():
                self.assertIsNone(data['portfolio24']['mean_inches'])
                self.assertIsNone(data['portfolio24']['lower_inches'])
                self.assertEqual(data['existing_neural'][1]['mean_inches'],3.)
                self.assertIsNone(data['hourly'][0]['display_snow_inches'])
            self.assertTrue((out/'cumulative_forecasts.png').is_file())
            self.assertTrue((out/'hourly_forecasts.png').is_file())
            self.assertEqual(json.loads((out/'current_forecast.json').read_text())['status'],'partial')


if __name__=='__main__':unittest.main()
