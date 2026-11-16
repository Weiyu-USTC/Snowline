"""Deployment quantities, clocks, frozen choices and graceful source failure."""
from datetime import datetime,timedelta,timezone
import csv,json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dashboard_portfolio import read_deployed,deployment_neural48_families,_integrate

UTC=timezone.utc
NOW=datetime(2026,11,11,10,30,tzinfo=UTC)
START=datetime(2026,11,11,11,tzinfo=UTC)
RUN=datetime(2026,11,11,6,tzinfo=UTC)
SITES=['alta','vail','mansfield','jay']


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        choices={}
        for site in SITES:
            choices[site]={str(n):dict(method_id=('original_MRMS_nowcastnet' if site=='alta' and n==1 else 'HRRR' if site=='vail' and n<=3 else 'NEXRAD_HRRR_portfolio' if n<=3 else 'Alta12_operational_portfolio' if site=='alta' and n==12 else 'HRRR_neural24_portfolio' if n==24 else 'neural48_weather_radar' if n==48 else 'HRRR_native_snowfall'),quantity='liquid_precipitation' if n<=3 else 'snowfall') for n in [1,3,6,12,24,48]}
            choices[site]['24']['weights']=dict(scale_fit_on_train=1,native_hrrr_component='hrrr_native',hrrr_weight=.3,prior_neural_weight=.7,validation_residual_p80_inches=1)
            choices[site]['48']['family']='weather_only' if site=='mansfield' else 'weather_radar'
        self.write('outputs/nwp-weather/forecast-portfolio/deployment_selection.json',{'sites':choices,'checkpoint_sha256':{'nowcastnet':'correct'}})
        provider=[]
        times=[RUN+timedelta(hours=h) for h in range(1,97)]
        for site in SITES:
            provider.append({'timezone':'GMT','utc_offset_seconds':0,'hourly_units':{'snowfall':'cm','precipitation':'mm'},'hourly':{'time':[t.replace(tzinfo=None).isoformat() for t in times],'snowfall':[2.54]*96,'precipitation':[4.]*96}})
        self.write('data/provider.json',provider)
        self.neural={'generated_at':NOW.isoformat(),'sites':{}}
        for site in SITES:
            family='weather_radar' if site in ['alta','vail'] else 'weather_only'
            self.neural['sites'][site]=dict(site_id=site,status='experimental',family=family,model_name='Old technical name',issued_at=(NOW-timedelta(minutes=5)).isoformat(),start=START.isoformat(),end=(START+timedelta(hours=48)).isoformat(),target={'name':site+' station','proxy':site in ['mansfield','jay']},sources={'provider':{'best_match':{'path':'data/provider.json','retrieved_at':NOW.isoformat()}}},warnings=[],totals=[dict(hours=h,end=(START+timedelta(hours=h)).isoformat(),mean_inches=10 if h==24 else 20,q10_inches=0,q90_inches=30) for h in [24,48]])
        self.write('outputs/nwp-weather/neural48-trained/current_forecast.json',self.neural)
        self.native=dict(run_time_utc=RUN.isoformat(),assumed_available_utc=(RUN+timedelta(hours=3)).isoformat(),retrieved_utc=NOW.isoformat(),forecast_end_utc=(RUN+timedelta(hours=48)).isoformat())
        self.write('data/hrrr_portfolio/current.json',self.native)
        self.weather=[];self.snow=[]
        for site in SITES:
            for h in range(1,49):
                row=dict(site=site,run_time_utc=RUN.isoformat(),forecast_hour=h,valid_time_utc=(RUN+timedelta(hours=h)).isoformat())
                self.weather.append(dict(**row,precip_mm=2))
                self.snow.append(dict(**row,snowfall_inches=1))
        self.csv('data/hrrr_portfolio/current_weather.csv',self.weather)
        self.csv('data/hrrr_portfolio/current_native_snowfall.csv',self.snow)
        self.radar={'issued_at':NOW.isoformat(),'units':'mm','quantity':'liquid_equivalent_precipitation','sites':{}}
        for site in SITES:
            self.radar['sites'][site]=dict(available=True,model='NEXRAD_HRRR_portfolio',issued_at=NOW.isoformat(),start=START.isoformat(),end=(START+timedelta(hours=3)).isoformat(),hourly=[dict(start=(START+timedelta(hours=h)).isoformat(),end=(START+timedelta(hours=h+1)).isoformat(),liquid_mm=h+1,liquid_lower_mm=h+.5,liquid_upper_mm=h+1.5) for h in range(3)],limitations=[])
        self.write('outputs/nexrad-short/current_forecast.json',self.radar)
        first=datetime(2026,11,11,10,30,tzinfo=UTC)
        self.mrms=dict(start=first.isoformat(),end=(first+timedelta(hours=3)).isoformat(),issued_at=(NOW-timedelta(minutes=10)).isoformat(),unit='mm liquid equivalent',checkpoint_sha256={'nowcastnet':'correct'},models=[dict(id='nowcastnet',hours=[dict(time=(first+timedelta(minutes=10*h)).isoformat(),mean_liquid_mm=h/3,q10_liquid_mm=0,q90_liquid_mm=h/2) for h in range(19)])])
        self.write('outputs/dashboard/alta/latest.json',self.mrms)
        self.alta12={'alta12':dict(available=True,start=START.isoformat(),end=(START+timedelta(hours=12)).isoformat(),issued_at=NOW.isoformat(),mean_inches=.7,q10_inches=0,q90_inches=2,method='actual_alta12_method')}
        self.write('outputs/nwp-weather/forecast-portfolio/current_forecast.json',self.alta12)

    def tearDown(self):self.temp.cleanup()
    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def csv(self,name,rows):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def read(self,site='alta',now=NOW):return read_deployed(site,now=now,root=self.root)
    def horizon(self,site,h,now=NOW):return next(r for r in self.read(site,now)['horizons'] if r['hours']==h)

    def test_all_horizons_have_same_model_label_and_explicit_units(self):
        p=self.read();self.assertEqual([r['hours'] for r in p['horizons']],[1,3,6,12,24,48])
        self.assertTrue(all(r['model_name']=='Our model' for r in p['horizons']))
        self.assertEqual(self.horizon('alta',1)['unit'],'mm');self.assertEqual(self.horizon('alta',12)['unit'],'inches')
    def test_alta1_uses_existing_trajectory_clock_hour_without_retimestamping(self):
        r=self.horizon('alta',1)
        self.assertEqual(r['method_id'],'original_MRMS_nowcastnet');self.assertEqual(r['start'],START.isoformat())
        self.assertEqual(r['issued_at'],self.mrms['issued_at']);self.assertAlmostEqual(r['mean'],2)
        self.assertEqual(r['provider_mean'],4);self.assertIsNone(r['lower'])
    def test_three_hour_liquid_is_sum_without_summing_quantile_bands(self):
        r=self.horizon('alta',3);self.assertEqual(r['mean'],6);self.assertEqual(r['provider_mean'],12)
        self.assertIsNone(r['upper']);self.assertEqual(r['quantity'],'liquid_precipitation')
    def test_snow_provider_is_converted_cm_to_inches_once(self):
        r=self.horizon('vail',6);self.assertEqual(r['mean'],6);self.assertAlmostEqual(r['provider_mean'],6)
    def test_alta12_is_exact_fixed_target_window(self):
        r=self.horizon('alta',12);self.assertEqual(r['mean'],.7);self.assertEqual(r['method_id'],'actual_alta12_method')
        self.alta12['alta12']['start']=(START+timedelta(hours=1)).isoformat();self.alta12['alta12']['end']=(START+timedelta(hours=13)).isoformat();self.write('outputs/nwp-weather/forecast-portfolio/current_forecast.json',self.alta12)
        self.assertEqual(self.horizon('alta',12)['status'],'fallback')
    def test_24_hour_portfolio_uses_only_original_first24_component(self):
        r=self.horizon('jay',24);self.assertAlmostEqual(r['mean'],14.2)
        self.neural['sites']['jay']['family']='weather_radar';self.write('outputs/nwp-weather/neural48-trained/current_forecast.json',self.neural)
        r=self.horizon('jay',24);self.assertEqual(r['status'],'fallback');self.assertEqual(r['mean'],24)
    def test_jay48_requires_new_selected_family_and_keeps24_unchanged(self):
        self.assertEqual(self.horizon('jay',48)['status'],'unavailable')
        self.neural['sites']['jay']['deployed_48']=dict(self.neural['sites']['jay']['totals'][1],family='weather_radar',mean_inches=50)
        self.write('outputs/nwp-weather/neural48-trained/current_forecast.json',self.neural)
        self.assertEqual(self.horizon('jay',48)['mean'],50);self.assertAlmostEqual(self.horizon('jay',24)['mean'],14.2)
    def test_missing_radar48_fallback_must_be_explicit(self):
        self.neural['sites']['jay']['deployed_48']=dict(self.neural['sites']['jay']['totals'][1],family='weather_only',status='fallback',reason='Recent radar unavailable')
        self.write('outputs/nwp-weather/neural48-trained/current_forecast.json',self.neural)
        r=self.horizon('jay',48);self.assertEqual(r['status'],'fallback');self.assertEqual(r['mean'],20)
        self.assertEqual(r['method_id'],'neural48_weather_only')
    def test_bad_or_stale_radar_falls_back_to_real_hrrr_not_zero(self):
        self.radar['sites']['mansfield']['issued_at']=(NOW-timedelta(hours=2)).isoformat();self.write('outputs/nexrad-short/current_forecast.json',self.radar)
        r=self.horizon('mansfield',3);self.assertEqual(r['status'],'fallback');self.assertEqual(r['mean'],6)
    def test_checkpoint_identity_change_is_not_silently_deployed(self):
        self.mrms['checkpoint_sha256']['nowcastnet']='different';self.write('outputs/dashboard/alta/latest.json',self.mrms)
        r=self.horizon('alta',1);self.assertEqual(r['status'],'fallback');self.assertEqual(r['method_id'],'HRRR_liquid')
    def test_missing_native_hour_is_unavailable_not_zero_or_provider_extension(self):
        rows=[r for r in self.snow if not(r['site']=='vail' and r['forecast_hour']==8)]
        self.csv('data/hrrr_portfolio/current_native_snowfall.csv',rows)
        r=self.horizon('vail',6);self.assertEqual(r['status'],'unavailable');self.assertIsNone(r['mean'])
    def test_native_end_is_never_extended_to_fortyeight_hours(self):
        self.native['forecast_end_utc']=(START+timedelta(hours=5)).isoformat();self.write('data/hrrr_portfolio/current.json',self.native)
        self.assertEqual(self.horizon('vail',6)['status'],'unavailable')
    def test_immutable_native_pair_takes_precedence_over_compatibility_files(self):
        self.csv('data/hrrr_portfolio/immutable.csv',self.snow)
        self.native['snowfall_path']='data/hrrr_portfolio/immutable.csv';self.write('data/hrrr_portfolio/current.json',self.native)
        (self.root/'data/hrrr_portfolio/current_native_snowfall.csv').unlink()
        self.assertEqual(self.horizon('vail',6)['mean'],6)
    def test_begun_neural48_window_is_retained_as_unavailable_not_moved(self):
        r=self.horizon('alta',48,START+timedelta(minutes=1));self.assertIsNone(r['mean'])
        self.assertTrue(r['window_started']);self.assertEqual(r['start'],START.isoformat())
    def test_stale_provider_does_not_destroy_current_model(self):
        self.neural['sites']['alta']['sources']['provider']['best_match']['retrieved_at']=(NOW-timedelta(hours=7)).isoformat();self.write('outputs/nwp-weather/neural48-trained/current_forecast.json',self.neural)
        r=self.horizon('alta',3);self.assertEqual(r['mean'],6);self.assertIsNone(r['provider_mean'])
    def test_duplicate_forecast_intervals_rejected(self):
        a={'start':START.isoformat(),'end':(START+timedelta(hours=1)).isoformat(),'value':1}
        with self.assertRaises(ValueError):_integrate([a,a],START,START+timedelta(hours=1),'value')
    def test_neural_family_mapping_comes_from_frozen_selection(self):
        self.assertEqual(deployment_neural48_families(self.root)['jay'],'weather_radar')
    def test_portfolio_issue_cannot_precede_newest_component_availability(self):
        self.neural['sites']['alta']['issued_at']=(NOW-timedelta(hours=1)).isoformat()
        self.write('outputs/nwp-weather/neural48-trained/current_forecast.json',self.neural)
        self.native['assumed_available_utc']=(NOW-timedelta(minutes=5)).isoformat()
        self.write('data/hrrr_portfolio/current.json',self.native)
        r=self.horizon('alta',24)
        self.assertEqual(r['issued_at'],self.native['assumed_available_utc'])

if __name__=='__main__':unittest.main()
