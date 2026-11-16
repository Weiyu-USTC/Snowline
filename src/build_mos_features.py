"""Align archived, issued MOS forecasts with fully future24-hour snow windows.

Forecasts are airport/regional covariates, not mountain observations. QPF remains
categorical. The weighted six-hour PoP is a predictor, not a24-hour probability.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/nwp_archive_probe'
SITES={'alta':('KSLC',11.),'vail':('KLXV',11.),'mansfield':('KMVL',21.),'jay':('KMVL',12.5)}
INSTANT={'temperature_c':'temp','dewpoint_c':'dewpoint','wind_speed_mps':'wind_speed','wind_u_mps':'wind_u','wind_v_mps':'wind_v'}


def instant_features(hours,values,start,end,prefix):
    """Trapezoid average of bounded linear interpolation, with no gap filling."""
    before=np.flatnonzero(hours<=start);after=np.flatnonzero(hours>=end)
    if not len(before) or not len(after):return None
    lo,hi=before[-1],after[0]
    x=hours[lo:hi+1];v=values[lo:hi+1]
    if not np.isfinite(v).all() or np.max(np.diff(x),initial=0)>6.01:return None
    knot=np.r_[start,x[(x>start)&(x<end)],end];z=np.interp(knot,x,v)
    unit='c' if prefix in ['temp','dewpoint'] else 'ms'
    return {f'forecast_{prefix}_mean_{unit}':float(np.trapezoid(z,knot)/(end-start)),
            f'forecast_{prefix}_min_{unit}':float(z.min()),f'forecast_{prefix}_max_{unit}':float(z.max()),
            f'forecast_{prefix}_change_{unit}':float(z[-1]-z[0])}


def daily_features(runframe,start,end):
    h=runframe.lead_hours.to_numpy(float);result={}
    for col,prefix in INSTANT.items():
        feature=instant_features(h,runframe[col].to_numpy(float),start,end,prefix)
        if feature is None:return None,'incomplete_instantaneous_forecast'
        result.update(feature)
    # Each P06/Q06 record is a disjoint6-hour interval ending atftime.
    p=runframe.precip_probability_6h.to_numpy(float);q=runframe.qpf_category_6h.to_numpy(float)
    ix=np.isfinite(p)&np.isfinite(q)
    hh=h[ix];weight=np.maximum(0,np.minimum(hh,end)-np.maximum(hh-6,start))
    if not np.isclose(weight.sum(),end-start):return None,'incomplete_precipitation_windows'
    if len(hh)>1 and np.min(np.diff(hh))<5.99:return None,'overlapping_precipitation_windows'
    pp=p[ix];qq=q[ix];keep=weight>0
    result.update(forecast_pop_6h_weighted_mean=float(np.sum(weight*pp)/(end-start)),forecast_pop_6h_max=float(pp[keep].max()),
                  forecast_qpf_ordinal_weighted_mean=float(np.sum(weight*qq)/(end-start)),forecast_qpf_ordinal_max=float(qq[keep].max()))
    for cat in range(7):result[f'forecast_qpf_category_{cat}_fraction']=float(np.sum(weight*(qq==cat))/(end-start))
    # A cold-times-precipitation summary is an engineering predictor, not snowinches.
    # It belongs to the temperature ablation, since it uses forecast temperature.
    result['forecast_temp_qpf_cold_interaction']=result['forecast_qpf_ordinal_weighted_mean']*max(0.,-result['forecast_temp_mean_c'])
    return result,None


def build(shift_mode='main'):
    rows=[];rejected=[];inputs={}
    for site,(station,clock) in SITES.items():
        if station not in inputs:
            path=DATA/f'mos_{station}_00utc.csv.gz'
            if not path.exists():continue
            d=pd.read_csv(path,parse_dates=['runtime','ftime']);d['runtime']=pd.to_datetime(d.runtime,utc=True);d['ftime']=pd.to_datetime(d.ftime,utc=True)
            inputs[station]=d
        shift=0.
        if shift_mode!='main':shift=(-1 if shift_mode=='minus' else 1)*({'vail':1.,'mansfield':6.,'jay':6.}.get(site,0.))
        for run,g in inputs[station].groupby('runtime',sort=True):
            g=g.sort_values('ftime');assert run.hour==0
            for lead in [1,2]:
                start=clock+shift+(lead-1)*24;end=start+24
                if start<6 or end>72:
                    rejected.append({'site':site,'run':str(run),'lead_day':lead,'reason':'outside_fully_future_forecast_support'});continue
                feature,reason=daily_features(g,start,end)
                if feature is None:
                    rejected.append({'site':site,'run':str(run),'lead_day':lead,'reason':reason});continue
                valid_start=run+pd.Timedelta(hours=start);valid_end=run+pd.Timedelta(hours=end);issue=run+pd.Timedelta(hours=6)
                assert valid_start>=issue and valid_end<=run+pd.Timedelta(hours=72)
                row={'site':site,'station':station,'model':'GFS_MOS','archive_type':'operational_forecast','forecast_reference_utc':run.isoformat(),
                     'issue_time_utc':issue.isoformat(),'valid_start_utc':valid_start.isoformat(),'valid_end_utc':valid_end.isoformat(),
                     'target_date':str((run+pd.Timedelta(days=lead)).date()),'lead_day':lead,'end_lead_hours_from_issue':end-6,
                     'assumed_publication_lag_hours':6,'observation_clock_shift_hours':shift,'forecast_metadata_source':'IEM GFS MOS archive',**feature}
                rows.append(row)
    result=pd.DataFrame(rows)
    suffix='' if shift_mode=='main' else f'_clock_{shift_mode}'
    result.to_csv(DATA/f'mos_daily_features{suffix}.csv',index=False)
    pd.DataFrame(rejected).to_csv(DATA/f'mos_feature_exclusions{suffix}.csv',index=False)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['main','all'],default='all');args=parser.parse_args();summary={}
    for mode in ['main','minus','plus'] if args.mode=='all' else ['main']:
        r=build(mode);summary[mode]=r.groupby('site').agg(rows=('target_date','count'),first_target=('target_date','min'),last_target=('target_date','max')).reset_index().to_dict('records');print(mode,summary[mode],flush=True)
    metadata={'source':'Archived operational NOAA GFS MOS forecast, hosted by Iowa State','publication_buffer_hours':6,'reference_cycle':'00UTC','target_windows_UTC':{site:clock for site,(station,clock) in SITES.items()},'station_map':{site:station for site,(station,clock) in SITES.items()},'clock_evidence':{'alta':'manual Collins04LST ->11UTC, dateisintervalendpoint','vail':'CAIC04displayclock;11UTCassumesMST;plus/minus1htestsclockambiguity','mansfield':'21UTCassumes16EST;historicalNOAAOBS-TIMEblank;plus/minus6htested','jay':'12:30UTCassumes07:30EST;documentedin1995NOAAstationhistorybutnotallarchive;plus/minus6htested'},'features':{'instantaneous':'temperature/dewpoint/winds averaged by bounded linearinterpolation between modelvalidtimes; noextrapolation or missing-value gapfilling','precipitation':'P06 probabilities andQ06 categories weightedbyoverlap of their6hperiodwithtargetwindow; require24hcoverage; categories neverconvertedtoexactprecipitation','pop_warning':'weightedmean6hPoP is apredictor, NOT probabilityofprecipitationover24h','wind':'10mstationwinds convertedmeteorologically toU/V; airportMOS notmountainridgewinds','target':'snowfallobservationsjoinedseparately; nofutureobservationshere'},'sensitivity':'plus/minus6hoursVT, plus/minus1hourVail; unavailableend>72hrows excluded; comparecommoncohort','summary':summary}
    (DATA/'mos_feature_metadata.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
