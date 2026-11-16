"""Fetch actual HRRR forecast chunks for Alta, preserving times and provenance.

The University of Utah public archive is a repackaging of operational HRRR GRIB.
Only ``fcst`` arrays are read. Missing chunks remain missing, including the
documented iris-grib constant-field bug; they are never interpreted as zeros.
Run with .venv-hrrr/bin/python (numcodecs and pyproj are the only extra readers).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numcodecs
import numpy as np
import pandas as pd
from pyproj import Proj
import requests
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/alta_nowcast/hrrr"
BASE = "https://hrrrzarr.s3.amazonaws.com/"
LAT, LON = 40.57607, -111.63889
PROJ = dict(a=6371229, b=6371229, proj="lcc", lon_0=262.5,
            lat_0=38.5, lat_1=38.5, lat_2=38.5)
X0, Y0, DX = -2697520.1425219304, -1587306.1525566636, 3000.
FIELDS = {
    "precip_mm": "surface/APCP_1hr_acc_fcst",
    "snow_depth_m": "surface/ASNOW_1hr_acc_fcst",
    "snow_water_mm": "surface/WEASD_1hr_acc_fcst",
    "temperature_k": "2m_above_ground/TMP",
    "dewpoint_k": "2m_above_ground/DPT",
    "wind_u_ms": "10m_above_ground/UGRD",
    "wind_v_ms": "10m_above_ground/VGRD",
    "relative_humidity_pct": "2m_above_ground/RH",
}
DEFAULT_FIELDS = [n for n in FIELDS if n != "snow_depth_m"]
EXPECTED_UNITS = {
    "precip_mm": {"kg m-2", "kg m**-2"},
    "snow_depth_m": {"m"}, "snow_water_mm": {"kg m-2", "kg m**-2"},
    "temperature_k": {"K"}, "dewpoint_k": {"K"},
    "wind_u_ms": {"m s-1", "m s**-1"},
    "wind_v_ms": {"m s-1", "m s**-1"},
    "relative_humidity_pct": {"%"},
}


def location():
    proj = Proj(**PROJ)
    x, y = proj(LON, LAT)
    ix, iy = round((x-X0)/DX), round((y-Y0)/DX)
    gx, gy = X0+ix*DX, Y0+iy*DX
    glon, glat = proj(gx, gy, inverse=True)
    return dict(latitude=LAT, longitude=LON, grid_x=ix, grid_y=iy,
                grid_latitude=glat, grid_longitude=glon,
                distance_projected_m=float(np.hypot(x-gx, y-gy)),
                chunk_y=iy//150, chunk_x=ix//150,
                internal_y=iy%150, internal_x=ix%150)


def get(key: str, cache: Path, retries: int=4):
    path = cache / key
    if path.exists():
        data = path.read_bytes()
    else:
        error = None
        for attempt in range(retries):
            try:
                r = requests.get(BASE+key, timeout=60)
                if r.status_code == 404:
                    raise FileNotFoundError(BASE+key)
                r.raise_for_status()
                data = r.content
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name+".tmp")
                temp.write_bytes(data)
                temp.replace(path)
                break
            except FileNotFoundError:
                raise
            except requests.RequestException as exc:
                error = exc
                time.sleep(min(2**attempt, 8))
        else:
            raise error
    return data, dict(url=BASE+key, cache_path=str(path.relative_to(cache.parent)),
                      bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def decode(data: bytes, meta: dict):
    if meta.get("filters"):
        raise ValueError("Unexpected Zarr filters; inspect metadata before decoding")
    raw = numcodecs.get_codec(meta["compressor"]).decode(data) if meta.get("compressor") else data
    a = np.frombuffer(raw, dtype=np.dtype(meta["dtype"]))
    return a.reshape(meta["chunks"] or ())


def field(run: pd.Timestamp, name: str, out: Path=OUT):
    spec = FIELDS[name]
    prefix = run.strftime("sfc/%Y%m%d/%Y%m%d_%Hz_fcst.zarr/")+spec+"/"
    raw_meta, prov_meta = get(prefix+".zmetadata", out/"raw")
    metadata = json.loads(raw_meta)["metadata"]
    specmeta = metadata[spec+"/.zarray"]
    attrs = metadata[spec+"/.zattrs"]
    units = attrs.get("units")
    units_basis = "source Zarr attribute"
    # Some archive fields omit units but preserve the unambiguous GRIB2 code.
    if name == "precip_mm" and units is None and attrs.get("GRIB_PARAM") == [2,0,1,8]:
        units = "kg m-2"
        units_basis = "NCEP HRRR GRIB inventory: APCP parameter 2/0/1/8, kg/m^2"
    if units not in EXPECTED_UNITS[name]:
        raise ValueError(f"Unexpected units for {name}: {units}")
    loc = location()
    chunkid = f"0.{loc['chunk_y']}.{loc['chunk_x']}"
    data, prov_data = get(prefix+spec+"/"+chunkid, out/"raw")
    values = decode(data, specmeta).astype("float32")
    fill = specmeta.get("fill_value")
    if isinstance(fill, (float,int)):
        values[values == fill] = np.nan
    period_data, prov_period = get(prefix+"forecast_period/0", out/"raw")
    period_meta = metadata["forecast_period/.zarray"]
    periods = decode(period_data, period_meta)
    if metadata["forecast_period/.zattrs"]["units"] != "hours":
        raise ValueError("Unexpected forecast_period units")
    periods = periods[:specmeta["shape"][0]]
    values = values[:len(periods)]
    refdata, prov_ref = get(prefix+"forecast_reference_time/0", out/"raw")
    reference = float(decode(refdata, metadata["forecast_reference_time/.zarray"]))
    assert abs(reference - run.timestamp()/3600) < 1e-7
    if not np.array_equal(periods, np.arange(1, len(periods)+1)):
        raise ValueError(f"Noncontiguous forecast hours: {periods}")
    iy, ix = loc["internal_y"],loc["internal_x"]
    # A 64x64 native grid (192 km wide) around the Alta nearest grid point.
    if min(iy,ix) < 32 or max(iy,ix)+32 > 150:
        raise ValueError("Patch crosses chunk boundary")
    patch = values[:,iy-32:iy+32,ix-32:ix+32]
    point = values[:,iy,ix]
    area = np.nanmean(values[:,iy-1:iy+2,ix-1:ix+2],axis=(1,2))
    extra = {}
    provenance = [prov_meta,prov_data,prov_period,prov_ref]
    if name == "precip_mm":
        # Match the radar agent's pooled 64x64 MRMS grid; this rectangle spans
        # two native HRRR chunks, y3/x3 and y4/x3. No extrapolation is used.
        data2, prov2 = get(prefix+spec+"/0.3.3", out/"raw")
        values2 = decode(data2,specmeta).astype("float32")[:len(periods)]
        if isinstance(fill,(float,int)):
            values2[values2 == fill] = np.nan
        joined = np.concatenate([values2,values],axis=1)
        lats = 41.85 - np.arange(64)*.04
        lons = -112.90 + np.arange(64)*.04
        lon2,lat2 = np.meshgrid(lons,lats)
        xx,yy = Proj(**PROJ)(lon2,lat2)
        cx,cy = (xx-X0)/DX-450,(yy-Y0)/DX-450
        assert np.all((cx>=0)&(cx<149)&(cy>=0)&(cy<299))
        extra["mrms64"] = np.stack([map_coordinates(v,[cy,cx],order=1,mode="constant",cval=np.nan) for v in joined])
        extra["mrms_latitudes"] = lats
        extra["mrms_longitudes"] = lons
        provenance.append(prov2)
    return dict(periods=periods,patch=patch,point=point,mean3x3=area,extra=extra,
                metadata=dict(name=name, field=spec, units=units, units_basis=units_basis, attrs=attrs,
                              zarray=specmeta, provenance=provenance))


def fetch_run(run, out=OUT, names=None):
    run = pd.Timestamp(run)
    run = run.tz_localize("UTC") if run.tzinfo is None else run.tz_convert("UTC")
    names = names or DEFAULT_FIELDS
    folder = out/"runs"/run.strftime("%Y%m%dT%H00Z")
    folder.mkdir(parents=True, exist_ok=True)
    saved, failed = {}, {}
    for name in names:
        try:
            result = field(run, name, out)
            np.savez_compressed(folder/(name+".npz"), periods=result["periods"],
                                patch=result["patch"],point=result["point"],
                                mean3x3=result["mean3x3"],**result["extra"])
            saved[name] = result["metadata"]
        except Exception as exc:
            failed[name] = {"type":type(exc).__name__,"message":str(exc)}
    rows = []
    for name in saved:
        data = np.load(folder/(name+".npz"))
        for i,lead in enumerate(data["periods"]):
            rows.append(dict(run_time_utc=run.isoformat(),
                             assumed_available_time_utc=(run+pd.Timedelta(hours=2)).isoformat(),
                             valid_time_utc=(run+pd.Timedelta(hours=float(lead))).isoformat(),
                             forecast_hour=float(lead), variable=name,
                             point_value=float(data["point"][i]),
                             mean3x3_value=float(data["mean3x3"][i])))
    pd.DataFrame(rows).to_csv(folder/"hourly_long.csv",index=False)
    manifest = dict(run_time_utc=run.isoformat(), forecast_only=True,
                    publication_buffer_hours=2, location=location(),
                    fields=saved, missing_fields=failed,
                    download_check_utc=datetime.now(timezone.utc).isoformat())
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest


def consolidate(out=OUT):
    files = sorted(p for p in (out/"runs").glob("*/hourly_long.csv")
                   if (p.parent/"manifest.json").exists())
    frames = [pd.read_csv(p) for p in files if p.stat().st_size>1]
    if frames:
        df=pd.concat(frames,ignore_index=True).sort_values(["run_time_utc","valid_time_utc","variable"])
        df.to_csv(out/"hourly_long.csv",index=False)
        wide=df.pivot(index=["run_time_utc","assumed_available_time_utc","valid_time_utc","forecast_hour"],
                 columns="variable",values="point_value").reset_index()
        wide.to_csv(out/"hourly_point.csv",index=False)
        df.pivot(index=["run_time_utc","assumed_available_time_utc","valid_time_utc","forecast_hour"],
                 columns="variable",values="mean3x3_value").reset_index().to_csv(out/"hourly_mean3x3.csv",index=False)
        daily=[]
        for rt, group in wide.groupby("run_time_utc"):
            run=pd.Timestamp(rt)
            if run.hour != 6:
                continue
            day=group[group.forecast_hour.between(6,29)].sort_values("forecast_hour")
            row=dict(date=str(run.date()),run_time_utc=rt,
                     target_start_utc=(run+pd.Timedelta(hours=5)).isoformat(),
                     target_end_utc=(run+pd.Timedelta(hours=29)).isoformat(),
                     complete_24h=len(day)==24 and set(DEFAULT_FIELDS).issubset(day.columns)
                     and not day.reindex(columns=DEFAULT_FIELDS).isna().any().any())
            for name in DEFAULT_FIELDS:
                if name not in day or len(day)!=24 or day[name].isna().any():
                    continue
                values=day[name].to_numpy()
                if name in ["precip_mm","snow_water_mm"]:
                    row[name+"_24h"]=float(values.sum())
                    row[name+"_first3h"]=float(values[:3].sum())
                else:
                    row[name+"_mean"]=float(values.mean())
                    row[name+"_min"]=float(values.min())
                    row[name+"_max"]=float(values.max())
            if {"wind_u_ms","wind_v_ms"}.issubset(day.columns):
                speed=np.hypot(day.wind_u_ms,day.wind_v_ms)
                row["wind_speed_ms_mean"]=float(speed.mean())
                row["wind_speed_ms_max"]=float(speed.max())
            daily.append(row)
        pd.DataFrame(daily).to_csv(out/"daily_features.csv",index=False)
    manifests=[json.loads(p.read_text()) for p in sorted((out/"runs").glob("*/manifest.json"))]
    summary=dict(source="University of Utah HRRR Zarr (operational NOAA HRRR forecasts)",
        source_docs="https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/python_data_loading.html",
        archive_errata="https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/zarr_variables.html",
        location=location(),projection=PROJ,grid_origin_m=[X0,Y0],grid_spacing_m=DX,
        patch_shape=[64,64],point_index_in_patch=[32,32],runs=len(manifests),
        total_raw_bytes=sum(p.stat().st_size for p in (out/"raw").rglob("*") if p.is_file()),
        timing="Forecast initializations and valid times preserved; assume availability two hours after initialization. Accumulation variables describe preceding one-hour interval ending at valid_time_utc.",
        warnings=["Missing chunks are missing, not zero. Archived Zarr conversion has documented missing-field bugs.",
                  "Temperature etc may be float16 before June2024; source dtype retained in manifests.",
                  "Grid-point values have no elevation downscaling. Wind components are native HRRR grid-relative unless source documentation specifies otherwise.",
                  "ASNOW is model-forecast new-snow depth, not an observed snowfall label.",
                  "ASNOW is excluded from default acquisition because its split Zarr time metadata needs further GRIB verification. WEASD_1hr_acc_fcst is accumulated new-snow water, not the instantaneous snowpack WEASD field.",
                  "A fixed two-hour availability buffer is an assumption; archive does not provide historical original publication times."],
        failures=[dict(run=m["run_time_utc"],missing=m["missing_fields"]) for m in manifests if m["missing_fields"]])
    cohort_file=out.parent/"radar/cohort_dates.csv"
    if cohort_file.exists() and (out/"daily_features.csv").exists():
        cohort=pd.read_csv(cohort_file,dtype={"case_id":str})
        status=pd.read_csv(out/"daily_features.csv")[["date","complete_24h"]]
        cohort=cohort.merge(status,on="date",how="left",validate="one_to_one")
        cohort["complete_24h"]=cohort.complete_24h.fillna(False).astype(bool)
        cohort.to_csv(out/"cohort_completeness.csv",index=False)
        summary["historical_cohort"]={"requested":len(cohort),
            "complete_24h":int(cohort.complete_24h.sum()),
            "by_split":cohort.groupby("split").complete_24h.agg(["count","sum"]).to_dict("index")}
    (out/"manifest.json").write_text(json.dumps(summary,indent=2))
    return summary


def audit_grib(run_text, lead=6, out=OUT):
    """Independent reference: NOAA original GRIB, ecCodes, nearest-grid search."""
    import eccodes as ec
    run=pd.Timestamp(run_text)
    run=run.tz_localize("UTC") if run.tzinfo is None else run.tz_convert("UTC")
    folder=out/"audit"/run.strftime("%Y%m%dT%H00Z")
    folder.mkdir(parents=True,exist_ok=True)
    base=run.strftime("https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.%Y%m%d/conus/hrrr.t%Hz.wrfsfc")+f"f{lead:02d}.grib2"
    idxfile=folder/f"f{lead:02d}.idx"
    if not idxfile.exists():
        rr=requests.get(base+".idx",timeout=60);rr.raise_for_status();idxfile.write_text(rr.text)
    lines=idxfile.read_text().splitlines()
    selectors={"precip_mm":f":APCP:surface:{lead-1}-{lead} hour acc",
               "snow_water_mm":f":WEASD:surface:{lead-1}-{lead} hour acc",
               "temperature_k":":TMP:2 m above ground:",
               "wind_u_ms":":UGRD:10 m above ground:"}
    results=[]
    for name,selector in selectors.items():
        j=next(j for j,l in enumerate(lines) if selector in l)
        start=int(lines[j].split(":")[1]);end=int(lines[j+1].split(":")[1])-1
        path=folder/(name+f"_f{lead:02d}.grib2")
        if not path.exists():
            rr=requests.get(base,headers={"Range":f"bytes={start}-{end}"},timeout=90)
            rr.raise_for_status()
            assert rr.status_code==206 and len(rr.content)==end-start+1
            path.write_bytes(rr.content)
        with path.open("rb") as handle:
            gid=ec.codes_grib_new_from_file(handle)
        nearest=dict(ec.codes_grib_find_nearest(gid,LAT,LON,npoints=1)[0])
        keys=["shortName","units","startStep","endStep","dataDate","dataTime","validityDate","validityTime"]
        metadata={key:ec.codes_get(gid,key) for key in keys}
        if name=="wind_u_ms":
            metadata["uvRelativeToGrid"]=ec.codes_get(gid,"uvRelativeToGrid")
        ec.codes_release(gid)
        source=out/"runs"/run.strftime("%Y%m%dT%H00Z")
        saved=np.load(source/(name+".npz"))
        i=list(saved["periods"]).index(lead)
        stored=float(saved["point"][i])
        manifest=json.loads((source/"manifest.json").read_text())
        dtype=np.dtype(manifest["fields"][name]["zarray"]["dtype"])
        expected=float(np.asarray(nearest["value"],dtype=dtype))
        assert stored==expected,(name,stored,expected)
        assert metadata["endStep"]==lead
        assert metadata["startStep"] == (lead-1 if name in ["precip_mm","snow_water_mm"] else lead)
        assert abs(nearest["lat"]-location()["grid_latitude"])<1e-6
        results.append(dict(variable=name,run=run.isoformat(),lead=lead,
            original_grib_value=nearest["value"],stored_value=stored,
            exact_after_source_precision_conversion=True,nearest=nearest,
            metadata=metadata,source_url=base,byte_range=[start,end],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (folder/f"comparison_f{lead:02d}.json").write_text(json.dumps(results,indent=2))
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",action="append",help="Explicit UTC initialization, ISO format")
    p.add_argument("--dates",help="CSV with date column, interpreted as initialization date")
    p.add_argument("--date-column",default="date")
    p.add_argument("--run-hour",type=int,default=6)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--fields",nargs="+",default=DEFAULT_FIELDS)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--audit-run",help="Independently verify a saved run against original NOAA GRIB")
    p.add_argument("--consolidate-only",action="store_true")
    a=p.parse_args()
    if a.audit_run:
        print(json.dumps(audit_grib(a.audit_run,out=a.out),indent=2))
        return
    if a.consolidate_only:
        print(json.dumps(consolidate(a.out),indent=2))
        return
    runs=list(a.run or [])
    if a.dates:
        ds=pd.to_datetime(pd.read_csv(a.dates)[a.date_column],utc=True)
        runs.extend((ds.dt.normalize()+pd.Timedelta(hours=a.run_hour)).astype(str))
    runs=sorted(set(runs))
    if not runs:
        p.error("Pass --run or --dates")
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(fetch_run,r,a.out,a.fields):r for r in runs}
        for done,future in enumerate(as_completed(futures),1):
            m=future.result()
            print(json.dumps(dict(done=done,total=len(runs),run=futures[future],
                                  fields=len(m["fields"]),missing=m["missing_fields"])),flush=True)
    print(json.dumps(consolidate(a.out),indent=2))


if __name__ == "__main__":
    main()
