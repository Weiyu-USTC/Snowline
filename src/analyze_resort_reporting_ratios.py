"""Gap-aware comparison of resort reports and archived NOAA new-snow observations.

This is a source audit, not a resort calibration model. Run from any directory:
    python3 src/analyze_resort_reporting_ratios.py
"""
from __future__ import annotations

import calendar
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/east_alta_research"
OUT = ROOT / "outputs/multi-resort/reporting-ratios"
ACCESS_DATE = "2026-11-04"
SNOW = "snowfall_inches"
SOURCES = {
    "ghcn_documentation": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt",
    "jay_noaa": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00434189.dly",
    "mansfield_noaa": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00435416.dly",
    "jay_resort": "https://jaypeakresort.com/skiing-riding/mountain",
    "stowe_third_party": "https://www.onthesnow.com/northeast/stowe-mountain-resort/historical-snowfall",
    "uvm_stowe_measurements": "https://www.uvm.edu/water/news/talking-snow-stowe-mountain-resort",
    "noaa_jay_station_history": "https://www.ncei.noaa.gov/monitoring-content/extremes/scec/reports/20210315-Vermont-24-Hour-Snowfall-Rejected.pdf",
    "jay_station_metadata": "https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND%3AUSC00434189/detail",
}
WINDOWS = {"Oct-May": (10, 5), "Sep-Jun": (9, 6), "Nov-Apr": (11, 4)}


def load_and_check(name: str, station: str) -> pd.DataFrame:
    """Verify all derived SNOW values/flags against the raw fixed-width archive."""
    flags = {f"{SNOW}_{suffix}": "string" for suffix in ("measurement_flag", "quality_flag", "source_flag")}
    d = pd.read_csv(DATA / "derived_noaa" / f"{name}_daily.csv", parse_dates=["date"], dtype=flags, low_memory=False)
    d = d.set_index("date").sort_index()
    assert d.index.is_unique
    raw = DATA / f"{name}_{station}.dly"
    checked = 0
    for line in raw.read_text().splitlines():
        if line[17:21] != "SNOW":
            continue
        year, month = int(line[11:15]), int(line[15:17])
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            at = 21 + (day - 1) * 8
            value = int(line[at:at + 5])
            date = pd.Timestamp(year, month, day)
            if value == -9999:
                assert date not in d.index or pd.isna(d.loc[date, SNOW])
                continue
            assert np.isclose(d.loc[date, SNOW], value / 25.4)
            for suffix, offset in [("measurement_flag", 5), ("quality_flag", 6), ("source_flag", 7)]:
                got = d.loc[date, f"{SNOW}_{suffix}"]
                assert ("" if pd.isna(got) else str(got)) == line[at + offset].strip()
            checked += 1
    d.attrs["raw_snow_records_checked"] = checked
    return d


def compare(d: pd.DataFrame, totals: pd.DataFrame, site: str, station: str, source_class: str) -> pd.DataFrame:
    records = []
    for item in totals.itertuples():
        y, reported = int(item.season_start), float(item.snowfall_inches)
        for window, (first, last) in WINDOWS.items():
            ix = pd.date_range(f"{y}-{first:02}-01", f"{y+1}-{last:02}-{calendar.monthrange(y+1, last)[1]}")
            block = d.reindex(ix)
            values = block[SNOW]
            quality_ok = block[f"{SNOW}_quality_flag"].fillna("").eq("")
            presumed = block[f"{SNOW}_measurement_flag"].fillna("").eq("P")
            usable = values.notna() & quality_ok
            measured = usable & ~presumed
            observed_sum = float(values.where(usable).sum(min_count=1))
            n = len(ix)
            clean = int(usable.sum())
            strict = int(measured.sum())
            records.append({
                "site": site, "station_id": station, "report_source_class": source_class,
                "season": f"{y}/{str(y+1)[-2:]}", "season_start": y,
                "window": window, "window_start": str(ix[0].date()), "window_end": str(ix[-1].date()),
                "resort_reported_inches": reported,
                "noaa_observed_sum_inches": observed_sum,
                "expected_days": n, "quality_usable_days_including_P": clean,
                "missing_or_quality_excluded_days": n-clean,
                "missing_value_days": int(values.isna().sum()),
                "quality_excluded_days": int((values.notna() & ~quality_ok).sum()),
                "presumed_zero_P_days": int((usable & presumed).sum()),
                "usable_days_excluding_P": strict,
                "coverage_including_P": clean/n,
                "coverage_excluding_P": strict/n,
                "complete_including_P": clean == n,
                "complete_excluding_P": strict == n,
                "coverage_at_least_95pct_including_P": clean/n >= .95,
                "coverage_at_least_95pct_excluding_P": strict/n >= .95,
                "reported_to_observed_sum_ratio": reported/observed_sum if observed_sum > 0 else np.nan,
                "ratio_status": "observed-denominator comparison; no matching-site calibration; missing snowfall makes this an upper bound conditional on reported values and fixed window" if clean else "no NOAA snowfall overlap",
            })
    return pd.DataFrame(records)


def table_html(d: pd.DataFrame, columns: dict[str, str]) -> str:
    shown = d[list(columns)].rename(columns=columns).copy()
    for c in shown:
        if pd.api.types.is_float_dtype(shown[c]):
            shown[c] = shown[c].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}" if "Ratio" in c else f"{v:.1f}")
    return shown.to_html(index=False, border=0, escape=True, classes="data")


def link(key: str, label: str) -> str:
    return f'<a href="{html.escape(SOURCES[key])}">{html.escape(label)}</a>'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jay_daily = load_and_check("jay_peak", "USC00434189")
    mansfield_daily = load_and_check("mount_mansfield", "USC00435416")
    jay_totals = pd.read_csv(DATA / "jay_peak_official_season_totals.csv")
    jay_totals["season_start"] = jay_totals.season.str[:4].astype(int)
    # These five numbers were transcribed from the cited third-party table, not
    # verified against a resort-hosted historical archive. Keep source class explicit.
    stowe_totals = pd.DataFrame({
        "season_start": [2012, 2013, 2014, 2015, 2016],
        "snowfall_inches": [218, 231, 249, 146, 330],
        "source_class": "third-party backup, not verified official Stowe archive",
        "source_url": SOURCES["stowe_third_party"], "access_date": ACCESS_DATE,
    })
    stowe_totals.to_csv(OUT / "stowe_third_party_backup_totals.csv", index=False)
    jay = compare(jay_daily, jay_totals, "Jay Peak", "USC00434189", "official resort season table")
    stowe = compare(mansfield_daily, stowe_totals, "Stowe versus Mount Mansfield NOAA", "USC00435416", "third-party backup; unverified resort attribution")
    all_rows = pd.concat([jay, stowe], ignore_index=True).sort_values(["site", "season_start", "window"])
    all_rows.to_csv(OUT / "all_windows_coverage_and_ratios.csv", index=False)
    primary = all_rows[all_rows.window.eq("Oct-May")].copy()
    primary.to_csv(OUT / "october_may_comparisons.csv", index=False)
    jp = primary[primary.site.eq("Jay Peak")]
    sp = primary[primary.station_id.eq("USC00435416")]
    j95 = jp[jp.coverage_at_least_95pct_including_P].copy()
    s95 = sp[sp.coverage_at_least_95pct_including_P].copy()
    assert len(j95) == 3 and not jp.complete_including_P.any()
    assert len(s95) == 5 and not sp.complete_including_P.any()
    assert all_rows.loc[all_rows.complete_including_P, "missing_or_quality_excluded_days"].eq(0).all()
    # Hold the cohort fixed for sensitivity: changing the window also changes its
    # coverage fraction, which must not silently change the seasons being compared.
    cohorts = {"Jay Peak": set(j95.season_start), "Stowe versus Mount Mansfield NOAA": set(s95.season_start)}
    fixed = all_rows[all_rows.apply(lambda r: r.season_start in cohorts[r.site], axis=1)].copy()
    fixed.to_csv(OUT / "window_sensitivity_fixed_cohort.csv", index=False)
    summary = {
        "access_date": ACCESS_DATE,
        "no_calibration_or_inflation_factor_endorsed": True,
        "noaa_archive_is_not_proof_of_independent_observer": True,
        "jay_primary": {
            "official_seasons": len(jay_totals), "seasons_with_any_NOAA_overlap": int((jp.quality_usable_days_including_P > 0).sum()),
            "complete_seasons_including_P": int(jp.complete_including_P.sum()),
            "complete_seasons_excluding_P": int(jp.complete_excluding_P.sum()),
            "seasons_at_95pct_including_P": j95.season.tolist(),
            "seasons_at_95pct_excluding_P": jp.loc[jp.coverage_at_least_95pct_excluding_P, "season"].tolist(),
            "ratio_min_at_95pct": float(j95.reported_to_observed_sum_ratio.min()),
            "ratio_max_at_95pct": float(j95.reported_to_observed_sum_ratio.max()),
            "pearson_r_diagnostic_only_3_incomplete_seasons": float(j95.resort_reported_inches.corr(j95.noaa_observed_sum_inches)),
        },
        "stowe_backup_primary": {
            "verified_official_history_found": False, "third_party_overlapping_seasons": len(sp),
            "complete_seasons": int(sp.complete_including_P.sum()),
            "ratio_min": float(s95.reported_to_observed_sum_ratio.min()),
            "ratio_max": float(s95.reported_to_observed_sum_ratio.max()),
            "pearson_r_diagnostic_only_5_incomplete_seasons": float(s95.resort_reported_inches.corr(s95.noaa_observed_sum_inches)),
        },
        "raw_SNOW_records_independently_verified": {
            "jay": jay_daily.attrs["raw_snow_records_checked"],
            "mansfield": mansfield_daily.attrs["raw_snow_records_checked"],
        },
        "sources": SOURCES,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    columns = {
        "season": "Season", "resort_reported_inches": "Reported in",
        "noaa_observed_sum_inches": "NOAA observed in", "missing_or_quality_excluded_days": "Missing / excluded days",
        "presumed_zero_P_days": "Presumed zero days", "reported_to_observed_sum_ratio": "Ratio ≤*",
    }
    j_table = table_html(j95, columns)
    s_table = table_html(s95, columns)
    sensitivity = table_html(fixed, {
        "site": "Comparison", "season": "Season", "window": "Window",
        "noaa_observed_sum_inches": "NOAA observed in", "missing_or_quality_excluded_days": "Missing / excluded days",
        "reported_to_observed_sum_ratio": "Ratio ≤*",
    })
    all_jay_table = table_html(jp[jp.quality_usable_days_including_P > 0], columns)
    result = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NOAA and resort snowfall: a gap-aware comparison</title>
<style>
body{{font:17px/1.6 system-ui,-apple-system,sans-serif;margin:0;color:#213547;background:#f4f7fa}}
main{{max-width:1100px;margin:0 auto;padding:44px 32px 80px;background:white}}
h1{{font-size:34px;line-height:1.2;max-width:850px}}h2{{margin-top:36px;font-size:23px}}
.eyebrow{{color:#55718a;font-weight:650;font-size:13px;letter-spacing:.08em;text-transform:uppercase}}
.note{{background:#edf4fa;border-left:4px solid #4077a3;padding:16px 20px;margin:24px 0}}
.caution{{background:#fff5df;border-left:4px solid #bd8b22;padding:16px 20px;margin:24px 0}}
.data{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}
.data th,.data td{{padding:9px 12px;border-bottom:1px solid #dfe7ee;text-align:right}}
.data th:first-child,.data td:first-child{{text-align:left}}.data th{{background:#eff4f7;white-space:normal}}
a{{color:#175f96}}.scroll{{overflow:auto}}.small{{font-size:14px;color:#496174}}details{{margin:22px 0}}
</style><main><div class="eyebrow">Source audit · accessed {ACCESS_DATE}</div>
<h1>How different are resort reports from NOAA snowfall?</h1>
<p><b>The available overlap does not support a trustworthy “bloating rate.”</b> Jay's official reported totals are larger than the snowfall recorded on available NOAA days, but every overlapping October–May season has gaps. Stowe's accessible comparison comes from a third party, and its reporting site differs from the Mansfield station. No multiplication or discount factor from this comparison is applied to forecasts.</p>
<div class="note"><b>What the numbers mean.</b> Each NOAA figure below is a sum of quality-accepted observations on available days, not an estimated complete-season total. Missing or quality-excluded snowfall remains unknown. GHCN <code>P</code> entries are “missing presumed zero”; they are counted separately and are not treated as independently observed dry days. {link('ghcn_documentation', 'NOAA flag and unit definitions')}.</div>
<h2>Jay Peak: official resort totals, incomplete station overlap</h2>
<p>The {link('jay_resort', 'official resort table')} contains 22 seasons (2004/05–2025/26). The downloaded {link('jay_noaa', 'NOAA Jay Peak archive')} has usable new-snow observations overlapping 10 of them. <b>Zero October–May overlap seasons are complete.</b> Three reach 95% coverage if NOAA's presumed-zero entries are included:</p>
<div class="scroll">{j_table}</div>
<p class="small">* The displayed number is reported total / observed NOAA sum. Because additional snowfall on missing days can only increase the denominator, this is an upper bound on the ratio using a completed NOAA window, conditional on the existing values being accurate. It is not an upper bound on intentional exaggeration and does not establish that the two sources measure the same site or season.</p>
<p>The observed-denominator ratios span <b>{j95.reported_to_observed_sum_ratio.min():.2f}–{j95.reported_to_observed_sum_ratio.max():.2f}</b>. Strictly excluding presumed-zero days leaves only 2010/11 at 95% coverage. Three incomplete seasons cannot establish a stable site-transfer relationship; their descriptive correlation is {summary['jay_primary']['pearson_r_diagnostic_only_3_incomplete_seasons']:.2f}, a particularly unstable estimate at this sample size.</p>
<p>{link('noaa_jay_station_history', 'NOAA’s station-history report (page 16)')} places the cooperative station at Jay Peak Ski Resort near 1,875 ft and records its closure in June 2014 when the observer could no longer report. Current {link('jay_station_metadata', 'station metadata')} list 1,840 ft (560.8 m). The precise correspondence to the resort’s historic reporting stake is not verified. NOAA archive hosting also does not imply an independent observer: the cooperative station was at the resort.</p>
<h2>Stowe and Mount Mansfield: useful backup, not an official-rate estimate</h2>
<p>A resort-hosted historical annual series was not verified. The following five reported totals are from {link('stowe_third_party', 'OnTheSnow’s Stowe history')} and are explicitly retained as an unverified third-party backup. Their comparator is {link('mansfield_noaa', 'NOAA Mount Mansfield snowfall')}, not the Stowe resort stake.</p>
<div class="scroll">{s_table}</div>
<p>All five October–May station windows have at least 95% coverage, but each has 1–5 missing or excluded days. The observed-denominator ratios span <b>{s95.reported_to_observed_sum_ratio.min():.2f}–{s95.reported_to_observed_sum_ratio.max():.2f}</b>; the correlation of these five incomplete sums with the third-party totals is <b>{summary['stowe_backup_primary']['pearson_r_diagnostic_only_5_incomplete_seasons']:.2f}</b>. That is descriptive evidence of shared year-to-year variation, not validated forecast skill or an inflation estimate.</p>
<p>{link('uvm_stowe_measurements', 'UVM’s January 2026 field visit')} documents Stowe’s 24-hour accumulation and snowpack stakes at roughly 3,000 ft, and the Mansfield snow stake around 3,900 ft. The NOAA station metadata place Mount Mansfield at 1,204 m (about 3,950 ft). Elevation, wind exposure, observation time and board-clearing practice can change measured snowfall. The 2026 site description does not prove that all historical reporting practices were unchanged.</p>
<h2>Window sensitivity and audit rules</h2>
<p>October–May is the primary comparison. September–June and November–April are sensitivity windows; the resort's exact annual counting dates were not established. The table below holds the primary 95%-coverage cohort fixed so a changed window cannot silently change which years are compared. The raw CSV also includes coverage under both treatments of presumed-zero entries.</p>
<details><summary>Show window sensitivity ({len(fixed)} comparisons)</summary><div class="scroll">{sensitivity}</div></details>
<details><summary>Show all 10 Jay overlap seasons, including poor coverage</summary><div class="scroll">{all_jay_table}</div></details>
<div class="caution"><b>Why no correction factor is fitted:</b> Different measurement locations, incomplete denominators, uncertain reporting windows and source provenance are confounded. Applying a single resort/NOAA ratio to daily predictions would convert those unknowns into a false precision. A defensible calibration needs simultaneous daily new-snow observations at the same or well-characterized sites, harmonized observation periods, and withheld-season validation.</div>
<p>Quality procedure: SNOW is converted from millimeters to inches using 25.4 mm/in; every nonmissing quality flag excludes that value; unavailable days stay missing; presumed-zero entries are explicitly counted; snow depth is never substituted for snowfall. All {summary['raw_SNOW_records_independently_verified']['jay']:,} Jay and {summary['raw_SNOW_records_independently_verified']['mansfield']:,} Mansfield nonmissing SNOW records in the derived files were independently checked against the original fixed-width source values and flags.</p>
<h2>Files</h2><ul>
<li><a href="october_may_comparisons.csv">Primary window comparisons</a></li>
<li><a href="all_windows_coverage_and_ratios.csv">All windows, coverage and ratios</a></li>
<li><a href="window_sensitivity_fixed_cohort.csv">Fixed-cohort window sensitivity</a></li>
<li><a href="stowe_third_party_backup_totals.csv">Stowe third-party backup totals and provenance</a></li>
<li><a href="summary.json">Machine-readable summary and sources</a></li>
</ul><p class="small">These comparisons audit data sources. They do not prove deliberate reporting bias, establish a resort snowfall correction, or replace evaluation of the forecast models.</p></main></html>'''
    (OUT / "report.html").write_text(result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
