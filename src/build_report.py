#!/usr/bin/env python3
"""Build the portable, static research report from saved analysis outputs.

Run after analysis.py: python src/build_report.py
The HTML embeds its figures and needs no network connection to render. Source
citations require a connection; optional local data/code links require the repo.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path


CSS = """
:root{--ink:#142b3a;--muted:#526b79;--teal:#176b87;--pale:#eaf3f5;--line:#d9e4e8;--orange:#a64e25}
*{box-sizing:border-box}body{margin:0;background:#eef2f3;color:var(--ink);font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1160px;margin:32px auto;background:#fff;box-shadow:0 8px 35px #142b3a0d}
header{padding:52px 64px 40px;background:var(--ink);color:white;border-top:8px solid #77b7c4}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#a8d3de}
h1{font-size:clamp(32px,4.2vw,51px);line-height:1.08;letter-spacing:-.035em;margin:18px 0 22px;max-width:850px}
.dek{font-size:21px;line-height:1.5;max-width:910px;color:#e2edf1;margin:0}.date{font-size:13px;color:#b9cdd5;margin-top:24px}
.body{padding:36px 64px 52px}h2{font-size:26px;line-height:1.2;letter-spacing:-.02em;margin:38px 0 15px}h3{font-size:19px;margin:24px 0 10px}
p{margin:0 0 16px}a{color:var(--teal);text-decoration-thickness:1px;text-underline-offset:3px}strong{font-weight:700}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:0 0 28px}.card{background:var(--pale);border-top:3px solid var(--teal);padding:20px}
.card .label{font-size:12px;text-transform:uppercase;letter-spacing:.07em;display:block;color:var(--muted)}.card b{font-size:34px;letter-spacing:-.03em;display:block;margin:6px 0}.card small{font-size:13px;line-height:1.4;display:block;color:var(--muted)}
.callout{border-left:4px solid var(--orange);background:#fcf5ef;padding:20px 24px;margin:24px 0}.callout p:last-child{margin-bottom:0}
figure{margin:30px -30px;background:#f5f8fc;padding:15px 10px;border:1px solid var(--line)}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:var(--muted);padding:12px 10px 0}
.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.45;font-variant-numeric:tabular-nums}th,td{padding:11px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}thead th{background:var(--pale);font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted)}tbody th{font-weight:600}td.number{text-align:right;white-space:nowrap}th.number{text-align:right}.baseline{background:#f5fafb}.small,.note{font-size:14px;color:var(--muted)}.note{margin-top:-5px}ul,ol{padding-left:23px;margin:0 0 18px}li{margin:9px 0}.tag{font-size:12px;color:var(--orange);font-weight:700;text-transform:uppercase;letter-spacing:.05em}.downloads{border-top:1px solid var(--line);padding-top:20px;margin-top:30px;font-size:14px}.downloads a{display:inline-block;margin-right:18px;margin-bottom:8px}footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:18px;margin-top:22px}code{font-size:.9em;background:#eef3f5;padding:2px 5px;border-radius:3px}
@media(max-width:760px){main{margin:0;box-shadow:none}header{padding:34px 24px}.body{padding:26px 24px 38px}.dek{font-size:19px}.cards{gap:10px}.card{padding:14px}.card b{font-size:27px}.card .label{font-size:10px}.card small{font-size:12px}figure{margin:24px -15px;padding:8px 4px}table{font-size:13px}th,td{padding:9px 8px}}
@media(max-width:460px){.cards{grid-template-columns:1fr}.card{display:grid;grid-template-columns:1fr 1fr;column-gap:10px}.card .label{grid-column:1}.card b{grid-column:1;margin:3px 0}.card small{grid-column:2;grid-row:1/3;align-self:center}}
@media print{@page{size:A4;margin:15mm}body{background:white;font-size:10.5pt;line-height:1.45}main{margin:0;max-width:none;box-shadow:none}header{padding:20px 24px;color:var(--ink);background:#edf3f5;border-top:4px solid var(--teal)}.eyebrow,.date,.dek{color:var(--muted)}h1{font-size:29pt}.dek{font-size:13pt}.body{padding:12px 0}h2{font-size:17pt;margin-top:24px;break-after:avoid}.cards{gap:10px}.card{padding:12px}.card b{font-size:22pt}table{font-size:9pt}th,td{padding:7px 8px}thead{display:table-header-group}tr,figure,.callout,.cards{break-inside:avoid}figure{margin:18px 0}figcaption,.small,.note{font-size:9pt}a{color:var(--ink)}.downloads{display:none}}
"""

LABELS = {
    "climatology": "Historical mean",
    "october_linear": "October only · linear",
    "enso_linear": "JAS ENSO only · linear",
    "october_enso_ridge": "October + JAS ENSO · ridge",
}


def f(value, digits=1):
    return f"{value:.{digits}f}"


def interval(values, digits=2):
    return f"{f(values[0], digits)} to {f(values[1], digits)}"


def figure(path: Path, alt: str, caption: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt, quote=True)}"><figcaption>{caption}</figcaption></figure>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: PROJECT/outputs/report.html")
    args = parser.parse_args()
    root = args.project_dir.resolve()
    output = args.output or root / "outputs/report.html"
    result_path = root / "outputs/results.json"
    r = json.loads(result_path.read_text())
    c = r["copper"]
    s = r["copper_sensitivity"]
    research_files = sorted((root / "research").glob("*.txt")) + sorted((root / "research").glob("*.json"))
    research = "\n".join(path.read_text() for path in research_files)
    audit = json.loads((root / "research/offseason_discrepancy_audit.json").read_text())
    enso = json.loads((root / "research/research_enso.json").read_text())
    metrics = {item["model"]: item for item in c["backtest_metrics"]}
    oct_association = c["october_vs_future"]
    strong = c["strong_oni_djf"]

    def cite(label: str, url: str) -> str:
        if url not in research:
            raise ValueError(f"Citation not present in reviewed research notes: {url}")
        return f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'

    def table(headers, rows):
        head = "".join(f"<th scope=\"col\">{h}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    correlations = table(
        ["Copper predictor → November–April", "Pearson r", "95% CI", "Two-sided p"],
        [(label, f(c[key]["pearson_r"], 2), interval(c[key]["pearson_ci95"]), f(c[key]["pearson_p_two_sided"], 3))
         for label, key in [("October snowfall", "october_vs_future"), ("JAS ONI · available before season", "jas_oni_vs_future"),
                            ("DJF ONI · retrospective only", "djf_oni_vs_future_retrospective")]])
    copper_metrics = table(
        ["Model", "Test seasons", "MAE · inches", "RMSE · inches"],
        [(LABELS[item["model"]], item["n_test"], f(item["mae_inches"]), f(item["rmse_inches"])) for item in c["backtest_metrics"]])
    station_rows = []
    for label, key in [("Dillon 1 E", "dillon"), ("Climax", "climax"), ("Dillon · recent subset", "dillon_recent")]:
        data = r[key]
        model_metrics = {item["model"]: item for item in data["backtest_metrics"]}
        station_rows.append((f'<strong>{label}</strong><br><span class="small">{data["first_season"]}–{data["last_season"]}</span>',
                             f'{data["n"]} / {model_metrics["climatology"]["n_test"]}',
                             f(data["october_vs_future"]["pearson_r"], 2),
                             *[f'{f(model_metrics[name]["mae_inches"])} / {f(model_metrics[name]["rmse_inches"])}' for name in LABELS]))
    stations_table = table(["Station / eligible span", "Seasons / tests", "October r", "Mean", "October", "JAS ENSO", "Combined"], station_rows)
    discrepancy_rows = []
    with (root / "outputs/copper_data_audit.csv").open() as source:
        for row in csv.DictReader(source):
            if float(row["reported_minus_observed"]) != 0:
                discrepancy_rows.append((row["season"], f(float(row["observed_month_sum"]), 0), row["reported_total"], f(float(row["reported_minus_observed"]), 0)))
    discrepancies = table(["Season", "Listed months · inches", "Printed total · inches", "Unallocated gap"], discrepancy_rows)
    off_climax = next(row for row in audit["monthly_station_context"] if row["station_name"] == "Climax" and row["year"] == 2025 and row["month"] == 9)
    pi = c["climatology_prediction_interval80"]
    difference_ci = strong["welch_ci95"]
    threshold_sensitivity = c["strong_oni_without_rounding_boundary"]
    october_gain = metrics["october_linear"]
    main_figure = figure(root / "outputs/copper_findings.png", "October correlation, retrospective ENSO comparison, and chronological forecast error at Copper Mountain.",
                         "Copper’s monthly data, NOAA ONI, and forecasts evaluated on later seasons. The winter ENSO panel describes association after the season; it is not an issue-time forecast input.")
    station_figure = figure(root / "outputs/station_comparison.png", "Separate October and following-season snowfall observations at Copper and Dillon, with different vertical scales.",
                            "Different observation sites and different snowfall scales. Nearby station data expand the evidence without being relabeled as Copper observations.")

    content = f"""
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Reproducible seasonal snowfall research at Copper Mountain, with NOAA station comparisons and chronological forecast evaluation.">
<title>Copper Mountain · Seasonal snowfall research</title><style>{CSS}</style></head><body><main>
<header><div class="eyebrow">Research demo · Colorado snowfall</div><h1>Does October snow—or El Niño—predict the winter?</h1>
<p class="dek">The present evidence does not establish either as a strong predictor at Copper Mountain. A simple historical average remains the defensible default for this demo.</p>
<div class="date">Analysis: {html.escape(r['analysis_date'])} · Seasonal outlook, not a storm forecast · Exploratory findings</div></header>
<div class="body">
<div class="cards"><div class="card"><span class="label">Copper observations</span><b>{c['n']} seasons</b><small>2005/06–2024/25; one unresolved season excluded</small></div>
<div class="card"><span class="label">October → winter</span><b>r = {f(oct_association['pearson_r'],2)}</b><small>95% CI {interval(oct_association['pearson_ci95'])}; weak and uncertain</small></div>
<div class="card"><span class="label">Later-season tests</span><b>{metrics['climatology']['n_test']} winters</b><small>Historical mean RMSE: {f(metrics['climatology']['rmse_inches'])} inches</small></div></div>
<p><strong>Target:</strong> snowfall from November 1 through April 30, in inches, forecast on November 1. October is assumed fully reported by issuance. This fixed window makes years comparable and tests whether October predicts <em>subsequent</em> snow. Including October in both predictor and target creates a built-in correlation.</p>
<p class="small">These findings neither prove no physical effect nor exclude a useful predictor with better data. Twenty seasons offer limited statistical power; several analyses below are sensitivity checks, not independent confirmations.</p>

<h2>1. Repair the measurement question first</h2>
<p>The {cite('official Copper table',audit['official_resort_source'])} reproduces both screenshot inconsistencies. They originate in the source, rather than transcription. The resort identifies its Cache Site near Aerie as the measurement location.</p>
{discrepancies}
<p>All original numbers are preserved. The primary analysis sums November–April directly, excludes 2025/26 pending correction, and checks whether removing 2019/20 changes the result. Missing May cells stay missing. The source’s October-inclusion footnote also conflicts with older arithmetic; May enters printed totals from 2022/23.</p>
<p><strong>Could the missing 60 inches have fallen outside October–May?</strong> No supporting evidence was found. Climax recorded {f(off_climax['sum_of_valid_snowfall_inches'],2)} inches in September 2025; the checked July–September 2025 and June–September 2026 station months were otherwise zero, including Dillon and Vail. These are separate sites: they cannot rule out localized Copper snow. The total remains unresolved until resort daily logs or a correction explain it. {cite('NOAA Climax daily observations',off_climax['source_url'])}.</p>

<h2>2. The two hypotheses</h2>
{correlations}
<p><strong>October:</strong> the estimated linear association with later snowfall explains about {f(100*oct_association['r_squared'],0)}% of observed variation in this sample. The rank correlation is also small (Spearman ρ = {f(oct_association['spearman_rho'],2)}). Correlation with October–April rises to {f(c['october_vs_oct_apr']['pearson_r'],2)}, partly because October contributes to that total; it does not establish predictive skill.</p>
<p><strong>Strong El Niño:</strong> define the descriptive proxy as published, rounded DJF ONI ≥1.5°C, including very strong events. Only {strong['n_strong']} winters qualify: 2009/10, 2015/16 and 2023/24. Their subsequent-season mean is {f(strong['strong_mean_inches'])} inches versus {f(strong['other_mean_inches'])} for the other {strong['n_other']}. The difference is <strong>{f(strong['difference_inches'])} inches</strong>, with Welch 95% CI {interval(difference_ci,1)} and exact permutation p = {f(strong['permutation_p_two_sided'],3)}. This wide range is compatible with substantial effects in either direction.</p>
<p>DJF occurs after issuance, so it is explanatory only. Forecast models use July–September (JAS) ONI. August–October publication cannot be assumed by November 1. The screenshot’s binary labels do not encode strength or a complete ENSO classification. {cite('NOAA index definitions',enso['oni']['table_url'])}.</p>
{main_figure}

<h2>3. What the trained models actually predict</h2>
<p>Models train on earlier seasons and predict the next eligible season, expanding the training set each time. Copper starts with eight training seasons and tests 2013/14–2024/25. Scaling is fitted inside each training fold; the combined model uses fixed ridge penalty α = 10, without selecting it on test performance. Twenty annual examples do not justify training a neural weather model.</p>
{copper_metrics}
<p class="note">MAE is average absolute error; RMSE gives greater weight to large misses. Lower is better. All four models use the same {metrics['climatology']['n_test']} Copper test winters.</p>
<p>October improves MAE by only {f(october_gain['mean_mae_improvement_inches'],2)} inches; the paired-bootstrap 95% interval for improvement is {interval(october_gain['mae_improvement_bootstrap_ci95'],1)}. Its RMSE is worse than the historical mean. ENSO alone and the combined model worsen both metrics. This is insufficient evidence for a useful forecast upgrade.</p>
<div class="callout"><span class="tag">Demo estimate · not a live 2026/27 forecast</span><p>The fitted Copper default is <strong>{f(c['mean_nov_apr_inches'],0)} inches for November–April</strong>, with a model-based 80% prediction interval of <strong>{interval(pi,0)} inches</strong>. For an October–April total, add the observed October amount; May is outside this target.</p><p class="small">The Student-t interval assumes independent, stable, approximately normal seasonal totals. It is not a validated climate-change adjustment. Its historical coverage was {f(100*c['climatology_pi80_coverage'],0)}% over just {metrics['climatology']['n_test']} test winters, too few to establish calibration. The experimental regressions have no validated prediction intervals.</p></div>

<h2>4. Independent NOAA station experiments</h2>
<p>Actual daily snowfall from {cite('NOAA GHCN-Daily','https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily')} was downloaded and modeled separately. Every October–April day must be observed and pass the specified quality policy; missing values and “missing presumed zero” flags are excluded. Trace measurements remain recorded zeros. This leaves 41 Dillon seasons, 22 Climax seasons, and a 20-season recent Dillon subset.</p>
{stations_table}
<p class="note">Model cells show MAE / RMSE in inches. Spans contain gaps. Dillon begins with 15 training seasons; Climax and recent Dillon with eight. The recent subset overlaps Dillon and is not a third independent station.</p>
<p>Dillon is lower (2,767.9 m) than Climax (3,457.7 m), with different snowfall climatology and eligible periods. No station rows are pooled as independent climate years. The historical mean wins both errors in these experiments, but success at a nearby station would not itself validate a Copper forecast. {cite('NOAA station metadata','https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt')}.</p>
{station_figure}

<h2>5. How robust is this conclusion?</h2>
<p>Including unresolved 2025/26 monthly values changes October r to {f(s['include_unresolved_monthly_2025']['october_vs_future']['pearson_r'],2)}; excluding both discrepant rows gives {f(s['exclude_both_discrepant_rows']['october_vs_future']['pearson_r'],2)}. Neither establishes a strong association. Starting the Copper backtest with 10 or 12 seasons also leaves the historical mean ahead on both errors.</p>
<p>ENSO classification is especially fragile: raw DJF 2010 ONI is 1.47°C, rounded to 1.5. Removing that boundary case leaves two “strong” winters and changes the estimated difference to +{f(threshold_sensitivity['difference_inches'])} inches. The same DJF threshold applied to RONI leaves only one. NOAA adopted RONI operationally in February 2026; this project preserves ONI and RONI separately. {cite('NOAA transition notice',enso['operational_change']['url'])}.</p>
<p class="small">Indices are revised historical ERSSTv6 values, so these are pseudo-real-time hindcasts, not historical forecast vintages. Fisher correlation intervals, Welch intervals, permutation tests and the season bootstrap rely on distributional or exchangeability assumptions; serial dependence and climate shifts can undermine them. Holm-adjusted primary permutation p-values are {f(r['primary_holm_adjusted_p']['october'],3)} (October) and {f(r['primary_holm_adjusted_p']['strong_oni'],3)} (strong ONI). The reported model comparison has no untouched final holdout.</p>

<h2>6. A useful path to mountain forecasts</h2>
<p><strong>Seasonal outlooks:</strong> extend the record with consistent resort daily observations and archived issue-time seasonal ensemble forecasts, such as {cite('ECMWF SEAS5','https://www.ecmwf.int/en/forecasts/datasets/set-v')}. Train a regularized hierarchical model that shares information across mountains while retaining site effects. {cite('Copper SNOTEL 415','https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=415')} can add snowpack, temperature and precipitation context; SWE and standing depth are not direct new-snowfall labels.</p>
<p><strong>Storm forecasts are a separate task:</strong> current {cite('ECMWF AIFS','https://www.ecmwf.int/en/forecasts/dataset/aifs-machine-learning-data')} and {cite('WeatherNext','https://developers.google.com/weathernext/guides/models')} provide probabilistic weather inputs on much shorter horizons. NOAA’s {cite('National Blend of Models','https://vlab.noaa.gov/web/mdl/nbm-documentation')} provides an operational comparison. Calibrate precipitation, rain–snow phase, snow-to-liquid ratio and terrain effects locally; none of these products is established here as universally best for Copper snowfall.</p>
<p><strong>Before launching:</strong> freeze the experiment, hold out entire future years and mountains, and archive forecasts prospectively. Require improvements over climatology and relevant operational guidance, including large-storm performance and probabilistic calibration (CRPS, threshold Brier scores and interval coverage). Publish issue time, valid window, measurement site and uncertainty with every estimate.</p>

<div class="downloads"><strong>Reproducible project files</strong><br>
<a href="results.json">Full results</a><a href="copper_backtest_predictions.csv">Copper test predictions</a><a href="copper_data_audit.csv">Data audit</a><a href="trained_models.json">Trained coefficients</a><a href="../src/analysis.py">Analysis source</a><a href="../research/methods_and_data.txt">Research notes</a></div>
<footer>Static report generated from saved results and reviewed source notes. Figures are embedded; no scripts, external fonts or tracking. Local file links work when the report remains in the project’s outputs folder. Research date: {html.escape(r['analysis_date'])}.</footer>
</div></main></body></html>
"""
    provenance = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [result_path, *research_files]}
    content += "\n<!-- Input SHA256: " + json.dumps(provenance) + " -->\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Wrote {output.resolve()} ({len(content.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
