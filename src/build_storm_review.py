"""Build the literature-backed storm-event design review; no model fitting."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/vail-storm-review'


def main():
    diagnostics=json.loads((OUT/'proxy_event_diagnostics.json').read_text())
    pooled=diagnostics['pooled']
    per=pd.read_csv(OUT/'proxy_event_diagnostics.csv')
    display=per[['winter','observed_days','adjacent_pairs','p_ge1_next_given_ge1_today','p_ge1_next_given_below1_today']].copy()
    display.iloc[:,-2:]=display.iloc[:,-2:]*100
    display.columns=['Winter','Observed days','Adjacent pairs','Next ≥1 in after ≥1 in (%)','Next ≥1 in after <1 in (%)']
    table=display.to_html(index=False,border=0,float_format=lambda x:f'{x:.1f}')
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Snowfall as storm events: research and Vail design</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef4f6;color:#203c49;font:16px/1.7 system-ui,-apple-system,sans-serif}}
main{{max-width:1120px;margin:30px auto;background:white;padding:46px 54px;border-radius:16px}}
h1{{font-size:39px;line-height:1.15;letter-spacing:-.03em;margin:12px 0 20px}}h2{{font-size:25px;line-height:1.25;margin-top:42px;border-top:1px solid #dce5e9;padding-top:26px}}h3{{font-size:18px;margin-top:25px}}
a{{color:#006d87;text-underline-offset:3px}}.eyebrow{{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:#007b91;font-weight:700}}
.lead{{font-size:19px;color:#4c6674}}.note{{padding:18px 22px;border-left:4px solid #b88037;background:#fff7ec;border-radius:4px}}
.callout{{padding:18px 24px;background:#eff6f8;border-radius:8px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:11px 13px;text-align:left;border-bottom:1px solid #dde6eb;vertical-align:top}}th{{background:#eff5f7}}.table{{overflow:auto}}img{{width:100%;height:auto}}.small{{font-size:13px;color:#60747f}}code,pre{{font-family:ui-monospace,monospace}}pre{{background:#eef5f7;padding:20px;border-radius:8px;overflow:auto;font-size:15px;line-height:1.65}}li{{margin:9px 0}}footer{{margin-top:36px;padding-top:20px;border-top:1px solid #dce5e9;font-size:13px;color:#60747f}}
@media(max-width:760px){{main{{margin:0;padding:25px 20px;border-radius:0}}h1{{font-size:31px}}}}
</style></head><body><main><div class="eyebrow">Literature review and engineering design · November 4, 2026</div>
<h1>Model snowfall as a sequence of storms</h1>
<p class="lead">Predict when a storm begins, how long it lasts, how much snow it produces, and how quiet intervals are distributed. Use atmospheric signals to condition those probabilities.</p>
<div class="callout"><strong>Recommendation:</strong> retain the hourly weather encoder and replace the daily marginal decoder with a weather-conditioned storm-event model. Start with explicit state durations and a compact marked point process. Compare it with a classical storm generator before adding more neural complexity.</div>
<p>This is a proposed redesign supported by the literature and a descriptive check of the saved Vail observations. No new storm model has been fitted, and no new forecast-skill improvement is claimed. The previous forecast remains unchanged.</p>

<h2>What the original Vail data show</h2>
<p>Using only screened October–April observations, excluding the two previously documented sensor-artifact windows, there are <strong>{pooled['observed_days']:,} daily endpoint readings</strong> and <strong>{pooled['adjacent_pairs']:,} pairs of adjacent observed days</strong>. Missing dates are excluded rather than treated as zero.</p>
<ul><li>After a reading ≥1 inch, the next reading also reaches 1 inch in <strong>{100*pooled['p_ge1_next_given_ge1_today']:.1f}%</strong> of cases ({pooled['n_11']}/{pooled['n_11']+pooled['n_10']}).</li>
<li>After a reading below 1 inch, that probability is <strong>{100*pooled['p_ge1_next_given_below1_today']:.1f}%</strong> ({pooled['n_01']}/{pooled['n_00']+pooled['n_01']}).</li></ul>
<div class="table">{table}</div>
<p>The direction is consistent across all six winters. This supports testing temporal persistence and event structure. It does not establish a particular Poisson or Hawkes law, nor prove that a front caused the association. Finite storm duration can itself create correlated wet days even when storm arrivals are Poisson.</p>
<div class="note"><strong>Measurement matters especially for event models.</strong> These are CAIC CAVMM 04:00 <code>Sno24</code> sensor readings, not established daily fresh-snow increments. A board reset can masquerade as a storm ending; unreset accumulation can masquerade as a long storm. “Below 1 inch” is not a verified dry day. Event labels require reset documentation or an overlapping manual snowfall record.</div>

<h2>A smooth mean can come from spiky storms</h2>
<p>If one six-inch storm is equally likely on any of six days, each possible outcome contains a single storm, while the conditional mean is one inch on each day. That mean is the optimal point forecast under squared-error loss. A guessed spike on the wrong day can increase MSE even if it looks more realistic.</p>
<img src="storm_events_vs_mean.png" alt="Synthetic storm sequences with peaks and quiet intervals, followed by their smooth ensemble mean">
<p class="small">Synthetic illustration only: invented marked Poisson arrivals, smooth finite-duration storm pulses, and exact daily integration. It is not fitted to Vail and is not a replacement weather forecast.</p>
<p>The preceding Vail model already had a point mass at zero and a positive-amount distribution. It also generated temporally dependent paths using historical rank templates. Its missing ingredient was an explicitly learned process for joint storm onset, duration, amount and weather-conditioned evolution. Temperature, dew point, humidity, wind and available pressure were already inputs. <a href="../vail-hourly/vail_hourly_report.html">Previous experiment</a>.</p>
<p>Modern ensemble weather models make the same distinction between individual plausible trajectories and the average over those trajectories. GenCast generates stochastic atmospheric trajectories; a single mean curve cannot display their full event structure. <a href="https://arxiv.org/html/2312.15796v2">Price et al., GenCast</a>.</p>

<h2>What the literature contributes</h2>
<div class="table"><table><thead><tr><th>Approach and primary source</th><th>What it adds</th><th>Implication for Vail</th></tr></thead><tbody>
<tr><td>Poisson-cluster rainfall pulses<br><a href="https://hess.copernicus.org/articles/24/2791/2020/">Onof &amp; Wang, 2020</a><br><a href="https://gmd.copernicus.org/articles/18/1357/2025/">Wei et al., pyBL, 2025</a></td><td>Storm arrivals generate clusters of finite-duration precipitation cells. pyBL supplies a practical calibration and simulation implementation, including short-record experiments.</td><td>A useful classical event baseline. Published simulation and extreme-statistics performance do not establish future storm timing. pyBL also reports difficulty reproducing monthly variability beyond the mean.</td></tr>
<tr><td>Smooth shot-noise precipitation<br><a href="https://link.springer.com/article/10.1007/s00477-017-1483-z">Ramesh, Garthwaite &amp; Onof, 2018</a></td><td>A doubly stochastic arrival process drives exponentially decaying precipitation pulses.</td><td>Close to the proposed “smoothed point process.” For exact quiet intervals, our design would use pulses with finite duration rather than indefinitely positive exponential tails.</td></tr>
<tr><td>Hidden semi-Markov rainfall states<br><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2009WR008602">Sansom &amp; Thomson, 2010</a></td><td>Explicit durations and a hierarchy of wet/dry periods, episodes, events and seasons.</td><td>Our preferred first fitted event model: it can learn how long quiet and active states persist, with fewer parameters than a large generative network.</td></tr>
<tr><td>Marked Hawkes rainfall generator<br><a href="https://espace.inrs.ca/id/eprint/16534/">Chapon, Ouarda &amp; Bertrand, 2025</a></td><td>Two-scale arrival clustering, distributions for amounts including extremes, and a copula for dependence between amounts.</td><td>A modern alternative for clustered bursts. It was assessed as a rainfall generator on 40 years of French hourly data, not as a local snowfall forecast. Statistical self-excitation is not proof that snowfall causes another front.</td></tr>
<tr><td>Neural inter-event timing<br><a href="https://arxiv.org/abs/1909.12127">Shchur, Biloš &amp; Günnemann, ICLR 2020</a></td><td>Directly learns conditional distributions of the waiting time to the next event, allowing flexible and efficient sampling.</td><td>Reuse the hourly encoder to predict time-to-next-storm and joint duration/amount marks. The paper establishes a general event-modeling method, not snow-specific skill.</td></tr>
<tr><td>Recent Hawkes research<br><a href="https://arxiv.org/abs/2607.27099">Deschatre, Hoffmann &amp; Rosenbaum, 2026 preprint</a></td><td>Generalizes rain-cell clustering and investigates power-law kernels and cross-scale behavior.</td><td>A research candidate after simpler models. Better statistical fit at high frequency is not demonstrated operational Vail forecast skill; six partial winters limit how much complexity is identifiable.</td></tr>
</tbody></table></div>

<h2>Connect events to fronts and mountain physics</h2>
<p>Colorado observations provide direct support for this direction. A SPLASH/SAIL study examined four storms during a period contributing 35% of that watershed’s winter precipitation. Temporal changes were tied to synoptic and frontal forcing; differences between nearby sites involved local winds and precipitation microphysics. This is physical support for event and terrain conditioning, not proof that all snow follows the same frontal sequence. <a href="https://repository.library.noaa.gov/view/noaa/67776">Heflin et al., 2024</a>.</p>
<p>The local encoder should use pressure tendency, temperature and dew-point changes, humidity, wind components and directional changes, and contrasts between stations. A wind shift alone is insufficient: moisture supply, lifting and the temperature profile also matter. The four Vail sites occupy different places and elevations; they are not a vertically collocated atmospheric sounding.</p>
<p>For incoming weather, add archived forecasts of upstream winds, moisture transport, thermal profiles and precipitation, available at the historical issue time. FrontFinder AI demonstrates detection of front types from gridded atmospheric fields, suggesting candidate inputs such as front probability, distance and approach. It does not turn four local station records into a complete front analysis. <a href="https://repository.library.noaa.gov/view/noaa/70211">Justin, McGovern &amp; Allen, 2025</a>.</p>
<p>A related rainfall study already allows point-process parameters to vary with pressure and temperature. That is a statistical precedent for conditioning storm behavior on weather; our forecast would require past or issued-future covariates rather than future observed conditions. <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4975607/">Kaczmarska, Isham &amp; Northrop, 2015</a>.</p>

<h2>The proposed model</h2>
<p>Let T<sub>k</sub> be a storm’s start, D<sub>k</sub> its duration, A<sub>k</sub> its total snowfall, and g a nonnegative pulse supported on [0,1] with integral one. A smooth storm trajectory and daily accumulation are:</p>
<pre>snowfall_rate(t) = Σₖ (Aₖ / Dₖ) · g((t − Tₖ) / Dₖ)
daily_snow(d)    = ∫ from d to d+1 snowfall_rate(t) dt

P(storm starts in [t,t+dt] | known history)
                  ≈ λ(t | weather, season, time in state) · dt</pre>
<p>Discrete arrivals and a continuous snowfall-rate curve are compatible. The factor A<sub>k</sub>/D<sub>k</sub> ensures each complete event integrates to its predicted total. A beta-shaped or spline pulse can rise and decay asymmetrically; clusters of pulses can represent multiple waves in one storm.</p>
<ol><li><strong>Onset and quiet intervals:</strong> estimate a weather-conditioned hazard or waiting-time distribution. A constant-rate Poisson model is the simplest comparator; a renewal or semi-Markov model lets duration and time since the previous event affect the next transition. Do not impose a rigid multi-day dry break after every storm.</li>
<li><strong>Storm marks:</strong> predict duration, total amount and pulse shape jointly. Dependence matters: duration and intensity need not be independent. Snow-phase probability and amount should depend on thermal and moisture conditions.</li>
<li><strong>Ongoing events:</strong> infer the active state and remaining duration at issue time. Do not restart the storm process at every forecast origin. Allow observations to update the state as the storm develops.</li>
<li><strong>Neural version:</strong> retain the multi-station TCN encoder, attach a small event-hazard and joint-mark decoder, and compare it with a simpler covariate-conditioned semi-Markov model. “Dry / developing / active / decaying” states are an engineering hypothesis to test, not verified labels.</li>
<li><strong>Joint forecasts:</strong> sample event sequences and integrate them into daily amounts. Sum each trajectory before taking monthly quantiles. Show several trajectories beside the mean, probability of a storm in the next 24/72 hours, and storm-total ranges.</li></ol>
<p>For event-time likelihood, dry exposure matters as much as observed onsets: the loss includes the integrated arrival hazard as well as the log hazards at events and the mark likelihood. Missing intervals and events crossing observation boundaries require censoring. If only reliable daily snowfall is available, fit daily integrated amounts and treat within-day event times as latent or interval-censored. Do not invent hourly snowfall by differencing an unverified board or snow-depth signal.</p>
<p>At weeks 3–4 and beyond, separate storm-frequency/amount outlooks from confidence in exact event dates. Recent probabilistic bias-correction work improves weekly precipitation distributions from issued ensembles, evaluated on days 19–25 and 26–32 at a 1.5° grid. It is useful evidence for a larger-scale prior, not a demonstration of exact daily Vail snowfall timing. <a href="https://www.nature.com/articles/s41586-026-11126-x">Guan et al., October 2026</a>.</p>

<h2>What would count as improvement</h2>
<p>Keep the same whole-winter test separation and a chronological holdout. Define event thresholds, gap-merging rules, censoring and timing tolerances inside the training protocol. Test sensitivity to those definitions; do not pick the definition that wins on test winters.</p>
<div class="table"><table><thead><tr><th>Forecast property</th><th>Evaluation</th></tr></thead><tbody>
<tr><td>Storm arrival</td><td>Brier score and reliability for onset within 24/72 hours; event likelihood including dry exposure.</td></tr>
<tr><td>Timing and detection</td><td>Missed events, false alarms and onset errors with a predeclared matching tolerance. Timing error alone must not hide missed storms.</td></tr>
<tr><td>Storm structure</td><td>Duration, storm total, peak amount, dry-gap distributions, and temporal dependence; use joint trajectory scores as well as marginal scores.</td></tr>
<tr><td>Daily and monthly quantities</td><td>Retain MSE/RMSE/MAE for predictive means; add CRPS, interval coverage/width, and calibrated monthly-total distributions.</td></tr>
<tr><td>Value of physics and architecture</td><td>Ablate event structure, front-related inputs, issued weather forecasts and neural complexity separately on matched held-out cases.</td></tr>
</tbody></table></div>
<p>More realistic simulated spikes would establish improved representation. Better held-out probabilities and timing would establish forecasting value. The existing station measurement uncertainty must be resolved before interpreting either as validated resort snowfall skill.</p>

<h2>Saved research and diagnostics</h2>
<ul><li><a href="../../research/storm_point_process_review.txt">Point-process literature notes</a></li>
<li><a href="../../research/front_conditioned_storm_review.txt">Front-conditioned and modern forecasting notes</a></li>
<li><a href="../../research/vail_event_data_audit.txt">Original-data audit and current-model inspection</a></li>
<li><a href="proxy_event_diagnostics.json">Diagnostic definitions and counts</a> · <a href="proxy_event_diagnostics.csv">Winter-by-winter table</a> · <a href="proxy_event_runs.csv">Censored and complete threshold runs</a></li>
<li><a href="storm_events_vs_mean.png">Explanatory figure PNG</a> · <a href="storm_events_vs_mean.svg">SVG</a> · <a href="illustration_manifest.json">Synthetic illustration specification</a></li></ul>
<footer>Research review and proposed experiment. No event model was trained for this review. Existing Vail model weights, forecasts and published error tables were not modified.</footer>
</main></body></html>'''
    (OUT/'storm_model_review.html').write_text(html)
    print(OUT/'storm_model_review.html')


if __name__=='__main__':main()
