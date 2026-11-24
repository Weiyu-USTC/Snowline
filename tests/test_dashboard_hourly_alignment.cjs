const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');

const script = readFileSync(join(__dirname, '../dashboard/app.js'), 'utf8');
const hour = 3600000;
const start = Date.UTC(2030, 10, 23, 6);
const at = hours => new Date(start + hours * hour).toISOString();
function series(snowCm, tempC) {
  return Array.from({length: 48}, (_, i) => ({
    start: at(i), time: at(i + 1), snowfall_inches: (i === 0 ? snowCm : 0) / 2.54,
    temperature_c: tempC, wind_kmh: tempC + 10, precipitation_mm: 0,
  }));
}
const comparison = {
  hours: 48, quantity: 'snowfall', unit: 'inches', start: at(0), end: at(48),
  status: 'ready', mean: 3.99 / 2.54, provider_mean: 1.33 / 2.54,
  sources: {provider: {target_label: 'Comparison station', retrieved_at: at(-1)}},
  provider_hourly: series(1.33, -3),
};

// Exercise the dashboard's regular fetch and render path with no browser or network.
async function display(row, units = 'metric', mountain = 'vail', observations = {}, quantity = 'water') {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      style: {}, classList: {add(){}, toggle(){}}, replaceChildren(){}, addEventListener(){},
      querySelectorAll: () => [], closest: () => element(id + '-card'),
      querySelector: selector => element(id + ' ' + selector),
    });
    return elements.get(id);
  }
  const payload = {site: {id: mountain}, 
    hourly: series(2.66, 4), source: {name: 'Rolling source'}, daily: [],
    deployed: mountain === 'mammoth' ? undefined : {site_id: mountain, horizons: [row]}, ...observations};
  const document = {getElementById: element, querySelectorAll: () => []};
  const fetch = async url => ({ok: true, json: async () => url.startsWith('/api/forecast') ? payload : {}});
  vm.runInNewContext(script, {document, location: {search: `?mountain=${mountain}&hours=${row.hours}&units=${units}&quantity=${quantity}`},
    URLSearchParams, fetch, AbortController, setTimeout, clearTimeout, console});
  await new Promise(resolve => setImmediate(resolve));
  assert.ok(!element('error-banner').textContent, element('error-banner').textContent);
  return element;
}

test('bar, headline, histogram and weather table use the identical 48-hour snapshot', async () => {
  for (const [units, expected] of [['metric', '1.33'], ['us', '0.52']]) {
    const e = await display(comparison, units);
    assert.equal(e('snow-total').textContent, expected);
    assert.match(e('neural-chart').innerHTML, new RegExp('Open-Meteo.*?' + expected.replace('.', '\\.')));
    assert.ok(e('snow-chart').innerHTML.includes(expected));
    assert.ok(e('weather-table-wrap').innerHTML.includes(expected));
    assert.equal(e('temperature-range').textContent, units === 'metric' ? '-3–-3' : '27–27');
    assert.equal(e('target-label').textContent, 'Comparison station');
  }
});

test('missing or discontinuous matched hours leave the provider amount unavailable', async () => {
  for (const rows of [null, [], comparison.provider_hourly.slice(1),
    [comparison.provider_hourly[1], ...comparison.provider_hourly.slice(1)]]) {
    const e = await display({...comparison, provider_hourly: rows});
    assert.equal(e('snow-total').textContent, '—');
    assert.match(e('snow-chart').innerHTML, /Hourly comparison data unavailable/);
    assert.match(e('neural-chart').innerHTML, /Open-Meteo[^<]+Unavailable/);
  }
});

test('a legacy export keeps its rolling outlook and identifies the missing comparison series', async () => {
  const {provider_hourly, ...old} = comparison;
  const e = await display(old);
  assert.equal(e('snow-total').textContent, '2.66');
  assert.match(e('neural-chart').innerHTML, /Open-Meteo[^<]+Unavailable/);
  assert.match(e('neural-details').innerHTML, /separate hourly guidance/);
});

test('a short liquid comparison continues to show independent rolling snowfall in the outlook', async () => {
  const e = await display({...comparison, hours: 1, end: at(1),
    quantity: 'liquid_precipitation', unit: 'mm', provider_mean: 5.1});
  assert.equal(e('snow-total').textContent, '2.66');
  assert.match(e('neural-chart').innerHTML, /5\.10 mm water/);
});

test('Mammoth separates hourly provider snowfall, a dry radar scan, and missing radar', async () => {
  const radar = {coverage: 'usable', timestamp: at(0), precip_rate_mm_h: 0, quality_index: 0.4};
  for (const [coverage, expected] of [['usable', '0.00'], ['poor_quality', '—']]) {
    const e = await display(comparison, 'metric', 'mammoth', {radar_observation: {...radar, coverage}});
    assert.equal(e('neural-panel').hidden, false);
    assert.equal(e('mammoth-observations').hidden, false);
    assert.equal(e('snow-total').textContent, '2.66');
    assert.equal(e('radar-rate').textContent, expected);
    assert.equal(e('radar-rate-unit').textContent, 'mm/h');
    assert.match(e('radar-coverage').textContent, /0.40/);
  }
});

test('Mammoth draws hourly snow water and compares only with same-quantity HRRR', async () => {
  const model = {hours:3, start:at(0),end:at(3),issued_at:at(0),status:'ready',unit:'mm',
    quantity:'snow_water_equivalent',mean:3,provider_mean:null,provider_name:'HRRR'};
  const deployed = {site_id:'mammoth', horizons:[model],
    hourly:[0,1,2].map((i)=>({horizon_hours:i+1,start:at(i),end:at(i+1),mean_mm:1,provider_mm:i<2?0:null}))};
  const e=await display(model,'metric','mammoth',{deployed});
  assert.match(e('neural-chart').innerHTML,/HRRR.*Unavailable/);
  assert.match(e('swe-chart').innerHTML,/Hourly snow-water equivalent/);
  assert.match(e('swe-coverage').textContent,/3 model hours · 2 HRRR hours/);
  assert.equal(e('daily-snow-unit').textContent,'mm SWE');
  assert.equal(e('daily-snow-amount').textContent,'3.00');
  assert.equal(e('snow-total-unit').textContent,'cm');
});

test('Main Lodge depth view matches forecast hours, keeps inches separate and excludes a late provider retrieval', async () => {
  const row = {hours:3, start:at(0),end:at(3),issued_at:at(-1),status:'ready',unit:'mm',
    quantity:'snow_water_equivalent',mean:7,provider_mean:3,depth_inches:1,provider_name:'HRRR'};
  const deployed = {site_id:'mammoth',start:at(0),horizons:[row],
    hourly:[0,1,2].map(i=>({horizon_hours:i+1,start:at(i),end:at(i+1),mean_mm:7/3,
      provider_mm:1,depth_inches:1/3}))};
  for (const [retrieved, valid] of [[at(-1),true],[at(1),false]]) {
    const e=await display(row,'metric','mammoth',{deployed,source:{retrieved_at:retrieved}},'depth');
    assert.match(e('swe-chart').innerHTML,/Hourly fresh snow depth/);
    assert.equal(e('daily-snow-unit').textContent,'cm');
    assert.equal(e('daily-snow-amount').textContent,'2.54');
    assert.match(e('neural-chart').innerHTML,valid?/Open-Meteo.*2.66/:/Open-Meteo.*Unavailable/);
    assert.equal(e('swe-ecmwf-key').hidden,true);
  }
});
