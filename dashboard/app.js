'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const mountainsFallback = [
    {id:'alta',name:'Alta',location:'Utah · Wasatch Range',timezone:'America/Denver',target_label:'Alta–Collins manual snowfall target'},
    {id:'vail',name:'Vail',location:'Colorado · Gore Range',timezone:'America/Denver',target_label:'Vail station snow-sensor proxy'},
    {id:'mansfield',name:'Stowe',location:'Vermont · Green Mountains',timezone:'America/New_York',target_label:'NOAA Mount Mansfield station target'},
    {id:'jay',name:'Jay Peak',location:'Vermont · Green Mountains',timezone:'America/New_York',target_label:'NOAA station target near Jay Peak'},
    {id:'mammoth',name:'Mammoth Mountain',location:'California · Sierra Nevada',timezone:'America/Los_Angeles',target_label:'Mammoth Mountain Main Lodge'}
  ];
  let savedUnits='us';try{if(localStorage.getItem('snowline.units.v1')==='metric')savedUnits='metric';}catch{}
  const state = {mountains:mountainsFallback,site:'alta',hours:48,view:'hourly',units:savedUnits,forecast:null,forecasts:{},imported:false,loading:false,request:0,seasons:{},seasonLoading:{},seasonErrors:{},seasonRequests:{},seasonView:'full',sweView:'hourly',mammothQuantity:'water'};
  const forecastHorizons = [1,3,6,12,24,48];
  const modelDisplay = {label:'Our model',color:'#be7c3a'};
  function syncForecastURL(){const url=new URL(location.href);url.searchParams.set('mountain',state.site);url.searchParams.set('hours',String(state.hours));url.searchParams.set('units',state.units);if(state.site==='mammoth')url.searchParams.set('quantity',state.mammothQuantity);else url.searchParams.delete('quantity');history.replaceState(null,'',`${url.pathname}${url.search}${url.hash}`);}
  const mountainIcon = '<svg viewBox="0 0 36 30" fill="none" aria-hidden="true"><path d="m2 25 12-20 8 13 4-7 9 14H2Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="m9 14 5 3 5-3" stroke="currentColor" stroke-width="1.6"/></svg>';
  const number = value => value!==null && value!=='' && value!==undefined && Number.isFinite(Number(value)) ? Number(value) : null;
  const val = (object,...keys) => { for(const k of keys)if(object && object[k]!==undefined && object[k]!==null)return object[k];return null; };
  const fmt = (value,digits=1) => number(value)===null ? '—' : Number(value).toLocaleString('en-US',{maximumFractionDigits:digits,minimumFractionDigits:digits});
  const metric = () => state.units==='metric';
  const displaySnow = v => number(v)===null ? null : Number(v)*(metric()?2.54:1);
  const displayTemp = v => number(v)===null ? null : metric()?Number(v):Number(v)*9/5+32;
  const displayWind = v => number(v)===null ? null : Number(v)/(metric()?1:1.609344);
  const displayLiquid = v => number(v)===null ? null : Number(v)/(metric()?1:25.4);
  const amountText = (value,digits=1) => number(value)===null?'—':Number(value)>0&&Number(value)<.01?'<0.01':fmt(value,Number(value)>0&&Number(value)<.1?Math.max(2,digits):digits);
  const snowText = (inches,digits=1) => amountText(displaySnow(inches),digits);
  const units = () => ({snow:metric()?'cm':'in',snowLong:metric()?'centimeters':'inches',temp:metric()?'°C':'°F',tempLong:metric()?'degrees Celsius':'degrees Fahrenheit',wind:metric()?'km/h':'mph',windLong:metric()?'kilometers per hour':'miles per hour',liquid:metric()?'mm':'in'});
  const setText = (id,text) => {if($(id))$(id).textContent=text;};
  const snowPalette = [[0,[78,157,166]],[.5,[66,107,171]],[1,[81,53,140]]];
  function snowColor(inches,maximum){
    if(number(inches)===null)return '#8a9996';
    const t=Math.min(1,Math.max(0,Number(inches)/maximum));
    const i=t<=.5?0:1,[lo,a]=snowPalette[i],[hi,b]=snowPalette[i+1],f=(t-lo)/(hi-lo);
    return `rgb(${a.map((v,j)=>Math.round(v+(b[j]-v)*f)).join(',')})`;
  }
  function renderColorLegend(id,maximum,hourly=false){
    if(!$(id))return;
    const u=units(),gradient=snowPalette.map(([at])=>`${snowColor(at*maximum,maximum)} ${at*100}%`).join(',');
    const labels=[0,.25,.5,.75,1].map((at,i)=>`${Number(displaySnow(at*maximum).toFixed(2)).toLocaleString('en-US',{maximumFractionDigits:2})}${i===4?'+':''}`);
    $(id).innerHTML=`<div class="snow-color-scale"><div class="snow-color-gradient" style="background:linear-gradient(90deg,${gradient})" aria-hidden="true"></div><div class="snow-color-labels">${labels.map(label=>`<span>${esc(label)}</span>`).join('')}</div><p class="snow-color-note">${esc(u.snow)}${hourly?' per hour':' accumulated'} · Color shows snowfall amount, not probability. The darkest color includes higher amounts.</p></div>`;
  }

  const esc = s => String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const siteInfo = () => state.mountains.find(m=>m.id===state.site)||mountainsFallback.find(m=>m.id===state.site)||mountainsFallback[0];
  const tz = () => val(state.forecast?.site,'timezone')||siteInfo().timezone||'America/Denver';
  const date = (s,options={}) => {if(!s||!Number.isFinite(Date.parse(s)))return 'Time unavailable';try{return new Date(s).toLocaleString('en-US',{timeZone:tz(),...options});}catch{return new Date(s).toLocaleString('en-US',options);} };
  const dateTime = s => date(s,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'});
  const localHour = s => date(s,{hour:'numeric'});
  const comparisonTitle = (start,end,unit) => {
    if(!Number.isFinite(Date.parse(start))||!Number.isFinite(Date.parse(end)))return `Forecast window · ${unit}`;
    const zone=s=>{try{return new Intl.DateTimeFormat('en-US',{timeZone:tz(),timeZoneName:'short'}).formatToParts(new Date(s)).find(p=>p.type==='timeZoneName')?.value;}catch{return '';}};
    const stamp=(s,showZone)=>date(s,{month:'short',day:'numeric',hour:'numeric',...(new Date(s).getUTCMinutes()?{minute:'2-digit'}:{}),...(showZone?{timeZoneName:'short'}:{})}).replace(/\s([AP]M)\b/g,'$1');
    return `${stamp(start,zone(start)!==zone(end))} – ${stamp(end,true)} · ${unit}`;
  };

  const notice = (id,message) => {$(id).hidden=!message;$(id).textContent=message||'';};
  const empty = (title,body) => `<div class="empty-state"><strong>${esc(title)}</strong><span>${esc(body)}</span></div>`;
  const warningsText = values => (Array.isArray(values)?values:values?[values]:[]).filter(x=>x!==null&&x!==undefined).map(x=>typeof x==='string'?x:x.message||x.note||JSON.stringify(x));
  const displayNotes = values => warningsText(values).flatMap(text=>String(text).split(/(?<=[.!?])\s+/)).filter(text=>!/(powder|five[\s-]*inch|5[\s-]*inch|12[.,]7\s*cm)/i.test(text));
  function normalize(raw){
    const p={...raw};
    if(typeof p.site==='string')p.site={...mountainsFallback.find(m=>m.id===p.site),id:p.site};
    p.site=p.site||{...siteInfo()};
    if(p.site.id==='mansfield')p.site={...p.site,name:'Stowe'};
    p.hourly=(p.hourly||[]).map(x=>({...x,time:val(x,'time','valid_time','valid_utc'),snowfall_inches:number(val(x,'snowfall_inches','snow_inches')),precipitation_mm:number(val(x,'precipitation_mm','precip_mm','liquid_mm')),temperature_c:number(val(x,'temperature_c','temp_c')),wind_kmh:number(val(x,'wind_kmh','wind_speed_kmh'))})).filter(x=>Number.isFinite(Date.parse(x.time))).sort((a,b)=>Date.parse(a.time)-Date.parse(b.time));
    p.daily=(p.daily||[]).map(x=>({...x,start:val(x,'start','start_utc','window_start'),end:val(x,'end','end_utc','window_end'),mean_inches:number(val(x,'mean_inches','predicted_inches','mean')),p_powder:number(val(x,'p_powder','powder_probability')),q10:number(val(x,'q10','q10_inches')),q90:number(val(x,'q90','q90_inches'))}));
    return p;
  }
  async function api(url,options){const response=await fetch(url,{...options,headers:{'Content-Type':'application/json',...(options?.headers||{})}});const data=await response.json().catch(()=>({error:'The local service returned an unreadable response.'}));if(!response.ok)throw new Error(typeof data.error==='string'?data.error:data.message||`Request failed (${response.status})`);return data;}
  function renderMountains(){
    $('mountain-list').innerHTML=state.mountains.map(m=>`<button class="mountain-button${m.id===state.site?' active':''}" data-site="${esc(m.id)}" ${state.loading?'disabled':''} aria-pressed="${m.id===state.site}">${mountainIcon}<span><strong>${esc(m.name)}</strong><small>${esc(m.location||m.region||'Mountain forecast')}</small></span><span class="selection-dot"></span></button>`).join('');
    $('mountain-list').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>selectMountain(b.dataset.site)));
  }
  async function selectMountain(id){
    if(state.loading)return;
    state.site=id;state.imported=false;state.forecast=null;syncForecastURL();renderMountains();render();notice('error-banner','');await loadForecast(id);
  }
  async function loadForecast(id){const request=++state.request;void loadSeason(id);try{const data=await api(`/api/forecast?site=${encodeURIComponent(id)}`);state.forecasts[id]=normalize(data);if(request!==state.request)return;state.forecast=state.forecasts[id];render();}catch(error){if(request!==state.request)return;notice('error-banner',`Could not load the forecast. ${error.message}`);$('data-status').textContent='Service unavailable';$('updated-at').textContent='Start the local forecast service and try Refresh all.';}}
  function hourlyOutlook(p, hours, row){
    const paired=deploymentAvailable(row)&&row.quantity==='snowfall';
    const rolling=(p?.hourly||[]).slice(0,hours);
    if(!paired||!Object.hasOwn(row,'provider_hourly')){
      return {rows:rolling,source:p?.source||{},paired:false,legacyPair:paired};
    }

    const source=row.sources?.provider||{};
    const empty={rows:[],source,paired:true,legacyPair:false,total:null};
    const rows=row.provider_hourly;
    if(!Array.isArray(rows)||rows.length!==hours||number(row.provider_mean)===null)return empty;
    let next=Date.parse(row.start),total=0;
    for(const hour of rows){
      const end=Date.parse(hour?.time);
      if(Date.parse(hour?.start)!==next||end!==next+3600000||
         !Number.isFinite(hour?.snowfall_inches)||hour.snowfall_inches<0)return empty;
      next=end;
      total+=hour.snowfall_inches;
    }
    if(next!==Date.parse(row.end)||Math.abs(total-row.provider_mean)>1e-8*Math.max(1,total))return empty;
    return {rows,source,paired:true,legacyPair:false,total};
  }
  function render(){
    const p=state.forecast,m={...siteInfo(),...(p?.site||{})},u=units();
    const mammoth=state.site==='mammoth';
    const selected=selectedDeployment(),outlook=hourlyOutlook(p,state.hours,selected);
    const neural=$('neural-panel'), season=$('season-panel');
    // Put the hourly forecast above the resort history when viewing Mammoth.
    if(neural.parentNode && season.parentNode===neural.parentNode){
      if(mammoth && neural.nextElementSibling!==season)season.before(neural);
      if(!mammoth && season.nextElementSibling!==neural)neural.before(season);
    }
    $('daily-model-panel').hidden=mammoth;
    $('mammoth-observations').hidden=!mammoth;
    $('mammoth-model-detail').hidden=!mammoth;
    $('mammoth-quantity').hidden=!mammoth;
    $('mammoth-model-report').hidden=!mammoth;
    $('other-model-reports').hidden=mammoth;
    setText('provider-chart-note',mammoth?'Open-Meteo estimates fresh-snow depth at Main Lodge. The hourly model above predicts snow water at the Jeff Dozier station.':'This hourly curve is provided by Open-Meteo. The selected model forecast appears in the comparison above.');
    const {rows:hours,source}=outlook;
    document.querySelectorAll('[data-hours]').forEach(b=>{const selected=Number(b.dataset.hours)===state.hours;b.classList.toggle('active',selected);b.setAttribute('aria-pressed',String(selected));});
    document.querySelectorAll('[data-units]').forEach(b=>{const selected=b.dataset.units===state.units;b.classList.toggle('active',selected);b.setAttribute('aria-pressed',String(selected));});
    setText('snow-total-unit',u.snow);setText('temperature-unit',u.temp);setText('wind-unit',u.wind);
    setText('daily-snow-unit',u.snow);
    setText('daily-eyebrow','Previous station model');
    setText('daily-title','Daily model comparison');
    setText('daily-subtitle','Expected fresh snow and uncertainty over each 24-hour reporting window.');
    renderColorLegend('snow-color-legend',state.view==='hourly'?1:24,state.view==='hourly');
    renderColorLegend('daily-color-legend',24);
    setText('snowfall-caption',`Fresh snowfall estimate · ${u.snowLong}`);
    $('mountain-title').textContent=m.name||siteInfo().name;$('mountain-location').textContent=m.location||m.region||siteInfo().location||'';$('target-label').textContent=source.target_label||m.target_label||m.target||'Station-based research forecast';
    $('export-button').disabled=!p;
    if(mammoth)setText('target-label','Main Lodge · Jeff Dozier Snow Study Site');
    const generated=source.retrieved_at||(outlook.paired?null:p?.generated_at||p?.generated_utc);
    const end=outlook.paired?selected.end:hours.at(-1)?.time;
    const expired=!!(end&&Date.parse(end)<=Date.now());
    const windowStart=outlook.paired?selected.start:hours[0]?.start;
    const savedWindow=Date.parse(windowStart)<Date.now()||outlook.paired&&(selected.stale||selected.window_started||selected.expired);
    const stale=!!p?.stale||!!source.stale||savedWindow||expired||(generated&&Date.now()-Date.parse(generated)>6*3600000);
    const unavailable=!p||!(p.hourly?.length||p.daily?.length||p.deployed?.horizons?.some(r=>number(r.mean)!==null));
    const partial=p?.status==='partial'||(Array.isArray(p?.errors)&&p.errors.length>0);
    const sourceCached=source.status==='cached';
    const statusText=state.imported?'Imported snapshot':unavailable?'Awaiting data':partial?'Partial update':stale?'Saved · refresh recommended':sourceCached?'Saved guidance':'Data ready';
    $('data-status').textContent=statusText;$('data-status').className=`badge ${state.imported||stale||partial?'warn':unavailable||sourceCached?'neutral':'live'}`;
    $('updated-at').textContent=generated?`Retrieved ${dateTime(generated)}`:p?'No current forecast available':'Checking saved forecasts';
    const hourlySource=source.name||(outlook.paired?'Open-Meteo':p?.hourly_source?.name||'Hourly weather guidance');
    $('source-label').textContent=hourlySource;
    $('hourly-model-label').textContent=[hourlySource,source.target_label].filter(Boolean).join(' · ');
    $('chart-zone').textContent=date(new Date().toISOString(),{timeZoneName:'short'}).split(' ').at(-1)+' · mountain local time';
    let snapshotMessage=state.imported?'Viewing an imported snapshot. Refresh replaces it with newly fetched online data.':expired?'This forecast window has passed. The curves below are a saved snapshot; refresh for current guidance.':stale?'Showing a saved forecast. Refresh latest data to check for a newer weather run.':null;
    if(sourceCached&&!snapshotMessage)snapshotMessage='Showing previously retrieved hourly guidance. Its original retrieval time is shown above.';
    if(partial){const errors=warningsText(p?.errors).join(' ');snapshotMessage=[snapshotMessage,'Some sources could not refresh. Available forecasts below retain their original source timestamps.',errors].filter(Boolean).join(' ');}
    if(hours.length&&hours.length<state.hours)snapshotMessage=[snapshotMessage,`Only ${hours.length} hourly intervals are available for the selected ${state.hours}-hour horizon.`].filter(Boolean).join(' ');
    notice('snapshot-banner',snapshotMessage);
    const snow=hours.map(x=>x.snowfall_inches).filter(Number.isFinite),temps=hours.map(x=>x.temperature_c).filter(Number.isFinite),winds=hours.map(x=>x.wind_kmh).filter(Number.isFinite);
    $('snow-total-label').textContent=hours.length&&hours.length<state.hours?`${hours.length} available hours`:`Next ${state.hours} full ${state.hours===1?'hour':'hours'}`;
    if(outlook.paired)$('snow-total-label').textContent=`${state.hours}-hour comparison window`;
    if(savedWindow||expired)$('snow-total-label').textContent=`Saved ${state.hours}-hour window`;
    const total=outlook.paired?outlook.total:(snow.length&&snow.length===hours.length?snow.reduce((a,b)=>a+b,0):null);
    $('snow-total').textContent=snowText(total,2);
    $('snow-total').style.color=total===null?'':snowColor(total,24);
    const snowDescription=$('snow-total').closest('.summary-card').querySelector('.metric-description');
    snowDescription.textContent=outlook.paired?
      (total===null?'Hourly comparison data unavailable':comparisonTitle(selected.start,selected.end,u.snow)):
      snow.length!==hours.length?'Snowfall has missing hourly values':hours.length&&hours.length<state.hours?'Total over available hours only':'Forecast fresh snowfall';
    $('temperature-range').textContent=temps.length?`${Math.round(displayTemp(Math.min(...temps)))}–${Math.round(displayTemp(Math.max(...temps)))}`:'—';
    $('peak-wind').textContent=winds.length?fmt(displayWind(Math.max(...winds)),0):'—';
    const daily=p?.daily||[],next=daily.find(d=>d.end&&Date.parse(d.end)>Date.now());
    setText('daily-snow-amount',next?snowText(next.mean_inches):'—');
    if($('daily-snow-amount'))$('daily-snow-amount').style.color='';
    setText('daily-snow-description',next?`Expected · ending ${date(next.end,{month:'short',day:'numeric',hour:'numeric'})}`:daily.length?'Saved model windows have passed':'Calibrated outlook unavailable');
    if(hours.length)$('chart-zone').textContent=`${dateTime(hours[0].start||hours[0].time)} → ${dateTime(hours.at(-1).time)}`;
    const chartHours=hours.length||state.hours;
    $('snow-chart-title').textContent=`${state.view==='cumulative'?'Accumulation':'Snowfall'} over ${chartHours} ${chartHours===1?'hour':'hours'}`;
    renderSnow(hours,outlook.paired);
    const shownDeployment = mammoth ? mammothDisplayForecast(p) : p?.deployed;
    renderDeployed(shownDeployment,outlook);
    if(mammoth){
      renderMammothCurve(shownDeployment);
      renderMammothRadar(p?.radar_observation);
      renderSesame(p?.sesame_observation);
    }
    renderWeather(hours);renderSources(p,m,source);renderSeason();
  }
  function niceTop(n){if(!n||n<=0)return .1;const power=10**Math.floor(Math.log10(n)),v=n/power;return (v<=1?1:v<=2?2:v<=4?4:v<=5?5:10)*power;}
  async function loadSeason(id){
    const request=(state.seasonRequests[id]||0)+1;state.seasonRequests[id]=request;state.seasonLoading[id]=true;delete state.seasonErrors[id];
    if(state.site===id&&!state.imported)renderSeason();
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);
    try{
      const payload=await api(`/api/season?site=${encodeURIComponent(id)}`,{signal:controller.signal});
      if(payload.site_id!==id||!Array.isArray(payload.history)||!payload.current||!Array.isArray(payload.current.points))throw new Error('The service returned an incomplete seasonal record.');
      if(state.seasonRequests[id]===request)state.seasons[id]=payload;
    }catch(error){if(state.seasonRequests[id]===request)state.seasonErrors[id]=error.name==='AbortError'?'The seasonal observations took too long to load. Try Refresh all again.':error.message;}
    finally{clearTimeout(timer);if(state.seasonRequests[id]===request){state.seasonLoading[id]=false;if(state.site===id&&!state.imported)renderSeason();}}
  }
  // Align calendar dates on a common leap winter, keeping February 29 in its
  // own place. Timestamped forecast endpoints retain mountain-local time.
  function seasonPosition(value){
    if(typeof value!=='string'||!/^\d{4}-\d{2}-\d{2}/.test(value)||!Number.isFinite(Date.parse(value)))return null;
    let month=Number(value.slice(5,7)),day=Number(value.slice(8,10)),hour=0,minute=0;
    if(value.includes('T')){
      const parts=new Intl.DateTimeFormat('en-US',{timeZone:tz(),month:'numeric',day:'numeric',hour:'numeric',minute:'numeric',hourCycle:'h23'}).formatToParts(new Date(value));
      const part=type=>Number(parts.find(p=>p.type===type)?.value);
      month=part('month');day=part('day');hour=part('hour');minute=part('minute');
    }
    return (Date.UTC(month>=10?1999:2000,month-1,day,hour,minute)-Date.UTC(1999,9,1))/86400000;
  }
  const seasonDateLabel=value=>typeof value==='string'&&value.includes('T')?dateTime(value):typeof value==='string'&&Number.isFinite(Date.parse(value))?new Date(`${value.slice(0,10)}T12:00:00Z`).toLocaleDateString('en-US',{timeZone:'UTC',month:'short',day:'numeric',year:'numeric'}):'Date unavailable';
  function seasonAxisLabel(position,withDay=true){return new Date(Date.UTC(1999,9,1)+position*86400000).toLocaleDateString('en-US',{timeZone:'UTC',month:'short',...(withDay?{day:'numeric'}:{})});}
  function seasonValueAxis(maximum){
    if(!(maximum>0))return {top:1,ticks:[0,.25,.5,.75,1]};
    const magnitude=10**Math.floor(Math.log10(maximum));
    const rounding=magnitude/10;
    const top=Math.ceil(maximum*1.04/rounding)*rounding;
    const idealStep=top/5;
    const stepMagnitude=10**Math.floor(Math.log10(idealStep));
    const step=[1,2,2.5,5,10].map(n=>n*stepMagnitude)
      .reduce((best,n)=>Math.abs(n-idealStep)<Math.abs(best-idealStep)?n:best);
    return {top,ticks:Array.from({length:Math.floor(top/step+1e-9)+1},(_,i)=>i*step)};
  }
  function seasonPoints(rows){return (Array.isArray(rows)?rows:[]).map(p=>({...p,date:p.time||p.date,x:seasonPosition(p.time||p.date),v:displaySnow(p.cumulative_inches),lo:displaySnow(p.lower_inches),hi:displaySnow(p.upper_inches)})).filter(p=>p.x!==null).sort((a,b)=>a.x-b.x);}
  function renderSeason(){
    if(!$('season-chart'))return;
    const payload=state.seasons[state.site],loading=state.seasonLoading[state.site],error=state.seasonErrors[state.site],u=units();
    const viewButtons=document.querySelectorAll('[data-season-view]');
    viewButtons.forEach(b=>{const active=b.dataset.seasonView===state.seasonView;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));b.textContent=b.dataset.seasonView==='full'?'Full season':state.site==='mammoth'?'Recent':'Recent + forecast';});
    $('season-tooltip').hidden=true;
    setText('season-title',`${payload?.quantity_label||'Season snowfall'}${payload?.season?' · '+payload.season:''}`);
    const target=payload?.history_target_label||payload?.target_label||'Comparing accumulated snowfall at the observation site.';
    const currentTarget=payload?.current_target_label||payload?.current?.source_label;
    setText('season-target',currentTarget&&currentTarget!==target?`${target} · Current observations: ${currentTarget}`:target);
    setText('season-total-label','Observed this season');setText('season-current-label','Observed this season');
    setText('season-total','—');setText('season-history-count','—');setText('season-observed-through','Waiting for observations');
    $('season-notices').innerHTML='';$('season-sources').innerHTML='';$('season-legend').hidden=true;
    const setStatus=(text,kind='neutral')=>{$('season-status').textContent=text;$('season-status').className=`badge ${kind}`;};
    if(state.imported){
      setStatus('Imported snapshot');setText('season-title','Season snowfall');setText('season-target','Season history is separate from saved forecast exports.');
      $('season-chart').innerHTML=empty('Season history is not part of this snapshot','Select a mountain or refresh to return to current observations. Saved forecasts are never joined to today’s observed total.');
      setText('season-chart-note','Live seasonal observations are hidden while viewing an imported forecast.');return;
    }
    if(!payload){
      setStatus(loading?'Loading history':'Awaiting observations');
      $('season-chart').innerHTML=empty(loading?'Loading the seasonal record':'Seasonal observations unavailable',loading?'Reading past seasons and the latest measured snowfall.':error||'Refresh all to download the available observation records.');
      setText('season-chart-note',state.site==='mammoth'?'Main Lodge reports monthly snowfall and its current season total.':'Past seasons, observations, and the model outlook will appear here as their data become available.');return;
    }
    const history=payload.history.map(h=>({...h,points:seasonPoints(h.points)})).filter(h=>h.points.some(p=>p.v!==null));
    const current=seasonPoints(payload.current.points),prediction=payload.forecast||{},forecast=prediction.status==='unavailable'?[]:seasonPoints(prediction.points);
    const observed=current.filter(p=>p.v!==null),future=forecast.filter(p=>p.v!==null);
    const through=payload.observed_through||observed.at(-1)?.date;
    const partialSubtotal=!!payload.current.total_is_lower_bound;
    const reportedTotal=payload.current.kind==='reported_season_total';
    const monthlyReports=payload.current.source?.cadence==='monthly'||history.some(h=>h.source?.cadence==='monthly');
    const incompleteHistory=history.some(h=>h.complete===false);
    const lowerBoundHistory=history.some(h=>h.total_is_lower_bound===true||(h.total_is_lower_bound===undefined&&h.complete===false&&h.kind!=='reported_season_total'));
    $('season-history-key').hidden=!history.length;$('season-current-key').hidden=!observed.length;$('season-forecast-key').hidden=!future.length;
    $('season-band-key').hidden=!forecast.some((p,i)=>i>0&&p.lo!==null&&p.hi!==null&&p.lo<=p.hi&&forecast[i-1].lo!==null&&forecast[i-1].hi!==null&&forecast[i-1].lo<=forecast[i-1].hi);
    setText('season-history-label',monthlyReports?'Past seasons · monthly totals':lowerBoundHistory?'Past seasons · recorded totals':incompleteHistory?'Past seasons · reporting gaps':'Past seasons');
    setText('season-total',number(payload.total_inches)!==null?`${snowText(payload.total_inches)} ${u.snow}`:'—');
    setText('season-observed-through',through?`Observed through ${seasonDateLabel(through)}`:'No complete current-season total available');
    if(partialSubtotal){setText('season-total-label','Season total');setText('season-total','Unavailable');setText('season-current-label','Recorded so far (incomplete)');setText('season-observed-through',number(payload.current.recorded_total_inches)!==null?`${snowText(payload.current.recorded_total_inches)} ${u.snow} recorded${through?' through '+seasonDateLabel(through):''}`:'Missing observations prevent a complete total');}
    else if(reportedTotal){setText('season-total-label','Reported season total');setText('season-current-label','Reported season total');}
    setText('season-history-count',`${history.length} ${history.length===1?'season':'seasons'}`);
    setStatus(loading?'Updating history':error?'Saved history':monthlyReports&&observed.length?'Resort reports':payload.current.complete&&future.length?'Season + forecast':payload.current.complete?'Observed season':observed.length?'Partial observations':'History available',error||(!monthlyReports&&!payload.current.complete)?'warn':'neutral');
    const notes=[...warningsText(payload.notes),payload.current.coverage_label,error?`The latest record could not be loaded: ${error}`:null].filter(Boolean);
    const coverageNotice=error?'Could not update observations; showing the saved record.':monthlyReports?'Markers show reported totals. Lines connect reports up to one month apart.':partialSubtotal?'The current record is incomplete. Recorded snowfall is a lower bound, not a full season total.':reportedTotal?'The current curve uses reported season totals; unreported dates remain gaps.':!payload.current.complete?'The latest observed date is shown above; missing reports remain gaps.':incompleteHistory?'Some historical dates have no report; their gaps remain visible.':'';
    $('season-notices').innerHTML=coverageNotice?`<p>${esc(coverageNotice)}</p>`:'';
    const safeSource=source=>{let url;try{url=new URL(source.url);if(!['http:','https:'].includes(url.protocol))url=null;}catch{}return url?`<a href="${esc(url.href)}" target="_blank" rel="noopener noreferrer">${esc(source.name||url.hostname)}</a>`:esc(source.name||'Observation archive');};
    const historyCoverage=history.length?`<div class="season-coverage-wrap"><table class="season-coverage"><caption>Historical observation coverage</caption><thead><tr><th scope="col">Season</th><th scope="col">Coverage</th></tr></thead><tbody>${history.map(h=>`<tr><th scope="row">${esc(h.season)}</th><td>${esc(h.coverage_label||(h.complete?'Complete record':'Incomplete observations; recorded values only'))}</td></tr>`).join('')}</tbody></table></div>`:'';
    $('season-sources').innerHTML=`<p>${esc(target)}</p>${(payload.sources||[]).length?`<ul>${payload.sources.map(s=>`<li>${safeSource(s)}</li>`).join('')}</ul>`:''}${[...new Set(notes)].map(note=>`<p>${esc(note)}</p>`).join('')}<p>Historical lines align by calendar date from October through May. Gaps are not filled with zero snowfall. Observation totals describe the named observation site and reporting convention.</p>${historyCoverage}`;
    setText('season-chart-note',monthlyReports?'The resort publishes monthly totals. Storm timing within each month is unknown.':prediction.note||'The dashed forecast continues from the latest complete measured total. It ends at the supported forecast horizon.');
    setText('season-interval-label',prediction.interval_label||'Model prediction range');
    const today=seasonPosition(payload.today),start=seasonPosition(payload.season_start)??0,end=seasonPosition(payload.season_end)??243;
    const historicalPoints=history.flatMap(h=>h.points);
    const fullSeason=state.seasonView==='full';
    let left=start,right=end;
    if(fullSeason&&history.length){
      // Past curves can contain missing-value placeholders through May 31.
      // A reported zero or an unchanged total still establishes coverage.
      right=Math.min(end,historicalPoints.reduce((last,p)=>p.v===null?last:Math.max(last,p.x),start));
    }else if(!fullSeason){
      const center=today??observed.at(-1)?.x??start;
      left=Math.max(start,center-14);right=Math.min(end,Math.max(center+3,future.at(-1)?.x??center+3));
      if(right<=left)left=Math.max(start,right-17);
    }
    const forecastInWindow=future.some(p=>p.x>=left&&p.x<=right&&p.x>(observed.at(-1)?.x??-Infinity));
    setText('season-window-note',`${seasonAxisLabel(left)} – ${seasonAxisLabel(right)}${forecastInWindow?' · forecast up to 48 hours':''}`);
    const all=[...historicalPoints,...current,...forecast];
    if(!all.some(p=>p.v!==null&&p.x>=left&&p.x<=right)){
      $('season-chart').innerHTML=empty('No measurements in this range',state.seasonView==='recent'?'Choose Full season to see the historical records. Missing snowfall observations are not treated as zero.':'Available records do not yet contain a measured seasonal curve. Refresh all to check for updates.');return;
    }
    $('season-legend').hidden=false;
    const width=1000,height=350,pad={l:57,r:25,t:35,b:43},plotW=width-pad.l-pad.r,plotH=height-pad.t-pad.b;
    const maximum=all.reduce((v,p)=>p.x>=left&&p.x<=right?Math.max(v,p.v??0,p.hi??0):v,0);
    const {top,ticks:valueTicks}=seasonValueAxis(maximum);
    const x=v=>pad.l+(v-left)/Math.max(.01,right-left)*plotW,y=v=>height-pad.b-v/top*plotH;
    let svg=svgGrid(width,height,pad,top,`${payload.quantity_label||'Accumulated snowfall'} · ${u.snow}`,valueTicks);
    svg+='<defs><clipPath id="seasonPlotClip"><rect x="'+pad.l+'" y="'+pad.t+'" width="'+plotW+'" height="'+plotH+'"/></clipPath></defs>';
    const monthlyTicks=fullSeason&&right-left>=60;
    const labelCutoff=monthlyTicks&&right<end;
    const tickDays=Math.min(5,Math.floor(right-left));
    const ticks=monthlyTicks?[0,31,61,92,123,152,183,213].filter(v=>v>=left&&v<=(labelCutoff?right-10:right))
      :tickDays>0?Array.from({length:tickDays+1},(_,i)=>i===tickDays?right:left+Math.round((right-left)*i/tickDays)):[right];
    if(labelCutoff)ticks.push(right);
    ticks.forEach(day=>{svg+=`<text x="${x(day)}" y="${height-18}" fill="#7f9192" font-size="11" text-anchor="middle">${esc(seasonAxisLabel(day,!monthlyTicks||(labelCutoff&&day===right)))}</text>`;});
    const pathFor=(rows,key='v',maxGapHours=Infinity)=>{let path='',last=null;for(const p of rows){const amount=p[key];if(amount===null||amount===undefined){last=null;continue;}const gap=last&&Date.parse(p.date)-Date.parse(last.date)>maxGapHours*3600000;path+=`${last&&!gap?'L':'M'}${x(p.x).toFixed(2)} ${y(amount).toFixed(2)} `;last=p;}return path;};
    svg+='<g clip-path="url(#seasonPlotClip)">';
    for(const h of history){
      const monthly=h.source?.cadence==='monthly',points=monthly?h.points.filter(p=>p.v!==null):h.points;
      svg+=`<path d="${pathFor(points,'v',monthly?31*24:36)}" fill="none" stroke="#4a7eac" stroke-opacity=".2" stroke-width="1.5" vector-effect="non-scaling-stroke"><title>${esc(h.season)}${h.complete===false?' · incomplete observations':''}</title></path>`;
      if(monthly)for(const p of points)svg+=`<circle cx="${x(p.x)}" cy="${y(p.v)}" r="2.5" fill="#4a7eac" opacity=".35"><title>${esc(h.season)} · ${esc(seasonDateLabel(p.date))} · ${esc(amountText(p.v))} ${esc(u.snow)}</title></circle>`;
    }
    // Separate uncertainty polygons at missing values instead of spanning gaps.
    let band=[];const bands=[];for(const p of forecast){if(p.lo!==null&&p.hi!==null&&p.lo<=p.hi){band.push(p);}else if(band.length){bands.push(band);band=[];}}if(band.length)bands.push(band);
    for(const points of bands){if(points.length<2)continue;const outline=points.map(p=>`${x(p.x).toFixed(2)} ${y(p.hi).toFixed(2)}`).concat([...points].reverse().map(p=>`${x(p.x).toFixed(2)} ${y(p.lo).toFixed(2)}`));svg+=`<path d="M${outline.join('L')}Z" fill="#be7c3a" fill-opacity=".16"/>`;}
    svg+=`<path d="${pathFor(monthlyReports?observed:current,'v',monthlyReports?31*24:36)}" fill="none" stroke="#176f73" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>`;
    for(let i=0;i<current.length;i++){
      const p=current[i];if(p.v===null)continue;
      const previous=current[i-1],next=current[i+1],last=p===observed.at(-1);
      const isolated=(!previous||previous.v===null||Date.parse(p.date)-Date.parse(previous.date)>36*3600000)&&(!next||next.v===null||Date.parse(next.date)-Date.parse(p.date)>36*3600000);
      if((monthlyReports||isolated)&&!last)svg+=`<circle cx="${x(p.x)}" cy="${y(p.v)}" r="2.5" fill="#176f73" stroke="white" stroke-width="1.5"/>`;
    }
    svg+=`<path d="${pathFor(forecast)}" fill="none" stroke="#be7c3a" stroke-width="2.6" stroke-dasharray="7 5" stroke-linecap="round" vector-effect="non-scaling-stroke"/>`;
    for(const p of future)if(p.x>=left&&p.x<=right)svg+=`<circle cx="${x(p.x)}" cy="${y(p.v)}" r="2.5" fill="#be7c3a"/>`;
    if(today!==null&&today>=left&&today<=right)svg+=`<path d="M${x(today)} ${pad.t}V${height-pad.b}" stroke="#9caeaa" stroke-dasharray="3 5" stroke-width="1"/><text x="${Math.min(width-pad.r-30,Math.max(pad.l+20,x(today)))}" y="${pad.t+13}" font-size="10" fill="#6e8580" text-anchor="middle">Today</text>`;
    svg+='<path id="season-hover-line" d="" stroke="#526f74" stroke-opacity=".45" stroke-dasharray="2 4"/></g>';
    const lastPoint=observed.at(-1);
    if(lastPoint&&lastPoint.x>=left&&lastPoint.x<=right){
      svg+=`<circle cx="${x(lastPoint.x)}" cy="${y(lastPoint.v)}" r="4" fill="#176f73" stroke="white" stroke-width="1.5"/>`;
    }
    $('season-chart').innerHTML=`<svg viewBox="0 0 ${width} ${height}" tabindex="0" role="img" aria-label="${esc(siteInfo().name)} accumulated snowfall. ${monthlyReports?'Markers show reported totals and faint lines connect monthly reports.':'Faint lines show past seasons, solid teal shows current observations, and dashed amber shows available model predictions.'} Use left and right arrow keys to inspect dates." aria-describedby="season-chart-note">${svg}</svg>`;
    const chart=$('season-chart').querySelector('svg');let cursor=Math.max(left,Math.min(right,today??left));
    const nearest=(rows,at)=>rows.reduce((best,p)=>p.v!==null&&Math.abs(p.x-at)<=.55&&(!best||Math.abs(p.x-at)<Math.abs(best.x-at))?p:best,null);
    const inspect=at=>{
      cursor=Math.max(left,Math.min(right,at));const c=nearest(current,cursor),f=nearest(forecast,cursor),past=history.map(h=>nearest(h.points,cursor)?.v).filter(v=>v!==null&&v!==undefined);
      const pieces=[seasonAxisLabel(cursor),c?`${partialSubtotal?'Recorded subtotal':reportedTotal?'Reported season total':'Observed'}: ${amountText(c.v)} ${u.snow}`:'Observed total unavailable'];
      if(f){pieces.push(`Our model: ${amountText(f.v)} ${u.snow}`);if(f.lo!==null&&f.hi!==null)pieces.push(`Range: ${amountText(f.lo)}–${amountText(f.hi)} ${u.snow}`);}
      if(past.length)pieces.push(`Past recorded accumulations: ${amountText(Math.min(...past))}–${amountText(Math.max(...past))} ${u.snow} (${past.length} seasons)`);
      $('season-tooltip').textContent=pieces.join(' · ');$('season-tooltip').hidden=false;
      $('season-hover-line').setAttribute('d',`M${x(cursor)} ${pad.t}V${height-pad.b}`);
    };
    chart.addEventListener('pointermove',event=>{const rect=chart.getBoundingClientRect();inspect(left+((event.clientX-rect.left)/rect.width*width-pad.l)/plotW*(right-left));});
    chart.addEventListener('pointerleave',()=>{if(document.activeElement!==chart){$('season-tooltip').hidden=true;$('season-hover-line').setAttribute('d','');}});
    chart.addEventListener('focus',()=>inspect(cursor));
    chart.addEventListener('keydown',event=>{if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();inspect(cursor+(event.key==='ArrowRight'?1:-1));}else if(event.key==='Escape'){$('season-tooltip').hidden=true;$('season-hover-line').setAttribute('d','');}});
  }
  function svgGrid(w,h,pad,top,ylabel,ticks=[0,top/4,top/2,3*top/4,top]){
    const precision=Math.min(6,Math.max(0,Math.ceil(-Math.log10(top/4))+2));
    const tickLabel=v=>v!==0&&top<.00001?v.toExponential(1):v.toLocaleString('en-US',{maximumFractionDigits:precision});
    let out='';
    for(const v of ticks){
      const y=h-pad.b-(h-pad.t-pad.b)*v/top;
      out+=`<path d="M${pad.l} ${y}H${w-pad.r}" stroke="#e8ede7" stroke-dasharray="3 4"/><text x="${pad.l-10}" y="${y+3}" fill="#8a9996" font-size="9" text-anchor="end">${esc(tickLabel(v))}</text>`;
    }
    return out+`<text x="${pad.l}" y="12" fill="#8a9996" font-size="9">${esc(ylabel)}</text>`;
  }
  function renderSnow(rows,paired){
    $('snow-tooltip').hidden=true;
    if(!rows.length){$('snow-chart').innerHTML=paired?empty('Hourly comparison data unavailable','Refresh all to fetch the hourly forecast for this model window.'):empty('Your next weather window starts here','Choose a mountain and refresh to pull the latest hourly weather forecast.');return;}
    const vals=rows.map(r=>displaySnow(r.snowfall_inches)),valid=vals.some(v=>v!==null);if(!valid){$('snow-chart').innerHTML=empty('Snowfall guidance unavailable','Temperature, wind and other fields may still be available below.');return;}
    if(state.view==='cumulative'&&vals.some(v=>v===null)){$('snow-chart').innerHTML=empty('Cumulative snowfall is unavailable','At least one hour has a missing snowfall value. Switch to Hourly to inspect the available intervals.');return;}
    const width=740,height=270,pad={l:44,r:12,t:27,b:36},plotW=width-pad.l-pad.r,plotH=height-pad.t-pad.b;
    let acc=0,missing=false;const vv=vals.map(v=>{if(v===null)missing=true;return state.view==='hourly'?v:missing?null:(acc+=v);});
    const top=niceTop(Math.max(0,...vv.filter(v=>v!==null))*1.15),y=v=>height-pad.b-plotH*v/top;
    const u=units();
    let svg=svgGrid(width,height,pad,top,`${u.snowLong}${state.view==='hourly'?' / hour':' accumulated'}`);
    if(state.view==='hourly'){
      rows.forEach((r,i)=>{if(vv[i]===null)return;const bw=Math.max(1,Math.min(plotW/rows.length*.58,42)),x=pad.l+plotW*(i+.5)/rows.length-bw/2;svg+=`<rect x="${x}" y="${y(vv[i])}" width="${bw}" height="${Math.max(0,plotH*vv[i]/top)}" rx="2" fill="${snowColor(r.snowfall_inches,1)}"/>`;});
      // Draw only forecast zeros; missing hours remain gaps.
      let zeroStart=null;for(let i=0;i<=vals.length;i++){if(i<vals.length&&vals[i]===0){if(zeroStart===null)zeroStart=i;}else if(zeroStart!==null){svg+=`<path class="zero-snow-line" d="M${pad.l+plotW*zeroStart/rows.length} ${y(0)}H${pad.l+plotW*i/rows.length}" fill="none" stroke="${snowColor(0,1)}" stroke-width="2.5" stroke-linecap="round" vector-effect="non-scaling-stroke"/>`;zeroStart=null;}}
    }else{
      let path=`M${pad.l} ${y(0)}`,canonicalTotal=0;
      const stops=[`<stop offset="0%" stop-color="${snowColor(0,24)}"/>`];
      vv.forEach((v,i)=>{if(v!==null){path+=`L${pad.l+plotW*(i+1)/rows.length} ${y(v)}`;canonicalTotal+=rows[i].snowfall_inches;stops.push(`<stop offset="${100*(i+1)/rows.length}%" stop-color="${snowColor(canonicalTotal,24)}"/>`);}});
      svg+=`<defs><linearGradient id="snowAmountGradient" gradientUnits="userSpaceOnUse" x1="${pad.l}" y1="0" x2="${width-pad.r}" y2="0">${stops.join('')}</linearGradient></defs><path d="${path}L${width-pad.r} ${y(0)}Z" fill="url(#snowAmountGradient)" opacity=".16"/><path d="${path}" fill="none" stroke="url(#snowAmountGradient)" stroke-width="2.6" vector-effect="non-scaling-stroke"/>`;
    }
    if(vals.every(v=>v===0))svg+=`<text x="${width/2}" y="${height/2-5}" fill="#8a9996" font-size="11" text-anchor="middle">No snowfall signaled in this weather window</text>`;
    const labelCount=Math.min(rows.length,6),ids=new Set(Array.from({length:labelCount},(_,i)=>Math.round(i*(rows.length-1)/Math.max(1,labelCount-1))));
    rows.forEach((r,i)=>{const x=pad.l+plotW*(i+.5)/rows.length;if(ids.has(i))svg+=`<text x="${x}" y="${height-14}" text-anchor="middle" fill="#8a9996" font-size="9">${esc(localHour(r.time))}</text>`;svg+=`<rect data-point="${i}" x="${pad.l+plotW*i/rows.length}" y="${pad.t}" width="${plotW/rows.length}" height="${plotH}" fill="transparent" tabindex="0" role="button" aria-label="${esc(dateTime(r.time))}: ${esc(amountText(vv[i],2))} ${esc(u.snowLong)}"><title>${esc(dateTime(r.time))}: ${esc(amountText(vv[i],2))} ${esc(u.snowLong)}</title></rect>`;});
    $('snow-chart').innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(state.view==='hourly'?'Hourly':'Cumulative')} snowfall guidance for ${esc(siteInfo().name)}">${svg}</svg>`;
    $('snow-chart').querySelectorAll('[data-point]').forEach(el=>{const show=()=>{const i=Number(el.dataset.point);$('snow-tooltip').textContent=`Hour ending ${dateTime(rows[i].time)} · ${amountText(vv[i],2)} ${u.snowLong}${state.view==='cumulative'?' accumulated':''}${rows[i].temperature_c!==null?' · '+fmt(displayTemp(rows[i].temperature_c),0)+u.temp:''}`;$('snow-tooltip').hidden=false;};el.addEventListener('mouseenter',show);el.addEventListener('focus',show);el.addEventListener('click',show);});
  }
  function selectedDeployment(deployed=state.forecast?.deployed){
    return Array.isArray(deployed?.horizons)?deployed.horizons.find(row=>row.hours===state.hours):null;
  }
  function deploymentQuantity(row){
    if(row?.quantity==='snow_water_equivalent'&&row.unit==='mm')return {convert:displayLiquid,unit:`${units().liquid} SWE`,noun:'snow water equivalent',short:'snow water'};
    if(row?.quantity==='fresh_snow_depth'&&row.unit==='inches')return {convert:displaySnow,unit:units().snow,noun:'fresh snow depth',short:'fresh snow'};
    if(row?.quantity==='snowfall'&&row.unit==='inches')return {convert:displaySnow,unit:units().snow,longUnit:units().snowLong,noun:'snowfall',short:'snow'};
    if(row?.quantity==='liquid_precipitation'&&row.unit==='mm')return {convert:displayLiquid,unit:`${units().liquid} water`,longUnit:`${units().liquid} water`,noun:'liquid precipitation',short:'water'};
    return null;
  }
  function deploymentAvailable(row){
    const start=Date.parse(row?.start),end=Date.parse(row?.end);
    return !!deploymentQuantity(row)&&row?.status!=='unavailable'&&number(row.mean)!==null&&Number(row.mean)>=0&&Number.isFinite(start)&&end-start===row.hours*3600000;
  }
  function renderHorizonList(deployed){
    setText('daily-eyebrow','');setText('daily-title','Our model');setText('daily-subtitle','The selected forecast for each horizon. Short radar forecasts measure liquid precipitation.');
    $('daily-color-legend').hidden=true;
    setText('daily-model-note','Each amount covers its stated window. A dash means the matching forecast is unavailable.');
    $('daily-cards').classList.add('horizon-outlook');
    const rows=forecastHorizons.map(hours=>deployed?.horizons?.find(row=>row.hours===hours)||{hours,status:'unavailable'});
    $('daily-cards').innerHTML=rows.map(row=>{
      const q=deploymentQuantity(row),available=deploymentAvailable(row),saved=row.stale||row.window_started||row.expired||Date.parse(row.start)<Date.now();
      const amount=available?`${amountText(q.convert(row.mean),2)} ${q.unit}`:'—';
      const window=available?`${saved?'Saved · ':''}${date(row.start,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})} → ${date(row.end,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})}`:'Refresh to check availability';
      return `<button class="horizon-outlook-row${row.hours===state.hours?' active':''}" type="button" data-outlook-hours="${row.hours}" aria-pressed="${row.hours===state.hours}"><span class="horizon-outlook-label">${row.hours}h</span><span><strong>${esc(amount)}</strong><small>${esc(window)}</small></span></button>`;
    }).join('');
    $('daily-cards').querySelectorAll('[data-outlook-hours]').forEach(button=>button.addEventListener('click',()=>{state.hours=Number(button.dataset.outlookHours);syncForecastURL();render();}));
  }
  function renderDeployed(deployed,outlook){
    const row=selectedDeployment(deployed),q=deploymentQuantity(row),available=deploymentAvailable(row),unit=q?.unit||(state.hours<=3?`${units().liquid} water`:units().snow);
    $('neural-panel').hidden=false;$('neural-eyebrow').hidden=true;$('neural-subtitle').hidden=true;
    setText('neural-title',comparisonTitle(row?.start,row?.end,unit));
    setText('model-summary-label',`Our model · ${state.hours}h${q?' '+q.short:''}`);
    setText('daily-snow-unit',q?.unit||unit);
    setText('daily-snow-amount',available?amountText(q.convert(row.mean),2):'—');
    $('daily-snow-amount').style.color=available?modelDisplay.color:'';
    const saved=available&&(row.stale||row.window_started||row.expired||Date.parse(row.start)<Date.now());
    setText('daily-snow-description',available?`${saved?'Saved window':'Window'} · ${date(row.start,{month:'short',day:'numeric',hour:'numeric'})} → ${date(row.end,{month:'short',day:'numeric',hour:'numeric'})}`:'Refresh all to check the selected forecast');
    renderHorizonList(deployed);
    if(!available){
      $('neural-chart').innerHTML=empty('Current comparison unavailable',displayNotes(row?.warnings||deployed?.warnings).join(' ')||'Refresh all to fetch radar and HRRR data and run the selected models.');
      setText('neural-status','Refresh needed');$('neural-status').className='badge warn';$('neural-legend').replaceChildren();$('neural-details').replaceChildren();setText('neural-note','Missing forecasts are not shown as zero.');return;
    }
    setText('neural-status',saved?'Saved window':row.status==='fallback'?'Fallback active':'Ready');$('neural-status').className=`badge ${saved||row.status==='fallback'?'warn':'live'}`;
    const providerAmount=row.quantity==='snowfall'?(outlook.total??null):number(row.provider_mean);
    const providerName=row.provider_name||'Open-Meteo';
    const series=[{label:providerName,color:'#167d87',amount:providerAmount}];
    if(row.quantity==='snow_water_equivalent'&&number(row.ecmwf_mean)!==null)
      series.push({label:'ECMWF',color:'#507bb6',amount:row.ecmwf_mean});
    series.push({label:'Our model',color:modelDisplay.color,amount:Number(row.mean)});
    const lo=number(row.lower),hi=number(row.upper),hasInterval=lo!==null&&hi!==null&&lo>=0&&hi>=lo;
    const w=1120,h=260,pad={l:55,r:18,t:30,b:42},plotW=w-pad.l-pad.r,plotH=h-pad.t-pad.b;
    const values=[...series.map(entry=>q.convert(entry.amount)),hasInterval?q.convert(hi):null].filter(v=>v!==null);
    const top=niceTop(Math.max(q.convert(.1),...values)*1.2),yy=v=>h-pad.b-v/top*plotH;
    const barW=75,gap=barW*.18,clusterW=series.length*barW+(series.length-1)*gap,center=pad.l+plotW/2;
    let svg=svgGrid(w,h,pad,top,q.unit);
    series.forEach((entry,i)=>{
      const x=center-clusterW/2+i*(barW+gap),value=q.convert(entry.amount);
      const label=`${entry.label} · ${dateTime(row.start)} to ${dateTime(row.end)} · ${value===null?'Unavailable':amountText(value,2)+' '+q.unit}`;
      if(value===null){svg+=`<text x="${x+barW/2}" y="${yy(0)-8}" text-anchor="middle" fill="#8a9996" font-size="13"><title>${esc(label)}</title>—</text>`;return;}
      svg+=value===0?`<path d="M${x} ${yy(0)}h${barW}" stroke="${entry.color}" stroke-width="3"><title>${esc(label)}</title></path>`:`<rect x="${x}" y="${yy(value)}" width="${barW}" height="${yy(0)-yy(value)}" rx="2" fill="${entry.color}"><title>${esc(label)}</title></rect>`;
      svg+=`<text x="${x+barW/2}" y="${yy(value)-8}" text-anchor="middle" fill="${entry.color}" font-size="12">${esc(amountText(value,2))}</text>`;
    });
    if(hasInterval){
      const x=center+clusterW/2+18;
      svg+=`<path d="M${x} ${yy(q.convert(lo))}V${yy(q.convert(hi))}M${x-6} ${yy(q.convert(lo))}H${x+6}M${x-6} ${yy(q.convert(hi))}H${x+6}" fill="none" stroke="${modelDisplay.color}" stroke-width="2" opacity=".65"><title>${esc(row.interval_label||'Model range')}: ${esc(amountText(q.convert(lo),2))}–${esc(amountText(q.convert(hi),2))} ${esc(q.unit)}</title></path>`;
    }
    svg+=`<text x="${center}" y="${h-15}" text-anchor="middle" fill="#627c82" font-size="11">${state.hours}-hour accumulation</text>`;
    $('neural-chart').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(siteInfo().name)} ${state.hours}-hour ${esc(q.noun)}: ${esc(providerName)} and Our model">${svg}</svg>`;
    $('neural-legend').innerHTML=series.map(entry=>`<span><i class="legend-line" style="background:${entry.color}"></i>${esc(entry.label)}</span>`).join('');
    const range=hasInterval?`${row.interval_label||'Model range'}: ${amountText(q.convert(lo),2)}–${amountText(q.convert(hi),2)} ${q.unit}. `:'';
    setText('neural-note',`${range}${row.quantity==='snow_water_equivalent'?'SWE is the depth of water the falling snow would yield when melted. ':row.quantity==='liquid_precipitation'?'Liquid-equivalent precipitation, not fresh-snow depth. ':'Snowfall accumulation. '}Both bars cover the same stated window and target. Missing provider guidance is shown as a dash.`);
    const notes=[outlook.legacyPair?'This snapshot has separate hourly guidance. Refresh all to obtain hourly data for the Open-Meteo comparison.':null,saved?'This is a saved forecast window. Refresh all for current guidance.':null,...displayNotes(row.warnings),deployed?.target?.note].filter(Boolean);
    $('neural-details').innerHTML=[...new Set(notes)].map(text=>`<p>${esc(text)}</p>`).join('');
  }

  function mammothDisplayForecast(payload) {
    const deployed = payload?.deployed;
    if (!deployed || state.mammothQuantity !== 'depth') return deployed;
    const readAt=Date.parse(payload.source?.retrieved_at);
    const timely=Number.isFinite(readAt) && readAt<=Date.parse(deployed.start);
    const hoursByStart = new Map((timely ? payload.hourly : []).map(row => [Date.parse(row.start),row]));
    const hourly = (deployed.hourly || []).map(row => {
      const weather=hoursByStart.get(Date.parse(row.start));
      const value=weather && Date.parse(weather.time)===Date.parse(row.end) ? number(weather.snowfall_inches) : null;
      return {...row,display_depth:number(row.depth_inches),provider_depth:value};
    });
    const horizons=(deployed.horizons || []).map(row => {
      const values=hourly.slice(0,row.hours).map(h=>h.provider_depth);
      const matched=values.length===row.hours&&values.every(v=>v!==null);
      return {...row,quantity:'fresh_snow_depth',unit:'inches',mean:number(row.depth_inches),
        provider_name:'Open-Meteo',provider_mean:matched?values.reduce((a,b)=>a+b,0):null,
        ecmwf_mean:null,lower:null,upper:null,interval_label:null,warnings:[]};
    });
    return {...deployed, hourly, horizons, target:{name:'Mammoth Main Lodge',
      note:'Hourly timing comes from issued weather forecasts. Depth calibration and error estimates use Main Lodge monthly reports.'}};
  }

  function renderMammothCurve(deployed) {
    const cumulative = state.sweView === 'cumulative';
    const depth = state.mammothQuantity === 'depth';
    const modelKey=depth?'display_depth':'mean_mm', refKey=depth?'provider_depth':'provider_mm';
    const reference=depth?'Open-Meteo':'HRRR', convert=depth?displaySnow:displayLiquid;
    setText('mammoth-model-title',depth?'Hourly fresh-snow estimate':'Hourly snow-water forecast');
    setText('mammoth-model-target',depth?'Main Lodge · depth calibrated against monthly totals':'Jeff Dozier Snow Study Site · optical snow sensor · 2,936 m');
    setText('swe-reference-name',reference);
    document.querySelectorAll('[data-mammoth-quantity]').forEach(button => {
      const selected = button.dataset.mammothQuantity === state.mammothQuantity;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    document.querySelectorAll('[data-swe-view]').forEach(button => {
      const selected = button.dataset.sweView === state.sweView;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    const rows = (deployed?.hourly || []).slice(0, state.hours);
    const container = $('swe-chart');
    $('swe-ecmwf-key').hidden=true;
    if (!rows.some(row => number(row[modelKey]) !== null)) {
      container.innerHTML = empty('Hourly model forecast unavailable', 'Refresh all to acquire the forecast inputs.');
      setText('swe-coverage','');
      return;
    }
    const unit = depth ? units().snow : units().liquid + ' SWE';
    const series = [
      {key:refKey, label:reference, color:'#167d87'},
      {key:modelKey, label:'Our model', color:modelDisplay.color},
    ];
    const hasEcmwf = !depth && rows.some(row => number(row.ecmwf_mm)!==null);
    if (hasEcmwf) series.splice(1,0,{key:'ecmwf_mm',label:'ECMWF',color:'#507bb6'});
    $('swe-ecmwf-key').hidden=!hasEcmwf;
    for (const item of series) {
      let sum = 0, complete = true;
      item.values = rows.map(row => {
        const amount = number(row[item.key]);
        if (amount === null) { complete = false; return null; }
        sum += amount;
        return cumulative ? (complete ? convert(sum) : null) : convert(amount);
      });
    }
    const w=920, h=238, pad={l:56,r:28,t:28,b:35};
    const top = niceTop(Math.max(convert(.05), ...series.flatMap(item => item.values.filter(n => n !== null))) * 1.1);
    const x = hour => pad.l + (w - pad.l - pad.r) * hour / state.hours;
    const y = value => h-pad.b - (h-pad.b-pad.t)*value/top;
    let svg = svgGrid(w,h,pad,top,unit+(cumulative?'':' per hour'));
    const ticks = [...new Set([0, ...Array.from({length:Math.min(6,state.hours)}, (_,i) =>
      Math.round((i+1) * state.hours / Math.min(6,state.hours)))])];
    for (const tick of ticks) {
      const stamp = tick === 0 ? rows[0].start : rows[tick-1]?.end;
      if(stamp) svg += `<text x="${x(tick)}" y="${h-9}" fill="#63797c" font-size="11" text-anchor="middle">${esc(localHour(stamp))}</text>`;
    }
    for (const item of series) {
      let last = cumulative ? [x(0),y(0)] : null;
      let lines = '', dots = '';
      item.values.forEach((v,i) => {
        if (v === null) { last=null; return; }
        const cx=x(i+1), cy=y(v);
        if (last) lines += `M${last[0]} ${last[1]}L${cx} ${cy} `;
        else if(!cumulative && i===0) lines+=`M${x(0)} ${cy}H${cx} `;
        const title=`${item.label} · ${dateTime(rows[i].end)} · ${amountText(v,3)} ${unit}`;
        dots += `<circle cx="${cx}" cy="${cy}" r="2.4" fill="${item.color}"><title>${esc(title)}</title></circle>`;
        last=[cx,cy];
      });
      svg += `<path d="${lines}" fill="none" stroke="${item.color}" stroke-width="${item.key===refKey?3.5:2}" ${item.key===modelKey?'stroke-dasharray="6 3"':''}/>${dots}`;
    }
    container.innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${cumulative?'Cumulative':'Hourly'} ${depth?'fresh snow depth':'snow-water equivalent'} from Our model and weather forecasts">${svg}</svg>`;
    const modelCount=rows.filter(r=>number(r[modelKey])!==null).length;
    const hrrrCount=rows.filter(r=>number(r[refKey])!==null).length;
    const row=selectedDeployment(deployed);
    setText('swe-coverage', `${modelCount} model hours · ${hrrrCount} ${reference} hours · Pacific time.`+
      (row?.stale||row?.window_started?' Saved forecast window.':'')+
      (!depth&&hrrrCount<state.hours?' The HRRR line ends where its issued forecast ends.':''));
  }

  function renderMammothRadar(observation){
    const rate=number(observation?.precip_rate_mm_h);
    const usable=observation?.coverage==='usable'&&Number.isFinite(Date.parse(observation.timestamp))&&rate!==null&&rate>=0;
    setText('radar-rate',usable?amountText(displayLiquid(rate),metric()?2:3):'—');
    setText('radar-rate-unit',units().liquid+'/h');
    setText('radar-time',observation?.timestamp?'Radar scan · '+dateTime(observation.timestamp):'No radar scan available');
    const quality=number(observation?.quality_index);
    const qualityTime=observation?.quality_timestamp;
    const qualityLabel=quality===null?'':`Radar quality index ${fmt(quality,2)} / 1${qualityTime&&qualityTime!==observation.timestamp?' · '+dateTime(qualityTime):''}.`;
    setText('radar-coverage',[usable?'':observation?.message||'The radar rate is unavailable for this scan.',qualityLabel].filter(Boolean).join(' '));
  }
  function renderSesame(observation){
    const observed=observation?.observed_at_utc;
    const available=Number.isFinite(Date.parse(observed))&&['ready','stale'].includes(observation?.status);
    setText('sesame-time',available?(observation.status==='stale'?'Saved observation · ':'Observed ')+dateTime(observed):'Station observations unavailable');
    const readings=[];
    if(available){
      const u=units();
      if(number(observation.temperature_f)!==null)readings.push(['Temperature',fmt(metric()?(observation.temperature_f-32)*5/9:observation.temperature_f,1)+' '+u.temp]);
      for(const [key,label] of [['snow_depth_inches','Snow depth'],['snow_board_inches','Snow on the board']]){
        if(number(observation[key])!==null)readings.push([label,snowText(observation[key],1)+' '+u.snow]);
      }
      for(const [key,label] of [['wind_mph','Wind speed'],['wind_gust_mph','Wind gust']]){
        if(number(observation[key])!==null)readings.push([label,fmt(observation[key]*(metric()?1.609344:1),0)+' '+u.wind]);
      }
    }
    $('sesame-values').innerHTML=readings.map(([name,value])=>`<div><dt>${esc(name)}</dt><dd>${esc(value)}</dd></div>`).join('');
  }
  function renderWeather(rows){
    const u=units();
    if(!rows.length){$('weather-table-wrap').innerHTML=empty('Weather details will appear here','Hourly temperature, precipitation and wind arrive with the weather refresh.');$('weather-period').textContent='';return;}
    $('weather-period').textContent=`${date(rows[0].time,{month:'short',day:'numeric'})} – ${date(rows.at(-1).time,{month:'short',day:'numeric'})}`;
    const indices=rows.length>12?Array.from({length:12},(_,i)=>Math.round(i*(rows.length-1)/11)):rows.map((_,i)=>i),sample=indices.map(i=>rows[i]);
    const row=(label,unit,fn,strong=false)=>`<tr><td>${label}<small>${unit}</small></td>${sample.map(r=>`<td${strong?' class="table-cell-strong"':''}>${fn(r)}</td>`).join('')}</tr>`;
    $('weather-table-wrap').innerHTML=`<table class="weather-table"><caption class="skip-link">Selected hourly weather forecast, sampled across the selected horizon</caption><thead><tr><th scope="col">Local time</th>${sample.map(r=>`<th scope="col" title="${esc(dateTime(r.time))}">${esc(localHour(r.time))}<br><span style="font-size:8px">${esc(date(r.time,{weekday:'short'}))}</span></th>`).join('')}</tr></thead><tbody>${row('Fresh snowfall',`${u.snowLong} / hour`,r=>snowText(r.snowfall_inches,2),true)}${row('Temperature',u.tempLong,r=>r.temperature_c!==null?fmt(displayTemp(r.temperature_c),0)+'°':'—')}${row('Wind',u.windLong,r=>fmt(displayWind(r.wind_kmh),0))}${row('Precipitation',`liquid equivalent · ${u.liquid}`,r=>fmt(displayLiquid(r.precipitation_mm),metric()?1:3))}</tbody></table>`;
  }
  function renderSources(p,m,source){
    const deployed=p?.deployed,row=selectedDeployment(deployed);
    const mammoth=m.id==='mammoth';
    const capabilities=mammoth?[
      ['Hourly snowfall model','The model predicts readings from the Jeff Dozier optical snow sensor. Its largest reported intensities have substantial uncertainty. The comparisons use issued HRRR and ECMWF forecasts.'],
      ['Fresh-snow depth','Main Lodge depth uses a separate conversion fitted to resort snowfall reports. The measured errors cover months. Hourly timing follows the weather forecast.'],
      ['Radar observations','NOAA’s Multi-Radar/Multi-Sensor system estimates liquid precipitation. Snowline checks the local radar quality before showing the rate.'],
      ['Season observations','Mammoth Mountain publishes monthly snowfall and the season-to-date total at Main Lodge.']
    ]:[['One selected model per horizon','The comparison uses the saved selection for this mountain and horizon. Refresh updates its inputs without retraining.'],['Radar and HRRR','Short-range radar forecasts measure liquid precipitation. HRRR provides the fallback where radar coverage or verification is insufficient.'],['Snowfall accumulation','The 12/24/48-hour models are evaluated at their stated accumulation windows; those scores do not establish hourly snowfall timing.'],['Observation target',deployed?.target?.note||m.target_note||m.target_label||'The available mountain-area station observations.']];
    $('capability-list').innerHTML=capabilities.map(([title,body])=>`<div class="capability"><span class="capability-mark">✓</span><span><strong>${esc(title)}</strong><br>${esc(body)}</span></div>`).join('');
    const warnings=displayNotes(p?.warnings);let details='';
    if(row){
      details+=`<p><strong>Our model · ${state.hours}h implementation:</strong> ${esc(row.method_id||'Unavailable')}${row.issued_at?'<br>Issued '+esc(dateTime(row.issued_at)):''}</p>`;
      if(row.status==='fallback')details+='<p>A fallback is active for this horizon. The model and provider bars still use the same quantity and valid window.</p>';
      if(row.interval_label)details+=`<p><strong>Model range:</strong> ${esc(row.interval_label)}</p>`;
      const timestamps=[],seen=new Set();
      const collect=(value,path=[])=>{if(!value||typeof value!=='object')return;for(const [key,child] of Object.entries(value)){const next=[...path,key];if(typeof child==='string'&&/issued|retrieved|run_time|available.*utc/.test(key)&&Number.isFinite(Date.parse(child))){const note=`${next.join(' · ')}: ${dateTime(child)}`;if(!seen.has(note)){timestamps.push(note);seen.add(note);}}else if(child&&typeof child==='object')collect(child,next);}};
      collect(row.sources);
      if(timestamps.length)details+=`<p><strong>Component source times</strong></p><ul>${timestamps.slice(0,24).map(text=>`<li>${esc(text.replaceAll('_',' '))}</li>`).join('')}</ul>`;
      if(displayNotes(row.warnings).length)details+=`<ul>${displayNotes(row.warnings).map(text=>`<li>${esc(text)}</li>`).join('')}</ul>`;
    }
    if(deployed?.target?.name)details+=`<p><strong>Model observation target:</strong> ${esc(deployed.target.name)}${number(deployed.target.elevation_m)!==null?' · '+fmt(deployed.target.elevation_m,0)+' m':''}. ${esc(deployed.target.note||'')}</p>`;
    if(source.target_label||m.target_note)details+=`<p><strong>External hourly forecast target:</strong> ${esc(source.target_label||m.target_note)}</p>`;
    const ds=p?.daily_source;if(ds?.name)details+=`<p><strong>Daily model weather source:</strong> ${esc(ds.name)}${ds.issued_at?'<br>Issued '+esc(dateTime(ds.issued_at)):''}${ds.retrieved_at?'<br>Retrieved '+esc(dateTime(ds.retrieved_at)):''}</p>`;
    if(source.name)details+=`<p><strong>Hourly source:</strong> ${esc(source.name)}</p>`;
    const issued=source.issued_at||p?.issued_at;if(issued)details+=`<p><strong>Issued:</strong> ${esc(dateTime(issued))}</p>`;
    const urls=[];if(ds?.url)urls.push([ds.name||'Daily model weather source',ds.url]);if(source.url)urls.push([source.name||'Hourly source',source.url]);if(Array.isArray(p?.sources))p.sources.forEach(x=>{if(x?.url)urls.push([x.name||x.label||'Data source',x.url]);});
    if(mammoth){
      const radar=p?.radar_observation;
      const sesame=p?.sesame_observation;
      if(sesame?.url)urls.push([sesame.name||'Sesame snow study site',sesame.url]);
      if(radar?.url)urls.push(['NOAA MRMS precipitation rate grid',radar.url]);
      if(radar?.quality_url)urls.push(['NOAA MRMS radar quality grid',radar.quality_url]);
      urls.push(['Main Lodge snowfall history','https://www.mammothmountain.com/on-the-mountain/snowfall-history']);
      urls.push(['Jeff Dozier Snow Study Site · historical ground observations','https://doi.org/10.5281/zenodo.17179963']);
    }
    urls.forEach(([name,url])=>{try{const u=new URL(url);if(['http:','https:'].includes(u.protocol))details+=`<p><a href="${esc(u.href)}" target="_blank" rel="noopener noreferrer">${esc(name)} ↗</a></p>`;}catch{}});
    if(!mammoth)details+=`<p>Refresh all updates the forecast inputs and observations for each mountain and runs its deployed models. Failed sources keep their original timestamps or use the indicated fallback.</p>`;
    if(warnings.length)details+=`<ul>${warnings.map(s=>`<li>${esc(s)}</li>`).join('')}</ul>`;
    if(!mammoth)details+='<p><a href="/reports/weather/forecast-portfolio/report.html" target="_blank" rel="noopener">Model selection, errors and forecast curves ↗</a> · <a href="/reports/weather/neural48-trained/report.html" target="_blank" rel="noopener">48-hour neural evaluation ↗</a></p>';
    $('source-details').innerHTML=details;
  }
  async function refresh(){
    if(state.loading)return;
    state.loading=true;state.request++;
    const request=async(url,options={})=>{const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);try{return await api(url,{...options,signal:controller.signal});}finally{clearTimeout(timer);}};
    $('refresh-button').disabled=true;$('refresh-button').classList.add('loading');$('refresh-button').querySelector('span').textContent='Refreshing all…';
    notice('error-banner','');notice('job-status','Updating weather and observations for all mountains…');renderMountains();render();
    let job=null,terminal=null;
    try{
      const response=await request('/api/refresh-all',{method:'POST',body:JSON.stringify({})});
      job=response.job_id||response.id;if(!job)throw new Error('The service did not return a refresh job.');
      const deadline=Date.now()+25*60*1000;let pollingErrors=0;
      while(Date.now()<deadline){
        await new Promise(resolve=>setTimeout(resolve,1200));
        let result;
        try{result=await request(`/api/jobs/${encodeURIComponent(job)}`);pollingErrors=0;}
        catch(error){pollingErrors++;if(pollingErrors>=3)throw new Error('Refresh status could not be checked. The server may still be working; check source timestamps before using this update.');notice('job-status','Reconnecting to the shared mountain update…');continue;}
        notice('job-status',result.message||'Updating weather and observations for all mountains…');
        if(['succeeded','partial','failed','error','completed','complete','success'].includes(result.status)){terminal=result;break;}
      }
      if(!terminal)throw new Error('The update is taking longer than expected and may still be running. Its original source timestamps remain visible.');
      const loaded=await Promise.allSettled(mountainsFallback.map(async mountain=>({id:mountain.id,payload:normalize(await request(`/api/forecast?site=${encodeURIComponent(mountain.id)}`))})));
      const loadErrors=[];
      loaded.forEach((result,i)=>{if(result.status==='fulfilled')state.forecasts[result.value.id]=result.value.payload;else loadErrors.push(`${mountainsFallback[i].name}: the updated forecast could not be loaded.`);});
      if(state.forecasts[state.site]){state.imported=false;state.forecast=state.forecasts[state.site];}
      await Promise.allSettled(mountainsFallback.map(mountain=>loadSeason(mountain.id)));
      const errors=[...warningsText(terminal.errors||terminal.result?.errors),...loadErrors];
      const sourceRows=terminal.sources||terminal.result?.sources||{};
      for(const [name,source] of Object.entries(sourceRows))if(['failed','error','unavailable'].includes(source?.status)&&source.message)errors.push(`${name}: ${source.message}`);
      const problems=[...new Set(errors)];
      if(['failed','error'].includes(terminal.status)&&!problems.length)problems.push(terminal.message||'Current sources could not refresh.');
      notice('error-banner',problems.length?problems.join(' '):'');
      const partial=terminal.status==='partial'||problems.length>0;
      notice('job-status',['failed','error'].includes(terminal.status)?'The update could not finish. Saved forecasts retain their original timestamps.':partial?'Update finished with some unavailable sources. Available forecasts keep their original source times; fallback choices are shown in the comparison.':'All mountains updated.');
    }catch(error){notice('error-banner',error.name==='AbortError'?'The service did not respond in time. The shared update may still be running.':error.message);notice('job-status',job?'Refresh status is uncertain. Check source timestamps before using the results.':'The all-mountain update could not be started.');}
    finally{state.loading=false;$('refresh-button').disabled=false;$('refresh-button').classList.remove('loading');$('refresh-button').querySelector('span').textContent='Refresh all';renderMountains();render();}
  }
  function exportForecast(){if(!state.forecast)return;const obj={schema:'snowline.forecast.v1',exported_at:new Date().toISOString(),forecast:state.forecast};const blob=new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`snowline-${state.site}-${new Date().toISOString().slice(0,10)}.json`;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
  function validateImport(object){
    if(!object||object.schema!=='snowline.forecast.v1'||!object.forecast||typeof object.forecast!=='object')throw new Error('Choose a Snowline forecast JSON export.');
    const p=object.forecast,id=typeof p.site==='string'?p.site:p.site?.id;
    if(!mountainsFallback.some(m=>m.id===id))throw new Error('This export does not identify a supported mountain.');
    if(p.site?.timezone){try{new Intl.DateTimeFormat('en-US',{timeZone:p.site.timezone}).format();}catch{throw new Error('This export contains an invalid time zone.');}}
    if(!Array.isArray(p.hourly)||p.hourly.length>168||!Array.isArray(p.daily)||p.daily.length>14)throw new Error('This export has invalid forecast arrays.');
    for(const field of ['source','daily_source'])if(p[field]&&(typeof p[field]!=='object'||Array.isArray(p[field])))throw new Error('This export has invalid source metadata.');
    for(const field of ['warnings','errors','sources'])if(p[field]&&!Array.isArray(p[field]))throw new Error('This export has invalid source notes.');
    for(const r of p.hourly){
      if(!r||!Number.isFinite(Date.parse(r.time)))throw new Error('This export contains an invalid forecast timestamp.');
      for(const key of ['snowfall_inches','precipitation_mm','temperature_c','wind_kmh'])if(r[key]!==undefined&&r[key]!==null&&(!Number.isFinite(Number(r[key]))||Math.abs(Number(r[key]))>10000))throw new Error('This export contains an invalid forecast value.');
    }
    for(const r of p.daily){
      if(!r||!Number.isFinite(Date.parse(r.start))||!Number.isFinite(Date.parse(r.end))||Date.parse(r.start)>=Date.parse(r.end))throw new Error('This export contains an invalid daily window.');
      for(const key of ['p_powder','historical_p_powder'])if(r[key]!==null&&r[key]!==undefined&&(!Number.isFinite(Number(r[key]))||Number(r[key])<0||Number(r[key])>1))throw new Error('This export contains invalid legacy probability metadata.');
      for(const key of ['mean_inches','q10','q90','historical_mean_inches'])if(r[key]!==null&&r[key]!==undefined&&(!Number.isFinite(Number(r[key]))||Number(r[key])<0||Number(r[key])>1000))throw new Error('This export contains an invalid snowfall value.');
    }
    if(p.neural48){
      const n=p.neural48;
      if(typeof n!=='object'||Array.isArray(n)||n.site_id!==id)throw new Error('This export contains an invalid neural forecast identity.');
      if(n.status!=='unavailable'){
        if(!Array.isArray(n.totals)||n.totals.length!==2||n.totals[0].hours!==24||n.totals[1].hours!==48)throw new Error('This export requires exact 24/48-hour neural totals.');
        const a=Date.parse(n.start),b=Date.parse(n.midpoint),c=Date.parse(n.end);
        if(!Number.isFinite(a)||b-a!==86400000||c-b!==86400000)throw new Error('This export contains invalid neural forecast windows.');
        n.totals.forEach((r,i)=>{if(Date.parse(r.end)!==[b,c][i]||number(r.mean_inches)===null)throw new Error('This export contains invalid neural totals.');for(const key of ['mean_inches','q10_inches','q90_inches','provider_inches'])if(r[key]!==null&&r[key]!==undefined&&(number(r[key])===null||Number(r[key])<0||Number(r[key])>1000))throw new Error('This export contains an invalid neural snowfall amount.');});
      }
    }
    if(p.deployed){
      const deployed=p.deployed;
      if(typeof deployed!=='object'||Array.isArray(deployed)||deployed.site_id!==id||!Array.isArray(deployed.horizons)||(deployed.horizons.length!==6&&!(id==='mammoth'&&deployed.horizons.length===0)))throw new Error('This export contains invalid selected-model forecasts.');
      if(id==='mammoth'&&deployed.hourly!=null){
        if(!Array.isArray(deployed.hourly)||deployed.hourly.length>48||deployed.hourly.some((r,i)=>
          !r||r.horizon_hours!==i+1||!Number.isFinite(Date.parse(r.start))||Date.parse(r.end)-Date.parse(r.start)!==3600000||
          (i>0&&Date.parse(r.start)!==Date.parse(deployed.hourly[i-1].end))||
          [r.mean_mm,r.provider_mm,r.ecmwf_mm].some(v=>v!=null&&(!Number.isFinite(v)||v<0))))throw new Error('Invalid hourly snow-water forecast.');
      }
      const seen=new Set();
      for(const row of deployed.horizons){
        if(!row||!forecastHorizons.includes(row.hours)||seen.has(row.hours))throw new Error('This export contains invalid model horizons.');seen.add(row.hours);
        if(row.status==='unavailable')continue;
        if(!deploymentAvailable(row)||!Number.isFinite(Date.parse(row.issued_at))||Date.parse(row.issued_at)>Date.parse(row.start))throw new Error('This export contains an invalid selected-model window.');
        for(const key of ['mean','lower','upper','provider_mean'])if(row[key]!==null&&row[key]!==undefined&&(number(row[key])===null||Number(row[key])<0||Number(row[key])>10000))throw new Error('This export contains an invalid selected-model amount.');
        if(number(row.lower)!==null&&number(row.upper)!==null&&Number(row.lower)>Number(row.upper))throw new Error('This export contains a reversed model range.');
      }
    }
    if(p.radar_observation!=null){
      const observation=p.radar_observation;
      if(typeof observation!=='object'||Array.isArray(observation))throw new Error('This export has invalid radar observation data.');
      for(const key of ['timestamp','quality_timestamp'])if(observation[key]!=null&&!Number.isFinite(Date.parse(observation[key])))throw new Error('This export contains an invalid radar scan time.');
      for(const key of ['precip_rate_mm_h','quality_index'])if(observation[key]!=null&&(!Number.isFinite(observation[key])||observation[key]<0||(key==='quality_index'&&observation[key]>1)))throw new Error('This export contains an invalid radar observation.');
    }
    if(p.radar){
      if(typeof p.radar!=='object'||(p.radar.models&&!Array.isArray(p.radar.models))||(p.radar.models?.length||0)>8)throw new Error('This export contains invalid radar data.');
      for(const m of p.radar.models||[]){const points=m.points||m.hours;if(!Array.isArray(points)||points.length>128)throw new Error('This export contains invalid radar curves.');for(const q of points)if(number(q.hours)===null||number(val(q,'mean','mean_liquid_mm'))===null)throw new Error('This export contains invalid radar values.');}
    }
    return {p,id};
  }
  async function importForecast(file){if(!file)return;if(state.loading){notice('error-banner','Wait for the current refresh to finish before importing a snapshot.');return;}try{if(file.size>2*1024*1024)throw new Error('Forecast files must be smaller than 2 MB.');const {p,id}=validateImport(JSON.parse(await file.text()));state.request++;state.site=id;state.forecast=normalize(p);state.imported=true;syncForecastURL();renderMountains();render();notice('error-banner','');notice('job-status','Imported a saved forecast. No data were uploaded or used to retrain a model.');}catch(error){notice('error-banner',`Could not import this file. ${error.message}`);}finally{$('import-file').value='';}}
  $('refresh-button').addEventListener('click',refresh);$('export-button').addEventListener('click',exportForecast);$('import-button').addEventListener('click',()=>$('import-file').click());$('import-file').addEventListener('change',e=>importForecast(e.target.files[0]));
  document.querySelectorAll('[data-units]').forEach(b=>b.addEventListener('click',()=>{if(!['us','metric'].includes(b.dataset.units))return;state.units=b.dataset.units;try{localStorage.setItem('snowline.units.v1',state.units);}catch{}syncForecastURL();render();}));
  document.querySelectorAll('[data-hours]').forEach(b=>b.addEventListener('click',()=>{const hours=Number(b.dataset.hours);if(!forecastHorizons.includes(hours))return;state.hours=hours;syncForecastURL();render();}));
  document.querySelectorAll('[data-mammoth-quantity]').forEach(b=>b.addEventListener('click',()=>{
    state.mammothQuantity=b.dataset.mammothQuantity;syncForecastURL();render();
  }));
  document.querySelectorAll('[data-swe-view]').forEach(b=>b.addEventListener('click',()=>{
    state.sweView=b.dataset.sweView;renderMammothCurve(mammothDisplayForecast(state.forecast));
  }));
  document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{state.view=b.dataset.view;document.querySelectorAll('[data-view]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});render();}));
  document.querySelectorAll('[data-season-view]').forEach(b=>b.addEventListener('click',()=>{if(!['full','recent'].includes(b.dataset.seasonView))return;state.seasonView=b.dataset.seasonView;renderSeason();}));
  ['dragenter','dragover'].forEach(event=>$('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.add('drag-over');}));['dragleave','drop'].forEach(event=>$('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.remove('drag-over');}));$('drop-zone').addEventListener('drop',e=>importForecast(e.dataTransfer.files[0]));
  async function init(){const query=new URLSearchParams(location.search),mountain=query.get('mountain'),hours=Number(query.get('hours')),queryUnits=query.get('units');if(mountain==='mammoth'&&query.get('quantity')==='depth')state.mammothQuantity='depth';if(mountainsFallback.some(m=>m.id===mountain))state.site=mountain;if(forecastHorizons.includes(hours))state.hours=hours;if(['us','metric'].includes(queryUnits))state.units=queryUnits;renderMountains();render();try{const data=await api('/api/mountains');const list=Array.isArray(data)?data:data.mountains;if(list?.length)state.mountains=list.map(m=>({...mountainsFallback.find(f=>f.id===m.id),...m,...(m.id==='mansfield'?{name:'Stowe'}:{})}));}catch{}renderMountains();await loadForecast(state.site);}
  init();
})();
