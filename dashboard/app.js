'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const mountainsFallback = [
    {id:'alta',name:'Alta',location:'Utah · Wasatch Range',timezone:'America/Denver',target_label:'Alta–Collins manual snowfall target'},
    {id:'vail',name:'Vail',location:'Colorado · Gore Range',timezone:'America/Denver',target_label:'Vail station snow-sensor proxy'},
    {id:'mansfield',name:'Stowe',location:'Vermont · Green Mountains',timezone:'America/New_York',target_label:'NOAA Mount Mansfield station target'},
    {id:'jay',name:'Jay Peak',location:'Vermont · Green Mountains',timezone:'America/New_York',target_label:'NOAA station target near Jay Peak'}
  ];
  let savedUnits='us';try{if(localStorage.getItem('snowline.units.v1')==='metric')savedUnits='metric';}catch{}
  const state = {mountains:mountainsFallback,site:'alta',hours:48,view:'hourly',units:savedUnits,forecast:null,forecasts:{},imported:false,loading:false,request:0};
  const forecastHorizons = [1,3,6,12,24,48];
  const modelDisplay = {label:'Our model',color:'#be7c3a'};
  function syncForecastURL(){const url=new URL(location.href);url.searchParams.set('mountain',state.site);url.searchParams.set('hours',String(state.hours));url.searchParams.set('units',state.units);history.replaceState(null,'',`${url.pathname}${url.search}${url.hash}`);}
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
  async function loadForecast(id){const request=++state.request;try{const data=await api(`/api/forecast?site=${encodeURIComponent(id)}`);state.forecasts[id]=normalize(data);if(request!==state.request)return;state.forecast=state.forecasts[id];render();}catch(error){if(request!==state.request)return;notice('error-banner',`Could not load the forecast. ${error.message}`);$('data-status').textContent='Service unavailable';$('updated-at').textContent='Start the local forecast service and try Refresh all.';}}
  function getHours(){return (state.forecast?.hourly||[]).slice(0,state.hours);}
  function render(){
    const p=state.forecast,m={...siteInfo(),...(p?.site||{})},u=units();
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
    $('mountain-title').textContent=m.name||siteInfo().name;$('mountain-location').textContent=m.location||m.region||siteInfo().location||'';$('target-label').textContent=m.target_label||m.target||'Station-based research forecast';
    $('export-button').disabled=!p;
    const source=p?.source||{},generated=source.retrieved_at||p?.generated_at||p?.generated_utc;
    const end=p?.hourly?.at(-1)?.time;
    const expired=!!(end&&Date.parse(end)<=Date.now());
    const stale=!!p?.stale||!!source.stale||expired||(generated&&Date.now()-Date.parse(generated)>6*3600000);
    const unavailable=!p||!(p.hourly?.length||p.daily?.length||p.deployed?.horizons?.some(r=>number(r.mean)!==null));
    const partial=p?.status==='partial'||(Array.isArray(p?.errors)&&p.errors.length>0);
    const sourceCached=source.status==='cached';
    const statusText=state.imported?'Imported snapshot':unavailable?'Awaiting data':partial?'Partial update':stale?'Saved · refresh recommended':sourceCached?'Saved guidance':'Data ready';
    $('data-status').textContent=statusText;$('data-status').className=`badge ${state.imported||stale||partial?'warn':unavailable||sourceCached?'neutral':'live'}`;
    $('updated-at').textContent=generated?`Retrieved ${dateTime(generated)}`:p?'No current forecast available':'Checking saved forecasts';
    $('source-label').textContent=source.name||p?.hourly_source?.name||'';
    $('hourly-model-label').textContent=source.name||p?.hourly_source?.name||'Hourly weather guidance';
    $('chart-zone').textContent=date(new Date().toISOString(),{timeZoneName:'short'}).split(' ').at(-1)+' · mountain local time';
    let snapshotMessage=state.imported?'Viewing an imported snapshot. Refresh replaces it with newly fetched online data.':expired?'This forecast window has passed. The curves below are a saved snapshot; refresh for current guidance.':stale?'Showing a saved forecast. Refresh latest data to check for a newer weather run.':null;
    if(sourceCached&&!snapshotMessage)snapshotMessage='Showing previously retrieved hourly guidance. Its original retrieval time is shown above.';
    if(partial){const errors=warningsText(p?.errors).join(' ');snapshotMessage=[snapshotMessage,'Some sources could not refresh. Available forecasts below retain their original source timestamps.',errors].filter(Boolean).join(' ');}
    if(p?.hourly?.length && p.hourly.length<state.hours)snapshotMessage=[snapshotMessage,`Only ${p.hourly.length} hourly intervals are available for the selected ${state.hours}-hour horizon.`].filter(Boolean).join(' ');
    notice('snapshot-banner',snapshotMessage);
    const hours=getHours(),snow=hours.map(x=>x.snowfall_inches).filter(x=>x!==null),temps=hours.map(x=>x.temperature_c).filter(x=>x!==null),winds=hours.map(x=>x.wind_kmh).filter(x=>x!==null);
    $('snow-total-label').textContent=hours.length&&hours.length<state.hours?`${hours.length} available hours`:`Next ${state.hours} full ${state.hours===1?'hour':'hours'}`;
    if(expired)$('snow-total-label').textContent=`Saved ${state.hours}-hour window`;
    $('snow-total').textContent=snow.length&&snow.length===hours.length?snowText(snow.reduce((a,b)=>a+b,0)):'—';
    $('snow-total').style.color=snow.length&&snow.length===hours.length?snowColor(snow.reduce((a,b)=>a+b,0),24):'';
    const snowDescription=$('snow-total').closest('.summary-card').querySelector('.metric-description');snowDescription.textContent=snow.length!==hours.length?'Snowfall has missing hourly values':hours.length&&hours.length<state.hours?'Total over available hours only':'Forecast fresh snowfall';
    $('temperature-range').textContent=temps.length?`${Math.round(displayTemp(Math.min(...temps)))}–${Math.round(displayTemp(Math.max(...temps)))}`:'—';
    $('peak-wind').textContent=winds.length?fmt(displayWind(Math.max(...winds)),0):'—';
    const daily=p?.daily||[],next=daily.find(d=>d.end&&Date.parse(d.end)>Date.now());
    setText('daily-snow-amount',next?snowText(next.mean_inches):'—');
    if($('daily-snow-amount'))$('daily-snow-amount').style.color='';
    setText('daily-snow-description',next?`Expected · ending ${date(next.end,{month:'short',day:'numeric',hour:'numeric'})}`:daily.length?'Saved model windows have passed':'Calibrated outlook unavailable');
    if(hours.length)$('chart-zone').textContent=`${dateTime(hours[0].start||hours[0].time)} → ${dateTime(hours.at(-1).time)}`;
    const chartHours=hours.length||state.hours;
    $('snow-chart-title').textContent=`${state.view==='cumulative'?'Accumulation':'Snowfall'} over ${chartHours} ${chartHours===1?'hour':'hours'}`;
    renderSnow(hours);renderDeployed(p?.deployed);renderWeather(hours);renderSources(p,m);
  }
  function niceTop(n){if(!n||n<=0)return .1;const power=10**Math.floor(Math.log10(n)),v=n/power;return (v<=1?1:v<=2?2:v<=4?4:v<=5?5:10)*power;}
  function svgGrid(w,h,pad,top,ylabel){const precision=Math.min(6,Math.max(0,Math.ceil(-Math.log10(top/4))+2));const tickLabel=v=>v!==0&&top<.00001?v.toExponential(1):v.toLocaleString('en-US',{maximumFractionDigits:precision});let out='';for(let i=0;i<=4;i++){const v=top*i/4,y=h-pad.b-(h-pad.t-pad.b)*i/4;out+=`<path d="M${pad.l} ${y}H${w-pad.r}" stroke="#e8ede7" stroke-dasharray="3 4"/><text x="${pad.l-10}" y="${y+3}" fill="#8a9996" font-size="9" text-anchor="end">${esc(tickLabel(v))}</text>`;}out+=`<text x="${pad.l}" y="12" fill="#8a9996" font-size="9">${esc(ylabel)}</text>`;return out;}
  function renderSnow(rows){
    $('snow-tooltip').hidden=true;
    if(!rows.length){$('snow-chart').innerHTML=empty('Your next weather window starts here','Choose a mountain and refresh to pull the latest hourly weather forecast.');return;}
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
  function renderDeployed(deployed){
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
    const series=[{label:'Open-Meteo',color:'#167d87',amount:number(row.provider_mean)},{label:'Our model',color:modelDisplay.color,amount:Number(row.mean)}];
    const lo=number(row.lower),hi=number(row.upper),hasInterval=lo!==null&&hi!==null&&lo>=0&&hi>=lo;
    const w=1120,h=260,pad={l:55,r:18,t:30,b:42},plotW=w-pad.l-pad.r,plotH=h-pad.t-pad.b;
    const values=[...series.map(entry=>q.convert(entry.amount)),hasInterval?q.convert(hi):null].filter(v=>v!==null);
    const top=niceTop(Math.max(q.convert(.1),...values)*1.2),yy=v=>h-pad.b-v/top*plotH;
    const barW=75,gap=barW*.18,clusterW=2*barW+gap,center=pad.l+plotW/2;
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
    $('neural-chart').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(siteInfo().name)} ${state.hours}-hour ${esc(q.noun)}: Open-Meteo and Our model">${svg}</svg>`;
    $('neural-legend').innerHTML=series.map(entry=>`<span><i class="legend-line" style="background:${entry.color}"></i>${esc(entry.label)}</span>`).join('');
    const range=hasInterval?`${row.interval_label||'Model range'}: ${amountText(q.convert(lo),2)}–${amountText(q.convert(hi),2)} ${q.unit}. `:'';
    setText('neural-note',`${range}${row.quantity==='liquid_precipitation'?'Liquid-equivalent precipitation, not fresh-snow depth. ':'Snowfall accumulation. '}Both bars cover the same stated window and target. Missing provider guidance is shown as a dash.`);
    const notes=[saved?'This is a saved forecast window. Refresh all for current guidance.':null,...displayNotes(row.warnings),deployed?.target?.note].filter(Boolean);
    $('neural-details').innerHTML=[...new Set(notes)].map(text=>`<p>${esc(text)}</p>`).join('');
  }

  function renderWeather(rows){
    const u=units();
    if(!rows.length){$('weather-table-wrap').innerHTML=empty('Weather details will appear here','Hourly temperature, precipitation and wind arrive with the weather refresh.');$('weather-period').textContent='';return;}
    $('weather-period').textContent=`${date(rows[0].time,{month:'short',day:'numeric'})} – ${date(rows.at(-1).time,{month:'short',day:'numeric'})}`;
    const indices=rows.length>12?Array.from({length:12},(_,i)=>Math.round(i*(rows.length-1)/11)):rows.map((_,i)=>i),sample=indices.map(i=>rows[i]);
    const row=(label,unit,fn,strong=false)=>`<tr><td>${label}<small>${unit}</small></td>${sample.map(r=>`<td${strong?' class="table-cell-strong"':''}>${fn(r)}</td>`).join('')}</tr>`;
    $('weather-table-wrap').innerHTML=`<table class="weather-table"><caption class="skip-link">Selected hourly weather forecast, sampled across the selected horizon</caption><thead><tr><th scope="col">Local time</th>${sample.map(r=>`<th scope="col" title="${esc(dateTime(r.time))}">${esc(localHour(r.time))}<br><span style="font-size:8px">${esc(date(r.time,{weekday:'short'}))}</span></th>`).join('')}</tr></thead><tbody>${row('Fresh snowfall',`${u.snowLong} / hour`,r=>snowText(r.snowfall_inches,2),true)}${row('Temperature',u.tempLong,r=>r.temperature_c!==null?fmt(displayTemp(r.temperature_c),0)+'°':'—')}${row('Wind',u.windLong,r=>fmt(displayWind(r.wind_kmh),0))}${row('Precipitation',`liquid equivalent · ${u.liquid}`,r=>fmt(displayLiquid(r.precipitation_mm),metric()?1:3))}</tbody></table>`;
  }
  function renderSources(p,m){
    const deployed=p?.deployed,row=selectedDeployment(deployed);
    const capabilities=[['One selected model per horizon','The comparison uses the saved selection for this mountain and horizon. Refresh updates its inputs without retraining.'],['Radar and HRRR','Short-range radar forecasts measure liquid precipitation. HRRR provides the fallback where radar coverage or verification is insufficient.'],['Snowfall accumulation','The 12/24/48-hour models are evaluated at their stated accumulation windows; those scores do not establish hourly snowfall timing.'],['Observation target',deployed?.target?.note||m.target_note||m.target_label||'The available mountain-area station observations.']];
    $('capability-list').innerHTML=capabilities.map(([title,body])=>`<div class="capability"><span class="capability-mark">✓</span><span><strong>${esc(title)}</strong><br>${esc(body)}</span></div>`).join('');
    const source=p?.source||{},warnings=displayNotes(p?.warnings);let details='';
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
    if(m.target_note)details+=`<p><strong>External hourly forecast target:</strong> ${esc(m.target_note)}</p>`;
    const ds=p?.daily_source;if(ds?.name)details+=`<p><strong>Daily model weather source:</strong> ${esc(ds.name)}${ds.issued_at?'<br>Issued '+esc(dateTime(ds.issued_at)):''}${ds.retrieved_at?'<br>Retrieved '+esc(dateTime(ds.retrieved_at)):''}</p>`;
    if(source.name)details+=`<p><strong>Hourly source:</strong> ${esc(source.name)}</p>`;
    const issued=source.issued_at||p?.issued_at;if(issued)details+=`<p><strong>Issued:</strong> ${esc(dateTime(issued))}</p>`;
    const urls=[];if(ds?.url)urls.push([ds.name||'Daily model weather source',ds.url]);if(source.url)urls.push([source.name||'Hourly source',source.url]);if(Array.isArray(p?.sources))p.sources.forEach(x=>{if(x?.url)urls.push([x.name||x.label||'Data source',x.url]);});
    urls.forEach(([name,url])=>{try{const u=new URL(url);if(['http:','https:'].includes(u.protocol))details+=`<p><a href="${esc(u.href)}" target="_blank" rel="noopener noreferrer">${esc(name)} ↗</a></p>`;}catch{}});
    details+=`<p>Refresh all updates weather-provider, radar and HRRR inputs for Alta, Vail, Stowe and Jay Peak in one shared job. It runs saved model weights and refreshes the available mountain observations. It does not retrain the models. Failed sources retain their original timestamps or trigger an identified fallback.</p><p>Hourly weather guidance, radar liquid precipitation and learned snowfall accumulations have different targets. Their quantities and forecast windows are shown separately.</p>`;
    if(warnings.length)details+=`<ul>${warnings.map(s=>`<li>${esc(s)}</li>`).join('')}</ul>`;
    details+='<p><a href="/reports/weather/forecast-portfolio/report.html" target="_blank" rel="noopener">Model selection, errors and forecast curves ↗</a> · <a href="/reports/weather/neural48-trained/report.html" target="_blank" rel="noopener">48-hour neural evaluation ↗</a></p>';
    $('source-details').innerHTML=details;
  }
  async function refresh(){
    if(state.loading)return;
    state.loading=true;state.request++;
    const request=async(url,options={})=>{const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);try{return await api(url,{...options,signal:controller.signal});}finally{clearTimeout(timer);}};
    $('refresh-button').disabled=true;$('refresh-button').classList.add('loading');$('refresh-button').querySelector('span').textContent='Refreshing all…';
    notice('error-banner','');notice('job-status','Updating weather, radar and HRRR for all four mountains…');renderMountains();render();
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
        notice('job-status',result.message||'Updating weather, radar and HRRR for all four mountains…');
        if(['succeeded','partial','failed','error','completed','complete','success'].includes(result.status)){terminal=result;break;}
      }
      if(!terminal)throw new Error('The update is taking longer than expected and may still be running. Its original source timestamps remain visible.');
      const loaded=await Promise.allSettled(mountainsFallback.map(async mountain=>({id:mountain.id,payload:normalize(await request(`/api/forecast?site=${encodeURIComponent(mountain.id)}`))})));
      const loadErrors=[];
      loaded.forEach((result,i)=>{if(result.status==='fulfilled')state.forecasts[result.value.id]=result.value.payload;else loadErrors.push(`${mountainsFallback[i].name}: the updated forecast could not be loaded.`);});
      if(state.forecasts[state.site]){state.imported=false;state.forecast=state.forecasts[state.site];}
      const errors=[...warningsText(terminal.errors||terminal.result?.errors),...loadErrors];
      const sourceRows=terminal.sources||terminal.result?.sources||{};
      for(const [name,source] of Object.entries(sourceRows))if(['failed','error','unavailable'].includes(source?.status)&&source.message)errors.push(`${name}: ${source.message}`);
      const problems=[...new Set(errors)];
      if(['failed','error'].includes(terminal.status)&&!problems.length)problems.push(terminal.message||'Current sources could not refresh.');
      notice('error-banner',problems.length?problems.join(' '):'');
      const partial=terminal.status==='partial'||problems.length>0;
      notice('job-status',['failed','error'].includes(terminal.status)?'The update could not finish. Saved forecasts retain their original timestamps.':partial?'Update finished with some unavailable sources. Available forecasts keep their original source times; fallback choices are shown in the comparison.':'All four mountains updated, including radar and HRRR.');
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
      if(typeof deployed!=='object'||Array.isArray(deployed)||deployed.site_id!==id||!Array.isArray(deployed.horizons)||deployed.horizons.length!==6)throw new Error('This export contains invalid selected-model forecasts.');
      const seen=new Set();
      for(const row of deployed.horizons){
        if(!row||!forecastHorizons.includes(row.hours)||seen.has(row.hours))throw new Error('This export contains invalid model horizons.');seen.add(row.hours);
        if(row.status==='unavailable')continue;
        if(!deploymentAvailable(row)||!Number.isFinite(Date.parse(row.issued_at))||Date.parse(row.issued_at)>Date.parse(row.start))throw new Error('This export contains an invalid selected-model window.');
        for(const key of ['mean','lower','upper','provider_mean'])if(row[key]!==null&&row[key]!==undefined&&(number(row[key])===null||Number(row[key])<0||Number(row[key])>10000))throw new Error('This export contains an invalid selected-model amount.');
        if(number(row.lower)!==null&&number(row.upper)!==null&&Number(row.lower)>Number(row.upper))throw new Error('This export contains a reversed model range.');
      }
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
  document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{state.view=b.dataset.view;document.querySelectorAll('[data-view]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});render();}));
  ['dragenter','dragover'].forEach(event=>$('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.add('drag-over');}));['dragleave','drop'].forEach(event=>$('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.remove('drag-over');}));$('drop-zone').addEventListener('drop',e=>importForecast(e.dataTransfer.files[0]));
  async function init(){const query=new URLSearchParams(location.search),mountain=query.get('mountain'),hours=Number(query.get('hours')),queryUnits=query.get('units');if(mountainsFallback.some(m=>m.id===mountain))state.site=mountain;if(forecastHorizons.includes(hours))state.hours=hours;if(['us','metric'].includes(queryUnits))state.units=queryUnits;renderMountains();render();try{const data=await api('/api/mountains');const list=Array.isArray(data)?data:data.mountains;if(list?.length)state.mountains=list.map(m=>({...mountainsFallback.find(f=>f.id===m.id),...m,...(m.id==='mansfield'?{name:'Stowe'}:{})}));}catch{}renderMountains();await loadForecast(state.site);}
  init();
})();
