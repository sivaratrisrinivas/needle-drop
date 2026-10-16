// Needle Drop front end. Two modes:
//  live   - talks to the FastAPI backend (LLM brief, clip analysis, agent, license RAG, traces)
//  static - no backend (e.g. GitHub Pages): precomputed showcase results, plus an in-browser
//           tag search and in-browser clip analysis over the same catalog and license rules.
const $ = s => document.querySelector(s);
const state = {mode: 'static', tab: 'text', clipFile: null, srtFile: null, sampleClip: null, lite: null,
               showcase: null, playing: null, result: null, clipURL: null};
const EXAMPLES = [
  'A hitman in a black suit casually dances around a bound cop in an empty warehouse before cutting off his ear.',
  'Two old friends say goodbye at a rainy train station; one boards and the train pulls away.',
  'A getaway driver launches into a screeching car chase through city traffic at night.',
  'Dawn over a quiet beach; an old man walks alone along the water remembering his father.',
];
const STATUS_LABEL = {cleared: 'Cleared', not_cleared: 'Not cleared', needs_review: 'Needs review', unverified: 'Unverified'};

async function boot() {
  try {
    const r = await fetch('api/health', {cache: 'no-store'});
    if (r.ok && (r.headers.get('content-type') || '').includes('json')) {
      const h = await r.json();
      state.mode = 'live';
      $('#mode').textContent = `live · ${h.catalog_tracks.toLocaleString()} tracks · ${h.llm ? h.llm_model : 'no LLM (heuristic)'}`;
      refreshSpend();
    }
  } catch (e) { /* static */ }
  if (state.mode === 'static') $('#mode').textContent = 'static demo · runs in your browser';
  for (const ex of EXAMPLES) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = ex.slice(0, 48) + '…';
    b.title = ex; b.onclick = () => { $('#scene').value = ex; }; $('#examples').append(b);
  }
  document.querySelectorAll('.tab').forEach(t => t.onclick = () => setTab(t.dataset.tab));
  wireDrop();
  $('#go').onclick = go;
  $('#energy').oninput = e => $('#energy-out').textContent = (+e.target.value).toFixed(2);
  $('#sample-chase').onclick = () => useSample('chase');
  $('#sample-dawn').onclick = () => useSample('dawn');
  loadLite().then(buildChips);
}

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll('.tab').forEach(t => { t.classList.toggle('active', t.dataset.tab === tab); t.setAttribute('aria-selected', t.dataset.tab === tab); });
  for (const p of ['text', 'clip', 'manual']) $('#pane-' + p).classList.toggle('hidden', p !== tab);
}

function wireDrop() {
  const d = $('#drop');
  ['dragenter', 'dragover'].forEach(ev => d.addEventListener(ev, e => { e.preventDefault(); d.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => d.addEventListener(ev, e => { e.preventDefault(); d.classList.remove('over'); }));
  d.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) setClip(f); });
  $('#clipfile').onchange = e => e.target.files[0] && setClip(e.target.files[0]);
  $('#srtfile').onchange = e => { state.srtFile = e.target.files[0] || null; };
}

function setClip(file, sampleName = null) {
  state.clipFile = file; state.sampleClip = sampleName;
  if (state.clipURL) URL.revokeObjectURL(state.clipURL);
  state.clipURL = URL.createObjectURL(file);
  const v = $('#preview'); v.src = state.clipURL; v.classList.remove('hidden');
  $('#drop').querySelector('span:last-child').innerHTML = `<b>${esc(file.name)}</b> · ${(file.size / 1e6).toFixed(1)} MB`;
}

async function useSample(name) {
  setStatus(`Loading sample clip…`, true);
  const r = await fetch(`samples/${name}.webm`);
  const b = await r.blob();
  setClip(new File([b], `${name}.webm`, {type: 'video/webm'}), name);
  try { const s = await fetch(`samples/${name}.srt`); if (s.ok) state.srtFile = new File([await s.blob()], `${name}.srt`); } catch (e) {}
  setStatus('');
}

function setStatus(msg, busy = false) {
  $('#status').innerHTML = esc(msg) + (busy ? '<div class="bar"><i></i></div>' : '');
}

async function go() {
  const btn = $('#go'); btn.disabled = true;
  try {
    let result;
    if (state.tab === 'manual') result = await liteSearch(manualBrief());
    else if (state.mode === 'live') result = await liveRun();
    else result = await staticRun();
    if (result) render(result);
  } catch (e) {
    setStatus('Something went wrong: ' + e.message);
  } finally { btn.disabled = false; }
}

// ------------------------------------------------------------------ live mode
async function liveRun() {
  const use = $('#use').value;
  let job;
  if (state.tab === 'clip') {
    if (!state.clipFile) { setStatus('Choose a clip first.'); return; }
    const fd = new FormData(); fd.append('file', state.clipFile); fd.append('use', use);
    fd.append('text', $('#clipnote').value);
    if (state.srtFile) fd.append('srt', state.srtFile);
    job = await (await fetch('api/clips', {method: 'POST', body: fd})).json();
  } else {
    const text = $('#scene').value.trim();
    if (!text) { setStatus('Describe the scene first.'); return; }
    job = await (await fetch('api/search', {method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({text, use})})).json();
  }
  if (job.detail) throw new Error(job.detail);
  const t0 = Date.now();
  while (true) {
    const j = await (await fetch(`api/jobs/${job.id}`)).json();
    const secs = ((Date.now() - t0) / 1000).toFixed(0);
    if (j.status === 'done') { setStatus(`Done in ${secs}s${j.result.cache_hit ? ' (from cache)' : ''}.`); refreshSpend(); return j.result; }
    if (j.status === 'error') throw new Error(j.error.split('\n')[0]);
    const step = await currentStep(j.trace_id);
    setStatus(j.status === 'queued' ? `Queued (position ${j.queue_position + 1})…` : `${step} · ${secs}s`, true);
    await new Promise(r => setTimeout(r, 1200));
  }
}

async function currentStep(tid) {
  try {
    const t = await (await fetch(`api/traces/${tid}`)).json();
    const last = (t.spans || []).filter(s => s.name.startsWith('tool.') || s.name.startsWith('clip.') || s.name.startsWith('llm.')).pop();
    const names = {'clip.analyze': 'Analyzing the clip', 'clip.cuts': 'Detecting cuts', 'clip.frames': 'Reading frames',
      'clip.audio': 'Listening to the audio', 'tool.analyze_clip': 'Analyzing the clip', 'llm.scene_profile': 'Writing the music brief',
      'tool.parse_scene': 'Writing the music brief', 'tool.search_catalog': 'Searching the catalog', 'tool.check_licenses': 'Checking licenses',
      'llm.agent_plan': 'Planning the next step', 'llm.explain': 'Explaining the picks', 'vlm.caption': 'Looking at keyframes'};
    return last ? (names[last.name] || last.name) : 'Working';
  } catch (e) { return 'Working'; }
}

async function refreshSpend() {
  if (state.mode !== 'live') return;
  const s = await (await fetch('api/spend')).json();
  $('#spend').textContent = `${s.requests} runs · ${(s.prompt_tokens + s.completion_tokens).toLocaleString()} tokens · API spend $${s.cost_usd.toFixed(4)} (local model) · cache hits ${s.cache_hits}`;
}

// ------------------------------------------------------------------ static mode
async function loadLite() {
  if (state.lite) return state.lite;
  try {
    const [lite, show] = await Promise.all([fetch('static/catalog-lite.json').then(r => r.json()),
                                            fetch('static/showcase.json').then(r => r.ok ? r.json() : {}).catch(() => ({}))]);
    state.lite = lite; state.showcase = show;
  } catch (e) { state.lite = null; }
  return state.lite;
}

function buildChips() {
  const L = state.lite; if (!L) return;
  const mk = (box, list, cls) => list.forEach(t => {
    const c = document.createElement('button'); c.type = 'button'; c.className = 'chip ' + cls; c.textContent = t;
    c.onclick = () => c.classList.toggle('on'); $(box).append(c);
  });
  mk('#mood-chips', L.top_moods, 'mood'); mk('#genre-chips', L.top_genres, 'genre');
}

function manualBrief() {
  const on = cls => [...document.querySelectorAll(`.chip.${cls}.on`)].map(c => c.textContent);
  const moods = on('mood'), genres = on('genre');
  if (!moods.length && !genres.length) throw new Error('pick at least one mood or genre');
  return {summary: 'Manual brief', moods, genres, instruments: [], energy: +$('#energy').value,
          relationship: 'match', source: 'manual'};
}

async function staticRun() {
  const show = state.showcase || {};
  if (state.tab === 'clip') {
    if (!state.clipFile) { setStatus('Choose a clip first.'); return; }
    if (state.sampleClip && show.clips && show.clips[state.sampleClip]) {
      setStatus('Showing the precomputed backend result for this sample clip.');
      return show.clips[state.sampleClip];
    }
    setStatus('Analyzing your clip in the browser…', true);
    const a = await analyzeInBrowser($('#preview'));
    const brief = briefFromAnalysis(a);
    const r = await liteSearch(brief); r.analysis = a; return r;
  }
  const text = $('#scene').value.trim();
  if (!text) { setStatus('Describe the scene first.'); return; }
  const pre = (show.scenes || []).find(s => s.text.trim() === text);
  if (pre) { setStatus('Showing the precomputed backend result for this example.'); return pre.result; }
  setStatus('No backend here, so this uses a keyword brief and in-browser tag search.');
  return liteSearch(keywordBrief(text));
}

const SYN = {rain: ['rain', 'melancholic', 'sad'], rainy: ['rain', 'melancholic', 'sad'], goodbye: ['sad', 'emotional', 'melancholic'],
  farewell: ['sad', 'emotional'], chase: ['action', 'energetic', 'fast'], car: ['driving', 'action'], drive: ['driving'],
  fight: ['action', 'aggressive', 'battle'], kiss: ['romantic', 'love'], love: ['romantic', 'love'], dawn: ['peaceful', 'hopeful', 'calm'],
  sunrise: ['hopeful', 'peaceful'], night: ['night', 'dark'], scary: ['horror', 'scary', 'suspense'], dance: ['dancing', 'fun', 'party'],
  dances: ['dancing', 'fun'], party: ['party', 'fun'], funeral: ['sad', 'emotional'], alone: ['melancholic', 'reflective'],
  space: ['space', 'cosmic'], beach: ['summer', 'beach', 'relaxing'], war: ['war', 'dramatic', 'epic'], hero: ['heroic', 'epic'],
  memory: ['nostalgic', 'reflective'], remembering: ['nostalgic', 'reflective'], father: ['emotional', 'nostalgic'], tense: ['tension', 'suspense'],
  violence: ['dark', 'aggressive'], heist: ['action', 'cool', 'suspense'], city: ['urban'], quiet: ['calm', 'soft', 'quiet']};
const FAST = /\b(chase|fight|run|race|explo\w*|frantic|fast|action|battle|heist|dance|party|screech\w*)\b/gi;
const SLOW = /\b(quiet|slow|alone|grief|funeral|sleep|calm|rain\w*|farewell|goodbye|lonely|still|sunset|dawn|melanchol\w*)\b/gi;

function keywordBrief(text) {
  const L = state.lite, moods = new Set(), genres = new Set();
  for (const w of text.toLowerCase().match(/[a-z]+/g) || []) {
    (SYN[w] || []).forEach(m => L.moods.includes(m) && moods.add(m));
    if (L.moods.includes(w)) moods.add(w);
    if (L.genres.includes(w)) genres.add(w);
  }
  const f = (text.match(FAST) || []).length, s = (text.match(SLOW) || []).length;
  const energy = Math.max(.05, Math.min(.95, .5 + .15 * (f - s)));
  if (!moods.size) ['film', 'emotional'].forEach(m => moods.add(m));
  return {summary: text, moods: [...moods].slice(0, 6), genres: [...genres], instruments: [], energy,
          relationship: 'match', source: 'keyword (static mode)'};
}

function briefFromAnalysis(a) {
  const moods = [];
  if (a.brightness < .3) moods.push('dark', 'night'); else if (a.brightness > .6) moods.push('happy', 'bright');
  if (a.warmth > .05) moods.push('warm', 'peaceful'); else if (a.warmth < -.05) moods.push('cold', 'tension');
  const energy = Math.max(.1, Math.min(.95, .2 + a.cuts_per_min / 60 + a.motion * 3));
  if (energy > .7) moods.push('action', 'energetic'); else if (energy < .35) moods.push('calm', 'reflective');
  return {summary: a.description, moods: [...new Set(moods)].filter(m => state.lite.moods.includes(m)), genres: [],
          instruments: [], energy, relationship: 'match', source: 'in-browser clip analysis'};
}

async function liteSearch(brief) {
  const L = await loadLite(); if (!L) throw new Error('catalog not available');
  const use = $('#use').value, clearedOnly = $('#cleared-only').checked;
  const pm = new Set(brief.moods), pg = new Set(brief.genres);
  const scored = [];
  for (const t of L.tracks) {
    const [id, title, artist, genres, moods, energy, lic] = t;
    const v = verdict(lic, use);
    if (clearedOnly && v.status !== 'cleared') continue;
    const mm = moods.filter(m => pm.has(m)), gm = genres.filter(g => pg.has(g));
    if (!mm.length && !gm.length) continue;
    const score = .5 * Math.min(1, mm.length / Math.max(1, Math.min(pm.size, 3))) + .3 * Math.min(1, gm.length / Math.max(1, Math.min(pg.size, 2)))
                + .2 * (1 - Math.abs(energy - brief.energy));
    scored.push({t, score, mm, gm, v});
  }
  scored.sort((a, b) => b.score - a.score);
  const per = {}, items = [];
  for (const s of scored) {
    const artist = s.t[2]; if ((per[artist] || 0) >= 2) continue; per[artist] = (per[artist] || 0) + 1;
    const [id, title, a, genres, moods, energy, lic] = s.t, jid = id.split(':')[1];
    items.push({rank: items.length + 1, score: s.score, cleared: s.v.status === 'cleared',
      track: {id, title, artist: a, genres, moods, instruments: [], energy, page_url: `https://www.jamendo.com/track/${jid}`,
              audio_url: `https://prod-1.storage.jamendo.com/?trackid=${jid}&format=mp31`},
      matched: {moods: s.mm, genres: s.gm}, why: [`mood match: ${s.mm.join(', ') || 'none'}`, `genre match: ${s.gm.join(', ') || 'none'}`,
              `energy ${energy.toFixed(2)} vs scene ${brief.energy.toFixed(2)}`],
      license: {status: s.v.status, license_id: lic, reason: s.v.reason, cost: s.v.cost, verification: lic ? 'dataset_snapshot' : 'none',
                source: lic ? L.license_source : '', citations: [], conditions: s.v.conditions,
                license_url: ''},
      flags: lic ? ['License verified against the 2019 MTG-Jamendo snapshot; reconfirm on the Jamendo track page before publishing.'] : ['No verified license source.'],
      blurb: ''});
    if (items.length >= 10) break;
  }
  const warnings = items.some(i => i.cleared) ? [] : ['No track in this shortlist is cleared for the selected use.'];
  return {profile: brief, items, warnings, use, analysis: null, steps: [], lite: true};
}

function verdict(lic, use) {
  const R = state.lite.rules;
  if (!lic || !R[lic]) return {status: 'unverified', reason: 'No verified license record.', cost: {amount_usd: null, basis: 'Unknown: no verified license record.'}, conditions: []};
  const [status, reason, conditions] = R[lic][use];
  const basis = state.lite.costs[status === 'cleared' ? 'cleared' : (lic.includes('nd') ? 'nd' : status)];
  return {status, reason, conditions, cost: {amount_usd: status === 'cleared' || status === 'needs_review' ? 0 : null, basis}};
}

async function analyzeInBrowser(video) {
  await new Promise(r => video.readyState >= 1 ? r() : video.addEventListener('loadedmetadata', r, {once: true}));
  const dur = Math.min(30, video.duration || 30), W = 64, H = 36, fps = 4;
  const c = document.createElement('canvas'); c.width = W; c.height = H; const g = c.getContext('2d', {willReadFrequently: true});
  let prev = null, diffs = [], cuts = [], br = 0, sat = 0, warm = 0, n = 0;
  for (let t = 0; t < dur; t += 1 / fps) {
    video.currentTime = t; await new Promise(r => video.addEventListener('seeked', r, {once: true}));
    g.drawImage(video, 0, 0, W, H); const d = g.getImageData(0, 0, W, H).data;
    let b = 0, s = 0, w = 0, df = 0;
    for (let i = 0; i < d.length; i += 4) {
      const R = d[i] / 255, G = d[i + 1] / 255, B = d[i + 2] / 255, mx = Math.max(R, G, B), mn = Math.min(R, G, B);
      b += (R + G + B) / 3; s += (mx - mn) / (mx + 1e-6); w += R - B; if (prev) df += Math.abs(R - prev[i] / 255);
    }
    const px = d.length / 4; br += b / px; sat += s / px; warm += w / px; n++;
    if (prev) { const m = df / px; diffs.push(m); if (m > .18) cuts.push(+t.toFixed(2)); }
    prev = d.slice();
    setStatus(`Analyzing your clip in the browser… ${Math.round(t / dur * 100)}%`, true);
  }
  const motion = diffs.length ? diffs.reduce((a, b) => a + b) / diffs.length : 0;
  const a = {duration: dur, cuts, cuts_per_min: cuts.length / dur * 60, avg_shot_s: dur / (cuts.length + 1), motion,
    brightness: br / n, saturation: sat / n, warmth: warm / n, keyframes: [], captions: [], subtitles: [], tempo_bpm: null};
  a.pacing = a.cuts_per_min >= 30 ? 'frantic, rapid cutting' : a.cuts_per_min >= 15 ? 'brisk cutting' : a.cuts_per_min >= 6 ? 'measured pacing' : 'long, lingering takes';
  a.palette = `${a.brightness < .3 ? 'dark' : a.brightness > .6 ? 'bright' : 'naturally lit'}, ${a.warmth > .05 ? 'warm' : a.warmth < -.05 ? 'cold' : 'neutral'} palette`;
  a.description = `${dur.toFixed(0)}s clip, ${a.pacing} (${cuts.length} cuts), ${a.palette}.`;
  return a;
}

// ------------------------------------------------------------------ rendering
function render(r) {
  state.result = r;
  $('#results').classList.remove('hidden');
  const p = r.profile || {};
  $('#brief').innerHTML = `<h2>Music brief <span class="small">(${esc(p.source || 'llm')})</span></h2>
    <p class="summary">${esc(p.summary || '')}</p>
    <div class="chips">${(p.moods || []).map(m => `<span class="chip static on">${esc(m)}</span>`).join('')}
      ${(p.genres || []).map(g => `<span class="chip static">${esc(g)}</span>`).join('')}
      ${(p.instruments || []).map(g => `<span class="chip static">${esc(g)}</span>`).join('')}</div>
    <div class="kv"><span>Energy <b>${(p.energy ?? 0).toFixed(2)}</b><span class="meter"><i style="width:${(p.energy || 0) * 100}%"></i></span></span>
      ${p.tempo ? `<span>Tempo <b>${p.tempo.min_bpm}-${p.tempo.max_bpm} BPM</b></span>` : ''}
      <span>Approach <b>${esc(p.relationship || 'match')}</b></span>
      ${p.vocals ? `<span>Vocals <b>${esc(p.vocals)}</b></span>` : ''}
      ${p.avoid && p.avoid.length ? `<span>Avoid <b>${esc(p.avoid.join(', '))}</b></span>` : ''}</div>
    ${p.music_description ? `<p class="small">${esc(p.music_description)}</p>` : ''}`;
  const a = r.analysis;
  if (a) {
    $('#analysis').classList.remove('hidden');
    $('#analysis').innerHTML = `<h2>Clip analysis</h2>
      <div class="kv"><span>Pacing <b>${esc(a.pacing)}</b></span><span>Cuts <b>${a.cuts.length}</b> (${a.cuts_per_min.toFixed(0)}/min)</span>
      <span>Avg shot <b>${a.avg_shot_s.toFixed(1)}s</b></span><span>Palette <b>${esc(a.palette)}</b></span>
      ${a.tempo_bpm ? `<span>Audio pulse <b>${a.tempo_bpm} BPM</b></span>` : ''}</div>
      <div class="cuts" title="cut points">${a.cuts.map(c => `<i style="left:${c / a.duration * 100}%"></i>`).join('')}</div>
      <div class="keys">${(a.keyframes || []).map(k => `<img alt="keyframe" src="${k.startsWith('data:') || k.startsWith('static/') ? k : 'api/media?path=' + encodeURIComponent(k)}">`).join('')}</div>
      ${a.captions && a.captions.length ? `<p class="small">Seen: ${esc(a.captions.join(' '))}</p>` : ''}
      ${a.subtitles && a.subtitles.length ? `<p class="small">Dialogue: “${esc(a.subtitles.slice(0, 6).join(' / '))}”</p>` : ''}`;
  } else $('#analysis').classList.add('hidden');
  $('#warnings').innerHTML = (r.warnings || []).map(w => `<div class="w">${esc(w)}</div>`).join('');
  $('#shortlist').innerHTML = '';
  for (const it of r.items) $('#shortlist').append(itemEl(it, r));
  renderTrace(r);
  $('#results').scrollIntoView({behavior: 'smooth', block: 'start'});
}

function itemEl(it, r) {
  const li = document.createElement('li'); li.className = 'item';
  const t = it.track, v = it.license, hitM = new Set(it.matched.moods || []), hitG = new Set(it.matched.genres || []);
  const cost = v.cost && v.cost.amount_usd === 0 ? '$0 + credit' : (v.cost && v.cost.amount_usd == null ? 'price unknown' : '');
  li.innerHTML = `<button class="play" aria-label="Play ${esc(t.title)}">▶</button>
    <div><div><span class="title">${it.rank}. ${esc(t.title)}</span> <span class="artist">· ${esc(t.artist)}</span></div>
      ${it.blurb ? `<p class="blurb">${esc(it.blurb)}</p>` : ''}
      <div class="why">${esc((it.why || []).join(' · '))}</div>
      <div class="tags">${t.moods.map(m => `<span class="${hitM.has(m) ? 'hit' : ''}">${esc(m)}</span>`).join('')}${t.genres.map(g => `<span class="${hitG.has(g) ? 'hit' : ''}">${esc(g)}</span>`).join('')}</div>
      ${it.flags && it.flags.length ? `<ul class="flags">${it.flags.map(f => `<li>${esc(f)}</li>`).join('')}</ul>` : ''}</div>
    <div class="lic"><span class="badge ${v.status}">${STATUS_LABEL[v.status]}</span><br>
      ${esc((v.license_id || 'no license').toUpperCase())}${v.version ? ' ' + esc(v.version) : ''} · ${esc(cost)}
      <details class="lic-detail"><summary>why</summary><div>${esc(v.reason)}</div>
        ${(v.conditions || []).map(c => `<div>• ${esc(c)}</div>`).join('')}
        <div>${esc((v.cost || {}).basis || '')}</div>
        ${(v.citations || []).map(c => `<blockquote>${esc(c.text.slice(0, 260))}${c.text.length > 260 ? '…' : ''}<br><a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title)} · ${esc(c.section)}</a></blockquote>`).join('')}
        ${v.source ? `<div>Source: <a href="${esc(v.source)}" target="_blank" rel="noopener">${v.verification === 'live_api' ? 'Jamendo API' : 'MTG-Jamendo license file'}</a></div>` : ''}
        ${v.attribution ? `<div>Credit: ${esc(v.attribution)}</div>` : ''}
        <div><a href="${esc(t.page_url)}" target="_blank" rel="noopener">Open on Jamendo</a></div></details>
      <div class="rate" title="Does this track fit the scene? Your labels calibrate the LLM judge.">
        <button class="pass" data-v="5">Fits</button><button class="fail" data-v="1">Doesn't fit</button></div></div>`;
  li.querySelector('.play').onclick = e => playAgainst(t, e.currentTarget);
  li.querySelectorAll('.rate button').forEach(b => b.onclick = () => rate(r, t, +b.dataset.v, b));
  return li;
}

function playAgainst(t, btn) {
  const p = $('#player'), vid = $('#preview');
  if (state.playing === t.id) { p.pause(); vid.pause(); btn.classList.remove('playing'); btn.textContent = '▶'; state.playing = null; return; }
  document.querySelectorAll('.play.playing').forEach(b => { b.classList.remove('playing'); b.textContent = '▶'; });
  p.src = t.audio_url; p.currentTime = 0;
  p.play().catch(() => setStatus('Could not stream this track here; open it on Jamendo instead.'));
  if (state.clipURL && state.tab === 'clip') { vid.muted = true; vid.currentTime = 0; vid.play(); }
  btn.classList.add('playing'); btn.textContent = '❚❚'; state.playing = t.id;
}

async function rate(r, t, v, btn) {
  btn.parentElement.querySelectorAll('button').forEach(b => b.classList.remove('on')); btn.classList.add('on');
  const rec = {trace_id: r.trace_id || 'static', track_id: t.id, rating: v, scene: (r.profile || {}).summary || ''};
  if (state.mode === 'live') await fetch('api/ratings', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(rec)});
  else { try { const k = 'needle-ratings'; const all = JSON.parse(localStorage.getItem(k) || '[]'); all.push(rec); localStorage.setItem(k, JSON.stringify(all)); } catch (e) {} }
}

async function renderTrace(r) {
  const box = $('#trace-body');
  if (!r.trace_id || state.mode !== 'live') {
    box.innerHTML = r.lite ? '<p class="small">In-browser search: no LLM calls, $0.</p>' :
      (r.cost ? `<p class="small">Precomputed run: ${r.cost.llm_calls} LLM calls, ${(r.cost.prompt_tokens + r.cost.completion_tokens).toLocaleString()} tokens, API spend $${r.cost.cost_usd.toFixed(4)}.</p>` : '');
    if (r.steps && r.steps.length) box.innerHTML += `<p class="small">Agent steps: ${r.steps.filter(s => s.tool).map(s => esc(s.tool)).join(' → ')}</p>`;
    return;
  }
  const t = await (await fetch(`api/traces/${r.trace_id}`)).json();
  const t0 = t.start, total = Math.max(1, t.ms);
  box.innerHTML = `<p class="small">Trace ${esc(t.id)} · ${(t.ms / 1000).toFixed(1)}s · ${t.totals.llm_calls} LLM calls · ${(t.totals.prompt_tokens + t.totals.completion_tokens).toLocaleString()} tokens · API spend $${t.totals.cost_usd.toFixed(4)} · same tokens on a hosted model at the configured shadow price: $${t.totals.shadow_cost_usd.toFixed(4)}</p>
    <p class="small">Agent steps: ${(r.steps || []).filter(s => s.tool).map(s => esc(s.tool) + (s.only_cleared ? ' (cleared only)' : '')).join(' → ')}</p>` +
    t.spans.map(s => `<div class="span"><span>${esc(s.name)}</span><span class="bar"><i style="left:${(s.start - t0) * 1000 / total * 100}%;width:${s.ms / total * 100}%"></i></span><span class="small">${s.ms} ms${s.prompt_tokens ? ` · ${s.prompt_tokens}+${s.completion_tokens} tok` : ''}</span></div>`).join('');
}

function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c])); }
boot();
