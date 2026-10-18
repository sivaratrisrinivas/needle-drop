// Needle Drop front end. One field; everything else appears only when there is something to do with it.
//  live   - talks to the FastAPI backend (LLM brief, clip analysis, agent, license RAG, traces)
//  static - no backend (e.g. GitHub Pages): precomputed showcase results, plus an in-browser
//           tag search and in-browser clip analysis over the same catalog and license rules.
const $ = s => document.querySelector(s);
const USES = [['monetized', 'monetized video'], ['personal', 'personal video'], ['client', 'client work']];
const state = {mode: 'static', llm: true, use: 'monetized', only: false, clipFile: null, srtFile: null, sampleClip: null, clipURL: null,
               lite: null, showcase: null, result: null, playing: null, playToken: 0, busy: false};
const EXAMPLES = [
  'Two old friends say goodbye at a rainy train station; one boards and the train pulls away.',
  'A getaway driver launches into a screeching car chase through city traffic at night.',
  'Dawn over a quiet beach; an old man walks alone along the water remembering his father.',
  'A hitman in a black suit casually dances around a bound cop in an empty warehouse before cutting off his ear.',
];
const LICENSE_WORD = {cleared: 'cleared*', needs_review: 'needs review*', not_cleared: 'not cleared', unverified: 'unverified'};
let exIdx = 0;

async function boot() {
  try {
    const r = await fetch('api/health', {cache: 'no-store'});
    if (r.ok && (r.headers.get('content-type') || '').includes('json')) {
      const h = await r.json(); state.mode = 'live'; state.llm = h.llm;
    }
  } catch (e) { /* static */ }
  // only say something about the mode when it changes what the user gets
  if (state.mode === 'static') setStatus('demo: runs in your browser, with a keyword brief');
  else if (!state.llm) setStatus('no language model running: keyword brief only');

  const f = $('#scene');
  const grow = () => { f.style.height = 'auto'; f.style.height = f.scrollHeight + 'px'; };
  f.addEventListener('input', grow);
  f.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); go(); }
    else if (e.key === 'Tab' && !e.shiftKey && !f.value && f.placeholder) { e.preventDefault(); f.value = f.placeholder; grow(); }
  });
  const rotate = () => { f.classList.add('swap'); setTimeout(() => { f.placeholder = EXAMPLES[exIdx++ % EXAMPLES.length]; f.classList.remove('swap'); }, 400); };
  f.placeholder = EXAMPLES[exIdx++ % EXAMPLES.length];
  setInterval(() => { if (!f.value && document.activeElement !== f && !document.body.classList.contains('has-out')) rotate(); }, 5000);
  f.addEventListener('focus', () => { if (!f.value) grow(); });

  const useBtn = $('#use'), paint = () => { useBtn.textContent = USES.find(u => u[0] === state.use)[1]; };
  paint();
  useBtn.onclick = () => { state.use = USES[(USES.findIndex(u => u[0] === state.use) + 1) % USES.length][0]; paint(); if (state.result) go(); };

  $('#pick').onclick = () => $('#file').click();
  $('#file').onchange = e => { takeFiles([...e.target.files]); e.target.value = ''; };
  document.querySelectorAll('[data-sample]').forEach(b => b.onclick = () => useSample(b.dataset.sample));
  $('#unclip').onclick = clearClip;
  $('#only').onclick = e => { state.only = !state.only; e.currentTarget.setAttribute('aria-pressed', state.only); if (state.result) render(state.result); };
  wireToggle('#brief', '#briefmore'); wireToggle('#clipline', '#clipmore'); wireToggle('#tracebtn', '#trace');

  let depth = 0;
  window.addEventListener('dragenter', e => { if (hasFiles(e)) { depth++; $('#drop').hidden = false; } });
  window.addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; $('#drop').hidden = true; } });
  window.addEventListener('dragover', e => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener('drop', e => { e.preventDefault(); depth = 0; $('#drop').hidden = true; takeFiles([...e.dataTransfer.files]); });
  loadLite();
}

const hasFiles = e => e.dataTransfer && [...(e.dataTransfer.types || [])].includes('Files');

function wireToggle(btn, box) {
  const b = $(btn), x = $(box);
  const flip = () => { const open = x.hidden; x.hidden = !open; b.setAttribute('aria-expanded', open); };
  b.onclick = flip; b.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); flip(); } };
}

// ------------------------------------------------------------------ clip
function takeFiles(files) {
  const srt = files.find(f => /\.srt$/i.test(f.name)), vid = files.find(f => f.type.startsWith('video/') || /\.(mp4|mov|webm|mkv|m4v)$/i.test(f.name));
  if (srt) state.srtFile = srt;
  if (vid) { setClip(vid); go(); }
  else if (srt && state.clipFile) go();
}

function setClip(file, sampleName = null) {
  state.clipFile = file; state.sampleClip = sampleName;
  if (state.clipURL) URL.revokeObjectURL(state.clipURL);
  state.clipURL = URL.createObjectURL(file);
  const v = $('#bg'); v.src = state.clipURL; v.play().catch(() => {});
  document.body.classList.add('has-clip');
  $('#clipname').textContent = `${file.name} · ${(file.size / 1e6).toFixed(1)} MB`;
  $('#clip').hidden = false;
}

function clearClip() {
  state.clipFile = state.srtFile = state.sampleClip = null;
  if (state.clipURL) URL.revokeObjectURL(state.clipURL);
  state.clipURL = null; const v = $('#bg'); v.pause(); v.removeAttribute('src'); v.load();
  document.body.classList.remove('has-clip'); $('#clip').hidden = true;
}

async function useSample(name) {
  setStatus('loading the clip', true);
  const b = await (await fetch(`samples/${name}.webm`)).blob();
  setClip(new File([b], `${name}.webm`, {type: 'video/webm'}), name);
  try { const s = await fetch(`samples/${name}.srt`); if (s.ok) state.srtFile = new File([await s.blob()], `${name}.srt`); } catch (e) {}
  go();
}

function setStatus(msg, busy = false) {
  const s = $('#status'); s.textContent = msg || ''; s.classList.toggle('busy', !!busy);
}

async function go() {
  if (state.busy) return;
  const text = $('#scene').value.trim();
  if (!state.clipFile && !text) { setStatus('describe a scene, or drop a clip'); $('#scene').focus(); return; }
  state.busy = true; stopPlayback();
  try {
    const result = state.mode === 'live' ? await liveRun(text) : await staticRun(text);
    if (result) render(result);
  } catch (e) { setStatus('something went wrong: ' + e.message); }
  finally { state.busy = false; }
}

// ------------------------------------------------------------------ live mode
async function liveRun(text) {
  let job;
  if (state.clipFile) {
    const fd = new FormData(); fd.append('file', state.clipFile); fd.append('use', state.use); fd.append('text', text);
    if (state.srtFile) fd.append('srt', state.srtFile);
    job = await (await fetch('api/clips', {method: 'POST', body: fd})).json();
  } else {
    job = await (await fetch('api/search', {method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({text, use: state.use})})).json();
  }
  if (job.detail) throw new Error(job.detail);
  const t0 = Date.now();
  while (true) {
    const j = await (await fetch(`api/jobs/${job.id}`)).json();
    const secs = ((Date.now() - t0) / 1000).toFixed(0);
    if (j.status === 'done') { setStatus(j.result.cache_hit ? 'from cache' : ''); return j.result; }
    if (j.status === 'error') throw new Error(j.error.split('\n')[0]);
    const step = await currentStep(j.trace_id);
    setStatus(j.status === 'queued' ? `queued, position ${j.queue_position + 1}` : `${step} · ${secs}s`, true);
    await new Promise(r => setTimeout(r, 1200));
  }
}

async function currentStep(tid) {
  try {
    const t = await (await fetch(`api/traces/${tid}`)).json();
    const last = (t.spans || []).filter(s => s.name.startsWith('tool.') || s.name.startsWith('clip.') || s.name.startsWith('llm.')).pop();
    const names = {'clip.analyze': 'analyzing the clip', 'clip.cuts': 'finding cuts', 'clip.frames': 'reading frames',
      'clip.audio': 'listening', 'tool.analyze_clip': 'analyzing the clip', 'llm.scene_profile': 'writing the brief',
      'tool.parse_scene': 'writing the brief', 'tool.search_catalog': 'searching the catalog', 'tool.check_licenses': 'checking licenses',
      'llm.agent_plan': 'planning', 'llm.explain': 'explaining the picks', 'vlm.caption': 'looking at keyframes'};
    return last ? (names[last.name] || 'working') : 'working';
  } catch (e) { return 'working'; }
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

async function staticRun(text) {
  await loadLite();
  const show = state.showcase || {};
  if (state.clipFile) {
    if (state.sampleClip && show.clips && show.clips[state.sampleClip]) { setStatus('precomputed result for this sample'); return show.clips[state.sampleClip]; }
    setStatus('analyzing your clip', true);
    $('#bg').pause(); const a = await analyzeInBrowser($('#bg')); $('#bg').play().catch(() => {});
    const r = await liteSearch(briefFromAnalysis(a)); r.analysis = a; setStatus(''); return r;
  }
  const pre = (show.scenes || []).find(s => s.text.trim() === text);
  if (pre) { setStatus('precomputed result for this example'); return pre.result; }
  setStatus('');
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
  const use = state.use, clearedOnly = state.only;
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
    setStatus(`analyzing your clip · ${Math.round(t / dur * 100)}%`, true);
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
  document.body.classList.add('has-out'); $('#out').hidden = false; $('#scene').dispatchEvent(new Event('input'));
  const p = r.profile || {}, a = r.analysis;
  let bt = p.music_description || p.summary || 'brief';
  if (a && a.description && bt.startsWith(a.description.slice(0, 40))) bt = [...(p.moods || []), ...(p.genres || [])].slice(0, 6).join(' · ') || 'brief'; $('#brief').textContent = bt.length > 150 ? bt.slice(0, 147).trimEnd() + '…' : bt;
  const bm = [];
  const tags = [...(p.moods || []), ...(p.genres || []), ...(p.instruments || [])];
  if (tags.length) bm.push(`<p><b>${esc(tags.join(' · '))}</b></p>`);
  const kv = [`energy <b>${(p.energy ?? 0).toFixed(2)}</b>`];
  if (p.tempo) kv.push(`tempo <b>${p.tempo.min_bpm}–${p.tempo.max_bpm} bpm</b>`);
  kv.push(`approach <b>${esc(p.relationship || 'match')}</b>`);
  if (p.vocals && p.vocals !== 'any') kv.push(`vocals <b>${esc(p.vocals)}</b>`);
  if (p.avoid && p.avoid.length) kv.push(`avoid <b>${esc(p.avoid.join(', '))}</b>`);
  bm.push(`<p>${kv.join(' · ')}</p>`);
  if (p.music_description) bm.push(`<p>${esc(p.music_description)}</p>`);
  if (p.source && p.source !== 'llm') bm.push(`<p>brief source: ${esc(p.source)}</p>`);
  $('#briefmore').innerHTML = bm.join('');
  $('#briefmore').hidden = true; $('#brief').setAttribute('aria-expanded', 'false');

  const cl = $('#clipline');
  if (a) {
    cl.hidden = false; cl.setAttribute('aria-expanded', 'false'); $('#clipmore').hidden = true;
    cl.textContent = [`${a.cuts.length} cuts`, a.tempo_bpm ? `${a.tempo_bpm} bpm pulse` : null, a.pacing, a.palette].filter(Boolean).join(' · ');
    $('#clipmore').innerHTML = `<div class="ticks" title="cut points">${a.cuts.map(c => `<i style="left:${c / a.duration * 100}%"></i>`).join('')}</div>
      <div>${(a.keyframes || []).map(k => `<img alt="keyframe" src="${k.startsWith('data:') || k.startsWith('static/') ? k : 'api/media?path=' + encodeURIComponent(k)}">`).join('')}</div>
      ${a.captions && a.captions.length ? `<p>seen: ${esc(a.captions.join(' '))}</p>` : ''}
      ${a.subtitles && a.subtitles.length ? `<p>dialogue: “${esc(a.subtitles.slice(0, 6).join(' / '))}”</p>` : ''}`;
  } else { cl.hidden = true; $('#clipmore').hidden = true; }

  $('#warn').innerHTML = (r.warnings || []).map(w => esc(w)).join('<br>');
  const list = $('#crate'); list.innerHTML = '';
  const items = state.only ? r.items.filter(i => i.license.status === 'cleared') : r.items;
  items.forEach((it, i) => list.append(itemEl(it, r, i)));
  $('#count').textContent = `${items.length} track${items.length === 1 ? '' : 's'}`;
  $('#note').textContent = items.some(i => ['cleared', 'needs_review'].includes(i.license.status))
    ? '* checked against the 2019 Jamendo license snapshot. Confirm on the track page before you publish.' : '';
  renderTrace(r);
  const top = $('#out').getBoundingClientRect().top;
  if (top > innerHeight * .6) $('#out').scrollIntoView({behavior: 'smooth', block: 'start'});
}

function itemEl(it, r, i) {
  const li = document.createElement('li'); const t = it.track, v = it.license;
  li.className = 't' + (v.status === 'cleared' || v.status === 'needs_review' ? '' : ' nope'); li.dataset.id = t.id;
  const lid = (v.license_id || 'no license').toUpperCase() + (v.version ? ' ' + v.version : '');
  li.innerHTML = `<div class="row">
      <button class="ti" aria-label="Play ${esc(t.title)} by ${esc(t.artist)}">${esc(t.title)}<span class="ar">${esc(t.artist)}</span></button>
      <span class="lic" title="${esc(v.reason || '')}"><i class="${v.status}"></i>${LICENSE_WORD[v.status] || v.status} · ${esc(lid)}</span>
      <button class="exp" aria-label="Details" aria-expanded="false">+</button></div>
    <div class="groove"><i></i></div><div class="d" hidden></div>`;
  li.querySelector('.ti').onclick = () => playAgainst(t, li);
  const exp = li.querySelector('.exp'), d = li.querySelector('.d');
  exp.onclick = () => {
    if (!d.innerHTML) fillDetail(d, it, r);
    d.hidden = !d.hidden; exp.setAttribute('aria-expanded', !d.hidden); exp.textContent = d.hidden ? '+' : '−';
  };
  return li;
}

function fillDetail(d, it, r) {
  const v = it.license, t = it.track, bits = [];
  if (it.blurb && it.blurb.trim().length > 24) bits.push(`<p><b>${esc(it.blurb.replace(/^[-–]\s*/, ''))}</b></p>`);
  bits.push(`<p>${esc((it.why || []).join(' · '))}</p>`);
  bits.push(`<p><b>${esc(v.reason || '')}</b></p>`);
  (v.conditions || []).forEach(c => bits.push(`<p>· ${esc(c)}</p>`));
  if (v.cost && v.cost.basis) bits.push(`<p>cost: ${esc(v.cost.basis)}</p>`);
  (v.citations || []).slice(0, 2).forEach(c => bits.push(`<blockquote>${esc(c.text.slice(0, 260))}${c.text.length > 260 ? '…' : ''}</blockquote><p><a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title)} · ${esc(c.section)}</a></p>`));
  (it.flags || []).filter(f => !/2019 MTG-Jamendo snapshot/.test(f)).forEach(f => bits.push(`<p>⚑ ${esc(f)}</p>`));
  if (v.attribution) bits.push(`<p>credit: ${esc(v.attribution)}</p>`);
  bits.push(`<p>${v.source ? `<a href="${esc(v.source)}" target="_blank" rel="noopener">license source</a> · ` : ''}<a href="${esc(t.page_url)}" target="_blank" rel="noopener">open on Jamendo</a></p>`);
  bits.push(`<div class="rate" title="Your labels calibrate the LLM judge"><button data-v="5">fits</button><button data-v="1">doesn't fit</button></div>`);
  d.innerHTML = bits.join('');
  d.querySelectorAll('.rate button').forEach(b => b.onclick = () => rate(r, t, +b.dataset.v, b));
}

// ------------------------------------------------------------------ playback
// Jamendo stream URL variants, tried in order until one actually loads (the host layout has changed over time).
function audioCandidates(t) {
  const id = (t.id || '').split(':')[1] || (t.audio_url.match(/trackid=(\d+)/) || [])[1];
  return [...new Set([t.audio_url, `https://prod-1.storage.jamendo.com/?trackid=${id}&format=mp31&from=app-devsite`,
          `https://mp3l.jamendo.com/?trackid=${id}&format=mp31&from=app-97dab294`])];
}

function stopPlayback() {
  const p = $('#player'); state.playToken++; p.pause(); p.removeAttribute('src'); p.load();
  document.body.classList.remove('playing');
  document.querySelectorAll('.t.on').forEach(l => { l.classList.remove('on'); l.querySelector('.groove i').style.width = '0'; });
  if (state.clipURL) $('#bg').play().catch(() => {});
  state.playing = null;
}

function playAgainst(t, li) {
  const p = $('#player'), vid = $('#bg');
  if (state.playing === t.id) { stopPlayback(); return; }
  stopPlayback(); state.playing = t.id; const token = state.playToken;
  const urls = audioCandidates(t); let i = 0;
  const fail = () => {
    if (state.playToken !== token) return;
    if (++i < urls.length) { p.src = urls[i]; p.play().catch(() => {}); return; }
    stopPlayback(); setStatus(`could not stream "${t.title}" here; open it on Jamendo from its details`);
  };
  p.onerror = fail;
  p.onplaying = () => {
    if (state.playToken !== token) return;
    setStatus(''); li.classList.add('on'); document.body.classList.add('playing');
    if (state.clipURL) { vid.muted = true; vid.currentTime = 0; vid.play().catch(() => {}); }
  };
  p.ontimeupdate = () => { if (state.playToken === token && p.duration) li.querySelector('.groove i').style.width = (p.currentTime / p.duration * 100) + '%'; };
  p.onended = () => { if (state.playToken === token) stopPlayback(); };
  setStatus('loading audio', true); p.src = urls[0];
  p.play().catch(e => { if (e && e.name === 'NotAllowedError') { stopPlayback(); setStatus('the browser blocked playback; click again'); } });
}

async function rate(r, t, v, btn) {
  btn.parentElement.querySelectorAll('button').forEach(b => b.classList.remove('on')); btn.classList.add('on');
  const rec = {trace_id: r.trace_id || 'static', track_id: t.id, rating: v, scene: (r.profile || {}).summary || ''};
  if (state.mode === 'live') await fetch('api/ratings', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(rec)});
  else { try { const k = 'needle-ratings'; const all = JSON.parse(localStorage.getItem(k) || '[]'); all.push(rec); localStorage.setItem(k, JSON.stringify(all)); } catch (e) {} }
}

async function renderTrace(r) {
  const box = $('#trace'); $('#trace').hidden = true; $('#tracebtn').setAttribute('aria-expanded', 'false');
  if (!r.trace_id || state.mode !== 'live') {
    box.innerHTML = r.lite ? '<p>in-browser search: no model calls, $0</p>' :
      (r.cost ? `<p>precomputed run: ${r.cost.llm_calls} model calls, ${(r.cost.prompt_tokens + r.cost.completion_tokens).toLocaleString()} tokens, spend $${r.cost.cost_usd.toFixed(4)}</p>` : '');
    if (r.steps && r.steps.length) box.innerHTML += `<p>steps: ${r.steps.filter(s => s.tool).map(s => esc(s.tool)).join(' → ')}</p>`;
    return;
  }
  const t = await (await fetch(`api/traces/${r.trace_id}`)).json();
  const total = Math.max(1, t.ms), t0 = t.start;
  box.innerHTML = `<p>${(t.ms / 1000).toFixed(1)}s · ${t.totals.llm_calls} model calls · ${(t.totals.prompt_tokens + t.totals.completion_tokens).toLocaleString()} tokens · spend $${t.totals.cost_usd.toFixed(4)} (local model; the same tokens on a hosted model would be $${t.totals.shadow_cost_usd.toFixed(4)})</p>
    <p>steps: ${(r.steps || []).filter(s => s.tool).map(s => esc(s.tool) + (s.only_cleared ? ' (cleared only)' : '')).join(' → ')}</p>` +
    t.spans.map(s => `<div class="span"><span>${esc(s.name)}</span><span class="bar"><i style="left:${(s.start - t0) * 1000 / total * 100}%;width:${s.ms / total * 100}%"></i></span><span>${s.ms} ms${s.prompt_tokens ? ` · ${s.prompt_tokens}+${s.completion_tokens} tok` : ''}</span></div>`).join('');
}

function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c])); }
boot();
