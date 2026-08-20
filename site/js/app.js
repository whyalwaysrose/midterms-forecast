/* =========================================================================
   2026 Senate Forecast — dashboard
   Vanilla JS, no dependencies. Charts are hand-built SVG so the site is a
   handful of static files that GitHub Pages can serve with no build step.
   ========================================================================= */

const SCHEMA_VERSION = 2;
const NS = 'http://www.w3.org/2000/svg';

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- format */

const pct = (v) => `${(100 * v).toFixed(0)}%`;
const pct1 = (v) => `${(100 * v).toFixed(1)}%`;

/** A margin as a party-leading label: 3.2 -> "D+3.2". */
function margin(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v >= 0 ? `D+${v.toFixed(1)}` : `R+${Math.abs(v).toFixed(1)}`;
}

function marginClass(v) {
  return v >= 0 ? 'dem-text' : 'rep-text';
}

function fmtDate(iso) {
  const d = new Date(iso + 'T00:00:00Z');
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
}

function fmtDateShort(iso) {
  const d = new Date(iso + 'T00:00:00Z');
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
}

/** Escape text before it goes anywhere near innerHTML. */
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/** Minimal inline markdown: **bold** only. Escapes everything else first. */
function inlineMd(s) {
  return esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

/* ------------------------------------------------------------------- svg */

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

/** Width to draw a chart at, matched to its container.
 *
 * The SVG scales uniformly to fit its host, so a viewBox wider than the
 * container shrinks the axis text with it — a 10px label in a 560-wide viewBox
 * rendered into 340px comes out at about 6px and is unreadable on a phone.
 * Drawing at roughly the container's own width keeps the scale near 1:1.
 */
function chartWidth(host, preferred) {
  const available = host.clientWidth || 0;
  if (!available) return preferred;
  return Math.round(Math.max(280, Math.min(preferred, available)));
}

function makeSvg(width, height) {
  const svg = svgEl('svg', {
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: 'xMidYMid meet',
    role: 'img',
  });
  return svg;
}

/* ------------------------------------------------------------- seat chart */

function renderSeatChart(forecast) {
  const host = $('seat-chart');
  host.textContent = '';

  const dist = forecast.chamber_forecast.seat_distribution;
  const threshold = forecast.chamber_forecast.dem_seats_for_majority;
  const entries = Object.entries(dist)
    .map(([seats, p]) => ({ seats: +seats, p }))
    .filter((d) => d.p > 0.0005)
    .sort((a, b) => a.seats - b.seats);

  if (!entries.length) return;

  const W = chartWidth(host, 560), H = 240, padL = 34, padR = 12, padT = 12, padB = 34;
  const svg = makeSvg(W, H);

  const minSeat = entries[0].seats, maxSeat = entries[entries.length - 1].seats;
  const nBars = maxSeat - minSeat + 1;
  const maxP = Math.max(...entries.map((d) => d.p));
  const bw = (W - padL - padR) / nBars;

  const x = (s) => padL + (s - minSeat) * bw;
  const y = (p) => padT + (H - padT - padB) * (1 - p / maxP);

  // horizontal grid
  for (let i = 0; i <= 3; i++) {
    const gy = padT + ((H - padT - padB) * i) / 3;
    svg.appendChild(svgEl('line', { x1: padL, x2: W - padR, y1: gy, y2: gy, class: 'grid-line' }));
  }

  for (const d of entries) {
    const h = Math.max(1, H - padB - y(d.p));
    const bar = svgEl('rect', {
      x: x(d.seats) + bw * 0.12,
      y: y(d.p),
      width: Math.max(1, bw * 0.76),
      height: h,
      rx: Math.min(2, bw * 0.3),
      fill: d.seats >= threshold ? 'var(--dem)' : 'var(--rep)',
      opacity: 0.9,
    });
    const title = svgEl('title');
    title.textContent = `${d.seats} Democratic seats — ${(100 * d.p).toFixed(1)}%`;
    bar.appendChild(title);
    svg.appendChild(bar);
  }

  // majority threshold sits on the boundary between the last R bar and first D bar
  const tx = x(threshold);
  svg.appendChild(svgEl('line', { x1: tx, x2: tx, y1: padT - 4, y2: H - padB, class: 'threshold-line' }));
  const label = svgEl('text', { x: tx + 5, y: padT + 6, class: 'threshold-text' });
  label.textContent = `${threshold} = majority`;
  svg.appendChild(label);

  // x ticks, roughly every 2-3 seats
  const step = nBars > 16 ? 3 : nBars > 9 ? 2 : 1;
  for (let s = minSeat; s <= maxSeat; s += step) {
    const t = svgEl('text', { x: x(s) + bw / 2, y: H - padB + 15, class: 'axis-label', 'text-anchor': 'middle' });
    t.textContent = s;
    svg.appendChild(t);
  }
  const xlab = svgEl('text', { x: (padL + W - padR) / 2, y: H - 4, class: 'axis-label', 'text-anchor': 'middle' });
  xlab.textContent = 'Democratic-caucus seats';
  svg.appendChild(xlab);

  host.appendChild(svg);
}

/* ---------------------------------------------------------- history chart */

function renderHistoryChart(history) {
  const host = $('history-chart');
  host.textContent = '';

  const runs = (history?.runs ?? []).filter((r) => typeof r.dem_control_prob === 'number');
  if (runs.length < 2) {
    $('history-empty').hidden = false;
    return;
  }
  $('history-empty').hidden = true;

  const W = chartWidth(host, 560), H = 240, padL = 40, padR = 14, padT = 14, padB = 34;
  const svg = makeSvg(W, H);

  const t0 = Date.parse(runs[0].run_date);
  const t1 = Date.parse(runs[runs.length - 1].run_date);
  const span = Math.max(1, t1 - t0);
  const x = (iso) => padL + ((Date.parse(iso) - t0) / span) * (W - padL - padR);
  const y = (p) => padT + (1 - p) * (H - padT - padB);

  for (let i = 0; i <= 4; i++) {
    const p = i / 4;
    const gy = y(p);
    svg.appendChild(svgEl('line', { x1: padL, x2: W - padR, y1: gy, y2: gy, class: 'grid-line' }));
    const t = svgEl('text', { x: padL - 7, y: gy + 3, class: 'axis-label', 'text-anchor': 'end' });
    t.textContent = pct(p);
    svg.appendChild(t);
  }

  // 50% reference
  svg.appendChild(svgEl('line', { x1: padL, x2: W - padR, y1: y(0.5), y2: y(0.5), class: 'threshold-line' }));

  const pts = runs.map((r) => `${x(r.run_date)},${y(r.dem_control_prob)}`).join(' ');
  svg.appendChild(svgEl('polyline', {
    points: pts, fill: 'none', stroke: 'var(--dem)', 'stroke-width': 2,
    'stroke-linejoin': 'round', 'stroke-linecap': 'round',
  }));

  for (const r of runs) {
    const c = svgEl('circle', { cx: x(r.run_date), cy: y(r.dem_control_prob), r: 2.6, fill: 'var(--dem)' });
    const title = svgEl('title');
    title.textContent = `${r.run_date}: ${pct1(r.dem_control_prob)} (${r.dem_seats_median} seats)`;
    c.appendChild(title);
    svg.appendChild(c);
  }

  for (const iso of [runs[0].run_date, runs[runs.length - 1].run_date]) {
    const t = svgEl('text', {
      x: x(iso), y: H - padB + 16, class: 'axis-label',
      'text-anchor': iso === runs[0].run_date ? 'start' : 'end',
    });
    t.textContent = fmtDateShort(iso);
    svg.appendChild(t);
  }

  host.appendChild(svg);
}

/* -------------------------------------------------- trajectory (band+line) */

function renderTrajectory(host, trajectory, opts = {}) {
  host.textContent = '';
  if (!trajectory?.length) return;

  const W = chartWidth(host, opts.width ?? 560), H = opts.height ?? 200;
  const padL = 38, padR = 12, padT = 12, padB = 26;
  const svg = makeSvg(W, H);

  const lo = Math.min(...trajectory.map((d) => d.p05));
  const hi = Math.max(...trajectory.map((d) => d.p95));
  const pad = Math.max(1, (hi - lo) * 0.08);
  const yMin = lo - pad, yMax = hi + pad;

  const t0 = Date.parse(trajectory[0].date);
  const t1 = Date.parse(trajectory[trajectory.length - 1].date);
  const span = Math.max(1, t1 - t0);
  const x = (iso) => padL + ((Date.parse(iso) - t0) / span) * (W - padL - padR);
  const y = (v) => padT + (1 - (v - yMin) / (yMax - yMin)) * (H - padT - padB);

  // zero line = tied race
  if (yMin < 0 && yMax > 0) {
    svg.appendChild(svgEl('line', { x1: padL, x2: W - padR, y1: y(0), y2: y(0), class: 'threshold-line' }));
  }

  // 90% band
  const top = trajectory.map((d) => `${x(d.date)},${y(d.p95)}`);
  const bottom = trajectory.slice().reverse().map((d) => `${x(d.date)},${y(d.p05)}`);
  svg.appendChild(svgEl('polygon', {
    points: [...top, ...bottom].join(' '),
    fill: 'var(--dem-soft)', stroke: 'none',
  }));

  svg.appendChild(svgEl('polyline', {
    points: trajectory.map((d) => `${x(d.date)},${y(d.p50)}`).join(' '),
    fill: 'none', stroke: 'var(--dem-line)', 'stroke-width': 2, 'stroke-linejoin': 'round',
  }));

  for (const v of [yMin, (yMin + yMax) / 2, yMax]) {
    const t = svgEl('text', { x: padL - 6, y: y(v) + 3, class: 'axis-label', 'text-anchor': 'end' });
    t.textContent = margin(v);
    svg.appendChild(t);
  }
  for (const [iso, anchor] of [[trajectory[0].date, 'start'], [trajectory[trajectory.length - 1].date, 'end']]) {
    const t = svgEl('text', { x: x(iso), y: H - 6, class: 'axis-label', 'text-anchor': anchor });
    t.textContent = fmtDateShort(iso);
    svg.appendChild(t);
  }

  host.appendChild(svg);
}

/* -------------------------------------------------------------- sparkline */

function sparkline(trajectory) {
  if (!trajectory?.length) return '';
  const W = 140, H = 26;
  const vals = trajectory.map((d) => d.p50);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const range = Math.max(0.5, hi - lo);
  const pts = vals.map((v, i) => {
    const x = (i / Math.max(1, vals.length - 1)) * W;
    const y = H - 3 - ((v - lo) / range) * (H - 6);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const zeroInside = lo < 0 && hi > 0;
  const zeroY = H - 3 - ((0 - lo) / range) * (H - 6);
  const zeroLine = zeroInside
    ? `<line x1="0" x2="${W}" y1="${zeroY.toFixed(1)}" y2="${zeroY.toFixed(1)}" stroke="var(--border)" stroke-width="1"/>`
    : '';

  const last = vals[vals.length - 1];
  const colour = last >= 0 ? 'var(--dem)' : 'var(--rep)';
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:26px">
    ${zeroLine}
    <polyline points="${pts}" fill="none" stroke="${colour}" stroke-width="1.6" stroke-linejoin="round"/>
  </svg>`;
}

/* ------------------------------------------------------------- commentary */

function commentaryHtml(entry) {
  const parts = [`<div class="entry">`];
  parts.push(`<div class="entry-date">${esc(fmtDate(entry.run_date))}</div>`);
  parts.push(`<div class="headline">${inlineMd(entry.headline)}</div>`);

  let listOpen = false;
  const closeList = () => { if (listOpen) { parts.push('</ul>'); listOpen = false; } };

  for (const block of entry.body ?? []) {
    for (const raw of String(block).split('\n')) {
      const line = raw.trimEnd();
      if (!line.trim()) { closeList(); continue; }

      if (/^\s{4,}-\s/.test(line)) {
        if (!listOpen) { parts.push('<ul>'); listOpen = true; }
        parts.push(`<li class="poll-line">${inlineMd(line.replace(/^\s*-\s/, ''))}</li>`);
      } else if (/^\s{2,}-\s/.test(line)) {
        if (!listOpen) { parts.push('<ul>'); listOpen = true; }
        parts.push(`<li>${inlineMd(line.replace(/^\s*-\s/, ''))}</li>`);
      } else {
        closeList();
        parts.push(`<p>${inlineMd(line)}</p>`);
      }
    }
  }
  closeList();
  parts.push('</div>');
  return parts.join('');
}

function renderCommentary(commentary) {
  const entries = commentary?.entries ?? [];
  if (!entries.length) {
    $('commentary-latest').innerHTML = '<p class="subtle">No commentary yet.</p>';
    $('commentary-archive-wrap').hidden = true;
    return;
  }
  $('commentary-latest').innerHTML = commentaryHtml(entries[0]);

  const rest = entries.slice(1, 30);
  if (!rest.length) {
    $('commentary-archive-wrap').hidden = true;
  } else {
    $('commentary-archive-wrap').hidden = false;
    $('commentary-archive').innerHTML = rest.map(commentaryHtml).join('');
  }
}

/* ------------------------------------------------------------------ races */

const INCUMBENCY_LABEL = {
  elected: 'incumbent running',
  appointed: 'appointed incumbent',
  open: 'open seat',
};

function raceRow(race) {
  const p = race.dem_win_prob;
  const meta = `${race.incumbent_party}-held · ${INCUMBENCY_LABEL[race.incumbent_status] ?? race.incumbent_status}` +
    (race.special ? ' · special' : '');
  const leader = p >= 0.5 ? 'dem-text' : 'rep-text';
  const shown = p >= 0.5 ? p : 1 - p;
  const party = p >= 0.5 ? 'D' : 'R';

  return `
  <button class="race-row" data-race="${esc(race.id)}">
    <span class="race-code">${esc(race.unit)}</span>
    <span>
      <span class="race-name">${esc(race.name)}</span><br>
      <span class="race-meta">${esc(meta)}</span>
    </span>
    <span class="race-prob ${leader}">${party} ${pct(shown)}</span>
    <span class="col-spark">
      <span class="pbar"><div style="width:${(100 * p).toFixed(1)}%"></div></span>
    </span>
    <span class="race-margin ${marginClass(race.margin.p50)}">${margin(race.margin.p50)}</span>
    <span class="race-polls col-polls">${race.poll_count}</span>
  </button>`;
}

let ALL_RACES = [];
let ACTIVE_FILTER = 'all';

function renderRaces() {
  const list = $('race-list');
  let races = ALL_RACES;
  if (ACTIVE_FILTER === 'competitive') {
    races = races.filter((r) => r.dem_win_prob > 0.05 && r.dem_win_prob < 0.95);
  } else if (ACTIVE_FILTER === 'polled') {
    races = races.filter((r) => r.poll_count > 0);
  }

  const header = `
  <div class="race-row race-head-row">
    <span>State</span><span>Race</span><span style="text-align:right">Favoured</span>
    <span class="col-spark">Win probability</span>
    <span style="text-align:right">Margin</span>
    <span class="col-polls" style="text-align:right">Polls</span>
  </div>`;

  list.innerHTML = header + (
    races.length
      ? races.map(raceRow).join('')
      : '<div class="race-row" style="cursor:default"><span></span><span class="subtle">No races match this filter.</span></div>'
  );

  list.querySelectorAll('[data-race]').forEach((btn) => {
    btn.addEventListener('click', () => openDrawer(btn.dataset.race));
  });
}

/* ----------------------------------------------------------------- drawer */

function openDrawer(raceId) {
  const race = ALL_RACES.find((r) => r.id === raceId);
  if (!race) return;

  $('drawer-title').textContent = `${race.name}${race.special ? ' (special election)' : ''}`;

  const notes = (race.notes ?? []).map((n) => `<div class="note">${esc(n)}</div>`).join('');

  const stats = `
  <div class="drawer-stats">
    <div class="dstat">
      <div class="dstat-value ${race.dem_win_prob >= 0.5 ? 'dem-text' : 'rep-text'}">${pct1(race.dem_win_prob)}</div>
      <div class="dstat-label">Dem win prob</div>
    </div>
    <div class="dstat">
      <div class="dstat-value ${marginClass(race.margin.p50)}">${margin(race.margin.p50)}</div>
      <div class="dstat-label">Median margin</div>
    </div>
    <div class="dstat">
      <div class="dstat-value">${pct1(race.tipping_point_prob)}</div>
      <div class="dstat-label">Tipping point</div>
    </div>
  </div>
  <p class="card-sub">90% interval <strong>${margin(race.margin.p05)}</strong> to <strong>${margin(race.margin.p95)}</strong>.
  Fundamentals alone (before polls, at a tied national environment) would put this race at
  <strong>${margin(race.fundamentals_prior_margin)}</strong>.</p>`;

  const pollRows = (race.polls ?? []).map((p) => {
    let tag = '';
    if (p.partisan_sign > 0) tag = '<span class="tag d">D sponsor</span>';
    else if (p.partisan_sign < 0) tag = '<span class="tag r">R sponsor</span>';
    const name = p.url
      ? `<a href="${esc(p.url)}" rel="noopener nofollow" target="_blank">${esc(p.pollster)}</a>`
      : esc(p.pollster);
    return `<tr>
      <td>${name}${tag}</td>
      <td class="num">${esc(fmtDateShort(p.date))}</td>
      <td class="num">${p.sample_size}</td>
      <td class="num">${esc(String(p.population).toUpperCase())}</td>
      <td class="num ${marginClass(p.margin)}">${margin(p.margin)}</td>
    </tr>`;
  }).join('');

  const pollTable = race.polls?.length
    ? `<h3 style="margin-top:22px">Recent polls (${race.poll_count} in window)</h3>
       <div class="table-scroll"><table class="polls">
         <thead><tr><th>Pollster</th><th style="text-align:right">Date</th><th style="text-align:right">N</th><th style="text-align:right">Screen</th><th style="text-align:right">Margin</th></tr></thead>
         <tbody>${pollRows}</tbody>
       </table></div>`
    : '<p class="subtle" style="margin-top:20px">No qualifying general-election polls in this race.</p>';

  $('drawer-body').innerHTML =
    notes + stats +
    '<h3 style="margin-top:20px">Estimated margin over time</h3>' +
    '<div id="drawer-chart" class="chart"></div>' +
    pollTable;

  // Reveal before drawing: a hidden container reports clientWidth 0, which
  // would send chartWidth() back to the unscaled fallback.
  $('drawer').hidden = false;
  document.body.style.overflow = 'hidden';

  renderTrajectory($('drawer-chart'), race.trajectory, { width: 540, height: 190 });
}

function closeDrawer() {
  $('drawer').hidden = true;
  document.body.style.overflow = '';
}

/* ------------------------------------------------------------------- main */

function renderDiagnostics(d) {
  const rHatOk = d.max_r_hat <= 1.05;
  const divOk = (d.divergences ?? 0) === 0;
  const items = [
    ['Max R-hat', d.max_r_hat?.toFixed(3), rHatOk],
    ['Min ESS (bulk)', Math.round(d.min_ess_bulk), d.min_ess_bulk >= 400],
    ['Divergences', d.divergences, divOk],
    ['Posterior draws', d.n_draws, true],
  ];
  $('diagnostics').innerHTML = items.map(([label, value, ok]) =>
    `<span class="diag-item">${esc(label)}: <b class="${ok ? 'diag-ok' : 'diag-warn'}">${esc(value)}</b></span>`
  ).join('');
}

let LAST_RENDER = null;

function render(forecast, history, commentary, geo) {
  // Re-runnable so a viewport change can redraw the charts at the new width.
  LAST_RENDER = () => render(forecast, history, commentary, geo);
  const cf = forecast.chamber_forecast;

  $('last-updated').textContent = fmtDate(forecast.run_date);
  $('days-left').textContent = forecast.days_to_election >= 0
    ? `${forecast.days_to_election} days to election day`
    : 'Election day has passed';

  $('dem-prob').textContent = pct(cf.dem_control_prob);
  $('rep-prob').textContent = pct(cf.rep_control_prob);
  $('prob-bar-dem').style.width = `${100 * cf.dem_control_prob}%`;
  $('prob-bar-rep').style.width = `${100 * cf.rep_control_prob}%`;

  $('seats-note').innerHTML =
    `Democrats are projected to hold <strong>${cf.dem_seats.median}</strong> seats ` +
    `(90% interval ${cf.dem_seats.p05}–${cf.dem_seats.p95}).`;
  $('tiebreak-note').textContent =
    `${cf.seats_not_up.D} Democratic and ${cf.seats_not_up.R} Republican seats are not up. ` +
    `Democrats need ${cf.dem_seats_for_majority}; the Vice President breaks a 50-50 tie for the ` +
    `${cf.tiebreaker_party === 'R' ? 'Republicans' : 'Democrats'}.`;

  $('n-sims').textContent = cf.n_simulations.toLocaleString();

  // Reveal the page before drawing anything. While `main` is hidden every
  // chart container reports clientWidth 0, so chartWidth() would fall back to
  // its unscaled default and the axis text would render at roughly half size.
  $('main').hidden = false;

  const gb = forecast.national.generic_ballot;
  $('gb-margin').textContent = margin(gb.dem_margin_median);
  $('gb-margin').className = `national-value ${marginClass(gb.dem_margin_median)}`;
  $('gb-range').textContent = `90%: ${margin(gb.dem_margin_p05)} to ${margin(gb.dem_margin_p95)}`;

  // The map goes first: it is the thing people look at, and it needs the page
  // visible so its container has a width to measure.
  renderMap(forecast, geo);

  renderSeatChart(forecast);
  renderHistoryChart(history);
  renderTrajectory($('national-chart'), forecast.national.trajectory, { width: 520, height: 190 });
  renderCommentary(commentary);

  ALL_RACES = forecast.races ?? [];
  renderRaces();
  renderDiagnostics(forecast.diagnostics ?? {});
}

async function loadJson(path, required) {
  // `midterms bundle` produces a single self-contained HTML file with the data
  // inlined, so it can be shared or opened from disk with no server at all.
  const embedded = window.__FORECAST_DATA__;
  if (embedded) {
    const key = path.replace(/^data\//, '').replace(/\.json$/, '');
    const value = embedded[key] ?? null;
    if (value === null && required) throw new Error(`missing embedded data: ${key}`);
    return value;
  }

  try {
    const res = await fetch(`${path}?v=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    if (required) throw err;
    return null;
  }
}

async function init() {
  try {
    const [forecast, history, commentary, geo] = await Promise.all([
      loadJson('data/forecast.json', true),
      loadJson('data/history.json', false),
      loadJson('data/commentary.json', false),
      loadJson('data/us-states.json', false),
    ]);

    if (forecast.schema_version !== SCHEMA_VERSION) {
      throw new Error(
        `Data is schema version ${forecast.schema_version} but this page expects ${SCHEMA_VERSION}. ` +
        `The site and the model are out of step — redeploy.`
      );
    }

    render(forecast, history, commentary, geo);
  } catch (err) {
    const banner = $('load-error');
    banner.hidden = false;
    banner.innerHTML =
      `<strong>Could not load the forecast.</strong> ${esc(err.message)}<br>` +
      `<span class="subtle">If you are running locally, serve the directory over HTTP ` +
      `(<code>python -m http.server</code>) — <code>file://</code> blocks fetch.</span>`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.filter').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      ACTIVE_FILTER = btn.dataset.filter;
      renderRaces();
    });
  });

  $('drawer-close').addEventListener('click', closeDrawer);
  $('drawer').addEventListener('click', (e) => { if (e.target.id === 'drawer') closeDrawer(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (LAST_RENDER) LAST_RENDER(); }, 150);
  });

  init();
});
