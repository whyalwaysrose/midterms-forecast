/* =========================================================================
   2026 Senate Forecast — dashboard
   Vanilla JS, no dependencies. Charts are hand-built SVG so the site is a
   handful of static files that GitHub Pages can serve with no build step.
   ========================================================================= */

const SCHEMA_VERSION = 5;
const NS = 'http://www.w3.org/2000/svg';

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- format */

const pct = (v) => `${(100 * v).toFixed(0)}%`;
/** A probability, to a tenth of a point, never rounded to 0% or 100%.
 *
 * The posterior assigns no race a probability of exactly zero or one, so
 * printing "100.0%" reports a certainty the model does not hold -- it is the
 * rounding, not the forecast, saying the race cannot be lost. Worse, it is the
 * safest-looking races where an upset would matter most. ">99.9%" is both
 * true and visibly a bound rather than a fact.
 *
 * The threshold is the display precision itself: anything that would round to
 * a bare 0.0 or 100.0 is shown as a bound instead. */
const pct1 = (v) => {
  const p = 100 * v;
  if (p >= 99.95) return '>99.9%';
  // Zero included: these are Monte Carlo estimates, so 0 of 4000 draws means
  // "below what 4000 draws can resolve", not "impossible".
  if (p < 0.05) return '<0.1%';
  return `${p.toFixed(1)}%`;
};

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

/** Short date, with the year only when it is needed to disambiguate.
 *
 * The trajectory charts run from mid-2025 to election day, so their two axis
 * endpoints read "Jul 15" and "Nov 3" — which any reader takes for the same
 * year. Passing the chart's full span lets a date carry its year exactly when
 * the span crosses one, and stay uncluttered when it does not.
 */
function fmtDateAxis(iso, spanIsos) {
  const year = (v) => v.slice(0, 4);
  const crossesYear = spanIsos.some((v) => year(v) !== year(spanIsos[0]));
  return crossesYear ? `${fmtDateShort(iso)} '${iso.slice(2, 4)}` : fmtDateShort(iso);
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

  // Who is actually on the ballot. Names come from the most recent poll in the
  // race, falling back to the nominee roster where nobody has polled it yet, so
  // an uncontested-in-the-polls race still names its candidates.
  const names = race.candidates ?? {};
  const demName = names.dem || 'Democratic candidate';
  const repName = names.rep || 'Republican candidate';
  const matchup = `<p class="matchup">
    <span class="dem-text"><strong>${esc(demName)}</strong> (D)</span>
    <span class="vs">vs</span>
    <span class="rep-text"><strong>${esc(repName)}</strong> (R)</span>
  </p>`;

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

  // The candidate chart carries everything the margin chart did -- the margin
  // and its interval are still in the tooltip -- plus the two shares and the
  // polls behind them, so showing both would be the same story drawn twice.
  const legend = `<div class="candidate-legend">
    <span><i style="background:var(--dem)"></i>${esc(demName)}</span>
    <span><i style="background:var(--rep)"></i>${esc(repName)}</span>
    <span class="subtle">Dots are individual polls</span>
  </div>`;

  $('drawer-body').innerHTML =
    notes + matchup + stats +
    '<h3 style="margin-top:20px">Share of the two-party vote</h3>' +
    legend +
    '<div id="drawer-chart" class="chart"></div>' +
    pollTable;

  // Reveal before drawing: a hidden container reports clientWidth 0, which
  // would send chartWidth() back to the unscaled fallback.
  $('drawer').hidden = false;
  document.body.style.overflow = 'hidden';

  renderCandidateChart($('drawer-chart'), race, { width: 540, height: 210 });
}

function closeDrawer() {
  $('drawer').hidden = true;
  document.body.style.overflow = '';
}

/* ------------------------------------------------------------------- main */

function renderMethodology(forecast) {
  const m = forecast.methodology ?? {};
  const summary = forecast.poll_summary ?? {};
  const cf = forecast.chamber_forecast ?? {};
  const set = (id, value) => { const el = $(id); if (el) el.textContent = value; };

  const unpolled = (forecast.races ?? []).filter((r) => r.poll_count === 0).length;
  set('m-unpolled', unpolled ? `${unpolled} races with no polling at all` : 'unpolled');
  set('m-nsims', (cf.n_simulations ?? 0).toLocaleString());
  set('m-racepolls', summary.n_race_polls ?? '—');
  set('m-genpolls', summary.n_national_polls ?? '—');
  set('m-pollsters', summary.n_pollsters ?? '—');

  // One decimal throughout: JSON drops the trailing zero, so 3.0 arrives as 3
  // and would read as a suspiciously round number next to 5.7.
  const pts = (v) => (typeof v === 'number' ? v.toFixed(1) : '—');
  const ede = m.election_day_error ?? {};
  set('m-nat', pts(ede.national_pts));
  set('m-state', pts(ede.state_pts));
  set('m-total', pts(ede.total_pts));

  // --- calibration ------------------------------------------------------
  const cal = m.calibration;
  if (cal) {
    set('m-calpolls', cal.n_polls.toLocaleString());
    set('m-calcycles', cal.n_cycles);
    set('m-calsource', cal.source);
    const rows = [
      ['National (shared by every race)', cal.fitted_national_pts, 'pts'],
      ['State-specific', cal.fitted_state_pts, 'pts'],
      ['Individual poll noise', cal.fitted_poll_pts, 'pts'],
      ['Poll design effect', cal.fitted_design_effect, '×'],
    ];
    $('m-caltable').innerHTML =
      '<thead><tr><th>Measured from history</th><th>Value</th></tr></thead><tbody>' +
      rows.map(([label, value, unit]) =>
        `<tr><td>${esc(label)}</td><td>${unit === '×' ? `${value.toFixed(2)}×` : `${value.toFixed(1)} pts`}</td></tr>`
      ).join('') + '</tbody>';
  }

  // --- backtest ---------------------------------------------------------
  const bt = m.backtest;
  if (bt) {
    set('m-btraces', bt.n_races);
    set('m-btcycles', bt.n_cycles);
    set('m-brier', bt.brier);
    set('m-skill', `${(100 * bt.skill_vs_naive).toFixed(0)}%`);

    $('m-reliability').innerHTML =
      '<thead><tr><th>Model said</th><th>Races</th><th>Actually won</th></tr></thead><tbody>' +
      bt.reliability.map((r) =>
        `<tr><td>${esc(r.bin)}</td><td>${r.n}</td>` +
        `<td>${pct(r.actual)}</td></tr>`
      ).join('') + '</tbody>';

    // Coverage: a bar for what happened, a notch for what was promised.
    $('m-coverage').innerHTML = Object.entries(bt.coverage)
      .sort((a, b) => +a[0] - +b[0])
      .map(([level, hit]) => {
        const target = +level;
        return `<div class="coverage-row">
          <span>${pct(target)} interval</span>
          <span class="coverage-track">
            <span class="coverage-fill" style="width:${(100 * hit).toFixed(1)}%"></span>
            <span class="coverage-target" style="left:${(100 * target).toFixed(1)}%"></span>
          </span>
          <span class="coverage-value">${pct(hit)}</span>
        </div>`;
      }).join('');
  }
}

/** Remember whether the reader collapsed the primer.
 *
 * It is written for a first-time visitor, and this page is meant to be checked
 * repeatedly. Someone coming back daily should not have to scroll past the
 * explanation of what the Senate is every time — but a newcomer should not have
 * to find it either, so it starts open and stays however they last left it.
 */
function attachPrimer() {
  const primer = $('primer');
  if (!primer) return;
  const KEY = 'primer-collapsed';

  let stored = null;
  try {
    stored = localStorage.getItem(KEY);
  } catch { /* private browsing blocks storage; fall through to the default */ }

  if (stored === '1') {
    primer.open = false;
  } else if (stored === null && window.innerWidth <= 640) {
    // No stated preference, and a narrow screen. Open, the primer runs past a
    // full phone screen and pushes the forecast itself out of reach — so it
    // starts collapsed here, with the summary line still visible and inviting.
    // An explicit choice always wins over this default.
    primer.open = false;
  }

  primer.addEventListener('toggle', () => {
    try {
      localStorage.setItem(KEY, primer.open ? '0' : '1');
    } catch { /* nothing to do; the preference simply will not persist */ }
  });
}

function attachMethodTabs() {
  const nav = $('method-nav');
  if (!nav) return;
  nav.addEventListener('click', (e) => {
    const tab = e.target.closest('.method-tab');
    if (!tab) return;
    for (const button of nav.querySelectorAll('.method-tab')) {
      const selected = button === tab;
      button.classList.toggle('active', selected);
      const panel = $(button.dataset.panel);
      if (panel) panel.hidden = !selected;
    }
  });
}

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
  renderMethodology(forecast);
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

  attachPrimer();
  attachMethodTabs();

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
