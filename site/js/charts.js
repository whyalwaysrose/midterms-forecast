/* =========================================================================
   2026 Senate Forecast — charts
   Loaded after app.js; uses its formatting helpers and svgEl()/chartWidth().
   ========================================================================= */

/** Round tick values covering [min, max] — the 1/2/5 x 10^n ladder.
 *
 * Axis labels should land on numbers a person would choose: 55%, 60%, 65%,
 * never 54.3%, 58.9%. Ticks are generated inside the domain rather than the
 * domain being stretched to them, so an auto-scaled axis stays tight.
 */
function niceTicks(min, max, target = 5) {
  if (!(max > min)) return [min];

  // Pick the step by which ladder rung the ideal step is *closest* to, rather
  // than always rounding up to the next rung. Rounding up systematically
  // undershoots the requested count — it left the national-environment axis
  // with two labels across a ten-point range. The sqrt thresholds are the
  // geometric midpoints between 1, 2, 5 and 10, so each rung wins the span it
  // is nearest to.
  const ideal = (max - min) / Math.max(1, target);
  let step = Math.pow(10, Math.floor(Math.log10(ideal)));
  const error = ideal / step;
  if (error >= Math.sqrt(50)) step *= 10;
  else if (error >= Math.sqrt(10)) step *= 5;
  else if (error >= Math.sqrt(2)) step *= 2;

  // Round to the step's own precision. Re-snapping via Math.round(v/step)*step
  // is not enough — the multiply puts the drift straight back, so 0.1 + 0.1 +
  // 0.1 still comes out as 0.30000000000000004.
  const decimals = Math.max(0, Math.ceil(-Math.log10(step)) + 1);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    ticks.push(Number(v.toFixed(decimals)));
  }
  return ticks;
}

/** A padded domain for a series, optionally forced to include a key value.
 *
 * The forecast-over-time chart is the reason this exists. Pinned to 0-100% its
 * line was a flat squiggle across the middle of the panel — technically honest,
 * practically unreadable, because every run has sat in a ten-point band. Scaling
 * to the data makes the movement visible; forcing 50% into the domain keeps the
 * one threshold that changes the story ("who is favoured") always on screen, so
 * the tighter axis cannot mislead.
 */
function paddedDomain(values, { include = null, padFraction = 0.25, minPad = 0, clamp = null } = {}) {
  const all = include === null ? values.slice() : values.concat([include]);
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  const pad = Math.max(minPad, (hi - lo) * padFraction);
  lo -= pad;
  hi += pad;
  if (clamp) {
    lo = Math.max(clamp[0], lo);
    hi = Math.min(clamp[1], hi);
  }
  if (!(hi > lo)) { lo -= 1; hi += 1; }
  return [lo, hi];
}

/** The floating tooltip for a chart, created once per host. */
function chartTip(host) {
  let tip = host.querySelector('.chart-tip');
  if (!tip) {
    tip = document.createElement('div');
    tip.className = 'chart-tip';
    tip.hidden = true;
    host.appendChild(tip);
  }
  return tip;
}

function placeTip(host, tip, clientX, clientY) {
  const hostBox = host.getBoundingClientRect();
  const tipBox = tip.getBoundingClientRect();
  let x = clientX - hostBox.left + 14;
  let y = clientY - hostBox.top - tipBox.height - 12;
  if (x + tipBox.width > hostBox.width) x = clientX - hostBox.left - tipBox.width - 14;
  if (y < 0) y = clientY - hostBox.top + 16;
  tip.style.left = `${Math.max(0, Math.min(x, hostBox.width - tipBox.width))}px`;
  tip.style.top = `${Math.max(0, y)}px`;
}

/** Pointer client-x to a value in the SVG's own viewBox coordinates. */
function toViewBoxX(svg, clientX) {
  const box = svg.getBoundingClientRect();
  const view = svg.getAttribute('viewBox').split(/\s+/).map(Number);
  if (!box.width) return view[0];
  return view[0] + ((clientX - box.left) / box.width) * view[2];
}

/* ------------------------------------------------------------- seat chart */

function renderSeatChart(forecast) {
  const host = $('seat-chart');
  host.textContent = '';

  const distribution = forecast.chamber_forecast.seat_distribution;
  const threshold = forecast.chamber_forecast.dem_seats_for_majority;
  const bars = Object.entries(distribution)
    .map(([seats, p]) => ({ seats: +seats, p }))
    .filter((d) => d.p > 0.0005)
    .sort((a, b) => a.seats - b.seats);
  if (!bars.length) return;

  // Probability of at least this many Democratic seats — the question a reader
  // actually has when looking at a seat histogram.
  let running = 0;
  for (let i = bars.length - 1; i >= 0; i--) {
    running += bars[i].p;
    bars[i].atLeast = running;
  }

  const W = chartWidth(host, 560), H = 250;
  const padL = 42, padR = 12, padT = 12, padB = 40;
  const svg = makeSvg(W, H);

  const minSeat = bars[0].seats, maxSeat = bars[bars.length - 1].seats;
  const nBars = maxSeat - minSeat + 1;
  const maxP = Math.max(...bars.map((d) => d.p));
  const barW = (W - padL - padR) / nBars;

  const x = (s) => padL + (s - minSeat) * barW;
  const y = (p) => padT + (H - padT - padB) * (1 - p / maxP);

  // y grid + labels, on round percentages
  for (const tick of niceTicks(0, maxP, 4)) {
    if (tick <= 0) continue;
    const gy = y(tick);
    svg.appendChild(svgEl('line', { x1: padL, x2: W - padR, y1: gy, y2: gy, class: 'grid-line' }));
    const label = svgEl('text', { x: padL - 7, y: gy + 3, class: 'axis-label', 'text-anchor': 'end' });
    label.textContent = `${(100 * tick).toFixed(tick < 0.02 ? 1 : 0)}%`;
    svg.appendChild(label);
  }
  svg.appendChild(svgEl('line', {
    x1: padL, x2: W - padR, y1: H - padB, y2: H - padB, class: 'axis-line',
  }));

  const barGroup = svgEl('g');
  for (const d of bars) {
    const rect = svgEl('rect', {
      x: x(d.seats) + barW * 0.12,
      y: y(d.p),
      width: Math.max(1, barW * 0.76),
      height: Math.max(1, H - padB - y(d.p)),
      rx: Math.min(2, barW * 0.3),
      fill: d.seats >= threshold ? 'var(--dem)' : 'var(--rep)',
      class: 'seat-bar',
      'data-seats': d.seats,
    });
    barGroup.appendChild(rect);
  }
  svg.appendChild(barGroup);

  const tx = x(threshold);
  svg.appendChild(svgEl('line', { x1: tx, x2: tx, y1: padT - 4, y2: H - padB, class: 'threshold-line' }));
  const thresholdLabel = svgEl('text', { x: tx + 5, y: padT + 6, class: 'threshold-text' });
  thresholdLabel.textContent = `${threshold} = majority`;
  svg.appendChild(thresholdLabel);

  const step = nBars > 16 ? 3 : nBars > 9 ? 2 : 1;
  for (let s = minSeat; s <= maxSeat; s += step) {
    const t = svgEl('text', {
      x: x(s) + barW / 2, y: H - padB + 16, class: 'axis-label', 'text-anchor': 'middle',
    });
    t.textContent = s;
    svg.appendChild(t);
  }
  const xLabel = svgEl('text', {
    x: (padL + W - padR) / 2, y: H - 6, class: 'axis-label', 'text-anchor': 'middle',
  });
  xLabel.textContent = 'Democratic-caucus seats';
  svg.appendChild(xLabel);

  host.appendChild(svg);

  // --- hover ---------------------------------------------------------------
  const tip = chartTip(host);
  const bySeat = new Map(bars.map((d) => [d.seats, d]));
  let active = null;

  svg.addEventListener('pointermove', (e) => {
    const seats = Math.round((toViewBoxX(svg, e.clientX) - padL - barW / 2) / barW) + minSeat;
    const d = bySeat.get(seats);
    if (!d) { hide(); return; }
    if (active !== seats) {
      active = seats;
      for (const bar of barGroup.children) {
        bar.classList.toggle('is-active', +bar.dataset.seats === seats);
      }
      const control = d.seats >= threshold ? 'Democratic majority' : 'Republican majority';
      tip.innerHTML =
        `<div class="tt-name">${d.seats} Democratic seats</div>` +
        `<div class="tt-meta">${esc(control)}</div>` +
        `<div class="tt-row"><span>Chance of exactly this</span><span>${pct1(d.p)}</span></div>` +
        `<div class="tt-row"><span>Chance of ${d.seats}+ seats</span><span>${pct1(d.atLeast)}</span></div>`;
    }
    tip.hidden = false;
    placeTip(host, tip, e.clientX, e.clientY);
  });

  function hide() {
    tip.hidden = true;
    active = null;
    for (const bar of barGroup.children) bar.classList.remove('is-active');
  }
  svg.addEventListener('pointerleave', hide);
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

  const W = chartWidth(host, 560), H = 250;
  const padL = 46, padR = 14, padT = 14, padB = 38;
  const svg = makeSvg(W, H);

  // Scale to the data, but never lose sight of 50%.
  const probs = runs.map((r) => r.dem_control_prob);
  const [lo, hi] = paddedDomain(probs, {
    include: 0.5, padFraction: 0.3, minPad: 0.02, clamp: [0, 1],
  });

  const t0 = Date.parse(runs[0].run_date);
  const t1 = Date.parse(runs[runs.length - 1].run_date);
  const span = Math.max(1, t1 - t0);
  const x = (iso) => padL + ((Date.parse(iso) - t0) / span) * (W - padL - padR);
  const y = (p) => padT + (1 - (p - lo) / (hi - lo)) * (H - padT - padB);

  for (const tick of niceTicks(lo, hi, 4)) {
    const gy = y(tick);
    svg.appendChild(svgEl('line', { x1: padL, x2: W - padR, y1: gy, y2: gy, class: 'grid-line' }));
    const label = svgEl('text', { x: padL - 8, y: gy + 3, class: 'axis-label', 'text-anchor': 'end' });
    label.textContent = pct(tick);
    svg.appendChild(label);
  }

  // The 50% line, drawn last of the reference marks so it reads as the one
  // that matters: above it Democrats are favoured, below it they are not.
  if (lo < 0.5 && hi > 0.5) {
    svg.appendChild(svgEl('line', {
      x1: padL, x2: W - padR, y1: y(0.5), y2: y(0.5), class: 'threshold-line',
    }));
    const evens = svgEl('text', { x: W - padR - 2, y: y(0.5) - 5, class: 'threshold-text', 'text-anchor': 'end' });
    evens.textContent = 'even odds';
    svg.appendChild(evens);
  }

  svg.appendChild(svgEl('polyline', {
    points: runs.map((r) => `${x(r.run_date)},${y(r.dem_control_prob)}`).join(' '),
    fill: 'none', stroke: 'var(--dem)', 'stroke-width': 2,
    'stroke-linejoin': 'round', 'stroke-linecap': 'round',
  }));

  for (const r of runs) {
    svg.appendChild(svgEl('circle', {
      cx: x(r.run_date), cy: y(r.dem_control_prob), r: 3, fill: 'var(--dem)',
    }));
  }

  const crosshair = svgEl('line', {
    y1: padT, y2: H - padB, class: 'crosshair', 'stroke-width': 1,
  });
  crosshair.style.display = 'none';
  svg.appendChild(crosshair);
  const marker = svgEl('circle', { r: 5.5, class: 'hover-dot' });
  marker.style.display = 'none';
  svg.appendChild(marker);

  const runSpan = [runs[0].run_date, runs[runs.length - 1].run_date];
  for (const iso of runSpan) {
    const t = svgEl('text', {
      x: x(iso), y: H - padB + 17, class: 'axis-label',
      'text-anchor': iso === runSpan[0] ? 'start' : 'end',
    });
    t.textContent = fmtDateAxis(iso, runSpan);
    svg.appendChild(t);
  }

  host.appendChild(svg);

  // --- hover ---------------------------------------------------------------
  const tip = chartTip(host);
  let activeIndex = -1;

  svg.addEventListener('pointermove', (e) => {
    const vx = toViewBoxX(svg, e.clientX);
    let best = 0, bestDistance = Infinity;
    runs.forEach((r, i) => {
      const distance = Math.abs(x(r.run_date) - vx);
      if (distance < bestDistance) { bestDistance = distance; best = i; }
    });

    if (activeIndex !== best) {
      activeIndex = best;
      const r = runs[best];
      const previous = best > 0 ? runs[best - 1] : null;
      const delta = previous ? r.dem_control_prob - previous.dem_control_prob : null;
      const deltaRow = delta === null
        ? ''
        : `<div class="tt-row"><span>Change on the run before</span><span>${
            (100 * delta >= 0 ? '+' : '') + (100 * delta).toFixed(1)} pts</span></div>`;
      tip.innerHTML =
        `<div class="tt-name">${esc(fmtDate(r.run_date))}</div>` +
        `<div class="tt-row"><span>Democratic control</span><span>${pct1(r.dem_control_prob)}</span></div>` +
        `<div class="tt-row"><span>Median seats</span><span>${r.dem_seats_median}</span></div>` +
        `<div class="tt-row"><span>Generic ballot</span><span>${margin(r.generic_ballot_dem_margin)}</span></div>` +
        `<div class="tt-row"><span>Polls in window</span><span>${r.n_race_polls}</span></div>` +
        deltaRow;

      const cx = x(r.run_date), cy = y(r.dem_control_prob);
      crosshair.setAttribute('x1', cx);
      crosshair.setAttribute('x2', cx);
      marker.setAttribute('cx', cx);
      marker.setAttribute('cy', cy);
    }
    crosshair.style.display = '';
    marker.style.display = '';
    tip.hidden = false;
    placeTip(host, tip, e.clientX, e.clientY);
  });

  svg.addEventListener('pointerleave', () => {
    tip.hidden = true;
    crosshair.style.display = 'none';
    marker.style.display = 'none';
    activeIndex = -1;
  });
}

/* ------------------------------------------------- trajectory (band + line) */

function renderTrajectory(host, trajectory, opts = {}) {
  host.textContent = '';
  if (!trajectory?.length) return;

  const W = chartWidth(host, opts.width ?? 560), H = opts.height ?? 200;
  const padL = 46, padR = 12, padT = 12, padB = 28;
  const svg = makeSvg(W, H);

  const [lo, hi] = paddedDomain(
    trajectory.map((d) => d.p05).concat(trajectory.map((d) => d.p95)),
    { include: 0, padFraction: 0.06, minPad: 0.5 },
  );

  const t0 = Date.parse(trajectory[0].date);
  const t1 = Date.parse(trajectory[trajectory.length - 1].date);
  const span = Math.max(1, t1 - t0);
  const x = (iso) => padL + ((Date.parse(iso) - t0) / span) * (W - padL - padR);
  const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);

  for (const tick of niceTicks(lo, hi, 4)) {
    const gy = y(tick);
    svg.appendChild(svgEl('line', {
      x1: padL, x2: W - padR, y1: gy, y2: gy,
      class: Math.abs(tick) < 1e-9 ? 'threshold-line' : 'grid-line',
    }));
    const label = svgEl('text', { x: padL - 7, y: gy + 3, class: 'axis-label', 'text-anchor': 'end' });
    label.textContent = margin(tick);
    svg.appendChild(label);
  }

  const top = trajectory.map((d) => `${x(d.date)},${y(d.p95)}`);
  const bottom = trajectory.slice().reverse().map((d) => `${x(d.date)},${y(d.p05)}`);
  svg.appendChild(svgEl('polygon', {
    points: top.concat(bottom).join(' '), fill: 'var(--dem-soft)', stroke: 'none',
  }));
  svg.appendChild(svgEl('polyline', {
    points: trajectory.map((d) => `${x(d.date)},${y(d.p50)}`).join(' '),
    fill: 'none', stroke: 'var(--dem-line)', 'stroke-width': 2, 'stroke-linejoin': 'round',
  }));

  const crosshair = svgEl('line', { y1: padT, y2: H - padB, class: 'crosshair' });
  crosshair.style.display = 'none';
  svg.appendChild(crosshair);
  const marker = svgEl('circle', { r: 5, class: 'hover-dot' });
  marker.style.display = 'none';
  svg.appendChild(marker);

  const endpoints = [trajectory[0].date, trajectory[trajectory.length - 1].date];
  for (const [iso, anchor] of [[endpoints[0], 'start'], [endpoints[1], 'end']]) {
    const t = svgEl('text', { x: x(iso), y: H - 7, class: 'axis-label', 'text-anchor': anchor });
    t.textContent = fmtDateAxis(iso, endpoints);
    svg.appendChild(t);
  }

  host.appendChild(svg);

  // --- hover ---------------------------------------------------------------
  const tip = chartTip(host);
  let activeIndex = -1;

  svg.addEventListener('pointermove', (e) => {
    const vx = toViewBoxX(svg, e.clientX);
    let best = 0, bestDistance = Infinity;
    trajectory.forEach((d, i) => {
      const distance = Math.abs(x(d.date) - vx);
      if (distance < bestDistance) { bestDistance = distance; best = i; }
    });

    if (activeIndex !== best) {
      activeIndex = best;
      const d = trajectory[best];
      tip.innerHTML =
        `<div class="tt-name">${esc(fmtDate(d.date))}</div>` +
        `<div class="tt-row"><span>Estimate</span>` +
        `<span class="${marginClass(d.p50)}">${margin(d.p50)}</span></div>` +
        `<div class="tt-row"><span>90% interval</span>` +
        `<span>${margin(d.p05)} to ${margin(d.p95)}</span></div>`;
      const cx = x(d.date), cy = y(d.p50);
      crosshair.setAttribute('x1', cx);
      crosshair.setAttribute('x2', cx);
      marker.setAttribute('cx', cx);
      marker.setAttribute('cy', cy);
    }
    crosshair.style.display = '';
    marker.style.display = '';
    tip.hidden = false;
    placeTip(host, tip, e.clientX, e.clientY);
  });

  svg.addEventListener('pointerleave', () => {
    tip.hidden = true;
    crosshair.style.display = 'none';
    marker.style.display = 'none';
    activeIndex = -1;
  });
}

/* ----------------------------------------------- candidate shares over time */

/** Both candidates' share of the two-party vote, with the polls behind it.
 *
 * The margin chart says the same thing mathematically, but "Ossoff 53, Collins
 * 47" is how people actually think about a race, and where the two lines cross
 * is instantly readable as the tie. Individual polls are drawn as dots so the
 * reader can see what the line is fitted to — and how much it is smoothing.
 *
 * Shares are of the two-party vote, so the pair always sums to 100. Undecideds
 * and minor candidates are excluded, which is what the model estimates.
 */
function renderCandidateChart(host, race, opts = {}) {
  host.textContent = '';
  const trajectory = race.trajectory ?? [];
  if (!trajectory.length) return;

  const W = chartWidth(host, opts.width ?? 560), H = opts.height ?? 210;
  const padL = 40, padR = 12, padT = 12, padB = 28;
  const svg = makeSvg(W, H);

  // margin (D minus R, in points) -> Democratic share of the two-party vote
  const demShare = (m) => 50 + m / 2;
  const polls = race.polls ?? [];

  const values = trajectory.flatMap((d) => [demShare(d.p05), demShare(d.p95)])
    .concat(polls.flatMap((p) => [demShare(p.margin), 100 - demShare(p.margin)]))
    .concat(trajectory.flatMap((d) => [100 - demShare(d.p05), 100 - demShare(d.p95)]));
  const [lo, hi] = paddedDomain(values, { include: 50, padFraction: 0.08, minPad: 1.5 });

  const t0 = Date.parse(trajectory[0].date);
  const t1 = Date.parse(trajectory[trajectory.length - 1].date);
  const span = Math.max(1, t1 - t0);
  const x = (iso) => padL + ((Date.parse(iso) - t0) / span) * (W - padL - padR);
  const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);

  for (const tick of niceTicks(lo, hi, 4)) {
    const gy = y(tick);
    svg.appendChild(svgEl('line', {
      x1: padL, x2: W - padR, y1: gy, y2: gy,
      class: Math.abs(tick - 50) < 1e-9 ? 'threshold-line' : 'grid-line',
    }));
    const label = svgEl('text', { x: padL - 7, y: gy + 3, class: 'axis-label', 'text-anchor': 'end' });
    label.textContent = `${tick.toFixed(0)}%`;
    svg.appendChild(label);
  }

  // Uncertainty band on the Democratic line. The Republican band is its mirror,
  // so drawing both would double the ink for no extra information.
  const top = trajectory.map((d) => `${x(d.date)},${y(demShare(d.p95))}`);
  const bottom = trajectory.slice().reverse().map((d) => `${x(d.date)},${y(demShare(d.p05))}`);
  svg.appendChild(svgEl('polygon', {
    points: top.concat(bottom).join(' '), fill: 'var(--dem-soft)', stroke: 'none',
  }));

  for (const [accessor, colour] of [
    [(d) => demShare(d.p50), 'var(--dem)'],
    [(d) => 100 - demShare(d.p50), 'var(--rep)'],
  ]) {
    svg.appendChild(svgEl('polyline', {
      points: trajectory.map((d) => `${x(d.date)},${y(accessor(d))}`).join(' '),
      fill: 'none', stroke: colour, 'stroke-width': 2, 'stroke-linejoin': 'round',
    }));
  }

  // The polls themselves, so the line is visibly answerable to something.
  for (const poll of polls) {
    if (Date.parse(poll.date) < t0) continue;
    for (const [value, colour] of [
      [demShare(poll.margin), 'var(--dem)'],
      [100 - demShare(poll.margin), 'var(--rep)'],
    ]) {
      const dot = svgEl('circle', {
        cx: x(poll.date), cy: y(value), r: 2.6,
        fill: colour, opacity: 0.5, class: 'poll-dot',
      });
      const title = svgEl('title');
      title.textContent = `${poll.pollster} (${poll.date}): ${margin(poll.margin)}`;
      dot.appendChild(title);
      svg.appendChild(dot);
    }
  }

  const crosshair = svgEl('line', { y1: padT, y2: H - padB, class: 'crosshair' });
  crosshair.style.display = 'none';
  svg.appendChild(crosshair);
  const demDot = svgEl('circle', { r: 4.5, class: 'hover-dot' });
  const repDot = svgEl('circle', { r: 4.5, class: 'hover-dot rep' });
  demDot.style.display = repDot.style.display = 'none';
  svg.appendChild(demDot);
  svg.appendChild(repDot);

  const endpoints = [trajectory[0].date, trajectory[trajectory.length - 1].date];
  for (const [iso, anchor] of [[endpoints[0], 'start'], [endpoints[1], 'end']]) {
    const t = svgEl('text', { x: x(iso), y: H - 7, class: 'axis-label', 'text-anchor': anchor });
    t.textContent = fmtDateAxis(iso, endpoints);
    svg.appendChild(t);
  }

  host.appendChild(svg);

  const names = race.candidates ?? {};
  const demName = names.dem || 'Democrat';
  const repName = names.rep || 'Republican';
  const tip = chartTip(host);
  let activeIndex = -1;

  svg.addEventListener('pointermove', (e) => {
    const vx = toViewBoxX(svg, e.clientX);
    let best = 0, bestDistance = Infinity;
    trajectory.forEach((d, i) => {
      const distance = Math.abs(x(d.date) - vx);
      if (distance < bestDistance) { bestDistance = distance; best = i; }
    });
    if (activeIndex !== best) {
      activeIndex = best;
      const d = trajectory[best];
      const dem = demShare(d.p50), rep = 100 - dem;
      tip.innerHTML =
        `<div class="tt-name">${esc(fmtDate(d.date))}</div>` +
        `<div class="tt-row"><span class="dem-text">${esc(demName)}</span>` +
        `<span class="dem-text">${dem.toFixed(1)}%</span></div>` +
        `<div class="tt-row"><span class="rep-text">${esc(repName)}</span>` +
        `<span class="rep-text">${rep.toFixed(1)}%</span></div>` +
        `<div class="tt-row"><span>Margin</span>` +
        `<span class="${marginClass(d.p50)}">${margin(d.p50)}</span></div>` +
        `<div class="tt-row"><span>90% interval</span>` +
        `<span>${margin(d.p05)} to ${margin(d.p95)}</span></div>`;
      const cx = x(d.date);
      crosshair.setAttribute('x1', cx);
      crosshair.setAttribute('x2', cx);
      demDot.setAttribute('cx', cx); demDot.setAttribute('cy', y(dem));
      repDot.setAttribute('cx', cx); repDot.setAttribute('cy', y(rep));
    }
    crosshair.style.display = demDot.style.display = repDot.style.display = '';
    tip.hidden = false;
    placeTip(host, tip, e.clientX, e.clientY);
  });

  svg.addEventListener('pointerleave', () => {
    tip.hidden = true;
    crosshair.style.display = demDot.style.display = repDot.style.display = 'none';
    activeIndex = -1;
  });
}
