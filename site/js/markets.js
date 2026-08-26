/* =========================================================================
   2026 Senate Forecast — prediction markets
   Shown beside the model, never fed into it.
   ========================================================================= */

/** Colour for an outcome, so a card reads at a glance rather than by label.
 *
 * Party colours where an outcome names a winner; a neutral ramp otherwise.
 * Seat counts are the awkward case: "49" means Democrats take the chamber and
 * "51" means they do not, and only the reader who remembers the 51-seat
 * threshold can tell. Colouring by which side wins says it without a caption.
 */
function outcomeColour(label, slug) {
  const text = label.toLowerCase();

  if (slug && slug.startsWith('republican-senate-seats')) {
    // Republican seats. Democrats need 51 of their own, so Republicans on 49
    // or fewer means a Democratic chamber.
    const seats = text.startsWith('≤') ? Number(text.slice(1)) : Number(text.replace(/\D/g, ''));
    if (Number.isFinite(seats)) return seats <= 49 ? 'var(--dem)' : 'var(--rep)';
    return 'var(--text-faint)';
  }
  // Split control: colour by who takes the SENATE, which is the half this site
  // forecasts, and tell the two apart with a dash rather than a third hue.
  // Both were grey before, which left two of the four lines identical.
  if (text.includes('senate') && text.includes('house')) {
    return text.startsWith('d senate') ? 'var(--dem)' : 'var(--rep)';
  }
  if (text.includes('democrat')) return 'var(--dem)';
  if (text.includes('republican')) return 'var(--rep)';
  return 'var(--text-faint)';
}

/** Dash pattern for split-control outcomes, empty for the rest.
 *
 * Carries the "one chamber each" idea without spending a colour on it: a
 * dashed red line is a Republican Senate alongside a Democratic House.
 */
function outcomeDash(label) {
  const text = label.toLowerCase();
  return text.includes('senate') && text.includes('house') ? '4 3' : '';
}

/** Our own number for a market, where one is comparable.
 *
 * Anything without an honest counterpart returns null and the card shows the
 * market alone rather than inventing a comparison.
 */
function modelCounterpart(slug, forecast) {
  const chamber = forecast.chamber_forecast;

  if (slug === 'which-party-will-win-the-senate-in-2026') {
    return {
      'Democratic Party': chamber.dem_control_prob,
      'Republican Party': 1 - chamber.dem_control_prob,
    };
  }

  if (slug.startsWith('republican-senate-seats')) {
    // Ours is over Democratic-caucus seats; theirs over Republican ones.
    const byRep = {};
    for (const [seats, p] of Object.entries(chamber.seat_distribution)) {
      const rep = 100 - Number(seats);
      const key = rep <= 47 ? '≤47' : String(rep);
      byRep[key] = (byRep[key] ?? 0) + p;
    }
    return byRep;
  }

  // Balance of Power needs a House forecast, which this model does not have.
  return null;
}

function marketCaveat(slug) {
  if (slug === 'balance-of-power-2026-midterms') {
    return 'This model forecasts only the Senate, so there is nothing to compare '
         + 'these against — the House half is not something it predicts.';
  }
  if (slug.startsWith('republican-senate-seats')) {
    return 'Blue bars are outcomes where Democrats take the chamber: they need 51 '
         + 'of their own, because the Vice President breaks ties.';
  }
  return null;
}

/** Width for a card chart, measured rather than assumed.
 *
 * Not chartWidth(): that has a hard 280px floor, and a market card's chart host
 * is 245px on a phone -- the page wrap and the card padding leave a 301px
 * track, so no card width can reach 280. The result was a viewBox wider than
 * its container, scaled down by the browser, rendering every label at 87% size.
 * These charts are simple enough to stay legible at 230.
 */
function marketChartWidth(host, preferred = 340) {
  const available = host.clientWidth || 0;
  if (!available) return preferred;
  return Math.round(Math.max(230, Math.min(preferred, available)));
}

/* ------------------------------------------------------ card: over time */

/** Lines of implied probability over time, one per outcome.
 *
 * This is the view that makes a market worth showing at all. A single day's
 * price is a number; thirteen months of them is the story — Democrats went from
 * 18% to 48% on the sweep over that period, which no snapshot can convey.
 */
function renderMarketTimeChart(host, event, opts = {}) {
  const series = event.outcomes.filter((o) => (o.history ?? []).length > 1);
  if (!series.length) return false;

  const W = marketChartWidth(host, opts.width ?? 340), H = opts.height ?? 150;
  const padL = 30, padR = 8, padT = 8, padB = 20;
  const svg = makeSvg(W, H);

  const dates = series[0].history.map((p) => Date.parse(p.date));
  const t0 = Math.min(...series.map((s) => Date.parse(s.history[0].date)));
  const t1 = Math.max(...series.map((s) => Date.parse(s.history[s.history.length - 1].date)));
  const span = Math.max(1, t1 - t0);
  const values = series.flatMap((s) => s.history.map((p) => p.p));
  const hi = Math.min(1, Math.max(...values) + 0.06);
  const lo = Math.max(0, Math.min(...values) - 0.06);

  const x = (iso) => padL + ((Date.parse(iso) - t0) / span) * (W - padL - padR);
  const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);

  for (const tick of niceTicks(lo * 100, hi * 100, 3)) {
    const gy = y(tick / 100);
    if (gy < padT || gy > H - padB) continue;
    svg.appendChild(svgEl('line', { x1: padL, x2: W - padR, y1: gy, y2: gy, class: 'grid-line' }));
    const label = svgEl('text', { x: padL - 6, y: gy + 3, class: 'axis-label', 'text-anchor': 'end' });
    label.textContent = `${tick.toFixed(0)}%`;
    svg.appendChild(label);
  }

  for (const outcome of series) {
    svg.appendChild(svgEl('polyline', {
      points: outcome.history.map((p) => `${x(p.date)},${y(p.p)}`).join(' '),
      fill: 'none',
      stroke: outcomeColour(outcome.label, event.slug),
      'stroke-width': 1.8,
      'stroke-linejoin': 'round',
      'stroke-dasharray': outcomeDash(outcome.label),
    }));
  }

  const ends = [series[0].history[0].date, series[0].history[series[0].history.length - 1].date];
  for (const [iso, anchor] of [[ends[0], 'start'], [ends[1], 'end']]) {
    const t = svgEl('text', { x: x(iso), y: H - 5, class: 'axis-label', 'text-anchor': anchor });
    t.textContent = fmtDateAxis(iso, ends);
    svg.appendChild(t);
  }

  host.appendChild(svg);

  const tip = chartTip(host);
  const crosshair = svgEl('line', { y1: padT, y2: H - padB, class: 'crosshair' });
  crosshair.style.display = 'none';
  svg.appendChild(crosshair);

  svg.addEventListener('pointermove', (e) => {
    const vx = toViewBoxX(svg, e.clientX);
    let best = 0, bestDistance = Infinity;
    dates.forEach((d, i) => {
      const distance = Math.abs(x(series[0].history[i].date) - vx);
      if (distance < bestDistance) { bestDistance = distance; best = i; }
    });
    const when = series[0].history[best].date;
    const rows = series.map((s) => {
      const point = s.history[best] ?? s.history[s.history.length - 1];
      return `<div class="tt-row"><span>${esc(s.label)}</span>`
           + `<span style="color:${outcomeColour(s.label, event.slug)}">${pct1(point.p)}</span></div>`;
    }).join('');
    tip.innerHTML = `<div class="tt-name">${esc(fmtDate(when))}</div>${rows}`;
    tip.hidden = false;
    const cx = x(when);
    crosshair.setAttribute('x1', cx);
    crosshair.setAttribute('x2', cx);
    crosshair.style.display = '';
    placeTip(host, tip, e.clientX, e.clientY);
  });
  svg.addEventListener('pointerleave', () => {
    tip.hidden = true;
    crosshair.style.display = 'none';
  });
  return true;
}

/* --------------------------------------------- card: a distribution */

/** Two distributions over the same outcome, drawn as paired bars.
 *
 * Replaces eleven rows of bar-plus-caption, which was unreadable and made the
 * card four times the height of its neighbours. Side by side, the disagreement
 * is the thing you see: the market spreads its mass across 48-52 seats where
 * this model piles it on 47 and below.
 */
function renderMarketDistribution(host, event, ours, opts = {}) {
  const bars = event.outcomes.filter((o) => o.probability >= 0.004);
  if (!bars.length) return false;

  const W = marketChartWidth(host, opts.width ?? 340), H = opts.height ?? 150;
  const padL = 30, padR = 8, padT = 10, padB = 34;
  const svg = makeSvg(W, H);

  const peak = Math.max(...bars.map((o) => Math.max(o.probability, ours?.[o.label] ?? 0)));
  const slot = (W - padL - padR) / bars.length;
  const y = (v) => padT + (1 - v / peak) * (H - padT - padB);

  for (const tick of niceTicks(0, peak * 100, 3)) {
    if (tick <= 0) continue;
    const gy = y(tick / 100);
    svg.appendChild(svgEl('line', { x1: padL, x2: W - padR, y1: gy, y2: gy, class: 'grid-line' }));
    const label = svgEl('text', { x: padL - 6, y: gy + 3, class: 'axis-label', 'text-anchor': 'end' });
    label.textContent = `${tick.toFixed(0)}%`;
    svg.appendChild(label);
  }

  bars.forEach((outcome, i) => {
    const left = padL + i * slot;
    const mine = ours?.[outcome.label];
    // Market bar solid, model bar hatched and narrower: the section is about
    // what the market says, so ours is the annotation, not the subject.
    const wide = mine === undefined ? slot * 0.68 : slot * 0.4;
    svg.appendChild(svgEl('rect', {
      x: left + slot * 0.08, y: y(outcome.probability),
      width: wide, height: Math.max(1, H - padB - y(outcome.probability)),
      fill: outcomeColour(outcome.label, event.slug), rx: 2,
    }));
    if (mine !== undefined) {
      svg.appendChild(svgEl('rect', {
        x: left + slot * 0.52, y: y(mine),
        width: slot * 0.4, height: Math.max(1, H - padB - y(mine)),
        fill: outcomeColour(outcome.label, event.slug), opacity: 0.35, rx: 2,
      }));
    }
    if (bars.length <= 12 || i % 2 === 0) {
      const label = svgEl('text', {
        x: left + slot / 2, y: H - padB + 12, class: 'axis-label', 'text-anchor': 'middle',
      });
      label.textContent = outcome.label;
      svg.appendChild(label);
    }
  });

  // Well clear of the tick labels: at padB 26 the two baselines were 10px
  // apart and the rows ran together.
  const caption = svgEl('text', {
    x: (padL + W - padR) / 2, y: H - 4, class: 'axis-label', 'text-anchor': 'middle',
  });
  caption.textContent = 'Republican seats';
  svg.appendChild(caption);

  host.appendChild(svg);
  return true;
}

/* ----------------------------------------------------------- the section */

function renderMarkets(forecast) {
  const section = $('markets');
  const markets = forecast.markets;
  if (!markets || !(markets.events ?? []).length) {
    section.hidden = true;
    return;
  }

  const track = $('markets-track');
  const dots = $('markets-dots');
  track.textContent = '';
  dots.textContent = '';

  if (markets.fetched_at) {
    $('markets-fetched').textContent =
      ` Odds as of ${fmtDate(markets.fetched_at.slice(0, 10))}.`;
  }
  if (markets.source_url) $('markets-source-link').href = markets.source_url;

  // Reveal before drawing: a hidden container reports clientWidth 0, which
  // sends chartWidth() back to its unscaled fallback and halves every chart.
  section.hidden = false;

  const pending = [];

  markets.events.forEach((event, index) => {
    const ours = modelCounterpart(event.slug, forecast);
    const caveat = marketCaveat(event.slug);
    const overTime = event.outcomes.some((o) => (o.history ?? []).length > 1);

    const card = document.createElement('article');
    card.className = 'market-card';
    card.setAttribute('role', 'group');
    card.setAttribute('aria-roledescription', 'slide');
    card.setAttribute('aria-label', `${index + 1} of ${markets.events.length}: ${event.title}`);

    const volume = event.volume >= 1e6
      ? `$${(event.volume / 1e6).toFixed(1)}M`
      : `$${Math.round(event.volume / 1000)}k`;

    // A compact key, rather than a bar and a caption per outcome.
    const legend = event.outcomes
      .filter((o) => o.probability >= 0.004)
      .slice(0, 5)
      .map((o) => {
        const mine = ours?.[o.label];
        const compare = mine === undefined ? ''
          : `<span class="market-mine">${pct1(mine)}</span>`;
        return `<li>
          <i style="background:${outcomeColour(o.label, event.slug)}${
            outcomeDash(o.label) ? ';opacity:.55' : ''}"></i>
          <span class="market-key-label">${esc(o.label)}</span>
          <span class="market-key-prob">${pct1(o.probability)}</span>
          ${compare}
        </li>`;
      }).join('');

    card.innerHTML =
      `<header class="market-card-head">
         <h3>${esc(event.title)}</h3>
         <span class="market-volume">${volume}</span>
       </header>
       <div class="market-chart chart"></div>
       <ul class="market-key${ours ? ' has-model' : ''}">
         ${ours ? '<li class="market-key-head"><i></i><span></span>'
                + '<span class="market-key-prob">market</span>'
                + '<span class="market-mine">model</span></li>' : ''}
         ${legend}
       </ul>
       ${caveat ? `<p class="market-caveat">${esc(caveat)}</p>` : ''}`;
    track.appendChild(card);

    // Charts are drawn in a second pass, below. Measuring one now would size
    // it against a track holding only the cards appended so far, and the
    // flex row re-splits its width every time another card lands -- the first
    // card measured 280px in a host that ended up 320.
    pending.push({ card, event, ours, overTime });

    const dot = document.createElement('button');
    dot.className = 'carousel-dot';
    dot.type = 'button';
    dot.setAttribute('role', 'tab');
    dot.setAttribute('aria-label', event.title);
    dot.addEventListener('click', () => scrollToCard(index));
    dots.appendChild(dot);
  });

  // Second pass: every card is in the DOM, so the flex row has settled and
  // each host reports the width it will keep.
  for (const { card, event, ours, overTime } of pending) {
    const host = card.querySelector('.market-chart');
    const drew = overTime
      ? renderMarketTimeChart(host, event)
      : renderMarketDistribution(host, event, ours);
    if (!drew) host.remove();
  }

  attachCarousel(track, dots);
}

/** Which card is nearest the middle of the track.
 *
 * Measured with getBoundingClientRect rather than offsetLeft. offsetLeft is
 * relative to the nearest *positioned* ancestor, which the track is not, so
 * mixed with scrollLeft it made every card look like card zero however far the
 * row had been scrolled.
 */
function activeCardIndex(track) {
  const box = track.getBoundingClientRect();
  const centre = box.left + box.width / 2;
  let best = 0, bestDistance = Infinity;
  [...track.children].forEach((card, i) => {
    const rect = card.getBoundingClientRect();
    const distance = Math.abs(rect.left + rect.width / 2 - centre);
    if (distance < bestDistance) { bestDistance = distance; best = i; }
  });
  return best;
}

function scrollToCard(index) {
  const track = $('markets-track');
  const card = track.children[index];
  if (!card) return;
  const box = track.getBoundingClientRect();
  const rect = card.getBoundingClientRect();
  const delta = (rect.left + rect.width / 2) - (box.left + box.width / 2);
  const target = track.scrollLeft + delta;
  const startedAt = track.scrollLeft;

  track.scrollTo({ left: target, behavior: 'smooth' });

  // Some engines refuse a smooth programmatic scroll on a scroll-snap
  // container and silently do nothing -- measured here: behavior:'auto' moved
  // the track 252px, behavior:'smooth' left it at 0 and stayed there. Arrows
  // that quietly do nothing are worse than a jump, so check and fall back.
  window.setTimeout(() => {
    if (Math.abs(track.scrollLeft - startedAt) < 2) track.scrollLeft = target;
  }, 120);
}

function attachCarousel(track, dots) {
  const sync = () => {
    // On a wide screen the cards all fit and there is nothing to scroll, so
    // arrows and dots would imply hidden content. The slack is a fraction of a
    // card rather than a pixel count: padding leaves ~18px at 1280px, and a
    // fixed number would be wrong at another card size.
    const card = track.firstElementChild;
    const slack = card ? card.offsetWidth * 0.25 : 24;
    const overflows = track.scrollWidth > track.clientWidth + slack;
    $('markets-prev').hidden = !overflows;
    $('markets-next').hidden = !overflows;
    dots.hidden = !overflows;
    if (!overflows) return;

    const active = activeCardIndex(track);
    [...dots.children].forEach((dot, i) => {
      dot.classList.toggle('is-active', i === active);
      dot.setAttribute('aria-selected', i === active ? 'true' : 'false');
    });
    $('markets-prev').disabled = track.scrollLeft <= 2;
    $('markets-next').disabled =
      track.scrollLeft >= track.scrollWidth - track.clientWidth - 2;
  };

  if (window.ResizeObserver) new ResizeObserver(sync).observe(track);
  else window.addEventListener('resize', sync);

  track.addEventListener('scroll', () => {
    window.clearTimeout(track._syncTimer);
    track._syncTimer = window.setTimeout(sync, 60);
  });
  $('markets-prev').addEventListener('click', () => scrollToCard(activeCardIndex(track) - 1));
  $('markets-next').addEventListener('click', () => scrollToCard(activeCardIndex(track) + 1));

  track.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    scrollToCard(activeCardIndex(track) + (e.key === 'ArrowRight' ? 1 : -1));
  });

  sync();
}
