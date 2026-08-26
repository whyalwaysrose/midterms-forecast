/* =========================================================================
   2026 House Forecast — the district cartogram
   Loaded after map.js; reuses its RATINGS/ratingFor and app.js helpers.

   The Senate gets a real map because states are the units and a state is a
   place. The House cannot: districts have equal population and wildly unequal
   area, so a shaded map of real district shapes hands almost all of its ink to
   the emptiest seats and shrinks the ones that decide the chamber to invisible
   specks. Every district here is therefore one square of the same size, which
   is the only arrangement in which the picture and the arithmetic agree.

   Layout comes from site/data/us-districts.json, computed from the same
   centroids the state map uses, so the two pictures can never drift apart.
   ========================================================================= */

/** Fill for a district square.
 *
 * Districts with no polling are the overwhelming majority -- around 90% of the
 * chamber -- and their probabilities come from the district's presidential
 * lean and the national environment rather than from anyone asking voters
 * there. They are drawn at reduced opacity so the eye can tell at a glance how
 * much of the map is measured and how much is inferred. It is the same
 * forecast either way; the difference is how much evidence sits behind it.
 */
function districtClass(race) {
  if (!race) return 'no-race';
  const rating = ratingFor(race.dem_win_prob);
  return rating.key + (race.poll_count ? ' polled' : ' unpolled');
}

let DISTRICT_BY_UNIT = {};

function renderCartogram(forecast, layout) {
  const host = $('house-map');
  if (!host || !layout || !layout.tiles) return;
  host.textContent = '';

  DISTRICT_BY_UNIT = {};
  for (const race of forecast.races || []) DISTRICT_BY_UNIT[race.unit] = race;

  const [vx, vy, vw, vh] = layout.view_box.split(/\s+/).map(Number);
  const cell = layout.cell;

  const svg = svgEl('svg', {
    viewBox: `${vx} ${vy} ${vw} ${vh}`,
    preserveAspectRatio: 'xMidYMid meet',
    role: 'img',
    'aria-label':
      'Cartogram of all 435 United States House districts, one equal square '
      + 'each, grouped by state and shaded by forecast rating',
  });

  const defs = svgEl('defs');
  const pattern = svgEl('pattern', {
    id: 'hatch-tossup-house', width: 5, height: 5,
    patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)',
  });
  pattern.appendChild(svgEl('rect', { width: 5, height: 5, fill: 'var(--tossup)' }));
  pattern.appendChild(svgEl('rect', {
    width: 1.8, height: 5, fill: 'var(--map-stroke)', opacity: 0.4,
  }));
  defs.appendChild(pattern);
  svg.appendChild(defs);

  const squares = svgEl('g', { class: 'district-squares' });
  const labels = svgEl('g', { class: 'district-labels' });
  svg.appendChild(squares);
  svg.appendChild(labels);

  // Track each state's block so its label can sit above it.
  const blocks = {};

  for (const tile of layout.tiles) {
    const race = DISTRICT_BY_UNIT[tile.district];
    const rect = svgEl('rect', {
      x: tile.x, y: tile.y, width: cell, height: cell, rx: 1.6,
      class: 'district ' + districtClass(race) + (race ? ' interactive' : ''),
      'data-unit': tile.district,
    });

    if (race) {
      const rating = ratingFor(race.dem_win_prob);
      rect.setAttribute('tabindex', '0');
      rect.setAttribute('role', 'button');
      rect.setAttribute(
        'aria-label',
        `${tile.district}: ${rating.label}, ${pct1(race.dem_win_prob)} `
        + `Democratic win probability`
      );
      const title = svgEl('title');
      title.textContent = `${tile.district} — ${rating.label}`;
      rect.appendChild(title);
    }
    squares.appendChild(rect);

    const b = blocks[tile.state] || (blocks[tile.state] = {
      minX: Infinity, maxX: -Infinity, minY: Infinity, n: 0,
    });
    b.minX = Math.min(b.minX, tile.x);
    b.maxX = Math.max(b.maxX, tile.x + cell);
    b.minY = Math.min(b.minY, tile.y);
    b.n += 1;
  }

  for (const [state, b] of Object.entries(blocks)) {
    const label = svgEl('text', {
      x: (b.minX + b.maxX) / 2, y: b.minY - 4, class: 'district-state-label',
    });
    label.textContent = state;
    labels.appendChild(label);
  }

  host.appendChild(svg);
  attachCartogramInteraction(host);
  renderCartogramLegend(forecast);
}

function renderCartogramLegend(forecast) {
  const el = $('house-map-legend');
  if (!el) return;

  const counts = {};
  let unpolled = 0;
  for (const race of forecast.races || []) {
    const key = ratingFor(race.dem_win_prob).key;
    counts[key] = (counts[key] || 0) + 1;
    if (!race.poll_count) unpolled += 1;
  }

  const items = RATINGS.map((r) =>
    `<span class="legend-item"><i class="legend-swatch ${r.key}"></i>${esc(r.label)} <b>${counts[r.key] || 0}</b></span>`
  );
  items.push(
    `<span class="legend-item legend-note"><i class="legend-swatch unpolled-swatch"></i>`
    + `Paler = no district polling <b>${unpolled}</b></span>`
  );
  el.innerHTML = items.join('');
}

function cartogramTooltipHtml(race) {
  const rating = ratingFor(race.dem_win_prob);
  const leadsDem = race.dem_win_prob >= 0.5;
  const shown = leadsDem ? race.dem_win_prob : 1 - race.dem_win_prob;
  const meta = `${race.incumbent_party}-held · ${INCUMBENCY_LABEL[race.incumbent_status] || race.incumbent_status}`;

  // How the forecast knows what it knows. In the House this is the single most
  // important line in the panel: nine districts in ten have never been polled,
  // and a reader is owed that fact before they read the number above it.
  const basis = race.poll_count
    ? `${race.poll_count} district poll${race.poll_count === 1 ? '' : 's'}`
    : 'No district polling — presidential lean and the national swing';

  const names = race.candidates ?? {};
  const matchup = (names.dem || names.rep)
    ? `<div class="tt-matchup">
         <span class="dem-text">${esc(surname(names.dem) || 'Democrat')}</span>
         <span class="tt-vs">v</span>
         <span class="rep-text">${esc(surname(names.rep) || 'Republican')}</span>
       </div>`
    : '';

  return `
    <div class="tt-name">${esc(race.name)}</div>
    <div class="tt-meta">${esc(meta)}</div>
    ${matchup}
    <div class="tt-bar"><div style="width:${(100 * race.dem_win_prob).toFixed(1)}%"></div></div>
    <div class="tt-row"><span>${esc(rating.label)}</span>
      <span class="${leadsDem ? 'dem-text' : 'rep-text'}">${leadsDem ? 'D' : 'R'} ${pct1(shown)}</span></div>
    <div class="tt-row"><span>Projected margin</span>
      <span class="${marginClass(race.margin.p50)}">${margin(race.margin.p50)}</span></div>
    <div class="tt-row"><span>90% interval</span>
      <span>${margin(race.margin.p05)} to ${margin(race.margin.p95)}</span></div>
    <div class="tt-hint">${esc(basis)} · click for detail</div>`;
}

function attachCartogramInteraction(host) {
  const tooltip = $('house-map-tooltip');
  if (!tooltip) return;
  let active = null;

  function place(clientX, clientY) {
    const hostBox = host.getBoundingClientRect();
    const ttBox = tooltip.getBoundingClientRect();
    let x = clientX - hostBox.left + 16;
    let y = clientY - hostBox.top + 16;
    if (x + ttBox.width > hostBox.width) x = clientX - hostBox.left - ttBox.width - 16;
    if (y + ttBox.height > hostBox.height) y = hostBox.height - ttBox.height;
    tooltip.style.left = `${Math.max(0, x)}px`;
    tooltip.style.top = `${Math.max(0, y)}px`;
  }

  function show(target, clientX, clientY) {
    const race = DISTRICT_BY_UNIT[target.dataset.unit];
    if (!race) return;
    if (active !== target) {
      if (active) active.classList.remove('is-active');
      active = target;
      active.classList.add('is-active');
      tooltip.innerHTML = cartogramTooltipHtml(race);
    }
    tooltip.hidden = false;
    place(clientX, clientY);
  }

  function hide() {
    tooltip.hidden = true;
    if (active) active.classList.remove('is-active');
    active = null;
  }

  host.addEventListener('pointermove', (e) => {
    const target = e.target.closest('[data-unit]');
    if (!target || !DISTRICT_BY_UNIT[target.dataset.unit]) { hide(); return; }
    show(target, e.clientX, e.clientY);
  });
  host.addEventListener('pointerleave', hide);

  host.addEventListener('click', (e) => {
    const target = e.target.closest('[data-unit]');
    const race = target && DISTRICT_BY_UNIT[target.dataset.unit];
    if (race) { hide(); openDrawer(race.id); }
  });

  // Keyboard parity. 435 focus stops is far too many to tab through, so only
  // the competitive districts are reachable -- the ones a reader could plausibly
  // want to inspect. The rest stay in the full race table below, which is the
  // real keyboard interface for the chamber.
  for (const target of host.querySelectorAll('[data-unit].interactive')) {
    const race = DISTRICT_BY_UNIT[target.dataset.unit];
    const competitive = race && race.dem_win_prob > 0.05 && race.dem_win_prob < 0.95;
    if (!competitive) { target.removeAttribute('tabindex'); continue; }

    target.addEventListener('focus', () => {
      const box = target.getBoundingClientRect();
      show(target, box.left + box.width / 2, box.top + box.height / 2);
    });
    target.addEventListener('blur', hide);
    target.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      e.preventDefault();
      hide();
      openDrawer(race.id);
    });
  }
}
