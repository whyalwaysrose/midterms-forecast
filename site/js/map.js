/* =========================================================================
   2026 Senate Forecast — the map
   Loaded after app.js; uses its formatting helpers and openDrawer().
   ========================================================================= */

/** Five-step diverging rating scale, keyed on Democratic win probability.
 *
 * Five steps, not seven. A seven-step ramp was built and measured first: near
 * the neutral midpoint its adjacent pairs fell to a perceptual separation of
 * 4-10, well under the floor of 15, meaning readers with ordinary colour
 * vision could not tell Lean from Likely. Five is what the colour space
 * actually supports at this surface. The precise probability is always one
 * hover away, so nothing is lost but false precision.
 */
const RATINGS = [
  { key: 'safe-r', label: 'Safe R',  max: 0.20 },
  { key: 'lean-r', label: 'Lean R',  max: 0.40 },
  { key: 'tossup', label: 'Toss-up', max: 0.60 },
  { key: 'lean-d', label: 'Lean D',  max: 0.80 },
  { key: 'safe-d', label: 'Safe D',  max: 1.01 },
];

function ratingFor(p) {
  return RATINGS.find((r) => p < r.max) || RATINGS[RATINGS.length - 1];
}

/** Bounding box of an SVG path built only from M/L/Z commands. */
function pathBounds(d) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const re = /(-?[\d.]+),(-?[\d.]+)/g;
  let m;
  while ((m = re.exec(d)) !== null) {
    const x = +m[1], y = +m[2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { minX, minY, maxX, maxY, w: maxX - minX, h: maxY - minY };
}

let RACE_BY_UNIT = {};

function renderMap(forecast, geo) {
  const host = $('us-map');
  if (!host || !geo || !geo.states) return;
  host.textContent = '';

  RACE_BY_UNIT = {};
  for (const race of forecast.races || []) RACE_BY_UNIT[race.unit] = race;

  const parts = geo.view_box.split(/\s+/).map(Number);
  const vx = parts[0], vy = parts[1], vw = parts[2], vh = parts[3];

  // How much of the map's furniture the available width can actually carry.
  //
  // The map's own aspect ratio fixes the relationship: an inline state label is
  // 11 units in a viewBox ~1100 wide, so it renders at 11 x (width / 1100). At
  // a 375px phone that is about 3px — and Rhode Island is 3x4px, far under any
  // usable tap target. Chips survive much further down, because they are drawn
  // at a fixed 60x22 units regardless of how small the state is.
  //
  //   >= 820px   inline labels are >= 8px: show everything
  //   >= 640px   labels too small, but chips stay legible and tappable
  //   <  640px   overview only; the race table below is the real interface
  const width = window.innerWidth;
  const showInlineLabels = width >= 820;
  const showChips = width >= 640;
  const compact = !showChips;

  // Reserve a column on the right only if chips are going into it.
  const CHIP_COLUMN = showChips ? 116 : 0;

  const svg = svgEl('svg', {
    viewBox: `${vx} ${vy} ${vw + CHIP_COLUMN} ${vh}`,
    preserveAspectRatio: 'xMidYMid meet',
    role: 'img',
    'aria-label': 'Map of the 2026 United States Senate races, shaded by forecast rating',
  });

  // Hatch for the toss-up category. It is the one rating sitting between the
  // two hues, so it is the one most at risk of being misread as either — a
  // texture means it never depends on colour alone.
  const defs = svgEl('defs');
  const pattern = svgEl('pattern', {
    id: 'hatch-tossup', width: 6, height: 6,
    patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)',
  });
  pattern.appendChild(svgEl('rect', { width: 6, height: 6, fill: 'var(--tossup)' }));
  pattern.appendChild(svgEl('rect', {
    width: 2.2, height: 6, fill: 'var(--map-stroke)', opacity: 0.4,
  }));
  defs.appendChild(pattern);
  svg.appendChild(defs);

  const shapes = svgEl('g');
  const overlays = svgEl('g');   // labels sit above every fill
  svg.appendChild(shapes);
  svg.appendChild(overlays);

  const needsChip = [];

  for (const code of Object.keys(geo.states)) {
    const d = geo.states[code];
    const race = RACE_BY_UNIT[code];
    const rating = race ? ratingFor(race.dem_win_prob) : null;

    const path = svgEl('path', {
      d,
      class: 'state ' + (rating ? rating.key : 'no-race') + (race ? ' interactive' : ''),
      'data-unit': code,
    });

    if (race) {
      path.setAttribute('tabindex', '0');
      path.setAttribute('role', 'button');
      path.setAttribute(
        'aria-label',
        `${race.name}: ${rating.label}, ${pct1(race.dem_win_prob)} Democratic win probability`
      );
      const title = svgEl('title');
      title.textContent = `${race.name} — ${rating.label}`;
      path.appendChild(title);
    }
    shapes.appendChild(path);

    if (!race) continue;

    const box = pathBounds(d);
    const centroid = (geo.centroids && geo.centroids[code])
      || [(box.minX + box.maxX) / 2, (box.minY + box.maxY) / 2];

    const fitsOwnLabel = box.w >= 34 && box.h >= 20;

    if (showInlineLabels && fitsOwnLabel) {
      const label = svgEl('text', { x: centroid[0], y: centroid[1], class: 'map-label' });
      label.textContent = code;
      overlays.appendChild(label);
    } else if (showChips && !fitsOwnLabel) {
      // Too small to hold its own label — give it a chip in the side column.
      needsChip.push({ code, centroid, rating, race });
    }
  }

  // Stack the chips north to south so their order still matches the geography
  // the leader lines point at.
  needsChip.sort((a, b) => a.centroid[1] - b.centroid[1]);
  const chipX = vx + vw + 16;
  const chipW = 60, chipH = 22, gap = 7;
  const stackH = needsChip.length * chipH + Math.max(0, needsChip.length - 1) * gap;
  let chipY = Math.max(vy + 12, vy + (vh - stackH) / 2);

  for (const entry of needsChip) {
    const cy = chipY + chipH / 2;
    overlays.appendChild(svgEl('path', {
      d: `M${entry.centroid[0]},${entry.centroid[1]}L${chipX - 7},${cy}`,
      class: 'map-leader',
    }));
    const chip = svgEl('rect', {
      x: chipX, y: chipY, width: chipW, height: chipH, rx: 4,
      class: `state map-chip-box interactive ${entry.rating.key}`,
      'data-unit': entry.code, tabindex: '0', role: 'button',
    });
    chip.setAttribute(
      'aria-label',
      `${entry.race.name}: ${entry.rating.label}, ${pct1(entry.race.dem_win_prob)} Democratic win probability`
    );
    const chipTitle = svgEl('title');
    chipTitle.textContent = `${entry.race.name} — ${entry.rating.label}`;
    chip.appendChild(chipTitle);
    overlays.appendChild(chip);

    const label = svgEl('text', { x: chipX + chipW / 2, y: cy, class: 'map-label' });
    label.textContent = entry.code;
    overlays.appendChild(label);

    chipY += chipH + gap;
  }

  const hint = $('map-hint');
  if (hint) hint.hidden = !compact;

  host.appendChild(svg);
  attachMapInteraction(host);
  renderMapLegend(forecast);
}

function renderMapLegend(forecast) {
  const counts = {};
  for (const race of forecast.races || []) {
    const key = ratingFor(race.dem_win_prob).key;
    counts[key] = (counts[key] || 0) + 1;
  }
  // Reads left to right in the same direction as the diverging scale.
  const items = RATINGS.map((r) =>
    `<span class="legend-item"><i class="legend-swatch ${r.key}"></i>${esc(r.label)} <b>${counts[r.key] || 0}</b></span>`
  );
  items.push('<span class="legend-item"><i class="legend-swatch no-race"></i>No race in 2026</span>');
  $('map-legend').innerHTML = items.join('');
}

function mapTooltipHtml(race) {
  const rating = ratingFor(race.dem_win_prob);
  const leadsDem = race.dem_win_prob >= 0.5;
  const shown = leadsDem ? race.dem_win_prob : 1 - race.dem_win_prob;
  const meta = `${race.incumbent_party}-held · ${INCUMBENCY_LABEL[race.incumbent_status] || race.incumbent_status}`
    + (race.special ? ' · special' : '');
  const polls = race.poll_count
    ? `${race.poll_count} poll${race.poll_count === 1 ? '' : 's'}`
    : 'No polls — carried by fundamentals';

  return `
    <div class="tt-name">${esc(race.name)}</div>
    <div class="tt-meta">${esc(meta)}</div>
    <div class="tt-bar"><div style="width:${(100 * race.dem_win_prob).toFixed(1)}%"></div></div>
    <div class="tt-row"><span>${esc(rating.label)}</span>
      <span class="${leadsDem ? 'dem-text' : 'rep-text'}">${leadsDem ? 'D' : 'R'} ${pct1(shown)}</span></div>
    <div class="tt-row"><span>Projected margin</span>
      <span class="${marginClass(race.margin.p50)}">${margin(race.margin.p50)}</span></div>
    <div class="tt-row"><span>90% interval</span>
      <span>${margin(race.margin.p05)} to ${margin(race.margin.p95)}</span></div>
    <div class="tt-row"><span>Tipping point</span><span>${pct1(race.tipping_point_prob)}</span></div>
    <div class="tt-hint">${esc(polls)} · click for detail</div>`;
}

function attachMapInteraction(host) {
  const tooltip = $('map-tooltip');
  let active = null;

  function place(clientX, clientY) {
    const hostBox = host.getBoundingClientRect();
    const ttBox = tooltip.getBoundingClientRect();
    let x = clientX - hostBox.left + 16;
    let y = clientY - hostBox.top + 16;
    // Flip rather than overflow when close to an edge.
    if (x + ttBox.width > hostBox.width) x = clientX - hostBox.left - ttBox.width - 16;
    if (y + ttBox.height > hostBox.height) y = hostBox.height - ttBox.height;
    tooltip.style.left = `${Math.max(0, x)}px`;
    tooltip.style.top = `${Math.max(0, y)}px`;
  }

  function show(target, clientX, clientY) {
    const race = RACE_BY_UNIT[target.dataset.unit];
    if (!race) return;
    if (active !== target) {
      if (active) active.classList.remove('is-active');
      active = target;
      active.classList.add('is-active');
      tooltip.innerHTML = mapTooltipHtml(race);
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
    if (!target || !RACE_BY_UNIT[target.dataset.unit]) { hide(); return; }
    show(target, e.clientX, e.clientY);
  });
  host.addEventListener('pointerleave', hide);

  host.addEventListener('click', (e) => {
    const target = e.target.closest('[data-unit]');
    const race = target && RACE_BY_UNIT[target.dataset.unit];
    if (race) { hide(); openDrawer(race.id); }
  });

  // Keyboard parity: focus shows the same detail, Enter/Space opens the race.
  //
  // Bound per element rather than delegated. Focus events raised on SVG
  // children do not reliably bubble to an HTML ancestor — verified here, where
  // a delegated `focusin` on the container never fired even though
  // document.activeElement was correctly the focused <path>. Keydown does
  // bubble, so that one stays delegated below.
  for (const target of host.querySelectorAll('[data-unit].interactive')) {
    target.addEventListener('focus', () => {
      const box = target.getBoundingClientRect();
      show(target, box.left + box.width / 2, box.top + box.height / 2);
    });
    target.addEventListener('blur', hide);
    target.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      const race = RACE_BY_UNIT[target.dataset.unit];
      if (!race) return;
      e.preventDefault();
      hide();
      openDrawer(race.id);
    });
  }
}
