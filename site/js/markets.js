/* =========================================================================
   2026 Senate Forecast — prediction markets
   Shown beside the model, never fed into it.
   ========================================================================= */

/** Our own number for a market, where one is comparable.
 *
 * Only three of these exist, and each is a judgement about what the market is
 * actually asking. Anything without an honest counterpart returns null and the
 * card shows the market alone rather than inventing a comparison.
 */
function modelCounterpart(slug, forecast) {
  const chamber = forecast.chamber_forecast;
  const demControl = chamber.dem_control_prob;

  if (slug === 'which-party-will-win-the-senate-in-2026') {
    return {
      'Democratic Party': demControl,
      'Republican Party': 1 - demControl,
    };
  }

  if (slug.startsWith('republican-senate-seats')) {
    // Our distribution is over Democratic-caucus seats; theirs is over
    // Republican ones. 100 - n, then bucketed the way they bucket it.
    const byRep = {};
    for (const [seats, p] of Object.entries(chamber.seat_distribution)) {
      const rep = 100 - Number(seats);
      const key = rep <= 47 ? '≤47' : String(rep);
      byRep[key] = (byRep[key] ?? 0) + p;
    }
    return byRep;
  }

  // Balance of Power needs a House forecast, which this model does not have.
  // Half of each outcome is genuinely unknown to us, so we say nothing rather
  // than implying the Senate half is the whole answer.
  return null;
}

/** A short, honest note about what the reader is comparing. */
function marketCaveat(slug) {
  if (slug === 'balance-of-power-2026-midterms') {
    return 'This model forecasts only the Senate, so there is nothing to compare '
         + 'these against — the House half is not something it predicts.';
  }
  if (slug.startsWith('republican-senate-seats')) {
    return 'Democrats need 51 seats for control, since the vice-president breaks '
         + 'ties — so the first three bars are the ones where they take the chamber.';
  }
  return null;
}

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
    $('markets-fetched').textContent = ` Prices from ${fmtDate(markets.fetched_at.slice(0, 10))}.`;
  }
  if (markets.source_url) $('markets-source-link').href = markets.source_url;

  markets.events.forEach((event, index) => {
    const ours = modelCounterpart(event.slug, forecast);
    const caveat = marketCaveat(event.slug);

    const card = document.createElement('article');
    card.className = 'market-card';
    card.id = `market-card-${index}`;
    card.setAttribute('role', 'group');
    card.setAttribute('aria-roledescription', 'slide');
    card.setAttribute('aria-label', `${index + 1} of ${markets.events.length}: ${event.title}`);

    const volume = event.volume >= 1e6
      ? `$${(event.volume / 1e6).toFixed(1)}M traded`
      : `$${Math.round(event.volume / 1000)}k traded`;

    const rows = event.outcomes.map((o) => {
      const mine = ours ? ours[o.label] : undefined;
      const width = Math.max(0.6, o.probability * 100);
      const compare = mine === undefined ? '' :
        `<div class="market-ours">
           <span class="market-ours-label">this model</span>
           <span class="${mine >= o.probability ? 'up' : 'down'}">${pct1(mine)}</span>
         </div>`;
      return `<li class="market-row">
        <div class="market-row-head">
          <span class="market-label">${esc(o.label)}</span>
          <span class="market-prob">${pct1(o.probability)}</span>
        </div>
        <div class="market-bar"><div style="width:${width}%"></div></div>
        ${compare}
      </li>`;
    }).join('');

    card.innerHTML =
      `<header class="market-card-head">
         <h3>${esc(event.title)}</h3>
         <span class="market-volume">${volume}</span>
       </header>
       <ul class="market-rows">${rows}</ul>
       ${caveat ? `<p class="market-caveat">${esc(caveat)}</p>` : ''}`;
    track.appendChild(card);

    const dot = document.createElement('button');
    dot.className = 'carousel-dot';
    dot.type = 'button';
    dot.setAttribute('role', 'tab');
    dot.setAttribute('aria-label', event.title);
    dot.addEventListener('click', () => scrollToCard(index));
    dots.appendChild(dot);
  });

  section.hidden = false;
  attachCarousel(track, dots);
}

/** Which card is nearest the middle of the track.
 *
 * Measured with getBoundingClientRect rather than offsetLeft. offsetLeft is
 * relative to the nearest *positioned* ancestor, which the track is not, so it
 * returned coordinates from somewhere further up the tree -- mixed with
 * scrollLeft it made every card look like card zero however far the row had
 * been scrolled. Rects are all in one coordinate system, so there is nothing
 * to get wrong.
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
  // Move by the difference between where the card is and where it should be,
  // rather than computing an absolute position. scrollTo rather than
  // scrollIntoView: the latter scrolls the whole page to bring the card into
  // view, yanking the reader away from the map.
  const delta = (rect.left + rect.width / 2) - (box.left + box.width / 2);
  const target = track.scrollLeft + delta;
  const startedAt = track.scrollLeft;

  track.scrollTo({ left: target, behavior: 'smooth' });

  // Some engines refuse a smooth programmatic scroll on a scroll-snap
  // container and silently do nothing at all -- measured here: behavior:'auto'
  // moved the track 252px, behavior:'smooth' left it at 0 and stayed there.
  // Arrows and arrow keys that quietly do nothing are far worse than a jump,
  // so check whether it actually started and fall back if not. A real smooth
  // scroll has visibly progressed well inside this window.
  window.setTimeout(() => {
    if (Math.abs(track.scrollLeft - startedAt) < 2) track.scrollLeft = target;
  }, 120);
}

function attachCarousel(track, dots) {
  const sync = () => {
    // On a wide screen all three cards fit and there is nothing to scroll, so
    // arrows and dots would be furniture implying content that is not hidden.
    //
    // The slack is a fraction of a card rather than a pixel count: padding and
    // sub-pixel layout leave scrollWidth ~18px above clientWidth at 1280px even
    // with everything visible, and a fixed threshold that swallowed that would
    // be wrong at a different card size. A quarter of a card is the point at
    // which something is actually hidden from the reader.
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
    // A control that cannot do anything should say so rather than sitting
    // there looking live.
    $('markets-prev').disabled = track.scrollLeft <= 2;
    $('markets-next').disabled =
      track.scrollLeft >= track.scrollWidth - track.clientWidth - 2;
  };

  // Whether it overflows depends on the viewport, so it has to be rechecked.
  if (window.ResizeObserver) new ResizeObserver(sync).observe(track);
  else window.addEventListener('resize', sync);

  track.addEventListener('scroll', () => {
    window.clearTimeout(track._syncTimer);
    track._syncTimer = window.setTimeout(sync, 60);
  });
  $('markets-prev').addEventListener('click', () => scrollToCard(activeCardIndex(track) - 1));
  $('markets-next').addEventListener('click', () => scrollToCard(activeCardIndex(track) + 1));

  // Keyboard parity: the track is focusable, so arrows should move it.
  track.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    scrollToCard(activeCardIndex(track) + (e.key === 'ArrowRight' ? 1 : -1));
  });

  sync();
}
