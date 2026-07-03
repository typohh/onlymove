// pwa/src/handlers/tournament-page.js
//
// GET /tournament — a standalone, directly-linkable, server-rendered landing page
// for the onlymove Go tournament (Finnish audience). Static content (no D1), styled
// to match onlymove's gold-on-dark goban identity. Registered before the SPA
// fallback in src/index.js. Links out to /leaderboard and the app front page (/).

// Design: dark "goban at dusk", warm gold (#CEB053) accent. The hero is a real
// corner tsumego with the single key point glowing gold — literally "the only move",
// the app's namesake. Serif display for gravitas, monospace for the measured data
// (dates, prizes, ranks), sans for prose. Go stones and a hoshi star-point are the
// recurring structural marks. Content is Finnish; the tournament is for Finnish
// players. Copy is intentionally plain and instructional.

export async function handleTournament() {
    return new Response(renderTournamentPage(), {
        status: 200,
        headers: {
            'Content-Type': 'text/html; charset=utf-8',
            'Cache-Control': 'public, max-age=60',
        },
    });
}

function renderTournamentPage() {
    return `<!doctype html>
<html lang="fi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#16130d">
<title>onlymove-turnaus · 13.–19. heinäkuuta 2026</title>
<meta name="description" content="Viikon mittainen tsumego-turnaus 13.–19.7.2026. Ratko go-tehtäviä, ja paras arvioitu vahvuutesi ratkaisee. Ilmainen, suomalaisille pelaajille — hyvä tapa hioa taitoja ennen Euroopan Go-kongressia.">
<meta property="og:type" content="website">
<meta property="og:title" content="onlymove-turnaus · 13.–19. heinäkuuta 2026">
<meta property="og:description" content="Viikon mittainen tsumego-turnaus. Paras arvioitu vahvuutesi ratkaisee. Ilmainen, suomalaisille pelaajille.">
<meta property="og:url" content="https://onlymove.net/tournament">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='13' fill='%23CEB053'/%3E%3C/svg%3E">
<style>
  :root {
    --bg: #16130d;          /* deep warm goban-shadow */
    --bg-lift: #1d1810;
    --panel: #221d13;       /* wood surface under stones */
    --panel-line: #2f2717;
    --gold: #CEB053;
    --gold-lite: #F0E7A7;
    --gold-deep: #9c7d38;
    --ink: #ece4d1;         /* primary text, warm ivory */
    --mist: #b0a68f;        /* secondary text */
    --faint: #948b74;       /* tertiary / captions (kept >= AA on the dark bg) */
    --line: rgba(206,176,83,.14);
    --line-soft: rgba(206,176,83,.08);
    --serif: Georgia, 'Iowan Old Style', 'Times New Roman', ui-serif, serif;
    --sans: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    --mono: ui-monospace, 'SF Mono', 'Cascadia Code', Menlo, Consolas, monospace;
    color-scheme: dark;
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body {
    margin: 0;
    font-family: var(--sans);
    color: var(--ink);
    background:
      radial-gradient(120% 80% at 50% -10%, rgba(206,176,83,.10), transparent 60%),
      radial-gradient(80% 50% at 85% 8%, rgba(240,231,167,.05), transparent 55%),
      var(--bg);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 720px; margin-inline: auto; padding: 0 1.25rem; }

  /* faint goban grid, ambient — sits behind everything, very low contrast */
  .grid-bg {
    position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background-image:
      linear-gradient(var(--line-soft) 1px, transparent 1px),
      linear-gradient(90deg, var(--line-soft) 1px, transparent 1px);
    background-size: 44px 44px;
    -webkit-mask-image: radial-gradient(120% 90% at 50% 0%, #000, transparent 75%);
            mask-image: radial-gradient(120% 90% at 50% 0%, #000, transparent 75%);
    opacity: .6;
  }

  .eyebrow {
    font-family: var(--mono);
    font-size: .72rem; letter-spacing: .28em; text-transform: uppercase;
    color: var(--gold); opacity: .85; margin: 0;
  }

  /* ---- hero ---- */
  header.hero { text-align: center; padding: 3.5rem 0 2.5rem; }
  .board-art { width: min(300px, 74vw); height: auto; margin: 1.6rem auto .4rem; display: block; }
  h1 {
    font-family: var(--serif); font-weight: 600;
    font-size: clamp(2.6rem, 9vw, 4rem); line-height: 1.02;
    letter-spacing: -.01em; margin: .3rem 0 .1rem;
    background: linear-gradient(180deg, var(--gold-lite), var(--gold) 55%, var(--gold-deep));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .dates {
    font-family: var(--mono); font-size: 1.02rem; letter-spacing: .04em;
    color: var(--ink); margin: .5rem 0 0;
  }
  .tagline { color: var(--mist); font-size: 1.06rem; max-width: 34rem; margin: 1rem auto 0; }
  .cta-row { display: flex; gap: .7rem; justify-content: center; flex-wrap: wrap; margin-top: 1.9rem; }
  .btn {
    font: inherit; font-weight: 600; text-decoration: none;
    padding: .7rem 1.3rem; border-radius: 999px; display: inline-flex; align-items: center; gap: .4rem;
    transition: transform .12s ease, background-color .15s ease, border-color .15s ease;
  }
  .btn .arrow { transition: transform .15s ease; }
  .btn:hover .arrow { transform: translateX(3px); }
  .btn-primary { background: var(--gold); color: #17130a; border: 1px solid var(--gold); }
  .btn-primary:hover { background: var(--gold-lite); border-color: var(--gold-lite); }
  .btn-ghost { background: transparent; color: var(--ink); border: 1px solid rgba(206,176,83,.35); }
  .btn-ghost:hover { border-color: var(--gold); color: var(--gold-lite); }
  .btn:active { transform: translateY(1px); }

  /* ---- hoshi divider (star point), not a number ---- */
  .hoshi { display: flex; align-items: center; gap: 1rem; margin: 3.2rem 0; }
  .hoshi::before, .hoshi::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, transparent, var(--line), transparent); }
  .hoshi span { width: 8px; height: 8px; border-radius: 50%; background: var(--gold); box-shadow: 0 0 0 4px rgba(206,176,83,.12); }

  section { margin: 0; }
  h2 {
    font-family: var(--serif); font-weight: 600; font-size: 1.7rem;
    margin: 0 0 .4rem; color: var(--ink); letter-spacing: -.01em;
  }
  .lead { color: var(--mist); margin: 0 0 1.4rem; }

  /* ---- prize cards ---- */
  .prizes { display: grid; grid-template-columns: 1fr 1fr; gap: .9rem; }
  .prize {
    border: 1px solid var(--panel-line); border-radius: 16px; padding: 1.25rem 1.15rem;
    background: linear-gradient(180deg, rgba(206,176,83,.04), transparent 70%), var(--bg-lift);
  }
  .prize .cat { font-family: var(--serif); font-size: 1.22rem; color: var(--gold-lite); margin: 0 0 .15rem; }
  .prize .cat-desc { color: var(--mist); font-size: .9rem; margin: 0 0 1rem; min-height: 2.4em; }
  .place { display: flex; align-items: center; gap: .6rem; padding: .5rem 0; border-top: 1px solid var(--line-soft); }
  .place:first-of-type { border-top: 0; }
  .place .medal { flex: none; width: 18px; height: 18px; border-radius: 50%; }
  .medal.first { background: radial-gradient(circle at 35% 30%, var(--gold-lite), var(--gold) 55%, var(--gold-deep)); box-shadow: 0 1px 3px rgba(0,0,0,.5); }
  .medal.second { background: transparent; border: 2px solid var(--gold-deep); }
  .place .lbl { color: var(--mist); font-size: .92rem; flex: 1; }
  .place .amt { font-family: var(--mono); font-weight: 600; color: var(--ink); font-size: 1.05rem; }
  .prize-note { color: var(--faint); font-size: .85rem; margin: 1rem 0 0; text-align: center; }

  /* ---- steps ---- */
  ol.steps { list-style: none; counter-reset: step; margin: 0; padding: 0; }
  ol.steps li { display: grid; grid-template-columns: auto 1fr; gap: .95rem; padding: .9rem 0; border-top: 1px solid var(--line-soft); }
  ol.steps li:first-child { border-top: 0; }
  ol.steps .stone {
    counter-increment: step; flex: none;
    width: 34px; height: 34px; border-radius: 50%;
    display: grid; place-items: center;
    font-family: var(--mono); font-size: .95rem; font-weight: 600; color: #17130a;
    background: radial-gradient(circle at 34% 30%, var(--gold-lite), var(--gold) 60%, var(--gold-deep));
    box-shadow: 0 2px 4px rgba(0,0,0,.45);
  }
  ol.steps .stone::before { content: counter(step); }
  ol.steps .s-title { font-weight: 600; color: var(--ink); }
  ol.steps .s-body { color: var(--mist); font-size: .95rem; margin-top: .1rem; }
  .col-note {
    margin: 1.3rem 0 0; padding: .85rem 1rem; border-left: 2px solid var(--gold-deep);
    background: rgba(206,176,83,.05); border-radius: 0 10px 10px 0;
    color: var(--mist); font-size: .92rem;
  }
  .col-note code { font-family: var(--mono); color: var(--gold-lite); font-size: .88em; }

  /* ---- scoring note ---- */
  .note-body { color: var(--mist); }
  .note-body strong { color: var(--ink); font-weight: 600; }

  /* ---- EGC band ---- */
  .egc {
    display: flex; gap: 1rem; align-items: center;
    border: 1px solid var(--panel-line); border-radius: 16px; padding: 1.15rem 1.25rem;
    background: linear-gradient(100deg, rgba(206,176,83,.08), transparent 70%), var(--bg-lift);
  }
  .egc .egc-mark { flex: none; width: 40px; height: 40px; }
  .egc .egc-k { font-family: var(--mono); font-size: .7rem; letter-spacing: .18em; text-transform: uppercase; color: var(--gold); opacity: .8; }
  .egc p { margin: .15rem 0 0; color: var(--mist); font-size: .95rem; }
  .egc strong { color: var(--ink); }

  /* ---- footer ---- */
  footer { margin: 3.4rem 0 3rem; padding-top: 1.6rem; border-top: 1px solid var(--line-soft); color: var(--faint); font-size: .88rem; }
  footer .org { color: var(--mist); }
  footer .org b { color: var(--ink); font-weight: 600; }
  footer .flinks { margin-top: .8rem; display: flex; gap: 1.1rem; flex-wrap: wrap; }
  footer a { color: var(--gold); text-decoration: none; }
  footer a:hover { color: var(--gold-lite); text-decoration: underline; }
  footer .fine { margin-top: 1rem; color: var(--faint); }

  a:focus-visible, .btn:focus-visible { outline: 2px solid var(--gold-lite); outline-offset: 3px; border-radius: 6px; }

  @media (max-width: 460px) {
    .prizes { grid-template-columns: 1fr; }
    header.hero { padding-top: 2.5rem; }
    .prize .cat-desc { min-height: 0; }
  }
  @media (prefers-reduced-motion: reduce) {
    .key-glow { animation: none !important; }
    .btn, .btn .arrow { transition: none; }
  }
  /* the "only move" gold key point breathes — the single deliberate motion */
  @keyframes keypulse { 0%,100% { opacity: .45; } 50% { opacity: 1; } }
  .key-glow { animation: keypulse 2.8s ease-in-out infinite; }
</style>
</head>
<body>
<div class="grid-bg" aria-hidden="true"></div>
<div class="wrap">

  <header class="hero">
    <p class="eyebrow">onlymove · go-turnaus</p>

    <svg class="board-art" viewBox="0 0 300 260" role="img" aria-label="Go-lauta, jossa yksi siirto hehkuu kultaisena — ainoa siirto.">
      <defs>
        <radialGradient id="wood" cx="42%" cy="30%" r="85%">
          <stop offset="0%" stop-color="#2c2517"/>
          <stop offset="100%" stop-color="#1a160e"/>
        </radialGradient>
        <radialGradient id="black" cx="36%" cy="30%" r="75%">
          <stop offset="0%" stop-color="#3a352b"/>
          <stop offset="55%" stop-color="#12100b"/>
          <stop offset="100%" stop-color="#050403"/>
        </radialGradient>
        <radialGradient id="white" cx="36%" cy="30%" r="80%">
          <stop offset="0%" stop-color="#fbf7ea"/>
          <stop offset="70%" stop-color="#e7ddc6"/>
          <stop offset="100%" stop-color="#c3b795"/>
        </radialGradient>
        <radialGradient id="gold" cx="36%" cy="30%" r="75%">
          <stop offset="0%" stop-color="#F7EDBD"/>
          <stop offset="55%" stop-color="#CEB053"/>
          <stop offset="100%" stop-color="#8f6f30"/>
        </radialGradient>
        <filter id="shadow" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="#000" flood-opacity="0.55"/>
        </filter>
        <filter id="glow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>

      <rect x="14" y="6" width="272" height="248" rx="12" fill="url(#wood)" stroke="#33291899" stroke-width="1"/>

      <!-- grid: corner crop. verticals x=40..280 step 40, horizontals y=20..220 step 40 -->
      <g stroke="#CEB053" stroke-opacity="0.22" stroke-width="1.4" stroke-linecap="round">
        <line x1="40" y1="20" x2="40" y2="220"/>
        <line x1="80" y1="20" x2="80" y2="220"/>
        <line x1="120" y1="20" x2="120" y2="220"/>
        <line x1="160" y1="20" x2="160" y2="220"/>
        <line x1="200" y1="20" x2="200" y2="220"/>
        <line x1="240" y1="20" x2="240" y2="220"/>
        <line x1="280" y1="20" x2="280" y2="220"/>
        <line x1="40" y1="20"  x2="280" y2="20"/>
        <line x1="40" y1="60"  x2="280" y2="60"/>
        <line x1="40" y1="100" x2="280" y2="100"/>
        <line x1="40" y1="140" x2="280" y2="140"/>
        <line x1="40" y1="180" x2="280" y2="180"/>
        <line x1="40" y1="220" x2="280" y2="220"/>
      </g>
      <!-- hoshi (4-4 star point) -->
      <circle cx="160" cy="100" r="3" fill="#CEB053" fill-opacity="0.55"/>

      <!-- stones (corner shape) -->
      <g filter="url(#shadow)">
        <!-- white group in the corner -->
        <circle cx="40"  cy="220" r="17" fill="url(#white)"/>
        <circle cx="80"  cy="220" r="17" fill="url(#white)"/>
        <circle cx="120" cy="220" r="17" fill="url(#white)"/>
        <circle cx="40"  cy="180" r="17" fill="url(#white)"/>
        <!-- black wall -->
        <circle cx="160" cy="220" r="17" fill="url(#black)"/>
        <circle cx="120" cy="180" r="17" fill="url(#black)"/>
        <circle cx="80"  cy="180" r="17" fill="url(#black)"/>
        <circle cx="40"  cy="140" r="17" fill="url(#black)"/>
      </g>

      <!-- the only move: glowing gold key point at an empty intersection -->
      <circle class="key-glow" cx="160" cy="180" r="20" fill="#CEB053" fill-opacity="0.25" filter="url(#glow)"/>
      <circle cx="160" cy="180" r="15" fill="none" stroke="url(#gold)" stroke-width="3"/>
      <circle cx="160" cy="180" r="4.5" fill="url(#gold)"/>
    </svg>

    <h1>onlymove-turnaus</h1>
    <p class="dates">13.–19. heinäkuuta 2026</p>
    <p class="tagline">Viikon mittainen tsumego-turnaus. Ratko go-tehtäviä, ja paras arvioitu vahvuutesi ratkaisee — hyvä tapa hioa taitoja ennen Euroopan Go-kongressia.</p>
    <div class="cta-row">
      <a class="btn btn-primary" href="/">Avaa onlymove <span class="arrow" aria-hidden="true">→</span></a>
      <a class="btn btn-ghost" href="/leaderboard">Tulostaulu <span class="arrow" aria-hidden="true">→</span></a>
    </div>
  </header>

  <div class="hoshi" aria-hidden="true"><span></span></div>

  <section aria-labelledby="palkinnot">
    <h2 id="palkinnot">Palkinnot</h2>
    <p class="lead">Kaksi sarjaa, molemmissa palkinto kahdelle parhaalle.</p>
    <div class="prizes">
      <div class="prize">
        <p class="cat">Absoluuttinen sarja</p>
        <p class="cat-desc">Korkein onlymove-arvio GoR-asteikolla.</p>
        <div class="place"><span class="medal first"></span><span class="lbl">1. sija</span><span class="amt">20 €</span></div>
        <div class="place"><span class="medal second"></span><span class="lbl">2. sija</span><span class="amt">10 €</span></div>
      </div>
      <div class="prize">
        <p class="cat">Suhteellinen sarja</p>
        <p class="cat-desc">Suurin ylisuoriutuminen: onlymove-arvio miinus oma GoR-lukusi.</p>
        <div class="place"><span class="medal first"></span><span class="lbl">1. sija</span><span class="amt">20 €</span></div>
        <div class="place"><span class="medal second"></span><span class="lbl">2. sija</span><span class="amt">10 €</span></div>
      </div>
    </div>
    <p class="prize-note">Osallistuminen on ilmaista.</p>
  </section>

  <div class="hoshi" aria-hidden="true"><span></span></div>

  <section aria-labelledby="osallistu">
    <h2 id="osallistu">Näin osallistut</h2>
    <p class="lead">Turnaus on tarkoitettu suomalaisille pelaajille.</p>
    <ol class="steps">
      <li>
        <span class="stone" aria-hidden="true"></span>
        <div>
          <div class="s-title">Hanki kutsulinkki</div>
          <div class="s-body">Pyydä linkki keneltä tahansa, joka on jo tulostaululla — kuka tahansa mukana oleva voi kutsua sinut.</div>
        </div>
      </li>
      <li>
        <span class="stone" aria-hidden="true"></span>
        <div>
          <div class="s-title">Liitä tilisi</div>
          <div class="s-body">Avaa linkki onlymove-sovelluksessa. Tilisi yhdistetään EGD-tunnukseesi ja pääset tulostaululle.</div>
        </div>
      </li>
      <li>
        <span class="stone" aria-hidden="true"></span>
        <div>
          <div class="s-title">Ratko viikon ajan</div>
          <div class="s-body">Ratko tsumegoja 13.–19.7. Paras (huippu) arviosi viikon aikana lasketaan turnaustulokseksi.</div>
        </div>
      </li>
    </ol>
    <p class="col-note">Tulokset näkyvät tulostaulun viikkosarakkeessa: turnauksen aikana kohdassa <code>This week peak</code> ja sen päätyttyä <code>Last week peak</code>. Voit vaihtaa <code>Absolute</code>- ja <code>Relative</code>-näkymien välillä.</p>
  </section>

  <div class="hoshi" aria-hidden="true"><span></span></div>

  <section aria-labelledby="arvio">
    <h2 id="arvio">Miten arvio lasketaan</h2>
    <p class="note-body">onlymove arvioi vahvuutesi viimeaikaisten ratkaisujesi perusteella. Myös <strong>tehtävään käytetty aika vaikuttaa arvioon</strong> — nopeat oikeat vastaukset kertovat suuremmasta vahvuudesta. Sinun ei kuitenkaan tarvitse murehtia poikkeuksellisen pitkiä aikoja, esimerkiksi jos sovellus jää auki: arvio käyttää mediaaniaikaa, joka sietää yksittäiset poikkeamat.</p>
  </section>

  <div class="hoshi" aria-hidden="true"><span></span></div>

  <section aria-labelledby="egc">
    <div class="egc">
      <svg class="egc-mark" viewBox="0 0 40 40" width="40" height="40" aria-hidden="true"><circle cx="20" cy="20" r="14" fill="none" stroke="#CEB053" stroke-width="3"/><circle cx="20" cy="20" r="4.5" fill="#CEB053"/></svg>
      <div>
        <div class="egc-k" id="egc">Ennen kongressia</div>
        <p><strong>Euroopan Go-kongressi alkaa 25. heinäkuuta.</strong> Käytä turnausviikko taitojen hiomiseen — ratkaisurutiini pysyy tuoreena, kun kisa pitää sinut liikkeellä.</p>
      </div>
    </div>
  </section>

  <footer>
    <p class="org">Järjestäjä: <b>Henry Hemming</b></p>
    <div class="flinks">
      <a href="/">onlymove.net</a>
      <a href="/leaderboard">Tulostaulu</a>
      <a href="https://discord.gg/ZU7jJNnkbB" target="_blank" rel="noopener">Discord</a>
    </div>
    <p class="fine">Turnaus on tarkoitettu suomalaisille osallistujille. Osallistuminen on ilmaista.</p>
  </footer>

</div>
</body>
</html>`;
}
