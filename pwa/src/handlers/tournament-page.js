// pwa/src/handlers/tournament-page.js
//
// GET /tournament — a standalone, server-rendered FINAL-RESULTS page for the
// concluded onlymove Go tournament (13–19 July 2026, Finnish audience). Styled to
// match onlymove's gold-on-dark goban identity. Registered before the SPA fallback
// in src/index.js.
//
// STATIC: the standings are baked in below (PARTICIPANTS) — a verbatim snapshot of
// the tournament week's peak values ("last week peak") taken from the production
// leaderboard on 2026-07-20, after the event. There is deliberately NO live data
// retrieval (no readLeaderboardRows / D1): the results are final and frozen, so a
// stale-scale recompute could only change what players actually saw during the
// event. `abs` = peak estimated GoR, `rel` = peak over-performance (estimate − own
// GoR), `gor` = the player's EGD GoR — all in GoR points, as the live board showed.

import { ESC_SNIPPET } from './page-shell.js';

// Final standings — the tournament-week peak per player (production last-week-peak
// snapshot, 2026-07-20). Order here is arbitrary; the page sorts client-side by the
// active series. Names are as stored in EGD (ASCII).
const PARTICIPANTS = [
    { name: "Henry Hemming", flag: "🇫🇮", gor: 2347, abs: 2397, rel: 50 },
    { name: "Pekka Keipi", flag: "🇫🇮", gor: 2282, abs: 2297, rel: 15 },
    { name: "Ilmari Jaaranen", flag: "🇫🇮", gor: 2009, abs: 2300, rel: 291 },
    { name: "Petri Haustola", flag: "🇫🇮", gor: 1878, abs: 2033, rel: 155 },
    { name: "Martti Rannanjarvi", flag: "🇫🇮", gor: 1302, abs: 1975, rel: 673 },
    { name: "Markus Koivisto", flag: "🇫🇮", gor: 98, abs: 1713, rel: 1615 },
    { name: "Pekka Lajunen", flag: "🇫🇮", gor: 2243, abs: 2274, rel: 31 },
    { name: "Juuso Takalainen", flag: "🇫🇮", gor: 1569, abs: 2112, rel: 543 },
    { name: "Jarmo Ahosola", flag: "🇫🇮", gor: 1731, abs: 1958, rel: 227 },
    { name: "Jussa Jutila", flag: "🇫🇮", gor: 1843, abs: 2134, rel: 291 },
    { name: "Olli Hella", flag: "🇫🇮", gor: 2355, abs: 2449, rel: 94 },
    { name: "Olli Ervelae", flag: "🇫🇮", gor: 2365, abs: 2258, rel: -107 },
    { name: "Ilkka Sovanto", flag: "🇫🇮", gor: 2257, abs: 2258, rel: 1 },
    { name: "Joni Hanski", flag: "🇫🇮", gor: 2193, abs: 2319, rel: 126 },
    { name: "Mika Lemstrom", flag: "🇫🇮", gor: 1340, abs: 1844, rel: 504 },
    { name: "Tuomas Hella", flag: "🇫🇮", gor: 2400, abs: 2325, rel: -75 },
    { name: "Mikko Siukola", flag: "🇫🇮", gor: 2442, abs: 2297, rel: -145 },
    { name: "Toni Kyottila", flag: "🇫🇮", gor: 2230, abs: 2273, rel: 43 },
    { name: "Antti Niemi", flag: "🇫🇮", gor: 2046, abs: 2165, rel: 119 },
    { name: "Ernestas Romeika", flag: "🇱🇹", gor: 2307, abs: 2134, rel: -173 },
    { name: "Ari Hietanen", flag: "🇫🇮", gor: 1958, abs: 2188, rel: 230 },
    { name: "Juho Heikkinen", flag: "🇫🇮", gor: 2152, abs: 2168, rel: 16 },
    { name: "Lauri Vihavainen", flag: "🇫🇮", gor: 1946, abs: 2083, rel: 137 },
    { name: "Kalle Timperi", flag: "🇫🇮", gor: 2192, abs: 2288, rel: 96 },
    { name: "Anttoni Maula", flag: "🇫🇮", gor: 2197, abs: 2213, rel: 16 },
    { name: "Marco Henkel", flag: "🇩🇪", gor: 2064, abs: 2035, rel: -29 },
    { name: "Heikki Lainela", flag: "🇫🇮", gor: 2226, abs: 2157, rel: -69 },
];

export function handleTournament() {
    return new Response(renderTournamentPage(), {
        status: 200,
        headers: {
            'Content-Type': 'text/html; charset=utf-8',
            // Static, final results — cache aggressively.
            'Cache-Control': 'public, max-age=3600',
        },
    });
}

// Server-side HTML escape (the client's esc() from ESC_SNIPPET isn't in scope here).
const escHtml = (s) => String(s).replace(/[&<>"]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// Top two of a series (highest value first); used to name the prize places.
function topTwo(key) {
    return PARTICIPANTS
        .filter((p) => Number.isFinite(p[key]))
        .slice().sort((a, b) => b[key] - a[key])
        .slice(0, 2);
}

function renderTournamentPage() {
    const data = PARTICIPANTS.map((p) => ({ name: p.name, flag: p.flag, gor: p.gor, abs: p.abs, rel: p.rel }));
    const json = JSON.stringify(data).replace(/</g, '\\u003c');
    const absWin = topTwo('abs'), relWin = topTwo('rel');
    const win = (arr, i) => (arr[i] ? arr[i].name : '—');
    return `<!doctype html>
<html lang="fi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#16130d">
<title>onlymove-turnaus · lopputulokset · 13.–19. heinäkuuta 2026</title>
<meta name="description" content="onlymove-tsumego-turnauksen lopputulokset (13.–19. heinäkuuta 2026). Kaksi sarjaa: absoluuttisesti paras onlymove-arvio sekä paras ylisuoriutuminen GoR-lukuihin nähden. Turnaus on päättynyt.">
<meta property="og:type" content="website">
<meta property="og:title" content="onlymove-turnaus · lopputulokset">
<meta property="og:description" content="onlymove-tsumego-turnauksen lopputulokset (13.–19. heinäkuuta 2026). Kaksi sarjaa: absoluuttisesti paras onlymove-arvio sekä paras ylisuoriutuminen. Turnaus on päättynyt.">
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
  /* no bottom padding: the hero ends with the results table, and the
     following hoshi divider's own margin provides the section gap */
  header.hero { text-align: center; padding: 3.5rem 0 0; }
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
  /* concluded badge — a quiet gold pill making it unmistakable the event is over */
  .status {
    display: inline-block; font-family: var(--mono); font-size: .72rem;
    letter-spacing: .16em; text-transform: uppercase; color: var(--gold-lite);
    border: 1px solid rgba(206,176,83,.45); border-radius: 999px;
    padding: .32rem .85rem; margin: .95rem 0 0;
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
  /* winner name is the emphasis; the place label + amount trail it */
  .place .who { color: var(--ink); font-weight: 600; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .place .lbl { color: var(--faint); font-size: .82rem; }
  .place .amt { font-family: var(--mono); font-weight: 600; color: var(--ink); font-size: 1.05rem; }
  .prize-note { color: var(--faint); font-size: .85rem; margin: 1rem 0 0; text-align: center; }

  /* ---- scoring note ---- */
  .note-body { color: var(--mist); }
  .note-body strong { color: var(--ink); font-weight: 600; }

  /* ---- results table (inside the hero, above the CTAs) ---- */
  /* left-aligned like the other sections, against the hero's centered text */
  .lb { margin: 0; text-align: left; }
  .lb .lead { margin-bottom: 1.1rem; }
  .lb-toggle { display: inline-flex; border: 1px solid rgba(206,176,83,.35); border-radius: 999px; overflow: hidden; margin: .2rem 0 1.1rem; }
  .lb-toggle button {
    font: inherit; font-size: .92rem; padding: .42rem .95rem; border: 0; background: transparent;
    color: var(--mist); cursor: pointer; white-space: nowrap;
    transition: background-color .15s ease, color .15s ease;
  }
  .lb-toggle button[aria-pressed="true"] { background: var(--gold); color: #17130a; font-weight: 600; }
  .lb-toggle button:focus-visible { outline: 2px solid var(--gold-lite); outline-offset: -2px; }
  .lb-table-wrap { overflow-x: auto; }
  .lb table { width: 100%; border-collapse: collapse; font-size: .95rem; }
  .lb th, .lb td { padding: .55rem .5rem; text-align: right; white-space: nowrap; }
  .lb th.player, .lb td.pl { text-align: left; width: 99%; }
  .lb thead th {
    font-family: var(--mono); font-size: .68rem; font-weight: 600;
    letter-spacing: .14em; text-transform: uppercase;
    color: var(--gold); opacity: .8; border-bottom: 1px solid var(--line);
  }
  .lb tbody tr { border-bottom: 1px solid var(--line-soft); }
  .lb td.pos { color: var(--faint); font-variant-numeric: tabular-nums; }
  .lb td.pos .medal { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: .4rem; }
  .lb td.gor { color: var(--mist); font-variant-numeric: tabular-nums; }
  .lb td.val { font-family: var(--mono); font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
  .lb .name { overflow: hidden; text-overflow: ellipsis; max-width: 14rem; display: inline-block; vertical-align: bottom; }
  .lb .empty { color: var(--faint); text-align: center; padding: 1.6rem 0; }

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
</style>
</head>
<body>
<div class="grid-bg" aria-hidden="true"></div>
<div class="wrap">

  <header class="hero">
    <p class="eyebrow">onlymove · go-turnaus</p>

    <svg class="board-art" viewBox="0 0 240 240" role="img" aria-label="onlymove-logon go-asema — yksi kivi on merkitty siirroksi.">
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
      </defs>

      <rect x="8" y="8" width="224" height="224" rx="16" fill="url(#wood)" stroke="#33291899" stroke-width="1"/>

      <!-- 5x5 grid, lines at 40,80,120,160,200 — the onlymove app-icon board -->
      <g stroke="#CEB053" stroke-opacity="0.22" stroke-width="1.4" stroke-linecap="round">
        <line x1="40"  y1="40" x2="40"  y2="200"/>
        <line x1="80"  y1="40" x2="80"  y2="200"/>
        <line x1="120" y1="40" x2="120" y2="200"/>
        <line x1="160" y1="40" x2="160" y2="200"/>
        <line x1="200" y1="40" x2="200" y2="200"/>
        <line x1="40" y1="40"  x2="200" y2="40"/>
        <line x1="40" y1="80"  x2="200" y2="80"/>
        <line x1="40" y1="120" x2="200" y2="120"/>
        <line x1="40" y1="160" x2="200" y2="160"/>
        <line x1="40" y1="200" x2="200" y2="200"/>
      </g>

      <!-- stones: the onlymove app-icon position, restyled in gold-on-dark -->
      <g filter="url(#shadow)">
        <circle cx="80"  cy="40"  r="18" fill="url(#black)"/>
        <circle cx="160" cy="40"  r="18" fill="url(#white)"/>
        <circle cx="40"  cy="80"  r="18" fill="url(#black)"/>
        <circle cx="80"  cy="80"  r="18" fill="url(#black)"/>
        <circle cx="120" cy="80"  r="18" fill="url(#black)"/>
        <circle cx="160" cy="80"  r="18" fill="url(#white)"/>
        <circle cx="200" cy="80"  r="18" fill="url(#white)"/>
        <circle cx="40"  cy="120" r="18" fill="url(#white)"/>
        <circle cx="80"  cy="120" r="18" fill="url(#white)"/>
        <circle cx="120" cy="120" r="18" fill="url(#white)"/>
        <circle cx="160" cy="120" r="18" fill="url(#black)"/>
        <circle cx="200" cy="120" r="18" fill="url(#black)"/>
        <circle cx="80"  cy="160" r="18" fill="url(#white)"/>
        <circle cx="120" cy="160" r="18" fill="url(#black)"/>
        <circle cx="160" cy="160" r="18" fill="url(#black)"/>
        <circle cx="200" cy="160" r="18" fill="url(#white)"/>
        <circle cx="40"  cy="200" r="18" fill="url(#black)"/>
        <circle cx="80"  cy="200" r="18" fill="url(#white)"/>
        <circle cx="160" cy="200" r="18" fill="url(#black)"/>
      </g>

      <!-- the only move: the black stone at (160,120), marked in gold -->
      <circle cx="160" cy="120" r="8.5" fill="none" stroke="url(#gold)" stroke-width="3"/>
      <circle cx="160" cy="120" r="2.6" fill="url(#gold)"/>
    </svg>

    <h1>onlymove-turnaus</h1>
    <p class="dates">13.–19. heinäkuuta 2026</p>
    <p class="status">Turnaus on päättynyt · lopputulokset</p>
    <p class="tagline">Viikon mittainen tsumego-turnaus on ohi. Alla lopulliset tulokset kummassakin sarjassa — turnausviikon korkeimmat onlymove-arviot.</p>

    <div class="hoshi" aria-hidden="true"><span></span></div>

    <section aria-labelledby="tulokset" class="lb">
      <h2 id="tulokset">Lopputulokset</h2>
      <p class="lead">Turnausviikon (13.–19.7.) huippuarvot. Vaihda sarjaa alta: absoluuttinen (korkein onlymove-arvio) tai suhteellinen (suurin ylisuoriutuminen GoR-lukuun nähden). Ykkös- ja kakkossija on merkitty kullalla.</p>
      <div class="lb-toggle" role="group" aria-label="sarja">
        <button id="mAbs" aria-pressed="true">Absoluuttinen</button>
        <button id="mRel" aria-pressed="false">Suhteellinen</button>
      </div>
      <div class="lb-table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th class="player">Pelaaja</th>
              <th>GoR</th>
              <th id="estHead">onlymove</th>
            </tr>
          </thead>
          <tbody id="lbRows"></tbody>
        </table>
      </div>
    </section>
  </header>

  <div class="hoshi" aria-hidden="true"><span></span></div>

  <section aria-labelledby="palkinnot">
    <h2 id="palkinnot">Palkinnot</h2>
    <p class="lead">Kaksi sarjaa, molemmissa palkinto kahdelle parhaalle.</p>
    <div class="prizes">
      <div class="prize">
        <p class="cat">Absoluuttinen sarja</p>
        <p class="cat-desc">Korkein onlymove-arvio GoR-asteikolla.</p>
        <div class="place"><span class="medal first"></span><span class="who">${escHtml(win(absWin, 0))}</span><span class="lbl">1. sija</span><span class="amt">20 €</span></div>
        <div class="place"><span class="medal second"></span><span class="who">${escHtml(win(absWin, 1))}</span><span class="lbl">2. sija</span><span class="amt">10 €</span></div>
      </div>
      <div class="prize">
        <p class="cat">Suhteellinen sarja</p>
        <p class="cat-desc">Suurin ylisuoriutuminen: onlymove-arvio miinus oma GoR-luku.</p>
        <div class="place"><span class="medal first"></span><span class="who">${escHtml(win(relWin, 0))}</span><span class="lbl">1. sija</span><span class="amt">20 €</span></div>
        <div class="place"><span class="medal second"></span><span class="who">${escHtml(win(relWin, 1))}</span><span class="lbl">2. sija</span><span class="amt">10 €</span></div>
      </div>
    </div>
    <p class="prize-note">Järjestäjä on yhteydessä palkittuihin. Onnittelut kaikille osallistujille!</p>
  </section>

  <div class="hoshi" aria-hidden="true"><span></span></div>

  <section aria-labelledby="arvio">
    <h2 id="arvio">Miten arvio laskettiin</h2>
    <p class="note-body">onlymove arvioi vahvuutesi viimeaikaisten ratkaisujesi perusteella, ja turnaustulokseksi otettiin viikon <strong>korkein arvio</strong>. Myös <strong>tehtävään käytetty aika vaikutti arvioon</strong> — nopeat oikeat vastaukset kertovat suuremmasta vahvuudesta — mutta yksittäiset erittäin pitkät ratkaisuajat (esimerkiksi sovelluksen jäätyä auki) eivät heikentäneet arviota.</p>
  </section>

  <div class="hoshi" aria-hidden="true"><span></span></div>

  <div class="cta-row">
    <a class="btn btn-primary" href="/">Avaa onlymove <span class="arrow" aria-hidden="true">→</span></a>
    <a class="btn btn-ghost" href="/leaderboard">Tulostaulu <span class="arrow" aria-hidden="true">→</span></a>
  </div>

  <footer>
    <p class="org">Järjestäjä: <b>Henry Hemming</b></p>
    <div class="flinks">
      <a href="/">onlymove.net</a>
      <a href="/leaderboard">Tulostaulu</a>
      <a href="https://discord.gg/ZU7jJNnkbB" target="_blank" rel="noopener">Discord</a>
    </div>
    <p class="fine">Tulokset ovat turnausviikon lopulliset huippuarvot. Kiitos kaikille pelaajille.</p>
  </footer>

</div>
<script>
(function () {
  var DATA = ${json};
  var mode = 'abs';   // 'abs' | 'rel' — the two prize series
  ${ESC_SNIPPET}
  var tbody = document.getElementById('lbRows');

  // Absolute = plain peak estimated GoR; relative = signed peak over-performance vs GoR.
  function fmt(v) {
    if (v === null || v === undefined) return '—';
    return mode === 'rel' ? (v >= 0 ? '+' : '−') + Math.abs(v) : String(v);
  }
  // Active series descending; rows without a value last (defensive — all are ranked).
  function cmp(a, b) {
    var va = a[mode], vb = b[mode];
    var na = (va === null || va === undefined), nb = (vb === null || vb === undefined);
    if (na && nb) return 0;
    if (na) return 1;
    if (nb) return -1;
    return vb - va;
  }
  function render() {
    document.getElementById('estHead').textContent = mode === 'abs' ? 'onlymove' : 'Ero';
    if (!DATA.length) { tbody.innerHTML = '<tr><td colspan="4" class="empty">Ei tuloksia.</td></tr>'; return; }
    var sorted = DATA.slice().sort(cmp);
    tbody.innerHTML = sorted.map(function (r, i) {
      // Gold/outline dots on the top two — the prize places.
      var medal = r[mode] !== null && r[mode] !== undefined && i < 2
        ? '<span class="medal ' + (i === 0 ? 'first' : 'second') + '" aria-hidden="true"></span>' : '';
      return '<tr><td class="pos">' + medal + (i + 1) + '</td>' +
        '<td class="pl">' + (r.flag ? r.flag + ' ' : '') + '<span class="name">' + esc(r.name) + '</span></td>' +
        '<td class="gor">' + (r.gor === null ? '—' : r.gor) + '</td>' +
        '<td class="val">' + fmt(r[mode]) + '</td></tr>';
    }).join('');
  }
  function setMode(m) {
    mode = m;
    document.getElementById('mAbs').setAttribute('aria-pressed', m === 'abs');
    document.getElementById('mRel').setAttribute('aria-pressed', m === 'rel');
    render();
  }
  document.getElementById('mAbs').onclick = function () { setMode('abs'); };
  document.getElementById('mRel').onclick = function () { setMode('rel'); };
  render();
})();
</script>
</body>
</html>`;
}
