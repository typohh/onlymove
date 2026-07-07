// pwa/src/handlers/tournament-page.js
//
// GET /tournament — a standalone, directly-linkable, server-rendered landing page
// for the onlymove Go tournament (Finnish audience). Styled to match onlymove's
// gold-on-dark goban identity. Registered before the SPA fallback in src/index.js.
// Embeds a compact participants table in the hero (the same readLeaderboardRows the
// /leaderboard page computes from, one timeframe auto-picked by tournament phase);
// the full /leaderboard page stays separate and is linked from the page-closing
// CTA row + footer.

// Design: dark "goban at dusk", warm gold (#CEB053) accent. The hero is the
// onlymove app-icon board position, restyled in gold-on-dark, with one black stone
// marked in gold as the current move. Serif display for gravitas, monospace for the measured data
// (dates, prizes, ranks), sans for prose. Go stones and a hoshi star-point are the
// recurring structural marks. Content is Finnish; the tournament is for Finnish
// players. Copy is intentionally plain and instructional.

import { readLeaderboardRows, weekIdAt, RANK_START_MIN } from '../leaderboard.js';
import { ESC_SNIPPET, numAbs, numRel, flag } from './page-shell.js';

// Monday 13.7.2026 00:00 UTC — the tournament week (Monday-UTC week bucketing,
// same weekIdAt the leaderboard's week rollover uses).
const TOURNAMENT_WEEK = weekIdAt(Date.UTC(2026, 6, 13));

export async function handleTournament(env) {
    let rows = [];
    try {
        rows = await readLeaderboardRows(env);
    } catch (e) {
        console.error('tournament leaderboard read failed:', e);
    }
    return new Response(renderTournamentPage(rows), {
        status: 200,
        headers: {
            'Content-Type': 'text/html; charset=utf-8',
            'Cache-Control': 'public, max-age=60',
        },
    });
}

// The standings section shows ONE timeframe, picked by where "now" sits relative
// to the tournament week: a preview of current estimates before, the this-week
// peak during (the live tournament standing), the frozen last-week peak the week
// after (the final results). Later than that the week data has rolled off the
// leaderboard's two-week window, so it falls back to current estimates.
function tournamentPhase(nowMs) {
    const nowWeek = weekIdAt(nowMs);
    if (nowWeek < TOURNAMENT_WEEK) {
        return { tf: 0, medals: false, note: 'Turnaus alkaa 13.7. — taulukko näyttää ilmoittautuneiden tämänhetkiset arviot.' };
    }
    if (nowWeek === TOURNAMENT_WEEK) {
        return { tf: 1, medals: true, note: 'Turnaus käynnissä — sija määräytyy viikon huippuarvion mukaan.' };
    }
    if (nowWeek === TOURNAMENT_WEEK + 1) {
        return { tf: 2, medals: true, note: 'Turnauksen lopputulokset, 13.–19. heinäkuuta 2026.' };
    }
    return { tf: 0, medals: false, note: 'Turnaus on päättynyt — taulukko näyttää ilmoittautuneiden tämänhetkiset arviot.' };
}

function renderTournamentPage(rows) {
    const phase = tournamentPhase(Date.now());
    const tf = phase.tf;
    // One row per registered player, only the phase's timeframe: abs = estimated
    // GoR, rel = the GoR gap (the two prize series). GoR follows the timeframe —
    // the frozen last-week GoR when showing last-week peaks (tf 2), else current.
    const data = rows.map((r) => {
        const gorRaw = tf === 2 ? r.prevWeekGor : r.gor;
        const row = {
            name: r.name || '—',
            flag: flag(r.country),
            gor: Number.isFinite(gorRaw) ? gorRaw : null,
            abs: numAbs([r.currentAbs, r.thisWeekAbs, r.prevWeekAbs][tf]),
            rel: numRel([r.currentRel, r.thisWeekRel, r.prevWeekRel][tf]),
        };
        // Not yet ranked: attempts/threshold progress replaces the estimate (abs/rel
        // null → sorts last), same convention as the /leaderboard page.
        if (!r.ranked) row.progress = Math.min(r.novelCount || 0, RANK_START_MIN) + '/' + RANK_START_MIN;
        return row;
    });
    const json = JSON.stringify(data).replace(/</g, '\\u003c');
    return `<!doctype html>
<html lang="fi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#16130d">
<title>onlymove-turnaus · 13.–19. heinäkuuta 2026</title>
<meta name="description" content="Viikon mittainen tsumego-turnaus 13.–19.7.2026 rahapalkinnoilla. Ratko go-tehtäviä, ja paras arvioitu vahvuutesi ratkaisee. Ilmainen, avoin suomalaisille ja Suomessa asuville — treeniä Euroopan Go-kongressia varten.">
<meta property="og:type" content="website">
<meta property="og:title" content="onlymove-turnaus · 13.–19. heinäkuuta 2026">
<meta property="og:description" content="Viikon mittainen tsumego-turnaus rahapalkinnoilla. Paras arvioitu vahvuutesi ratkaisee. Ilmainen, avoin suomalaisille ja Suomessa asuville.">
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
  /* no bottom padding: the hero ends with the participants table, and the
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
  /* ---- scoring note ---- */
  .note-body { color: var(--mist); }
  .note-body strong { color: var(--ink); font-weight: 600; }

  /* ---- embedded participants table (inside the hero, above the CTAs) ---- */
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
  .lb td.val.progress { font-family: var(--mono); font-weight: 400; color: var(--faint); }
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
  @media (prefers-reduced-motion: reduce) {
    .key-glow { animation: none !important; }
    .btn, .btn .arrow { transition: none; }
  }
  /* the gold current-move mark breathes — the single deliberate motion */
  @keyframes keypulse { 0%,100% { opacity: .45; } 50% { opacity: 1; } }
  .key-glow { animation: keypulse 2.8s ease-in-out infinite; }
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
        <filter id="glow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
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

      <!-- current move: the black stone at (160,120), marked in gold with a soft glow -->
      <circle class="key-glow" cx="160" cy="120" r="14" fill="#CEB053" fill-opacity="0.28" filter="url(#glow)"/>
      <circle cx="160" cy="120" r="8.5" fill="none" stroke="url(#gold)" stroke-width="3"/>
      <circle cx="160" cy="120" r="2.6" fill="url(#gold)"/>
    </svg>

    <h1>onlymove-turnaus</h1>
    <p class="dates">13.–19. heinäkuuta 2026</p>
    <p class="tagline">Viikon mittainen, ilmainen tsumego-turnaus rahapalkinnoilla. Ratko go-tehtäviä, ja paras arvioitu vahvuutesi ratkaisee — treeniä Euroopan Go-kongressia varten.</p>

    <div class="hoshi" aria-hidden="true"><span></span></div>

    <section aria-labelledby="ilmoittautuneet" class="lb">
      <h2 id="ilmoittautuneet">Ilmoittautuneet</h2>
      <p class="lead">${phase.note}</p>
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
    <p class="lead">Turnaus on avoin suomalaisille ja Suomessa asuville.</p>
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
          <div class="s-body">Avaa saamasi kutsulinkki. Tilisi yhdistetään EGD-tunnukseesi ja pääset tulostaululle.</div>
        </div>
      </li>
      <li>
        <span class="stone" aria-hidden="true"></span>
        <div>
          <div class="s-title">Ratko viikon ajan</div>
          <div class="s-body">Ratko tsumegoja 13.–19.7. Paras (huippu) arviosi viikon aikana lasketaan turnaustulokseksi. Arvio edellyttää yhteensä vähintään 60 ratkaistua tehtävää, jotka voit kerätä jo ennen turnausta — uutena pelaajana aloita siis ajoissa.</div>
        </div>
      </li>
    </ol>
  </section>

  <div class="hoshi" aria-hidden="true"><span></span></div>

  <section aria-labelledby="arvio">
    <h2 id="arvio">Miten arvio lasketaan</h2>
    <p class="note-body">onlymove arvioi vahvuutesi viimeaikaisten ratkaisujesi perusteella. Myös <strong>tehtävään käytetty aika vaikuttaa arvioon</strong> — nopeat oikeat vastaukset kertovat suuremmasta vahvuudesta. Sinun ei kuitenkaan tarvitse murehtia poikkeuksellisen pitkiä aikoja, esimerkiksi jos sovellus jää auki: yksittäiset erittäin pitkät ratkaisuajat eivät heikennä arviotasi.</p>
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
    <p class="fine">Turnaus on avoin suomalaisille ja Suomessa asuville. Järjestäjä pidättää oikeuden hylätä osallistujan epäillyn vilpin vuoksi.</p>
  </footer>

</div>
<script>
(function () {
  var DATA = ${json};
  var MEDALS = ${phase.medals ? 'true' : 'false'};
  var mode = 'abs';   // 'abs' | 'rel' — the two prize series
  ${ESC_SNIPPET}
  var tbody = document.getElementById('lbRows');

  // Absolute = plain estimated GoR; relative = signed GoR gap vs the player's own GoR.
  function fmt(v) {
    if (v === null || v === undefined) return '—';
    return mode === 'rel' ? (v >= 0 ? '+' : '−') + Math.abs(v) : String(v);
  }
  // Active series descending; rows without a value (idle week / not yet ranked) last.
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
    if (!DATA.length) { tbody.innerHTML = '<tr><td colspan="4" class="empty">Ei pelaajia vielä.</td></tr>'; return; }
    var sorted = DATA.slice().sort(cmp);
    tbody.innerHTML = sorted.map(function (r, i) {
      // Gold/outline dots on the top two — the prize places — only while the shown
      // timeframe IS the tournament standing (during the week + the results week).
      var medal = MEDALS && !r.progress && r[mode] !== null && i < 2
        ? '<span class="medal ' + (i === 0 ? 'first' : 'second') + '" aria-hidden="true"></span>' : '';
      var val = r.progress
        ? '<td class="val progress">' + esc(r.progress) + '</td>'
        : '<td class="val">' + fmt(r[mode]) + '</td>';
      return '<tr><td class="pos">' + medal + (i + 1) + '</td>' +
        '<td class="pl">' + (r.flag ? r.flag + ' ' : '') + '<span class="name">' + esc(r.name) + '</span></td>' +
        '<td class="gor">' + (r.gor === null ? '—' : r.gor) + '</td>' +
        val + '</tr>';
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
