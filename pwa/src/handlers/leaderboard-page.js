// pwa/src/handlers/leaderboard-page.js
//
// GET /leaderboard — a standalone, directly-linkable, server-rendered page (loads
// without booting the tsumego SPA). Computes on read (no cron blob); a short edge
// cache bounds D1 reads under traffic. Lists every player active in the past week
// who has at least RANK_START_MIN solves here (so their onlymove rank is known):
// invite-registered players with their EGD/AGA name, everyone else as "Anonymous".
// Players still below the gate are only counted, in a footer tally.
// A UNITS selector (onlymove / EGF / AGA, persisted in localStorage) picks the
// frame both value columns speak. EVERY value cell renders as "(pct%) rank" —
// the rank in the selected frame plus the floored progress toward the next one
// (raw GoR / AGA rating numbers are not displayed; they remain the sort keys):
//   * Verified — from the player's linked federation account, machine heartbeats
//     only (EGD GoR heartbeat / AGA Ratings Explorer heartbeat), activity-gated
//     per the EGF criteria in readLeaderboardRows. EGF units decompose the GoR
//     round-band (2389 -> "(39%) 4d" — the GoR's own convention), AGA units the
//     AGA rating through AGA's floor convention (agaPctRank; 5.7 -> "(70%) 5d");
//     onlymove units leave the column blank (there is no raw onlymove-frame
//     source datum). Human survey declarations are NOT revealed here (they feed
//     only calibration + the flag fallback).
//   * onlymove — the current estimate from their solving here, decomposed in the
//     selected frame (onlymove units via the shared rankProgress, matching the
//     front-page rank list; EGF/AGA units via that federation's convention).
// Any header sorts (cells sort by their underlying continuous scalar).

import { readLeaderboardRows, RANK_START_MIN, gorToRankScalar } from '../leaderboard.js';
import { rankScalarToLabel, rankProgress } from '../../shared/rank-math.js';
import { baseCss, ESC_SNIPPET, flag } from './page-shell.js';

const ACTIVE_DAYS = 7;   // the visible board = players active in the past week

// Shown when we can't resolve a country for a row (no GoR/invitation/geo-IP, or a
// hidden profile). The UN flag reads as "nation unspecified" and renders as a real
// regional-indicator glyph like every other flag; swap to '🏴‍☠️' etc. here to taste.
const UNKNOWN_FLAG = '🇺🇳';

// Localization for this server-rendered page. The page is edge-cached for 60s, so
// language MUST be resolved on the CLIENT (navigator.language) — an Accept-Language
// server split would cache one language and serve it to every viewer. The dict is
// inlined into the page and applied by the prelude script; it mirrors the SPA's
// getLang() prefix match (zh/ja/ko/en) and {param} interpolation (js/i18n.js).
// `{discord}` in inviteLine is filled with the anchor at runtime. "onlymove" is a
// brand name and stays untranslated (the third column header). NOTE: the tournament
// page (Finnish-only) and the invite page (English-only) are deliberate exceptions
// and do NOT share this dict.
const STR = {
    en: {
        leaderboard: 'Leaderboard',
        player: 'Player',
        verified: 'Verified',
        units: 'Units',
        anonymous: 'Anonymous',
        empty: 'No players yet.',
        belowNoteOne: '1 more active player has fewer than {min} solves and isn\'t shown yet.',
        belowNoteMany: '{n} more active players have fewer than {min} solves and aren\'t shown yet.',
        staleNoteOne: '1 player\'s verified rating is not shown because it is too out of date (no recent enough tournament).',
        staleNoteMany: '{n} players\' verified ratings are not shown because they are too out of date (no recent enough tournament).',
        inactiveNote: 'Players not active in the past 7 days are not listed.',
        inviteLine: 'Want your name shown instead of "Anonymous"? Registration is by invite — ask in {discord}.',
        hideLabel: 'Hide my profile — withhold my name, country, and verified rating from this board. My onlymove rank still shows, but anonymously.',
        saving: 'Saving…',
        hiddenMsg: 'Your profile is hidden. The board updates within a minute.',
        visibleMsg: 'Your profile is visible. The board updates within a minute.',
        saveError: 'Could not save — please try again.',
    },
    zh: {
        leaderboard: '排行榜',
        player: '玩家',
        verified: '已验证',
        units: '单位',
        anonymous: '匿名',
        empty: '暂无玩家。',
        belowNoteOne: '还有 1 名活跃玩家的解题数不足 {min}，暂未显示。',
        belowNoteMany: '还有 {n} 名活跃玩家的解题数不足 {min}，暂未显示。',
        staleNoteOne: '有 1 名玩家的已验证等级分因过于陈旧（近期无比赛记录）而未显示。',
        staleNoteMany: '有 {n} 名玩家的已验证等级分因过于陈旧（近期无比赛记录）而未显示。',
        inactiveNote: '过去 7 天内没有活动的玩家不会列出。',
        inviteLine: '想显示您的名字而不是「匿名」？注册需邀请——请在 {discord} 询问。',
        hideLabel: '隐藏我的资料——不在此榜单显示我的名字、国家/地区和已验证等级分。我的 onlymove 段位仍会显示，但为匿名。',
        saving: '正在保存…',
        hiddenMsg: '您的资料已隐藏。榜单将在一分钟内更新。',
        visibleMsg: '您的资料已公开。榜单将在一分钟内更新。',
        saveError: '保存失败——请重试。',
    },
    ja: {
        leaderboard: 'ランキング',
        player: 'プレイヤー',
        verified: '認証済み',
        units: '単位',
        anonymous: '匿名',
        empty: 'まだプレイヤーがいません。',
        belowNoteOne: 'あと1人、解答数が {min} 未満のアクティブなプレイヤーがいますが、まだ表示されていません。',
        belowNoteMany: 'あと{n}人、解答数が {min} 未満のアクティブなプレイヤーがいますが、まだ表示されていません。',
        staleNoteOne: '1人のプレイヤーの認証済みレーティングは、古すぎるため（最近の大会出場がないため）表示されていません。',
        staleNoteMany: '{n}人のプレイヤーの認証済みレーティングは、古すぎるため（最近の大会出場がないため）表示されていません。',
        inactiveNote: '過去7日間に活動のないプレイヤーは表示されません。',
        inviteLine: '「匿名」の代わりに名前を表示しますか？登録は招待制です——{discord} で聞いてください。',
        hideLabel: 'プロフィールを隠す——このボードで名前・国・認証済みレーティングを表示しません。onlymove ランクは匿名で引き続き表示されます。',
        saving: '保存中…',
        hiddenMsg: 'プロフィールは非表示です。ボードは1分以内に更新されます。',
        visibleMsg: 'プロフィールは表示されています。ボードは1分以内に更新されます。',
        saveError: '保存できませんでした——もう一度お試しください。',
    },
    ko: {
        leaderboard: '순위표',
        player: '플레이어',
        verified: '인증됨',
        units: '단위',
        anonymous: '익명',
        empty: '아직 플레이어가 없습니다.',
        belowNoteOne: '해결한 문제가 {min}개 미만인 활동 플레이어가 1명 더 있으며 아직 표시되지 않습니다.',
        belowNoteMany: '해결한 문제가 {min}개 미만인 활동 플레이어가 {n}명 더 있으며 아직 표시되지 않습니다.',
        staleNoteOne: '플레이어 1명의 인증된 레이팅이 너무 오래되어(최근 대회 기록 없음) 표시되지 않습니다.',
        staleNoteMany: '플레이어 {n}명의 인증된 레이팅이 너무 오래되어(최근 대회 기록 없음) 표시되지 않습니다.',
        inactiveNote: '지난 7일간 활동이 없는 플레이어는 표시되지 않습니다.',
        inviteLine: '"익명" 대신 이름을 표시하고 싶으신가요? 등록은 초대제입니다 — {discord}에서 문의하세요.',
        hideLabel: '내 프로필 숨기기 — 이 보드에서 내 이름, 국가, 인증된 레이팅을 표시하지 않습니다. onlymove 랭크는 익명으로 계속 표시됩니다.',
        saving: '저장 중…',
        hiddenMsg: '프로필이 숨겨졌습니다. 보드는 1분 이내에 업데이트됩니다.',
        visibleMsg: '프로필이 표시됩니다. 보드는 1분 이내에 업데이트됩니다.',
        saveError: '저장하지 못했습니다 — 다시 시도해 주세요.',
    },
};

export async function handleLeaderboard(env) {
    let rows = [], belowMin = 0, staleVerified = 0;
    try {
        ({ rows, belowMin, staleVerified } = await readLeaderboardRows(env, { activeDays: ACTIVE_DAYS }));
    } catch (e) {
        console.error('leaderboard read failed:', e);
    }
    return new Response(renderPage(rows, belowMin, staleVerified), {
        status: 200,
        headers: {
            'Content-Type': 'text/html; charset=utf-8',
            'Cache-Control': 'public, max-age=60',
        },
    });
}

// { pct, t } from a continuous rank scalar via the SHARED rankProgress — the same
// rank + floored progress-to-next-rank decomposition the front-page rank list
// uses. Every cell in every unit mode renders as "(pct%) rank"; the percent is
// derived from the underlying rating scalar, so it always agrees with the shown
// band (for GoR this IS the EGF convention: 2389 -> "(39%) 4d").
function toPctRank(scalar) {
    if (!Number.isFinite(scalar)) return null;
    const p = rankProgress(scalar);
    return { pct: p.percent, t: p.label };
}

// The AGA decomposition — AGA's official convention is FLOOR-based (dan =
// floor(rating), kyu = floor(|rating|); see src/aga.js), NOT the round-band
// convention above, so AGA cells decompose the rating in AGA's own frame: label
// from the floor band, percent = fractional progress through it (5.7 ->
// "(70%) 5d", -3.8 -> "(20%) 3k"). A kyu band's top edge IS its integer
// (rating -3.0 = the strongest 3k), hence the 99 clamp there. The (-1, 1) gap
// holds no real AGA rating; a converted ESTIMATE landing in it snaps to the
// nearer boundary.
function agaPctRank(rating) {
    if (!Number.isFinite(rating)) return null;
    if (rating >= 1) {
        const d = Math.floor(rating);
        return { pct: Math.floor((rating - d) * 100), t: d + 'd' };
    }
    if (rating > -1) return rating >= 0 ? { pct: 0, t: '1d' } : { pct: 99, t: '1k' };
    const k = Math.floor(-rating);
    return { pct: Math.min(99, Math.floor((rating + k + 1) * 100)), t: k + 'k' };
}

// Continuous internal AGA rank scalar (1k = 0, 1d = 1) -> an AGA-style rating:
// dan region maps 1:1, kyu region shifts past the AGA (-1, 1) gap (scalar 0 =
// 1k -> -1.0), the sub-1-band (0, 1) interval interpolates across the gap.
function agaScalarToRating(s) {
    return s >= 1 ? s : s <= 0 ? s - 1 : 2 * s - 1;
}

function renderPage(rows, belowMin, staleVerified) {
    // Per row: `est` = the onlymove estimate in the three unit frames — `v` the
    // UNTIMED internal scalar (the om-mode sort key), `om`/`pct` its rank label
    // + progress, and `egf`/`aga` the TIME-ADJUSTED server-frame conversions as
    // { v: that frame's sort scalar, pct, t } (the frames the verified ratings
    // natively speak — see readLeaderboardRows). `ver` = the verified official
    // data per federation:
    // { v: raw rating (the sort key), pct, t } with the band + percent derived
    // from that rating (the official rank_label only drives the activity gate).
    const data = rows.map((r) => {
        let est = null;
        if (Number.isFinite(r.currentInternal)) {
            const p = rankProgress(r.currentInternal);
            est = { v: r.currentInternal, om: p.label, pct: p.percent };
            // Each frame carries its own sort scalar `v`: the om order (untimed)
            // and the EGF/AGA orders (time-adjusted server frames) genuinely
            // differ per user by the personal time dock, so switching units must
            // re-sort by the displayed frame's own value.
            const egf = toPctRank(r.currentEgf);
            if (egf) est.egf = { v: r.currentEgf, ...egf };
            const agaRating = Number.isFinite(r.currentAga) ? agaScalarToRating(r.currentAga) : null;
            const aga = agaRating !== null ? agaPctRank(agaRating) : null;
            if (aga) est.aga = { v: agaRating, ...aga };
        }
        const ver = {};
        if (r.egd) {
            const p = toPctRank(gorToRankScalar(r.egd.gor));
            if (p) ver.egf = { v: r.egd.gor, pct: p.pct, t: p.t };
        }
        if (r.aga) {
            const p = agaPctRank(r.aga.rating);
            if (p) ver.aga = { v: r.aga.rating, pct: p.pct, t: p.t };
        }
        const row = {
            // Anonymous (unregistered) rows carry anon: true and no name — the client
            // renders a muted "Anonymous" label for them.
            name: r.registered ? (r.name || '—') : null,
            anon: !r.registered,
            flag: flag(r.country) || UNKNOWN_FLAG,
            ver,
            est,
        };
        return row;
    });
    const json = JSON.stringify(data).replace(/</g, '\\u003c');
    const strJson = JSON.stringify(STR).replace(/</g, '\\u003c');
    // Footer tallies: active players still below the list-entry gate (no per-user
    // sample count is shown; this aggregate is the only nod to them), and shown
    // players whose verified rating was withheld as out of date (the EGF activity
    // gate). The English text here is the no-JS default; the prelude re-renders
    // both localized.
    const belowNote = belowMin
        ? `<p id="lbBelow">${belowMin} more active player${belowMin === 1 ? '' : 's'} ${belowMin === 1 ? 'has' : 'have'} fewer than ${RANK_START_MIN} solves and ${belowMin === 1 ? "isn't" : "aren't"} shown yet.</p>`
        : '';
    const staleNote = staleVerified
        ? `<p id="lbStale">${staleVerified === 1
            ? "1 player's verified rating is not shown because it is too out of date (no recent enough tournament)."
            : `${staleVerified} players' verified ratings are not shown because they are too out of date (no recent enough tournament).`}</p>`
        : '';

    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>onlymove — Leaderboard</title>
<style>
  ${baseCss(760)}
  th, td { padding: .5rem .5rem; text-align: right; white-space: nowrap; }
  td:nth-child(2) { text-align: left; width: 99%; }
  th.player { text-align: left; }
  thead th { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; opacity: .55; font-weight: 600;
             border-bottom: 1px solid; }
  /* vertical divider separating the declared range from the onlymove estimate */
  th.om-start, td.om-start { border-left: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
  thead th[data-sort] { cursor: pointer; user-select: none; -webkit-user-select: none; }
  thead th[data-sort]:hover { opacity: .85; }
  thead th.sorted { opacity: .95; }
  td.rank { opacity: .5; font-variant-numeric: tabular-nums; }
  td.val { font-variant-numeric: tabular-nums; font-weight: 600; }
  /* progress-to-next-rank "(64%)" — same relative size/mute as the front-page
     rank list's percentage span (rank-translation.js: 0.8em, opacity .6).
     Both value columns render it (pctRank), so scope on the class, not on
     td.val — the Verified cell is a td.decl. */
  .pct { font-size: .8em; opacity: .6; font-weight: 400; }
  td.decl { opacity: .7; font-variant-numeric: tabular-nums; }
  .name { overflow: hidden; text-overflow: ellipsis; max-width: 16rem; display: inline-block; vertical-align: bottom; }
  /* Unregistered players — muted so named (registered) rows stand out. */
  .name.anon { opacity: .5; font-style: italic; }
  .empty { opacity: .5; padding: 2rem 0; text-align: center; }
  footer { margin-top: 1.5rem; opacity: .55; font-size: .82rem; }
  footer p { margin: .35rem 0; }
  footer a { color: inherit; }
  /* "Hide my profile" control — shown only to visitors with a userId. */
  .hide-profile { margin-top: 1.1rem; padding-top: 1rem; border-top: 1px solid color-mix(in srgb, currentColor 12%, transparent); }
  .hide-profile label { display: inline-flex; align-items: flex-start; gap: .5rem; cursor: pointer; }
  .hide-profile input { margin: .15rem 0 0; flex: none; }
  .hide-profile input:disabled + span { opacity: .5; }
  .hide-status { display: block; margin-top: .4rem; min-height: 1.2em; opacity: .85; }
  /* Units selector — a small segmented control above the table. */
  .units { display: inline-flex; margin: 0 0 .9rem; border: 1px solid color-mix(in srgb, currentColor 25%, transparent);
           border-radius: .5rem; overflow: hidden; }
  .units button { font: inherit; font-size: .8rem; padding: .3rem .7rem; border: 0; background: none;
                  color: inherit; cursor: pointer; opacity: .6; }
  .units button + button { border-left: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
  .units button.on { opacity: 1; background: color-mix(in srgb, currentColor 10%, transparent); font-weight: 600; }
  @media (max-width: 420px) {
    body { padding-left: .6rem; padding-right: .6rem; }
  }
</style>
</head>
<body>
<h1 id="lbH1">Leaderboard</h1>
<div class="units" id="unitsBar" role="group" aria-label="Units">
  <button type="button" data-units="om" class="on">OnlyMove</button>
  <button type="button" data-units="egf">EGF</button>
  <button type="button" data-units="aga">AGA</button>
</div>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th class="player" data-sort="name" id="thPlayer">Player</th>
      <th data-sort="ver" id="thVer">Verified</th>
      <th class="om-start" data-sort="est">onlymove</th>
    </tr>
  </thead>
  <tbody id="rows"></tbody>
</table>
<footer>
  ${belowNote}
  ${staleNote}
  <p id="lbInactive">Players not active in the past 7 days are not listed.</p>
  <p id="lbInvite">Want your name shown instead of "Anonymous"? Registration is by invite — ask in <a href="https://discord.gg/ZU7jJNnkbB" target="_blank" rel="noopener">Discord</a>.</p>
  <div id="hideProfile" class="hide-profile" hidden>
    <label>
      <input type="checkbox" id="hideChk" disabled>
      <span id="hideLabel">Hide my profile — withhold my name, country, and declared ranks from this board. My onlymove rank still shows, but anonymously.</span>
    </label>
    <span id="hideStatus" class="hide-status" role="status" aria-live="polite"></span>
  </div>
</footer>
<script>
// Localization prelude. Resolves the viewer's language on the CLIENT (see STR note
// in the module) and rewrites the static text; must run BEFORE the render script so
// the localized column headers become each th's stored data-label. Exposes window.LB
// for the render + hide-profile scripts.
(function () {
  var STR = ${strJson};
  var CFG = { min: ${RANK_START_MIN}, belowMin: ${belowMin}, stale: ${staleVerified} };
  function getLang() {
    var l = navigator.language || 'en';
    if (l.indexOf('zh') === 0) return 'zh';
    if (l.indexOf('ja') === 0) return 'ja';
    if (l.indexOf('ko') === 0) return 'ko';
    return 'en';
  }
  var LANG = getLang();
  function t(key, params) {
    var db = STR[LANG] || STR.en;
    var s = (key in db) ? db[key] : STR.en[key];
    if (params && typeof s === 'string') {
      s = s.replace(/\\{(\\w+)\\}/g, function (m, n) { return (n in params) ? String(params[n]) : m; });
    }
    return s;
  }
  window.LB = { t: t, lang: LANG, cfg: CFG };

  function setText(id, s) { var el = document.getElementById(id); if (el) el.textContent = s; }
  document.documentElement.lang = LANG;
  document.title = 'onlymove — ' + t('leaderboard');
  setText('lbH1', t('leaderboard'));
  setText('thPlayer', t('player'));
  setText('thVer', t('verified'));
  setText('hideLabel', t('hideLabel'));
  var ub = document.getElementById('unitsBar');
  if (ub) ub.setAttribute('aria-label', t('units'));
  var invite = document.getElementById('lbInvite');
  if (invite) {
    invite.innerHTML = t('inviteLine', {
      discord: '<a href="https://discord.gg/ZU7jJNnkbB" target="_blank" rel="noopener">Discord</a>',
    });
  }
  var below = document.getElementById('lbBelow');
  if (below && CFG.belowMin > 0) {
    below.textContent = t(CFG.belowMin === 1 ? 'belowNoteOne' : 'belowNoteMany',
                          { n: CFG.belowMin, min: CFG.min });
  }
  var stale = document.getElementById('lbStale');
  if (stale && CFG.stale > 0) {
    stale.textContent = t(CFG.stale === 1 ? 'staleNoteOne' : 'staleNoteMany', { n: CFG.stale });
  }
  setText('lbInactive', t('inactiveNote'));
})();
</script>
<script>
(function () {
  var DATA = ${json};
  var sortKey = 'est'; // 'name' | 'ver' | 'est'
  var sortDir = -1;    // 1 ascending, -1 descending
  var tbody = document.getElementById('rows');

  // Units selector state: 'om' (onlymove) | 'egf' | 'aga'; persisted per device.
  var UNITS = ['om', 'egf', 'aga'];
  var units = 'om';
  try { var u0 = localStorage.getItem('lbUnits'); if (UNITS.indexOf(u0) >= 0) units = u0; } catch (e) {}

  ${ESC_SNIPPET}

  function nameOf(r) { return r.anon ? window.LB.t('anonymous') : (r.name || ''); }
  // Verified cell: blank in onlymove units (no raw onlymove-frame source datum);
  // in EGF/AGA units the pre-formatted "rating · rank" of that federation.
  function verText(r) {
    if (units === 'om') return '';
    var v = r.ver && r.ver[units === 'egf' ? 'egf' : 'aga'];
    return v ? pctRank(v.pct, v.t) : '—';
  }
  // Every value cell, in every unit mode, renders as "(pct%) rank" — the muted
  // percentage is progress toward the next rank in the selected frame (as on the
  // front-page rank list; for EGF that's the GoR's own convention, for AGA the
  // official floor-band one).
  function pctRank(pct, label) {
    return (typeof pct === 'number'
      ? '<span class="pct">(' + esc(pct) + '%)</span> ' : '') + esc(label);
  }
  function estText(r) {
    if (!r.est) return '—';
    if (units !== 'om') {
      var c = r.est[units];
      if (c) return pctRank(c.pct, c.t);
    }
    return pctRank(r.est.pct, r.est.om);
  }

  function keyOf(th) { return th.getAttribute('data-sort'); }   // 'name' | 'ver' | 'est'
  function sortValue(r) {
    if (sortKey === 'name') return nameOf(r);
    if (sortKey === 'ver') {
      if (units === 'om') return null;
      var v = r.ver && r.ver[units];
      return v ? v.v : null;
    }
    // 'est' — sort by the DISPLAYED frame's own scalar: the om (untimed) and
    // EGF/AGA (time-adjusted) orders differ per user by the personal time dock.
    // A row missing the frame conversion falls back to the untimed scalar (its
    // cell displays the om form then too).
    if (!r.est) return null;
    if (units !== 'om') {
      var c = r.est[units];
      if (c && typeof c.v === 'number') return c.v;
    }
    return r.est.v;
  }
  function cmp(a, b) {
    var va = sortValue(a), vb = sortValue(b);
    var na = (va === null || va === undefined), nb = (vb === null || vb === undefined);
    if (na && nb) return 0;
    if (na) return 1;                         // missing values always sort to the bottom
    if (nb) return -1;
    if (sortKey === 'name') return sortDir * String(va).localeCompare(String(vb));
    return sortDir * (va - vb);
  }

  var headers = Array.prototype.slice.call(document.querySelectorAll('th[data-sort]'));
  headers.forEach(function (th) {
    th.setAttribute('data-label', th.textContent);
    th.addEventListener('click', function () {
      var key = keyOf(th);
      if (key === sortKey) { sortDir = -sortDir; }               // same column -> flip
      else { sortKey = key; sortDir = (key === 'name') ? 1 : -1; }  // names A→Z, ranks high→low
      render();
    });
  });
  function updateHeaders() {
    headers.forEach(function (th) {
      var active = keyOf(th) === sortKey;
      th.classList.toggle('sorted', active);
      th.textContent = th.getAttribute('data-label') + (active ? (sortDir === 1 ? ' ▲' : ' ▼') : '');
    });
  }

  // Fit-to-width: the nowrap table can overflow narrow phone screens. After a
  // render, if the page's natural width exceeds the screen, rewrite the viewport
  // so the browser shows the page zoomed OUT just enough to fit it exactly (the
  // scale is computed against the device width captured before any override).
  // Zoom-out only — when the screen already fits the content, the meta is left
  // alone (scale 1, intended font size); pinch-zoom stays available either way.
  var baseWidth = 0;
  function fitToWidth() {
    var doc = document.documentElement;
    if (!baseWidth) baseWidth = doc.clientWidth;
    var w = doc.scrollWidth;
    if (w <= doc.clientWidth) return;
    var scale = baseWidth / w;
    document.querySelector('meta[name=viewport]')
      .setAttribute('content', 'width=' + w + ', initial-scale=' + scale);
  }

  function render() {
    updateHeaders();
    if (!DATA.length) { tbody.innerHTML = '<tr><td colspan="4" class="empty">' + esc(window.LB.t('empty')) + '</td></tr>'; return; }
    var sorted = DATA.slice().sort(cmp);
    tbody.innerHTML = sorted.map(function (r, i) {
      var estCell = '<td class="val om-start">' + estText(r) + '</td>';
      return '<tr><td class="rank">' + (i + 1) + '</td>' +
        '<td>' + (r.flag ? r.flag + ' ' : '') +
        '<span class="name' + (r.anon ? ' anon' : '') + '">' + esc(nameOf(r)) + '</span></td>' +
        '<td class="decl">' + verText(r) + '</td>' +
        estCell + '</tr>';
    }).join('');
    fitToWidth();
  }

  var unitButtons = Array.prototype.slice.call(document.querySelectorAll('#unitsBar button'));
  unitButtons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var next = btn.getAttribute('data-units');
      if (UNITS.indexOf(next) < 0 || next === units) return;
      units = next;
      try { localStorage.setItem('lbUnits', units); } catch (e) {}
      unitButtons.forEach(function (b) { b.classList.toggle('on', b === btn); });
      render();
    });
  });
  unitButtons.forEach(function (b) { b.classList.toggle('on', b.getAttribute('data-units') === units); });

  render();
})();
</script>
<script>
// "Hide my profile" control. This page doesn't boot the SPA, but it's same-origin
// with it, so the account's userId is readable from localStorage. Shown only when
// a userId exists (a visitor who never used the app has no profile to hide).
(function () {
  var uid = null;
  try { uid = localStorage.getItem('userId'); } catch (e) {}
  if (!uid || !/^[A-Za-z0-9_-]{1,40}$/.test(uid)) return;

  var box = document.getElementById('hideProfile');
  var chk = document.getElementById('hideChk');
  var status = document.getElementById('hideStatus');
  box.hidden = false;

  // Reflect the current server state before enabling the checkbox.
  fetch('/profile/hide?userId=' + encodeURIComponent(uid))
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) { if (j) chk.checked = !!j.hidden; })
    .catch(function () {})
    .then(function () { chk.disabled = false; });

  chk.addEventListener('change', function () {
    var want = chk.checked;
    chk.disabled = true;
    status.textContent = window.LB.t('saving');
    fetch('/profile/hide', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId: uid, hidden: want }),
    })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
      .then(function (j) {
        chk.checked = !!j.hidden;
        status.textContent = window.LB.t(j.hidden ? 'hiddenMsg' : 'visibleMsg');
      })
      .catch(function () {
        chk.checked = !want;   // revert on failure
        status.textContent = window.LB.t('saveError');
      })
      .then(function () { chk.disabled = false; });
  });
})();
</script>
</body>
</html>`;
}
