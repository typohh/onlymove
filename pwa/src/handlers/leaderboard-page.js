// pwa/src/handlers/leaderboard-page.js
//
// GET /leaderboard — a standalone, directly-linkable, server-rendered page (loads
// without booting the tsumego SPA). Computes on read (no cron blob); a short edge
// cache bounds D1 reads under traffic. Lists every player active in the past week
// who has at least RANK_START_MIN solves here (so their onlymove rank is known):
// invite-registered players with their EGD/AGA name, everyone else as "Anonymous".
// Players still below the gate are only counted, in a footer tally.
// A UNITS selector (onlymove / EGF / AGA, persisted in localStorage) picks the
// frame both value columns speak:
//   * Verified — the official rating of the player's linked federation account,
//     from the machine heartbeats only (EGD GoR heartbeat / AGA Ratings Explorer
//     heartbeat), activity-gated per the EGF criteria in readLeaderboardRows.
//     EGF units show the GoR + official grade ("2830 · 4d"), AGA units the AGA
//     rating + official rank ("5.1 · 5d"); onlymove units leave the column blank
//     (there is no raw onlymove-frame source datum). Human survey declarations
//     are NOT revealed here (they feed only calibration + the flag fallback).
//   * onlymove — the current estimate from their solving here: onlymove units
//     show the internal rank label ("3k"), EGF/AGA units the estimate converted
//     to that server's frame in the same rating+rank format as Verified.
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
        sub: 'Every player active in the past week with at least {min} solves, ordered by their onlymove rank estimate. "Verified" is the official rating of the player\'s linked EGF or AGA account, shown in the selected units and only while tournament-active.',
        player: 'Player',
        verified: 'Verified',
        units: 'Units',
        anonymous: 'Anonymous',
        empty: 'No players yet.',
        belowNoteOne: '1 more active player has fewer than {min} solves and isn\'t shown yet.',
        belowNoteMany: '{n} more active players have fewer than {min} solves and aren\'t shown yet.',
        inviteLine: 'Want your name shown instead of "Anonymous"? Registration is by invite — ask in {discord}.',
        hideLabel: 'Hide my profile — withhold my name, country, and verified rating from this board. My onlymove rank still shows, but anonymously.',
        saving: 'Saving…',
        hiddenMsg: 'Your profile is hidden. The board updates within a minute.',
        visibleMsg: 'Your profile is visible. The board updates within a minute.',
        saveError: 'Could not save — please try again.',
    },
    zh: {
        leaderboard: '排行榜',
        sub: '过去一周内至少解出 {min} 题的所有活跃玩家，按其 onlymove 段位估计排序。「已验证」是玩家绑定的 EGF/AGA 账号的官方等级分，以所选单位显示，且仅限近期参加过比赛的棋手。',
        player: '玩家',
        verified: '已验证',
        units: '单位',
        anonymous: '匿名',
        empty: '暂无玩家。',
        belowNoteOne: '还有 1 名活跃玩家的解题数不足 {min}，暂未显示。',
        belowNoteMany: '还有 {n} 名活跃玩家的解题数不足 {min}，暂未显示。',
        inviteLine: '想显示您的名字而不是「匿名」？注册需邀请——请在 {discord} 询问。',
        hideLabel: '隐藏我的资料——不在此榜单显示我的名字、国家/地区和已验证等级分。我的 onlymove 段位仍会显示，但为匿名。',
        saving: '正在保存…',
        hiddenMsg: '您的资料已隐藏。榜单将在一分钟内更新。',
        visibleMsg: '您的资料已公开。榜单将在一分钟内更新。',
        saveError: '保存失败——请重试。',
    },
    ja: {
        leaderboard: 'ランキング',
        sub: '過去1週間に少なくとも {min} 問解いたアクティブなプレイヤー全員を、onlymove ランク推定値順に並べています。「認証済み」は、プレイヤーが連携した EGF / AGA アカウントの公式レーティングで、選択した単位で表示され、最近大会に参加しているプレイヤーのみ表示されます。',
        player: 'プレイヤー',
        verified: '認証済み',
        units: '単位',
        anonymous: '匿名',
        empty: 'まだプレイヤーがいません。',
        belowNoteOne: 'あと1人、解答数が {min} 未満のアクティブなプレイヤーがいますが、まだ表示されていません。',
        belowNoteMany: 'あと{n}人、解答数が {min} 未満のアクティブなプレイヤーがいますが、まだ表示されていません。',
        inviteLine: '「匿名」の代わりに名前を表示しますか？登録は招待制です——{discord} で聞いてください。',
        hideLabel: 'プロフィールを隠す——このボードで名前・国・認証済みレーティングを表示しません。onlymove ランクは匿名で引き続き表示されます。',
        saving: '保存中…',
        hiddenMsg: 'プロフィールは非表示です。ボードは1分以内に更新されます。',
        visibleMsg: 'プロフィールは表示されています。ボードは1分以内に更新されます。',
        saveError: '保存できませんでした——もう一度お試しください。',
    },
    ko: {
        leaderboard: '순위표',
        sub: '지난 한 주 동안 최소 {min}문제를 푼 모든 활동 플레이어를 onlymove 랭크 추정치 순으로 정렬했습니다. "인증됨"은 플레이어가 연결한 EGF/AGA 계정의 공식 레이팅으로, 선택한 단위로 표시되며 최근 대회 활동이 있는 경우에만 표시됩니다.',
        player: '플레이어',
        verified: '인증됨',
        units: '단위',
        anonymous: '익명',
        empty: '아직 플레이어가 없습니다.',
        belowNoteOne: '해결한 문제가 {min}개 미만인 활동 플레이어가 1명 더 있으며 아직 표시되지 않습니다.',
        belowNoteMany: '해결한 문제가 {min}개 미만인 활동 플레이어가 {n}명 더 있으며 아직 표시되지 않습니다.',
        inviteLine: '"익명" 대신 이름을 표시하고 싶으신가요? 등록은 초대제입니다 — {discord}에서 문의하세요.',
        hideLabel: '내 프로필 숨기기 — 이 보드에서 내 이름, 국가, 인증된 레이팅을 표시하지 않습니다. onlymove 랭크는 익명으로 계속 표시됩니다.',
        saving: '저장 중…',
        hiddenMsg: '프로필이 숨겨졌습니다. 보드는 1분 이내에 업데이트됩니다.',
        visibleMsg: '프로필이 표시됩니다. 보드는 1분 이내에 업데이트됩니다.',
        saveError: '저장하지 못했습니다 — 다시 시도해 주세요.',
    },
};

export async function handleLeaderboard(env) {
    let rows = [], belowMin = 0;
    try {
        ({ rows, belowMin } = await readLeaderboardRows(env, { activeDays: ACTIVE_DAYS }));
    } catch (e) {
        console.error('leaderboard read failed:', e);
    }
    return new Response(renderPage(rows, belowMin), {
        status: 200,
        headers: {
            'Content-Type': 'text/html; charset=utf-8',
            'Cache-Control': 'public, max-age=60',
        },
    });
}

// Continuous internal AGA rank scalar (1k = 0, 1d = 1) -> an AGA-style rating
// number: dan region maps 1:1 (scalar 1.5 -> 1.5), kyu region shifts past the
// AGA (-1, 1) gap (scalar 0 = 1k -> -1.0), and the sub-1-band (0, 1) interval
// interpolates linearly across the gap. Display-only (the estimate in AGA units).
function agaScalarToRating(s) {
    return s >= 1 ? s : s <= 0 ? s - 1 : 2 * s - 1;
}
const fmtAgaNum = (r) => (Math.round(r * 10) / 10).toFixed(1);
// AGA official rank from the rating (dan = floor(rating), kyu = floor(|rating|)) —
// fallback when a heartbeat row predates the stored rank_label.
const agaLabelFromRating = (r) => (r >= 1 ? Math.floor(r) + 'd' : Math.floor(-r) + 'k');

function renderPage(rows, belowMin) {
    // Per row: `est` = the onlymove estimate in the three unit frames — `v` the
    // UNTIMED internal scalar (the sort key + the onlymove-mode label source),
    // `om` its rank label, and `egf`/`aga` the TIME-ADJUSTED server-frame
    // conversions as { n: rating number, t: rank label } (the frames the verified
    // ratings natively speak — see readLeaderboardRows). `ver` = the verified
    // official data per federation, pre-formatted: { v: sort scalar, t: display }.
    const data = rows.map((r) => {
        let est = null;
        if (Number.isFinite(r.currentInternal)) {
            // om units show "(percent) rank" — the same rank + progress-to-next-rank
            // decomposition the front-page rank list uses (shared rankProgress),
            // percent-first so the format matches the "rating · rank" of EGF/AGA units.
            const p = rankProgress(r.currentInternal);
            est = { v: r.currentInternal, om: p.label, pct: p.percent };
            if (Number.isFinite(r.currentEgf)) {
                est.egf = {
                    n: String(Math.max(100, Math.round(2000 + 100 * r.currentEgf))),
                    t: rankScalarToLabel(r.currentEgf),
                };
            }
            if (Number.isFinite(r.currentAga)) {
                est.aga = {
                    n: fmtAgaNum(agaScalarToRating(r.currentAga)),
                    t: rankScalarToLabel(r.currentAga),
                };
            }
        }
        const ver = {};
        if (r.egd) {
            const label = r.egd.label || rankScalarToLabel(gorToRankScalar(r.egd.gor));
            ver.egf = { v: r.egd.gor, t: r.egd.gor + ' · ' + label };
        }
        if (r.aga) {
            const label = r.aga.label || agaLabelFromRating(r.aga.rating);
            ver.aga = { v: r.aga.rating, t: fmtAgaNum(r.aga.rating) + ' · ' + label };
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
    // Footer tally of active players still below the list-entry gate (no per-user
    // sample count is shown; this aggregate is the only nod to them). The English
    // text here is the no-JS default; the prelude re-renders it localized.
    const belowNote = belowMin
        ? `<p id="lbBelow">${belowMin} more active player${belowMin === 1 ? '' : 's'} ${belowMin === 1 ? 'has' : 'have'} fewer than ${RANK_START_MIN} solves and ${belowMin === 1 ? "isn't" : "aren't"} shown yet.</p>`
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
     rank list's percentage span (rank-translation.js: 0.8em, opacity .6). */
  td.val .pct { font-size: .8em; opacity: .6; font-weight: 400; }
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
<p class="sub" id="lbSub">Every player active in the past week with at least ${RANK_START_MIN} solves, ordered by their onlymove rank estimate. "Verified" is the official rating of the player's linked EGF or AGA account, shown in the selected units and only while tournament-active.</p>
<div class="units" id="unitsBar" role="group" aria-label="Units">
  <button type="button" data-units="om" class="on">onlymove</button>
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
  var CFG = { min: ${RANK_START_MIN}, belowMin: ${belowMin} };
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
  setText('lbSub', t('sub', { min: CFG.min }));
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
    return v ? esc(v.t) : '—';
  }
  // Estimate cell: onlymove units show "(percent) rank" — progress to the next
  // rank, as on the front-page rank list, percent-first to match the rating-first
  // EGF/AGA format; EGF/AGA units show the converted "rating · rank" (falling
  // back to the onlymove form if a coupling is unavailable).
  function omText(e) {
    return (typeof e.pct === 'number'
      ? '<span class="pct">(' + esc(e.pct) + '%)</span> ' : '') + esc(e.om);
  }
  function estText(r) {
    if (!r.est) return '—';
    if (units !== 'om') {
      var c = r.est[units];
      if (c) return esc(c.n + ' · ' + c.t);
    }
    return omText(r.est);
  }

  function keyOf(th) { return th.getAttribute('data-sort'); }   // 'name' | 'ver' | 'est'
  function sortValue(r) {
    if (sortKey === 'name') return nameOf(r);
    if (sortKey === 'ver') {
      if (units === 'om') return null;
      var v = r.ver && r.ver[units];
      return v ? v.v : null;
    }
    return r.est ? r.est.v : null;              // 'est' — the current estimate
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
