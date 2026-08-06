// pwa/src/handlers/leaderboard-page.js
//
// GET /leaderboard — a standalone, directly-linkable, server-rendered page (loads
// without booting the tsumego SPA). Computes on read (no cron blob); a short edge
// cache bounds D1 reads under traffic. Lists every player active in the past week
// who has at least RANK_START_MIN solves here (so their onlymove rank is known):
// invite-registered players with their EGD/AGA name, everyone else as "Anonymous".
// Players still below the gate are only counted, in a footer tally.
// A UNITS selector — onlymove, or any ranking system we hold a coupling for
// (RANK_SYSTEMS), persisted in localStorage — picks the frame both value columns
// speak. Every value cell renders as "rank" or "rank (pct%)", the percent being
// the floored progress toward the next rank in the selected frame; the raw GoR /
// AGA rating numbers are never displayed, they only reach here as the shared-scale
// scalars the cells sort by.
//   * Declared — the player's rank on the SELECTED system: their own survey
//     declaration, or the machine heartbeat from a linked federation account,
//     which takes precedence on its own system (EGD GoR on EGF, the AGA Ratings
//     Explorer rating on AGA) and is activity-gated per the EGF criteria in
//     readLeaderboardRows. A declaration is a band label, so it shows the band
//     alone; a heartbeat resolves WITHIN the band and shows the percent too
//     (GoR 2389 -> "4d (39%)"). Blank in onlymove units — an onlymove rank is
//     measured here, never declared.
//   * Predicted — the current estimate from their solving here, coupled into the
//     selected system's frame (onlymove units: the untimed internal rank itself,
//     matching the front-page rank list).
// Any header sorts (cells sort by their underlying continuous scalar).

import { readLeaderboardRows, RANK_START_MIN, RANK_SYSTEMS } from '../leaderboard.js';
import { rankProgress, rankToLabel, clampRankBand } from '../../shared/rank-math.js';
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
// brand name and stays untranslated (it names one of the units options). NOTE: the
// tournament page (Finnish-only) and the invite page (English-only) are deliberate
// exceptions and do NOT share this dict.
const STR = {
    en: {
        leaderboard: 'Weekly Leaderboard',
        activeWindow: '(active in the past {days} days)',
        player: 'Player',
        declared: 'Declared',
        predicted: 'Predicted',
        units: 'Units',
        anonymous: 'Anonymous',
        empty: 'No players yet.',
        belowNoteOne: '1 more active player has fewer than {min} solves and isn\'t shown yet.',
        belowNoteMany: '{n} more active players have fewer than {min} solves and aren\'t shown yet.',
        inviteLine: 'Want your name shown instead of "Anonymous"? Registration is by invite — ask in {discord}.',
        hideLabel: 'Hide my profile — withhold my name, country, and declared ranks from this board. My predicted rank still shows, but anonymously.',
        saving: 'Saving…',
        hiddenMsg: 'Your profile is hidden. The board updates within a minute.',
        visibleMsg: 'Your profile is visible. The board updates within a minute.',
        saveError: 'Could not save — please try again.',
    },
    zh: {
        leaderboard: '每周排行榜',
        activeWindow: '（过去 {days} 天内活跃）',
        player: '玩家',
        declared: '申报',
        predicted: '预测',
        units: '单位',
        anonymous: '匿名',
        empty: '暂无玩家。',
        belowNoteOne: '还有 1 名活跃玩家的解题数不足 {min}，暂未显示。',
        belowNoteMany: '还有 {n} 名活跃玩家的解题数不足 {min}，暂未显示。',
        inviteLine: '想显示您的名字而不是「匿名」？注册需邀请——请在 {discord} 询问。',
        hideLabel: '隐藏我的资料——不在此榜单显示我的名字、国家/地区和申报段位。我的预测段位仍会显示，但为匿名。',
        saving: '正在保存…',
        hiddenMsg: '您的资料已隐藏。榜单将在一分钟内更新。',
        visibleMsg: '您的资料已公开。榜单将在一分钟内更新。',
        saveError: '保存失败——请重试。',
    },
    ja: {
        leaderboard: '週間ランキング',
        activeWindow: '（過去{days}日間に活動）',
        player: 'プレイヤー',
        declared: '申告',
        predicted: '予測',
        units: '単位',
        anonymous: '匿名',
        empty: 'まだプレイヤーがいません。',
        belowNoteOne: 'あと1人、解答数が {min} 未満のアクティブなプレイヤーがいますが、まだ表示されていません。',
        belowNoteMany: 'あと{n}人、解答数が {min} 未満のアクティブなプレイヤーがいますが、まだ表示されていません。',
        inviteLine: '「匿名」の代わりに名前を表示しますか？登録は招待制です——{discord} で聞いてください。',
        hideLabel: 'プロフィールを隠す——このボードで名前・国・申告した段級位を表示しません。予測ランクは匿名で引き続き表示されます。',
        saving: '保存中…',
        hiddenMsg: 'プロフィールは非表示です。ボードは1分以内に更新されます。',
        visibleMsg: 'プロフィールは表示されています。ボードは1分以内に更新されます。',
        saveError: '保存できませんでした——もう一度お試しください。',
    },
    ko: {
        leaderboard: '주간 순위표',
        activeWindow: '(지난 {days}일간 활동)',
        player: '플레이어',
        declared: '신고',
        predicted: '예측',
        units: '단위',
        anonymous: '익명',
        empty: '아직 플레이어가 없습니다.',
        belowNoteOne: '해결한 문제가 {min}개 미만인 활동 플레이어가 1명 더 있으며 아직 표시되지 않습니다.',
        belowNoteMany: '해결한 문제가 {min}개 미만인 활동 플레이어가 {n}명 더 있으며 아직 표시되지 않습니다.',
        inviteLine: '"익명" 대신 이름을 표시하고 싶으신가요? 등록은 초대제입니다 — {discord}에서 문의하세요.',
        hideLabel: '내 프로필 숨기기 — 이 보드에서 내 이름, 국가, 신고한 급/단을 표시하지 않습니다. 예측 랭크는 익명으로 계속 표시됩니다.',
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

// One value cell from a continuous rank on the shared scale: `v` the sort key, `t`
// the rank label, and `pct` the floored progress toward the next rank — the SHARED
// rankProgress decomposition the front-page rank list uses, so a cell always agrees
// with the band it names (for a GoR this IS the EGF convention: 2389 -> "4d (39%)",
// and for an AGA rating its floor convention, since agaRatingToRank has already put
// AGA's bands onto ours). `pct` is omitted — the cell renders as a bare label — when
// the value is a survey declaration, which names a band and says nothing about where
// in it the player sits, so `v` is only that band's centre.
//
// A SERVER frame clamps the LABEL to the real 30k..9d ladder, the same treatment the
// client's rank-translation modal gives a coupled rank, so an extreme extrapolation
// can't print "12 dan"; the onlymove frame is left unclamped there and here.
function cell(v, serverFrame, exact = true) {
    if (!Number.isFinite(v)) return null;
    const p = rankProgress(v, serverFrame ? (b) => rankToLabel(clampRankBand(b)) : rankToLabel);
    return exact ? { v, t: p.label, pct: p.percent } : { v, t: p.label };
}

function renderPage(rows, belowMin) {
    // Per row, two maps keyed by unit mode ('om' or a RANK_SYSTEMS name), each entry
    // a cell as above:
    //   `est`  — the onlymove estimate. 'om' is the UNTIMED internal rank; each
    //            system's entry is the TIME-ADJUSTED coupling into that frame. Every
    //            frame carries its OWN sort value, because the untimed and
    //            time-adjusted orders genuinely differ per user by the personal time
    //            dock — so switching units must re-sort by the displayed frame.
    //   `decl` — the player's rank on that system (declaration, or the linked
    //            federation heartbeat where there is one — see readLeaderboardRows).
    //            No 'om' entry: an onlymove rank is measured, never declared.
    const data = rows.map((r) => {
        const est = {};
        const om = cell(r.currentInternal, false);
        if (om) {
            est.om = om;
            for (const s of RANK_SYSTEMS) {
                const c = cell(r.est[s], true);
                if (c) est[s] = c;
            }
        }
        const decl = {};
        for (const s of RANK_SYSTEMS) {
            const d = r.decl[s];
            const c = d ? cell(d.v, true, d.exact) : null;
            if (c) decl[s] = c;
        }
        return {
            // Anonymous (unregistered) rows carry anon: true and no name — the client
            // renders a muted "Anonymous" label for them.
            name: r.registered ? (r.name || '—') : null,
            anon: !r.registered,
            flag: flag(r.country) || UNKNOWN_FLAG,
            decl,
            est,
        };
    });
    const json = JSON.stringify(data).replace(/</g, '\\u003c');
    const strJson = JSON.stringify(STR).replace(/</g, '\\u003c');
    // The unit modes, in selector order: onlymove first, then every coupled system.
    const unitsJson = JSON.stringify(['om', ...RANK_SYSTEMS]).replace(/</g, '\\u003c');
    // System names are brand names and go untranslated, so the <option> list is
    // rendered server-side rather than through the localization prelude.
    const unitOptions = ['<option value="om">OnlyMove</option>']
        .concat(RANK_SYSTEMS.map((s) => `<option value="${s}">${s}</option>`))
        .join('\n  ');
    // Footer tally: active players still below the list-entry gate (no per-user
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
<title>onlymove — Weekly Leaderboard</title>
<style>
  ${baseCss(760)}
  /* The activity window, inline after the title — the board IS the past week, so
     it belongs in the heading rather than as a footnote about who is missing. */
  h1 .hnote { font-size: .62em; font-weight: 400; opacity: .55; white-space: nowrap; }
  th, td { padding: .5rem .5rem; text-align: right; white-space: nowrap; }
  td:nth-child(2) { text-align: left; width: 99%; }
  th.player { text-align: left; }
  thead th { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; opacity: .55; font-weight: 600;
             border-bottom: 1px solid; }
  /* vertical divider separating the declared rank from the predicted one */
  th.pred-start, td.pred-start { border-left: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
  thead th[data-sort] { cursor: pointer; user-select: none; -webkit-user-select: none; }
  thead th[data-sort]:hover { opacity: .85; }
  thead th.sorted { opacity: .95; }
  td.rank { opacity: .5; font-variant-numeric: tabular-nums; }
  td.val { font-variant-numeric: tabular-nums; font-weight: 600; }
  /* progress-to-next-rank "(64%)" — same relative size/mute as the front-page
     rank list's percentage span (rank-translation.js: 0.8em, opacity .6).
     Both value columns render it (pctRank), so scope on the class, not on
     td.val — the Declared cell is a td.decl. */
  .pct { font-size: .8em; opacity: .6; font-weight: 400; }
  /* A cell with no percent to show still RESERVES its width, so the rank labels
     stay a column — across rows within a system and across systems alike. */
  .pct.blank { visibility: hidden; }
  /* The percent digits sit in a fixed two-digit box, right-aligned. The cells are
     right-aligned, so without it a one-digit percent shortens the whole string and
     pushes that row's rank label — and the parentheses — a digit out of column. */
  .pctn { display: inline-block; min-width: 2ch; text-align: right; font-variant-numeric: tabular-nums; }
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
  /* Units selector — a dropdown above the table. There are as many options as
     there are ranking systems we couple to, so this is a <select> rather than a
     segmented control: the list grows with rank_map.json without ever wrapping.
     Colours are the FORM-CONTROL system pair (Field/FieldText), not inherit +
     transparent: the popup list is painted by the OS, which leaves it light in
     dark mode, so inheriting the page's light-on-dark text renders the open list
     white on white. The system pair flips with the theme on both parts. */
  .units { margin: 0 0 .9rem; }
  .units select { font: inherit; font-size: .8rem; padding: .25rem 1.4rem .25rem .5rem;
                  color: FieldText; background: Field;
                  border: 1px solid color-mix(in srgb, currentColor 25%, transparent);
                  border-radius: .5rem; cursor: pointer; }
  .units select option { color: FieldText; background: Field; }
  @media (max-width: 420px) {
    body { padding-left: .6rem; padding-right: .6rem; }
  }
</style>
</head>
<body>
<h1><span id="lbH1">Weekly Leaderboard</span> <span class="hnote" id="lbActive">(active in the past ${ACTIVE_DAYS} days)</span></h1>
<div class="units">
  <select id="unitsSel" aria-label="Units">
  ${unitOptions}
  </select>
</div>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th class="player" data-sort="name" id="thPlayer">Player</th>
      <th data-sort="decl" id="thDecl">Declared</th>
      <th class="pred-start" data-sort="est" id="thPred">Predicted</th>
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
      <span id="hideLabel">Hide my profile — withhold my name, country, and declared ranks from this board. My predicted rank still shows, but anonymously.</span>
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
  var CFG = { min: ${RANK_START_MIN}, belowMin: ${belowMin}, activeDays: ${ACTIVE_DAYS} };
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
  setText('lbActive', t('activeWindow', { days: CFG.activeDays }));
  setText('thPlayer', t('player'));
  setText('thDecl', t('declared'));
  setText('thPred', t('predicted'));
  setText('hideLabel', t('hideLabel'));
  var us = document.getElementById('unitsSel');
  if (us) us.setAttribute('aria-label', t('units'));   // the only remaining use of the units string
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
  var sortKey = 'est'; // 'name' | 'decl' | 'est'
  var sortDir = -1;    // 1 ascending, -1 descending
  var tbody = document.getElementById('rows');

  // Units selector state: 'om' (onlymove) or a ranking-system name; persisted per
  // device. The two units that predate the any-system selector were stored
  // lowercase, so map those forward rather than resetting the reader to onlymove.
  var UNITS = ${unitsJson};
  var units = 'om';
  try {
    var u0 = localStorage.getItem('lbUnits');
    if (u0 === 'egf') u0 = 'EGF'; else if (u0 === 'aga') u0 = 'AGA';
    if (UNITS.indexOf(u0) >= 0) units = u0;
  } catch (e) {}

  ${ESC_SNIPPET}

  function nameOf(r) { return r.anon ? window.LB.t('anonymous') : (r.name || ''); }
  // Declared cell: blank in onlymove units (an onlymove rank is measured, never
  // declared); otherwise this player's rank on the selected system, em-dash when
  // they have neither a declaration nor a linked account there.
  // The em-dash goes through pctRank too (as a label with no percent), so it sits
  // in the rank column rather than where the percent would be.
  function declText(r) {
    if (units === 'om') return '';
    var c = r.decl && r.decl[units];
    return c ? pctRank(c.pct, c.t) : pctRank(null, '—');
  }
  // A value cell renders as "rank", plus a muted "(pct%)" — progress toward the
  // next rank in the selected frame — whenever the underlying value resolves
  // within the band (see cell()).
  //
  // The parenthetical is ALWAYS emitted, merely hidden when there is no percent to
  // show, and its width is fixed (.pctn holds two digit cells). So the rank labels
  // line up as a column in three situations that would otherwise each break the
  // alignment of a RIGHT-aligned cell: a declaration (a bare band) next to a
  // verified rating (which resolves within the band) in the same system; an em-dash
  // where a player has no rank on the selected system; and — because the reserved
  // space does not depend on whether any row in the system HAS a percent — the
  // switch between systems, where the column would else shift by the width of the
  // parenthetical whenever the selected system happened to have no verified rows.
  function pctRank(pct, label) {
    var has = typeof pct === 'number';
    return esc(label) + ' <span class="pct' + (has ? '' : ' blank') + '">(<span class="pctn">'
      + esc(has ? pct : 0) + '</span>%)</span>';
  }
  // A row with no conversion for the selected system falls back to the onlymove
  // frame, so the column is never blank for a ranked player.
  function estCellOf(r) { return (r.est && (r.est[units] || r.est.om)) || null; }
  function estText(r) {
    var c = estCellOf(r);
    return c ? pctRank(c.pct, c.t) : pctRank(null, '—');
  }

  function keyOf(th) { return th.getAttribute('data-sort'); }   // 'name' | 'decl' | 'est'
  function sortValue(r) {
    if (sortKey === 'name') return nameOf(r);
    if (sortKey === 'decl') {
      if (units === 'om') return null;
      var c = r.decl && r.decl[units];
      return c ? c.v : null;
    }
    // 'est' — sort by the DISPLAYED frame's own scalar: the om (untimed) and the
    // per-system (time-adjusted) orders differ per user by the personal time dock.
    var e = estCellOf(r);
    return e ? e.v : null;
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
      return '<tr><td class="rank">' + (i + 1) + '</td>' +
        '<td>' + (r.flag ? r.flag + ' ' : '') +
        '<span class="name' + (r.anon ? ' anon' : '') + '">' + esc(nameOf(r)) + '</span></td>' +
        '<td class="decl">' + declText(r) + '</td>' +
        '<td class="val pred-start">' + estText(r) + '</td></tr>';
    }).join('');
    fitToWidth();
  }

  var unitsSel = document.getElementById('unitsSel');
  unitsSel.value = units;
  unitsSel.addEventListener('change', function () {
    var next = unitsSel.value;
    if (UNITS.indexOf(next) < 0 || next === units) return;
    units = next;
    try { localStorage.setItem('lbUnits', units); } catch (e) {}
    render();
  });

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
