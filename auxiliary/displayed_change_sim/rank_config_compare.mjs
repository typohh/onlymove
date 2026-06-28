// Compare 3 windowMLE configurations on REAL user histories, with the shipped
// linear-info-ramp window faithfully re-implemented (validated to ~1e-13 vs the
// engine in rank_mechanism_diag.mjs):
//   baseline : cap 500, info ramp w=1−acc/16  (shipped — boundary sample leaves at ~0.93 w)
//   expand   : cap 3500, info ramp w=1−acc/16 (ramp can decay before the cap; few hit it)
//   linpos   : cap 500, POSITIONAL ramp w=(M−j)/M, M=min(#att,500)  (oldest in-window
//              leaves at 1/M≈0 BY CONSTRUCTION — kills the boundary artifact without widening the cap)
//
// Reports, pooled over users: the correct→down / wrong→up contradiction (the thing
// the edge causes), plus the COST proxies — mean |Δrank| per attempt (volatility)
// and the median posterior SE (precision) and effective window N.
//
// Run: node rank_config_compare.mjs [user1,user2,...]
import { readFileSync } from 'fs';
import vm from 'vm';
import path from 'path';
import { pathToFileURL } from 'url';

const REPO = '/home/ootakamoku/Repositories/onlymove';
const HERE = path.join(REPO, 'auxiliary', 'displayed_change_sim');
vm.runInThisContext(readFileSync(path.join(REPO, 'pwa', 'lib', 'math.min.js'), 'utf8'));
const E = await import(pathToFileURL(path.join(HERE, 'engine.bundle.mjs')).href);
E.loadPopulationConstants();

const POP = JSON.parse(readFileSync(path.join(REPO, 'pwa', 'tsumegos', 'population_constants.json'), 'utf8'));
const FLOOR_MAX = 0.78, CAL_P = 0.8, TOTAL_RANKS = 29;
const ff = POP.fat_finger_rate;
const LAPSE = (typeof ff === 'number' && ff >= 0 && ff < 1 - FLOOR_MAX) ? ff : 0.0026;
const WIN_PRIOR_SD = 4, WIN_PRIOR_INFO = 1 / (WIN_PRIOR_SD * WIN_PRIOR_SD);
const X_LO = -15, X_HI = 45, SLOPE_LO = 0.05, SLOPE_HI = 3.0, MSLOPE_FLOOR = 1e-3, SLOPE_FALLBACK = 0.28, A_REF = 1.6;

const sigmoid = z => 1 / (1 + Math.exp(-z));
const clampFloor = c => Math.min(Math.max(typeof c === 'number' ? c : 0, 0), FLOOR_MAX);
const clearSpan = c => Math.max(0, 1 - Math.max(c, 0) - LAPSE);
const clearKappa = c => { const cc = Math.max(c, 0); const q = Math.min(Math.max((CAL_P - cc) / (1 - cc - LAPSE), 1e-6), 1 - 1e-6); return Math.log(q / (1 - q)); };
const applyM = (rating, size) => E.normalizedRankForSize(rating, size) * TOTAL_RANKS;
const Mslope = (rating, size) => { const h = 1e-3; return (applyM(rating + h, size) - applyM(rating - h, size)) / (2 * h); };
const clampSlope = s => (!Number.isFinite(s) || s <= 0) ? SLOPE_FALLBACK : Math.min(SLOPE_HI, Math.max(SLOPE_LO, s));
function makeAttempt(rating, slopeAp, floor, size, correct) {
  const x = applyM(rating, size);
  const slope = clampSlope((slopeAp > 0 ? slopeAp : A_REF) / Math.max(Math.abs(Mslope(rating, size)), MSLOPE_FLOOR));
  const flr = clampFloor(floor);
  return { x, slope, floor: flr, kappa: clearKappa(flr), correct };
}

// windowMLE parameterized by {cap, itarget, mode}; returns {R, se, effN, departW}.
//   mode 'info'   : shipped Fisher-information ramp  w = 1 − acc/itarget
//   mode 'linpos' : positional linear ramp  w = (M − posFromNewest)/M, M=min(#att,cap)
//                   — the oldest in-window sample leaves at 1/M ≈ 0 BY CONSTRUCTION,
//                   so the low-info cap-boundary artifact cannot occur.
function windowMLE(att, refR, cfg) {
  const { cap, itarget, mode } = cfg;
  const mWin = Math.min(att.length, cap);   // ramp spans the AVAILABLE window
  const win = []; let acc = 0;
  for (let j = att.length - 1; j >= 0 && win.length < cap; j--) {
    let w;
    if (mode === 'linpos') {
      w = (mWin - win.length) / mWin;
    } else {
      w = 1 - acc / itarget;
      if (w <= 0) break;
    }
    const a = att[j]; win.push({ a, w });
    const span = clearSpan(a.floor);
    const praw = sigmoid(a.slope * (refR - a.x) + a.kappa);
    let p = a.floor + span * praw; p = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
    const g = span * a.slope * praw * (1 - praw);
    acc += g * g / (p * (1 - p));
  }
  let sw = 0, sw2 = 0; for (const { w } of win) { sw += w; sw2 += w * w; }
  const effN = sw2 > 0 ? (sw * sw) / sw2 : 0;
  const departW = win.length ? win[win.length - 1].w : 0;
  let R = refR, info = WIN_PRIOR_INFO;
  for (let it = 0; it < 15; it++) {
    let score = WIN_PRIOR_INFO * (refR - R); info = WIN_PRIOR_INFO;
    for (const { a, w } of win) {
      const span = clearSpan(a.floor);
      const praw = sigmoid(a.slope * (R - a.x) + a.kappa);
      let p = a.floor + span * praw; p = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
      const g = span * a.slope * praw * (1 - praw);
      score += w * (a.correct - p) * g / (p * (1 - p));
      info += w * g * g / (p * (1 - p));
    }
    if (info < 1e-9) break;
    let step = score / info; if (step > 2) step = 2; else if (step < -2) step = -2;
    R += step; if (R < X_LO) R = X_LO; else if (R > X_HI) R = X_HI;
    if (Math.abs(step) < 1e-6) break;
  }
  return { R, se: info > 0 ? 1 / Math.sqrt(info) : Infinity, effN, departW };
}

// Three levers on the cap-edge weight. The shipped INFO ramp can't drive it to 0 in
// the served-at-rank regime (only ~0.8 info accumulates across all 500 samples, so the
// edge weight 1−0.8/16 ≈ 0.95 stays high — the contradiction). The two fixes: `expand`
// raises the cap so the ramp has more samples to decay over; `linpos` replaces the info
// ramp with a positional one whose edge is 1/cap≈0 regardless of information — fixing the
// edge at the same 500 cap, at the cost of down-weighting the whole window (effN ≈ 2cap/3).
const CONFIGS = [
  { label: 'baseline 500/16', cap: 500, itarget: 16, mode: 'info' },
  { label: 'expand  3500/16', cap: 3500, itarget: 16, mode: 'info' },
  { label: 'linpos  500', cap: 500, itarget: 0, mode: 'linpos' },
];

const data = JSON.parse(readFileSync(path.join(HERE, 'sim_input_new.json'), 'utf8'));
const problems = data.problems;
const users = process.argv[2] ? process.argv[2].split(',') : Object.keys(data.users);

const q = (arr, p) => { const s = [...arr].sort((a, b) => a - b); return s.length ? s[Math.floor(p * (s.length - 1))] : NaN; };
const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : NaN;

console.log(`\nReal-data window-config comparison — users: ${users.join(', ')}\n`);
console.log(`${'config'.padEnd(16)}${'corr→down'.padStart(10)}${'wrong→up'.padStart(10)}${'mean|Δrank|'.padStart(12)}${'medEffN'.padStart(9)}${'medSE'.padStart(8)}${'medDepartW'.padStart(11)}`);
console.log('-'.repeat(76));

for (const cfg of CONFIGS) {
  let cAll = [], wAll = [], deltas = [], effNs = [], ses = [], departWs = [];
  let corrDown = 0, nCorr = 0, wrongUp = 0, nWrong = 0;
  for (const uid of users) {
    const att = []; let R = 0;
    for (const [tid, succ, size, , , isr] of data.users[uid]) {
      if (isr === 1) continue;
      const m = problems[tid];
      const a = makeAttempt(m[0], m[1], m[2], size, succ === 1);
      const refR = R;
      att.push(a);
      const r = windowMLE(att, refR, cfg);
      const total = r.R - refR;
      deltas.push(Math.abs(total)); effNs.push(r.effN); ses.push(r.se);
      if (att.length >= cfg.cap) departWs.push(r.departW);
      if (succ === 1) { nCorr++; if (total < 0) corrDown++; }
      else { nWrong++; if (total > 0) wrongUp++; }
      R = r.R;
    }
  }
  const pc = (a, b) => b ? (100 * a / b).toFixed(1) + '%' : '–';
  console.log(`${cfg.label.padEnd(16)}${pc(corrDown, nCorr).padStart(10)}${pc(wrongUp, nWrong).padStart(10)}` +
    `${mean(deltas).toFixed(4).padStart(12)}${q(effNs, .5).toFixed(0).padStart(9)}${q(ses, .5).toFixed(3).padStart(8)}` +
    `${(departWs.length ? q(departWs, .5).toFixed(2) : '–').padStart(11)}`);
}
console.log('-'.repeat(76));
console.log(`corr→down/wrong→up = contradiction (lower better).  mean|Δrank| = per-attempt volatility (stones).`);
console.log(`medSE = posterior precision (lower = tighter/more stable).  medDepartW = weight of the sample aging`);
console.log(`out at the cap edge (baseline ~0.93 = hard drop; a soft fix wants this ~0).\n`);
