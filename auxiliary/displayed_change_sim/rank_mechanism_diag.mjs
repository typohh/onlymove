// Why does a CORRECT challenge answer sometimes tick the rank DOWN, when problems
// are served at the user's rank (P≈0.8)? This faithfully re-implements the shipped
// windowMLE (rank-decomposition.js) and decomposes each first-time attempt's rank
// change into:
//   own   = the new attempt's OWN evidence pull (R_after − R_after|newest-zeroed)
//   shift = everything else (R_after|newest-zeroed − R_before): the 500-cap drop +
//           gradual re-weight
// and separately measures the 500-cap boundary drop and the effective window size.
//
// Run: node rank_mechanism_diag.mjs <user> [sinceISO]   (user is mandatory)
import { readFileSync } from 'fs';
import vm from 'vm';
import path from 'path';
import { pathToFileURL } from 'url';

const REPO = '/home/ootakamoku/Repositories/onlymove';
const HERE = path.join(REPO, 'auxiliary', 'displayed_change_sim');
vm.runInThisContext(readFileSync(path.join(REPO, 'pwa', 'lib', 'math.min.js'), 'utf8'));
const E = await import(pathToFileURL(path.join(HERE, 'engine.bundle.mjs')).href);
E.loadPopulationConstants();

const USER = process.argv[2];
if (!USER) {
    console.error('usage: node rank_mechanism_diag.mjs <user> [sinceISO]  '
        + '(pick a user id from the local DB)');
    process.exit(1);
}
const SINCE_MS = process.argv[3] ? Date.parse(process.argv[3]) : Date.parse('2026-06-22');

// --- shipped irt.js constants/helpers (population fat_finger_rate) ---
const POP = JSON.parse(readFileSync(path.join(REPO, 'pwa', 'tsumegos', 'population_constants.json'), 'utf8'));
const FLOOR_MAX = 0.78, CAL_P = 0.8;
const ff = POP.fat_finger_rate;
const LAPSE = (typeof ff === 'number' && ff >= 0 && ff < 1 - FLOOR_MAX) ? ff : 0.0026;
const sigmoid = z => 1 / (1 + Math.exp(-z));
const clampFloor = c => Math.min(Math.max(typeof c === 'number' ? c : 0, 0), FLOOR_MAX);
const clearSpan = c => Math.max(0, 1 - Math.max(c, 0) - LAPSE);
const clearKappa = c => { const cc = Math.max(c, 0); const q = Math.min(Math.max((CAL_P - cc) / (1 - cc - LAPSE), 1e-6), 1 - 1e-6); return Math.log(q / (1 - q)); };

// --- shipped rank-decomposition.js constants ---
const WIN_ITARGET = 16, WIN_CAP = 500;
const WIN_PRIOR_SD = 4, WIN_PRIOR_INFO = 1 / (WIN_PRIOR_SD * WIN_PRIOR_SD);
const X_LO = -15, X_HI = 45, SLOPE_LO = 0.05, SLOPE_HI = 3.0, MSLOPE_FLOOR = 1e-3, SLOPE_FALLBACK = 0.28, A_REF = 1.6;

const applyM = (rating, size) => E.rankForSize(rating, size);
function Mslope(rating, size) { const h = 1e-3; return (applyM(rating + h, size) - applyM(rating - h, size)) / (2 * h); }
function clampSlope(s) { if (!Number.isFinite(s) || s <= 0) return SLOPE_FALLBACK; return Math.min(SLOPE_HI, Math.max(SLOPE_LO, s)); }

function makeAttempt(rating, slopeAp, floor, size, correct) {
  const x = applyM(rating, size);
  const slope = clampSlope((slopeAp > 0 ? slopeAp : A_REF) / Math.max(Math.abs(Mslope(rating, size)), MSLOPE_FLOOR));
  const flr = clampFloor(floor);
  return { x, slope, floor: flr, kappa: clearKappa(flr), correct };
}

// Faithful windowMLE with options: cap, and zeroNewest (zero the newest attempt's
// LIKELIHOOD weight while keeping it in the ramp). Returns {R, effN, departW,
// departCorrect, windowLen, rampHitZero}.
function windowMLE(att, refR, { cap = WIN_CAP, zeroNewest = false } = {}) {
  const win = []; let acc = 0; let rampHitZero = false;
  for (let j = att.length - 1; j >= 0 && win.length < cap; j--) {
    const w = 1 - acc / WIN_ITARGET;
    if (w <= 0) { rampHitZero = true; break; }
    const a = att[j];
    win.push({ a, w });
    const span = clearSpan(a.floor);
    const praw = sigmoid(a.slope * (refR - a.x) + a.kappa);
    let p = a.floor + span * praw; p = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
    const g = span * a.slope * praw * (1 - praw);
    acc += g * g / (p * (1 - p));
  }
  // effective N of the ramp weights, and the departing (oldest in-window) sample.
  let sw = 0, sw2 = 0;
  for (const { w } of win) { sw += w; sw2 += w * w; }
  const effN = sw2 > 0 ? (sw * sw) / sw2 : 0;
  const oldest = win[win.length - 1];
  const departW = oldest ? oldest.w : 0;
  const departCorrect = oldest ? oldest.a.correct : null;

  let R = refR;
  for (let it = 0; it < 15; it++) {
    let score = WIN_PRIOR_INFO * (refR - R), info = WIN_PRIOR_INFO;
    for (let wi = 0; wi < win.length; wi++) {
      const { a, w } = win[wi];
      const lw = (zeroNewest && wi === 0) ? 0 : w;          // zero newest's evidence
      if (lw === 0) continue;
      const span = clearSpan(a.floor);
      const praw = sigmoid(a.slope * (R - a.x) + a.kappa);
      let p = a.floor + span * praw; p = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
      const g = span * a.slope * praw * (1 - praw);
      score += lw * (a.correct - p) * g / (p * (1 - p));
      info += lw * g * g / (p * (1 - p));
    }
    if (info < 1e-9) break;
    let step = score / info; if (step > 2) step = 2; else if (step < -2) step = -2;
    R += step; if (R < X_LO) R = X_LO; else if (R > X_HI) R = X_HI;
    if (Math.abs(step) < 1e-6) break;
  }
  return { R, effN, departW, departCorrect, windowLen: win.length, rampHitZero };
}

// ---- replay ----
const data = JSON.parse(readFileSync(path.join(HERE, 'sim_input_new.json'), 'utf8'));
const problems = data.problems;
const attempts = data.users[USER];

const att = [];                 // ported first-time attempts, in order
let R = 0;                      // running displayed rank (index)
// validation vs engine on the first chunk
let valMaxDiff = 0, valN = 0;
const engHist = [];

const rows = [];
for (const [tid, succ, size, , , isr] of attempts) {
  const m = problems[tid];
  const correct = succ === 1;
  // keep an engine-format history for validation
  engHist.push({ problemId: tid, problemRating: m[0], problemRank: E.rankForSize(m[0], size),
    problemSlope: m[1], problemFloor: m[2], correct, date: 1, isRepeat: isr === 1, size });
  if (isr === 1) continue;       // rank uses first-time only

  const a = makeAttempt(m[0], m[1], m[2], size, correct);
  const refR = R;
  att.push(a);

  const after = windowMLE(att, refR, { cap: WIN_CAP });
  const noK = windowMLE(att, refR, { cap: WIN_CAP, zeroNewest: true });
  const cap1 = windowMLE(att, refR, { cap: WIN_CAP + 1 });   // keep the departing sample

  const own = after.R - noK.R;                 // this attempt's own evidence pull
  const shift = noK.R - refR;                  // boundary drop + gradual reweight
  const boundary = after.R - cap1.R;           // effect of the 500-cap dropping the oldest
  const total = after.R - refR;
  const P_served = (() => { const span = clearSpan(a.floor); return a.floor + span * sigmoid(a.slope * (refR - a.x) + a.kappa); })();

  rows.push({ correct, total, own, shift, boundary,
    effN: after.effN, departW: after.departW, departCorrect: after.departCorrect,
    capped: after.windowLen >= WIN_CAP && !after.rampHitZero,
    gap: refR - a.x, P: P_served });

  // validation: engine's estimate on the same prefix
  if (valN < 300) {
    const est = E.estimateWindowRank(engHist.map(h => ({ ...h })), refR);
    if (est) { valMaxDiff = Math.max(valMaxDiff, Math.abs(est.rank - after.R)); valN++; }
  }
  R = after.R;
}

// ---- summarize ----
const q = (arr, p) => { const s = [...arr].sort((a, b) => a - b); return s[Math.floor(p * (s.length - 1))]; };
const mean = arr => arr.reduce((a, b) => a + b, 0) / (arr.length || 1);
const cappedRows = rows.filter(r => r.capped);
const corrDown = rows.filter(r => r.correct && r.total < 0);
const wrongUp = rows.filter(r => !r.correct && r.total > 0);
const corr = rows.filter(r => r.correct), wrong = rows.filter(r => !r.correct);

console.log(`\nUSER ${USER}  first-time attempts=${rows.length}  (port↔engine max|Δ| over first ${valN} = ${valMaxDiff.toExponential(2)} rank — should be ~0)\n`);
console.log(`served-at-rank check (refR − problem_x, in stones; ~0 ⇒ served at rank):  p10=${q(rows.map(r=>r.gap),.1).toFixed(2)}  p50=${q(rows.map(r=>r.gap),.5).toFixed(2)}  p90=${q(rows.map(r=>r.gap),.9).toFixed(2)}`);
console.log(`served P(clear):  p10=${q(rows.map(r=>r.P),.1).toFixed(2)}  p50=${q(rows.map(r=>r.P),.5).toFixed(2)}  p90=${q(rows.map(r=>r.P),.9).toFixed(2)}`);
console.log(`\neffective window N:  p10=${q(rows.map(r=>r.effN),.1).toFixed(0)}  p50=${q(rows.map(r=>r.effN),.5).toFixed(0)}  p90=${q(rows.map(r=>r.effN),.9).toFixed(0)}   (WIN_CAP=${WIN_CAP})`);
console.log(`rows where 500-cap BINDS (ramp never hit 0): ${cappedRows.length}/${rows.length} = ${(100*cappedRows.length/rows.length).toFixed(0)}%`);
if (cappedRows.length) console.log(`  departing-sample weight at the cap edge (should be ~0 per the code comment):  p10=${q(cappedRows.map(r=>r.departW),.1).toFixed(2)}  p50=${q(cappedRows.map(r=>r.departW),.5).toFixed(2)}  p90=${q(cappedRows.map(r=>r.departW),.9).toFixed(2)}`);
console.log(`\nmean |rank change| per attempt (stones): ${mean(rows.map(r=>Math.abs(r.total))).toFixed(4)}`);
console.log(`contradiction: correct→down ${corrDown.length}/${corr.length}=${(100*corrDown.length/corr.length).toFixed(1)}%   wrong→up ${wrongUp.length}/${wrong.length}=${(100*wrongUp.length/wrong.length).toFixed(1)}%`);

const fmt = rs => `own=${mean(rs.map(r=>r.own)).toFixed(4)}  shift=${mean(rs.map(r=>r.shift)).toFixed(4)}  boundary=${mean(rs.map(r=>r.boundary)).toFixed(4)}  total=${mean(rs.map(r=>r.total)).toFixed(4)}`;
console.log(`\nDECOMPOSITION (mean stones; total = own + shift):`);
console.log(`  all correct       : ${fmt(corr)}`);
console.log(`  CORRECT→DOWN cases: ${fmt(corrDown)}`);
if (corrDown.length) {
  const depCorr = corrDown.filter(r => r.departCorrect === true).length;
  console.log(`    of these, |boundary| ≥ |own|: ${(100*corrDown.filter(r=>Math.abs(r.boundary)>=Math.abs(r.own)).length/corrDown.length).toFixed(0)}%   departing sample was CORRECT: ${(100*depCorr/corrDown.length).toFixed(0)}%`);
}
console.log(`  all wrong         : ${fmt(wrong)}`);
console.log('');
