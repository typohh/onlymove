// Synthetic ground-truth tracking test for the 3 window configs — the cost side
// that real data can't measure (true ability is unknown there). A simulated user
// with a KNOWN, evolving ability answers problems served at the live estimate
// (UCB), correct/wrong drawn from the true 4PL. We then ask: how well does each
// config's windowMLE track the truth, and how fast does it react to a change?
//
// Configs (cap / i_target), matching rank_config_compare.mjs:
//   baseline 500/16   expand 3500/16   lowinfo 500/4
//
// Scenarios (ability in stones; 0 == 20k):
//   improve : +0 → +12 linearly over the run (a steadily improving player)
//   step    : flat, then +6 jump at the midpoint
//   stable  : flat (volatility-only baseline)
//
// Metrics (steady region): RMSE(estimate − true), mean signed lag (estimate − true;
// negative on `improve` = falling behind), step settle (#attempts to within 1 stone
// after the jump), and per-attempt volatility.
import { readFileSync } from 'fs';

const REPO = '/home/ootakamoku/Repositories/onlymove';
const POP = JSON.parse(readFileSync(`${REPO}/pwa/tsumegos/population_constants.json`, 'utf8'));
const FLOOR_MAX = 0.78, CAL_P = 0.8;
const ff = POP.fat_finger_rate;
const LAPSE = (typeof ff === 'number' && ff >= 0 && ff < 1 - FLOOR_MAX) ? ff : 0.0026;
const WIN_PRIOR_INFO = 1 / 16, X_LO = -15, X_HI = 45;
const sigmoid = z => 1 / (1 + Math.exp(-z));
const clearSpan = c => Math.max(0, 1 - Math.max(c, 0) - LAPSE);
const clearKappa = c => { const cc = Math.max(c, 0); const q = Math.min(Math.max((CAL_P - cc) / (1 - cc - LAPSE), 1e-6), 1 - 1e-6); return Math.log(q / (1 - q)); };

// Representative served problem: served AT the estimate, with corpus-typical
// discrimination (~0.28 logit/rank ⇒ the low-info, served-at-rank regime that
// makes the 500-cap bind) and a typical guessing floor.
const SLOPE = 0.28, FLOOR = 0.2, KAPPA = clearKappa(FLOOR), SPAN = clearSpan(FLOOR);
const WIN_UCB_K = 1, WIN_SE_FLOOR = 1 / Math.sqrt(WIN_PRIOR_INFO + 16 / 2);

function pClear(theta, x) { return FLOOR + SPAN * sigmoid(SLOPE * (theta - x) + KAPPA); }

function windowMLE(att, refR, cap, itarget) {
  const win = []; let acc = 0;
  for (let j = att.length - 1; j >= 0 && win.length < cap; j--) {
    const w = 1 - acc / itarget; if (w <= 0) break;
    const a = att[j]; win.push({ a, w });
    const praw = sigmoid(SLOPE * (refR - a.x) + KAPPA);
    let p = FLOOR + SPAN * praw; p = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
    const g = SPAN * SLOPE * praw * (1 - praw);
    acc += g * g / (p * (1 - p));
  }
  let R = refR, info = WIN_PRIOR_INFO;
  for (let it = 0; it < 15; it++) {
    let score = WIN_PRIOR_INFO * (refR - R); info = WIN_PRIOR_INFO;
    for (const { a, w } of win) {
      const praw = sigmoid(SLOPE * (R - a.x) + KAPPA);
      let p = FLOOR + SPAN * praw; p = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
      const g = SPAN * SLOPE * praw * (1 - praw);
      score += w * (a.correct - p) * g / (p * (1 - p));
      info += w * g * g / (p * (1 - p));
    }
    if (info < 1e-9) break;
    let step = score / info; if (step > 2) step = 2; else if (step < -2) step = -2;
    R += step; if (R < X_LO) R = X_LO; else if (R > X_HI) R = X_HI;
    if (Math.abs(step) < 1e-6) break;
  }
  return { R, se: info > 0 ? 1 / Math.sqrt(info) : Infinity };
}

// deterministic RNG (mulberry32) for reproducibility
function rng(seed) { return () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

const N = 4000, WARM = 600, REPS = 40;
function trueTheta(scn, i) {
  const u = i / (N - 1);
  if (scn === 'improve') return 12 * u;
  if (scn === 'step') return i < N / 2 ? 0 : 6;
  return 4; // stable
}

const CONFIGS = [
  { label: 'baseline 500/16', cap: 500, itarget: 16 },
  { label: 'expand  3500/16', cap: 3500, itarget: 16 },
  { label: 'lowinfo 500/4', cap: 500, itarget: 4 },
];
const SCN = ['improve', 'step', 'stable'];

function runOne(scn, cfg, seed) {
  const rand = rng(seed);
  const att = []; let R = 0; let serve = 0;
  let sqErr = 0, lagSum = 0, nSteady = 0, volSum = 0, prevR = 0;
  let settle = -1;
  for (let i = 0; i < N; i++) {
    const theta = trueTheta(scn, i);
    const x = serve;                         // served at current serve-rank
    const correct = rand() < pClear(theta, x) ? 1 : 0;
    att.push({ x, correct });
    const refR = R;
    const { R: Rn, se } = windowMLE(att, refR, cfg.cap, cfg.itarget);
    R = Rn;
    serve = R + WIN_UCB_K * Math.max(0, se - WIN_SE_FLOOR);
    if (i >= WARM) {
      const e = R - theta; sqErr += e * e; lagSum += e; nSteady++;
      volSum += Math.abs(R - prevR);
    }
    if (scn === 'step' && i >= N / 2 && settle < 0 && Math.abs(R - 6) <= 1) settle = i - N / 2;
    prevR = R;
  }
  return { rmse: Math.sqrt(sqErr / nSteady), lag: lagSum / nSteady, vol: volSum / nSteady, settle };
}

const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
console.log(`\nSynthetic ground-truth tracking — N=${N} attempts, ${REPS} reps, warmup ${WARM} (served at rank, slope ${SLOPE}).\n`);
for (const scn of SCN) {
  console.log(`Scenario: ${scn}${scn === 'improve' ? '  (+0→+12 stones)' : scn === 'step' ? '  (+6 jump at midpoint)' : '  (flat +4)'}`);
  console.log(`  ${'config'.padEnd(16)}${'RMSE'.padStart(8)}${'meanLag'.padStart(9)}${'volatility'.padStart(12)}${scn === 'step' ? 'settle(att)'.padStart(12) : ''}`);
  for (const cfg of CONFIGS) {
    const rs = Array.from({ length: REPS }, (_, k) => runOne(scn, cfg, 1000 + k));
    const settles = rs.map(r => r.settle).filter(s => s >= 0);
    console.log(`  ${cfg.label.padEnd(16)}${mean(rs.map(r => r.rmse)).toFixed(2).padStart(8)}${mean(rs.map(r => r.lag)).toFixed(2).padStart(9)}` +
      `${mean(rs.map(r => r.vol)).toFixed(4).padStart(12)}` +
      (scn === 'step' ? (settles.length ? mean(settles).toFixed(0) : '>'+(N/2)).padStart(12) : ''));
  }
  console.log('');
}
console.log(`RMSE/lag in stones (lower better). lag<0 on 'improve' = estimate trails the rising true ability.`);
console.log(`volatility = mean |Δrank|/attempt. settle = attempts after the step to come within 1 stone.\n`);
