// js/motif-stats.js
//
// Per-motif statistics, computed ENTIRELY from the user's own attempts. Nothing
// here reads a population constant, a fitted prior, or any other player's data —
// which is what makes it immune to the rank confound that a population centre
// carries (motif prevalence varies several-fold across the rank ladder, so any
// centre averaged over users is really a statement about which problems that
// user's rank gets served).
//
// Two independent readings per axis, answering different questions:
//
//   TENDENCY — of your wrong answers, what share erred TOWARD the motif rather
//   than away from it? Per wrong attempt, d = correct_emb[k] - picked_emb[k]:
//   d < 0 means the move you played loaded HIGHER on the motif than the answer
//   did (you reached for it), d > 0 means it loaded lower (you missed it). The
//   statistic is the share of errors with d < 0, and the null is 50% — no
//   directional bias in your own mistakes. That null is a property of your
//   errors, not a comparison against anyone: 50/50 is balance. It is a plain
//   SIGN TEST, so it needs no model, no fitting and no prior, and its interval
//   is binomial on your own error count.
//
//   STRENGTH — does a problem loading on the motif cost you accuracy beyond
//   what its DIFFICULTY already predicts? The shipped 3PL supplies the
//   per-problem prediction, so difficulty is held fixed by construction; what is
//   left is regressed on the motif loading. Reported in ranks via the local
//   slope of the same curve. This half genuinely needs the difficulty control —
//   motif loading correlates with problem rank, so a bare correlation between
//   loading and outcome would score a user weak at whatever motif happens to
//   live in hard problems.
//
// Both are read on the SAME window the rest of the client models use, and both
// are pure functions of that window, so two devices with the same history agree.

import { state, getMergedAnswerHistory } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH, MOTIF_Z } from './constants.js';
import { sigmoid } from './math-utils.js';
import { clearSpan } from './irt.js';
import { flattenAttempt } from '../shared/rank-window.js';
import { curveForSize } from './rank-decomposition.js';

// Attempts a user needs before either reading is attempted at all. Below this
// the interval is so wide that nothing clears the gate anyway; the floor just
// avoids computing on a handful of rows.
const MIN_ATTEMPTS = 20;
// Wrong attempts needed before the sign test is reported. A sign test on a
// handful of errors is dominated by its own interval.
const MIN_ERRORS = 10;

// One-sided share of errors that erred TOWARD the motif, with the binomial z
// against the 50% balance point. Ties (identical loading) carry no direction and
// are dropped rather than split, so they neither inflate n nor pull toward 0.5.
function signTest(nOver, nUnder) {
    const n = nOver + nUnder;
    if (n < MIN_ERRORS) return null;
    const share = nOver / n;
    // se of a proportion under the null p = 0.5 is sqrt(0.25/n), which is also
    // the widest it can be — so the gate is the conservative one.
    const z = (share - 0.5) / Math.sqrt(0.25 / n);
    // The band is the GATE's own width, not a conventional 95%: a reading is
    // flagged exactly when |z| > MOTIF_Z, so drawing the interval at any other
    // multiple lets a flagged row show a band straddling the neutral point — a
    // reported finding contradicting the number printed under it. Locked to
    // MOTIF_Z so "band clear of the neutral line" IS "flagged", by construction.
    const se = Math.sqrt(Math.max(share * (1 - share), 1e-9) / n);
    return { share, n, z,
             lo: Math.max(0, share - MOTIF_Z * se), hi: Math.min(1, share + MOTIF_Z * se) };
}

// Ordinary least squares of `y` on `x` with an intercept, returning the slope
// and its standard error. Used for the residual-accuracy regression, where x is
// the motif loading and y the 3PL residual.
function olsSlope(x, y) {
    const n = x.length;
    if (n < MIN_ATTEMPTS) return null;
    let sx = 0, sy = 0;
    for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; }
    const mx = sx / n, my = sy / n;
    let sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
        const dx = x[i] - mx;
        sxx += dx * dx;
        sxy += dx * (y[i] - my);
    }
    if (!(sxx > 0)) return null;
    const slope = sxy / sxx;
    let sse = 0;
    for (let i = 0; i < n; i++) {
        const r = y[i] - my - slope * (x[i] - mx);
        sse += r * r;
    }
    const se = Math.sqrt((sse / Math.max(n - 2, 1)) / sxx);
    if (!Number.isFinite(slope) || !(se > 0)) return null;
    return { slope, se, n };
}

// Per-axis statistics over the user's own window. Returns null when there is no
// usable embedding data at all; otherwise an array of K entries, each of which
// may carry a null `tendency` or `strength` if that half lacks data.
export function computeMotifStats(history = null) {
    const theta = state.userRank;
    if (typeof theta !== 'number' || !Number.isFinite(theta)) return null;

    const all = history || getMergedAnswerHistory();
    const rows = all
        .filter(a => a && !a.isRepeat && !a.linked &&
                     Array.isArray(a.embedding) &&
                     typeof a.correct === 'boolean')
        .slice(-MODEL_INPUT_HISTORY_LENGTH);
    if (rows.length < MIN_ATTEMPTS) return null;

    const K = rows[0].embedding.length;
    if (rows.some(a => a.embedding.length !== K)) return null;

    // --- TENDENCY: sign test on d, over wrong attempts with a resolved pick ---
    const nOver = new Array(K).fill(0);
    const nUnder = new Array(K).fill(0);
    for (const a of rows) {
        if (a.correct) continue;
        if (!Array.isArray(a.pickedEmbedding) || a.pickedEmbedding.length !== K) continue;
        for (let k = 0; k < K; k++) {
            const d = a.embedding[k] - a.pickedEmbedding[k];
            if (!Number.isFinite(d) || d === 0) continue;
            if (d < 0) nOver[k]++; else nUnder[k]++;
        }
    }

    // --- STRENGTH: 3PL residual regressed on the loading ---
    // The residual is in PROBABILITY units, so the slope converts to ranks by
    // the mean local dP/drank over the same rows — the curve's own steepness
    // where this user sits, which is what makes "ranks" mean the same thing here
    // as everywhere else.
    const xs = Array.from({ length: K }, () => []);
    const resid = [];
    let dpSum = 0, dpN = 0;
    for (const a of rows) {
        const curve = curveForSize(a.size);
        const f = flattenAttempt(a.problemRating, a.problemSlope, a.problemFloor,
                                 a.correct, curve);
        if (!f) continue;
        const span = clearSpan(f.floor);
        const praw = sigmoid(f.slope * (theta - f.x) + f.kappa);
        const p = f.floor + span * praw;
        if (!Number.isFinite(p)) continue;
        resid.push((a.correct ? 1 : 0) - p);
        for (let k = 0; k < K; k++) xs[k].push(a.embedding[k]);
        dpSum += f.slope * span * praw * (1 - praw);
        dpN++;
    }
    const dpPerRank = dpN > 0 ? dpSum / dpN : 0;

    const out = [];
    for (let k = 0; k < K; k++) {
        const tendency = signTest(nOver[k], nUnder[k]);
        let strength = null;
        const fit = resid.length ? olsSlope(xs[k], resid) : null;
        if (fit && dpPerRank > 0) {
            // Positive residual slope = you beat the 3PL on motif-heavy problems
            // = stronger at it. Divide by dP/drank to state it in ranks.
            strength = {
                ranks: fit.slope / dpPerRank,
                // Same gate-width band as the sign test above, same reason.
                lo: (fit.slope - MOTIF_Z * fit.se) / dpPerRank,
                hi: (fit.slope + MOTIF_Z * fit.se) / dpPerRank,
                z: fit.slope / fit.se,
                n: fit.n,
            };
        }
        out.push({ k, tendency, strength });
    }
    return out;
}

// Whether a reading clears the shared confidence gate. Confidence only — a
// finding the user is reliably making is reported however small it is.
export const isFlagged = (r) => !!r && Number.isFinite(r.z) && Math.abs(r.z) > MOTIF_Z;
