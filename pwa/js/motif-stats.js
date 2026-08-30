// js/motif-stats.js
//
// Per-motif statistics, computed ENTIRELY from the user's own attempts. Nothing
// here reads a population constant, a fitted prior, or any other player's data —
// which is what makes it immune to the rank confound that a population centre
// carries (motif prevalence varies several-fold across the rank ladder, so any
// centre averaged over users is really a statement about which problems that
// user's rank gets served).
//
// Two independent readings per axis, answering different questions. The
// error-composition reading (tendency) stands against the AVAILABILITY NULL:
// what a direction-blind error on each served problem would have looked like,
// taken per problem over the alternatives its board actually offered, each
// weighed by the visits its own evaluation rests on (computeAltStats, and the
// masses packaging publishes for the same quantity). That null is still a
// property of the user's own served problems and nobody else's play; without
// it, the composition statistic reports the problem MENU as a property of the
// player (the answer to a tsumego is by construction not a tenuki, so nearly
// every alternative loads higher on that axis — a fixed 50% balance point flags
// almost everyone).
//
//   TENDENCY — of your wrong answers, what share erred TOWARD the motif rather
//   than away from it, against the share a typical error on the same problems
//   would have? Per wrong attempt, d = correct_emb[k] - picked_emb[k]: d < 0
//   means the move you played loaded HIGHER on the motif than the answer did
//   (you reached for it), d > 0 means it loaded lower (you missed it). Each
//   error contributes its problem's own null toward-probability
//   (altAbove/(altAbove+altBelow)), so the test is Poisson-binomial: observed
//   toward-count against the sum of those probabilities. Still a sign test —
//   no model, no fitting, no prior — just with the balance point supplied by
//   each problem instead of assumed at 50%.
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
import { SHIPPED_K } from './data-version.js';
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

// Share of errors that erred TOWARD the motif, against the availability null:
// each error carries its problem's own null toward-probability p, so the null
// count is Sum(p) and its variance Sum(p(1-p)) — Poisson-binomial, normal-
// approximated, which at MIN_ERRORS and interior p is what a fixed-null
// binomial would also be. Ties (pick loading identical to the answer's) carry
// no direction and are dropped rather than split — and the null drops them
// symmetrically, its masses being built from strict comparisons only, a tie
// falling out with its whole weight (computeAltStats).
function tendencyTest(nOver, n, sumP, sumPQ) {
    if (n < MIN_ERRORS) return null;
    // sumPQ can only be 0 when every problem's null was one-sided (p of 0 or
    // 1); no variance means no test.
    if (!(sumPQ > 0)) return null;
    const share = nOver / n;
    const nullShare = sumP / n;
    const se = Math.sqrt(sumPQ) / n;
    const z = (nOver - sumP) / Math.sqrt(sumPQ);
    // The band is the GATE's own width, not a conventional 95%: a reading is
    // flagged exactly when |z| > MOTIF_Z, so drawing the interval at any other
    // multiple lets a flagged row show a band straddling the neutral point — a
    // reported finding contradicting the number printed under it. Locked to
    // MOTIF_Z, and drawn at the NULL's standard error, so "band clear of the
    // null share" IS "flagged", by construction.
    return { share, nullShare, n, z,
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

    // Every stored vector is from the device's current fit by construction — the
    // wipe drops all of them when the fit changes, and no build writes one while
    // its own fit and the device's disagree (motifDataIsCurrent). So provenance is
    // settled before a reader sees a row, and this only has to check that a row
    // HAS a vector of the right shape, so the per-axis loops stay in range.
    const K = SHIPPED_K;
    if (!K) return null;
    const usable = a => Array.isArray(a.embedding) && a.embedding.length === K;

    const all = history || getMergedAnswerHistory();
    const rows = all
        .filter(a => a && !a.isRepeat && !a.linked && usable(a) &&
                     typeof a.correct === 'boolean')
        .slice(-MODEL_INPUT_HISTORY_LENGTH);
    if (rows.length < MIN_ATTEMPTS) return null;

    // --- TENDENCY: error composition against the availability null. Rows
    // without the null masses (a fill that hasn't landed yet) contribute
    // nothing — no fixed-null fallback, which would mix two different neutral
    // points into one statistic.
    //
    // altAbove/altBelow are MASSES (summed visits over the alternatives), not
    // counts, so nothing here may read them as a sample size: they enter only
    // as the >0 admission test and as the denominator of the toward-probability,
    // both of which are scale-free. The number of errors, which IS a sample
    // size, is counted separately in nDir.
    const nOver = new Array(K).fill(0);
    const nDir = new Array(K).fill(0);
    const sumP = new Array(K).fill(0);
    const sumPQ = new Array(K).fill(0);
    const altUsable = (a) =>
        Array.isArray(a.altAbove) && a.altAbove.length === K &&
        Array.isArray(a.altBelow) && a.altBelow.length === K;
    for (const a of rows) {
        if (a.correct) continue;
        if (!Array.isArray(a.pickedEmbedding) || a.pickedEmbedding.length !== K) continue;
        if (!altUsable(a)) continue;
        for (let k = 0; k < K; k++) {
            const d = a.embedding[k] - a.pickedEmbedding[k];
            const availMass = a.altAbove[k] + a.altBelow[k];
            if (Number.isFinite(d) && d !== 0 && availMass > 0) {
                const p = a.altAbove[k] / availMass;   // null P(this error leans toward)
                nDir[k]++;
                sumP[k] += p;
                sumPQ[k] += p * (1 - p);
                if (d < 0) nOver[k]++;
            }
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
        const tendency = tendencyTest(nOver[k], nDir[k], sumP[k], sumPQ[k]);
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
