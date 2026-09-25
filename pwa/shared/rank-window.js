// pwa/shared/rank-window.js
//
// The positional-linear WINDOW rank estimator — the live front-page rank — plus
// the matching positional-weighted solve-time median, factored out of
// rank-decomposition.js so the Cloudflare Worker (EGD leaderboard) and the browser
// client run BIT-IDENTICAL rank code.
//
// Both consumers feed already-"flattened" attempts in INTERNAL-RANK space, sorted
// oldest -> newest: { x, slope, correct(0/1), floor, kappa }. The client flattens
// via rank-decomposition.js `flattenAttempts` (state/curve-aware); the Worker
// flattens from its buffer + problem_params + the inlined M curve. Keeping the fit
// itself here guarantees the two produce identical numbers.
//
// Imported helpers are pure + Worker-safe: sigmoid / weightedQuantile (math-utils.js
// touches the `math` global only inside normCdf, never called here) and clearSpan
// (irt.js, reads only the inlined population_constants.json).

import { sigmoid, weightedQuantile } from '../js/math-utils.js';
import { interpExtrap } from './rank-math.js';
import { clearSpan, clampFloor, clearKappa } from '../js/irt.js';
// Baked at build time, like irt.js's LAPSE — the population rank prior below is a
// deployment constant, not runtime state, so both the client and the Worker get it
// from the same inlined blob with no fetch and no injection seam.
import POPULATION_CONSTANTS from '../tsumegos/population_constants.json';
// The corpus's mean per-attempt information vs rank (pipeline.package_for_pwa
// emit_se_floor_curve). Inlined for the same reason as the constants above: it is a
// deployment constant, and both the client and the Worker must price the SE floor
// from the identical numbers or they show different ranks for the same history.
import SE_FLOOR_CURVE from '../tsumegos/se_floor.json';

// --- Per-attempt transform to internal-rank space -------------------------------
// Shared so the client's flattenAttempts (rank-decomposition.js) and the Worker's
// buffer flatten produce IDENTICAL inputs. The problem's raw rating maps to the
// internal rank x = applyM(rating); the success-curve slope is the problem's
// own 2PL discrimination a_p converted to logit-per-internal-rank via M's local
// steepness.
const SLOPE_LO = 0.05, SLOPE_HI = 3.0;
const MSLOPE_FLOOR = 1e-3;

// Raw rating -> internal rank through a size's M curve {ratings, ranks}.
export function applyM(rating, curve) {
    return interpExtrap(rating, curve.ratings, curve.ranks);
}

function mSlope(rating, curve) {
    const h = 1e-3;
    return (applyM(rating + h, curve) - applyM(rating - h, curve)) / (2 * h);
}

// Clamp a converted per-attempt slope into [SLOPE_LO, SLOPE_HI].
export function clampSlope(s) {
    return Math.min(SLOPE_HI, Math.max(SLOPE_LO, s));
}

// a_p (raw 2PL discrimination, positive) -> logit-per-internal-rank at `rating`,
// guarded against pathological tiny |M'|.
export function slopeInRankSpace(rating, aP, curve) {
    return clampSlope(aP / Math.max(Math.abs(mSlope(rating, curve)), MSLOPE_FLOOR));
}

// One flattened attempt in internal-rank space: { x, slope, correct, floor, kappa }.
// `curve` is the attempt's size curve; `rating` its raw rating; `aP` its problemSlope;
// `floorRaw` its problemFloor (3PL guessing floor). Returns null for an entry
// missing any of them: every shipped problem carries all three, so such an entry
// is malformed and is skipped, never filled with a default.
export function flattenAttempt(rating, aP, floorRaw, correct, curve) {
    if (!curve || !Number.isFinite(rating) || !Number.isFinite(floorRaw)
        || !Number.isFinite(aP) || aP <= 0) return null;
    const floor = clampFloor(floorRaw);
    return {
        x: applyM(rating, curve),
        slope: slopeInRankSpace(rating, aP, curve),
        correct: correct ? 1 : 0,
        floor,
        kappa: clearKappa(floor),
    };
}

export const WIN_CAP = 500;                   // hard sample-count window + the ramp's denominator

// Minimum VALID (still-in-corpus) first-time attempts before a user is treated as
// carrying a rank at all. Shared because every consumer has to draw the line in the
// same place: the Worker emits `pred_rank`/lists the player on /leaderboard only
// above it (computeRank, src/leaderboard.js), the Community rank histogram is built
// from exactly those players, and the client's own "you are here" marker on that
// histogram must appear on the same condition or it marks someone who is not among
// the bars.
export const RANK_START_MIN = 30;

// --- Population rank prior ------------------------------------------------------
// The estimate is a POSTERIOR over the user's rank, and this is its prior: the
// measured distribution of rank across the user population
// (pipeline.fit_population_constants → rank_prior). It replaced a Gaussian centred
// on the user's OWN previous estimate, which damped each step but anchored the LEVEL
// to nothing: an early excursion re-centred the prior on itself and stuck. The two
// things that made those excursions large are both fixed by any proper prior — a
// user's first attempts are frequently all-correct or all-wrong, where the
// unpenalised likelihood has no interior maximum at all.
//
// Evaluated on a GRID rather than by Newton + Laplace because the population is
// bimodal and so is the posterior while the window is short; a mode-and-curvature
// approximation would report one mode with a falsely tight se, which is the unsafe
// direction for everything downstream that scales with se. The shipped grid is
// coarse (the JSON is inlined into the bundle); it is resampled onto a finer working
// grid here because a converged posterior has se BELOW the shipped step and se is
// what the display and serve offsets are built from.
//
// PRIOR_X's bounds are the one place the rank coordinate is bounded, and that is a
// statement about the prior's support, not a ladder clamp: they sit far outside the
// observed population, and the density floor in the pipeline keeps the tails
// non-zero so an off-scale user drifts to the edge rather than breaking.
const PRIOR_WORK_STEP = 0.1;                  // working resolution, in stones
const { PRIOR_X, PRIOR_LOG } = (() => {
    const p = POPULATION_CONSTANTS.rank_prior;
    const n = p.log_density.length;
    const hi = p.rank_lo + (n - 1) * p.rank_step;
    const src = [];
    for (let i = 0; i < n; i++) src.push(p.rank_lo + i * p.rank_step);
    const m = Math.round((hi - p.rank_lo) / PRIOR_WORK_STEP) + 1;
    const xs = new Float64Array(m), lg = new Float64Array(m);
    for (let i = 0; i < m; i++) {
        xs[i] = p.rank_lo + i * PRIOR_WORK_STEP;
        lg[i] = interpExtrap(xs[i], src, p.log_density);
    }
    return { PRIOR_X: xs, PRIOR_LOG: lg };
})();
export const PRIOR_MEAN = POPULATION_CONSTANTS.rank_prior.mean;
export const PRIOR_SD = POPULATION_CONSTANTS.rank_prior.sd;
const WIN_PRIOR_INFO = 1 / (PRIOR_SD * PRIOR_SD);

// --- Display and serve offsets --------------------------------------------------
// Both are the posterior mean plus a multiple of its EXCESS se — the part above the
// irreducible floor (seFloorForMeanInfo below), so both collapse onto the mean at
// convergence and neither leaves a permanent bias. Over-rating a user is far more
// costly than under-rating (too-hard problems, and a rank that later walks back), so
// the DISPLAY sits K_DISPLAY excess-se below the mean.
//
// Serving does NOT follow the display down. Serving where you display starves the
// estimator: the user clears almost everything, successes on far-below problems carry
// almost no information, the posterior never narrows, and the display never comes up.
// Serving is therefore INFORMATION-SEEKING — above the mean, where an outcome
// discriminates in both directions:
//     serve = min(mean + SERVE_UCB_K·excess,  display + serveGapCap(attempts))
//
// The cap is the binding half and it is a product constraint, not a tuning knob: an
// uncapped UCB serves a 20-kyu a dan-level problem on their first attempt, which is
// unacceptable whatever it does for onboarding speed.
//
// Note what the gap actually resolves to. Substituting the display:
//     serve − display = min((SERVE_UCB_K − K_DISPLAY)·excess,  cap)
// so the cap is one of TWO terms, and while the posterior is wide the other one is
// large (three excess-se, tens of ranks on a fresh account). The cap is therefore the
// binding term throughout onboarding, which makes it — not the UCB — the thing that
// sets how fast a strong user is found.
//
// A CONSTANT cap is the wrong shape for that job, because it does not just limit the
// bonus, it INVERTS it: display + c sits below the mean whenever
// excess > c/(K_DISPLAY_MAGNITUDE), i.e. for the whole of onboarding at c = 5. The app
// then serves below its own best guess, which is the starvation case above.
//
// The cap is instead the CURRENT run of correct answers, plus one:
//     cap = 1 + (trailing consecutive correct in the window)
// — one rank of licence per consecutive clear, reset by any miss, never zero. This is
// a closed loop on serving difficulty rather than an open-loop constant: clearing many
// in a row is the signal that serving is too easy, and a miss is the signal to come
// back. It is tight exactly when the estimator knows least (cap 1 on the first attempt,
// where a constant 5 is already five ranks out) and it opens as evidence arrives, so
// the too-hard first problem cannot recur. Measured against the constant, on the
// shipped corpus: an unlimited-strength player reaches a dan badge in ~25 first-time
// attempts rather than ~38, and the served-difficulty distribution stays nearer the
// 80% calibration point across every simulated ability.
//
// The trade is that the badge-vs-served gap is not bounded by a constant: a long run
// opens it, and run lengths are long-tailed, so it is bounded only in distribution
// (typically ~2 ranks, occasionally 10+ during a streak while excess is still wide).
// A ceiling on the streak term bounds it outright, at the cost of the endgame speed.
// Below the cap the UCB term takes over and both collapse onto the mean as excess → 0,
// so K_DISPLAY still governs the endgame.
export const K_DISPLAY = -2;
export const SERVE_UCB_K = 1;

// An UNBLEMISHED record (no incorrect attempt anywhere in the window, which includes
// the empty window of a brand-new account) earns this on top of the run. Without it a
// fresh account's first problem lands one rank above its badge — too little for the
// first solve to move the badge at all, which reads as the app ignoring the answer.
// At 2 the first problem sits three ranks above the badge and the first clear moves
// the shown rank a full rank. It is self-limiting in the way the run itself is: the
// first miss ends it for good, so a player who is genuinely at the starting rank pays
// for it once. Raising it buys little — measured, the first-solve movement flattens
// out above 2 while a true 20-kyu's share of over-hard problems keeps climbing.
export const FRESH_RECORD_BONUS = 2;

// The serve-gap cap for a window of flattened attempts (oldest → newest). Counts the
// trailing run of correct answers. `attempts` holds FIRST-TIME attempts only (repeats
// never enter the window), so a missed recall does not reset the run — deliberate: the
// cap is a pure function of the same array the posterior is built from, which is what
// keeps the client and the Worker running identical code.
export function serveGapCap(attempts) {
    let streak = 0;
    for (let i = attempts.length - 1; i >= 0 && attempts[i].correct; i--) streak++;
    // streak === length means nothing in the window was ever missed.
    return 1 + streak + (streak === attempts.length ? FRESH_RECORD_BONUS : 0);
}

// The posterior mean -> the SHOWN rank. One definition, so the live badge
// (estimateWindowRankFromFlat) and the Rank chart's front-page-rank line
// (windowRankSeries) can never plot different numbers for the same history.
export function displayRankFor(rank, se) {
    return rank + K_DISPLAY * Math.max(0, se - seFloorAt(rank));
}

// --- Rank-dependent SE floor for the display and serve offsets -------------------
// Both offsets are k·(se − se_floor): move off the estimate while the posterior is
// loose, collapse onto the user's OWN rank as it tightens. But "tight" is not a
// global constant. The floor is the irreducible SE the window MLE would report with a
// FULL window of problems AT the user's rank, each cleared at the 80% calibration
// point — i.e. the SE at which serving MORE at-rank problems stops helping. That
// summed Fisher information is Σw · <I>: Σw the positional-weight mass of a full
// window, and <I> the mean per-attempt information of problems near the user's rank.
// Flat kyu curves (small slope) genuinely can't pin a rank as tightly as sharp dan
// ones, so a constant floor mis-scales by ~4× across the ladder and leaves flat-region
// users perpetually served above their rank, never converging.
//
// <I> is the FULL 4PL information anchorInfo(floor)·slope² (irt.js), matching the
// g²/(p(1−p)) windowMLE accumulates below. It used to be the 2PL constant
// 0.8·0.2·<slope²>, which ignored the guessing floor and the lapse and so claimed
// STRICTLY more information than a real at-rank window can deliver — putting the
// floor below the se the estimator actually converges to. Two consequences, both
// structural rather than a matter of degree: the bonus never reached zero, so every
// converged user carried a permanent over-serve; and rankConverged (se < 2×floor)
// fired later than its "within 2× of the best achievable" intent. Keeping the two
// definitions in one formula is what makes the floor REACHABLE at all.
// The mean is UNWEIGHTED on purpose: it describes the post-convergence steady state,
// where selection is weakness-driven rather than information-driven.
export const WIN_STEADY_WEIGHT = (WIN_CAP + 1) / 2;   // Σ of the (M−age)/M ramp over a full window
export function seFloorForMeanInfo(meanInfo) {
    const info = WIN_PRIOR_INFO + WIN_STEADY_WEIGHT * meanInfo;
    return info > 0 ? 1 / Math.sqrt(info) : Infinity;
}

// <I> at a rank, off the shipped corpus curve (package_for_pwa emit_se_floor_curve).
// It is a corpus property, so it is MEASURED ONCE over every problem shipped rather
// than by each reader over the problems it happens to hold. A reader-side scan is what
// this replaces, and it was wrong in two ways at once: the client saw only the shards
// loaded this session — which cluster around the SERVE rank, above the mean the floor
// is priced at — while the Worker saw the whole params map, so the badge and the
// stored pred_rank could disagree on identical history; and a 50-problem sample mean
// moved as the estimate drifted or a shard landed, which the display doubles into
// stones and which re-crosses band edges under a user whose skill has not moved.
//
// The knots are the rank prior's own grid. windowMLE evaluates the posterior on that
// grid and returns a weighted mean over it, so a rank reaching this function is always
// inside — interpExtrap therefore never extrapolates, and there is no end convention
// to get wrong. That matters more than it looks: extending interpolated INFORMATION
// linearly can take it below zero, and a non-positive information gives an infinite
// floor, which drops the excess term entirely and silently shows the raw mean.
const FLOOR_X = (() => {
    const { rank_lo, rank_step, mean_info } = SE_FLOOR_CURVE;
    const xs = new Float64Array(mean_info.length);
    for (let i = 0; i < mean_info.length; i++) xs[i] = rank_lo + i * rank_step;
    return xs;
})();
const FLOOR_INFO = Float64Array.from(SE_FLOOR_CURVE.mean_info);

// The lookup is CLAMPED to the knot range rather than extrapolated. On the shipped
// grids that changes nothing — the prior's support is the knot range, so no reachable
// rank is outside — and that is the point: it makes "never extrapolated" structural
// instead of a property two files have to keep agreeing on. Extending information
// linearly takes it negative within a few stones of either end, and seFloorForMeanInfo
// maps a non-positive information to Infinity, which drops the excess term and shows
// the raw posterior mean — a silent failure with no symptom at the call site. Holding
// the end knots is also the right answer on its own terms: past the last measured
// rank the best available problems are the ones at that rank.
export function seFloorAt(rank) {
    if (!Number.isFinite(rank)) return seFloorForMeanInfo(FLOOR_INFO[0]);
    const x = Math.min(Math.max(rank, FLOOR_X[0]), FLOOR_X[FLOOR_X.length - 1]);
    return seFloorForMeanInfo(interpExtrap(x, FLOOR_X, FLOOR_INFO));
}

// Weighted 4PL Newton for the internal rank R over the positionally-weighted window
// (walking back from the newest: weight 1 at the newest, ~1/M at the oldest in-window,
// M = min(#attempts, WIN_CAP)). `attempts` sorted oldest -> newest.
// One attempt's log-likelihood across PRIOR_X. Memoized per flattened attempt: the
// grid is fixed, so an attempt's contribution never changes, and windowRankSeries
// re-evaluates overlapping windows hundreds of times over the same objects. The cache
// is weak, and flattenAttempts builds fresh objects each pass, so it never outlives
// its inputs. ONE definition of the likelihood — the series path and the live path
// must not drift, which is the whole reason this module is shared with the Worker.
const LL_CACHE = new WeakMap();
function attemptLogLik(a) {
    let ll = LL_CACHE.get(a);
    if (ll) return ll;
    const n = PRIOR_X.length;
    ll = new Float64Array(n);
    const span = clearSpan(a.floor);
    for (let i = 0; i < n; i++) {
        const praw = sigmoid(a.slope * (PRIOR_X[i] - a.x) + a.kappa);
        let p = a.floor + span * praw;
        if (p < 1e-9) p = 1e-9; else if (p > 1 - 1e-9) p = 1 - 1e-9;
        ll[i] = a.correct ? Math.log(p) : Math.log(1 - p);
    }
    LL_CACHE.set(a, ll);
    return ll;
}

// Grid posterior for the internal rank over the positionally-weighted window
// (walking back from the newest: weight 1 at the newest, ~1/M at the oldest
// in-window, M = min(#attempts, WIN_CAP)). `attempts` sorted oldest -> newest.
// Returns the posterior MEAN and SD — R is the honest estimate every downstream
// consumer scores against; the display and serve offsets are applied separately.
export function windowMLE(attempts) {
    const n = PRIOR_X.length;
    const lp = new Float64Array(PRIOR_LOG);
    const M = Math.min(attempts.length, WIN_CAP);
    for (let j = attempts.length - 1, age = 0; j >= 0 && age < WIN_CAP; j--, age++) {
        const w = (M - age) / M;
        const ll = attemptLogLik(attempts[j]);
        for (let i = 0; i < n; i++) lp[i] += w * ll[i];
    }
    return posteriorFromLp(lp);
}

// The posterior (mean + se) from an UNNORMALISED log-density on the prior grid.
// ONE definition, shared by windowMLE (which builds `lp` by summing the window)
// and windowRankWalk (which maintains the same `lp` incrementally) — so the two
// paths cannot normalize, centre or rescale differently.
function posteriorFromLp(lp) {
    const n = PRIOR_X.length;
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (lp[i] > mx) mx = lp[i];
    if (!Number.isFinite(mx)) return { R: PRIOR_MEAN, se: PRIOR_SD };
    let s = 0, sx = 0, sxx = 0;
    for (let i = 0; i < n; i++) {
        const g = Math.exp(lp[i] - mx);
        s += g; sx += g * PRIOR_X[i]; sxx += g * PRIOR_X[i] * PRIOR_X[i];
    }
    if (!(s > 0)) return { R: PRIOR_MEAN, se: PRIOR_SD };
    const mean = sx / s;
    return { R: mean, se: Math.sqrt(Math.max(sxx / s - mean * mean, 0)) };
}

// Steps between from-scratch resyncs of the incremental sums below. The rescale
// formulas are algebraically exact but accumulate floating-point error because
// every step multiplies the whole `lp` vector, so `lp` is rebuilt from the window
// every WALK_RESYNC steps to bound the drift. Resync is O(WIN_CAP·grid) and the
// walks are hundreds of thousands of steps, so the amortised cost is negligible.
const WALK_RESYNC = 256;

// The SHOWN window rank after EVERY attempt in a history, in one forward pass —
// the same posterior windowMLE computes (same trailing WIN_CAP positional
// weights, same grid) but maintained incrementally instead of re-summed from
// scratch at each step. `flatOrNull[i]` is the flattened i-th attempt in BUFFER
// order (oldest -> newest), or null when the attempt is unusable (left the
// corpus / malformed). Returns a Float64Array aligned 1:1 with the input, each
// element displayRankFor(mean, se) or NaN below RANK_START_MIN usable attempts.
//
// Why incremental: the offline monthly-history rebuild walks a user's WHOLE
// report history (hundreds of thousands of attempts corpus-wide), and calling
// windowMLE per step is O(WIN_CAP) windows of grid work each — hours. `lp` here
// is the weighted sum of the cached per-attempt log-likelihood vectors WITHOUT
// the prior (which windowMLE folds in at construction and which no window update
// touches), and the window only ever gains a tail and loses a head, so both
// updates are O(grid):
//
//   append v (M -> M+1):  lp' = ll_v + (M/(M+1))·lp
//   drop oldest u (M -> M-1):  lp' = (M/(M-1))·(lp - S/M),  S = Σ ll over the window
//
// (Appending shifts every remaining attempt one age older, which is exactly the
// factor M/(M+1); dropping the oldest renormalises the remaining weights.) A
// step that does both is the drop followed by the append. Only the valid
// attempts enter `lp`, so M = usable count and never exceeds WIN_CAP. The prior
// is added back once per output to form the full log-density.
export function windowRankWalk(flatOrNull, cap = WIN_CAP) {
    const n = flatOrNull.length;
    const out = new Float64Array(n);
    out.fill(NaN);
    const G = PRIOR_X.length;
    const lp = new Float64Array(G);   // weighted Σ ll, NO prior
    const S = new Float64Array(G);    // Σ ll over the usable window
    const full = new Float64Array(G); // PRIOR_LOG + lp, for the posterior
    let len = 0;                      // usable attempts in the window (= M)
    for (let i = 0; i < n; i++) {
        if (i >= cap) {
            const removed = flatOrNull[i - cap];
            if (removed) {
                const ll = attemptLogLik(removed);
                if (len > 1) {
                    const f = len / (len - 1);
                    for (let k = 0; k < G; k++) lp[k] = f * (lp[k] - S[k] / len);
                } else {
                    lp.fill(0);
                }
                for (let k = 0; k < G; k++) S[k] -= ll[k];
                len--;
            }
        }
        const added = flatOrNull[i];
        if (added) {
            const ll = attemptLogLik(added);
            const f = len / (len + 1);
            for (let k = 0; k < G; k++) lp[k] = ll[k] + f * lp[k];
            for (let k = 0; k < G; k++) S[k] += ll[k];
            len++;
        }
        if (len >= RANK_START_MIN) {
            for (let k = 0; k < G; k++) full[k] = PRIOR_LOG[k] + lp[k];
            const p = posteriorFromLp(full);
            out[i] = displayRankFor(p.R, p.se);
        }
        if ((i + 1) % WALK_RESYNC === 0) {
            // Rebuild lp/S from the window in the SAME order windowMLE sums it.
            const start = Math.max(0, i - cap + 1);
            const usable = [];
            for (let j = start; j <= i; j++) if (flatOrNull[j]) usable.push(flatOrNull[j]);
            const M = usable.length;
            lp.fill(0);
            S.fill(0);
            for (let a = 0; a < M; a++) {
                const ll = attemptLogLik(usable[M - 1 - a]);
                const w = (M - a) / M;
                for (let k = 0; k < G; k++) lp[k] += w * ll[k];
                for (let k = 0; k < G; k++) S[k] += ll[k];
            }
            len = M;
        }
    }
    return out;
}

// Public window rank estimate from PRE-FLATTENED attempts (oldest -> newest).
// Returns { rank, se, serveRank, displayRank } or null when there are no attempts.
//
//   rank        the posterior MEAN — the honest estimate. Everything that has to
//               stay calibrated scores against this: the leaderboard's stored
//               pred_rank, the server translations, the rank-map anchors, the
//               recall and weakness models. Skewing it would close the calibration
//               loop on a shifted number.
//   displayRank what the front page SHOWS. Conservative by K_DISPLAY excess-se.
//   serveRank   what problem selection targets. Ramps from displayRank to rank.
//
// No warm start: the posterior is computed from the prior each call, so there is no
// path by which a previous estimate can carry itself forward.
export function estimateWindowRankFromFlat(attempts) {
    // null = "no history yet", which callers branch on to leave a stored rank alone.
    // The estimate for an empty window is not missing, though — see coldStartEstimate.
    if (!attempts.length) return null;
    return estimateFromWindow(attempts);
}

// Every rank a device shows or serves before it has answered anything. An empty
// window is not an absent posterior — windowMLE over zero attempts returns the
// population prior exactly — so all three slots come from the ORDINARY rule applied
// to that posterior, and no slot needs a stand-in constant of its own.
export function coldStartEstimate() {
    return estimateFromWindow([]);
}

function estimateFromWindow(attempts) {
    const { R, se } = windowMLE(attempts);
    const excess = Math.max(0, se - seFloorAt(R));
    const displayRank = displayRankFor(R, se);
    return {
        rank: R,
        se,
        displayRank,
        serveRank: Math.min(R + SERVE_UCB_K * excess, displayRank + serveGapCap(attempts)),
    };
}

// Positional-weighted median of a per-attempt series over the SAME last-WIN_CAP
// window and the SAME (M − age)/M weights as the rank fit. `values` is per-attempt
// (oldest -> newest, novel-only, aligned 1:1 with the rank window); non-finite or
// <= 0 entries are treated as missing (zero weight, and don't count toward the
// minimum — there is deliberately NO upper cap, since a median is robust to a few
// idle-tab outliers). Returns the weighted median, or null when fewer than
// `minCount` valid values. Client + Worker share it so the time-adjustment matches.
export function weightedMedianOverWindow(values, minCount) {
    const M = Math.min(values.length, WIN_CAP);
    const pairs = [];
    let total = 0;
    for (let j = values.length - 1, age = 0; j >= 0 && age < WIN_CAP; j--, age++) {
        const v = values[j];
        if (Number.isFinite(v) && v > 0) {
            const w = (M - age) / M;
            pairs.push({ v, w });
            total += w;
        }
    }
    if (pairs.length < minCount || total <= 0) return null;
    pairs.sort((a, b) => a.v - b.v);
    return weightedQuantile(pairs, total, 0.5);
}
