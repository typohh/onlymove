// js/irt.js
//
// Shared 4PL item-response math for the client's two rank-aware consumers:
// the recall model (recall-model.js) and the per-attempt rank estimator
// (rank-decomposition.js). Both place a user on a problem's success curve
//
//     P(clear) = floor + (1 − floor − LAPSE)·σ(slope·(rank gap) + κ(floor))
//
// anchored so P(clear) = CALIBRATION_P when the user's rank equals the
// problem's. The calibration point, floor convention and lapse have a single
// source of truth here, shared by both consumers.

// Baked at build time. We read only the fat-finger (lapse) rate from it here;
// population-constants.js is the validated loader for the rest.
import POPULATION_CONSTANTS from '../tsumegos/population_constants.json';

// The margin the PWA serves and anchors at: a problem's "rating" is the ability
// at which the user clears it CALIBRATION_P of the time (matches the backend
// fit_user_rating / fit_rank_curves and map_rating_to_rank).
export const CALIBRATION_P = 0.8;

// Guessing floors are shipped raw by package_for_pwa. Clamp below CALIBRATION_P
// so κ(c) = logit((CALIBRATION_P − c)/(1 − c − LAPSE)) stays finite (κ → ∞ as
// c → P).
export const FLOOR_MAX = 0.78;

// LAPSE: the 4PL upper-asymptote slip ("fat-finger") rate — the irreducible
// failure probability on a problem the user should clear with certainty (a
// misclick / misread, most common when strong players are bored by trivial
// problems). Fitted by pipeline.fit_population_constants as the plateau of the
// observed failure rate as modeled-clear → 1. Capping the success curve at
// (1 − LAPSE) instead of 1 is what stops a single careless failure on a
// far-below-rank problem from yanking the rank/recall estimate: at the top the
// likelihood's P(1−P) stays bounded by LAPSE·(1−LAPSE) while the gradient → 0,
// so the slip carries almost no weight. Read straight from the inlined
// population constants (mandatory + build-time, like weakness.js's AXIS_LAMBDA);
// the fitter guarantees 0 ≤ LAPSE < 1 − FLOOR_MAX so (1 − floor − LAPSE) > 0.
export const LAPSE = POPULATION_CONSTANTS.fat_finger_rate;

// σ(κ(c)): the RAW (inner) sigmoid value at a zero rank gap — where the 4PL
// curve passes through CALIBRATION_P — i.e. (CALIBRATION_P − c)/(1 − c − LAPSE).
// One expression, shared by κ and by the anchor information below.
function anchorRawClear(c) {
    const cc = Math.max(typeof c === 'number' ? c : 0, 0);
    return Math.min(Math.max((CALIBRATION_P - cc) / (1 - cc - LAPSE), 1e-6), 1 - 1e-6);
}

// κ(c): the offset that keeps P(clear) = CALIBRATION_P at a zero rank gap for a
// 4PL problem with guessing floor c and upper asymptote (1 − LAPSE). The lapse
// term is folded in so the rating's 80%-clear definition holds exactly; with
// LAPSE = 0 it reduces to the 3PL anchor.
export function clearKappa(c) {
    const q = anchorRawClear(c);
    return Math.log(q / (1 - q));
}

// The Fisher information ONE at-rank attempt carries about the user's rank, per
// unit of squared success-curve slope: I(anchor) = anchorInfo(floor)·slope².
//
// A 2PL would give the constant P(1−P) = 0.16 at the 80% point, and that constant
// is what the selection weight and the UCB SE floor used to assume. It is WRONG
// under the 4PL actually in use: a guessing floor and a lapse ceiling both flatten
// dP/dθ at the anchor without flattening the Bernoulli variance, so the same 80%
// clear rate carries strictly LESS information about where the user sits. With
// q = σ(κ(c)) and span = 1 − c − LAPSE,
//
//     dP/dinner = span·q(1−q)   and   P(1−P) = CALIBRATION_P·(1 − CALIBRATION_P)
//     I = a²·(span·q(1−q))² / (P(1−P))
//
// — the SAME g²/(p(1−p)) windowMLE accumulates, evaluated at a zero rank gap. It
// equals 0.16 exactly when floor = LAPSE = 0, and is strictly below it otherwise,
// falling monotonically as the floor rises toward CALIBRATION_P. So the 2PL
// constant always OVERSTATES a problem's information, by a factor set by the
// fitted floor distribution — which moves with every re-rate, hence no number
// here; see CLAUDE.md for the measured range at a given ship.
export function anchorInfo(floorRaw) {
    const c = clampFloor(floorRaw);
    const q = anchorRawClear(c);
    const g = clearSpan(c) * q * (1 - q);
    return (g * g) / (CALIBRATION_P * (1 - CALIBRATION_P));
}

// The success-curve span (upper asymptote − floor) = (1 − floor − LAPSE). The
// per-attempt clear probability is `floor + clearSpan(floor)·praw`, and the
// gradient dP/dinner is `clearSpan(floor)·praw·(1−praw)`. Shared so every call
// site computes the span identically. Guarded ≥ 0.
export function clearSpan(c) {
    return Math.max(0, 1 - Math.max(typeof c === 'number' ? c : 0, 0) - LAPSE);
}

// Coerce a possibly-missing per-problem floor into [0, FLOOR_MAX]; legacy rows
// without a floor fall back to 0 (i.e. plain 2PL-with-lapse behaviour).
export function clampFloor(c) {
    return Math.min(Math.max(typeof c === 'number' ? c : 0, 0), FLOOR_MAX);
}
