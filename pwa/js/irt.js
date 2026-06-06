// js/irt.js
//
// Shared 3PL item-response math for the client's two rank-aware consumers:
// the recall model (model.js) and the per-attempt rank estimator
// (rank-decomposition.js). Both place a user on a problem's success curve
//
//     P(clear) = floor + (1 − floor)·σ(slope·(rank gap) + κ(floor))
//
// anchored so P(clear) = CALIBRATION_P when the user's rank equals the
// problem's. These constants and κ/floor helpers were duplicated byte-for-byte
// in both files; centralized here so the calibration point and floor convention
// have a single source of truth.

// The margin the PWA serves and anchors at: a problem's "rating" is the ability
// at which the user clears it CALIBRATION_P of the time (matches the backend
// fit_user_rating / fit_rank_curves and map_rating_v2_to_egf).
export const CALIBRATION_P = 0.8;
export const LOGIT_CALIBRATION = Math.log(CALIBRATION_P / (1 - CALIBRATION_P));

// Guessing floors are shipped raw by package_for_pwa. Clamp below CALIBRATION_P
// so κ(c) = logit((CALIBRATION_P − c)/(1 − c)) stays finite (κ → ∞ as c → P).
export const FLOOR_MAX = 0.78;

// κ(c): the offset that keeps P(clear) = CALIBRATION_P at a zero rank gap for a
// 3PL problem with guessing floor c. For c ≤ 0 it degenerates to the 2PL anchor.
export function clearKappa(c) {
    if (!(c > 0)) return LOGIT_CALIBRATION;
    const q = Math.min(Math.max((CALIBRATION_P - c) / (1 - c), 1e-6), 1 - 1e-6);
    return Math.log(q / (1 - q));
}

// Coerce a possibly-missing per-problem floor into [0, FLOOR_MAX]; legacy rows
// without a floor fall back to 0 (i.e. plain 2PL behaviour).
export function clampFloor(c) {
    return Math.min(Math.max(typeof c === 'number' ? c : 0, 0), FLOOR_MAX);
}
