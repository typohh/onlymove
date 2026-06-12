// js/momentum-rank.js
//
// The front-page momentum-rank control loop, as a pure step function. One
// source of truth shared by the live per-attempt update (user-stats.js) and
// the account-sync anchored replay (account-sync.js) — the replay is only
// valid if it runs EXACTLY the live dynamics, so don't fork this logic.
//
// State: rank (normalized internal rank, 1/TOTAL_RANKS per grade; may be
// negative = below 20k — there is deliberately no lower clamp) and momentum
// (the adaptive step size). The loop is a controller, not an estimator: its
// level-correctness comes from the serving feedback (wrong rank → mismatched
// difficulty → skewed clear rate → corrective drift), which is also why a
// replay of frozen outcomes recovers momentum (a contraction) but not the
// rank level (an integrator) — see replayMomentumRank.

import { TOTAL_RANKS } from './constants.js';

// Momentum must stay positive and bounded: the failure recurrence
// m ← m − sf·(2m−1)/3 has its stable fixed point at 0.5, and with the bounded
// scalingFactor healthy play self-limits momentum to well under ~150. A
// negative or runaway value inverts/explodes the rank update (wins lower the
// rank, losses raise it). The unambiguous corruption signal is a NEGATIVE
// momentum (the oscillator's sign flip); MOMENTUM_MAX sits clearly above any
// legitimate value so a reset never clobbers a genuine fast-improver, while
// still catching positive runaway (the historical a_p² scalingFactor bug).
export const DEFAULT_MOMENTUM = 30;
export const MOMENTUM_MIN = 0.5;
export const MOMENTUM_MAX = 300;

export function clampMomentum(m) {
    if (typeof m !== 'number' || !Number.isFinite(m)) return DEFAULT_MOMENTUM;
    return Math.min(MOMENTUM_MAX, Math.max(MOMENTUM_MIN, m));
}

// Per-correct-answer rank gain used to asymptote to 0 as the normalized rank
// approached 1.0 (the `(1 - rank)/9` headroom term), pinning even the
// strongest players just under 10d. Instead the step decays toward 1/10th of
// a rank rather than 0: it slows near the top but never stops, so a strong
// player keeps drifting up — slowly — into the extrapolated range above 9d.
const MIN_RANK_INCREASE_PER_CORRECT = (1 / TOTAL_RANKS) / 10;

// Loss-step size relative to the win step. The loop is designed around a
// loss:win ratio whose fixed point is the 80% clear rate (the serving target).
// The penalty is charged on the PRE-cut momentum (symmetric with the win
// branch, which steps on the pre-growth momentum); 3.6 re-centers the
// realized clear rate on 80.0% given that losses tend to follow win streaks,
// when momentum is elevated.
const LOSS_PENALTY_MULTIPLIER = 3.6;

// Smoothly lower-bound the headroom term at MIN_RANK_INCREASE_PER_CORRECT.
// A plain Math.max(headroom, floor) is only C0 — it kinks where the two meet.
// This is a C-infinity smooth maximum (the "soft-absolute" form): it tracks
// `headroom` when headroom >> floor, eases into `floor` as headroom drops
// through it, and asymptotes to `floor` from above as headroom goes negative
// (rank past 1.0). eps = floor^2 sets the corner rounding.
function softFloor(headroom, floor) {
    const d = headroom - floor;
    return 0.5 * (headroom + floor + Math.sqrt(d * d + floor * floor));
}

// Per-attempt informativeness in [0, 1], from the problem's raw Fisher
// selection weight a_p² (`quality` = slope², see slopeToQuality). The step of
// a stochastic update should scale with the information the attempt carries
// (I ∝ a_p² at the served ~80% margin), normalized so the median problem
// behaves like the saturated pre-mapping loop did — i.e. the calibrated
// dynamics (LOSS_PENALTY_MULTIPLIER, onboarding speed) are preserved and only
// genuinely flat problems get discounted, linearly in their information.
// NB: the current rating backend ridge-shrinks slopes hard (corpus a_p²
// p25–p95 ≈ 5.0–5.7, median ≈ 5.2), so today this barely modulates; it
// becomes meaningful if a future model ships unshrunk slopes. Missing/invalid
// quality keeps the historical 0.5 half-step default.
const QUALITY_SATURATION = 5.2;

export function qualityToConfidence(quality) {
    if (typeof quality !== 'number' || !Number.isFinite(quality) || quality <= 0) return 0.5;
    return Math.min(1, quality / QUALITY_SATURATION);
}

// One attempt through the loop. `quality` is the raw a_p² stored on the
// problem/answer entry; repeats step at one fifth and never grow momentum.
// Returns the new state; momentum is clamped every step so a mis-scaled input
// can never let the recurrence run away.
export function momentumRankStep(rank, momentum, { correct, isRepeat = false, quality } = {}) {
    const scalingFactor = qualityToConfidence(quality) * (isRepeat ? 1 / 5 : 1);

    if (correct) {
        const maxIncrease = softFloor((1.0 - rank) / 9, MIN_RANK_INCREASE_PER_CORRECT);
        let intendedRatingIncrease = scalingFactor * momentum / 600;
        let intendedMomentumIncrease = scalingFactor * 0.1;
        if (intendedRatingIncrease > maxIncrease) {
            // While the headroom cap binds, momentum growth is throttled by
            // the same ratio so it can't compound freely against the cap.
            intendedMomentumIncrease *= maxIncrease / intendedRatingIncrease;
            intendedRatingIncrease = maxIncrease;
        }
        if (!isRepeat) {
            momentum *= 1 + intendedMomentumIncrease;
        }
        rank += intendedRatingIncrease;
    } else {
        // Penalize on the PRE-cut momentum, THEN shrink (see
        // LOSS_PENALTY_MULTIPLIER). No lower clamp: rank below 0 means below
        // 20k, which the rest of the stack (labels, selection, the rank→server
        // translation) extrapolates fine.
        rank -= LOSS_PENALTY_MULTIPLIER * scalingFactor * (momentum / 600);
        momentum -= scalingFactor * (momentum - (momentum + 1) / 3);
    }

    return { rank, momentum: clampMomentum(momentum) };
}

// Momentum warm-up length for the account-sync hybrid re-anchor (3000 matches
// the other long-history cutoffs, e.g. RECALL_CANDIDATE_HISTORY_LENGTH), and
// the much shorter window over which the RANK is integrated on top of the
// anchor. The split is load-bearing, measured on real histories:
//   - Momentum is a contraction (losses cut it ~×1/3) and converges backward
//     to the true loop state within ~100 attempts — warm it over everything.
//   - Rank is an integrator with no serving feedback inside a replay, so
//     replicas from different starting levels never converge to the true
//     level, and every attempt of replayed rank accumulates era-mismatch bias
//     (outcomes generated under an older loss multiplier replayed under the
//     current one drifted up to −0.3/1000 attempts). Worse, a cold-started
//     momentum of 30 makes the first replayed losses worth −0.18 rank each.
//     Warming momentum first and integrating rank only over the last ~200
//     attempts bounds the deviation from the anchor to ~±2 grades — which at
//     that horizon is mostly the user's genuine recent streak, the thing the
//     replay exists to preserve.
export const MOMENTUM_REPLAY_LENGTH = 3000;
export const RANK_REPLAY_WINDOW = 200;

// Hybrid state recovery over a (merged, date-sorted) answer history:
// momentum replays from the start of the suffix (rank held at the anchor for
// the headroom-cap term), then rank is anchored at the externally supplied
// level — the skill decomposition evaluated AT the rank window's start — and
// both integrate through the final rankWindow attempts. The anchor supplies
// the one thing a replay of frozen outcomes provably can't recover (level);
// the replay supplies what it provably can (converged momentum, recent
// streak dynamics). Deterministic in the history, so synced devices agree.
export function replayMomentumRank(history, anchorRank, {
    maxLength = MOMENTUM_REPLAY_LENGTH,
    rankWindow = RANK_REPLAY_WINDOW,
} = {}) {
    const suffix = history.slice(Math.max(0, history.length - maxLength));
    const rankStart = Math.max(0, suffix.length - rankWindow);
    let rank = anchorRank;
    let momentum = DEFAULT_MOMENTUM;
    for (let i = 0; i < suffix.length; i++) {
        const a = suffix[i];
        const next = momentumRankStep(rank, momentum, {
            correct: !!a.correct,
            isRepeat: !!a.isRepeat,
            quality: a.problemQuality,
        });
        momentum = next.momentum;
        // Rank integrates only inside the window; during warm-up it stays
        // pinned to the anchor so the cap term sees a realistic level.
        if (i >= rankStart) rank = next.rank;
    }
    return { rank, momentum };
}
