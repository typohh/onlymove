// js/rank-decomposition.js
//
// Two-component rank decomposition:  R(t) = S(t) + M(t)
//   S(t) — slow "skill"   GP, ℓ_S ~ years,    σ_S ~ 0.7 ranks
//   M(t) — fast "form"    GP, ℓ_M ~ 1 hour,   σ_M ~ 1 rank
//   y_i | t_i, r_i  ~  Bernoulli( σ( κ · (R(t_i) − r_i) ) )
//
// What the *correct* fit would be
// -------------------------------
// Joint state-space filter on the stacked Markov state (S_t, M_t):
//   - Transition: per-component OU decay exp(−Δt/ℓ) + process noise.
//   - Observation: Bernoulli(σ(κ·(S_t + M_t − r_t))) — non-Gaussian.
//   - Inference: extended Kalman / Laplace-approx EKF stepping forward
//     through attempts, with EM (or marginal-likelihood maximization) for
//     the hyperparameters (ℓ_S, ℓ_M, σ_S, σ_M, κ).
// That's roughly 10× the code for a marginal gain when ℓ_S / ℓ_M ≈ 10⁴ —
// at that separation the joint posterior covariance is nearly block-diagonal
// and backfitting recovers the same answer.
//
// What we actually do (this file)
// -------------------------------
//   1. Time-bucket the attempts (30 min). Within a bucket we treat outcomes
//      as one cluster (intra-bucket correlation = 1, conservative): a bucket
//      is one "independent moment of evidence" no matter how many tsumegos
//      were solved in it. This is what keeps bursts from inflating effN.
//   2. Per bucket b, form the rank pseudo-observation
//        R_b = r̄_b + (logit(p̂_b) − logit(0.8)) / κ
//      with a Beta(α=0.8·pseudo, β=0.2·pseudo) prior on the win rate p̂_b so
//      single-attempt buckets stay finite.
//   3. Backfit two passes:
//        Ŝ ← smooth_τ_S( R_b − M̂_b )
//        M̂ ← smooth_τ_M( R_b − Ŝ_b )
//      Gaussian kernels at the two timescales. Two iterations is enough
//      because of the scale separation.
//
// What we lose vs the joint EKF
// -----------------------------
//   - Pseudo-observations are MLEs on the rank scale, not posterior means.
//     The Beta prior mitigates the tail bias.
//   - SE bands on Ŝ and M̂ use kernel-Kish effN and don't propagate
//     iteration-to-iteration uncertainty (assume "R_b is the data," not
//     "R_b − M̂_b is the data"). Fine here because of scale separation.
//   - Hyperparameters are fixed priors. No automatic ℓ/σ learning.

import { TOTAL_RANKS } from './constants.js';

const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;

// Bucket at 6h: roughly one session per bucket for typical users. We no
// longer render a per-attempt form curve (the data simply isn't dense enough
// for hour-scale form to read as anything but noise on a months-long chart),
// so the only thing the bucketing has to support is the session-to-session
// variance estimate and the long-term skill smoother. 6h gives ~720 buckets
// for a 6-month profile of 6500 attempts — backfit is now O(720²) instead of
// O(4000²), cutting load time from seconds to milliseconds.
const BUCKET_MS = 6 * 60 * 60 * 1000;      // 6 h

// Kernel timescales. Gaussian — both directions, both kernels.
const TAU_S_MS = 30 * DAY_MS;              // skill: 1-month-ish scale
const TAU_M_MS = 60 * 60 * 1000;           // form: 60 min

// Truncate the Gaussian at 4σ. Per-evaluation cost stays bounded; the
// contribution past 4σ is e^(−8) ≈ 3e-4, negligible.
const KERNEL_TRUNC_SIGMA = 4;

// Problem-rank calibration target (matches model.js).
const CALIBRATION_P = 0.8;
const LOGIT_CALIBRATION = Math.log(CALIBRATION_P / (1 - CALIBRATION_P));

// Logit-per-rank — keep in sync with model.js LOGIT_PER_RANK_PRIOR.
const LOGIT_PER_RANK = 0.28;

// Conservative per-bucket pseudo-observation noise variance, using the
// calibration-point p ≈ 0.8:
//   Var(R_b) ≈ 1 / [κ² · p(1−p)]   (delta-method on logit)
// Applied uniformly so the SE bands on Ŝ and M̂ become 1/√effN · const.
const PER_BUCKET_NOISE_VAR =
    1 / (LOGIT_PER_RANK * LOGIT_PER_RANK * CALIBRATION_P * (1 - CALIBRATION_P));

// Physical prior on the marginal stdev of M(t), in ranks. Even severe form
// swings (sleep-deprived, drunk, emotionally rattled) cap out around ±3
// ranks (≈3σ), so σ_M ≈ 1 rank. Without this prior, the per-bucket noise
// (≈9 ranks at p=0.8) makes the unregularized kernel-MLE for M(t) run wild
// on sparse data — single-bucket sessions produce ±10-rank excursions.
//
// MAP estimator with prior M(t) ~ N(0, σ_M²) and obs r_b | M(t) ~ N(M, σ_obs²):
//   M̂(t)         = Σ w_b · r_b / (Σ w_b + λ_M)
//   Var(M̂(t))    = σ_obs² / (Σ w_b + λ_M)         ← bounded above by σ_M²
//   λ_M          = σ_obs² / σ_M²                  ≈ 80
//
// Shrinks point estimate to 0 when data is thin, lets it drift toward the
// data-driven value once Σw exceeds λ. S(t) gets no such prior — there's
// plenty of long-window data and absolute rank can sit anywhere on the
// scale, so shrinking toward 0 (= 20k) would be wrong.
const SIGMA_M_PRIOR = 1.0;
const LAMBDA_M = PER_BUCKET_NOISE_VAR / (SIGMA_M_PRIOR * SIGMA_M_PRIOR);

// Per-bucket pseudo-rank R_b is the MAP of a single-parameter logistic
// regression on the actual per-attempt outcomes within the bucket:
//
//   P(correct_i | R_b, r_i) = σ( κ·(R_b − r_i) + logit(0.8) )
//
// This is the *correct* sufficient-statistic-free formulation: a win on a
// hard problem and a loss on an easy one carry different information about
// R_b than the reverse, which the aggregate (n, w, r̄) lost. Fisher info at
// the MLE gives σ_ε,b² automatically and correctly downweights attempts on
// problems far above/below the user (where σ(η)(1−σ(η)) ≈ 0).
//
// Prior: pseudo virtual attempts at problem rank r̄_b distributed 0.8:0.2
// wins/losses (so they're consistent with R_b = r̄_b → p = 0.8). Same prior
// strength as the previous Beta-on-p̂ formulation.
function fitBucketMLE(attempts, rBar, pseudo) {
    let R = rBar;
    for (let iter = 0; iter < 30; iter++) {
        let score = 0, info = 0;
        for (const a of attempts) {
            const eta = LOGIT_PER_RANK * (R - a.rank) + LOGIT_CALIBRATION;
            const p = 1 / (1 + Math.exp(-eta));
            score += LOGIT_PER_RANK * (a.correct - p);
            info += LOGIT_PER_RANK * LOGIT_PER_RANK * p * (1 - p);
        }
        // Virtual prior attempts at r̄_b with mean win rate 0.8.
        const etaPrior = LOGIT_PER_RANK * (R - rBar) + LOGIT_CALIBRATION;
        const pPrior = 1 / (1 + Math.exp(-etaPrior));
        score += LOGIT_PER_RANK * (CALIBRATION_P * pseudo - pseudo * pPrior);
        info += LOGIT_PER_RANK * LOGIT_PER_RANK * pseudo * pPrior * (1 - pPrior);
        if (info < 1e-9) break;
        const delta = score / info;
        R += delta;
        if (Math.abs(delta) < 1e-6) break;
    }
    let info = 0;
    for (const a of attempts) {
        const eta = LOGIT_PER_RANK * (R - a.rank) + LOGIT_CALIBRATION;
        const p = 1 / (1 + Math.exp(-eta));
        info += LOGIT_PER_RANK * LOGIT_PER_RANK * p * (1 - p);
    }
    const etaPriorFinal = LOGIT_PER_RANK * (R - rBar) + LOGIT_CALIBRATION;
    const pPriorFinal = 1 / (1 + Math.exp(-etaPriorFinal));
    info += LOGIT_PER_RANK * LOGIT_PER_RANK * pseudo * pPriorFinal * (1 - pPriorFinal);
    return { R, sigma2: 1 / Math.max(info, 1e-9) };
}

function buildBuckets(answerHistory) {
    const entries = answerHistory
        .filter(a => typeof a.problemRating === 'number'
                     && typeof a.date === 'number'
                     && !a.isRepeat
                     && !a.linked)
        .slice()
        .sort((a, b) => a.date - b.date);
    if (entries.length === 0) return [];

    const map = new Map();
    for (const a of entries) {
        const key = Math.floor(a.date / BUCKET_MS);
        let b = map.get(key);
        if (!b) { b = { tSum: 0, attempts: [] }; map.set(key, b); }
        b.tSum += a.date;
        b.attempts.push({
            rank: a.problemRating * TOTAL_RANKS,
            correct: a.correct ? 1 : 0,
        });
    }

    const arr = [...map.values()];
    for (const b of arr) {
        b.n = b.attempts.length;
        b.t = b.tSum / b.n;
        b.rBar = b.attempts.reduce((s, a) => s + a.rank, 0) / b.n;
        const pseudo = Math.max(1, b.n / 5);
        const { R, sigma2 } = fitBucketMLE(b.attempts, b.rBar, pseudo);
        b.R = R;
        b.sigma2 = sigma2;
        // Drop the per-attempt list once R/σ² are computed — saves memory on
        // large profiles and downstream code only needs R, sigma2, t, n.
        delete b.attempts;
        delete b.tSum;
    }
    arr.sort((a, b) => a.t - b.t);
    return arr;
}

function gaussianWeight(dtOverTau) {
    if (Math.abs(dtOverTau) > KERNEL_TRUNC_SIGMA) return 0;
    return Math.exp(-0.5 * dtOverTau * dtOverTau);
}

function smoothAtBuckets(buckets, tau, getValue, lambda = 0) {
    const N = buckets.length;
    const values = new Array(N);
    const effN = new Array(N);
    for (let i = 0; i < N; i++) {
        const t = buckets[i].t;
        let sumW = 0, sumWV = 0, sumW2 = 0;
        for (let j = 0; j < N; j++) {
            const w = gaussianWeight((buckets[j].t - t) / tau);
            if (w === 0) continue;
            sumW += w;
            sumW2 += w * w;
            sumWV += w * getValue(j);
        }
        // Bayesian MAP: data sum plus prior pseudocount λ in the denominator.
        // λ = 0 reduces to the classical Nadaraya–Watson estimator.
        const denom = sumW + lambda;
        values[i] = denom > 0 ? sumWV / denom : 0;
        effN[i] = sumW > 0 ? (sumW * sumW) / sumW2 : 0;
    }
    return { values, effN };
}

export function computeRankDecomposition(answerHistory, options = {}) {
    const iterations = options.iterations || 2;
    const buckets = buildBuckets(answerHistory);
    if (buckets.length < 10) return null;

    const N = buckets.length;
    let S = new Array(N).fill(0);
    let M = new Array(N).fill(0);

    for (let iter = 0; iter < iterations; iter++) {
        ({ values: S } = smoothAtBuckets(buckets, TAU_S_MS, j => buckets[j].R - M[j]));
        ({ values: M } = smoothAtBuckets(buckets, TAU_M_MS, j => buckets[j].R - S[j], LAMBDA_M));
    }

    // Variance of the kernel-weighted estimator with heterogeneous per-bucket
    // noise variances σ_ε,j² (= buckets[j].sigma2):
    //   Var(Ŝ(t)) = Σ w_j² · σ_ε,j² / (Σ w_j)²
    // When all σ_ε,j² are equal this reduces to σ_obs² / effN_Kish, but with
    // mixed bucket sizes (some sparse, some dense) the dense buckets
    // correctly drive the SE down. Computed at each bucket centroid.
    const effNS = new Array(N);
    const effNM = new Array(N);
    const seS = new Array(N);
    const seM = new Array(N);
    for (let i = 0; i < N; i++) {
        const t = buckets[i].t;
        let sumWS = 0, sumW2S = 0, sumW2sigS = 0;
        let sumWM = 0, sumW2M = 0, sumW2sigM = 0;
        for (let j = 0; j < N; j++) {
            const wS = gaussianWeight((buckets[j].t - t) / TAU_S_MS);
            if (wS > 0) {
                sumWS += wS;
                sumW2S += wS * wS;
                sumW2sigS += wS * wS * buckets[j].sigma2;
            }
            const wM = gaussianWeight((buckets[j].t - t) / TAU_M_MS);
            if (wM > 0) {
                sumWM += wM;
                sumW2M += wM * wM;
                sumW2sigM += wM * wM * buckets[j].sigma2;
            }
        }
        effNS[i] = sumWS > 0 ? (sumWS * sumWS) / sumW2S : 0;
        effNM[i] = sumWM > 0 ? (sumWM * sumWM) / sumW2M : 0;
        seS[i] = sumWS > 0 ? Math.sqrt(sumW2sigS) / sumWS : Infinity;
        // M uses the MAP estimator with prior, so posterior variance includes
        // λ_M in the denominator. With per-bucket precisions:
        //   Var(M̂) ≈ (Σ w² σ_ε,j² + λ_M · σ_M²) / (Σ w + λ_M)²
        // Bounded above by σ_M² when sumW → 0.
        const denomM = sumWM + LAMBDA_M;
        const numM = sumW2sigM + LAMBDA_M * SIGMA_M_PRIOR * SIGMA_M_PRIOR;
        seM[i] = denomM > 0 ? Math.sqrt(numM) / denomM : SIGMA_M_PRIOR;
    }

    // Summary stats: "session-to-session swing" and "long-term trend".
    //
    // sigmaM — precision-weighted MLE on the raw per-bucket residuals
    // r_b = R_b − Ŝ_b. The model is r_b ~ N(M(t_b), σ_ε,b²) with
    // M(t_b) ~ N(0, σ_M²), so r_b ~ N(0, σ_M² + σ_ε,b²) marginally and
    //
    //   ℓ(σ_M²) = −½ Σ [ log(σ_M² + σ_ε,b²) + r_b² / (σ_M² + σ_ε,b²) ]
    //
    // Maximize by Newton–Raphson. Bucket-level noise variance for n_b attempts:
    //   σ_ε,b² = 1 / [κ² · p̂_b(1 − p̂_b) · n_b]    (delta-method on logit)
    //
    // Within a bucket the attempts are independent given M(t_b) — within-
    // bucket correlation is *not* an additional source of noise on R_b; it
    // only matters for sample-mean inference, not latent-state inference.
    //
    // Unweighted MoM (what we had) is dominated by sparse buckets (n=1 gives
    // σ_ε² ≈ 80, contributing nothing useful but inflating the noise floor).
    // MLE weights each bucket by 1/(σ_M² + σ_ε,b²)², so a few dense buckets
    // can drive a tight detection floor even when most buckets are sparse.
    //
    // SE comes from the inverse Fisher information at the MLE. Detection
    // floor on σ_M is set at the value where the lower 95% CI bound on σ_M²
    // first exceeds 0 — under H₀ (σ_M=0), Fisher info at σ_M²=0 is
    // ½ Σ 1/σ_ε,b⁴, so SE₀(σ_M²) = √(2 / Σ 1/σ_ε,b⁴), and the floor is
    // √(1.645 · SE₀). Different from MoM the floor scales as 1/√(N_dense)
    // when there are dense buckets, not 1/√N_total.
    const residuals = buckets.map((b, i) => b.R - S[i]);
    const noiseVars = buckets.map(b => b.sigma2);

    let sigmaM2 = 0;
    for (let iter = 0; iter < 30; iter++) {
        let score = 0, info = 0;
        for (let i = 0; i < residuals.length; i++) {
            const v = sigmaM2 + noiseVars[i];
            const r2 = residuals[i] * residuals[i];
            score += (r2 - v) / (2 * v * v);
            info += 1 / (2 * v * v);
        }
        if (sigmaM2 === 0 && score <= 0) break;                   // no signal above noise
        const delta = score / Math.max(info, 1e-12);
        const next = Math.max(0, sigmaM2 + delta);
        if (Math.abs(next - sigmaM2) < 1e-7) { sigmaM2 = next; break; }
        sigmaM2 = next;
    }
    const sigmaM = Math.sqrt(sigmaM2);

    // Floor under H₀ (σ_M=0): inverse Fisher info evaluated at σ_M²=0.
    let info0 = 0;
    for (const sv of noiseVars) info0 += 1 / (2 * sv * sv);
    const seSigmaM2_null = info0 > 0 ? 1 / Math.sqrt(info0) : Infinity;
    const sigmaMDetectionFloor = Math.sqrt(Math.max(0, 1.645 * seSigmaM2_null));

    // longTermDrift — Ŝ at the last well-supported bucket minus Ŝ at the
    // first. "Well-supported" = effN_S ≥ 10, so endpoints aren't being
    // dominated by edge-of-window prior pull.
    let firstIdx = -1, lastIdx = -1;
    for (let i = 0; i < N; i++) {
        if (effNS[i] >= 10) { if (firstIdx < 0) firstIdx = i; lastIdx = i; }
    }
    const haveSpan = firstIdx >= 0 && lastIdx > firstIdx;
    const longTermDrift = haveSpan ? S[lastIdx] - S[firstIdx] : null;
    const longTermSpanMs = haveSpan ? buckets[lastIdx].t - buckets[firstIdx].t : null;

    const currentSkill = lastIdx >= 0 ? S[lastIdx] : null;
    const currentSkillSE = lastIdx >= 0 ? seS[lastIdx] : null;
    const currentForm = lastIdx >= 0 ? M[lastIdx] : null;
    const currentFormSE = lastIdx >= 0 ? seM[lastIdx] : null;

    return {
        buckets,
        S, M,
        effNS, effNM,
        seS, seM,
        summary: {
            sigmaM,
            sigmaMDetectionFloor,
            longTermDrift,
            longTermSpanMs,
            currentSkill,
            currentSkillSE,
            currentForm,
            currentFormSE,
            bucketCount: N,
        },
    };
}

// Evaluate Ŝ(t), M̂(t) and their effNs at an arbitrary time t. Use after
// computeRankDecomposition() if you want to render a fine-grained line/band
// independent of bucket centroids.
export function evaluateAt(decomp, t) {
    if (!decomp) return null;
    const { buckets, S, M } = decomp;
    let sumWS = 0, sumW2S = 0, sumW2sigS = 0, sumWVS = 0;
    let sumWM = 0, sumW2M = 0, sumW2sigM = 0, sumWVM = 0;
    for (let j = 0; j < buckets.length; j++) {
        const sig = buckets[j].sigma2;
        const dtS = (buckets[j].t - t) / TAU_S_MS;
        const wS = gaussianWeight(dtS);
        if (wS > 0) {
            sumWS += wS;
            sumW2S += wS * wS;
            sumW2sigS += wS * wS * sig;
            sumWVS += wS * (buckets[j].R - M[j]);
        }
        const dtM = (buckets[j].t - t) / TAU_M_MS;
        const wM = gaussianWeight(dtM);
        if (wM > 0) {
            sumWM += wM;
            sumW2M += wM * wM;
            sumW2sigM += wM * wM * sig;
            sumWVM += wM * (buckets[j].R - S[j]);
        }
    }
    const sHat = sumWS > 0 ? sumWVS / sumWS : 0;
    const mHat = sumWVM / (sumWM + LAMBDA_M);
    const effNS = sumWS > 0 ? (sumWS * sumWS) / sumW2S : 0;
    const effNM = sumWM > 0 ? (sumWM * sumWM) / sumW2M : 0;
    const denomM = sumWM + LAMBDA_M;
    return {
        S: sHat,
        M: mHat,
        effNS,
        effNM,
        // Variance of weighted-mean estimator with heterogeneous σ_ε,j².
        seS: sumWS > 0 ? Math.sqrt(sumW2sigS) / sumWS : Infinity,
        seM: denomM > 0
            ? Math.sqrt(sumW2sigM + LAMBDA_M * SIGMA_M_PRIOR * SIGMA_M_PRIOR) / denomM
            : SIGMA_M_PRIOR,
    };
}
