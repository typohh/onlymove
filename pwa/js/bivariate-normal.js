// js/bivariate-normal.js
//
// Bivariate normal CDF + sign-quadrant probabilities for the weakness UI.
//
// Used to convert the augmented-weakness model's 2x2 posterior on
// (β_correct[k], β_picked[k]) into three sign-quadrant probabilities the
// stats panel renders as "too often / not often enough / bad at" per axis.
//
// Implementation: Owen's (1956) closed form for the bivariate normal CDF
// expressed via Owen's T. T(h, a) is computed by Simpson's rule on the
// (well-behaved) integrand — accurate to ~1e-7 with N=128 subdivisions,
// which is plenty for a UI that thresholds at 60-70% probability.

// Standard normal CDF via math.js's erf. Mirrors the existing usage in
// weakness.js (where erf is also pulled off the math globals).
export function normCdf(z) {
    return 0.5 * (1 + math.erf(z / Math.SQRT2));
}

// Owen's T function:
//   T(h, a) = (1/(2π)) ∫₀^a exp(-h²(1+x²)/2) / (1+x²) dx
//
// Symmetries: T(-h, a) = T(h, a), T(h, -a) = -T(h, a). We exploit those to
// keep the integration over a positive interval. The integrand is monotone-
// decreasing and bounded by 1/(1+x²), so adaptive Simpson over a fixed grid
// converges fast.
export function owensT(h, a) {
    if (a === 0) return 0;
    const sign = a < 0 ? -1 : 1;
    const absA = Math.abs(a);
    const absH = Math.abs(h);

    // Closed-form anchor: T(0, a) = atan(a)/(2π). Avoids the trivial integral.
    if (absH < 1e-14) return sign * Math.atan(absA) / (2 * Math.PI);

    // N=128 even subdivisions of [0, absA]; integrand never spikes inside, so
    // a fixed-step composite Simpson's rule is the simplest accurate option.
    const N = 128;
    const dx = absA / N;
    let sum = 0;
    for (let i = 0; i <= N; i++) {
        const x = i * dx;
        const f = Math.exp(-absH * absH * (1 + x * x) * 0.5) / (1 + x * x);
        // Composite Simpson weights: 1 at endpoints, 4 on odd, 2 on even.
        const w = (i === 0 || i === N) ? 1 : (i % 2 === 1 ? 4 : 2);
        sum += w * f;
    }
    return sign * (sum * dx / 3) / (2 * Math.PI);
}

// Bivariate standard normal CDF: P(X ≤ a, Y ≤ b) where (X, Y) ~ N(0, 1, ρ).
// Owen's (1956) decomposition:
//   Φ₂(a, b; ρ) = ½[Φ(a) + Φ(b)] - T(a, q_a) - T(b, q_b) - δ
//
// where q_a = (b/a - ρ) / √(1-ρ²), q_b = (a/b - ρ) / √(1-ρ²),
// and δ ∈ {0, ¼} depending on the quadrant of (a, b) and the sign of ρ.
//
// Degenerate cases (ρ → ±1, a=0 or b=0) handled via limits.
export function biNormCdf(a, b, rho) {
    // Clamp ρ inside (-1, 1) to avoid divide-by-zero in the q_a, q_b terms.
    if (rho > 0.99999) rho = 0.99999;
    else if (rho < -0.99999) rho = -0.99999;

    // Trivial: independent case.
    if (rho === 0) return normCdf(a) * normCdf(b);

    // (0, 0): exact closed form.
    if (a === 0 && b === 0) {
        return 0.25 + Math.asin(rho) / (2 * Math.PI);
    }

    const sq = Math.sqrt(1 - rho * rho);

    // q_a and q_b need protection against a=0 or b=0; in those cases the
    // corresponding Owen's T term resolves via T(0, q) = atan(q)/(2π) which
    // our owensT handles. The (b/a) and (a/b) limits at zero are ±∞; the
    // multiplicative outer factor (exp(-h²(1+x²)/2)) collapses the integral
    // to zero. So setting q = ±∞ effectively zeroes the corresponding T.
    const HUGE = 1e15;
    const q_a = a !== 0 ? (b / a - rho) / sq : (b > 0 ? HUGE : -HUGE);
    const q_b = b !== 0 ? (a / b - rho) / sq : (a > 0 ? HUGE : -HUGE);

    // δ correction: 1/2 when (a, b) sit in opposite half-planes through 0,
    // 0 when in the same half-plane. Owen 1956 eq. 3.3.
    let delta;
    if (a * b > 0 || (a * b === 0 && a + b >= 0)) {
        delta = 0;
    } else {
        delta = 0.5;
    }

    const result = 0.5 * (normCdf(a) + normCdf(b))
        - owensT(a, q_a) - owensT(b, q_b) - delta * 0.5;

    // Numerical artifacts at the boundary; clamp to a valid probability.
    return Math.max(0, Math.min(1, result));
}

// Sign-quadrant probabilities for the augmented weakness model.
//
// Inputs: posterior mean (μ_c, μ_p) and 2x2 covariance [[σ_c², cov],
// [cov, σ_p²]] for one axis's (β_correct, β_picked) joint estimate.
//
// Returns:
//   tooMuch    = P(β_c > 0, β_p < 0)   -- problem-recognition fine, mis-applies the motif
//   notEnough  = P(β_c < 0, β_p > 0)   -- motif problems hard but the user's applications work
//   badAt      = P(β_c < 0, β_p < 0)   -- motif problems hard AND user mis-applies
//   noIssue    = P(β_c > 0, β_p > 0)   -- strength along this axis
//
// All four sum to ~1; small numerical mass leaks across the axes from the
// CDF integration. Suitable for a UI gate: surface a category when its prob
// is above threshold T (e.g. 0.65).
export function quadrantProbs(mu_c, mu_p, var_c, var_p, cov) {
    const sigma_c = Math.sqrt(Math.max(var_c, 1e-12));
    const sigma_p = Math.sqrt(Math.max(var_p, 1e-12));
    const rho = cov / (sigma_c * sigma_p);

    // Z-scores of zero w.r.t. the posterior: a = -μ_c/σ_c puts the question
    // "is β_c below zero?" in standard-normal coordinates.
    const a = -mu_c / sigma_c;
    const b = -mu_p / sigma_p;

    const phi_a = normCdf(a);  // P(β_c < 0)
    const phi_b = normCdf(b);  // P(β_p < 0)
    const phi_ab = biNormCdf(a, b, rho);  // P(β_c < 0 ∧ β_p < 0)

    const badAt = phi_ab;
    const notEnough = phi_a - phi_ab;
    const tooMuch = phi_b - phi_ab;
    const noIssue = 1 - phi_a - phi_b + phi_ab;

    return {
        tooMuch: Math.max(0, Math.min(1, tooMuch)),
        notEnough: Math.max(0, Math.min(1, notEnough)),
        badAt: Math.max(0, Math.min(1, badAt)),
        noIssue: Math.max(0, Math.min(1, noIssue)),
    };
}
