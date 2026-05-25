"""Reconstruct the per-user recall-time distribution offline from the D1 log.

The PWA recall model (pwa/js/model.js) predicts, per problem, the gap at which a
re-shown problem decays to a target recall probability; the stats page surfaces
this in hours. Repeat attempts are logged as `is_repeat=1` (since 2026-05-15), so
each repeat is in principle a recall measurement.

CRITICAL DATA CONSTRAINT — only post-cutoff chains are usable.
Repeats were not logged before the 2026-05-15 cutoff. If a problem's ORIGINAL
solve predates the cutoff we cannot know how many unlogged repeats happened in
between, so both the gap-since-last-attempt and the success streak `r` are
corrupted. This script therefore keeps ONLY (user, problem) chains whose first
(non-repeat) attempt occurred AFTER the cutoff — then the whole chain is captured
and `r` / gaps are trustworthy. (Side effect: every clean gap is necessarily
< the elapsed window, ~7 days, which conveniently removes the long-gap tail that
was really hidden intermediate reviews.)

Two reconstructions, both using the model's spaced-repetition factor 0.5^r
(r = consecutive prior successes since the last failure, as of the preceding
attempt — mirrors model.js `successSinceFailure`):

  A. TIMING-ONLY (ignore success). The scheduler only re-shows a problem when it
     predicts recall in [0.6, 0.8] (getRecallCandidates), so *when* a user returns
     already encodes the model's recall time. Collapse the spaced stretch with
     gap * 0.5^r and read the distribution of those normalized gaps. This is the
     robust estimator — it does not depend on outcomes at all. Its center (~median
     normalized gap) is the ~0.7-recall time; the early/low percentiles approximate
     the 0.8 window opening.

  B. OUTCOME-BASED 80%. Fit logit P(success) = a_u + b * log(1 + gap_days * 0.5^r)
     with a shared decay slope b and a per-user intercept (random intercept, EM),
     then invert to the per-user gap that yields 80% recall. Caveat: at the gaps
     the scheduler actually picks, observed success is ~0.74, so the 80% crossing
     extrapolates to short gaps and the slope is weak (within-user forgetting is
     real and monotonic but only ~z=-1 over a 7-day window) — treat B's per-user
     numbers as directional, and prefer A for the distribution.

Works straight from `attempts/database.sql` (no ratings/ or GPU artifacts).

Run:
  ./.venv/bin/python3 -m auxiliary.recall_time_distribution
  ./.venv/bin/python3 -m auxiliary.recall_time_distribution --size 9
  ./.venv/bin/python3 -m auxiliary.recall_time_distribution --min-gap-hours 2 --no-plot
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import scipy.stats as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# tsumego_reports: id, user_id, problem, success, created_at, size, move,
# time_used, review_time, is_repeat. move is a quoted GTP coord or NULL.
REPORT_RE = re.compile(
    r'INSERT INTO "tsumego_reports".*?VALUES\('
    r"(\d+),'([^']+)','([^']+)',([01]),'([^']+)',(\d+|NULL),"
    r"(?:'[^']*'|NULL),(\d+|NULL),(\d+|NULL),([01])\);"
)


def parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load_reports(sql_path, target_size=None):
    rows = []
    with open(sql_path) as f:
        for line in f:
            if 'INSERT INTO "tsumego_reports"' not in line:
                continue
            m = REPORT_RE.search(line)
            if not m:
                continue
            _id, uid, prob, succ, ts, size, _tu, _rv, isrep = m.groups()
            sz = int(size) if size != "NULL" else 9
            if target_size is not None and sz != target_size:
                continue
            try:
                dt = parse_ts(ts)
            except ValueError:
                continue
            rows.append({"user_id": uid, "problem": prob, "success": int(succ),
                         "dt": dt, "size": sz, "is_repeat": int(isrep)})
    rows.sort(key=lambda r: r["dt"])
    return rows


def build_clean_measurements(rows):
    """Keep only chains whose original (first) attempt is after the repeat-logging
    cutoff. For each repeat in such a chain emit gap_h, success, r, and the
    spaced-normalized gap (gap_h * 0.5^r). Returns (measurements, cutoff, stats)."""
    repeat_times = [r["dt"] for r in rows if r["is_repeat"] == 1]
    if not repeat_times:
        return [], None, {}
    cutoff = min(repeat_times)

    by_up = defaultdict(list)
    for row in rows:
        by_up[(row["user_id"], row["problem"])].append(row)

    out = []
    n_chains = n_clean_chains = n_clean_with_repeat = 0
    n_excluded_precutoff = 0
    for (uid, _prob), seq in by_up.items():
        seq.sort(key=lambda r: r["dt"])
        n_chains += 1
        # Original solve must be after the cutoff and itself a non-repeat.
        if seq[0]["dt"] < cutoff or seq[0]["is_repeat"] != 0:
            n_excluded_precutoff += 1
            continue
        n_clean_chains += 1
        streak = 0
        had_repeat = False
        for i, row in enumerate(seq):
            if row["is_repeat"] == 1 and i > 0:
                had_repeat = True
                gap_h = (row["dt"] - seq[i - 1]["dt"]).total_seconds() / 3600.0
                out.append({"user_id": uid, "gap_h": gap_h, "r": streak,
                            "success": row["success"], "size": row["size"],
                            "norm_gap_h": gap_h * (0.5 ** streak)})
            streak = streak + 1 if row["success"] == 1 else 0
        if had_repeat:
            n_clean_with_repeat += 1

    stats = {"n_chains": n_chains, "n_clean_chains": n_clean_chains,
             "n_clean_with_repeat": n_clean_with_repeat,
             "n_excluded_precutoff": n_excluded_precutoff}
    return out, cutoff, stats


# ---------- helpers ----------

def logit(p):
    return np.log(p / (1 - p))


def pctl(arr, qs):
    a = np.asarray([x for x in arr if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.percentile(a, q)) for q in qs}


def fit_best_distribution(values):
    """Fit positive right-skewed candidates (log-normal, gamma, Weibull) with loc
    pinned at 0 and return the lowest-AIC one. Recall/forgetting intervals are
    positive and heavy-tailed, so these three are the natural candidates."""
    data = np.asarray([x for x in values if np.isfinite(x) and x > 0], dtype=float)
    if data.size < 8:
        return None
    candidates = {"log-normal": st.lognorm, "gamma": st.gamma,
                  "Weibull": st.weibull_min}
    best = None
    for name, dist in candidates.items():
        try:
            params = dist.fit(data, floc=0)
            ll = float(np.sum(dist.logpdf(data, *params)))
            if not np.isfinite(ll):
                continue
            k = len(params) - 1  # loc is fixed
            aic = 2 * k - 2 * ll
            ks_p = float(st.kstest(data, dist.cdf, args=params).pvalue)
        except Exception:  # noqa: BLE001
            continue
        rec = {"name": name, "dist": dist, "params": params, "aic": aic,
               "ks_p": ks_p, "n": data.size,
               "median": float(dist.median(*params)),
               "mean": float(dist.mean(*params))}
        if best is None or aic < best["aic"]:
            best = rec
    return best


def print_fit(best, indent="    "):
    if best is None:
        print(f"{indent}(too few points to fit a distribution)")
        return
    print(f"{indent}best fit: {best['name']} "
          f"(AIC={best['aic']:.1f}, KS p={best['ks_p']:.2f}, n={best['n']})  "
          f"-> median={best['median']:.0f}h, mean={best['mean']:.0f}h")


def text_hist(values, edges, label_fmt="{:.0f}"):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    for i in range(len(edges) - 1):
        c = int(np.sum((vals >= edges[i]) & (vals < edges[i + 1])))
        hi = "inf" if edges[i + 1] > 1e8 else label_fmt.format(edges[i + 1])
        print(f"    {label_fmt.format(edges[i]):>5s}-{hi:<5s}h | {'#'*c} {c}")


def fit_pooled(feat, y, ridge=1e-6, iters=80):
    X = np.column_stack([np.ones_like(feat), feat])
    beta = np.zeros(2)
    H = np.eye(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = (X * w[:, None]).T @ X + ridge * np.eye(2)
        g = X.T @ (y - p) - ridge * beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    cov = np.linalg.inv(H)
    return beta, cov


def fit_random_intercept(feat, y, uarr, U, em_iters=40):
    """Random-intercept logistic: logit P = a_u + b*feat, b shared, a_u ~ N(a0,tau^2).
    Coordinate ascent: per-user MAP intercepts (E/M), then a 1-D Newton step for b
    using a_u as an offset, then update (a0, tau^2)."""
    beta0, _ = fit_pooled(feat, y)
    a0, b = float(beta0[0]), float(beta0[1])
    tau2 = 0.5
    a_u = np.full(U, a0)
    idx_by_u = [np.where(uarr == u)[0] for u in range(U)]

    for _ in range(em_iters):
        # E/M step: per-user intercept (1-D Newton with N(a0,tau2) prior).
        post_var = np.zeros(U)
        for u in range(U):
            idx = idx_by_u[u]
            f, yy = feat[idx], y[idx]
            a = a_u[u]
            w_sum = 1.0
            for _ in range(50):
                eta = a + b * f
                p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
                w = np.clip(p * (1 - p), 1e-9, None)
                w_sum = float(np.sum(w))
                grad = float(np.sum(yy - p)) - (a - a0) / tau2
                hess = -w_sum - 1.0 / tau2
                step = grad / hess
                a -= step
                if abs(step) < 1e-10:
                    break
            a_u[u] = a
            post_var[u] = 1.0 / (w_sum + 1.0 / tau2)
        a0 = float(a_u.mean())
        tau2 = float(np.mean((a_u - a0) ** 2 + post_var))

        # M step for shared slope b, with per-user intercept as offset.
        offset = a_u[uarr]
        b_info = 1e-6
        for _ in range(50):
            eta = offset + b * feat
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            w = np.clip(p * (1 - p), 1e-9, None)
            grad = float(np.sum((y - p) * feat))
            b_info = float(np.sum(w * feat * feat)) + 1e-6
            step = grad / b_info
            b += step
            if abs(step) < 1e-10:
                break
    se_b = 1.0 / np.sqrt(b_info)
    return a_u, b, a0, tau2, se_b


def recall_time_hours(a, b, target, r=0):
    """Invert logit(target) = a + b*log(1 + gap_days*0.5^r) for gap_h.
    Returns 0.0 if P is already below target at t=0, inf if no decreasing crossing."""
    if b >= 0:
        return np.inf
    feat_star = (logit(target) - a) / b
    gap_days = (np.exp(feat_star) - 1.0) * (2.0 ** r)
    if gap_days <= 0:
        return 0.0
    return gap_days * 24.0


# ---------- sections ----------

def section_sufficiency(meas, cutoff, stats, n_rows, n_repeat_rows, min_gap_h):
    print(f"\n{'='*72}\n1. DATA SUFFICIENCY (post-cutoff clean chains only)\n{'='*72}")
    print(f"  repeat-logging cutoff:            {cutoff.isoformat()}")
    print(f"  total attempt rows parsed:        {n_rows}")
    print(f"  repeat rows (is_repeat=1):        {n_repeat_rows}")
    print(f"  total (user,problem) chains:      {stats['n_chains']}")
    print(f"  chains excluded (orig pre-cutoff):{stats['n_excluded_precutoff']}")
    print(f"  clean chains (orig after cutoff): {stats['n_clean_chains']} "
          f"(w/ >=1 repeat: {stats['n_clean_with_repeat']})")
    gaps = np.array([m["gap_h"] for m in meas])
    succ = np.array([m["success"] for m in meas])
    print(f"  clean repeat measurements:        {len(meas)}")
    print(f"  distinct users with clean repeats:{len({m['user_id'] for m in meas})}")
    print(f"  overall clean repeat success:     {succ.mean():.3f}")
    p = pctl(gaps, [5, 25, 50, 75, 95])
    print(f"  raw gap hours: p5={p[5]:.1f} p25={p[25]:.1f} p50={p[50]:.1f} "
          f"p75={p[75]:.1f} p95={p[95]:.1f}")
    print(f"  fraction of gaps < {min_gap_h:g}h (retries):  {np.mean(gaps<min_gap_h):.2f}")
    counts = sorted(Counter(m["user_id"] for m in meas).values(), reverse=True)
    print("  users with >= N clean repeats:    " +
          "  ".join(f"N{thr}={sum(1 for c in counts if c>=thr)}"
                    for thr in (3, 5, 8, 10, 20)))


def section_timing(meas, min_gap_h, min_repeats):
    print(f"\n{'='*72}\n2. STRATEGY A — TIMING-ONLY RECALL TIME (ignore success)\n{'='*72}")
    kept = [m for m in meas if m["gap_h"] >= min_gap_h]
    print(f"  using {len(kept)} repeats with gap >= {min_gap_h:g}h "
          f"(excluded {len(meas)-len(kept)} within-session retries)")
    print("  recall time = gap * 0.5^r  (collapses the spaced-repetition stretch;")
    print("  re-shows fire at predicted P in [0.6,0.8], so this is the recall window)")

    ng = np.array([m["norm_gap_h"] for m in kept])
    p = pctl(ng, [10, 25, 50, 75, 90])
    print(f"\n  Pooled normalized gap (all clean repeats):")
    print(f"    p10={p[10]:.0f}h  p25={p[25]:.0f}h  median={p[50]:.0f}h  "
          f"p75={p[75]:.0f}h  p90={p[90]:.0f}h")
    print(f"    fraction in 10-200h band: {np.mean((ng>=10)&(ng<=200)):.2f}")

    by_user = defaultdict(list)
    for m in kept:
        by_user[m["user_id"]].append(m["norm_gap_h"])
    users = {u: v for u, v in by_user.items() if len(v) >= min_repeats}
    print(f"\n  Per-user recall time ({len(users)} users with >= {min_repeats} repeats):")
    if not users:
        print("    too few users; skipping per-user distribution.")
        return None
    # Per-user median ~= 0.7-recall time; per-user p20 ~= 0.8 window opening.
    med_u = np.array([np.median(v) for v in users.values()])
    p20_u = np.array([np.percentile(v, 20) for v in users.values()])
    for name, arr in (("median (≈0.70 recall)", med_u),
                      ("p20    (≈0.80 recall)", p20_u)):
        pp = pctl(arr, [10, 50, 90])
        inb = np.mean((arr >= 10) & (arr <= 200))
        print(f"    per-user {name}: across users p10={pp[10]:.0f}h "
              f"median={pp[50]:.0f}h p90={pp[90]:.0f}h  (in 10-200h: {inb:.2f})")
    print("\n  Cross-user histogram of per-user median recall time:")
    text_hist(med_u, [0, 10, 20, 40, 80, 160, 320, 1e9])
    fit = fit_best_distribution(med_u)
    print("\n  Fitted distribution to per-user median recall time:")
    print_fit(fit)
    return {"med_u": med_u, "p20_u": p20_u, "norm_gaps": ng, "fit": fit}


def section_outcome(meas, target=0.80, min_repeats=8):
    print(f"\n{'='*72}\n3. STRATEGY B — OUTCOME-BASED {target:.0%} RECALL TIME\n{'='*72}")
    # Use ALL clean gaps (incl. short) so the t->0 recall level is anchored and
    # the target crossing is identifiable.
    gaps = np.array([m["gap_h"] for m in meas])
    succ = np.array([m["success"] for m in meas], dtype=float)
    rr = np.array([m["r"] for m in meas], dtype=float)
    feat = np.log(1.0 + (gaps / 24.0) * np.power(0.5, rr))

    beta, cov = fit_pooled(feat, succ)
    a_p, b_p = beta
    print(f"  pooled fit:        a={a_p:+.3f} b={b_p:+.4f} (SE {cov[1,1]**.5:.4f}, "
          f"z={b_p/cov[1,1]**.5:+.2f})  [confounded by between-user selection]")

    uids = sorted({m["user_id"] for m in meas})
    uidx = {u: i for i, u in enumerate(uids)}
    uarr = np.array([uidx[m["user_id"]] for m in meas])
    a_u, b, a0, tau2, se_b = fit_random_intercept(feat, succ, uarr, len(uids))
    print(f"  random-intercept:  within-user b={b:+.4f} (SE {se_b:.4f}, "
          f"z={b/se_b:+.2f})  a0={a0:+.3f} tau={tau2**.5:.3f}")

    # Within-user binned check: at longer-than-your-usual gap, does recall drop?
    feat_user_mean = np.zeros(len(uids))
    cnt = np.zeros(len(uids))
    np.add.at(feat_user_mean, uarr, feat)
    np.add.at(cnt, uarr, 1)
    feat_user_mean /= np.maximum(cnt, 1)
    centered = feat - feat_user_mean[uarr]
    print("\n  Within-user forgetting check (success vs gap relative to your own usual):")
    qs = np.quantile(centered, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        mk = (centered >= qs[i]) & (centered <= qs[i+1] if i == 4 else centered < qs[i+1])
        if mk.sum() > 0:
            tag = "shorter" if i < 2 else ("usual" if i == 2 else "longer")
            print(f"    {tag:>7s} gap  n={int(mk.sum()):>4d}  success={succ[mk].mean():.3f}")

    if not (b < 0):
        print(f"\n  VERDICT: within-user slope is not negative — {target:.0%} recall time")
        print("  not identifiable from outcomes. Use Strategy A.")
        return None

    counts = Counter(m["user_id"] for m in meas)
    sel = [i for i, u in enumerate(uids) if counts[u] >= min_repeats]
    times = np.array([recall_time_hours(a_u[i], b, target) for i in sel])
    finite = times[np.isfinite(times) & (times > 0)]
    n_below = int(np.sum(times == 0))
    print(f"\n  Per-user {target:.0%}-recall time ({len(sel)} users >= {min_repeats} repeats):")
    print(f"    {n_below} users already below {target:.0%} at t->0 "
          f"(observed success ~0.74 < {target:.0%}), so their {target:.0%}-time ~= 0h.")
    fit = None
    if finite.size:
        pp = pctl(finite, [10, 50, 90])
        print(f"    of the rest: p10={pp[10]:.0f}h median={pp[50]:.0f}h p90={pp[90]:.0f}h")
        fit = fit_best_distribution(finite)
        print(f"\n  Fitted distribution to per-user {target:.0%}-recall time "
              f"(positive only):")
        print_fit(fit)
    print("  NOTE: scheduled gaps sit at ~0.70 recall, so the 0.80 crossing is short.")
    print("  The slope is steep partly because r is endogenous (a high success streak")
    print("  marks an already-well-learned problem), and the 0.5^r term maps those to")
    print("  feat~0 / high success — which inflates the apparent decay. Treat B's")
    print("  per-user times as directional; Strategy A is the robust distribution.")
    return {"a_u": a_u, "b": b, "times": times, "finite": finite,
            "n_below": n_below, "target": target, "fit": fit}


def _dist_panel(ax, data, fit, xmax, color, title, xlabel, annotate=None):
    """Linear-x histogram of a recall-time distribution with the best-fit PDF
    overlaid (scaled to counts)."""
    data = np.asarray([x for x in data if np.isfinite(x) and 0 < x <= xmax], dtype=float)
    if data.size == 0:
        ax.axis("off")
        return
    bins = np.linspace(0, xmax, 16)
    bw = bins[1] - bins[0]
    ax.hist(data, bins=bins, color=color, alpha=0.80)
    ax.axvspan(10, 200, color="green", alpha=0.08, label="expected 10-200h")
    ax.axvline(np.median(data), color="C3", ls="--",
               label=f"empirical median {np.median(data):.0f}h")
    if fit is not None:
        x = np.linspace(bins[0] + 1e-6, bins[-1], 400)
        pdf = fit["dist"].pdf(x, *fit["params"])
        ax.plot(x, pdf * data.size * bw, color="black", lw=2,
                label=f"{fit['name']} fit (KS p={fit['ks_p']:.2f})")
    if annotate:
        ax.text(0.97, 0.97, annotate, transform=ax.transAxes, ha="right",
                va="top", fontsize=8, color="gray")
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("users")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def make_plot(meas, min_gap_h, timing, outcome, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (plot skipped: {e})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if timing is not None:
        xmax = float(np.percentile(timing["med_u"], 97))
        _dist_panel(
            axes[0], timing["med_u"], timing.get("fit"), xmax, "C0",
            "Strategy A: timing-only recall time\n(per-user median, ~0.70-recall band)",
            "per-user recall time (hours)")
    else:
        axes[0].axis("off")

    if outcome is not None and outcome.get("finite") is not None and outcome["finite"].size:
        xmax = float(np.percentile(outcome["finite"], 97))
        ann = (f"{outcome['n_below']} more users already\n"
               f"below {outcome['target']:.0%} at t->0 (~0h)")
        _dist_panel(
            axes[1], outcome["finite"], outcome.get("fit"), xmax, "C4",
            f"Strategy B: outcome-based {outcome['target']:.0%}-recall time\n"
            "(per-user; positive only)",
            f"per-user {outcome['target']:.0%}-recall time (hours)", annotate=ann)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"\n  plot written to {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, choices=(9, 13),
                    help="restrict to one board size (default: both)")
    ap.add_argument("--min-gap-hours", type=float, default=1.0,
                    help="Strategy A: exclude smaller gaps as within-session retries")
    ap.add_argument("--min-repeats", type=int, default=5,
                    help="min clean repeats to include a user in per-user distributions")
    ap.add_argument("--target-prob", type=float, default=0.80,
                    help="Strategy B target recall probability")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "auxiliary", "recall_time_distribution.png"))
    args = ap.parse_args()

    sql_path = os.path.join(PROJECT_ROOT, "attempts", "database.sql")
    if not os.path.exists(sql_path):
        print(f"missing {sql_path} — run `python -m pipeline.fetch_database` first")
        return

    rows = load_reports(sql_path, args.size)
    n_repeat_rows = sum(1 for r in rows if r["is_repeat"] == 1)
    meas, cutoff, stats = build_clean_measurements(rows)
    if args.size:
        print(f"(board size filter: {args.size})")
    if not meas:
        print("no clean repeat measurements found — nothing to reconstruct.")
        return

    section_sufficiency(meas, cutoff, stats, len(rows), n_repeat_rows, args.min_gap_hours)
    timing = section_timing(meas, args.min_gap_hours, args.min_repeats)
    outcome = section_outcome(meas, target=args.target_prob, min_repeats=args.min_repeats)
    if not args.no_plot:
        make_plot(meas, args.min_gap_hours, timing, outcome, args.out)


if __name__ == "__main__":
    main()
