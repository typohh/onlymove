#!/usr/bin/env python3
"""SPLIT-HALF reliability of the per-user axis positions.

Every reliability figure read off the fit's own posterior compares var(v_hat)
against a posterior variance, and once the prior is wider than the real
between-user spread var(v_hat) starts absorbing estimation noise -- so relaxing
the ridge raises the number whether or not any signal is there. Splitting a
user's attempts by PARITY gives two halves whose noise is independent: fit that
user's position separately on each, with the loadings, ability and problem
parameters held fixed, and correlate the halves across users. Noise cannot
correlate with itself, so what survives is real.

Parity rather than a time split, deliberately: a first-half/second-half cut
confounds a stable trait with learning over the period.

Spearman-Brown lifts the half-length correlation to the full-length reliability.

TWO REPORTS, ANSWERING DIFFERENT QUESTIONS.

`buckets` is DISJOINT history bands. Cumulative floors (>=250, >=500, >=1000)
are nested -- the deep users sit inside every shallower level -- so a rising
sequence is partly just the progressive removal of shallow users, and the levels
are not independent observations. Disjoint bands also line up with the fit's own
per-bucket reliability table, which makes the two directly comparable.

The curve is SUMMARISED rather than left to the eye. Reading monotonicity off
four numbers is a judgement call that flips on noise, and there is a principled
statistic to hand: r(n) = tau^2 / (tau^2 + sigma^2/n) rearranges to
sigma^2/tau^2 = n(1-r)/r, which is exactly the attempt count at which reliability
reaches 0.5. So every point implies an N50, they agree if the model holds, and
the pooled fit is a product-meaningful number -- how much history a motif needs
before it can be reported. An axis with no between-user variance gives N50
infinite or wildly inconsistent, which is the same verdict monotonicity gives but
without the eyeballing. `trend` is the rank correlation of r against n over the
caps: +1.00 is a clean climb.

`history curve` holds the POPULATION FIXED and varies only how much of each
user's history is used: the deep users, scored on their first N attempts. Bands
still confound volume with population -- heavy users differ from light users in
more than attempt count -- so this is the one that isolates the effect. It is
also the sharpest test of whether an axis is real at all: reliability is
tau^2 / (tau^2 + sigma^2/n), so an axis with genuine between-user variance MUST
climb toward 1 as n grows. A flat curve says tau^2 is ~0, whichever level it
sits at.

THE INNER FIT IS A DAMPED NEWTON and a user whose fit does not converge is
DROPPED and counted. Plain Newton at a unit step diverges on a few users per
hundred -- the loadings carry outliers an order of magnitude past their own sd,
and a half that happens to contain one is ill-conditioned -- and an undamped run
then silently returns whatever the last iteration held. Four such users out of
224 once took the positive control from +0.77 to -0.18.

Correlation is reported BOTH ways for the same reason: Pearson is not robust to
one far-out user, Spearman is. Where they disagree the estimate is being carried
by its extremes, which is a fact about the fit rather than a choice of summary.

`xTHETA` is the leak the positive control cannot catch. The control column spans
what a uniform ability shift expresses, through a_p; an axis that tracks ability
enters through g_k instead, so it slips past. `resid` is the half-half
correlation after projecting each half's positions off the user's fitted ability.
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

HALF_RIDGE = 0.5          # numerical only, and identical across arms so the
                          # arms stay comparable whatever ridge each was fit at
BUCKETS = ((125, 250), (250, 500), (500, 1000), (1000, 10 ** 9))
HISTORY_CAPS = (125, 250, 500, 1000)
MAX_NEWTON = 200


def _fit_v(g, off, cc, y, lam, ridge):
    """Damped Newton on v with everything else FIXED. z = a(theta-b) + g.v, and
    the 4PL floor/lapse enter through dp/dz, so the weights are not p(1-p).
    Returns (v, converged)."""
    rng = 1 - cc - lam

    def obj(v):
        s = 1.0 / (1.0 + np.exp(-np.clip(off + g @ v, -35, 35)))
        p = np.clip(cc + rng * s, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p)).sum() + ridge * (v @ v)

    v = np.zeros(g.shape[1])
    f = obj(v)
    for _ in range(MAX_NEWTON):
        s = 1.0 / (1.0 + np.exp(-np.clip(off + g @ v, -35, 35)))
        p = np.clip(cc + rng * s, 1e-9, 1 - 1e-9)
        dp = rng * s * (1 - s)
        grad = g.T @ ((y - p) * dp / (p * (1 - p))) - 2 * ridge * v
        w = dp * dp / (p * (1 - p))
        H = (g * w[:, None]).T @ g + 2 * ridge * np.eye(g.shape[1])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return v, False
        dec = float(grad @ step)          # Newton decrement, curvature-weighted
        if dec < 0:                       # not an ascent direction
            return v, False
        if dec < 1e-10:
            return v, True
        t = 1.0
        while t > 1e-12:                  # Armijo on the penalized objective
            fn = obj(v + t * step)
            if fn <= f - 1e-4 * t * dec:
                v, f = v + t * step, fn
                break
            t /= 2
        else:
            return v, False
    return v, False


def _load(dump):
    d = np.load(dump, allow_pickle=False)
    need = ("loading", "difficulty", "slope", "floor", "lapse", "theta", "tid", "user")
    missing = [n for n in need if n not in d.files]
    if missing:
        raise SystemExit(f"{dump} lacks {missing}: split-half needs a --dump-params fit")
    return {
        "G": np.asarray(d["loading"], float),
        "b": np.asarray(d["difficulty"], float),
        "a": np.asarray(d["slope"], float),
        "c": np.asarray(d["floor"], float),
        "lam": float(np.asarray(d["lapse"]).ravel()[0]),
        "th": np.asarray(d["theta"], float).ravel(),
        "tid": {str(t): i for i, t in enumerate(d["tid"])},
        "uid": {str(u): i for i, u in enumerate(d["user"])},
    }


def _sequences(P, sql_path=None):
    """Each dump user's attempts in TIME ORDER, which the history curve needs --
    'their first N' is only meaningful against the order they happened in."""
    import sqlite3
    path = sql_path or os.path.join(PROJECT_ROOT, "db/d1.sqlite")
    con = sqlite3.connect(path)
    rows = {}
    for u, p_, s in con.execute(
            "SELECT user_id,problem,success FROM tsumego_reports ORDER BY id"):
        iu, ip = P["uid"].get(u), P["tid"].get(p_)
        if iu is None or ip is None:
            continue
        rows.setdefault(iu, []).append((ip, int(s)))
    con.close()
    return rows


def _halves(P, ipx, y, uidx, ridge):
    """Fit both parity halves. Column 0 is a per-user ABILITY shift (a shift d in
    theta enters the logit as a_p*d, so its column is the slope); columns 1..K
    are the axes. Fitting them together is what makes the axis halves comparable
    -- and column 0 doubles as the POSITIVE CONTROL, a trait nobody doubts is
    real, so a null on the axes only means something if this comes back high."""
    ia = np.arange(len(ipx)) % 2 == 0
    out = []
    for m in (ia, ~ia):
        ip = ipx[m]
        if y[m].std() == 0:
            return None
        g = np.column_stack([P["a"][ip], P["G"][ip]])
        off = P["a"][ip] * (P["th"][uidx] - P["b"][ip])
        v, ok = _fit_v(g, off, np.clip(P["c"][ip], 0.0, 1.0), y[m], P["lam"], ridge)
        if not ok:
            return None
        out.append(v)
    return out


def _n50_and_trend(caps, rs):
    """(N50, trend) from the reliability curve.

    N50 fits (1/r - 1) = rho/n through the origin over the caps where r is in
    (0, 1), rho being sigma^2/tau^2 in attempt units -- so rho IS the attempts at
    which reliability reaches 0.5. Through the origin because the intercept is
    pinned at 1 by the model rather than free.

    trend is Spearman on (n, r). With four caps it is coarse by construction, so
    it is a direction, not an effect size."""
    from scipy.stats import spearmanr
    ok = [(n, r) for n, r in zip(caps, rs) if np.isfinite(r) and 0 < r < 1]
    n50 = float("inf")
    if len(ok) >= 2:
        x = np.array([1.0 / n for n, _ in ok])
        y = np.array([1.0 / r - 1.0 for _, r in ok])
        denom = float((x * x).sum())
        if denom > 0:
            rho = float((x * y).sum() / denom)
            n50 = rho if rho > 0 else float("inf")
    fin = [(n, r) for n, r in zip(caps, rs) if np.isfinite(r)]
    trend = (spearmanr([n for n, _ in fin], [r for _, r in fin]).statistic
             if len(fin) >= 3 else float("nan"))
    return n50, trend


def _boot_ci(a, b, n_boot=2000, seed=0):
    """Percentile CI on the Spearman-Brown reliability, resampling USERS.

    The halves are already fitted, so a resample is just a re-correlation --
    cheap enough to run per axis. This is what says whether two seeds differing
    by 0.39 vs 0.74 are disagreeing or merely both noisy at n=58."""
    rng = np.random.default_rng(seed)
    n = len(a)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        aa, bb = a[i], b[i]
        if aa.std() == 0 or bb.std() == 0:
            continue
        r = np.corrcoef(aa, bb)[0, 1]
        if np.isfinite(r) and r > -1:
            out.append(2 * r / (1 + r))
    if len(out) < n_boot // 4:
        return float("nan"), float("nan")
    return tuple(np.percentile(out, [2.5, 97.5]))


def _table(A, B, CA, CB, TH, labels, head):
    from scipy.stats import spearmanr
    sb = lambda r: 2 * r / (1 + r) if (np.isfinite(r) and r > -1) else float("nan")
    thc = TH - TH.mean()
    print(head)
    print(f"    {'axis':<18}{'pearson':>9}{'S-B':>8}{'spearman':>10}{'S-B':>8}"
          f"{'xTHETA':>8}{'resid':>8}")
    for k in range(A.shape[1]):
        rp = np.corrcoef(A[:, k], B[:, k])[0, 1]
        rs = spearmanr(A[:, k], B[:, k]).statistic
        xa = 0.5 * (np.corrcoef(A[:, k], thc)[0, 1] + np.corrcoef(B[:, k], thc)[0, 1])
        pa = lambda x: x - thc * (np.dot(x, thc) / np.dot(thc, thc))
        rr = np.corrcoef(pa(A[:, k]), pa(B[:, k]))[0, 1]
        nm = labels[k] if k < len(labels) and labels[k] else f"axis {k}"
        print(f"    {nm:<18}{rp:>9.3f}{sb(rp):>8.3f}{rs:>10.3f}{sb(rs):>8.3f}"
              f"{xa:>8.3f}{sb(rr):>8.3f}")
    rp = np.corrcoef(CA, CB)[0, 1]
    rs = spearmanr(CA, CB).statistic
    print(f"    {'CTRL (ability)':<18}{rp:>9.3f}{sb(rp):>8.3f}{rs:>10.3f}{sb(rs):>8.3f}")


def split_half_report(dump, labels=(), ridge=HALF_RIDGE, sql_path=None):
    """Disjoint history bands, then the fixed-population history curve."""
    P = _load(dump)
    rows = _sequences(P, sql_path)
    K = P["G"].shape[1]
    print(f"\n  split-half reliability — dump users, halves fit at ridge {ridge}")
    print("  (Spearman-Brown corrected; xTHETA = ability leak the control misses)")

    for lo, hi in BUCKETS:
        A, B, CA, CB, TH, drop = [], [], [], [], [], 0
        for uidx, r in rows.items():
            if not (lo <= len(r) < hi):
                continue
            arr = np.array(r)
            res = _halves(P, arr[:, 0], arr[:, 1].astype(float), uidx, ridge)
            if res is None:
                drop += 1
                continue
            fa, fb = res
            A.append(fa[1:]); B.append(fb[1:])
            CA.append(fa[0]); CB.append(fb[0]); TH.append(P["th"][uidx])
        if len(A) < 20:
            print(f"\n  {lo}-{'inf' if hi > 10**8 else hi}: {len(A)} users — too few, skipped")
            continue
        band = f"{lo}-{'inf' if hi > 10 ** 8 else hi - 1}"
        _table(np.array(A), np.array(B), np.array(CA), np.array(CB), np.array(TH),
               list(labels), f"\n  attempts {band}   n={len(A)}"
                             f"{f'  ({drop} dropped, not converged)' if drop else ''}")

    deep = {u: r for u, r in rows.items() if len(r) >= max(HISTORY_CAPS)}
    if len(deep) < 20:
        print(f"\n  history curve: only {len(deep)} users at "
              f"{max(HISTORY_CAPS)}+ attempts — too few, skipped")
        return
    print(f"\n  history curve — SAME {len(deep)} users at {max(HISTORY_CAPS)}+ "
          f"attempts, scored on their first N")
    print("  (population fixed, so only data volume varies; a real axis must "
          "climb toward 1)")
    per_cap = {}
    for cap in HISTORY_CAPS:
        A, B, CA, CB, TH, who, drop = [], [], [], [], [], [], 0
        for uidx, r in deep.items():
            arr = np.array(r[:cap])
            res = _halves(P, arr[:, 0], arr[:, 1].astype(float), uidx, ridge)
            if res is None:
                drop += 1
                continue
            fa, fb = res
            A.append(fa[1:]); B.append(fb[1:])
            CA.append(fa[0]); CB.append(fb[0]); TH.append(P["th"][uidx])
            who.append(uidx)
        if len(A) < 20:
            print(f"\n  first {cap}: {len(A)} users — too few, skipped")
            continue
        A, B, CA, CB, TH = (np.array(x) for x in (A, B, CA, CB, TH))
        per_cap[cap] = (A, B, CA, CB, who)
        _table(A, B, CA, CB, TH, list(labels),
               f"\n  first {cap} attempts   n={len(A)}"
               f"{f'  ({drop} dropped, not converged)' if drop else ''}")
    if len(per_cap) >= 2:
        _curve_summary(per_cap, list(labels), K)


def _curve_summary(per_cap, labels, K):
    """The curve as numbers rather than a shape to squint at.

    N50 is the attempts at which reliability reaches 0.5, fitted across the caps
    (see _n50_and_trend). `inf` means the curve carries no climb to extrapolate,
    which is the same statement as tau^2 ~ 0. The CI is on the DEEPEST cap and is
    over users, so it says how much of a gap between two seeds is just n=58.

    The CTRL row is the calibration: a trait nobody doubts is real, measured the
    same way, so its N50 is the scale everything else is read against."""
    caps = sorted(per_cap)
    sb = lambda r: 2 * r / (1 + r) if (np.isfinite(r) and r > -1) else float("nan")
    top = caps[-1]
    print(f"\n  curve summary — trend, attempts needed for reliability 0.5, "
          f"and user-bootstrap CIs (95%)")
    print(f"    {'axis':<18}" + "".join(f"{('r@' + str(c)):>8}" for c in caps)
          + f"{'trend':>8}{'N50':>9}{'N50 CI':>18}{'r CI @' + str(top):>18}")
    rows = [(k, labels[k] if k < len(labels) and labels[k] else f"axis {k}")
            for k in range(K)] + [(None, "CTRL (ability)")]
    # Users present at EVERY cap, so one bootstrap replicate resamples a whole
    # curve rather than four independent ones — the caps are nested within user
    # and pretending otherwise would understate how correlated they are.
    common = sorted(set.intersection(*[set(per_cap[c][4]) for c in caps]))
    pos = {c: {u: i for i, u in enumerate(per_cap[c][4])} for c in caps}
    for k, nm in rows:
        cols = {}
        for c in caps:
            A, B, CA, CB, _ = per_cap[c]
            cols[c] = ((CA, CB) if k is None else (A[:, k], B[:, k]))
        rs = [sb(np.corrcoef(*cols[c])[0, 1]) for c in caps]
        n50, trend = _n50_and_trend(caps, rs)
        lo, hi = _boot_ci(*cols[top])
        n50lo, n50hi = _boot_n50(caps, cols, pos, common)
        n50s = "inf" if not np.isfinite(n50) else f"{n50:,.0f}"
        rng = ("—" if not np.isfinite(n50lo)
               else f"[{n50lo:,.0f}, {'inf' if not np.isfinite(n50hi) else format(n50hi, ',.0f')}]")
        ci = ("—" if not np.isfinite(lo) else f"[{lo:+.2f}, {hi:+.2f}]")
        tr = "—" if not (np.isfinite(trend) and max(rs) > 0) else f"{trend:+.2f}"
        print(f"    {nm:<18}" + "".join(f"{r:>8.2f}" for r in rs)
              + f"{tr:>8}{n50s:>9}{rng:>18}{ci:>18}")
    print("    N50 inf = no climb to extrapolate, i.e. no between-user variance.")
    print("    trend — = every point at or below zero, so a rank climb means nothing.")


def _boot_n50(caps, cols, pos, common, n_boot=600, seed=0):
    """Percentile CI on N50, resampling users ONCE per replicate across all caps."""
    rng = np.random.default_rng(seed)
    sb = lambda r: 2 * r / (1 + r) if (np.isfinite(r) and r > -1) else float("nan")
    out = []
    if len(common) < 20:
        return float("nan"), float("nan")
    for _ in range(n_boot):
        pick = [common[i] for i in rng.integers(0, len(common), len(common))]
        rs = []
        for c in caps:
            idx = [pos[c][u] for u in pick]
            a, b = cols[c][0][idx], cols[c][1][idx]
            rs.append(sb(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan"))
        out.append(_n50_and_trend(caps, rs)[0])
    fin = [x for x in out if np.isfinite(x)]
    if len(fin) < n_boot // 10:
        return float("nan"), float("nan")
    lo = float(np.percentile(fin, 2.5))
    hi = (float(np.percentile(fin, 97.5)) if len(fin) > 0.75 * n_boot
          else float("inf"))          # >25% of replicates had no climb at all
    return lo, hi


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--ridge", type=float, default=HALF_RIDGE)
    a = ap.parse_args()
    split_half_report(a.dump, a.labels, a.ridge)


if __name__ == "__main__":
    main()
