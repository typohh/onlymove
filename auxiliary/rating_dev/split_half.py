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
    for cap in HISTORY_CAPS:
        A, B, CA, CB, TH, drop = [], [], [], [], [], 0
        for uidx, r in deep.items():
            arr = np.array(r[:cap])
            res = _halves(P, arr[:, 0], arr[:, 1].astype(float), uidx, ridge)
            if res is None:
                drop += 1
                continue
            fa, fb = res
            A.append(fa[1:]); B.append(fb[1:])
            CA.append(fa[0]); CB.append(fb[0]); TH.append(P["th"][uidx])
        if len(A) < 20:
            print(f"\n  first {cap}: {len(A)} users — too few, skipped")
            continue
        _table(np.array(A), np.array(B), np.array(CA), np.array(CB), np.array(TH),
               list(labels), f"\n  first {cap} attempts   n={len(A)}"
                             f"{f'  ({drop} dropped, not converged)' if drop else ''}")


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
