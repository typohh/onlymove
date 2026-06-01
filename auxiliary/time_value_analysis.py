"""Population-level estimate of the rank value of *time usage*.

One-off diagnostic (not part of the pipeline). Question: across many users, how
many ranks of measured ability is associated with spending more time per problem,
once we strip out (a) the user's overall speed habit, (b) problem-rating variance
(stronger users get harder problems that intrinsically take longer), and (c) the
user's external baseline ability (declared rank)?

We build, per user, the four constructs discussed:
  - results_rank   : ability from a pooled-slope Rasch fit over their successes,
                     mapped to rank units via M (rating_to_rank.json).
  - time_usage     : mean residual log-time after regressing log(time) on rating
                     over the whole population -> "longer than the difficulty
                     warrants", i.e. usage with the difficulty confound removed.
  - efficiency     : per-user within slope of success on residual-time (controlling
                     for rating), ridge-shrunk to the population mean.
  - declared_rank  : rank_surveys -> internal rank scalar (EGF identity, other
                     servers translated through rank_map.json).

then regress results_rank on time_usage with/without efficiency and declared_rank,
and report the time coefficient in *ranks per doubling of difficulty-adjusted time*.

NOTE: this is an observational between-user association. Controlling for declared
rank is the closest we get to "time bought ability beyond baseline", but it is not
a clean causal dose-response (that needs a randomized time-floor). The script prints
the caveat-relevant diagnostics (how much declared rank attenuates the coefficient).

Time data exists only for ~the trailing two weeks of the dump.

    ./.venv/bin/python3 auxiliary/time_value_analysis.py
"""
import json
import math
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "attempts", "database.sqlite")
RATINGS = {sz: os.path.join(ROOT, "ratings_v2", str(sz), "tsumego_ratings_final.csv")
           for sz in (9, 13)}
M_PATH = os.path.join(ROOT, "ratings_v2", "rating_to_rank.json")
RANKMAP_PATH = os.path.join(ROOT, "ratings_v2", "rank_map.json")

# Time sanity window (seconds). Below = misclick/instant; above = AFK tail.
TIME_LO, TIME_HI = 1.0, 180.0
MIN_ATTEMPTS = 20          # per-user inclusion threshold (timed first-time attempts)
RIDGE_EFF = 5.0            # shrinkage of per-user efficiency toward population mean


# --------------------------------------------------------------------------- M
def load_M():
    d = json.load(open(M_PATH))
    k = "9" if "9" in d else list(d.keys())[0]
    r = np.asarray(d[k]["ratings"], float)
    ri = np.asarray(d[k]["rank_indices"], float)
    order = np.argsort(r)
    return r[order], ri[order]

def rating_to_rank(x, mr, mri):
    return np.interp(x, mr, mri)


# ----------------------------------------------------------------- declared rank
def label_to_scalar(label):
    label = label.strip().lower()
    try:
        if label.endswith("k"):
            return -(int(label[:-1]) - 1)      # 1k=0, 5k=-4
        if label.endswith("d"):
            return int(label[:-1])             # 1d=1, 3d=3
    except ValueError:
        return None
    return None

def declared_internal(server, label, rankmap):
    s = label_to_scalar(label)
    if s is None or server not in rankmap:
        return None
    km = rankmap[server]
    return float(np.interp(s, km["server"], km["our"]))


# ------------------------------------------------------------------ data loading
def load_attempts():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT user_id, problem, success, size, time_used, created_at "
        "FROM tsumego_reports WHERE is_repeat=0", con)
    con.close()

    rmap = {}
    for sz, path in RATINGS.items():
        r = pd.read_csv(path)
        for tid, rating in zip(r.tsumego_id, r.rating):
            rmap[(sz, tid)] = float(rating)
    df["rating"] = [rmap.get((sz, p)) for sz, p in zip(df["size"], df["problem"])]
    df = df.dropna(subset=["rating"]).copy()
    df["success"] = df["success"].astype(float)
    df["t_s"] = df["time_used"] / 1000.0
    df["timed"] = df["t_s"].between(TIME_LO, TIME_HI)
    return df


# ------------------------------------------------------- pooled-slope Rasch fit
def fit_rasch(df):
    """logit P = a_u + s*rating_centered ; returns per-user theta (rating where P=.5)."""
    users = df["user_id"].unique()
    uidx = {u: i for i, u in enumerate(users)}
    ui = df["user_id"].map(uidx).to_numpy()
    rc = (df["rating"] - df["rating"].mean()).to_numpy()
    y = df["success"].to_numpy()
    nU = len(users)

    def nll(p):
        a = p[:nU]
        s = p[nU]
        z = a[ui] + s * rc
        # stable log-loss + grad
        pr = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        loss = -(y * np.log(pr + eps) + (1 - y) * np.log(1 - pr + eps)).sum()
        loss += 1e-3 * (a @ a)          # tiny ridge on user effects for conditioning
        g = pr - y
        ga = np.bincount(ui, weights=g, minlength=nU) + 2e-3 * a
        gs = (g * rc).sum()
        return loss, np.concatenate([ga, [gs]])

    p0 = np.zeros(nU + 1)
    p0[nU] = -0.1
    res = minimize(nll, p0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=500))
    a = res.x[:nU]
    s = res.x[nU]
    mean_rating = df["rating"].mean()
    # theta in raw rating units: P=.5 -> a_u + s*(theta-mean)=0
    theta = mean_rating - a / s
    return pd.Series(theta, index=users), s


# ------------------------------------------------------------------------- main
def main():
    if not os.path.exists(DB):
        sys.exit(f"missing {DB} (import the dump first, see CLAUDE.md)")
    mr, mri = load_M()
    rankmap = json.load(open(RANKMAP_PATH))

    df = load_attempts()
    print(f"first-time attempts with a rating: {len(df):,}  "
          f"({df['timed'].sum():,} within the {TIME_LO:.0f}-{TIME_HI:.0f}s time window)")

    # population log(time)~rating baseline on timed rows -> residual time
    td = df[df["timed"]].copy()
    td["logt"] = np.log(td["t_s"])
    A = np.column_stack([np.ones(len(td)), td["rating"].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, td["logt"].to_numpy(), rcond=None)
    td["resid_t"] = td["logt"] - A @ coef
    print(f"population log(time) ~ rating slope = {coef[1]:+.4f} ln-s per rating unit "
          f"(harder problems take {'longer' if coef[1] > 0 else 'less'})")

    # per-user inclusion
    counts = td.groupby("user_id").size()
    keep = counts[counts >= MIN_ATTEMPTS].index
    td = td[td["user_id"].isin(keep)].copy()
    print(f"users with >= {MIN_ATTEMPTS} timed first-time attempts: {len(keep)}")

    # ability from ALL in-window first-time attempts of these users (timed or not)
    win_lo = td["created_at"].min()
    abil_df = df[(df["user_id"].isin(keep)) & (df["created_at"] >= win_lo)].copy()
    theta, s = fit_rasch(abil_df)
    print(f"Rasch shared rating slope s = {s:+.4f} (expect <0: harder -> lower P)")

    # per-user efficiency: success ~ rating + resid_t, take resid_t coef, ridge-shrunk
    def user_eff(g):
        if len(g) < MIN_ATTEMPTS or g["success"].nunique() < 2:
            return np.nan
        X = np.column_stack([np.ones(len(g)), g["rating"], g["resid_t"]])
        yy = g["success"].to_numpy()
        b = np.zeros(3)
        for _ in range(50):                     # IRLS w/ ridge
            z = X @ b
            pr = 1 / (1 + np.exp(-z))
            W = np.clip(pr * (1 - pr), 1e-6, None)
            grad = X.T @ (pr - yy) + RIDGE_EFF * b
            H = X.T @ (X * W[:, None]) + RIDGE_EFF * np.eye(3)
            step = np.linalg.solve(H, grad)
            b -= step
            if np.max(np.abs(step)) < 1e-7:
                break
        return b[2]
    eff = td.groupby("user_id").apply(user_eff, include_groups=False)

    # assemble per-user table
    tbl = pd.DataFrame({
        "results_rating": theta.reindex(keep),
        "median_time_s": td.groupby("user_id")["t_s"].median(),
        "time_usage": td.groupby("user_id")["resid_t"].mean(),   # difficulty-purged
        "efficiency": eff,
        "n": counts.reindex(keep),
    })
    tbl["results_rank"] = rating_to_rank(tbl["results_rating"].to_numpy(), mr, mri)

    # declared rank (EGF identity, else translate; prefer EGF, else mean of servers)
    con = sqlite3.connect(DB)
    sv = pd.read_sql_query("SELECT user_id, server, rank FROM rank_surveys", con)
    con.close()
    decl = {}
    for uid, g in sv.groupby("user_id"):
        egf = g[g["server"] == "EGF"]
        if len(egf):
            v = declared_internal("EGF", egf.iloc[0]["rank"], rankmap)
        else:
            vals = [declared_internal(r.server, r["rank"], rankmap) for _, r in g.iterrows()]
            vals = [x for x in vals if x is not None]
            v = float(np.mean(vals)) if vals else None
        if v is not None:
            decl[uid] = v
    tbl["declared"] = tbl.index.map(decl)

    # de-contamination: baseline ability from PRE-window first-time attempts
    # (disjoint from the timed attempts, so it can't be inflated by time usage).
    pre = df[(df["user_id"].isin(keep)) & (df["created_at"] < win_lo)].copy()
    pre_counts = pre.groupby("user_id").size()
    pre_keep = pre_counts[pre_counts >= MIN_ATTEMPTS].index
    pre = pre[pre["user_id"].isin(pre_keep)]
    base_theta, base_s = fit_rasch(pre)
    tbl["baseline_rank"] = rating_to_rank(
        base_theta.reindex(keep).to_numpy(), mr, mri)
    print(f"users with an independent pre-window baseline (>= {MIN_ATTEMPTS} attempts): "
          f"{len(pre_keep)}")

    # cleanest de-contaminated outcome: in-window over-performance vs that baseline.
    # success - P_expected(rating | baseline ability), averaged per user. Outcome
    # uses baseline from DISJOINT attempts, so it isn't circular with time_usage.
    bt = base_theta.reindex(td["user_id"]).to_numpy()
    p_exp = 1.0 / (1.0 + np.exp(-base_s * (td["rating"].to_numpy() - bt)))
    td["overperf"] = td["success"].to_numpy() - p_exp
    tbl["overperf"] = td.groupby("user_id")["overperf"].mean()

    out = os.path.join(ROOT, "auxiliary", "time_value_per_user.csv")
    tbl.to_csv(out)
    print(f"\nper-user table -> {out}  ({len(tbl)} users, "
          f"{tbl['declared'].notna().sum()} with declared rank)\n")

    # ---- correlations
    print("=== correlations (Pearson) ===")
    cc = tbl[["results_rank", "median_time_s", "time_usage", "efficiency", "declared"]].corr()
    print(cc.round(3).to_string())

    # ---- regressions: results_rank ~ predictors  (OLS w/ z-scored predictors -> report ranks)
    LN2 = math.log(2.0)
    def ols(cols, data):
        d = data.dropna(subset=["results_rank"] + cols)
        X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
        yv = d["results_rank"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        dof = len(d) - X.shape[1]
        sigma2 = (resid @ resid) / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        ss_tot = ((yv - yv.mean()) ** 2).sum()
        r2 = 1 - (resid @ resid) / ss_tot
        return d, beta, se, r2, dof

    print("\n=== results_rank (in ranks) regressed on time usage ===")
    print("time_usage coef reported as RANKS PER DOUBLING of difficulty-adjusted time "
          f"(x ln2={LN2:.3f}).\n")
    for cols in (["time_usage"],
                 ["time_usage", "efficiency"],
                 ["time_usage", "declared"],
                 ["time_usage", "efficiency", "declared"]):
        d, beta, se, r2, dof = ols(cols, tbl)
        parts = []
        for j, c in enumerate(cols):
            b, e = beta[j + 1], se[j + 1]
            if c in ("time_usage",):
                parts.append(f"{c}={b * LN2:+.2f}±{e * LN2:.2f} rk/2x (t={b/e:+.1f})")
            else:
                parts.append(f"{c}={b:+.2f}±{e:.2f} (t={b/e:+.1f})")
        print(f"  n={len(d):3d} R2={r2:.3f} | " + " | ".join(parts))

    print("\n=== de-contaminated: independent pre-window baseline ===")
    for cols in (["time_usage", "baseline_rank"],
                 ["time_usage", "efficiency", "baseline_rank"]):
        d, beta, se, r2, dof = ols(cols, tbl)
        parts = []
        for j, c in enumerate(cols):
            b, e = beta[j + 1], se[j + 1]
            unit = " rk/2x" if c == "time_usage" else ""
            scale = LN2 if c == "time_usage" else 1.0
            parts.append(f"{c}={b*scale:+.2f}±{e*scale:.2f}{unit} (t={b/e:+.1f})")
        print(f"  n={len(d):3d} R2={r2:.3f} | " + " | ".join(parts))

    # over-performance outcome (probability units; outcome is baseline-anchored,
    # so the time coefficient is the clean within-window dose, not a fitted-on-itself
    # artifact). Reported as Δ(success prob) per doubling of difficulty-adjusted time.
    d = tbl.dropna(subset=["overperf", "time_usage"])
    X = np.column_stack([np.ones(len(d)), d["time_usage"]])
    b, *_ = np.linalg.lstsq(X, d["overperf"].to_numpy(), rcond=None)
    r = d["overperf"].to_numpy() - X @ b
    se = math.sqrt((r @ r) / (len(d) - 2) * np.linalg.inv(X.T @ X)[1, 1])
    print(f"\n  overperf ~ time_usage : {b[1]*LN2:+.3f}±{se*LN2:.3f} success-prob per "
          f"doubling (n={len(d)}, t={b[1]/se:+.1f})")
    print("  (positive = users who linger beat their own established baseline; this "
          "outcome can't be inflated by time because the baseline is from other attempts.)")

    print("\nReading: the time_usage coefficient is the headline 'ranks bought per "
          "doubling of (difficulty-adjusted) time'. Compare across rows: how much it\n"
          "attenuates when 'declared' (external baseline ability) is added tells you how\n"
          "much is plausibly time-buys-ability vs strong-players-also-linger.")


if __name__ == "__main__":
    main()
