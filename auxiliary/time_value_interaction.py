"""Rank value of time from (time-tendency x enforced-server-speed) interaction.

The cross-server main-effect approach fails: a server's time control is a server
constant, perfectly collinear with that server's rank inflation. But we ALSO know,
from our own logs, each user's *time-usage tendency* -- and a fixed blitz clock
handicaps a slow, time-reliant player far more than a naturally fast one. That
heterogeneity is NOT a server constant, so it is identified separately from
inflation.

Model (declared ranks in native per-server scalar):

    native_rank[u,s] = alpha_user[u] + gamma_server[s]
                       + lambda * ( speed[s] * time_tendency[u] ) + eps

  - alpha_user : ability (within-user identification; kills between-user selection)
  - gamma_server : ALL server constants -- inflation AND average speed effect
  - speed[s] : +1 fast (GoQuest), -1 slow (EGF/AGA), 0 mixed
  - time_tendency[u] : standardized log median solving time from our timed logs
  - lambda : the prize. lambda<0 means time-heavy users sit relatively LOWER on
    fast servers / HIGHER on slow servers -> being denied their clock costs them
    rank -> time has value. Inflation can't drive lambda (it's in gamma_server).

    ./.venv/bin/python3 auxiliary/time_value_interaction.py
"""
import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "attempts", "database.sqlite")

SPEED = {"GoQuest": +1.0, "EGF": -1.0, "AGA": -1.0,
         "OGS": 0.0, "KGS": 0.0, "Fox": 0.0, "Tygem": 0.0, "IGS": 0.0, "WBaduk": 0.0}
TIME_LO, TIME_HI = 1.0, 180.0
MIN_TIMED = 20


def label_to_scalar(label):
    label = str(label).strip().lower()
    try:
        if label.endswith("k"):
            return -(int(label[:-1]) - 1)
        if label.endswith("d"):
            return int(label[:-1])
    except ValueError:
        return None
    return None


def user_time_tendency(con, pct=0.5):
    """Standardized log time-usage per user at percentile `pct`. The median (0.5)
    is compressed (all tsumego are ~10s); the upper tail (e.g. 0.9) better captures
    a player's willingness to sink time into hard reading -- the trait a fixed clock
    truncates."""
    df = pd.read_sql_query(
        "SELECT user_id, time_used FROM tsumego_reports "
        "WHERE is_repeat=0 AND time_used>0", con)
    df["t_s"] = df.time_used / 1000.0
    df = df[df.t_s.between(TIME_LO, TIME_HI)]
    g = df.groupby("user_id").t_s
    stat = g.quantile(pct)[g.count() >= MIN_TIMED]
    lt = np.log(stat)
    return (lt - lt.mean()) / lt.std(), stat


def fit_lambda(d):
    """OLS native_rank ~ user FE + server FE + speed*tendency. Returns lambda."""
    users = sorted(d.user_id.unique())
    servers = sorted(d.server.unique())
    uidx = {u: i for i, u in enumerate(users[1:], start=1)}   # drop user[0]
    sidx = {s: j for j, s in enumerate(servers[1:], start=1)}  # drop server[0]
    cols = 1 + (len(users) - 1) + (len(servers) - 1) + 1       # +intercept +inter
    X = np.zeros((len(d), cols))
    X[:, 0] = 1.0
    base_u = 1
    base_s = base_u + (len(users) - 1)
    inter_c = cols - 1
    for row, (_, r) in enumerate(d.iterrows()):
        if r.user_id in uidx:
            X[row, base_u + uidx[r.user_id] - 1] = 1.0
        if r.server in sidx:
            X[row, base_s + sidx[r.server] - 1] = 1.0
        X[row, inter_c] = SPEED[r.server] * r.tendency
    y = d.scalar.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(d) - np.linalg.matrix_rank(X))
    sigma2 = (resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(sigma2 * xtx_inv[inter_c, inter_c])
    return beta[inter_c], se, dof


def run(sv0, tend, label, med):
    sv = sv0.copy()
    sv["tendency"] = sv.user_id.map(tend)
    sv = sv.dropna(subset=["tendency"])
    nserv = sv.groupby("user_id").server.nunique()
    multi = nserv[nserv >= 2].index
    d = sv[sv.user_id.isin(multi)].copy()
    if d.user_id.nunique() < 5:
        print(f"[{label}] too few users"); return
    lam, se, dof = fit_lambda(d)

    rng = np.random.default_rng(0)
    us = list(multi)
    boot = []
    for _ in range(3000):
        samp = rng.choice(us, len(us), replace=True)
        rows = pd.concat([d[d.user_id == u].assign(user_id=f"{u}#{i}")
                          for i, u in enumerate(samp)])
        try:
            boot.append(fit_lambda(rows)[0])
        except Exception:
            pass
    boot = np.array(boot)
    print(f"[{label:9s}] n_users={d.user_id.nunique():2d} GoQuest={ (d.server=='GoQuest').sum() } "
          f"EGF/AGA={ d.server.isin(['EGF','AGA']).sum() } | "
          f"lambda={lam:+.3f}±{se:.3f} (t={lam/se:+.1f}) "
          f"95%CI[{np.percentile(boot,2.5):+.2f},{np.percentile(boot,97.5):+.2f}] "
          f"P(<0)={np.mean(boot<0):.2f} | per-SD fast-vs-slow={lam*2:+.2f} rk")


def main():
    con = sqlite3.connect(DB)
    sv = pd.read_sql_query("SELECT user_id, server, rank FROM rank_surveys", con)
    sv["scalar"] = sv["rank"].map(label_to_scalar)
    sv = sv.dropna(subset=["scalar"]).drop_duplicates(["user_id", "server"], keep="last")

    print("lambda<0 = time-heavy players handicapped on blitz => time has rank value")
    print("(server FE absorbs inflation; identification is within-user x time-tendency)\n")
    for pct, name in [(0.5, "median"), (0.75, "p75"), (0.9, "p90"), (0.95, "p95")]:
        tend, med = user_time_tendency(con, pct)
        run(sv, tend, name, med)
    con.close()


if __name__ == "__main__":
    main()
