"""Fit server↔EGF coupling SHAPES from the 2018 cross-server survey.

rank_survey_2018.csv: each respondent declared their rank on several servers at
once. That directly calibrates the relative server↔server (hence server↔EGF)
couplings, independent of any rating model — useful as a PRIOR on the coupling
SHAPE (relative rank spacing within a server), especially for servers our current
survey barely covers.

Model: an ordinal factor model. Each respondent r has a latent EGF rank θ_r; for a
declared (server s, rank h):  P(declared ≤ h | θ_r) = expit(τ_{s,h} − θ_r). The EGF
column is the identity anchor (pins θ_r to the EGF scale via a tight Gaussian).

Returns per-server coupling curves {server: (server_scalars, egf_scalars)} (ascending
server-rank boundary scalars → internal/EGF scalar), directly interpolatable. NOTE:
this is the 2018 *shape*; callers should allow a per-server affine (offset+scale) to
absorb 7 years of (non-uniform) online-rank inflation — see
docs/rank_mapping_experiments.md (the dan end has drifted, the kyu/SDK end has not).
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pipeline.fit_rank_curves import (  # noqa: E402
    ordinal_nll_vec, _unpack_thresholds, build_server_curves)

CSV_DEFAULT = os.path.join(PROJECT_ROOT, "rank_survey_2018.csv")
# cross-server-survey column header -> our server key, per file (drop columns for
# servers we don't ship couplings for).
COLMAP_2018 = {"OGS": "OGS", "KGS": "KGS", "IGS": "IGS", "Foxwq": "Fox",
               "Tygem": "Tygem", "WBaduk": "WBaduk", "AGA": "AGA", "EGF": "EGF"}
COLMAP_2024 = {"OGS": "OGS", "KGS": "KGS", "Fox": "Fox", "Tygem": "Tygem",
               "IGS": "IGS", "GoQuest rank": "GoQuest", "EGF": "EGF", "AGA": "AGA"}
COLMAP = COLMAP_2018  # back-compat default
EGF_ANCHOR_SD = 0.7   # tightness of the EGF-identity anchor on θ (rank units)
SHAPE_LAM = 5.0       # uniform-spacing prior on each 2018 server's own thresholds


def _scalar(v):
    if not isinstance(v, str):
        return None
    m = re.fullmatch(r"(\d+)\s*([kd])", v.strip().lower())
    if not m:
        return None
    n = int(m.group(1))
    return float((1 - n) if m.group(2) == "k" else n)


def fit_crossserver_couplings(csv_path, colmap, min_server_samples=5, rank_padding=1):
    """Fit server↔EGF couplings from a cross-server survey CSV (ordinal factor
    model). Returns {server: (server_scalars, egf_scalars)} — build_server_curves
    shape (EGF identity included), interpolatable at any server-rank scalar."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path).rename(columns=colmap)
    recs = defaultdict(list)                       # resp -> [(server, scalar)]
    for i, row in df.iterrows():
        for s in colmap.values():
            if s in df.columns:
                g = _scalar(row[s])
                if g is not None:
                    recs[i].append((s, g))
    recs = {r: v for r, v in recs.items() if v}
    resp = list(recs)
    ridx = {r: i for i, r in enumerate(resp)}

    servers = [s for s in colmap.values() if s != "EGF"]
    blocks = {}
    for s in servers:
        decl = [int(round(g)) for r in recs for (sv, g) in recs[r] if sv == s]
        if len(decl) < min_server_samples:
            continue
        g_min, g_max = min(decl) - rank_padding, max(decl) + rank_padding
        K = g_max - g_min
        if K < 1:
            continue
        blocks[s] = dict(g_min=g_min, g_max=g_max, N=g_max - g_min + 1,
                         baseline=np.ones(K - 1) if K > 1 else np.array([]))

    nR = len(resp)
    layout, x0 = [], []
    for r in resp:                                  # init θ from EGF else mean
        egf = [g for sv, g in recs[r] if sv == "EGF"]
        x0.append(float(egf[0]) if egf else float(np.mean([g for _, g in recs[r]])))
    idx = nR
    for s, blk in blocks.items():
        n = len(blk["baseline"]) + 1
        layout.append((s, idx, n))
        x0.append(float(blk["g_min"]))
        x0.extend([0.0] * len(blk["baseline"]))
        idx += n
    x0 = np.asarray(x0, float)

    def unpack(p):
        th = p[:nR]
        tau = {s: _unpack_thresholds(p[st:st + n], blocks[s]["baseline"])
               for s, st, n in layout}
        return th, tau

    def obj(p):
        th, tau = unpack(p)
        nll = 0.0
        for r in resp:
            t = th[ridx[r]]
            for sv, g in recs[r]:
                if sv == "EGF":
                    nll += 0.5 * ((t - g) / EGF_ANCHOR_SD) ** 2
                elif sv in blocks:
                    b = blocks[sv]
                    k = max(0, min(b["N"] - 1, int(round(g)) - b["g_min"]))
                    nll += ordinal_nll_vec(tau[sv], np.array([t]), np.array([k]))
        pen = SHAPE_LAM * sum(float(np.sum(p[st + 1:st + n] ** 2))
                              for _, st, n in layout)
        return nll + pen

    res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": 20000})
    _, tau = unpack(res.x)
    fits = {s: ({g: float(tau[s][i])
                 for i, g in enumerate(range(blocks[s]["g_min"] + 1,
                                             blocks[s]["g_max"] + 1))}, 0, 0)
            for s in blocks}
    return build_server_curves(fits)


def fit_2018_couplings(csv_path=CSV_DEFAULT, **kw):
    return fit_crossserver_couplings(csv_path, COLMAP_2018, **kw)


def fit_2024_couplings(csv_path=os.path.join(PROJECT_ROOT, "rank_survey_2024.csv"), **kw):
    return fit_crossserver_couplings(csv_path, COLMAP_2024, **kw)


if __name__ == "__main__":
    cur = fit_2018_couplings()
    print(f"fitted 2018 couplings for: {[s for s in cur if s != 'EGF']}")
    for s, c in cur.items():
        if s == "EGF":
            continue
        print(f"  {s}: {len(c['server'])} knots, "
              f"server {c['server'][0]:+.0f}..{c['server'][-1]:+.0f} → "
              f"EGF {c['our'][0]:+.1f}..{c['our'][-1]:+.1f}")
