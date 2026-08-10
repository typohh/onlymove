"""Production residual-axes machinery: supervised per-problem SKILL AXES.

The model behind pipeline.build_axes (which replaced the skew-ICA embedding
pipeline on 2026-07-18; the validation story lives in
docs/methodology.md). On top of a per-user difficulty baseline, K
per-problem axis loadings are learned so that per-user positions explain the
between-user response residuals — axes are DEFINED by validated human
variation, not activation statistics.

    logit P(success) = alpha_u + beta_u * rank_p      (per-user baseline)
                       + r(x_p)                        (population residual —
                                                        quarantined in its own
                                                        head, never shipped)
                       + sum_k g_k(x_p) * v_u[k]       (the axes)

Fixed training regimen (each element is load-bearing; see the doc):
  - g is a DIRECT LINEAR head on the features with unit-RMS columns (free
    loading scale lets one strong direction dodge the v-prior via cloning;
    a trunk-shared g can only combine r-relevant features).
  - lenient orthogonality penalty (mutual + vs r, batch cosines, weight 0.5).
  - EVERY axis warm-starts at a data-derived direction: the top eigenvectors
    of the centred between-user residual score moment (eigen_directions).
    At v=0 the loading gradient is identically zero, so an axis left at a
    random init escapes along its draw rather than along the data; these are
    the closed-form escape directions of that same linearization.

Features are the rating trainer's EXACT inputs (same loaders, same joint-PCA
cache/seed) plus the rating model's outputs — LITERAL parity with
pipeline.train_rating; the axes must never depend on the retired ICA pipeline.

K selection is INCREMENTAL: K grows from K_START while the held-out
walk-forward LL gain of the axes block (production-form 4PL baseline, 2-fold
by user) stays within STOP_FRACTION of the running peak; the selected K is the
last one before the breach, so the axis set grows naturally with data volume.
"""

import hashlib
import json
import os
from collections import defaultdict

import numpy as np

from utils.rating.features import (
    _cache_ids_path, load_backend_problem_params, load_joint_activations,
    load_joint_bot_winrate, load_joint_game_phase, load_joint_move_features,
    load_tightness_gaps, phase_ratio_z, zscore_neglog, SQLITE_PATH,
)
from utils.rating.fit_rank_curves import interp_extrap
from utils.rating.slope_calibration import load_client_lapse

from utils.paths import PROJECT_ROOT

# training regimen
# EPOCHS sits at the plateau of the held-out gain, and stopping short of it is
# NOT the conservative choice: an under-trained block leaves gain on the table
# invisibly, so the K search then compares candidates none of which converged.
# The upper side is bounded by a different quantity — the per-user coefficient
# displacement that distinguishes between-user weakness from population
# difficulty saturates EARLIER than the gain does, so there is a window where
# more training is uncompensated and, past the gain plateau, only displacement
# keeps creeping. This is the end of that window. Figures in CLAUDE.md.
EPOCHS = 400
SEED = 0
ORTHO_WEIGHT = 0.5
SEED_RIDGE = 10.0
BATCH = 8192
MIN_USER_ATTEMPTS = 50

# incremental-K selection
K_START = 3                 # floor; every candidate count is warm-started alike
K_MAX = 12                  # runaway backstop
STOP_FRACTION = 0.03        # stop when gain < (1 - this) * peak; select K-1

# walk-forward eval protocol (client replica — same as utils.rating.slope_calibration)
WINDOW = 500
EVAL_BLOCK = 50
WARMUP = 120
FLOOR_MAX = 0.78
CALIBRATION_P = 0.8


# ------------------------------------------------------------------ features
def corpus_tid_size():
    """tid -> board size from the tsumegos/<size>/ SGF sets — the GROUND TRUTH
    corpus. Never derive size membership from the activation-cache id lists:
    they are incremental and can carry stale ids from purged problems, and a
    tid landing in both lists silently mis-sizes its features and its
    per_problem.csv row (the 000d0d75 packaging abort, 2026-07-18)."""
    from pipeline.train_rating import SIZES
    out = {}
    for s in SIZES:
        d = os.path.join(PROJECT_ROOT, "tsumegos", str(s))
        for f in os.listdir(d):
            if f.endswith(".sgf"):
                out[f[:-4]] = s
    return out


def build_axis_features():
    """Literal rating-trainer feature parity. Returns (feats, scalars, rank,
    params, act_dim): feats tid -> raw feature vec (z-scored by the caller),
    scalars tid -> [wr, sc, phase] (eval baselines + seed targets), rank tid ->
    normalized problemRank, params tid -> (rating, slope, floor), act_dim =
    width of the leading joint-PCA activation block (per-cell fanout needs it)."""
    from pipeline.train_rating import PCA_DIM, SEED as PCA_SEED, SIZES

    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    act, act_dim = load_joint_activations(SIZES, tids_by_size, PCA_DIM, PCA_SEED)
    bw, _nb = load_joint_bot_winrate(SIZES, tids_by_size)
    mv, nmv = load_joint_move_features(SIZES, tids_by_size)
    ph, nph = load_joint_game_phase(SIZES, tids_by_size)
    zero_mv = np.zeros(nmv, dtype=np.float32)
    zero_ph = np.zeros(nph, dtype=np.float32)

    f_sc = zscore_neglog(load_tightness_gaps("score_gap"))
    f_wr = zscore_neglog(load_tightness_gaps("wr_gap"))
    f_ph = phase_ratio_z()

    curve = _load_rank_curve()
    rsf = {t: np.array(v, dtype=np.float32)
           for t, v in load_backend_problem_params().items()}

    tid_size = corpus_tid_size()
    feats, scalars, rank, params = {}, {}, {}, {}
    for key in act:
        size, t = key
        t = str(t)
        # corpus filter: skip cache entries whose tid isn't actually a problem
        # of this size (stale ids from purges; see corpus_tid_size)
        if tid_size.get(t) != size:
            continue
        if (key not in bw or t not in f_sc or t not in f_wr or t not in f_ph
                or t not in rsf):
            continue
        # M is applied with END-SLOPE EXTRAPOLATION, never bare np.interp: the
        # curve's knots span only where human declarations reach, so a bare
        # np.interp would saturate every problem past either end onto the end
        # knot. That is the client's problemRank (rank-decomposition.js
        # rankForSize) and fit_population_constants' weakness-prior rank feature,
        # both unbounded — a clamp here would break parity with both, flatten the
        # rank ordering across the whole hard tail, and hand the axes a column
        # that is constant exactly where the residual head has the most room.
        rk = float(interp_extrap(rsf[t][0], curve[0], curve[1]))
        feats[t] = np.concatenate([
            act[key], bw[key], mv.get(key, zero_mv), ph.get(key, zero_ph),
            [1.0 if size == 13 else 0.0],
            rsf[t], [rk],
        ]).astype(np.float32)
        scalars[t] = np.array([f_wr[t], f_sc[t], f_ph[t]], dtype=np.float32)
        rank[t] = rk
        params[t] = rsf[t]
    return feats, scalars, rank, params, int(act_dim)


def _load_rank_curve():
    import json as _json
    path = os.path.join(PROJECT_ROOT, "ratings", "rating_to_rank.json")
    m = _json.load(open(path))
    c = m["9"] if "9" in m else next(iter(m.values()))
    return np.array(c["ratings"]), np.array(c["ranks"])


def load_axis_attempts():
    """Per-user time-ordered first-time attempt rows [(user, tid, success)]."""
    import sqlite3
    con = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
    rows = con.execute(
        """SELECT user_id, problem, success FROM tsumego_reports
            WHERE is_repeat = 0 GROUP BY user_id, problem, created_at
            ORDER BY user_id, created_at, id"""
    ).fetchall()
    con.close()
    return rows


def seed_directions(F, S_mat, k_seed):
    """Ridge-regress each hand-crafted scalar (wr gap, score gap, phase ratio)
    onto the (z-scored) features; unit rows.

    NOT ON THE PRODUCTION PATH — eigen_directions is. Retained because the
    auxiliary comparison harnesses reproduce this arm, and because it is the
    only way to re-measure what the hand-crafted start was worth. It warm-starts
    at most three axes and leaves every later one random, which is what made the
    axis count and the initialization confounded with each other."""
    G0 = np.linalg.solve(F.T @ F + SEED_RIDGE * np.eye(F.shape[1]),
                         F.T @ S_mat[:, :k_seed])
    return (G0 / (np.linalg.norm(G0, axis=0, keepdims=True) + 1e-9)).T


def eigen_directions(rows_by_user, train_users, F, tid_index, rank, params,
                     lapse, k):
    """The production warm start: the top-k directions of BETWEEN-USER residual
    variation, computed from `train_users` only. Returns (k, n_feat), unit rows.

    At v=0 the bilinear term contributes nothing to the logit and the loading
    gradient is identically zero, so an axis left at a random init escapes along
    whichever way its draw happened to point. The direction the model wants is
    available in closed form. Linearizing around v=0, each user pulls the
    loadings along their own residual score under the axis-free baseline,

        s_u = sum_p (y_up - p_hat_up) * x_p

    and the dominant escape directions are the leading eigenvectors of the
    CENTRED second moment of those scores.

    Centring is not a detail. The uncentred leading direction is the one in
    which the baseline mispredicts for EVERYONE, which is population residual
    difficulty — the quantity the r head is quarantined to absorb and the axes
    are forbidden to carry. Seeding from it would put difficulty into every axis
    at initialization, for the mean-zero prior to fight back out.

    Train users only: directions taken from the users an axis set is scored on
    would leak, and the K selection would then reward the leak."""
    all_r0 = np.array([v[0] for v in params.values()])
    grid = np.arange(all_r0.min() - 3.0, all_r0.max() + 3.0, 0.05)
    scores = []
    for u in train_users:
        a = rows_by_user[u]
        tids = [t for t, _ in a]
        y = np.array([s for _, s in a], dtype=float)
        R0 = np.array([params[t][0] for t in tids])
        A = np.array([params[t][1] for t in tids])
        C = np.clip(np.array([params[t][2] for t in tids]), 0.0, FLOOR_MAX)
        span = np.maximum(1.0 - C - lapse, 1e-6)
        q = np.clip((CALIBRATION_P - C) / span, 1e-6, 1 - 1e-6)
        kap = np.log(q / (1 - q))
        th = _fit_theta(R0, A, C, span, kap, y, grid)
        P4 = _p4pl(th, R0, A, C, span, kap)
        off = np.log(P4 / (1 - P4))
        X = np.column_stack([np.ones(len(tids)),
                             np.array([rank[t] for t in tids])])
        w = _irls_off(X, y, off, ridge=1e-6)
        p = 1.0 / (1.0 + np.exp(-np.clip(off + X @ w, -35, 35)))
        idx = np.array([tid_index[t] for t in tids])
        scores.append(F[idx].astype(np.float64).T @ (y - p))
    S = np.stack(scores)
    S = S - S.mean(axis=0, keepdims=True)
    _ev, vec = np.linalg.eigh(S.T @ S / S.shape[0])
    top = vec[:, ::-1][:, :k].T
    return (top / (np.linalg.norm(top, axis=1, keepdims=True) + 1e-9)
            ).astype(np.float32)


# ------------------------------------------------------------------ training
def train_axes(rows_by_user, train_users, F, tid_index, k_axes, g_init, device,
               seed=SEED, epochs=EPOCHS, ortho_weight=ORTHO_WEIGHT):
    """One fixed-regimen fit on `train_users`. Returns (loadings (n_problems,
    k) scaled to unit-variance v, raw weight matrix W (k, fdim), rms (k,),
    v_sd (k,)). Loadings = (F_std @ W.T)/rms * v_sd."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    uids = sorted(train_users)
    uindex = {u: i for i, u in enumerate(uids)}
    rows = [(uindex[u], tid_index[t], s)
            for u in uids for t, s in rows_by_user[u]]
    U = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
    P = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    Y = torch.tensor([float(r[2]) for r in rows], device=device)
    counts = torch.bincount(U, minlength=len(uids)).clamp(min=1).float()
    X = torch.tensor(F, device=device)
    RANK = X[:, -1].clone()

    n_u, n_feat = len(uids), F.shape[1]
    trunk = nn.Sequential(
        nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
    ).to(device)
    head_r = nn.Linear(32, 1).to(device)
    g_lin = nn.Linear(n_feat, k_axes, bias=False).to(device)
    if g_init is not None:
        with torch.no_grad():
            g_lin.weight[:g_init.shape[0]].copy_(
                torch.tensor(g_init, dtype=torch.float32, device=device))
    alpha = torch.zeros(n_u, device=device, requires_grad=True)
    beta = torch.zeros(n_u, device=device, requires_grad=True)
    v = torch.zeros(n_u, k_axes, device=device, requires_grad=True)

    opt = torch.optim.Adam([
        {"params": list(trunk.parameters()) + list(head_r.parameters()),
         "lr": 1e-3, "weight_decay": 1e-4},
        {"params": list(g_lin.parameters()), "lr": 1e-3},
        {"params": [alpha, beta], "lr": 1e-2},
        {"params": [v], "lr": 1e-2},
    ])
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    n = len(rows)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            u, p, y = U[idx], P[idx], Y[idx]
            h = trunk(X[p])
            rh = head_r(h)
            gh = g_lin(X[p])
            gh = gh / (gh.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-6)
            logit = (alpha[u] + beta[u] * RANK[p] + rh.squeeze(-1)
                     + (gh * v[u]).sum(-1))
            loss = bce(logit, y) \
                + 0.5 * ((v[u] ** 2).sum(dim=1) / counts[u]).sum() / len(idx) \
                + 10.0 * (v.mean(dim=0) ** 2).sum()
            M = torch.cat([gh, rh], dim=1)
            Mc = M - M.mean(dim=0, keepdim=True)
            Mn = Mc / (Mc.norm(dim=0, keepdim=True) + 1e-9)
            Cm = Mn.T @ Mn
            loss = loss + ortho_weight * ((Cm - torch.eye(
                Cm.shape[0], device=Cm.device)) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        gh = g_lin(X)
        rms = gh.pow(2).mean(dim=0).sqrt() + 1e-6
        g_all = (gh / rms).cpu().numpy()
        v_sd = v.std(dim=0).cpu().numpy()
        g_all = g_all * v_sd[None, :]
        W = g_lin.weight.detach().cpu().numpy()
    return g_all, W, rms.cpu().numpy(), v_sd


# ------------------------------------------------------------------ eval
def _p4pl(theta, R0, A, C, span, kap):
    z = np.clip(A * (theta - R0) + kap, -35, 35)
    return np.clip(C + span / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)


def _fit_theta(R0, A, C, span, kap, y, grid):
    P = _p4pl(grid[:, None], R0[None, :], A[None, :], C[None, :],
              span[None, :], kap[None, :])
    ll = (y[None, :] * np.log(P) + (1 - y[None, :]) * np.log(1 - P)).sum(axis=1)
    i = int(np.argmax(ll))
    if 0 < i < len(grid) - 1:
        d = (ll[i - 1] - ll[i + 1]) / (2 * (ll[i - 1] - 2 * ll[i] + ll[i + 1]) + 1e-12)
        return float(grid[i] + np.clip(d, -1, 1) * (grid[1] - grid[0]))
    return float(grid[i])


def _irls_off(X, y, off, ridge=1e-6, w0=None, penalty=None):
    """Newton fit of a logistic model with a fixed offset.

    `penalty` is an optional per-parameter Gaussian prior precision (one entry
    per column). It REPLACES the scalar `ridge`; a zero entry leaves that
    parameter unpenalized, so the intercept and rank columns can stay free
    while the axis coefficients are shrunk. With `penalty=None` this is exactly
    the scalar-ridge fit it has always been."""
    pen = (np.full(X.shape[1], float(ridge)) if penalty is None
           else np.asarray(penalty, dtype=float))
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    for _ in range(100):
        z = np.clip(off + X @ w, -35, 35)
        p = 1 / (1 + np.exp(-z))
        Wd = p * (1 - p) + 1e-9
        H = X.T @ (X * Wd[:, None]) + np.diag(pen)
        step = np.linalg.solve(H, X.T @ (y - p) - pen * w)
        w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def _logloss(off, y):
    p = np.clip(1 / (1 + np.exp(-np.clip(off, -35, 35))), 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def _walk_forward(rows_by_user, eval_users, g, rank, params, tid_index, lapse,
                  penalty=None):
    """One causal pass per user: fit ability and the per-user logistic on a
    trailing window, score the next block. Returns the pooled baseline offset,
    axis score and outcome, plus every per-user axis-coefficient vector the
    pass produced (the empirical-Bayes input)."""
    all_r0 = np.array([v[0] for v in params.values()])
    grid = np.arange(all_r0.min() - 3.0, all_r0.max() + 3.0, 0.05)
    offs, scores, ys, coefs = [], [], [], []
    for u in eval_users:
        a = rows_by_user[u]
        if len(a) < WARMUP + EVAL_BLOCK:
            continue
        tids = [t for t, _s in a]
        y = np.array([s for _t, s in a], dtype=float)
        rk = np.array([rank[t] for t in tids])
        G = np.stack([g[tid_index[t]] for t in tids])
        R0 = np.array([params[t][0] for t in tids])
        A = np.array([params[t][1] for t in tids])
        C = np.clip(np.array([params[t][2] for t in tids]), 0.0, FLOOR_MAX)
        span = np.maximum(1.0 - C - lapse, 1e-6)
        q = np.clip((CALIBRATION_P - C) / span, 1e-6, 1 - 1e-6)
        kap = np.log(q / (1 - q))
        X_b = np.column_stack([np.ones(len(a)), rk])
        X_ax = np.column_stack([np.ones(len(a)), rk, G])
        w_b = w_ax = None
        for t0 in range(WARMUP, len(a), EVAL_BLOCK):
            lo = max(0, t0 - WINDOW)
            hi = min(t0 + EVAL_BLOCK, len(a))
            try:
                th = _fit_theta(R0[lo:t0], A[lo:t0], C[lo:t0], span[lo:t0],
                                kap[lo:t0], y[lo:t0], grid)
                P4 = _p4pl(th, R0[lo:hi], A[lo:hi], C[lo:hi], span[lo:hi],
                           kap[lo:hi])
                off4 = np.log(P4 / (1 - P4))
                o_tr, o_ev = off4[:t0 - lo], off4[t0 - lo:]
                w_b = _irls_off(X_b[lo:t0], y[lo:t0], o_tr, w0=w_b)
                w_ax = _irls_off(X_ax[lo:t0], y[lo:t0], o_tr, w0=w_ax,
                                 penalty=penalty)
            except np.linalg.LinAlgError:
                continue
            offs.append(np.clip(o_ev + X_b[t0:hi] @ w_b, -35, 35))
            scores.append(X_ax[t0:hi, 2:] @ w_ax[2:])
            ys.append(y[t0:hi])
            coefs.append(w_ax[2:].copy())
    return (np.concatenate(offs), np.concatenate(scores), np.concatenate(ys),
            np.stack(coefs))


def _axis_prior(coefs):
    """Per-axis Gaussian prior precision from the spread of the per-user axis
    coefficients -- the same empirical-Bayes form the client ships as
    weakness_axis_lambda, estimated here because the candidate K under
    selection has no fitted vector yet. The spread is the MEDIAN ABSOLUTE
    DEVIATION (scaled to a normal standard deviation), not the standard
    deviation: an unpenalized pass is what supplies these coefficients, and a
    single runaway fit would otherwise set the scale for every axis. An axis
    with no between-user spread is one the prior should pin at zero, so the
    precision rises without bound and only a division guard bounds it."""
    mad = np.median(np.abs(coefs - np.median(coefs, axis=0)), axis=0)
    tau = np.maximum(1.4826 * mad, 1e-9)
    return np.concatenate([[0.0, 0.0], 1.0 / tau ** 2])   # intercept, rank free


def axes_ll_gain(rows_by_user, eval_users, g, rank, params, tid_index, lapse):
    """Held-out walk-forward LL gain (nats TOTAL, plus n) of the axes block
    over the production-form 4PL baseline, with the global calibration fit
    (intercept + kappa) applied to the pooled axis scores.

    Two passes: the first supplies the between-user coefficient spread, the
    second is the scored fit under that prior. The per-user coefficients are
    shrunk exactly as the client shrinks them -- unpenalized, a window whose
    design is near-singular admits an arbitrarily large coefficient, and the
    resulting scores swamp the pooled calibration."""
    _o, _s, _y, coefs = _walk_forward(rows_by_user, eval_users, g, rank,
                                      params, tid_index, lapse)
    off, sc, y, _c = _walk_forward(rows_by_user, eval_users, g, rank, params,
                                   tid_index, lapse,
                                   penalty=_axis_prior(coefs))
    # Standardize the pooled axis score before calibrating. A converged logistic
    # fit is invariant to an affine rescale of a regressor -- the coefficient
    # absorbs it and the intercept absorbs the centering -- so this leaves the
    # fitted probabilities and the returned gain alone. What it removes is the
    # dependence on the score's ARBITRARY scale: that scale is set by the
    # per-user IRLS above, which is unbounded on a separable window, so against
    # the intercept column the raw Hessian can be ill-conditioned enough that
    # rounding decides the solve.
    sc = (sc - sc.mean()) / (sc.std() + 1e-9)
    Xk = np.column_stack([np.ones(len(y)), sc])
    w = _irls_off(Xk, y, off, ridge=1e-9)
    return _logloss(off, y) - _logloss(off + Xk @ w, y), len(y)


# ------------------------------------------------------------------ selection
def select_and_fit(device="cuda"):
    """The full production procedure. Returns a dict:
      k_sel, gains {K: mnat/attempt}, loadings (n, k_sel) from the FINAL
      all-users fit, tids (aligned), W/rms/v_sd (per-cell fanout), feat_mu/
      feat_sd (the F standardization), act_dim, axis_mu/axis_sd (loading
      standardization for shipping)."""
    feats, _scalars, rank, params, act_dim = build_axis_features()
    tids = sorted(feats)
    tid_index = {t: i for i, t in enumerate(tids)}
    F = np.stack([feats[t] for t in tids])
    feat_mu, feat_sd = F.mean(axis=0), F.std(axis=0) + 1e-9
    F = ((F - feat_mu) / feat_sd).astype(np.float32)
    lapse = load_client_lapse(0.0328)

    rows = load_axis_attempts()
    rows_by_user = defaultdict(list)
    for u, p, s in rows:
        if p in tid_index:
            rows_by_user[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rows_by_user.items()
                   if len(a) >= MIN_USER_ATTEMPTS)
    fold_of = {u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 2
               for u in users}
    print(f"axes: {len(tids)} problems, {len(users)} users "
          f"({sum(len(rows_by_user[u]) for u in users)} attempts)")

    # Warm-start directions per fold, computed ONCE at the widest count the
    # search can reach and sliced per candidate: they are the eigenvectors of a
    # fold's residual score moment, so they do not depend on K, and recomputing
    # them per candidate would repeat a full baseline pass for nothing.
    folds = {f: [u for u in users if fold_of[u] != f] for f in (0, 1)}
    eigen_by_fold = {
        f: eigen_directions(rows_by_user, tr, F, tid_index, rank, params,
                            lapse, K_MAX)
        for f, tr in folds.items()}

    gains, peak, breached = {}, -np.inf, None
    k = K_START
    while k <= K_MAX:
        total_gain, total_n = 0.0, 0
        for fold in (0, 1):
            tr = folds[fold]
            ev = [u for u in users if fold_of[u] == fold]
            g, _W, _rms, _sd = train_axes(rows_by_user, tr, F, tid_index, k,
                                          eigen_by_fold[fold][:k], device)
            gain, n = axes_ll_gain(rows_by_user, ev, g, rank, params,
                                   tid_index, lapse)
            total_gain += gain
            total_n += n
        gains[k] = total_gain / total_n * 1000.0
        print(f"  K={k}: axes LL gain {gains[k]:.3f} mnat/attempt")
        peak = max(peak, gains[k])
        # The stop rule below is a FRACTION OF THE PEAK, which orders candidates
        # only while the peak is positive. Against a non-positive peak the
        # fraction sits ABOVE it, so the very first candidate breaches against
        # itself and selection falls through to K_START - 1 -- an axis count
        # below the floor the search starts at, written out as if it were a
        # decision. A peak that is not positive means the axes failed to beat
        # the baseline at all, which is never a shippable state, so refuse here
        # rather than let a degenerate count reach the artifacts. Only the first
        # iteration can trip this: peak is a running maximum.
        if not np.isfinite(peak) or peak <= 0.0:
            raise RuntimeError(
                f"axes selection: no positive gain by K={k} (peak {peak:.3f} "
                "mnat/attempt). The axes block did not beat the production-form "
                "4PL baseline on held-out users, so there is no axis count "
                "worth shipping. Refusing rather than selecting a count below "
                f"the K_START={K_START} floor.")
        if gains[k] < peak * (1.0 - STOP_FRACTION):
            breached = k
            break
        k += 1
    k_sel = (breached - 1) if breached is not None else K_MAX
    if k_sel < K_START:
        raise RuntimeError(
            f"axes selection: selected K={k_sel} below the K_START={K_START} "
            f"floor (breached at K={breached}, gains {gains}). Unreachable "
            "with a positive peak -- treat as a bug in the stop rule rather "
            "than a result.")
    print(f"  selected K = {k_sel} (peak {peak:.3f}"
          + (f", K={breached} breached at {gains[breached]:.3f})"
             if breached is not None else ", K_MAX backstop reached)"))

    # The final fit trains on ALL users, so its warm start is derived from all
    # of them too — the per-fold directions above exist only to keep the
    # selection honest, and reusing one here would start the shipped fit from
    # half the data.
    g_init = eigen_directions(rows_by_user, users, F, tid_index, rank, params,
                              lapse, k_sel)
    g_final, W, rms, v_sd = train_axes(rows_by_user, users, F, tid_index,
                                       k_sel, g_init, device)
    axis_mu = g_final.mean(axis=0)
    axis_sd = g_final.std(axis=0) + 1e-9
    return {
        "k_sel": k_sel, "gains": gains, "tids": tids,
        "loadings": ((g_final - axis_mu) / axis_sd).astype(np.float32),
        "W": W.astype(np.float32), "rms": rms.astype(np.float32),
        "v_sd": v_sd.astype(np.float32),
        "feat_mu": feat_mu.astype(np.float32),
        "feat_sd": feat_sd.astype(np.float32),
        "axis_mu": axis_mu.astype(np.float32),
        "axis_sd": axis_sd.astype(np.float32),
        "act_dim": act_dim,
    }
