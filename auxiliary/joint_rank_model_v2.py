"""Unified joint rank model (v2): all ranking systems feed the problems at once.

v1 (joint_rank_model.py) anchored users by converting every declaration to internal
rank through the FROZEN rank_map.json — i.e. it placed the other servers onto EGF in a
separate step and then fed that in. That made the server↔EGF coupling a pre-baked,
EGF-anchored extrapolation (broken at DDK, where no EGF declarations exist). v2 fixes
the structure: ONE simultaneous fit in which every server is a first-class observation
of a user's latent ability.

    P(user u solves problem p) = σ( cubic(θ_u) − cubic(d_p) + logit(0.8) )
        d_p = w · PCA(activations_p) + β · size_is_13          (problem difficulty)
        θ_u = latent ability (internal-rank units), one per user
    declarations (every server s, per-server ordinal cumulative-link coupling):
        P(declared_s ≤ g | θ_u) = σ( τ_{s,g} − θ_u )
        EGF: τ fixed at integer boundaries (identity) — the SOLE location pin (1d).
        others: τ = offset α_s + cubic-baseline spacing · exp(log-dev), shape-prior on log-dev.

All of {θ_u}, w, β, and every server's {α_s, log-dev} are estimated in one objective, so
an OGS (or Fox, KGS, …) declaration constrains the problems through θ_u exactly as an EGF
declaration does — and the 23 OGS DDK declarers + their outcomes calibrate the OGS↔internal
coupling where no EGF cross-declaration reaches. cubic = the fixed statistics.txt
rank→win-logit shape (governs BOTH the outcome link and the server-threshold spacing).

No per-problem free parameter (problems can't run to ±∞). No θ ridge: declarers are
pinned by their declarations; non-declarers are kept only if they have ≥1 win AND ≥1 loss
(finite MLE), all-win/all-loss ones dropped rather than leashed.

The squash fix (`--theta-prior`): with only ~6% of users (declarers) anchored and
adaptive selection pinning everyone's solve-rate near SOLVE_OFFSET (~80%, declarer sd
≈0.04), the outcomes carry almost no ability signal — so the free non-declarer θ's
collapse toward the centre and drag the global difficulty map w's spread in with them
(the rank scale is under-identified, NOT a convergence or ridge artefact). The fix is an
independent per-user ability prior: each user's θ is soft-anchored to an MLE ability fit
on the production mixture params (compare_rank_surveys), affine-calibrated to internal
rank on the declarer overlap (see load_user_theta_priors). This turns all ~1.5k users
into scale-bearing anchors. Effect: placed-difficulty range +~60%, Val-3 ρ 0.65→0.71,
Val-2 mid-band bias →~0 (outcome log-loss rises modestly — the free-θ baseline only
"won" it by overfitting the collapsed scale). Residual squash beyond this is regression
attenuation in w (activations explain ~0.6 of difficulty variance) — a separate ceiling.

Usage:
  ./.venv/bin/python3 -m auxiliary.joint_rank_model_v2
  ./.venv/bin/python3 -m auxiliary.joint_rank_model_v2 --theta-prior          # de-squashed
  ./.venv/bin/python3 -m auxiliary.joint_rank_model_v2 --lam-decl 30 --pca-dim 256
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.compare_rank_surveys import (  # noqa: E402
    SQL_PATH_DEFAULT, fit_theta, latest_per_user_server,
    load_attempts_for_users, load_problem_params, load_rank_surveys,
    theta_to_kyudan,
)
from auxiliary.fit_statistics_shapes import fit_shape  # noqa: E402
from auxiliary.joint_rank_model import (  # noqa: E402
    RANK_INDEX_OFFSET, SOLVE_OFFSET, load_joint_features,
)
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402


# ---- cubic shape helpers (shared by the outcome link and server spacing) ----

_CUBIC_PARAMS, _ = fit_shape("cubic")  # [a, b, d] for θ(x)=a·x+b·x²+d·x³, x=rank_index


def cubic_np(rank_scalar):
    a, b, d = _CUBIC_PARAMS
    x = np.asarray(rank_scalar, float) + RANK_INDEX_OFFSET
    return a * x + b * x * x + d * x ** 3


def cubic_baseline_spacing(g_lo, g_hi):
    """Latent-unit spacing the cubic implies between consecutive server ranks over
    [g_lo, g_hi] — the shape prior baseline for that server's thresholds. Boundaries
    sit at g+0.5; spacing is constant 1 rank here (server ranks ARE rank units), but
    we express it through the cubic so 'apply the cubic to all servers' is literal:
    the prior spacing is the cubic's local rank step, normalized to ~1 at 1k."""
    bounds = np.arange(g_lo, g_hi) + 0.5
    # local cubic slope at each boundary, normalized so 1k(≈0) → 1.0
    a, b, d = _CUBIC_PARAMS
    x = bounds + RANK_INDEX_OFFSET
    slope = a + 2 * b * x + 3 * d * x * x
    slope0 = a + 2 * b * RANK_INDEX_OFFSET + 3 * d * RANK_INDEX_OFFSET ** 2
    return np.maximum(slope / slope0, 0.05)


def load_user_theta_priors(user_ids, declarer_ids, sizes, sql_path, surveys,
                           min_attempts=8, include_declarers=False):
    """Per-user ability prior in INTERNAL-rank units, for soft-anchoring θ.

    The squash this fixes: the model leaves ~96% of users' θ free and
    unanchored, and because adaptive selection pins everyone's solve-rate near
    the SOLVE_OFFSET (~80%), those outcomes carry almost no ability signal — so
    the free θ's collapse toward the centre and drag the global difficulty map
    w's spread in with them. Each user's ability, however, is independently
    estimable from their OWN first-time attempts via the production 3-expert
    mixture params (exactly as compare_rank_surveys does): fit_theta → the
    handicap-calibrated rank scalar (theta_to_kyudan), max across boards.

    That scalar lives on the handicap-stone axis (RANK_PRIOR: 3k=0), NOT the
    internal/EGF axis the model's θ uses (1k=0), so it is calibrated with ONE
    affine fit on the OVERLAP set — declarers, who carry both an MLE θ and a
    declaration. The declaration's internal rank (via the frozen rank_map.json,
    EGF=identity) is the affine target. The affine is FIXED, not learned
    jointly: a learned slope could shrink to 0 and let θ collapse anyway, so
    fixing it is what makes the prior carry a real, non-collapsible spread.

    Returns ({user_id: prior_internal_rank}, (slope, intercept), n_with_mle).
    By default the prior is emitted for NON-declarers only (declarers stay
    pinned by their declarations alone); include_declarers=True emits it for
    everyone (ablation)."""
    import json as _json
    want = set(user_ids) | set(declarer_ids)
    pp = {s: load_problem_params(s) for s in sizes}
    att = load_attempts_for_users(sql_path, want)
    hcap = {}  # user_id -> handicap-rank scalar (max over boards)
    for u in want:
        best = None
        for s in sizes:
            params, ta, sa, _ = pp.get(s, (None, None, None, None))
            if params is None or ta is None:
                continue
            a = att.get((u[:8], s), [])
            if len(a) < min_attempts:
                continue
            th, _n = fit_theta(a, params)
            if th is None:
                continue
            ks = theta_to_kyudan(th, ta, sa)
            if ks is not None:
                best = ks if best is None else max(best, ks)
        if best is not None:
            hcap[u] = best
    # affine hcap → internal, fit on declarers via frozen rank_map (EGF=id)
    rm_path = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rank_map.json")
    rm = _json.load(open(rm_path)) if os.path.exists(rm_path) else {}

    def declared_internal(server, g):
        if server == "EGF":
            return float(g)
        if server in rm and len(rm[server]["our"]) >= 2:
            return float(np.interp(g, rm[server]["server"], rm[server]["our"]))
        return None

    xs, ys = [], []
    for u, sdf in surveys.groupby("user_id"):
        if u not in hcap:
            continue
        vals = [declared_internal(r.server, r.rank_scalar) for r in sdf.itertuples()]
        vals = [v for v in vals if v is not None]
        if vals:
            xs.append(hcap[u])
            ys.append(float(np.mean(vals)))
    if len(xs) < 5:
        raise RuntimeError("too few declarers with MLE θ to calibrate the prior affine")
    A = np.vstack([np.array(xs), np.ones(len(xs))]).T
    (slope, intc), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    priors = {}
    for u, h in hcap.items():
        if not include_declarers and u in declarer_ids:
            continue
        priors[u] = float(slope * h + intc)
    return priors, (float(slope), float(intc)), len(hcap)


class UnifiedModel(nn.Module):
    def __init__(self, n_users, k, servers, use_slope=False, slope_hidden=()):
        """servers: {name: (g_min, g_max, baseline_spacing[np], is_egf)}.
        Difficulty d_p = w·activations is LINEAR (confirmed). The slope head g may be
        linear (slope_hidden=()) or an MLP (slope_hidden=[h1,..]) — discrimination is
        plausibly nonlinear even though rank is linear, so we test both."""
        super().__init__()
        self.use_slope = use_slope
        self.theta = nn.Embedding(n_users, 1)
        nn.init.zeros_(self.theta.weight)
        self.w = nn.Linear(k, 1, bias=False)
        nn.init.zeros_(self.w.weight)
        if use_slope:
            # Per-problem discrimination a_p = softplus(bias + g(activations)): a SHARED
            # activation head (no per-problem free param). bias FIXED so a_p≈1 baseline,
            # output layer zero-init, clamp + slope_ridge keep the scale from degenerating
            # (a_p→0 would free d_p and explode it).
            if slope_hidden:
                layers, d_in = [], k
                for h in slope_hidden:
                    layers += [nn.Linear(d_in, h), nn.ReLU()]
                    d_in = h
                head = nn.Linear(d_in, 1, bias=False)
                nn.init.zeros_(head.weight)
                self.g = nn.Sequential(*layers, head)
            else:
                self.g = nn.Linear(k, 1, bias=False)
                nn.init.zeros_(self.g.weight)
            self.register_buffer("slope_bias", torch.tensor(0.5413))
        self.board = nn.Parameter(torch.zeros(1))
        a, b, d = [float(p) for p in _CUBIC_PARAMS]
        for nm, v in (("ca", a), ("cb", b), ("cd", d)):
            self.register_buffer(nm, torch.tensor(v))

        self.servers = {}
        self.coupling = nn.ParameterDict()
        for name, (g_min, g_max, baseline, is_egf) in servers.items():
            K = g_max - g_min  # number of thresholds (categories 0..K)
            self.servers[name] = (g_min, g_max, K, is_egf)
            if is_egf:
                # Fixed identity boundaries (internal rank ≡ EGF rank) → location pin.
                tau = torch.tensor([(g_min + j) + 0.5 for j in range(K)], dtype=torch.float32)
                self.register_buffer(f"tau_{name}", tau)
                self.register_buffer(f"base_{name}", torch.zeros(max(K - 1, 0)))
            else:
                # [α_s, log-dev_1..K-1]; τ = α + cumsum(baseline·exp(log-dev)).
                p = torch.zeros(K)
                p[0] = float(g_min) + 0.5  # init offset at identity
                self.coupling[name] = nn.Parameter(p)
                self.register_buffer(f"base_{name}",
                                     torch.tensor(baseline[:K - 1], dtype=torch.float32))

    def _tau(self, name):
        g_min, g_max, K, is_egf = self.servers[name]
        if is_egf:
            return getattr(self, f"tau_{name}")
        p = self.coupling[name]
        base = getattr(self, f"base_{name}")
        if K <= 1:
            return p[:1]
        spac = base * torch.exp(p[1:])
        return torch.cat([p[:1], p[0] + torch.cumsum(spac, 0)])

    def difficulty(self, feat, size13):
        return self.w(feat).squeeze(-1) + self.board * size13

    def slope(self, feat):
        if not self.use_slope:
            return 1.0
        a = torch.nn.functional.softplus(self.slope_bias + self.g(feat).squeeze(-1))
        return a.clamp(0.2, 5.0)

    def _cubic(self, rank):
        x = (rank + RANK_INDEX_OFFSET).clamp(-10.0, 40.0)
        return self.ca * x + self.cb * x * x + self.cd * x ** 3

    def solve_logit(self, u, feat, size13):
        gap = (self._cubic(self.theta(u).squeeze(-1))
               - self._cubic(self.difficulty(feat, size13)))
        return self.slope(feat) * gap + SOLVE_OFFSET

    def weight_l2(self):
        return self.w.weight.pow(2).sum()  # slope head ridged separately (slope_ridge)

    def slope_l2(self):
        return sum(p.pow(2).sum() for p in self.g.parameters())

    def decl_nll(self, name, u_idx, cat):
        """Per-declaration ordinal cumulative-link NLL (one term per row). Caller
        sums (optionally with per-user weights)."""
        _, _, K, _ = self.servers[name]
        tau = self._tau(name)
        th = self.theta(u_idx).squeeze(-1)
        cu = torch.clamp(cat, max=K - 1)
        upper = torch.where(cat < K, torch.sigmoid(tau[cu] - th), torch.ones_like(th))
        cl = torch.clamp(cat - 1, min=0)
        lower = torch.where(cat > 0, torch.sigmoid(tau[cl] - th), torch.zeros_like(th))
        return -torch.log((upper - lower).clamp(min=1e-9))  # per-row, NOT summed

    def logdev_penalty(self):
        pen = self.theta.weight.new_zeros(())
        for name in self.coupling:
            pen = pen + (self.coupling[name][1:] ** 2).sum()
        return pen


def build_servers(surveys, ucat, min_decl):
    """Return ({name: (g_min,g_max,baseline,is_egf)}, {name: (uidx_arr, cat_arr)})."""
    servers, decls = {}, {}
    for name, sdf in surveys.groupby("server"):
        sdf = sdf[sdf["user_id"].isin(ucat)]
        if len(sdf) < min_decl:
            continue
        g = sdf["rank_scalar"].round().astype(int)
        g_min, g_max = int(g.min()), int(g.max())
        if g_max == g_min:
            g_max += 1  # need ≥1 threshold
        is_egf = (name == "EGF")
        baseline = cubic_baseline_spacing(g_min, g_max)
        servers[name] = (g_min, g_max, baseline, is_egf)
        uidx = np.array([ucat[u] for u in sdf["user_id"]], np.int64)
        cat = np.clip(g.to_numpy() - g_min, 0, g_max - g_min).astype(np.int64)
        decls[name] = (uidx, cat)
    return servers, decls


def train(model, tX, tu, ts, ty, decl_t, device, epochs, lr, lam_decl, ridge,
          shape_lam, slope_ridge=1.0, user_weights=None,
          prior_mean=None, prior_mask=None, theta_prior_w=0.0):
    """user_weights: 1d tensor of length n_users, applied per-attempt as
    weight=user_weights[u_attempt]. Same weighting is applied to the per-declaration
    NLL. Normalized externally so mean weight ≈ 1 (keeps loss scale comparable to
    the unweighted run, so lam_decl/ridge/shape_lam don't need re-tuning).

    prior_mean/prior_mask: per-user (len n_users) Gaussian θ-prior. Adds
    theta_prior_w·Σ_u mask_u·(θ_u − prior_mean_u)² — the independent ability
    anchor (see load_user_theta_priors) that stops the unanchored majority from
    collapsing the rank scale."""
    model.to(device)
    tX, tu, ts, ty = tX.to(device), tu.to(device), ts.to(device), ty.to(device)
    decl_t = {n: (u.to(device), c.to(device)) for n, (u, c) in decl_t.items()}
    use_prior = prior_mean is not None and theta_prior_w > 0
    if use_prior:
        prior_mean = prior_mean.to(device)
        prior_mask = prior_mask.to(device)
    use_w = user_weights is not None
    if use_w:
        w_t = user_weights.to(device)
        bce_none = nn.BCEWithLogitsLoss(reduction="none")
    else:
        bce_sum = nn.BCEWithLogitsLoss(reduction="sum")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model.solve_logit(tu, tX, ts)
        if use_w:
            loss = (bce_none(logits, ty) * w_t[tu]).sum()
        else:
            loss = bce_sum(logits, ty)
        for name, (u, c) in decl_t.items():
            nll_per = model.decl_nll(name, u, c)
            if use_w:
                loss = loss + lam_decl * (nll_per * w_t[u]).sum()
            else:
                loss = loss + lam_decl * nll_per.sum()
        loss = loss + ridge * model.weight_l2() + shape_lam * model.logdev_penalty()
        if use_prior:
            th_all = model.theta.weight.squeeze(-1)
            loss = loss + theta_prior_w * (((th_all - prior_mean) ** 2) * prior_mask).sum()
        if model.use_slope:
            loss = loss + slope_ridge * model.slope_l2()
        loss.backward()
        opt.step()
    return model


def compute_user_weights(theta_np, mode="inverse-density", bw=None,
                         clip=(0.1, 30.0)):
    """Per-user loss weights to compensate user-rank-density imbalance: w_u ∝
    1/KDE(θ_u). Sparse-θ users (DDK declarers, strong dans) get amplified so
    their non-saturated outcomes are not drowned out by the SDK majority's
    saturated ones. Normalized so mean=1 (post-clip), then clipped again to
    keep the bounds honest."""
    if mode == "uniform":
        return np.ones_like(theta_np, dtype=np.float32)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(theta_np, bw_method=bw or "scott")
    dens = np.maximum(kde(theta_np), 1e-6)
    w = 1.0 / dens
    w = w / w.mean()
    w = np.clip(w, *clip)
    w = w / w.mean()
    return w.astype(np.float32)


def stage2_slope(stage1_model, n_users, k, servers, slope_hidden, tX, tu, ts, ty,
                 device, epochs, lr, slope_ridge):
    """Two-stage 2PL: copy theta+w+board+coupling from a converged 1PL fit into a
    slope-enabled twin, freeze everything except the slope head g, then optimize
    BCE on outcomes (decl_nll is constant under frozen theta — dropped). Breaks
    the joint a_p↔d_p scale degeneracy by construction: with d_p pinned, a_p
    cannot trade scale with difficulty, only rescale the existing rank gap to
    match per-problem outcome sharpness."""
    m = UnifiedModel(n_users, k, servers, use_slope=True, slope_hidden=slope_hidden)
    m.theta.weight.data.copy_(stage1_model.theta.weight.data)
    m.w.weight.data.copy_(stage1_model.w.weight.data)
    m.board.data.copy_(stage1_model.board.data)
    for name in stage1_model.coupling:
        m.coupling[name].data.copy_(stage1_model.coupling[name].data)
    for nm, p in m.named_parameters():
        if not nm.startswith("g."):
            p.requires_grad_(False)
    m.to(device)
    tX, tu, ts, ty = tX.to(device), tu.to(device), ts.to(device), ty.to(device)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        loss = bce(m.solve_logit(tu, tX, ts), ty) + slope_ridge * m.slope_l2()
        loss.backward()
        opt.step()
    return m


def report_slope(model, feat, tids_by_size, sizes, device, label=""):
    """a_p stats + Spearman vs shipped bot quality on each board."""
    from scipy.stats import spearmanr as _sp
    with torch.no_grad():
        af0 = torch.tensor(np.stack(list(feat.values())).astype(np.float32)).to(device)
        a_p = model.slope(af0).cpu().numpy()
    ap_by_tid = dict(zip(feat.keys(), a_p))
    prefix = f"{label} " if label else ""
    print(f"{prefix}per-problem slope a_p: min {a_p.min():.2f}, median {np.median(a_p):.2f}, "
          f"max {a_p.max():.2f}  (validate vs shipped bot 'quality':)")
    for s in sizes:
        qp = os.path.join(PROJECT_ROOT, "ratings", str(s), "tsumego_quality_final.csv")
        if not os.path.exists(qp):
            continue
        q = pd.read_csv(qp).set_index("tsumego_id")["quality"]
        ct = [t for t in tids_by_size[s] if t in ap_by_tid and t in q.index]
        if len(ct) < 10:
            continue
        av = np.array([ap_by_tid[t] for t in ct])
        qv = q.loc[ct].to_numpy()
        print(f"    size {s}: Spearman(a_p, bot quality) = {_sp(av, qv).statistic:+.3f} "
              f"over {len(ct)} problems")
    return ap_by_tid


def implied_internal(model, name, g_scalar):
    """Internal rank a declaration of (server, g_scalar) maps to under the fitted
    coupling — midpoint of the category's θ-interval. EGF → identity."""
    g_min, g_max, K, _ = model.servers[name]
    tau = model._tau(name).detach().cpu().numpy()
    cat = int(np.clip(round(g_scalar) - g_min, 0, K))
    lo = tau[cat - 1] if cat > 0 else tau[0] - 1.0
    hi = tau[cat] if cat < K else tau[-1] + 1.0
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="9,13")
    ap.add_argument("--min-attempts", type=int, default=10)
    ap.add_argument("--min-decl", type=int, default=3, help="Skip a server with fewer declarations.")
    ap.add_argument("--pca-dim", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam-decl", type=float, default=30.0,
                    help="Weight on the declaration likelihood vs the (summed) outcome BCE.")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--slope", action="store_true",
                    help="Add a per-problem discrimination head a_p=softplus(g·activations) (2PL).")
    ap.add_argument("--two-stage-slope", action="store_true",
                    help="Fit 1PL first, then freeze theta+w+board+coupling and learn ONLY "
                         "the slope head g on user outcomes. Eliminates the joint-2PL "
                         "a_p↔d_p scale degeneracy by construction; tests whether user "
                         "outcomes carry per-problem discrimination given a fixed rank scale.")
    ap.add_argument("--slope-ridge", type=float, default=1.0,
                    help="Ridge on the slope head g (keeps a_p≈1; low values let the "
                         "scale degenerate and difficulty explode).")
    ap.add_argument("--slope-hidden", type=int, nargs="*", default=[],
                    help="Slope-head MLP hidden sizes (empty = linear). Rank is linear "
                         "in activations but discrimination may not be — test e.g. 64.")
    ap.add_argument("--shape-lam", type=float, default=10.0, help="Shape prior on per-server spacing.")
    ap.add_argument("--theta-prior", action="store_true",
                    help="Soft-anchor every user's θ with an independent ability "
                         "estimate (MLE on the production mixture params, affine-"
                         "calibrated to internal rank on the declarer overlap). Stops "
                         "the unanchored majority from collapsing the rank scale — the "
                         "principled de-squash. See load_user_theta_priors.")
    ap.add_argument("--theta-prior-w", type=float, default=1.0,
                    help="Weight of the per-user θ-prior (Σ_u (θ_u−prior_u)²). ρ and "
                         "Val-2 recovery saturate by ~1.0; higher only widens the "
                         "range further at a rising outcome-fit cost.")
    ap.add_argument("--theta-prior-min-attempts", type=int, default=8,
                    help="Min joined attempts to fit a user's MLE ability prior.")
    ap.add_argument("--theta-prior-all", action="store_true",
                    help="Also prior-anchor declarers (default: declarers stay pinned "
                         "by their declarations alone). Ablation.")
    ap.add_argument("--user-weight", choices=("uniform", "inverse-density"),
                    default="uniform",
                    help="inverse-density: two-pass — uniform-weight pre-fit to "
                         "estimate θ, then refit with w_u ∝ 1/KDE(θ_u). Amplifies "
                         "tail-rank users whose non-saturated outcomes carry the "
                         "DDK/dan-end placement signal that the SDK majority drowns out.")
    ap.add_argument("--weight-bw", type=float, default=None,
                    help="KDE bandwidth for the density (Scott's rule if unset).")
    ap.add_argument("--weight-clip", type=float, nargs=2, default=(0.1, 30.0),
                    metavar=("LO", "HI"),
                    help="Clip per-user weights to [LO, HI] after 1/density normalization.")
    ap.add_argument("--all-embedded", action="store_true",
                    help="Load activations for ALL embedded problems (not just those with "
                         "attempts), so the trained w can be projected over the full corpus at "
                         "output time. PCA basis is then fit on the larger set; training "
                         "(BCE+decl) still only uses problems with attempts.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "auxiliary",
                                                  "joint_rank_placements_v2.csv"))
    ap.add_argument("--out-rankmap", default=None,
                    help="If set, dump the fitted per-server couplings as a rank_map.json "
                         "(format the PWA expects: {server: {our:[...], server:[...]}}).")
    args = ap.parse_args()
    if args.slope and args.two_stage_slope:
        ap.error("--slope (joint 2PL) and --two-stage-slope are mutually exclusive")
    sizes = [int(s) for s in args.sizes.split(",")]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_first_time_attempts()
    df = df[df["size"].isin(sizes)].copy()
    surveys = latest_per_user_server(load_rank_surveys(SQL_PATH_DEFAULT))
    declarer_ids = set(surveys["user_id"])

    # User selection: declarers always (anchored by declaration, can't blow up);
    # non-declarers only with enough attempts AND both outcomes (finite MLE, no ridge).
    agg = df.groupby("user_id")["success"].agg(n="count", win="sum")
    agg["loss"] = agg["n"] - agg["win"]
    nondecl_ok = set(agg.index[(agg["n"] >= args.min_attempts) &
                               (agg["win"] > 0) & (agg["loss"] > 0)]) - declarer_ids
    keep_users = (declarer_ids | nondecl_ok)
    df = df[df["user_id"].isin(keep_users)].copy()
    n_dropped_allwl = len(agg.index[(agg["n"] >= args.min_attempts) &
                                    ((agg["win"] == 0) | (agg["loss"] == 0))] )

    tids_by_size = {s: df[df["size"] == s]["problem"].unique() for s in sizes}
    if args.all_embedded:
        # Union with every cached activation tid for each size — so feat[] (and thus
        # the final projected internal_rank) covers the full corpus, not just problems
        # users have attempted. Training set (BCE) is still the attempted subset.
        import json as _json
        from auxiliary.inspect_activation_signals import _cache_ids_path
        for s in sizes:
            all_ids = _json.load(open(_cache_ids_path(s)))
            tids_by_size[s] = np.array(sorted(set(tids_by_size[s]) | set(all_ids)))
    feat, k = load_joint_features(sizes, tids_by_size, args.pca_dim, args.seed)
    df = df[df["problem"].isin(feat.keys())].reset_index(drop=True)

    # users = union of attempters kept + all declarers (some declarers may have no attempts)
    user_ids = sorted(set(df["user_id"]) | declarer_ids)
    ucat = {u: i for i, u in enumerate(user_ids)}
    n_users = len(user_ids)

    servers, decls = build_servers(surveys, ucat, args.min_decl)
    print(f"attempts {len(df)} | users {n_users} (declarers {len(declarer_ids)}, "
          f"all-win/loss non-declarers dropped {n_dropped_allwl}) | "
          f"problems {df['problem'].nunique()} | PCs {k} | device {device.type}")
    print(f"servers fit jointly: {', '.join(f'{n}({len(decls[n][0])})' for n in servers)}\n")

    uidx = np.array([ucat[u] for u in df["user_id"]], np.int64)
    X = np.stack(df["problem"].map(feat).to_numpy()).astype(np.float32)
    s13 = (df["size"] == 13).to_numpy(np.float32)
    y = df["success"].to_numpy(np.float32)
    groups = pd.Categorical(df["problem"]).codes
    tX, tu, ts, ty = (torch.tensor(X), torch.tensor(uidx),
                      torch.tensor(s13), torch.tensor(y))
    decl_t = {n: (torch.tensor(u), torch.tensor(c)) for n, (u, c) in decls.items()}
    common = dict(epochs=args.epochs, lr=args.lr, lam_decl=args.lam_decl,
                  ridge=args.ridge, shape_lam=args.shape_lam, slope_ridge=args.slope_ridge)

    # ---- Independent per-user ability prior (optional, the de-squash anchor) ----
    if args.theta_prior:
        priors, (a_, b_), n_mle = load_user_theta_priors(
            keep_users, declarer_ids, sizes, SQL_PATH_DEFAULT, surveys,
            min_attempts=args.theta_prior_min_attempts,
            include_declarers=args.theta_prior_all)
        pm = np.zeros(n_users, np.float32)
        mask = np.zeros(n_users, np.float32)
        for u, v in priors.items():
            if u in ucat:
                pm[ucat[u]] = v
                mask[ucat[u]] = 1.0
        common.update(prior_mean=torch.tensor(pm), prior_mask=torch.tensor(mask),
                      theta_prior_w=args.theta_prior_w)
        print(f"θ-prior: MLE ability fit for {n_mle} users | calibration "
              f"internal ≈ {a_:.3f}·hcap {b_:+.3f} | prior applied to "
              f"{int(mask.sum())} users (w={args.theta_prior_w}, "
              f"{'incl' if args.theta_prior_all else 'excl'} declarers)\n")

    # ---- Two-pass weighting (optional): pre-fit (uniform) → KDE on θ → weights ----
    user_weights = None
    if args.user_weight != "uniform":
        print(f"Pre-fit (uniform weights) for θ estimation → KDE → {args.user_weight} reweighting")
        pre = UnifiedModel(n_users, k, servers, args.slope, args.slope_hidden)
        train(pre, tX, tu, ts, ty, decl_t, device, **common)
        pre.eval()
        with torch.no_grad():
            theta_pre = pre.theta.weight.squeeze(-1).cpu().numpy()
        u_w_np = compute_user_weights(theta_pre, mode=args.user_weight,
                                      bw=args.weight_bw, clip=tuple(args.weight_clip))
        user_weights = torch.tensor(u_w_np)
        # Diagnostic — what got upweighted?
        # bins by pre-fit θ to show that DDK/dan-end users got the largest weights
        bands = [(-30, -10, "≤10k"), (-10, -5, "10k-5k"), (-5, -1, "5k-1k"),
                 (-1,  1, "1k-1d"),   (1,  4, "2-3d"),   (4,  30, "4d+")]
        print(f"  θ_pre range [{theta_pre.min():.2f}, {theta_pre.max():.2f}]; "
              f"weight stats: min {u_w_np.min():.3f}, median {np.median(u_w_np):.3f}, "
              f"max {u_w_np.max():.3f}")
        for lo, hi, lab in bands:
            mask = (theta_pre >= lo) & (theta_pre < hi)
            if mask.any():
                w_mean = float(u_w_np[mask].mean())
                print(f"    {lab:>7s}: n_users={mask.sum():>4d}  mean_weight={w_mean:>6.2f}")
        common["user_weights"] = user_weights
        print()

    # ---- Validation 1: outcome generalization (GroupKFold by problem) ----
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
    oof = np.empty(len(y))
    oof_2s = np.empty(len(y)) if args.two_stage_slope else None
    for tr, te in GroupKFold(args.folds).split(y, groups=groups):
        m = UnifiedModel(n_users, k, servers, args.slope, args.slope_hidden)
        train(m, tX[tr], tu[tr], ts[tr], ty[tr], decl_t, device, **common)
        m.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(m.solve_logit(
                tu[te].to(device), tX[te].to(device), ts[te].to(device))).cpu().numpy()
        if args.two_stage_slope:
            m2 = stage2_slope(m, n_users, k, servers, args.slope_hidden,
                              tX[tr], tu[tr], ts[tr], ty[tr], device,
                              args.epochs, args.lr, args.slope_ridge)
            m2.eval()
            with torch.no_grad():
                oof_2s[te] = torch.sigmoid(m2.solve_logit(
                    tu[te].to(device), tX[te].to(device), ts[te].to(device))).cpu().numpy()
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    print(f"Validation 1 — outcome generalization (GroupKFold by problem):")
    print(f"  1PL:       LogLoss {log_loss(y, oof):.4f} | AUC {roc_auc_score(y, oof):.4f}")
    if args.two_stage_slope:
        oof_2s = np.clip(oof_2s, 1e-6, 1 - 1e-6)
        print(f"  +2-stage:  LogLoss {log_loss(y, oof_2s):.4f} | "
              f"AUC {roc_auc_score(y, oof_2s):.4f}")
    print()

    # ---- Full fit ----
    model = UnifiedModel(n_users, k, servers, args.slope, args.slope_hidden)
    train(model, tX, tu, ts, ty, decl_t, device, **common)
    model.eval()
    print(f"Full fit: board FE (13x13) {float(model.board.item()):+.3f} rank")
    print("Fitted per-server coupling (internal rank implied for sample declared ranks):")
    from auxiliary.compare_rank_surveys import scalar_to_rank
    for name in servers:
        g_min, g_max, K, is_egf = model.servers[name]
        pts = sorted({g_min, (g_min + g_max) // 2, g_max})
        cells = ", ".join(f"{scalar_to_rank(g)}→{implied_internal(model, name, g):+.1f}" for g in pts)
        print(f"  {name:>8s}{' (egf,id)' if is_egf else '        '}: {cells}")

    if args.slope:
        report_slope(model, feat, tids_by_size, sizes, device, label="Joint-2PL")

    if args.two_stage_slope:
        print("\n--- Stage 2: freeze theta+w+board+coupling, train only slope head g ---")
        model_2s = stage2_slope(model, n_users, k, servers, args.slope_hidden,
                                tX, tu, ts, ty, device,
                                args.epochs, args.lr, args.slope_ridge)
        model_2s.eval()
        report_slope(model_2s, feat, tids_by_size, sizes, device, label="Two-stage")

    # ---- Validation 2: leave-one-declarer-out absolute recovery, stratified ----
    # Hold out ALL of a declarer's declarations; read θ from outcomes; compare to the
    # internal rank their declaration implies under the full-fit coupling (true GT for
    # EGF, self-consistency for others). Only declarers with both outcomes & enough
    # attempts (finite outcome-θ).
    with torch.no_grad():
        gt = {u: implied_internal(model, sdf["server"].iloc[0], sdf["rank_scalar"].iloc[0])
              for u, sdf in surveys.groupby("user_id")}
    elig = [u for u in declarer_ids if u in ucat and u in set(agg.index[
        (agg["n"] >= args.min_attempts) & (agg["win"] > 0) & (agg["loss"] > 0)])]
    # average GT over a user's declarations (via fitted couplings)
    gt_user = {}
    for u, sdf in surveys.groupby("user_id"):
        if u not in elig:
            continue
        vals = [implied_internal(model, r.server, r.rank_scalar)
                for r in sdf.itertuples() if r.server in model.servers]
        if vals:
            gt_user[u] = float(np.mean(vals))
    elig = [u for u in elig if u in gt_user]
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import pearsonr, spearmanr
    declared = np.array([gt_user[u] for u in elig])
    eu = np.array([ucat[u] for u in elig])
    strata = pd.qcut(declared, q=min(5, len(np.unique(declared))), labels=False, duplicates="drop")
    held = set(u for sdf in [surveys] for u in elig)  # users to drop from decl loss per fold
    oof_theta = np.full(len(elig), np.nan)
    for tr_a, te_a in StratifiedKFold(min(5, len(elig)), shuffle=True,
                                      random_state=args.seed).split(np.arange(len(elig)), strata):
        drop = set(eu[te_a].tolist())
        decl_fold = {}
        for n, (u, c) in decls.items():
            mask = ~np.isin(u, list(drop))
            if mask.any():
                decl_fold[n] = (torch.tensor(u[mask]), torch.tensor(c[mask]))
        m = UnifiedModel(n_users, k, servers, args.slope, args.slope_hidden)
        train(m, tX, tu, ts, ty, decl_fold, device, **common)
        with torch.no_grad():
            th = m.theta.weight.squeeze(-1).cpu().numpy()
        oof_theta[te_a] = th[eu[te_a]]
    rmse = float(np.sqrt(np.mean((oof_theta - declared) ** 2)))
    print(f"\nValidation 2 — leave-declarer-out recovery ({len(elig)} declarers; GT = "
          f"declaration via fitted coupling, identity for EGF):")
    print(f"  overall: Spearman {spearmanr(oof_theta, declared).statistic:.3f} | "
          f"Pearson {pearsonr(oof_theta, declared)[0]:.3f} | RMSE {rmse:.2f} ranks")
    bands = pd.qcut(declared, q=min(5, len(np.unique(declared))), duplicates="drop")
    bdf = pd.DataFrame({"declared": declared, "recovered": oof_theta, "band": bands})
    print(f"  {'GT band':>16s} {'n':>3s} {'mean_GT':>8s} {'mean_rec':>9s} {'bias':>7s} {'RMSE':>6s}")
    for b, g in bdf.groupby("band", observed=True):
        e = g.recovered - g.declared
        print(f"  {f'{b.left:+.1f}..{b.right:+.1f}':>16s} {len(g):>3d} {g.declared.mean():>+8.2f} "
              f"{g.recovered.mean():>+9.2f} {e.mean():>+7.2f} {np.sqrt((e**2).mean()):>6.2f}")

    # ---- Validation 3 + placement output ----
    size_of = {}
    for s in sizes:
        for t in tids_by_size[s]:
            size_of.setdefault(t, s)
    # Bake the per-board FE into the shipped internal_rank — earlier versions passed
    # zeros for size13 here, which dropped the 13×13 offset from the CSV (benign for
    # the Validation-3 Spearman, but wrong for any consumer that ships these as ranks).
    s13_mask = np.array([1.0 if size_of.get(t) == 13 else 0.0 for t in feat],
                        dtype=np.float32)
    with torch.no_grad():
        af = torch.tensor(np.stack(list(feat.values())).astype(np.float32)).to(device)
        s13_t = torch.tensor(s13_mask).to(device)
        d = model.difficulty(af, s13_t).cpu().numpy()
    placed = pd.DataFrame({"tsumego_id": list(feat.keys()),
                           "size": [size_of.get(t, -1) for t in feat],
                           "internal_rank": d})
    print("\nValidation 3 — placed internal_rank vs current bot rating (Spearman):")
    for s in sizes:
        rp = os.path.join(PROJECT_ROOT, "ratings", str(s), "tsumego_ratings_final.csv")
        if not os.path.exists(rp):
            continue
        rat = pd.read_csv(rp).set_index("tsumego_id")["rating"]
        sub = placed[placed["size"] == s].set_index("tsumego_id").join(rat, how="inner")
        print(f"  size {s}: ρ={spearmanr(sub.internal_rank, sub.rating).statistic:.3f} "
              f"({len(sub)} problems) | range [{sub.internal_rank.min():.1f}, "
              f"{sub.internal_rank.max():.1f}]")
    placed.sort_values("internal_rank").to_csv(args.out, index=False)
    print(f"\nWrote {len(placed)} placed ranks → {args.out}")

    if args.out_rankmap:
        import json as _json
        rm = {}
        for name, (g_min, g_max, K, is_egf) in model.servers.items():
            gs = list(range(int(g_min), int(g_max) + 1))
            ours = [float(g) if is_egf else float(implied_internal(model, name, g))
                    for g in gs]
            rm[name] = {"our": ours, "server": [float(g) for g in gs]}
        with open(args.out_rankmap, "w") as f:
            _json.dump(rm, f, indent=2)
        print(f"Wrote rank_map (per-server couplings) → {args.out_rankmap}")


if __name__ == "__main__":
    main()
