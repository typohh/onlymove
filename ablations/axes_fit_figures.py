#!/usr/bin/env python3
"""Three properties of the SHIPPED joint fit, read back off `ratings/` + `axes/`
without training anything: how much rating leaks into the axis tails, whether any
axis is doubling as a second ability estimate, and how far the shipped axes sit
from the rule columns they were taught onto.

All three are things the fit ASSERTS about itself, so they are measured rather
than assumed. Nothing here refits the trunk — the loadings, the per-problem
triple and the abilities are read exactly as `pipeline.train_mirt` wrote them.

  axes.rating_leak_worst      the largest per-tail correlation between any axis
                              END and any of the shipped triple (rating, slope,
                              floor). This is the quantity ORTHO_RATING is scored
                              on, in the same shape: each end is the shared tail
                              weight of +g or -g (utils.rating.tail_agreement),
                              each triple column centred and normalised, so an
                              entry is a cosine between two centred columns. A
                              WHOLE-AXIS correlation cannot stand in for it — it
                              is linear and averages a symmetric dependence to
                              zero — and it is printed alongside as context only.
  axes.ability_corr_worst     the largest |corr(axis position, ability)| over the
                              fitted users. The fit does not persist its per-user
                              positions, so they are RE-DERIVED here: the
                              penalized MLE of one user's position with the
                              loadings, the abilities and the response parameters
                              all held at their shipped values, under the
                              trainer's own V_RIDGE and its per-attempt scaling.
                              Ability is NOT refitted alongside — refitting it
                              would orthogonalize the position against it by
                              construction and guarantee the answer. The verdict
                              is insensitive to the ridge (main() prints the
                              sweep).
  axes.taught_agreement_mean  hinged tail agreement between each axis's TAUGHT
  axes.taught_agreement_min   end and the rule column it was taught from, meaned
                              and minimised over the axes — `report_bank_identity`
                              recomputed, on the same population (problems that
                              are both attempted and banked) through the same
                              `hinged_r`. Every shipped component is one-sided,
                              so the taught end is the positive one; the negative
                              end is untaught residual and is printed, never
                              summarised.

Two population choices are stated rather than defaulted. The leakage runs over
EVERY fitted problem, unweighted: what ships is one loading per problem over the
whole corpus, and that is the population a client draws from. The trainer
optimises the same quantity attempt-weighted over attempted problems, which is a
different number and is printed beside it. The identity report runs over
attempted+banked problems because that is what `report_bank_identity` scores —
the never-attempted half's loadings are unconstrained extrapolation whose
extremes carry no claim.

    ./.venv/bin/python3 -m ablations.axes_fit_figures
"""

import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from utils.rating.features import (  # noqa: E402
    load_backend_problem_params, load_first_time_attempts)
from utils.rating.model_constants import (  # noqa: E402
    K_AXES, LAPSE_CAP, SIZES, clear_logit_vec)
from utils.rating.tail_agreement import TAIL_HINGE, hinged_r, tail_weight  # noqa: E402

TIER = "methodology"

AXES_DIR = os.path.join(PROJECT_ROOT, "axes")
RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")
PER_PROBLEM = os.path.join(AXES_DIR, "per_problem.csv")
LABELS = os.path.join(AXES_DIR, "labels.json")
USER_THETA = os.path.join(RATINGS_DIR, "user_theta.csv")
RATING_CONSTANTS = os.path.join(RATINGS_DIR, "rating_constants.json")

# The trainer's own per-user position penalty, and its scaling. The objective is
# mean-over-attempts BCE plus V_RIDGE * mean-over-attempts ||v||^2, so one user's
# share of it is their own negative log-likelihood plus V_RIDGE * n_attempts *
# ||v||^2 — the ridge grows with the record rather than shrinking against it.
V_RIDGE = 0.5
MAX_NEWTON = 200
# Ridge settings the ability-correlation verdict is checked against, as multiples
# of the per-user total: the shipped form and two constants that do not scale
# with the record at all. Reported, never quoted — the figure is the first.
RIDGE_ARMS = ("shipped", "constant-0.5", "constant-5")
# A user whose damped Newton does not converge is dropped rather than scored at
# whatever the last iteration held. It should be a handful; past this the
# positions are not being recovered and the figure is not measuring what it says.
MAX_DROP_SHARE = 0.01


# --- the shipped fit, as artifacts -------------------------------------------

def load_shipped():
    """Loadings, the per-problem triple, the abilities and the fit's lapse, keyed
    the way `train_mirt` keyed them: (size, tsumego_id) in `per_problem.csv` row
    order, which is the fit's own problem order."""
    import pandas as pd
    require(os.path.exists(PER_PROBLEM), f"no {PER_PROBLEM} — run pipeline.train_mirt")
    per = pd.read_csv(PER_PROBLEM)
    cols = [f"embed_{k}" for k in range(K_AXES)]
    missing = [c for c in cols + ["tsumego_id", "size"] if c not in per.columns]
    require(not missing, f"{PER_PROBLEM} lacks {missing}")
    extra = [c for c in per.columns if c.startswith("embed_") and c not in cols]
    require(not extra, f"{PER_PROBLEM} carries {extra} past K_AXES={K_AXES}")
    G = per[cols].to_numpy(np.float64)
    require(np.all(np.isfinite(G)), f"{PER_PROBLEM}: non-finite loading")

    # The axis gauge divides every loading by the corpus RMS of its own axis, so
    # a loading is already in axis-standard-deviation units. Everything below
    # reports in those units, so the claim is checked rather than inherited.
    rms = np.sqrt((G ** 2).mean(axis=0))
    require(np.all(np.abs(rms - 1.0) < 0.01),
            f"axis loadings are not standardized: per-axis RMS {rms.round(4)}")

    keys = list(zip(per["size"].astype(int), per["tsumego_id"].astype(str)))
    params = load_backend_problem_params(tuple(SIZES))
    absent = [t for _, t in keys if t not in params]
    require(not absent, f"{len(absent)} fitted problems have no shipped triple "
                        f"(e.g. {absent[:3]}) — ratings/ and axes/ are from "
                        f"different runs")
    tri = np.array([params[t] for _, t in keys], dtype=np.float64)
    require(np.all(np.isfinite(tri)), "a shipped rating/slope/floor is not finite")
    require(np.all(tri[:, 1] > 0.0), "a shipped slope is not positive")
    require(np.all((tri[:, 2] >= 0.0) & (tri[:, 2] < 1.0)), "a shipped floor is outside [0, 1)")

    require(os.path.exists(RATING_CONSTANTS), f"no {RATING_CONSTANTS}")
    with open(RATING_CONSTANTS) as f:
        lapse = float(json.load(f)["fat_finger_rate"])
    require(0.0 <= lapse <= LAPSE_CAP, f"lapse {lapse} outside [0, {LAPSE_CAP}]")

    require(os.path.exists(USER_THETA), f"no {USER_THETA}")
    ut = pd.read_csv(USER_THETA)
    for c in ("profile", "theta", "n_attempts"):
        require(c in ut.columns, f"{USER_THETA} lacks {c}")
    theta = dict(zip(ut["profile"].astype(str), ut["theta"].astype(float)))
    require(len(theta) == len(ut), f"{USER_THETA} repeats a profile")
    require(np.all(np.isfinite(list(theta.values()))), f"{USER_THETA}: non-finite ability")
    return {"keys": keys, "G": G, "tri": tri, "lapse": lapse, "theta": theta,
            "fit_attempts": int(ut["n_attempts"].sum())}


def attempt_rows(keys, theta):
    """The fit's own attempt rows, rebuilt by the trainer's filter: first-time
    attempts on the shipped board sizes, users carrying both outcomes, problems
    the fit holds. Returns (problem index per row, user id per row).

    The attempt log only ever grows, so this is the fit's population plus
    whatever landed after it. Restricting to the profiles the fit actually
    scored keeps the extra rows to the tail of an existing user's history."""
    import pandas as pd
    key_idx = {k: i for i, k in enumerate(keys)}
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)].copy()
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    keep = set(g.index[(g["c"] >= 1) & (g["w"] >= 1)])
    df = df[df["user_id"].isin(keep)].copy()
    prob2size = dict(zip(df["problem"], df["size"]))
    df["pi"] = df["problem"].map(lambda t: key_idx.get((prob2size.get(t), t)))
    df = df[df["pi"].notna()].copy()
    df["uid"] = df["user_id"].astype(str)
    scored = set(df["uid"])
    lost = [u for u in theta if u not in scored]
    require(not lost, f"{len(lost)} users the fit scored have no attempts left "
                      f"(e.g. {lost[:3]}) — the attempt log is behind the fit")
    df = df[df["uid"].isin(theta)]
    return (df["pi"].to_numpy(np.int64), df["uid"].to_numpy(object),
            df["success"].to_numpy(np.float64))


# --- leakage: each axis END against the shipped triple ------------------------

def tail_columns(M, sign):
    """The shared tail weight applied PER COLUMN. `tail_weight` z-scores over
    everything finite it is handed, so a whole (problem, axis) matrix would be
    hinged against pooled statistics; the trainer's torch twin works per axis and
    this has to match it."""
    return np.column_stack([tail_weight(sign * M[:, k]) for k in range(M.shape[1])])


def _unit_columns(M):
    M = M - M.mean(axis=0)
    n = np.linalg.norm(M, axis=0)
    require(np.all(n > 0), "a column is constant and cannot be normalised")
    return M / n


def leakage(G, tri):
    """(K, 3) per-tail correlation for each end, as {'+': …, '-': …}. Entry
    [k, j] is the cosine between axis k's hinged end and centred triple column j
    — exactly the matrix ORTHO_RATING squares and sums."""
    tz = _unit_columns(tri)
    return {nm: _unit_columns(tail_columns(G, sg)).T @ tz
            for sg, nm in ((1.0, "+"), (-1.0, "-"))}


def whole_axis(G, tri):
    """The linear statistic the per-tail form replaces, for context only."""
    return _unit_columns(G).T @ _unit_columns(tri)


# --- identity: the taught ends against their rule columns ---------------------

def bank_identity(keys, G, prob_idx):
    """(+tail, -tail, bulk) per axis against the taught rule columns, plus the
    spec and the scored population.

    The target columns come from `pipeline.train_mirt.load_bank`, not from a
    restatement of it: the `"A/B"` pairing, the tail-weighted glue and the
    z-scoring are the fit's own, and a second copy of them would agree with the
    fit only by convention."""
    import hashlib
    from pipeline.train_mirt import load_bank
    key_idx = {k: i for i, k in enumerate(keys)}
    tgt, ok, spec = load_bank(key_idx)
    require(len(spec) == K_AXES, f"the bank spec is {len(spec)} columns, not K_AXES={K_AXES}")
    with open(LABELS) as f:
        components = [str(c) for c in json.load(f)["components"]]
    require(len(components) >= K_AXES,
            f"{LABELS} lists {len(components)} components against K_AXES={K_AXES}")
    # The fit stamps a digest of the spec it was actually taught, so the rules
    # scored here can be proved to be the rules the shipped axes were pulled
    # toward — a `components` list edited after the fit would otherwise be
    # scored against loadings that never saw it, and nothing else would notice.
    shipped = np.load(os.path.join(AXES_DIR, "axes.npz"), allow_pickle=False)
    require(int(shipped["k"]) == K_AXES,
            f"axes.npz ships K={int(shipped['k'])} against K_AXES={K_AXES}")
    stamped = str(shipped["axis_bank"])
    digest = hashlib.sha256("\n".join(spec).encode()).hexdigest()[:16]
    require(stamped == digest,
            f"the shipped fit was taught bank {stamped}, these rules hash to "
            f"{digest} — labels.json has moved since the fit")

    att = np.zeros(len(G), bool)
    att[np.unique(prob_idx)] = True
    sel = np.where(att & ok)[0]
    require(len(sel) > 1000, f"only {len(sel)} attempted+banked problems to score on")
    pos, neg, bulk = [], [], []
    for k in range(K_AXES):
        x, y = G[sel, k], tgt[sel, k]
        pos.append(hinged_r(x, y))
        neg.append(hinged_r(-x, -y))
        bulk.append(float(np.corrcoef(x, y)[0, 1]))
    pos = np.array(pos, float)
    require(np.all(np.isfinite(pos)),
            "a taught tail scores NaN — its hinged column is constant, so the "
            "axis makes no claim at that end")
    # Every shipped component is one-sided, so the taught end is the positive
    # one. A bipolar "A/B" column teaches BOTH ends and the summary below would
    # then be dropping half the evidence.
    two_sided = [s for s in spec if "/" in s]
    require(not two_sided,
            f"bipolar components {two_sided} teach both ends; the taught-end "
            f"summary here covers the positive tail only")
    return {"spec": spec, "pos": pos, "neg": np.array(neg, float),
            "bulk": np.array(bulk, float), "n_scored": len(sel),
            "n_banked": int(ok.sum())}


# --- the per-user axis positions, re-derived ----------------------------------

def fit_position(g, off, floor, y, lapse, ridge):
    """Damped Newton on one user's axis position with everything else FIXED.

    z = slope·(ability − difficulty) + g·v is the model's own logit, and the 4PL
    floor and lapse enter through dp/dz, so the Newton weights are not p(1−p).
    Undamped Newton diverges on a few users per hundred here — the loadings carry
    outliers an order of magnitude past their own spread — and an undamped run
    then returns whatever the last iteration held, so the step is Armijo-tested
    on the penalized objective and a non-convergence is reported as one.

    Returns (position, converged)."""
    span = 1.0 - floor - lapse

    def obj(v):
        s = 1.0 / (1.0 + np.exp(-np.clip(off + g @ v, -35, 35)))
        p = np.clip(floor + span * s, 1e-9, 1.0 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p)).sum() + ridge * (v @ v)

    v = np.zeros(g.shape[1])
    f = obj(v)
    for _ in range(MAX_NEWTON):
        s = 1.0 / (1.0 + np.exp(-np.clip(off + g @ v, -35, 35)))
        p = np.clip(floor + span * s, 1e-9, 1.0 - 1e-9)
        dp = span * s * (1 - s)
        grad = g.T @ ((y - p) * dp / (p * (1 - p))) - 2 * ridge * v
        w = dp * dp / (p * (1 - p))
        H = (g * w[:, None]).T @ g + 2 * ridge * np.eye(g.shape[1])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return v, False
        dec = float(grad @ step)
        if dec < 0:
            return v, False
        if dec < 1e-10:
            return v, True
        t = 1.0
        while t > 1e-12:
            fn = obj(v + t * step)
            if fn <= f - 1e-4 * t * dec:
                v, f = v + t * step, fn
                break
            t /= 2
        else:
            return v, False
    return v, False


def user_positions(shipped, rows, arm="shipped"):
    """Every fitted user's axis position and ability. `arm` names the ridge:
    "shipped" is the trainer's V_RIDGE scaled by the user's attempt count, the
    two constants do not scale with the record at all."""
    G, tri, lapse = shipped["G"], shipped["tri"], shipped["lapse"]
    rating, slope, floor = tri[:, 0], tri[:, 1], tri[:, 2]
    # The fit's difficulty, recovered from what it shipped: the rating IS
    # b + kappa(floor)/slope, so this inverts the same expression train_mirt
    # applied rather than re-deriving anything.
    diff = rating - clear_logit_vec(floor, lapse) / np.maximum(slope, 1e-6)

    pi, uid, y = rows
    order = np.argsort(uid, kind="stable")
    pi, uid, y = pi[order], uid[order], y[order]
    bounds = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1], True])

    pos, ability, counts, dropped = [], [], [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        u = str(uid[lo])
        idx = pi[lo:hi]
        th = shipped["theta"][u]
        ridge = {"shipped": V_RIDGE * (hi - lo),
                 "constant-0.5": 0.5, "constant-5": 5.0}[arm]
        v, ok = fit_position(G[idx], slope[idx] * (th - diff[idx]),
                             np.clip(floor[idx], 0.0, 1.0), y[lo:hi], lapse, ridge)
        if not ok:
            dropped.append(u)
            continue
        pos.append(v)
        ability.append(th)
        counts.append(hi - lo)
    P = np.array(pos)
    require(len(P) > 100, f"only {len(P)} users' positions recovered")
    require(len(dropped) <= MAX_DROP_SHARE * (len(P) + len(dropped)),
            f"{len(dropped)} of {len(P) + len(dropped)} users' positions did not "
            f"converge — the recovery is not measuring what it claims")
    # Centred, exactly as the model reads them: the population mean is not a
    # position, and the likelihood never sees it.
    return (P - P.mean(axis=0), np.array(ability, float),
            np.array(counts, np.int64), len(dropped))


def ability_correlation(P, ability):
    return np.array([float(np.corrcoef(P[:, k], ability)[0, 1])
                     for k in range(P.shape[1])])


# --- the figures --------------------------------------------------------------

def measure():
    shipped = load_shipped()
    G, tri = shipped["G"], shipped["tri"]
    rows = attempt_rows(shipped["keys"], shipped["theta"])
    pi = rows[0]

    ends = leakage(G, tri)
    leak_all = np.stack([ends["+"], ends["-"]])
    leak_attempt = np.stack(list(leakage(G[pi], tri[pi]).values()))

    ident = bank_identity(shipped["keys"], G, pi)
    P, ability, counts, n_drop = user_positions(shipped, rows)
    corr = ability_correlation(P, ability)
    bulk = whole_axis(G, tri)

    return {
        "n_problems": len(G), "n_attempts": len(pi),
        "n_users": len(P), "n_dropped": n_drop,
        "fit_attempts": shipped["fit_attempts"],
        "lapse": shipped["lapse"],
        "ends": ends,
        "leak_worst": float(np.abs(leak_all).max()),
        "leak_worst_attempt_weighted": float(np.abs(leak_attempt).max()),
        "whole_axis": bulk,
        "whole_axis_worst": float(np.abs(bulk).max()),
        "ident": ident,
        "taught_mean": float(ident["pos"].mean()),
        "taught_min": float(ident["pos"].min()),
        "taught_max": float(ident["pos"].max()),
        "ability_corr": corr,
        "ability_corr_worst": float(np.abs(corr).max()),
        "position_sd": P.std(axis=0),
        "attempts_per_user": counts,
        "shipped": shipped, "rows": rows,
    }


def figures():
    m = measure()
    return [
        Figure("axes.rating_leak_worst", m["leak_worst"], ".2f",
               "largest correlation between one END of a shipped axis (the shared "
               "tail weight of +g or -g) and one of the shipped per-problem "
               "quantities (rating, slope, floor), over every fitted problem — the "
               "quantity the rating-orthogonality penalty is scored on"),
        Figure("axes.ability_corr_worst", m["ability_corr_worst"], ".2f",
               "largest |correlation| between a user's position on a shipped axis "
               "and their fitted ability; positions re-derived as the penalized MLE "
               "against the shipped loadings, abilities and response parameters, "
               "which the fit does not persist"),
        Figure("axes.taught_agreement_mean", m["taught_mean"], ".2f",
               f"hinged tail agreement (z >= {TAIL_HINGE}) between each axis's "
               f"taught end and the rule column it was taught from, meaned over "
               f"the {K_AXES} axes, on attempted+banked problems"),
        Figure("axes.taught_agreement_min", m["taught_min"], ".2f",
               "the same agreement at the weakest-held axis"),
    ]


def main():
    m = measure()
    ident = m["ident"]
    names = [s[:20] for s in ident["spec"]]
    print(f"\nthe shipped fit, read back off ratings/ + axes/ — {m['n_problems']:,d} "
          f"problems, {m['n_users']:,d} users, {m['n_attempts']:,d} attempt rows "
          f"(the fit recorded {m['fit_attempts']:,d}), lapse {m['lapse']}")

    print("\n=== rating leakage, PER TAIL ===")
    print("cosine between one hinged axis END and one centred shipped quantity,")
    print("over every fitted problem. This is the penalised statistic; the "
          "whole-axis\ncolumn beside it is the linear one it replaces, which "
          "averages a symmetric\ndependence to zero and is context only.")
    print(f"{'axis':<22}{'+rating':>9}{'+slope':>8}{'+floor':>8}"
          f"{'-rating':>10}{'-slope':>8}{'-floor':>8}   {'whole-axis r,s,f':>22}")
    for k in range(K_AXES):
        w = m["whole_axis"][k]
        print(f"{names[k]:<22}" + "".join(f"{v:>9.3f}" for v in m["ends"]["+"][k])
              + "".join(f"{v:>{10 if j == 0 else 8}.3f}" for j, v in enumerate(m["ends"]["-"][k]))
              + "   " + " ".join(f"{v:+.3f}" for v in w))
    print(f"  worst per-tail |cosine|           {m['leak_worst']:.3f}   "
          f"(attempt-weighted over attempted problems, the population the "
          f"penalty sees: {m['leak_worst_attempt_weighted']:.3f})")
    print(f"  worst whole-axis |correlation|    {m['whole_axis_worst']:.3f}")

    print("\n=== identity against the taught rules ===")
    print(f"tail-weighted agreement at z >= {TAIL_HINGE}, 0 = independent, over "
          f"{ident['n_scored']:,d}\nattempted+banked problems of "
          f"{ident['n_banked']:,d} banked. Every component is one-sided, so the\n"
          f"+tail is the taught end and the -tail is untaught residual.")
    print(f"{'axis':<22}{'+tail (taught)':>16}{'-tail':>9}{'bulk r':>9}")
    for k in range(K_AXES):
        print(f"{names[k]:<22}{ident['pos'][k]:>16.3f}{ident['neg'][k]:>9.3f}"
              f"{ident['bulk'][k]:>+9.3f}")
    print(f"  taught ends: mean {m['taught_mean']:.3f}, "
          f"min {m['taught_min']:.3f}, max {m['taught_max']:.3f}")

    print("\n=== axis positions against ability ===")
    print("positions re-derived per user at the trainer's own ridge, with the "
          "loadings,\nabilities and response parameters frozen. Ability is NOT "
          "refitted alongside:\ndoing so would orthogonalize the position against "
          "it by construction.")
    print(f"{'axis':<22}{'corr w/ ability':>17}{'position sd':>13}")
    for k in range(K_AXES):
        print(f"{names[k]:<22}{m['ability_corr'][k]:>+17.3f}{m['position_sd'][k]:>13.4f}")
    print(f"  worst |correlation| {m['ability_corr_worst']:.3f} over "
          f"{m['n_users']:,d} users"
          + (f", {m['n_dropped']} dropped for non-convergence" if m["n_dropped"] else "")
          + f"; median history {int(np.median(m['attempts_per_user']))} attempts")
    print("  ridge sensitivity (the shipped per-attempt scaling against two "
          "constants):")
    for arm in RIDGE_ARMS:
        P, ability, _, drop = user_positions(m["shipped"], m["rows"], arm)
        c = ability_correlation(P, ability)
        print(f"    {arm:<14} worst |r| {np.abs(c).max():.3f}   "
              f"position sd {P.std(axis=0).mean():.3f}"
              + (f"   ({drop} dropped)" if drop else ""))


if __name__ == "__main__":
    main()
