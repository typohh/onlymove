#!/usr/bin/env python3
"""Where along an axis does a motif actually PREDICT? Splits the held-out-user
walk's axis contribution by which TAIL of the axis the problem sits in.

The shipped rules are ONE-SIDED (`axes/labels.json` `components`): the positive
tail is the taught criterion, the negative tail is untaught residual — the rule
column is a constant there, so the fit is pulled toward a plateau it cannot hold
and spreads the corpus out however the likelihood wants. Whether that untaught
end does any predictive work is not something the identity report can answer: for
a one-sided rule the `-tail` figure it prints only asks whether the fitted
negative tail avoids the rule's positives, which ~84% of the corpus satisfies by
being ordinary. So it is measured here instead, on held-out prediction.

THE ATTRIBUTION IS PER ROW AND MARGINAL, NOT AN ABLATED RE-WALK. Zeroing K-1
loadings and re-walking is measured to give a NEGATIVE contribution for an axis
worth +4.41 mnat beside its neighbours, because the walk's axis prior is stated
on the TOTAL contribution `g @ v` in prediction space — drop K-1 axes and the
prior on the survivor loosens K-fold, so what the ablation measures is mostly its
own estimation noise. Here the walk is run ONCE at the full K, so `v` is the same
vector in every column, and axis k's contribution on a row is the loss it would
lose by dropping ITS term alone from that row's logit:

    delta_k = loss(z - g[k]*v[k]) - loss(z)

Summed over every row this is not a partition of the axis block's total (the
terms interact through the logistic), so both are printed: the per-axis marginals
and the block's own value with the whole axis term removed.

Buckets are the SAME tail convention the identity report and the teaching term
use — the loading z-scored over ATTEMPTED problems, tail at `TAIL_HINGE`. A third
of the corpus is never attempted and its loadings are unconstrained extrapolation,
so it has no place in a z-score the walk's rows are then judged against.

Usage:
  # shipped fit — loadings are IN-SAMPLE for every user, so the level is
  # optimistic; run it for the shape, not for a magnitude
  ./.venv/bin/python3 -m auxiliary.rating_dev.axis_by_tail

  # leak-free: a dev fit that never saw the scored accounts
  ./.venv/bin/python3 -m auxiliary.rating_dev.axis_by_tail \\
      --dump /tmp/uh_s0.npz --holdout-seed 0 --frac 0.15
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.rating_dev.eval_user_holdout import _load, forward_walk  # noqa: E402
from utils.rating.features import load_first_time_attempts  # noqa: E402
from utils.rating.tail_agreement import TAIL_HINGE  # noqa: E402

SIZES = [9, 13]
AXES_DIR = os.path.join(PROJECT_ROOT, "axes")
EPS = 1e-9


def load_shipped(ratings_dir, per_problem):
    """rate/slope/floor/lapse from a ratings dir, loadings from per_problem.csv.

    The MIRT writes its loadings per PROBLEM, so they are read straight off the
    csv rather than projected through a linear map the way the retired two-stage
    axes had to be — `eval_user_holdout.project_loadings` is that older path and
    does not apply to this model."""
    import pandas as pd
    from auxiliary.rating_dev.validate_ratings import load_ratings
    rate, slope, floor, lam, _th = load_ratings(ratings_dir)
    df = pd.read_csv(per_problem)
    cols = [c for c in df.columns if c.startswith("embed_")]
    cols.sort(key=lambda c: int(c.split("_")[1]))
    G = df[cols].to_numpy(float)
    load = {str(t): G[i] for i, t in enumerate(df["tsumego_id"].astype(str))}
    return rate, slope, floor, lam, load


def rule_columns():
    """{rule name: {tid: bool}} from the proxies, plus the shipped component spec.

    Selection is BY RULE NAME through the npz's own `rules` array — safe here in
    a way it is not for the motif store, because a rule name identifies exactly
    one column by construction."""
    p = os.path.join(AXES_DIR, "motif_proxies.npz")
    if not os.path.exists(p):
        return {}, []
    z = np.load(p, allow_pickle=False)
    rules = [str(x) for x in z["rules"]]
    lab = np.asarray(z["labels"], bool)
    tids = [str(t) for t in (z["tids"] if "tids" in z.files else z["tid"])]
    out = {r: dict(zip(tids, lab[:, i])) for i, r in enumerate(rules)}
    spec = []
    lp = os.path.join(AXES_DIR, "labels.json")
    if os.path.exists(lp):
        spec = [str(x) for x in json.load(open(lp)).get("components", [])]
    return out, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None,
                    help="a train_joint_mirt --dump-params npz. Use one trained "
                         "with --holdout-user-frac so the loadings never saw the "
                         "scored accounts.")
    ap.add_argument("--ratings-dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--per-problem",
                    default=os.path.join(AXES_DIR, "per_problem.csv"))
    ap.add_argument("--holdout-seed", type=int, default=None,
                    help="score only this seed's held-out users. Omit to score "
                         "everyone, which is only honest with a --dump whose "
                         "training excluded them, or as an explicitly in-sample "
                         "read of the shipped fit.")
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--min-prefix", type=int, default=0,
                    help="score only attempts with at least this much history. "
                         "The axis block is measured to pay ~nothing below a few "
                         "hundred attempts, so the whole-population figure is an "
                         "average over rows that carry no axis signal either way.")
    ap.add_argument("--halves", action="store_true",
                    help="also run the HALF ABLATION: separate causal walks with "
                         "each axis clipped to its positive side, to its negative "
                         "side, and per axis. Costs one extra walk per arm and is "
                         "the measurement the per-row table cannot make, since "
                         "that table's buckets share one fitted v.")
    ap.add_argument("--pin-prior", action="store_true",
                    help="freeze the v-space prior precision at the full model's "
                         "instead of letting each arm derive its own. This makes "
                         "the clipped arms MORE shrunk, not equally shrunk, and "
                         "reverses the result; it is here to show that, not to "
                         "be used.")
    ap.add_argument("--boot", type=int, default=1000,
                    help="user-clustered bootstrap reps; 0 disables")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    if args.dump:
        rate, slope, floor, lam, load = _load(None, args.dump)
        src = os.path.basename(args.dump)
    else:
        rate, slope, floor, lam, load = load_shipped(args.ratings_dir,
                                                     args.per_problem)
        src = "SHIPPED ratings/ + axes/per_problem.csv"
    if not load:
        sys.exit("no loadings in this source — nothing to split by tail")
    K = len(next(iter(load.values())))

    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)].sort_values("created_at")
    if args.holdout_seed is not None:
        from auxiliary.rating_dev.holdout_set import held_out_users
        hu_list, hu_hash = held_out_users(args.holdout_seed, args.frac)
        hu = set(hu_list)
        is_held = df["user_id"].astype(str).isin(hu)
        seen = set(df.loc[~is_held, "problem"].astype(str))
        df = df[is_held]
        scope = f"held-out users, seed {args.holdout_seed} (hash {hu_hash})"
    else:
        seen = set(df["problem"].astype(str))
        scope = "ALL users — loadings are IN-SAMPLE, level is optimistic"

    seqs = defaultdict(list)
    for u, t, ok in zip(df["user_id"].astype(str), df["problem"].astype(str),
                        df["success"]):
        if t in rate and t in load:
            seqs[u].append((t, int(ok)))

    # z-score each axis over the ATTEMPTED problems only — the same population
    # the identity report scores on, and the only one whose loadings are
    # constrained by anything.
    att = sorted({t for v in seqs.values() for t, _ in v})
    A = np.stack([load[t] for t in att])
    mu, sd = A.mean(axis=0), A.std(axis=0) + 1e-12
    zof = {t: (load[t] - mu) / sd for t in att}

    rules, spec = rule_columns()
    names = [spec[k] if k < len(spec) else f"axis {k}" for k in range(K)]

    # THE COUNTERFACTUAL SUBTRACTS THE CENTRED LOADING, NOT THE RAW ONE. `g` is
    # not centred over problems — the axis gauge normalises its RMS and `center_v`
    # removes the population mean of `v`, neither of which touches the per-axis
    # mean over the corpus — so `g @ v` carries a term `mu @ v` that is CONSTANT
    # across problems for one user. That term is a second ability parameter, not
    # motif information, and dropping the raw `g[k]*v[k]` would credit the axis
    # for it on every row including one sitting exactly at the mean. Subtracting
    # `(g[k] - mu[k]) * v[k]` asks what the tails are actually about: given that
    # we know where the USER sits, does knowing where THIS PROBLEM sits on motif k
    # help? The intercept is reported once, on its own line.
    if args.halves:
        # HALF ABLATION — the decisive form of the question, and the reason the
        # per-row table below cannot be it. Every bucket in that table shares ONE
        # fitted `v`, and `v` is fitted so the weighted residual is orthogonal to
        # the centred loading, so a term that helps in the tail is forced to hurt
        # slightly everywhere else whatever the truth is: the per-bucket signs are
        # partly a zero-sum redistribution rather than independent readings. Here
        # each arm is its OWN causal walk with its own `v`, scoring the SAME rows,
        # so the arms are paired and nothing is being split that must sum.
        #
        # `mu + max(g - mu, 0)` keeps the positive tail and flattens everything at
        # or below the mean onto one value — the axis can still say "this problem
        # is a ko" and can no longer say anything about how far from ko a problem
        # is. The prior precision is pinned at the FULL model's, since clipping
        # shrinks |g| and would otherwise loosen the prior in the arm being judged.
        # THE PRIOR MUST NOT BE PINNED IN v-SPACE, and pinning it is the one
        # thing that reverses this table's answer. `PRIOR_AX_SD` states the prior
        # on the CONTRIBUTION `g @ v` in prediction space, which is the only
        # scale-free way to state it, and the walk derives the v-space precision
        # as `mean|g|^2 / PRIOR_AX_SD^2` so that every model gets the same 0.5
        # logits however its loadings are scaled. Freezing that v-space number
        # across arms instead hands a clipped arm a TIGHTER prediction-space
        # prior in proportion to how much loading it lost: measured, clipping to
        # the negative side drops rms|g| from 2.46 to 0.85 and the effective
        # prior from 0.500 logits to 0.172. Under it "negative half only" wins
        # every configuration — because the axes' failure in this regime is v
        # estimation noise and 3x more shrinkage cures it, which is a finding
        # about shrinkage wearing a tail's clothes. Left unpinned each arm gets
        # the same 0.5 logits of licence to move a prediction using whatever
        # information it still has, which is the comparison the question wants.
        prior_prec = None
        if args.pin_prior:
            G = np.stack([load[t] for t in att])
            prior_prec = float((G ** 2).sum(axis=1).mean()) / (0.5 ** 2)
            print("    PRIOR PINNED in v-space — the clipped arms run under "
                  "tighter shrinkage; read the unpinned run for the tail answer")
        pos_only = {t: mu + np.maximum(load[t] - mu, 0.0) for t in load}
        neg_only = {t: mu + np.minimum(load[t] - mu, 0.0) for t in load}
        # WINSORIZING SEPARATES THE TWO EXPLANATIONS a bare half-ablation cannot.
        # A UNIFORM rescale of the loadings is exactly neutral here — scaling g by
        # alpha and v by 1/alpha leaves every prediction identical, and the prior
        # is derived from mean|g|^2 so it rescales with them — so a difference
        # between the halves is NOT about overall magnitude. What does differ is
        # the SHAPE: the taught side is heavy-tailed (skew +1.6 to +4.6), so a few
        # problems carry several times the average loading and therefore several
        # times the leverage on a `v` estimated from a short history. Capping the
        # tail at one sd keeps every problem's MEMBERSHIP of the motif and removes
        # only that leverage, so if the capped arm recovers, the taught tail was
        # being punished for its shape rather than for carrying no signal.
        sd_att = np.stack([load[t] for t in att]).std(axis=0)
        wins_pos = {t: mu + np.minimum(np.maximum(load[t] - mu, 0.0), sd_att)
                    for t in load}
        wins_all = {t: mu + np.clip(load[t] - mu, -sd_att, sd_att) for t in load}
        arms = [("rating only", None), ("full axes", load),
                ("positive half only", pos_only), ("negative half only", neg_only),
                ("pos half, capped at +1sd", wins_pos),
                ("full, capped at +/-1sd", wins_all)]
        for k in range(K):
            v = {t: load[t].copy() for t in load}
            for t in v:
                v[t][k] = mu[k] + max(v[t][k] - mu[k], 0.0)
            arms.append((f"  axis {k} {names[k]}: NEG side removed", v))
        for k in range(K):
            v = {t: load[t].copy() for t in load}
            for t in v:
                v[t][k] = mu[k] + min(v[t][k] - mu[k], 0.0)
            arms.append((f"  axis {k} {names[k]}: POS side removed", v))

        L, ru = [], None
        for tag, lv in arms:
            acc, us = [], []

            def on_row(u, tid, y, pos, th, vv, g, a, c, r, kap, span, p,
                       _a=acc, _u=us):
                if pos < args.min_prefix:
                    return
                _a.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
                _u.append(u)

            forward_walk(rate, slope, floor, lam, seqs, seen,
                         loadings=lv, k_axes=(K if lv else 0), on_row=on_row,
                         prior_prec=(prior_prec if lv else None))

            L.append(np.array(acc))
            if ru is None:
                ru = np.array(us)
            print(f"    walked {tag} ({len(acc)} rows)", flush=True)
        base = L[0]
        uu2, inv2 = np.unique(ru, return_inverse=True)
        by_u2 = [np.where(inv2 == i)[0] for i in range(len(uu2))]
        rng2 = np.random.default_rng(0)
        pick2 = (rng2.integers(0, len(uu2), (args.boot, len(uu2)))
                 if args.boot else None)

        def paired(d):
            """User-clustered bootstrap on a PAIRED per-row difference. Clusters
            are users: attempts within one share an ability estimate and an axis
            position, so resampling rows would understate the interval in the
            direction that makes a difference look real."""
            if pick2 is None:
                return None
            su = np.array([d[ix].sum() for ix in by_u2])
            nu = np.array([len(ix) for ix in by_u2], float)
            return np.percentile(su[pick2].sum(1) / nu[pick2].sum(1),
                                 [2.5, 97.5]) * 1000

        print(f"\n  HALF ABLATION — {len(base)} rows, every arm the same rows, "
              + ("prior PINNED in v-space (see the note in the source)"
                 if args.pin_prior else
                 "each arm at the same 0.5-logit prediction-space prior"))
        print(f"    {'arm':<34}{'logloss':>10}{'axis adds':>12}{'paired 95% CI':>22}")
        print(f"    {'rating only':<34}{base.mean():>10.4f}{'—':>12}{'—':>22}")
        for (tag, _lv), l in list(zip(arms, L))[1:]:
            c = paired(base - l)
            print(f"    {tag:<34}{l.mean():>10.4f}"
                  f"{(base - l).mean() * 1000:>+12.2f}"
                  + (f"{f'[{c[0]:+.2f}, {c[1]:+.2f}]':>22}" if c is not None
                     else f"{'—':>22}"))
        full = L[1]
        print(f"\n    vs FULL (positive = the full axis beats this arm):")
        for (tag, _lv), l in list(zip(arms, L))[2:]:
            c = paired(l - full)
            print(f"    {tag:<34}{'':>10}{(l - full).mean() * 1000:>+12.2f}"
                  + (f"{f'[{c[0]:+.2f}, {c[1]:+.2f}]':>22}" if c is not None
                     else f"{'—':>22}"))
        print()

    rows_u, rows_t, rows_y = [], [], []
    d_axis, d_block, d_icept = [], [], []

    def on_row(u, tid, y, pos, th, v, g, a, c, r, kap, span, p):
        if pos < args.min_prefix:
            return
        base = a * (th - r) + kap
        full = base + float(g @ v)
        gc = g - mu

        def loss_at(zz):
            zz = np.clip(zz, -35, 35)
            q = np.clip(c + span / (1 + np.exp(-zz)), EPS, 1 - EPS)
            return -(y * np.log(q) + (1 - y) * np.log(1 - q))

        lf = loss_at(full)
        rows_u.append(u)
        rows_t.append(tid)
        rows_y.append(y)
        d_axis.append([loss_at(full - gc[k] * v[k]) - lf for k in range(K)])
        d_block.append(loss_at(base) - lf)
        # every problem-specific component removed, the user's own offset kept:
        # the axis block acting as nothing but an ability shift
        d_icept.append(loss_at(base) - loss_at(base + float(mu @ v)))

    forward_walk(rate, slope, floor, lam, seqs, seen, loadings=load, k_axes=K,
                 on_row=on_row)

    if not rows_u:
        sys.exit("no rows scored")
    D = np.array(d_axis)                      # (n, K), mnat after scaling
    B = np.array(d_block)
    I = np.array(d_icept)
    Z = np.stack([zof[t] for t in rows_t])    # (n, K)
    users = np.array(rows_u)
    uu, inv = np.unique(users, return_inverse=True)
    n = len(B)

    print(f"\n{args.label or src}")
    print(f"  scope: {scope}")
    print(f"  {len(seqs)} users, {n} scored attempts"
          + (f", prefix >= {args.min_prefix}" if args.min_prefix else "")
          + f" | K={K} | tail at z >= {TAIL_HINGE}")
    print(f"  whole axis BLOCK adds {B.mean() * 1000:+.2f} mnat/attempt "
          f"(all K terms removed at once)")
    print(f"  of which mu @ v — a per-user INTERCEPT carrying no problem "
          f"information — is {I.mean() * 1000:+.2f} mnat")

    rng = np.random.default_rng(0)
    boot_pick = (rng.integers(0, len(uu), (args.boot, len(uu)))
                 if args.boot else None)
    by_u = [np.where(inv == i)[0] for i in range(len(uu))] if args.boot else None

    def ci(vals_per_row, mask):
        """User-clustered bootstrap CI on the masked mean. Clusters are USERS —
        attempts within one share an ability estimate and an axis position, so
        resampling rows would understate the interval in the direction that makes
        a difference look real."""
        if not args.boot or not mask.any():
            return None
        s = np.where(mask, vals_per_row, 0.0)
        us = np.array([s[ix].sum() for ix in by_u])
        un = np.array([mask[ix].sum() for ix in by_u], float)
        num = us[boot_pick].sum(axis=1)
        den = un[boot_pick].sum(axis=1)
        ok = den > 0
        return np.percentile(num[ok] / den[ok], [2.5, 97.5]) * 1000

    # A ONE-SIDED taught rule leaves NO NEGATIVE TAIL to threshold. The target is
    # two-valued — the rule's positives high, every other problem on one plateau
    # just below the mean — so the fitted loading is a spike slightly under zero
    # with a long positive tail, and z <= -1 holds 0.0% of the corpus on `ko` and
    # `life and death`. A fixed hinge can therefore report "the negative tail is
    # worth nothing" when what it measured is that the bucket is empty. Quantiles
    # populate both ends by construction and are the primary split here; the
    # hinge rows are printed underneath so the number stays comparable with the
    # identity report and the client's own gate.
    # The last column is ADDITIVE, not a percentage. A share-of-total reads
    # cleanly only while the total is comfortably non-zero, and these totals sit
    # near zero by construction on a population most of whose attempts carry no
    # axis signal — one axis printed 1124% and -2415% on neighbouring buckets of
    # a +0.01 mnat total, which is arithmetic, not a finding. `contrib` is the
    # bucket's own share of the axis's per-attempt mean, so the five quintiles
    # sum to the marginal total whatever its sign.
    hdr = (f"    {'bucket':<14}{'n':>9}{'share':>8}{'mean z':>9}"
           f"{'on-rule':>9}{'mnat/att':>11}{'95% CI':>18}{'contrib':>10}")
    QS = ((0, 20, "Q1 lowest"), (20, 40, "Q2"), (40, 60, "Q3"), (60, 80, "Q4"),
          (80, 100, "Q5 highest"))
    for k in range(K):
        z = Z[:, k]
        rname = names[k].split("/")[0]
        col = rules.get(rname)
        print(f"\n  axis {k}  {names[k]}"
              f"   (marginal total {D[:, k].mean() * 1000:+.2f} mnat/attempt)")
        print(hdr)
        # Quantiles of the loading OVER THE SCORED ROWS, so each bucket carries a
        # fifth of the attempts rather than a fifth of the corpus — serving is
        # difficulty-matched and motif occupancy is strongly non-uniform over
        # rank, so the two are not the same partition.
        cuts = np.percentile(z, [20, 40, 60, 80])
        buckets = []
        for lo, hi, lbl in QS:
            m = np.ones(len(z), bool)
            if lo:
                m &= z > cuts[lo // 20 - 1]
            if hi < 100:
                m &= z <= cuts[hi // 20 - 1]
            buckets.append((lbl, m))
        buckets.append((f"  z >= +{TAIL_HINGE:g}", z >= TAIL_HINGE))
        buckets.append((f"  z <= -{TAIL_HINGE:g}", z <= -TAIL_HINGE))
        for lbl, m in buckets:
            if not m.any():
                print(f"    {lbl:<14}{0:>9}{'':>8}{'':>9}{'':>9}{'empty':>11}")
                continue
            c = ci(D[:, k], m)
            onr = ""
            if col is not None:
                idx = np.where(m)[0]
                onr = f"{np.mean([bool(col.get(rows_t[i], False)) for i in idx]):.1%}"
            print(f"    {lbl:<14}{int(m.sum()):>9}{m.mean():>7.1%}"
                  f"{z[m].mean():>+9.2f}{onr:>9}"
                  f"{D[m, k].mean() * 1000:>+11.2f}"
                  + (f"{f'[{c[0]:+.2f}, {c[1]:+.2f}]':>18}" if c is not None
                     else f"{'—':>18}")
                  + f"{D[m, k].sum() / len(z) * 1000:>+10.2f}")

    print("\n  mnat/att = mean loss this axis's own term saves on the bucket's rows.")
    print("  contrib  = that bucket's share of the axis's per-attempt mean, so")
    print("             the five quintiles sum to the marginal total above.")
    print("  The two hinge rows OVERLAP the quintiles — they are the same rows")
    print("  re-cut at the shared tail threshold, not a sixth and seventh bucket.")


if __name__ == "__main__":
    main()
