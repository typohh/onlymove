"""Which inputs drive a shipped skill axis, and how concentrated is that reliance?

Companion to auxiliary/rating_dev/inspect_residual_axes.py: that one shows the
extreme PROBLEMS on an axis (what it looks like), this one shows the extreme
INPUTS (what it reads). Together they are what a human needs to name a new axis
that build_axes grew.

Three views, because they answer different questions:

  WEIGHT        The axis head is a linear map on standardized columns, so
                |W[k, j]| IS input j's influence per standard deviation. Needs
                nothing but axes.npz -- instant, and available with --quick.

  CONCENTRATION Whether the axis reads a few columns hard or many columns
                weakly. Reported as the PARTICIPATION RATIO, (sum w^2)^2 /
                sum w^4 -- the effective number of columns, equal to the block
                width when weight is spread evenly and falling toward 1 as it
                concentrates. A block can dominate the total simply by being
                wide (the activation block has 128 columns to work with), so
                the per-column density and the effective count say far more
                about importance than the raw share does.

  TAIL PROFILE  The two tails separately. The head is linear and therefore
                symmetric in the weights, but the FEATURE DISTRIBUTIONS are not,
                so what puts a problem in the positive tail need not mirror what
                puts one in the negative tail. Reported as the exact additive
                decomposition: an axis value is sum_j W[k,j] * z_j, so the mean
                of W[k,j]*z_j over a tail's problems is precisely that column's
                contribution to the tail, and the parts sum to the whole.

Every block is named except the activations, which are joint-PCA components and
carry no individual meaning -- they are reported in aggregate only.

One naming caveat: the "bot winrate" block is whatever attempts/<size>/*.txt the
loader globs, which is mostly bots but also picks up per-problem pass-rate-like
scalars (heuristic, sgf_move). They are printed under their own file names, so
read the block as "pass-rate columns" rather than assuming every row is a bot.

Run:
  ./.venv/bin/python3 -m auxiliary.inspect_axis_inputs                # every axis
  ./.venv/bin/python3 -m auxiliary.inspect_axis_inputs --axis 4       # just one
  ./.venv/bin/python3 -m auxiliary.inspect_axis_inputs --quick        # no feature load
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

EMBED_DIR = os.path.join(PROJECT_ROOT, "axes")
N_ALT_SLOTS = 10                      # winning move + 9 alternatives
MOVE_COLS_PER_SLOT = 2                # (logit win-prob, score)


def bot_column_names(n_expected):
    """The bot-winrate columns are one per attempts/<size>/*.txt file, in the
    sorted intersection across boards -- the exact order load_joint_bot_winrate
    builds. That loader then drops pooled-zero-variance columns, which would
    shift every name after the dropped one, so the names are only returned when
    the count still matches; otherwise the caller falls back to indices rather
    than printing a confidently wrong bot."""
    from utils.rating.model_constants import SIZES
    sets = []
    for s in SIZES:
        d = os.path.join(PROJECT_ROOT, "attempts", str(s))
        sets.append({os.path.basename(f) for f in glob.glob(os.path.join(d, "*.txt"))})
    common = sorted(set.intersection(*sets)) if sets else []
    names = [c[:-4] if c.endswith(".txt") else c for c in common]
    if len(names) != n_expected:
        return None
    return names


def column_names(n_feat, act_dim):
    """(names, blocks) for the feature vector build_axis_features concatenates:
    activations | bot winrate | move eval | phase | size | rating,slope,floor | rank."""
    n_move = N_ALT_SLOTS * MOVE_COLS_PER_SLOT
    n_bot = n_feat - act_dim - n_move - 2 - 1 - 3 - 1
    names = [f"act_PC{i}" for i in range(act_dim)]
    blocks = ["activation"] * act_dim

    bots = bot_column_names(n_bot)
    for i in range(n_bot):
        names.append(bots[i] if bots else f"bot_col_{i}")
        blocks.append("bot winrate")

    for slot in range(N_ALT_SLOTS):
        who = "solution" if slot == 0 else f"alt{slot}"
        names += [f"{who}_winprob", f"{who}_score"]
        blocks += ["move eval"] * 2

    names += ["phase_moves_before", "phase_main_line_len"]
    blocks += ["phase"] * 2
    names += ["size_is_13"]
    blocks += ["size"]
    names += ["rating", "slope", "floor", "rank"]
    blocks += ["rating model"] * 4
    assert len(names) == n_feat, (len(names), n_feat)
    return names, blocks


def _fit(name, width=42):
    """Bot identifiers run long (network, steps and temperature); truncate from
    the LEFT so the distinguishing tail -- the rank or temperature -- survives."""
    return name if len(name) <= width else "\u2026" + name[-(width - 1):]


def participation_ratio(w):
    """Effective number of columns carrying the weight: equals len(w) when the
    weight is spread evenly, falls toward 1 as it concentrates on one column."""
    s2 = float((w ** 2).sum())
    s4 = float((w ** 4).sum())
    return (s2 * s2 / s4) if s4 > 0 else 0.0


def cover_count(w, frac):
    """How many columns, largest first, hold `frac` of the squared weight."""
    v = np.sort(w ** 2)[::-1]
    if v.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(v) / v.sum(), frac) + 1)


def load_axes():
    ax = np.load(os.path.join(EMBED_DIR, "axes.npz"), allow_pickle=True)
    labels = {}
    lp = os.path.join(EMBED_DIR, "labels.json")
    if os.path.exists(lp):
        labels = json.load(open(lp))
    return ax, labels


def describe_label(labels, k):
    comps = (labels or {}).get("components", [])
    if k >= len(comps):
        return "?", "(no label entry)"
    c = comps[k] or {}
    pos, neg = c.get("positive"), c.get("negative")
    tail = f"+{pos or '<unnamed>'} / -{neg or '<unnamed>'}"
    return c.get("motif_id", "?"), tail


def report_weights(k, w, names, blocks, top):
    print(f"\n  WEIGHT AND CONCENTRATION  (effective columns = participation ratio)")
    print(f"    {'block':<14}{'cols':>6}{'share |W|^2':>13}{'eff cols':>10}"
          f"{'per col':>9}{'50%/90% in':>13}")
    order = ["activation", "bot winrate", "move eval", "phase", "size", "rating model"]
    tot = float((w ** 2).sum())
    for b in order:
        idx = [i for i, bb in enumerate(blocks) if bb == b]
        if not idx:
            continue
        wb = w[idx]
        share = float((wb ** 2).sum() / tot)
        print(f"    {b:<14}{len(idx):>6}{share:>12.1%}{participation_ratio(wb):>10.1f}"
              f"{share / len(idx):>8.2%}{cover_count(wb, 0.5):>7}/{cover_count(wb, 0.9):<5}")
    print(f"    {'ALL':<14}{len(w):>6}{1.0:>12.1%}{participation_ratio(w):>10.1f}"
          f"{'':>8} {cover_count(w, 0.5):>6}/{cover_count(w, 0.9):<5}")

    named = [i for i, b in enumerate(blocks) if b != "activation"]
    named.sort(key=lambda i: -abs(w[i]))
    print(f"\n  TOP {top} NAMED INPUTS BY WEIGHT  (sign = direction of the POSITIVE tail)")
    for i in named[:top]:
        arrow = "raises" if w[i] > 0 else "lowers"
        print(f"    {_fit(names[i]):<43}{blocks[i]:<13}W {w[i]:+7.3f}  {arrow} the axis")


def report_tails(k, w, names, blocks, F, y, tail_frac, top):
    """Exact additive decomposition per tail: axis value = sum_j W[j]*z_j, so the
    mean of W[j]*z_j over a tail's problems is that column's contribution to it."""
    n = len(y)
    m = max(1, int(round(n * tail_frac)))
    order = np.argsort(y)
    lo, hi = order[:m], order[-m:]
    contrib_hi = (F[hi] * w[None, :]).mean(0)
    contrib_lo = (F[lo] * w[None, :]).mean(0)
    act = [i for i, b in enumerate(blocks) if b == "activation"]
    named = [i for i, b in enumerate(blocks) if b != "activation"]

    for tag, contrib, idxs in (("POSITIVE", contrib_hi, hi), ("NEGATIVE", contrib_lo, lo)):
        total = float(contrib.sum())
        a = float(contrib[act].sum())
        nmd = float(contrib[named].sum())
        print(f"\n  {tag} TAIL  ({m} problems, mean axis value {y[idxs].mean():+.3f})")
        print(f"    total contribution {total:+.3f}   "
              f"activations {a:+.3f} ({abs(a) / max(abs(total), 1e-9):.0%})   "
              f"named {nmd:+.3f} ({abs(nmd) / max(abs(total), 1e-9):.0%})")
        ranked = sorted(named, key=lambda i: -abs(contrib[i]))[:top]
        print(f"    {'named input':<43}{'block':<13}{'contrib':>9}{'mean z':>9}")
        for i in ranked:
            print(f"    {_fit(names[i]):<43}{blocks[i]:<13}{contrib[i]:>+9.3f}"
                  f"{F[idxs, i].mean():>+9.2f}")


def report_identity(k, pp, ycol):
    """Correlation against every frozen anchor -- the same quantity build_axes
    matches on, so it shows whether this axis is a known motif, a near-miss, or
    genuinely new."""
    ap = os.path.join(EMBED_DIR, "label_anchors.npz")
    if not os.path.exists(ap):
        return
    a = np.load(ap, allow_pickle=False)
    ids = [str(x) for x in a["ids"]]
    atids = [str(t) for t in a["tids"]]
    idx = {t: i for i, t in enumerate(pp["tsumego_id"].astype(str))}
    rows = [(i, idx[t]) for i, t in enumerate(atids) if t in idx]
    if not rows:
        return
    ai = np.array([r[0] for r in rows])
    pi = np.array([r[1] for r in rows])
    y = pp[ycol].to_numpy()[pi]
    ys = (y - y.mean()) / (y.std() + 1e-12)
    print(f"\n  IDENTITY  correlation with each frozen anchor (build_axes matches at 0.8)")
    for j, mid in enumerate(ids):
        x = a["M"][j][ai]
        xs = (x - x.mean()) / (x.std() + 1e-12)
        c = float((xs * ys).mean())
        note = ""
        if abs(c) >= 0.8:
            note = "  <- matched"
        elif abs(c) >= 0.4:
            note = "  <- related, below the bar"
        print(f"    {mid:<16}{c:+.3f}{note}")


def report_overlap(k, pp, labels, n_axes):
    """How much this axis shares with each already-named one. The axes carry only
    a LENIENT mutual orthogonality penalty, so they are not forced independent --
    overlap here is real shared structure, and a new axis that sits close to a
    named one is a candidate for being that motif under another name rather than
    a separate finding."""
    y = pp[f"embed_{k}"].to_numpy(float)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    print(f"\n  OVERLAP WITH THE OTHER SHIPPED AXES")
    for j in range(n_axes):
        if j == k or f"embed_{j}" not in pp.columns:
            continue
        o = pp[f"embed_{j}"].to_numpy(float)
        os_ = (o - o.mean()) / (o.std() + 1e-12)
        c = float((os_ * ys).mean())
        _mid, tail = describe_label(labels, j)
        flag = "  <- substantial" if abs(c) >= 0.3 else ""
        print(f"    axis {j}  {tail:<34}{c:+.3f}{flag}")


def report_matrix(pp, labels, n_axes):
    cols = [j for j in range(n_axes) if f"embed_{j}" in pp.columns]
    if len(cols) < 2:
        return
    Z = pp[[f"embed_{j}" for j in cols]].to_numpy(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    C = Z.T @ Z / len(Z)
    print("\n" + "=" * 78)
    print("SHARED STRUCTURE BETWEEN SHIPPED AXES (pairwise correlation)")
    print("=" * 78)
    print("      " + "".join(f"{'ax' + str(j):>9}" for j in cols))
    for a, j in enumerate(cols):
        print(f"  ax{j:<3}" + "".join(f"{C[a, b]:>+9.3f}" for b in range(len(cols))))
    off = [abs(C[a, b]) for a in range(len(cols)) for b in range(len(cols)) if a != b]
    print(f"\n  off-diagonal |corr|: max {max(off):.3f}   mean {np.mean(off):.3f}")
    for a, j in enumerate(cols):
        _m, tail = describe_label(labels, j)
        print(f"    ax{j}  {tail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", type=int, default=None, help="axis index (default: all)")
    ap.add_argument("--top", type=int, default=12, help="rows per ranking")
    ap.add_argument("--tail-frac", type=float, default=0.02,
                    help="fraction of problems in each tail")
    ap.add_argument("--quick", action="store_true",
                    help="weights only — skips the feature load, so no tail profiles")
    args = ap.parse_args()

    ax, labels = load_axes()
    W = ax["W"]
    act_dim = int(ax["act_dim"])
    names, blocks = column_names(W.shape[1], act_dim)
    if bot_column_names(sum(1 for b in blocks if b == "bot winrate")) is None:
        print("NOTE: bot column count does not match the file set (zero-variance "
              "columns were dropped); bot columns shown by index.")

    F = pp = None
    if not args.quick:
        import utils.rating.residual_axes as ra
        print("loading features (use --quick to skip) ...")
        feats, _rk, _pa, _ad = ra.build_axis_features()
        tids = sorted(feats)
        F = (np.stack([feats[t] for t in tids]) - ax["feat_mu"]) / ax["feat_sd"]
        pp = pd.read_csv(os.path.join(EMBED_DIR, "per_problem.csv"))
        pos = {t: i for i, t in enumerate(pp["tsumego_id"].astype(str))}
        keep = [i for i, t in enumerate(tids) if t in pos]
        F = F[keep]
        rowpos = [pos[tids[i]] for i in keep]
    else:
        p = os.path.join(EMBED_DIR, "per_problem.csv")
        pp = pd.read_csv(p) if os.path.exists(p) else None

    axes = [args.axis] if args.axis is not None else list(range(W.shape[0]))
    for k in axes:
        mid, tail = describe_label(labels, k)
        print("\n" + "=" * 78)
        print(f"AXIS {k}   motif {mid}   labels {tail}")
        print("=" * 78)
        report_weights(k, W[k], names, blocks, args.top)
        if F is not None and f"embed_{k}" in pp.columns:
            y = pp[f"embed_{k}"].to_numpy()[rowpos]
            report_tails(k, W[k], names, blocks, F, y, args.tail_frac, args.top)
        if pp is not None and f"embed_{k}" in pp.columns:
            report_overlap(k, pp, labels, W.shape[0])
            report_identity(k, pp, f"embed_{k}")

    if pp is not None:
        report_matrix(pp, labels, W.shape[0])


if __name__ == "__main__":
    main()
