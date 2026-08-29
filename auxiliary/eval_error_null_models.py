#!/usr/bin/env python3
"""Which wrong variation does a user pick, given that they erred?

The client's motif TENDENCY reading (pwa/js/motif-stats.js) is a sign test of the
picked variation's axis loading against the correct one's, with a FIXED 50% null.
That null assumes an erring user is equally likely to land on either side of the
answer on every axis -- which is only true if the problem OFFERS a balanced menu
of wrong moves. This script measures the alternative: a per-problem null built
from what the decision node actually offers, optionally weighted by how good each
wrong move is.

Phase 1 asks the empirical question underneath it: given an error on a problem,
how well does each candidate null predict WHICH wrong variation was picked?
Phase 2 carries the winner through to the directional statistic the client uses.

Nothing here writes to the repo. Run:

    ./.venv/bin/python3 -m auxiliary.eval_error_null_models
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize_scalar, minimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.katago.coords import sgf_to_gtp, gtp_to_xy  # noqa: E402

DB_PATH = os.path.join(PROJECT_ROOT, "db", "d1.sqlite")
TSUMEGO_DIR = os.path.join(PROJECT_ROOT, "tsumegos")
PER_CELL_DIR = os.path.join(PROJECT_ROOT, "axes", "per_cell")
LABELS_JSON = os.path.join(PROJECT_ROOT, "axes", "labels.json")

N_BOOTSTRAP = 2000
RNG_SEED = 0
# Winrate floor for the multiplicative models: a variation at a reported 0.0%
# still gets clicked, and log(0) is not a probability model.
EPS_WR = 1e-3


# --------------------------------------------------------------------------
# SGF: decision node children, read straight off the raw text.
#
# sgfmill parses these files correctly but costs ~10 ms each and we need ~34k of
# them; the scanner below is ~100x cheaper and is CHECKED against sgfmill on a
# random sample every run (see verify_parser), so the speed is not bought with
# trust.
# --------------------------------------------------------------------------

_SZ_RE = re.compile(r"SZ\[(\d+)\]")
_PROP_RE = re.compile(r"([A-Z]{1,2})\[((?:[^\]\\]|\\.)*)\]")


def _skip_brackets(text, i):
    """Given text[i] == '[', return the index just past the matching ']'."""
    i += 1
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "]":
            return i + 1
        i += 1
    return n


def decision_children(text):
    """Raw SGF text -> (size, [child, ...]) for the FIRST branch point.

    Each child is a dict with the first node's move (SGF coordinate string or
    None for a pass / moveless node), its colour, and the winrate/score parsed
    off its comment. Children are returned in file order, so children[0] is the
    answer the generator promoted.
    """
    m = _SZ_RE.search(text)
    size = int(m.group(1)) if m else None

    # Walk to the first '(' that opens INSIDE the root paren: that is the first
    # variation of the decision node. Every '(' at that depth is one of its
    # siblings -- depth cannot return to 1 until the last of them closes.
    depth = 0
    i = 0
    n = len(text)
    starts = []
    while i < n:
        c = text[i]
        if c == "[":
            i = _skip_brackets(text, i)
            continue
        if c == "(":
            depth += 1
            if depth == 2:
                starts.append(i)
        elif c == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1

    children = []
    for s in starts:
        children.append(_first_node(text, s + 1))
    return size, children


def _first_node(text, i):
    """Parse the first node of a variation beginning at index i (just past '(')."""
    n = len(text)
    while i < n and text[i] in " \t\r\n":
        i += 1
    if i < n and text[i] == ";":
        i += 1
    start = i
    while i < n:
        c = text[i]
        if c == "[":
            i = _skip_brackets(text, i)
            continue
        if c in ";()":
            break
        i += 1
    seg = text[start:i]

    move = None
    colour = None
    comment = None
    for prop, val in _PROP_RE.findall(seg):
        if prop == "B" or prop == "W":
            colour = prop
            move = val
        elif prop == "C":
            comment = val
    winrate = score = None
    if comment:
        # UNANCHORED, like every other parser in the repo (parse_eval_comment in
        # utils/katago/add_variations.py): a comment may carry text before the
        # fields, and the third "/<visits>" field simply falls outside the match.
        mm = re.search(r"(-?[\d.]+)%/(-?[\d.]+)", comment)
        if mm:
            winrate = float(mm.group(1))
            score = float(mm.group(2))
    is_pass = (move is None) or (move == "") or (move == "tt")
    return {"move": move, "colour": colour, "is_pass": is_pass,
            "winrate": winrate, "score": score}


def verify_parser(paths, limit=300):
    """Cross-check the fast scanner against sgfmill on a random sample."""
    import sgfmill.sgf
    from utils.katago.sgf_parsing import parse_tsumego_sgf
    rng = np.random.default_rng(RNG_SEED)
    sample = [paths[i] for i in rng.choice(len(paths), size=min(limit, len(paths)),
                                           replace=False)]
    bad = 0
    for p in sample:
        raw = open(p, "rb").read()
        size, kids = decision_children(raw.decode("utf-8", "replace"))
        game = sgfmill.sgf.Sgf_game.from_bytes(raw)
        _, nodes = parse_tsumego_sgf(game)
        if len(nodes) != len(kids):
            bad += 1
            continue
        for node, kid in zip(nodes, kids):
            try:
                colour, mv = node.get_raw_move()
            except Exception:
                colour, mv = None, None
            ref_move = mv.decode() if mv is not None else None
            ref_pass = ref_move in (None, "", "tt")
            if bool(ref_pass) != bool(kid["is_pass"]):
                bad += 1
                break
            if not ref_pass and ref_move != kid["move"]:
                bad += 1
                break
            if colour and colour.upper() != (kid["colour"] or ""):
                bad += 1
                break
    return len(sample), bad


# --------------------------------------------------------------------------
# Problem table
# --------------------------------------------------------------------------

def _load_problem(args):
    size, tid = args
    path = os.path.join(TSUMEGO_DIR, str(size), tid + ".sgf")
    try:
        with open(path, "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return (size, tid, None)
    sz, kids = decision_children(text)
    if not kids or sz != size:
        return (size, tid, None)
    correct = kids[0]
    # Mover-perspective: the comments are BLACK-perspective throughout
    # (bots/analysis.cfg sets reportAnalysisWinratesAs = BLACK), so a White-to-play
    # decision node has to be flipped before "higher is better" means anything.
    to_play = correct["colour"] or (kids[1]["colour"] if len(kids) > 1 else "B")
    black = (to_play == "B")

    def mover(kid):
        if kid["winrate"] is None:
            return None, None
        w = kid["winrate"] / 100.0
        s = kid["score"]
        return (w, s) if black else (1.0 - w, -s)

    cw, cs = mover(correct)
    # Which child the client's evaluation rule would call the answer -- a
    # disagreement with children[0] is a corpus ordering fault, reported not fixed.
    best_i, best_key = None, None
    for i, kid in enumerate(kids):
        mw, ms = mover(kid)
        if mw is None:
            continue
        key = (mw, ms if ms is not None else -1e9)
        if best_key is None or key > best_key:
            best_key, best_i = key, i

    wrong = []
    seen = set()
    for kid in kids[1:]:
        if kid["is_pass"]:
            continue
        mw, ms = mover(kid)
        if mw is None:
            continue
        try:
            gtp = sgf_to_gtp(kid["move"], size)
        except Exception:
            continue
        # get_child_move reuses an existing child, so a repeat here would be a
        # corpus fault; keep the first so a move maps to exactly one row.
        if gtp in seen:
            continue
        seen.add(gtp)
        wrong.append({"gtp": gtp, "w": mw, "s": ms})

    correct_gtp = None
    if not correct["is_pass"]:
        try:
            correct_gtp = sgf_to_gtp(correct["move"], size)
        except Exception:
            correct_gtp = None

    return (size, tid, {
        "n_children": len(kids),
        "n_pass": sum(1 for k in kids[1:] if k["is_pass"]),
        "correct_gtp": correct_gtp,
        "correct_w": cw, "correct_s": cs,
        "eval_best_is_first": (best_i == 0),
        "wrong": wrong,
    })


def load_problems(keys, workers=None):
    workers = workers or max(2, (os.cpu_count() or 4) - 1)
    out = {}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for size, tid, info in ex.map(_load_problem, keys, chunksize=200):
            out[(size, tid)] = info
            done += 1
            if done % 5000 == 0:
                print(f"  parsed {done}/{len(keys)} SGFs "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def group_logsumexp(logits, offsets):
    """logsumexp within contiguous groups given reduceat offsets."""
    mx = np.maximum.reduceat(logits, offsets)
    ex = np.exp(logits - np.repeat(mx, np.diff(np.append(offsets, len(logits)))))
    return mx + np.log(np.add.reduceat(ex, offsets))


class Model:
    """A null over a problem's wrong variations.

    `fn` returns per-row LOG PROBABILITIES already normalised within each
    problem, so a model can be a mixture and not only a softmax over logits.
    `from_logits` wraps the common case.
    """

    def __init__(self, name, n_params, fn, x0, bounds):
        self.name = name
        self.n_params = n_params
        self.fn = fn
        self.x0 = x0
        self.bounds = bounds

    def logp(self, feats, offsets, sizes, params):
        return self.fn(feats, offsets, sizes, params)


def from_logits(logits_fn):
    def fn(feats, offsets, sizes, params):
        logits = logits_fn(feats, params)
        return logits - np.repeat(group_logsumexp(logits, offsets), sizes)
    return fn


def build_models():
    logw = lambda f: np.log(np.maximum(f["w"], EPS_WR))     # noqa: E731
    logr = lambda f: np.log(f["r"] + 1.0)                   # noqa: E731

    def mix_with_uniform(head_logits, offsets, sizes, mix):
        """(1 - mix) * softmax(head) + mix * uniform, per problem.

        Structural, not cosmetic: the decision node's children are KataGo's own
        search-ranked refutations FIRST (generate_variations_directly_to_sgf walks
        moveInfos, which come back visit-sorted) and then every remaining legal
        move in board-scan order (add_shallow_variations). So the head of the list
        is ranked and the tail is arbitrary, and a null has to be able to say so.
        """
        mix = float(np.clip(mix, 1e-6, 1 - 1e-6))
        lse = np.repeat(group_logsumexp(head_logits, offsets), sizes)
        p = (1 - mix) * np.exp(head_logits - lse) + mix / np.repeat(sizes, sizes)
        return np.log(np.maximum(p, 1e-300))

    return [
        Model("uniform", 0, from_logits(lambda f, p: np.zeros(len(f["w"]))), [], []),
        Model("winrate-linear", 0, from_logits(lambda f, p: logw(f)), [], []),
        Model("winrate-power(alpha)", 1, from_logits(lambda f, p: p[0] * logw(f)),
              [1.0], [(0.0, 20.0)]),
        Model("softmax-winrate(temperature)", 1,
              from_logits(lambda f, p: f["w"] / p[0]), [0.3], [(0.02, 50.0)]),
        Model("softmax-score(temperature)", 1,
              from_logits(lambda f, p: f["s"] / p[0]), [5.0], [(0.05, 500.0)]),
        Model("softmax-winrate+score(2 temperatures)", 2,
              from_logits(lambda f, p: f["w"] / p[0] + f["s"] / p[1]),
              [0.3, 5.0], [(0.02, 50.0), (0.05, 500.0)]),
        Model("variation-order zipf(beta)", 1,
              from_logits(lambda f, p: -p[0] * logr(f)), [1.0], [(0.0, 5.0)]),
        Model("variation-order exponential(tau)", 1,
              from_logits(lambda f, p: -(f["r"]) / p[0]), [8.0], [(0.5, 500.0)]),
        Model("variation-order zipf + uniform tail", 2,
              lambda f, o, s, p: mix_with_uniform(-p[0] * logr(f), o, s, p[1]),
              [1.0, 0.2], [(0.0, 5.0), (1e-4, 0.999)]),
        Model("order zipf + winrate, uniform tail", 3,
              lambda f, o, s, p: mix_with_uniform(
                  -p[0] * logr(f) + p[2] * logw(f), o, s, p[1]),
              [1.0, 0.2, 0.5], [(0.0, 5.0), (1e-4, 0.999), (0.0, 20.0)]),
    ]


def fit_model(model, feats, offsets, sizes, counts):
    """MLE of the model's parameters on the given (training) rows."""
    total = counts.sum()
    if model.n_params == 0:
        return []

    def nll(params):
        lp = model.logp(feats, offsets, sizes, np.atleast_1d(params))
        return -float((counts * lp).sum()) / total

    if model.n_params == 1:
        lo, hi = model.bounds[0]
        res = minimize_scalar(lambda t: nll([t]), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-4})
        return [float(res.x)]
    res = minimize(nll, np.array(model.x0), method="L-BFGS-B", bounds=model.bounds)
    return [float(v) for v in res.x]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Error-null model evaluation: which wrong variation gets picked?")
    print("=" * 78)

    # ---- reports -----------------------------------------------------------
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT problem, size, move, user_id FROM tsumego_reports "
        "WHERE success = 0 AND is_repeat = 0 AND move IS NOT NULL").fetchall()
    n_wrong_total = con.execute(
        "SELECT COUNT(*) FROM tsumego_reports "
        "WHERE success = 0 AND is_repeat = 0").fetchone()[0]
    con.close()
    print(f"\nwrong, non-repeat attempts                 : {n_wrong_total:,}")
    print(f"  ... with a recorded move                 : {len(rows):,} "
          f"({100.0 * len(rows) / n_wrong_total:.1f}%)")

    keys = sorted({(int(r[1]), r[0]) for r in rows})
    print(f"distinct problems referenced               : {len(keys):,}")

    # ---- SGFs --------------------------------------------------------------
    paths = [os.path.join(TSUMEGO_DIR, str(s), t + ".sgf") for s, t in keys]
    paths = [p for p in paths if os.path.exists(p)]
    n_sample, n_bad = verify_parser(paths)
    print(f"parser cross-check vs sgfmill              : "
          f"{n_sample - n_bad}/{n_sample} decision nodes identical")
    if n_bad:
        print("  !! fast SGF scanner disagrees with sgfmill -- results below are suspect")

    print("parsing decision nodes ...", flush=True)
    problems = load_problems(keys)
    n_missing = sum(1 for v in problems.values() if v is None)
    print(f"problems missing / unparseable on disk     : {n_missing:,}")

    ordering_bad = sum(1 for v in problems.values()
                       if v is not None and not v["eval_best_is_first"])
    n_ok = sum(1 for v in problems.values() if v is not None)
    print(f"problems where children[0] is NOT the      : {ordering_bad:,} / {n_ok:,} "
          f"({100.0 * ordering_bad / max(n_ok, 1):.3f}%)")
    print("  evaluation-best child (corpus ordering)")

    # ---- match reported moves to variations --------------------------------
    n_no_problem = 0
    n_hit_correct = 0
    n_hit_wrong = 0
    n_nomatch = 0
    n_trivial = 0          # matched, but the problem offers only ONE wrong variation
    picks = defaultdict(lambda: defaultdict(int))   # key -> gtp -> count
    nvar_hist = defaultdict(int)
    nvar_by_size = defaultdict(list)
    nomatch_by_problem = defaultdict(int)
    matched = []                                    # (user_id, key, gtp)

    for problem, size, move, user_id in rows:
        key = (int(size), problem)
        info = problems.get(key)
        if info is None:
            n_no_problem += 1
            continue
        if move == info["correct_gtp"]:
            n_hit_correct += 1
            continue
        gtps = {v["gtp"] for v in info["wrong"]}
        if move not in gtps:
            n_nomatch += 1
            nomatch_by_problem[key] += 1
            continue
        m = len(info["wrong"])
        nvar_hist[m] += 1
        nvar_by_size[int(size)].append(m)
        if m < 2:
            n_trivial += 1
            continue
        n_hit_wrong += 1
        picks[key][move] += 1
        matched.append((user_id, key, move))

    n_considered = len(rows) - n_no_problem
    print("\n--- match rates (of wrong attempts with a move) ---")
    print(f"problem not on disk / unparseable          : {n_no_problem:,}")
    print(f"move == the CORRECT variation              : {n_hit_correct:,} "
          f"({100.0 * n_hit_correct / max(n_considered, 1):.2f}%)  "
          f"[expected ~0 on success=0]")
    print(f"move matched a WRONG variation             : "
          f"{n_hit_wrong + n_trivial:,} "
          f"({100.0 * (n_hit_wrong + n_trivial) / max(n_considered, 1):.2f}%)")
    print(f"move matched NOTHING at the decision node  : {n_nomatch:,} "
          f"({100.0 * n_nomatch / max(n_considered, 1):.2f}%)")
    print(f"  ... spread over problems                 : {len(nomatch_by_problem):,}")
    print(f"matched but only 1 wrong variation offered : {n_trivial:,} "
          f"({100.0 * n_trivial / max(n_hit_wrong + n_trivial, 1):.2f}% of matches; "
          f"every model is trivially exact there)")
    print(f"SCORED picks (>= 2 wrong variations)       : {n_hit_wrong:,} "
          f"over {len(picks):,} problems")

    tot = sum(nvar_hist.values())
    print("\nwrong variations offered, per matched attempt:")
    for m in sorted(nvar_hist):
        if m > 12:
            break
        print(f"  {m:>3} wrong variation(s): {nvar_hist[m]:>7,} "
              f"({100.0 * nvar_hist[m] / tot:5.2f}%)")
    big = sum(v for k, v in nvar_hist.items() if k > 12)
    if big:
        print(f"  >12 wrong variations: {big:>7,} ({100.0 * big / tot:5.2f}%)")
    ks = sorted(nvar_hist)
    med = np.median(np.repeat(np.array(ks), [nvar_hist[k] for k in ks]))
    print(f"  median offered per attempt: {med:.0f}")
    for size in sorted(nvar_by_size):
        arr = np.array(nvar_by_size[size])
        print(f"  size {size}: median {np.median(arr):.0f}, "
              f"p10 {np.percentile(arr, 10):.0f}, p90 {np.percentile(arr, 90):.0f}, "
              f"max {arr.max()} (board has {size * size} cells)")

    # ---- flat arrays -------------------------------------------------------
    prob_keys = sorted(picks)
    offsets = []
    pidx = []
    w_all, s_all, r_all, cnt_all, gtp_all = [], [], [], [], []
    cursor = 0
    for pi, key in enumerate(prob_keys):
        info = problems[key]
        offsets.append(cursor)
        for r, v in enumerate(info["wrong"]):
            w_all.append(v["w"])
            s_all.append(v["s"] if v["s"] is not None else 0.0)
            r_all.append(r)          # position in the SGF's own variation list
            cnt_all.append(picks[key].get(v["gtp"], 0))
            gtp_all.append(v["gtp"])
            pidx.append(pi)
            cursor += 1
    offsets = np.array(offsets, dtype=np.int64)
    feats = {"w": np.array(w_all), "s": np.array(s_all),
             "r": np.array(r_all, dtype=np.float64)}
    counts = np.array(cnt_all, dtype=np.float64)
    pidx = np.array(pidx, dtype=np.int64)
    sizes = np.diff(np.append(offsets, len(counts)))

    # ---- paired two-fold split by problem id hash --------------------------
    fold = np.array([
        int(hashlib.sha256(f"{k[0]}/{k[1]}".encode()).hexdigest()[:8], 16) & 1
        for k in prob_keys], dtype=np.int64)
    fold_row = fold[pidx]
    print(f"\npaired split by problem-id hash            : "
          f"fold0 {int((fold == 0).sum()):,} problems / "
          f"fold1 {int((fold == 1).sum()):,} problems")

    models = build_models()
    heldout_logp = {}
    fitted = {}
    for model in models:
        lp = np.zeros(len(counts))
        params_seen = []
        for train_f in (0, 1):
            tr = fold_row == train_f
            te = ~tr
            tr_off, tr_sizes = _subgroup(tr, offsets, sizes)
            sub_feats = {k: v[tr] for k, v in feats.items()}
            params = fit_model(model, sub_feats, tr_off, tr_sizes, counts[tr])
            params_seen.append(params)
            te_off, te_sizes = _subgroup(te, offsets, sizes)
            sub_feats_te = {k: v[te] for k, v in feats.items()}
            lp[te] = model.logp(sub_feats_te, te_off, te_sizes, np.atleast_1d(params))
        heldout_logp[model.name] = lp
        fitted[model.name] = params_seen

    # ---- NLL table ---------------------------------------------------------
    npick = counts.sum()
    nll = {m.name: -float((counts * heldout_logp[m.name]).sum()) / npick
           for m in models}
    base = nll["uniform"]

    rng = np.random.default_rng(RNG_SEED)
    nprob = len(prob_keys)
    per_problem_pick = np.bincount(pidx, weights=counts, minlength=nprob)
    per_problem_ll = {m.name: np.bincount(pidx, weights=counts * heldout_logp[m.name],
                                          minlength=nprob) for m in models}
    boot = defaultdict(list)
    for _ in range(N_BOOTSTRAP):
        mult = rng.multinomial(nprob, np.full(nprob, 1.0 / nprob))
        denom = float((mult * per_problem_pick).sum())
        if denom <= 0:
            continue
        b_nll = {m.name: -float((mult * per_problem_ll[m.name]).sum()) / denom
                 for m in models}
        for m in models:
            boot[m.name].append(b_nll[m.name] - b_nll["uniform"])

    print("\n" + "=" * 78)
    print("PHASE 1 -- held-out mean negative log-likelihood per pick")
    print(f"({int(npick):,} picks, {nprob:,} problems, cross-fitted over the two folds)")
    print("=" * 78)
    print(f"{'model':<40} {'NLL':>8} {'delta':>8} {'95% interval':>20}  params")
    print("-" * 78)
    for m in models:
        d = np.array(boot[m.name])
        lo, hi = (np.percentile(d, 2.5), np.percentile(d, 97.5)) if len(d) else (0, 0)
        ps = fitted[m.name]
        pstr = "" if m.n_params == 0 else " | ".join(
            "[" + ", ".join(f"{v:.3f}" for v in p) + "]" for p in ps)
        print(f"{m.name:<40} {nll[m.name]:8.4f} {nll[m.name] - base:+8.4f} "
              f"[{lo:+7.4f}, {hi:+7.4f}]  {pstr}")
    print("\n(delta < 0 = better than the uniform-over-available null; the interval is a\n"
          " paired bootstrap over PROBLEMS, same resample for every model.)")

    best = min(models, key=lambda m: nll[m.name])
    print(f"\nbest model: {best.name}")

    # A reference point, though a degenerate one: with ~1.3 picks per problem the
    # per-problem empirical frequency is memorization, not an attainable ceiling.
    perfect = _oracle_nll(counts, pidx, nprob)
    print(f"in-sample ORACLE (per-problem empirical pick frequencies): {perfect:.4f}")
    print(f"  -- degenerate: {npick / nprob:.2f} picks per problem, so this is\n"
          "     memorization rather than an attainable ceiling.")

    # Discrimination: does the model put the picked move near the top of the menu?
    print("\n--- rank of the picked variation under each model (held-out) ---")
    print(f"{'model':<40} {'top-1':>8} {'top-3':>8} {'top-5':>8} {'mean pct':>9}")
    print("-" * 78)
    for m in models:
        lp = heldout_logp[m.name]
        # rank of each row inside its problem, 0 = highest predicted probability
        rank = np.zeros(len(lp), dtype=np.int64)
        for lo, sz in zip(offsets, sizes):
            seg = lp[lo:lo + sz]
            rank[lo:lo + sz] = np.argsort(np.argsort(-seg, kind="stable"),
                                          kind="stable")
        pct = rank / np.maximum(np.repeat(sizes, sizes) - 1, 1)
        w = counts
        t1 = float((w * (rank < 1)).sum()) / npick
        t3 = float((w * (rank < 3)).sum()) / npick
        t5 = float((w * (rank < 5)).sum()) / npick
        mp = float((w * pct).sum()) / npick
        print(f"{m.name:<40} {t1:>8.3f} {t3:>8.3f} {t5:>8.3f} {mp:>9.3f}")
    print("(mean pct = mean normalised rank of the move the user actually played;\n"
          " 0.5 is chance, 0 would mean the model always names the pick first.)")

    # ---- calibration -------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"CALIBRATION of {best.name} (held-out)")
    print("=" * 78)
    p_best = np.exp(heldout_logp[best.name])
    trials = per_problem_pick[pidx]          # attempts on this problem
    edges = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20,
                      0.30, 0.45, 0.65, 1.01])
    print(f"{'predicted bin':>16} {'rows':>9} {'attempts':>10} "
          f"{'predicted':>10} {'observed':>10} {'ratio':>7}")
    print("-" * 70)
    for i in range(len(edges) - 1):
        sel = (p_best >= edges[i]) & (p_best < edges[i + 1]) & (trials > 0)
        if not sel.any():
            continue
        tr = trials[sel].sum()
        pred = float((trials[sel] * p_best[sel]).sum()) / tr
        obs = float(counts[sel].sum()) / tr
        print(f"  [{edges[i]:.2f}, {edges[i+1]:.2f})".rjust(16) +
              f" {int(sel.sum()):>9,} {int(tr):>10,} "
              f"{pred:>10.4f} {obs:>10.4f} {obs / max(pred, 1e-9):>7.2f}")

    # ---- Phase 2 -----------------------------------------------------------
    phase2(prob_keys, problems, heldout_logp[best.name], best.name,
           offsets, sizes, counts, gtp_all, matched)


def _subgroup(mask, offsets, sizes):
    """Group offsets/sizes restricted to a boolean row mask (groups stay contiguous)."""
    new_sizes = np.add.reduceat(mask.astype(np.int64), offsets)
    keep = new_sizes > 0
    new_sizes = new_sizes[keep]
    new_off = np.zeros(len(new_sizes), dtype=np.int64)
    if len(new_sizes):
        new_off[1:] = np.cumsum(new_sizes)[:-1]
    return new_off, new_sizes


def _oracle_nll(counts, pidx, nprob):
    tot = np.bincount(pidx, weights=counts, minlength=nprob)
    p = counts / np.maximum(tot[pidx], 1e-9)
    nz = counts > 0
    return -float((counts[nz] * np.log(p[nz])).sum()) / counts.sum()


def phase2(prob_keys, problems, heldout_logp, best_name,
           offsets, sizes, counts, gtp_all, matched):
    print("\n" + "=" * 78)
    print("PHASE 2 -- directional (toward-the-motif) share per axis")
    print("=" * 78)

    labels = json.load(open(LABELS_JSON))["components"]
    K = len(labels)
    nprob = len(prob_keys)
    p_null = np.exp(heldout_logp)

    n_rows = len(counts)
    row_toward = np.zeros((n_rows, K), dtype=bool)
    row_nontie = np.zeros((n_rows, K), dtype=bool)
    # Per-problem null probability of erring TOWARD the motif, conditioned on a
    # non-tie exactly as the client's sign test conditions.
    prob_p_w = np.full((nprob, K), np.nan)
    prob_p_u = np.full((nprob, K), np.nan)

    n_no_npz = 0
    n_unmapped = 0
    n_used = 0

    t0 = time.time()
    for pi, key in enumerate(prob_keys):
        if pi % 8000 == 0 and pi:
            print(f"  {pi}/{nprob} problems ({time.time() - t0:.0f}s)", flush=True)
        size, tid = key
        path = os.path.join(PER_CELL_DIR, str(size), tid + ".npz")
        try:
            with np.load(path, allow_pickle=False) as d:
                emb = np.asarray(d["embeddings"], dtype=np.float64)
                xys = np.asarray(d["xys"], dtype=np.int32)
        except OSError:
            n_no_npz += 1
            continue
        if emb.shape[1] != K:
            n_no_npz += 1
            continue
        lut = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}

        info = problems[key]
        cxy = gtp_to_xy(info["correct_gtp"]) if info["correct_gtp"] else None
        if cxy is None or cxy not in lut:
            n_unmapped += 1
            continue
        c_emb = emb[lut[cxy]]

        lo, hi = int(offsets[pi]), int(offsets[pi] + sizes[pi])
        rows = np.arange(lo, hi)
        v_emb = np.full((len(rows), K), np.nan)
        ok = np.zeros(len(rows), dtype=bool)
        for j, r in enumerate(rows):
            idx = lut.get(gtp_to_xy(gtp_all[r]))
            if idx is not None:
                v_emb[j] = emb[idx]
                ok[j] = True
        if not ok.any():
            n_unmapped += 1
            continue
        n_used += 1

        # d = correct - picked; d < 0 means the pick loaded HIGHER = "toward".
        with np.errstate(invalid="ignore"):
            d = c_emb[None, :] - v_emb
            toward = (d < 0) & ok[:, None]
            away = (d > 0) & ok[:, None]
        nontie = toward | away
        row_toward[lo:hi] = toward
        row_nontie[lo:hi] = nontie

        for weights, dest in ((p_null[rows], prob_p_w),
                              (ok / max(ok.sum(), 1), prob_p_u)):
            mass_nt = (weights[:, None] * nontie).sum(axis=0)
            mass_tw = (weights[:, None] * toward).sum(axis=0)
            good = mass_nt > 1e-12
            dest[pi, good] = mass_tw[good] / mass_nt[good]

    print(f"\nproblems with a usable per-cell embedding  : {n_used:,} of {nprob:,}")
    print(f"  missing / wrong-K npz                    : {n_no_npz:,}")
    print(f"  correct cell or all variations unmapped  : {n_unmapped:,}")

    # ---- aggregate, weighted by the SAME per-axis error counts -------------
    row_pidx = np.repeat(np.arange(nprob), sizes)
    per_prob_tw = np.zeros((nprob, K))
    per_prob_n = np.zeros((nprob, K))
    for k in range(K):
        per_prob_tw[:, k] = np.bincount(row_pidx, weights=counts * row_toward[:, k],
                                        minlength=nprob)
        per_prob_n[:, k] = np.bincount(row_pidx, weights=counts * row_nontie[:, k],
                                       minlength=nprob)
    ok_w = np.isfinite(prob_p_w)
    ok_u = np.isfinite(prob_p_u)
    mass_w = np.where(ok_w, per_prob_n * np.nan_to_num(prob_p_w), 0.0)
    mass_u = np.where(ok_u, per_prob_n * np.nan_to_num(prob_p_u), 0.0)
    n_w = np.where(ok_w, per_prob_n, 0.0)
    n_u = np.where(ok_u, per_prob_n, 0.0)

    obs = per_prob_tw.sum(0) / np.maximum(per_prob_n.sum(0), 1)
    pw = mass_w.sum(0) / np.maximum(n_w.sum(0), 1)
    pu = mass_u.sum(0) / np.maximum(n_u.sum(0), 1)

    # bootstrap over problems, paired across the three quantities
    rng = np.random.default_rng(RNG_SEED)
    b_obs, b_pw, b_res = [], [], []
    for _ in range(500):
        mult = rng.multinomial(nprob, np.full(nprob, 1.0 / nprob))[:, None]
        o = (mult * per_prob_tw).sum(0) / np.maximum((mult * per_prob_n).sum(0), 1)
        w = (mult * mass_w).sum(0) / np.maximum((mult * n_w).sum(0), 1)
        b_obs.append(o)
        b_pw.append(w)
        b_res.append(o - w)
    b_obs = np.array(b_obs)
    b_pw = np.array(b_pw)
    b_res = np.array(b_res)

    print(f"\n{'axis':<16} {'observed':>19} {'null(weighted)':>19} "
          f"{'null(uniform)':>14} {'client':>7} {'residual (obs - null)':>25}")
    print("-" * 106)
    for k in range(K):
        lo_o, hi_o = np.percentile(b_obs[:, k], [2.5, 97.5])
        lo_w, hi_w = np.percentile(b_pw[:, k], [2.5, 97.5])
        lo_r, hi_r = np.percentile(b_res[:, k], [2.5, 97.5])
        print(f"{labels[k]:<16} {obs[k]:>7.4f} [{lo_o:.4f},{hi_o:.4f}] "
              f"{pw[k]:>7.4f} [{lo_w:.4f},{hi_w:.4f}] {pu[k]:>14.4f} "
              f"{0.5:>7.2f} {obs[k] - pw[k]:>+9.4f} [{lo_r:+.4f},{hi_r:+.4f}]")
    print("\nA residual clear of zero is a real directional tendency; a residual\n"
          "straddling zero means the whole deviation from the client's 50% null is\n"
          "explained by WHICH wrong moves the problem offers.")
    print(f"\nnon-tie error counts per axis: "
          f"{', '.join(f'{labels[k]}={int(per_prob_n.sum(0)[k]):,}' for k in range(K))}")

    # ---- what it would change for a user ----------------------------------
    print("\n" + "=" * 78)
    print("PER-USER IMPACT -- would the reading the client prints change?")
    print("=" * 78)
    row_of = {}
    for pi, key in enumerate(prob_keys):
        lo, hi = int(offsets[pi]), int(offsets[pi] + sizes[pi])
        for r in range(lo, hi):
            row_of[(key, gtp_all[r])] = r

    users = {}
    for user_id, key, gtp in matched:
        r = row_of.get((key, gtp))
        if r is None:
            continue
        pi = int(row_pidx[r])
        acc = users.get(user_id)
        if acc is None:
            acc = users[user_id] = {"n": np.zeros(K), "tw": np.zeros(K),
                                    "p": np.zeros(K), "v": np.zeros(K)}
        nt = row_nontie[r]
        p = prob_p_w[pi]
        use = nt & np.isfinite(p)
        acc["n"][use] += 1
        acc["tw"][use] += row_toward[r][use]
        acc["p"][use] += p[use]
        acc["v"][use] += p[use] * (1 - p[use])

    MOTIF_Z = 1.0
    MIN_ERRORS = 10
    n_users = 0
    agree = np.zeros(K)
    flag_old = np.zeros(K)
    flag_new = np.zeros(K)
    flip_sign = np.zeros(K)
    lost = np.zeros(K)
    gained = np.zeros(K)
    tested = np.zeros(K)
    for acc in users.values():
        ok = acc["n"] >= MIN_ERRORS
        if not ok.any():
            continue
        n_users += 1
        share = np.where(ok, acc["tw"] / np.maximum(acc["n"], 1), np.nan)
        z_old = (share - 0.5) / np.sqrt(0.25 / np.maximum(acc["n"], 1))
        z_new = (acc["tw"] - acc["p"]) / np.sqrt(np.maximum(acc["v"], 1e-9))
        fo = ok & (np.abs(z_old) > MOTIF_Z)
        fn = ok & (np.abs(z_new) > MOTIF_Z)
        tested += ok
        flag_old += fo
        flag_new += fn
        agree += ok & (fo == fn) & (~fo | (np.sign(z_old) == np.sign(z_new)))
        flip_sign += fo & fn & (np.sign(z_old) != np.sign(z_new))
        lost += fo & ~fn
        gained += fn & ~fo

    print(f"users with >= {MIN_ERRORS} scored errors on an axis : {n_users:,}")
    print(f"\n{'axis':<16} {'tested':>8} {'flagged now':>13} {'flagged w/ null':>16} "
          f"{'lost':>7} {'gained':>7} {'sign flip':>10}")
    print("-" * 82)
    for k in range(K):
        t = max(tested[k], 1)
        print(f"{labels[k]:<16} {int(tested[k]):>8,} "
              f"{int(flag_old[k]):>6,} ({100*flag_old[k]/t:4.1f}%) "
              f"{int(flag_new[k]):>7,} ({100*flag_new[k]/t:4.1f}%) "
              f"{int(lost[k]):>7,} {int(gained[k]):>7,} {int(flip_sign[k]):>10,}")
    print("\n'flagged now' = the shipped 50% sign test; 'flagged w/ null' = the same\n"
          "count against the per-problem availability null (Poisson-binomial z).")


if __name__ == "__main__":
    main()
