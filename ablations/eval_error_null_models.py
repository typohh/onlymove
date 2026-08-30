#!/usr/bin/env python3
"""Which wrong variation does a user pick, given that they erred?

The client's motif TENDENCY reading (pwa/js/motif-stats.js) is a sign test of the
picked variation's axis loading against the correct one's, against the UNIFORM
AVAILABILITY NULL: uniform over the alternatives the board offered, per problem,
parameter-free. That null assumes an erring user is equally likely to land on any
alternative -- which ignores that a decision node's wrong moves are not equally
tempting. This script measures the alternative: the same per-problem null WEIGHTED
by how good (and how deeply searched) each wrong move is.

Phase 1 asks the empirical question underneath it: given an error on a problem,
how well does each candidate null predict WHICH wrong variation was picked?
Phase 2 carries a fixed panel of them through to the directional statistic the
client computes, and to the per-user flags it prints -- observed against the
SHIPPED uniform-availability null and against the weighted ones, side by side.
(The fixed-50% null the client used before the availability null shipped
survives only as a footnote here.)

Phase 2b is the one that ADJUDICATES. Phase 1 scores a null on a question the
client never asks -- which of ~58 wrong moves gets clicked -- while the tendency
statistic only ever consumes each problem's toward-probability p, one number per
axis, against a binary outcome. So 2b scores exactly that: the per-error
DIRECTIONAL log-loss (and Brier) of toward/away against each null's p, per axis,
paired over the same errors. A null can win phase 1 and lose here, because
concentrating the pick distribution on the right move says nothing about
splitting the alternatives correctly ABOVE and BELOW the answer.

Reliability is deliberately NOT the criterion anywhere in this file. Its
companion `tendency_null_variants` shows why: the retired fixed-50% null scores
as by far the most split-half-reliable arm, on all seven axes and both repeats,
precisely because it RETAINS the per-problem menu confound -- impersonal
structure that is stable within a user replicates across halves like a real
lean. Reliability therefore rises with retained confound rather than with
measurement quality, and that argument does not stop at the fixed null: it
applies to any pair of nulls differing in how much problem structure they divide
out. What is left is validity, which is what 2b measures.

Nothing here writes to the repo. Run:

    ./.venv/bin/python3 -m ablations.eval_error_null_models

Phase 2 needs a per-cell axis projection. `--embeddings shipped` reads the
production `axes/per_cell/` store; the DEFAULT `--embeddings actbot` projects the
variation cells on the fly through a named dev checkpoint (`--model`,
`--features`), which is the trunk input the reading-level measurements now stand
on. That default names a `dev_runs/` checkpoint, which the directory's contract
otherwise reserves for an explicitly-named comparison arm -- `--embeddings
shipped` is the production-artifact path and stays exact.
"""

import argparse
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

from utils.katago.add_variations import parse_eval_comment, parse_eval_visits  # noqa: E402
from utils.katago.coords import sgf_to_gtp, gtp_to_xy  # noqa: E402

DB_PATH = os.path.join(PROJECT_ROOT, "db", "d1.sqlite")
TSUMEGO_DIR = os.path.join(PROJECT_ROOT, "tsumegos")
PER_CELL_DIR = os.path.join(PROJECT_ROOT, "axes", "per_cell")
LABELS_JSON = os.path.join(PROJECT_ROOT, "axes", "labels.json")
DEFAULT_DEV_MODEL = os.path.join(PROJECT_ROOT, "dev_runs", "actbot_s3.pt")
DEFAULT_DEV_FEATURES = "activations+bot"

N_BOOTSTRAP = 2000
RNG_SEED = 0
# Winrate floor for the multiplicative models: a variation at a reported 0.0%
# still gets clicked, and log(0) is not a probability model.
EPS_WR = 1e-3
# The gate the client flags a tendency reading at (MOTIF_Z / MIN_ERRORS in
# pwa/js/constants.js and pwa/js/motif-stats.js).
MOTIF_Z = 1.0
MIN_ERRORS = 10

# The nulls phase 2 carries, as (column tag, build_models name). The uniform
# availability null -- what ships -- is column 0 always and is not listed here.
# NAMED, not selected by phase 1: phase 2 asks which null predicts the DIRECTION
# of an error, which is a different question from which move gets picked, and a
# panel chosen by the first question could not disagree with it. The three
# parameter-free entries span the signals a decision node offers (how good the
# move is, how far the search looked, where it sits in the list); the fitted one
# is a reference bound, not a shippable candidate.
PHASE2_PANEL = (
    ("winrate-lin", "winrate-linear"),
    ("visits-lin", "visits-linear"),
    ("1/order", "1/order"),
    ("zipf+wr*", "order zipf + winrate, uniform tail"),
)


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
    None for a pass / moveless node), its colour, and the winrate/score/visits
    parsed off its comment. Children are returned in file order, so children[0]
    is the answer the generator promoted.
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
    winrate = score = visits = None
    if comment:
        # UNANCHORED, like every other parser in the repo (parse_eval_comment in
        # utils/katago/add_variations.py): a comment may carry text before the
        # fields, and the third "/<visits>" field simply falls outside the match.
        mm = re.search(r"(-?[\d.]+)%/(-?[\d.]+)", comment)
        if mm:
            winrate = float(mm.group(1))
            score = float(mm.group(2))
            # The third field is the SEARCH DEPTH behind the estimate. Read
            # through the repo's own reader rather than a fourth regex here, so
            # the "no third field means the shallow floor by construction"
            # convention is stated in exactly one place.
            visits = parse_eval_visits(comment)
    is_pass = (move is None) or (move == "") or (move == "tt")
    return {"move": move, "colour": colour, "is_pass": is_pass,
            "winrate": winrate, "score": score, "visits": visits}


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
            # The evaluation comment, all three fields, against the repo's own
            # readers. The third one is what the visits-weighted nulls below run
            # on, so a scanner that captured it differently would silently move
            # every one of their weights.
            try:
                ref_comment = node.get("C")
            except KeyError:
                ref_comment = None
            ref_ws = parse_eval_comment(ref_comment)
            kid_ws = (None if kid["winrate"] is None
                      else (kid["winrate"], kid["score"]))
            if ref_ws != kid_ws:
                bad += 1
                break
            if parse_eval_visits(ref_comment) != (kid["visits"] or 0):
                bad += 1
                break
    return len(sample), bad


# --------------------------------------------------------------------------
# Problem table
# --------------------------------------------------------------------------

def _mover_reader(kids):
    """A (winrate, score) reader in MOVER perspective for a decision node.

    The comments are BLACK-perspective throughout (bots/analysis.cfg sets
    reportAnalysisWinratesAs = BLACK), so a White-to-play decision node has to be
    flipped before "higher is better" means anything.
    """
    to_play = kids[0]["colour"] or (kids[1]["colour"] if len(kids) > 1 else "B")
    black = (to_play == "B")

    def mover(kid):
        if kid["winrate"] is None:
            return None, None
        w = kid["winrate"] / 100.0
        s = kid["score"]
        return (w, s) if black else (1.0 - w, -s)

    return mover


def variation_cells(size, tid):
    """Every non-pass root variation of a problem, in SGF order.

    Each entry is {"gtp", "xy", "w", "s", "v"} with w/s in MOVER perspective and
    v the visit count behind that estimate. cells[0] is the ANSWER -- generation
    pins it first (promote_winning_first_move) and every backend stage reads it
    positionally. A variation whose comment carries no evaluation, and a repeat
    of a cell already listed, are both dropped, so a board point names at most
    one entry. Returns None when the file is missing, unparseable, or declares a
    different board size.

    This is the WEIGHT side of a weighted availability null: the same per-cell
    numbers phase 1 scores its candidate nulls on, exposed so an ablation that
    already holds a per-cell projection (ablations.tendency_null_variants) can
    build the same weights without restating the scanner.
    """
    path = os.path.join(TSUMEGO_DIR, str(size), tid + ".sgf")
    try:
        with open(path, "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    sz, kids = decision_children(text)
    if not kids or sz != size:
        return None
    mover = _mover_reader(kids)
    cells = []
    seen = set()
    for kid in kids:
        if kid["is_pass"]:
            continue
        mw, ms = mover(kid)
        if mw is None:
            continue
        try:
            gtp = sgf_to_gtp(kid["move"], size)
        except Exception:
            continue
        xy = gtp_to_xy(gtp)
        if xy is None or gtp in seen:
            continue
        seen.add(gtp)
        cells.append({"gtp": gtp, "xy": (int(xy[0]), int(xy[1])),
                      "w": mw, "s": ms, "v": float(kid["visits"])})
    return cells


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
    mover = _mover_reader(kids)

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
        wrong.append({"gtp": gtp, "w": mw, "s": ms, "v": float(kid["visits"])})

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
    # Visits are how deeply KataGo searched that move, and the generator spends
    # search where the search wanted to go -- so it is a SALIENCE signal that
    # owes nothing to the move's evaluation. Floored at 1 only because log(0)
    # is not a weight; nothing in the corpus is written below the shallow floor.
    logv = lambda f: np.log(np.maximum(f["v"], 1.0))        # noqa: E731

    def lcb_logits(f, p):
        """A one-parameter lower-confidence-bound proxy on the winrate: a move
        looks good either by being good or by being under-searched, and this is
        the cheapest statement of that trade. c is in winrate units per
        1/sqrt(visit)."""
        return np.log(np.maximum(
            f["w"] - p[0] / np.sqrt(np.maximum(f["v"], 1.0)), EPS_WR))

    def lcb_fixed(f, p):
        """The same bound with NO fitted constant: the binomial standard error of
        the winrate at its own visit count, at a conventional z of 1. Paired with
        lcb(c) on purpose -- the two differ only in whether the radius scale is
        estimated, so the gap between them is what fitting that scale buys."""
        w = np.clip(f["w"], 0.0, 1.0)
        se = np.sqrt(w * (1.0 - w) / np.maximum(f["v"], 1.0))
        return np.log(np.maximum(w - se, EPS_WR))

    def norm(logits, offsets, sizes):
        return logits - np.repeat(group_logsumexp(logits, offsets), sizes)

    def linear_order(f, offsets, sizes, p):
        """w proportional to (alternatives offered - position), i.e. a straight
        line down the variation list rather than zipf's curve. Needs the group
        sizes, so it cannot go through from_logits."""
        return norm(np.log(np.maximum(np.repeat(sizes, sizes) - f["r"], 1.0)),
                    offsets, sizes)

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
        Model("visits-linear", 0, from_logits(lambda f, p: logv(f)), [], []),
        Model("visits-power(alpha)", 1, from_logits(lambda f, p: p[0] * logv(f)),
              [1.0], [(0.0, 20.0)]),
        Model("winrate*visits", 0, from_logits(lambda f, p: logw(f) + logv(f)),
              [], []),
        Model("lcb(c)", 1, from_logits(lcb_logits), [0.2], [(0.0, 5.0)]),
        Model("lcb-fixed", 0, from_logits(lcb_fixed), [], []),
        Model("1/order", 0, from_logits(lambda f, p: -logr(f)), [], []),
        Model("linear-order", 0, linear_order, [], []),
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

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embeddings", choices=("actbot", "shipped"), default="actbot",
                    help="per-cell axis projection phase 2 reads. 'shipped' uses "
                         "the production axes/per_cell/ store; 'actbot' (default) "
                         "projects the variation cells through --model on the fly")
    ap.add_argument("--model", default=DEFAULT_DEV_MODEL, metavar="CKPT",
                    help="checkpoint for --embeddings actbot")
    ap.add_argument("--features", default=DEFAULT_DEV_FEATURES,
                    help="the per-problem block layout --model was FITTED on; it "
                         "is not recoverable from the file")
    return ap.parse_args(argv)


def main():
    args = parse_args()
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
    w_all, s_all, v_all, r_all, cnt_all, gtp_all = [], [], [], [], [], []
    cursor = 0
    for pi, key in enumerate(prob_keys):
        info = problems[key]
        offsets.append(cursor)
        for r, v in enumerate(info["wrong"]):
            w_all.append(v["w"])
            s_all.append(v["s"] if v["s"] is not None else 0.0)
            v_all.append(v["v"])
            r_all.append(r)          # position in the SGF's own variation list
            cnt_all.append(picks[key].get(v["gtp"], 0))
            gtp_all.append(v["gtp"])
            pidx.append(pi)
            cursor += 1
    offsets = np.array(offsets, dtype=np.int64)
    feats = {"w": np.array(w_all), "s": np.array(s_all),
             "v": np.array(v_all, dtype=np.float64),
             "r": np.array(r_all, dtype=np.float64)}
    counts = np.array(cnt_all, dtype=np.float64)
    pidx = np.array(pidx, dtype=np.int64)
    sizes = np.diff(np.append(offsets, len(counts)))

    vv = feats["v"]
    print(f"\nvisits behind a wrong variation's estimate : "
          f"median {np.median(vv):.0f}, p10 {np.percentile(vv, 10):.0f}, "
          f"p90 {np.percentile(vv, 90):.0f}, max {vv.max():.0f}")
    print(f"  at the shallow floor (30) or below       : "
          f"{100.0 * float((vv <= 30).mean()):.1f}% of rows")

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
    # A FIXED PANEL, not a phase-1 winner. Phase 1 ranks nulls on which wrong
    # MOVE gets picked; phase 2 asks a different question of the same nulls and
    # must be free to disagree, so the panel is named rather than selected --
    # three parameter-free candidates spanning the signals available (move
    # quality, search depth, list position) plus the fitted best for reference.
    arms = []
    for tag, name in PHASE2_PANEL:
        if name not in heldout_logp:
            raise SystemExit(f"phase 2 panel names {name!r}, which build_models "
                             f"does not define")
        arms.append((tag, name, np.exp(heldout_logp[name])))
    phase2(prob_keys, problems, arms, offsets, sizes, counts, gtp_all, matched,
           args)


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


# --------------------------------------------------------------------------
# Phase 2: per-cell axis loadings.
#
# Two sources, same interface. `shipped` reads the production axes/per_cell/
# store; `actbot` projects a problem's variation cells through a named
# checkpoint on the fly. BOTH quantize to the client's own resolution before
# any comparison, because the null and the observed sign are both STRICT
# comparisons and a difference below 0.01 is a tie to the client.
# --------------------------------------------------------------------------

class ShippedCells:
    """Per-cell axis vectors as pipeline.write_mirt_per_cell published them."""

    source = f"the shipped axes/per_cell/ store ({PER_CELL_DIR})"

    def __init__(self, K):
        self.K = K

    def cells(self, size, tid, wanted):
        path = os.path.join(PER_CELL_DIR, str(size), tid + ".npz")
        try:
            with np.load(path, allow_pickle=False) as d:
                emb = np.asarray(d["embeddings"])
                xys = np.asarray(d["xys"], dtype=np.int32)
        except OSError:
            return None
        if emb.ndim != 2 or emb.shape[1] != self.K:
            return None
        q = _quantize(emb)
        out = {}
        for i, (x, y) in enumerate(xys):
            xy = (int(x), int(y))
            if xy in wanted:
                out[xy] = q[i]
        return out


class FittedCells:
    """The same vectors projected on the fly through a named checkpoint, so a
    dev fit can be read at the cells the client actually compares. Only the
    wanted cells go through the trunk -- the availability null lives on the
    problem's root variations, never on the whole board scan."""

    def __init__(self, K, model_path, features):
        import torch
        from ablations._projection import FittedProjection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            # One problem's variation cells is a few dozen rows, and at that
            # size the intra-op thread pool spends more on synchronisation than
            # on the matmuls: measured ~13x slower at 4 threads than at 1.
            torch.set_num_threads(1)
        self.K = K
        self.P = FittedProjection(device, model_path, features)
        self.source = (f"{model_path} projected on the fly "
                       f"(blocks={features}, fdim {self.P.fdim}, device {device})")

    def cells(self, size, tid, wanted):
        from ablations._projection import problem_arrays
        try:
            cand, xys, _ = problem_arrays(size, tid)
        except OSError:
            return None
        lut = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}
        keys = [xy for xy in wanted if xy in lut]
        if not keys:
            return None
        rows = np.array([lut[xy] for xy in keys], dtype=np.intp)
        inp = np.concatenate(
            [self.P.pca(cand[rows]),
             np.tile(self.P.tail(size, tid), (len(rows), 1))], axis=1)
        q = _quantize(self.P.axes(inp.astype(np.float32)))
        return {xy: q[i] for i, xy in enumerate(keys)}


def _quantize(a):
    """Client resolution, single-sourced with the other ablations."""
    from ablations._projection import quantize
    return quantize(np.asarray(a, dtype=np.float64))


def phase2(prob_keys, problems, arms, offsets, sizes, counts, gtp_all, matched,
           args):
    """Carry the candidate nulls through to the statistic the client prints.

    `arms` is [(tag, model name, per-row probability)] for the WEIGHTED nulls;
    the uniform-availability null (what ships) is always the first column and
    the reference every residual and every flag delta is taken against.
    """
    print("\n" + "=" * 78)
    print("PHASE 2 -- directional (toward-the-motif) share per axis")
    print("=" * 78)

    labels = json.load(open(LABELS_JSON))["components"]
    K = len(labels)
    nprob = len(prob_keys)

    src = (ShippedCells(K) if args.embeddings == "shipped"
           else FittedCells(K, args.model, args.features))
    print(f"per-cell axis source: {src.source}")

    # Null column 0 is ALWAYS the shipped uniform-availability null.
    tags = ["uniform"] + [t for t, _, _ in arms]
    names = ["uniform over the alternatives offered"] + [n for _, n, _ in arms]
    row_p = [None] + [p for _, _, p in arms]
    n_null = len(tags)

    n_rows = len(counts)
    row_toward = np.zeros((n_rows, K), dtype=bool)
    row_nontie = np.zeros((n_rows, K), dtype=bool)
    # Per-problem null probability of erring TOWARD the motif, conditioned on a
    # non-tie exactly as the client's sign test conditions.
    prob_p = np.full((n_null, nprob, K), np.nan)

    n_no_cells = 0
    n_unmapped = 0
    n_used = 0

    t0 = time.time()
    for pi, key in enumerate(prob_keys):
        if pi % 4000 == 0 and pi:
            print(f"  {pi}/{nprob} problems ({time.time() - t0:.0f}s)", flush=True)
        size, tid = key
        info = problems[key]
        cxy = gtp_to_xy(info["correct_gtp"]) if info["correct_gtp"] else None
        if cxy is None:
            n_unmapped += 1
            continue
        cxy = (int(cxy[0]), int(cxy[1]))

        lo, hi = int(offsets[pi]), int(offsets[pi] + sizes[pi])
        rows = np.arange(lo, hi)
        # (x, y) always, never a position: the npz is a board scan, the SGF's
        # variations are a pruned list, and the two orders do not correspond.
        v_xy = [gtp_to_xy(gtp_all[r]) for r in rows]
        v_xy = [None if xy is None else (int(xy[0]), int(xy[1])) for xy in v_xy]
        wanted = {xy for xy in v_xy if xy is not None}
        wanted.add(cxy)

        cellmap = src.cells(size, tid, wanted)
        if cellmap is None:
            n_no_cells += 1
            continue
        c_emb = cellmap.get(cxy)
        if c_emb is None:
            n_unmapped += 1
            continue

        v_emb = np.full((len(rows), K), np.nan)
        ok = np.zeros(len(rows), dtype=bool)
        for j, xy in enumerate(v_xy):
            e = cellmap.get(xy) if xy is not None else None
            if e is not None:
                v_emb[j] = e
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

        for j in range(n_null):
            w = ok.astype(np.float64) if row_p[j] is None else row_p[j][rows] * ok
            # p = mass strictly ABOVE the correct cell / mass strictly above or
            # below it. Ties drop WITH their weight, which is what makes this
            # the same conditioning computeAltStats' strict counts apply.
            mass_nt = (w[:, None] * nontie).sum(axis=0)
            mass_tw = (w[:, None] * toward).sum(axis=0)
            good = mass_nt > 1e-12
            prob_p[j, pi, good] = mass_tw[good] / mass_nt[good]

    print(f"\nproblems with a usable per-cell projection : {n_used:,} of {nprob:,}")
    print(f"  no per-cell vectors for the problem      : {n_no_cells:,}")
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
    okj = np.isfinite(prob_p)
    mass = np.where(okj, per_prob_n[None] * np.nan_to_num(prob_p), 0.0)
    nmass = np.where(okj, per_prob_n[None], 0.0)

    obs = per_prob_tw.sum(0) / np.maximum(per_prob_n.sum(0), 1)
    pnull = mass.sum(1) / np.maximum(nmass.sum(1), 1)          # (n_null, K)

    # bootstrap over problems, paired across observed and every null
    rng = np.random.default_rng(RNG_SEED)
    b_obs, b_null = [], []
    for _ in range(500):
        mult = rng.multinomial(nprob, np.full(nprob, 1.0 / nprob))[:, None]
        b_obs.append((mult * per_prob_tw).sum(0)
                     / np.maximum((mult * per_prob_n).sum(0), 1))
        b_null.append((mult[None] * mass).sum(1)
                      / np.maximum((mult[None] * nmass).sum(1), 1))
    b_obs = np.array(b_obs)                                    # (B, K)
    b_null = np.array(b_null)                                  # (B, n_null, K)

    # Wide enough for the longest tag PHASE2_PANEL can carry, plus a space.
    wcol = max(10, max(len(t) for t in tags) + 2)
    head = "".join(f"{t:>{wcol}}" for t in tags)
    print(f"\n--- toward-share per axis: observed, and each availability null ---")
    print(f"{'axis':<16}{'observed':>22}{head}")
    print("-" * (38 + wcol * n_null))
    for k in range(K):
        lo_o, hi_o = np.percentile(b_obs[:, k], [2.5, 97.5])
        cols = "".join(f"{pnull[j, k]:>{wcol}.4f}" for j in range(n_null))
        print(f"{labels[k]:<16}{obs[k]:>8.4f} [{lo_o:.4f},{hi_o:.4f}]{cols}")

    print(f"\n--- residual (observed - null), 95% paired bootstrap over problems ---")
    print(f"{'axis':<16}" + "".join(f"{'vs ' + t:>26}" for t in tags))
    print("-" * (16 + 26 * n_null))
    for k in range(K):
        cells = ""
        for j in range(n_null):
            res = b_obs[:, k] - b_null[:, j, k]
            lo_r, hi_r = np.percentile(res, [2.5, 97.5])
            cells += f"{obs[k] - pnull[j, k]:>+9.4f} [{lo_r:+.4f},{hi_r:+.4f}]".rjust(26)
        print(f"{labels[k]:<16}{cells}")
    print("\nA residual clear of zero is a directional tendency the null does NOT\n"
          "explain; a residual straddling zero means the whole deviation is\n"
          "explained by WHICH wrong moves the problem offers, weighted that way.")
    for t, nm in zip(tags, names):
        print(f"  {t:<13} = {nm}")
    print(f"\nnon-tie error counts per axis: "
          f"{', '.join(f'{labels[k]}={int(per_prob_n.sum(0)[k]):,}' for k in range(K))}")
    print("FOOTNOTE: the retired fixed-50% null sits at 0.5000 on every axis and\n"
          "every problem; it is not a column here because it is not a null the\n"
          "client stands against any more.")

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
            acc = users[user_id] = {"n": np.zeros((n_null, K)),
                                    "tw": np.zeros((n_null, K)),
                                    "p": np.zeros((n_null, K)),
                                    "v": np.zeros((n_null, K))}
        nt = row_nontie[r]
        tw = row_toward[r]
        for j in range(n_null):
            p = prob_p[j, pi]
            use = nt & np.isfinite(p)
            acc["n"][j][use] += 1
            acc["tw"][j][use] += tw[use]
            acc["p"][j][use] += p[use]
            acc["v"][j][use] += p[use] * (1 - p[use])

    n_users = 0
    tested = np.zeros((n_null, K))
    flag = np.zeros((n_null, K))
    lost = np.zeros((n_null, K))
    gained = np.zeros((n_null, K))
    flip = np.zeros((n_null, K))
    flag_legacy = np.zeros(K)
    for acc in users.values():
        ok = acc["n"] >= MIN_ERRORS
        if not ok.any():
            continue
        n_users += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            z = ((acc["tw"] - acc["p"])
                 / np.sqrt(np.maximum(acc["v"], 1e-9)))
        f = ok & (np.abs(z) > MOTIF_Z)
        tested += ok
        flag += f
        # Everything is measured against null 0, the shipped uniform one.
        for j in range(1, n_null):
            both = f[0] & f[j]
            lost[j] += f[0] & ~f[j]
            gained[j] += f[j] & ~f[0]
            flip[j] += both & (np.sign(z[0]) != np.sign(z[j]))
        # The retired fixed-50% test, for the footnote only.
        with np.errstate(invalid="ignore", divide="ignore"):
            share = acc["tw"][0] / np.maximum(acc["n"][0], 1)
            z50 = (share - 0.5) / np.sqrt(0.25 / np.maximum(acc["n"][0], 1))
        flag_legacy += ok[0] & (np.abs(z50) > MOTIF_Z)

    print(f"users with >= {MIN_ERRORS} scored errors on an axis : {n_users:,}")
    print(f"flag gate: |Poisson-binomial z| > {MOTIF_Z} (MOTIF_Z), both sides")
    print(f"\n{'axis':<16}{'null':<13}{'tested':>8}{'flagged':>16}{'lost':>8}"
          f"{'gained':>8}{'sign flip':>11}")
    print("-" * 80)
    for k in range(K):
        for j in range(n_null):
            t = max(tested[j, k], 1)
            head = labels[k][:15] if j == 0 else ""
            deltas = ("" if j == 0 else
                      f"{int(lost[j, k]):>8,}{int(gained[j, k]):>8,}"
                      f"{int(flip[j, k]):>11,}")
            print(f"{head:<16}{tags[j]:<13}{int(tested[j, k]):>8,}"
                  f"{int(flag[j, k]):>9,} ({100 * flag[j, k] / t:4.1f}%)"
                  f"{deltas}")
    print("\n'lost'/'gained'/'sign flip' are all against that axis' 'uniform' row --\n"
          "the null the client ships. FOOTNOTE: the retired fixed-50% test would\n"
          "flag "
          + ", ".join(f"{labels[k]}={int(flag_legacy[k]):,}" for k in range(K))
          + ".")

    phase2b(labels, tags, names, prob_p, row_pidx, row_toward, row_nontie,
            counts, nprob, K)


def phase2b(labels, tags, names, prob_p, row_pidx, row_toward, row_nontie,
            counts, nprob, K):
    """Score each null on the ONE number the tendency statistic consumes.

    The client never asks a null which of ~58 wrong moves gets clicked. It asks
    it for p -- the probability that a direction-blind error on THIS problem
    lands above the answer on THIS axis -- and then scores a binary outcome
    against it. So the honest criterion is the proper score of that binary
    prediction, per error, per axis: log-loss (and Brier beside it, since a
    clipped log-loss is sensitive to the clip and Brier is not).

    Errors are POOLED across users on purpose. A null is a player-free object;
    anything per-user here would be scoring the player, not the null.

    Every candidate is unfitted, so it needs no folds -- p is a deterministic
    function of the problem. The fitted arm carries its phase-1 cross-fitting
    with it, which is what keeps it an honest reference rather than a bound it
    reaches by having seen these errors.
    """
    print("\n" + "=" * 78)
    print("PHASE 2b -- per-error DIRECTIONAL log-loss: does the null predict "
          "the SIDE?")
    print("=" * 78)

    n_null = len(tags)
    per_prob_ll = np.zeros((n_null, K, nprob))
    per_prob_br = np.zeros((n_null, K, nprob))
    per_prob_n = np.zeros((K, nprob))
    clipped = np.zeros((n_null, K))
    total_n = np.zeros(K)

    for k in range(K):
        pk_all = prob_p[:, row_pidx, k]                   # (n_null, n_rows)
        # One scored set for every null, so the comparison is paired per error.
        fin = np.isfinite(pk_all).all(axis=0) & row_nontie[:, k]
        w = counts * fin
        y = row_toward[:, k].astype(np.float64)
        per_prob_n[k] = np.bincount(row_pidx, weights=w, minlength=nprob)
        total_n[k] = float(w.sum())
        for j in range(n_null):
            raw = pk_all[j]
            # A one-sided menu gives p exactly 0 or 1, and an error that lands on
            # the impossible side then costs infinity. Clip at the same floor the
            # multiplicative models use, and REPORT the rate: a null that owes
            # its score to the clip is being scored on the clip.
            with np.errstate(invalid="ignore"):
                hit = (raw <= EPS_WR) | (raw >= 1.0 - EPS_WR)
            clipped[j, k] = float((w * hit).sum())
            p = np.clip(np.nan_to_num(raw, nan=0.5), EPS_WR, 1.0 - EPS_WR)
            ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            br = (y - p) ** 2
            per_prob_ll[j, k] = np.bincount(row_pidx, weights=w * ll,
                                            minlength=nprob)
            per_prob_br[j, k] = np.bincount(row_pidx, weights=w * br,
                                            minlength=nprob)

    ll = per_prob_ll.sum(2) / np.maximum(total_n, 1)       # (n_null, K)
    br = per_prob_br.sum(2) / np.maximum(total_n, 1)
    ll_all = per_prob_ll.sum((1, 2)) / max(total_n.sum(), 1)
    br_all = per_prob_br.sum((1, 2)) / max(total_n.sum(), 1)

    # Paired bootstrap over PROBLEMS, one resample scored by every null.
    rng = np.random.default_rng(RNG_SEED)
    b_d = np.zeros((500, n_null, K))
    b_d_all = np.zeros((500, n_null))
    for b in range(500):
        mult = rng.multinomial(nprob, np.full(nprob, 1.0 / nprob)).astype(float)
        num = per_prob_ll @ mult                           # (n_null, K)
        den = per_prob_n @ mult                            # (K,)
        val = num / np.maximum(den, 1)
        b_d[b] = val - val[0]
        tot = num.sum(1) / max(den.sum(), 1)
        b_d_all[b] = tot - tot[0]

    print(f"\n{'axis':<16}{'null':<13}{'errors':>9}{'logloss':>9}"
          f"{'delta':>9}{'delta 95% CI':>20}{'brier':>8}{'clip%':>8}")
    print("-" * 92)
    for k in range(K):
        for j in range(n_null):
            head = labels[k][:15] if j == 0 else ""
            n_s = f"{int(total_n[k]):,}" if j == 0 else ""
            if j == 0:
                d, ci = "", ""
            else:
                lo, hi = np.percentile(b_d[:, j, k], [2.5, 97.5])
                d = f"{ll[j, k] - ll[0, k]:>+9.4f}"
                ci = f"[{lo:+.4f}, {hi:+.4f}]"
            print(f"{head:<16}{tags[j]:<13}{n_s:>9}{ll[j, k]:>9.4f}{d:>9}"
                  f"{ci:>20}{br[j, k]:>8.4f}"
                  f"{100 * clipped[j, k] / max(total_n[k], 1):>8.2f}")
    print("-" * 92)
    for j in range(n_null):
        if j == 0:
            d, ci = "", ""
        else:
            lo, hi = np.percentile(b_d_all[:, j], [2.5, 97.5])
            d = f"{ll_all[j] - ll_all[0]:>+9.4f}"
            ci = f"[{lo:+.4f}, {hi:+.4f}]"
        n_s = f"{int(total_n.sum()):,}" if j == 0 else ""
        print(f"{'ALL AXES':<16}{tags[j]:<13}{n_s:>9}{ll_all[j]:>9.4f}{d:>9}"
              f"{ci:>20}{br_all[j]:>8.4f}"
              f"{100 * clipped[j].sum() / max(total_n.sum(), 1):>8.2f}")

    print("\nEach row scores one null on the binary outcome the tendency test "
          "actually\nconsumes: did this error land ABOVE the answer on this "
          "axis, against that\nnull's p for this problem. delta < 0 = better "
          "than the uniform availability\nnull the client ships; the interval "
          "is a paired bootstrap over PROBLEMS,\nthe same resample for every "
          "null. 'clip%' is the share of scored errors whose\np hit the "
          f"{EPS_WR:g} floor or ceiling -- a one-sided menu -- and is a "
          "property of\nthe null, not of the players.")
    for t, nm in zip(tags, names):
        print(f"  {t:<13} = {nm}")
    print("\nThis is the ADJUDICATING table. Phase 1 ranks nulls on which wrong "
          "move gets\npicked, which the client never asks; reliability "
          "(tendency_null_variants) rises\nwith RETAINED problem structure and "
          "so cannot referee between nulls either.\nA null belongs on the "
          "client when it predicts the SIDE better than uniform.")


if __name__ == "__main__":
    main()
