"""Reproduce the PWA's "Likely issues" (weakness) results offline from the local
DB copy, for one user or for every user with an EGD profile attached.

This is a faithful Python port of the client's weakness pipeline — the same math,
same constants, same two-level detection — so the output matches what a user sees
in the stats modal's Weaknesses pane (up to the usual "backend artifacts may be a
re-run ahead of the last package_for_pwa" caveat). The ported surfaces are:

  - pwa/js/weakness.js          — trainWeakness(): the 2K+2 logistic regression
                                  (α + γ·rank + Σ β_correct·correct_emb
                                   + Σ β_picked·picked_emb), per-axis ridge prior
                                  centred on half the shipped sum mean, and the
                                  Laplace posterior (SE + per-axis 2×2 cov on
                                  (β_correct[k], β_picked[k])).
  - pwa/js/optimizer.js         — newtonBlockStep(): the damped-Newton step the fit
                                  takes, its Armijo backtrack and the Newton
                                  decrement it stops on.
  - pwa/js/stats-weaknesses.js  — buildLikelyIssuesElement(): level-1 flagging on
                                  the identified SUM posterior against the shipped
                                  population centre, the false-flag error budget,
                                  and the level-2 direction.
  - pwa/js/stats-compute.js     — computeAxisDirections()/betaProbAboveHalf(): the
                                  paired picked−correct sign test.

Per-attempt features are assembled exactly as the client's answerHistory carries
them after backfill:
  - embedding      = the problem's correct-move axis vector (axes/
                     per_problem.csv, the same source the shipped lookup sidecar +
                     shard metadata are emitted from).
  - pickedEmbedding = the axis vector at the cell the user actually clicked, for
                     WRONG non-repeat attempts. The DB `move` is GTP notation in
                     the canonical orientation-0 frame (game-logic.buildReportPayload
                     inverse-transforms the click), so gtp_to_xy(move) indexes the
                     per-cell npz (axes/per_cell/<size>/<tid>.npz) by
                     coordinate — the same (x,y) frame extraction labels cells with.
                     (Validated 300/300 on non-repeat correct attempts; repeats are
                     served-frame and unrecoverable, and are excluded from training.)
  - problemRank    = rankForSize(rating): applyM(rating), the
                     board-independent internal rank the model reads as `rating`.

Usage:
    ./.venv/bin/python3 -m auxiliary.user_weakness_report --user <userId>
    ./.venv/bin/python3 -m auxiliary.user_weakness_report --egd
    ./.venv/bin/python3 -m auxiliary.user_weakness_report --egd --json

Reads only local artifacts (no D1 / network).
"""

import argparse
import csv
import json
import math
import os
import sqlite3
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.paths import DB_SQLITE  # noqa: E402
from utils.katago.coords import gtp_to_xy  # noqa: E402

# --- Artifact locations (backend + shipped) --------------------------------
DEFAULT_DB = DB_SQLITE
PER_PROBLEM_CSV = os.path.join(PROJECT_ROOT, "axes", "per_problem.csv")
PER_CELL_DIR = os.path.join(PROJECT_ROOT, "axes", "per_cell")
RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")
# Shipped copies (what the client actually reads) preferred; backend fallbacks.
CURVE_JSON = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rating_to_rank.json")
CURVE_JSON_BACKEND = os.path.join(PROJECT_ROOT, "ratings", "rating_to_rank.json")
EMBED_META = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "embedding_meta.json")
POP_CONSTANTS = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "population_constants.json")

# --- Constants (verbatim from the JS sources) ------------------------------
MIN_TRAINING_ATTEMPTS = 20         # weakness.js
MODEL_INPUT_HISTORY_LENGTH = 500   # constants.js
MAX_NEWTON_ITERS = 50              # weakness.js (runaway guard, not a budget)
NEWTON_TOL = 1e-8                  # weakness.js, on decrement/2
ARMIJO_C = 1e-4                    # optimizer.js
MAX_BACKTRACKS = 30                # optimizer.js
RIDGE_GUARD = 1e-6                 # weakness.js lamVec guard on intercept/rating

# Level-1 / level-2 detection thresholds (stats-weaknesses.js).
WEAK_PROB_THRESHOLD = 0.80
WEAK_FALSE_FLAG_BUDGET = 0.5       # expected spurious issues the pane will show
DIRECTION_PROB_THRESHOLD = 0.85
MIN_DIRECTION_N = 3

# English "Likely issues" phrasings (i18n.js en block).
PHRASES = {
    "tooMuch": "You rely on «{motif}» too much.",
    "notEnough": "You don't trust «{motif}» enough.",
    "badAt": "You are weak at «{motif}».",
}


# ---------------------------------------------------------------------------
# Small numeric helpers, ported to match the client bit-for-bit where it matters.
# ---------------------------------------------------------------------------
def sigmoid(z):  # math-utils.js
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def norm_cdf(z):  # math-utils.js: 0.5*(1+erf(z/sqrt2))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def interp_extrap(x, xs, ys):  # shared/rank-math.js
    n = len(xs)
    if x <= xs[0]:
        return ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0])
    if x >= xs[n - 1]:
        return ys[n - 1] + (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]) * (x - xs[n - 1])
    a, b = 0, n - 1
    while b - a > 1:
        m = (a + b) >> 1
        if xs[m] <= x:
            a = m
        else:
            b = m
    return ys[a] + (ys[b] - ys[a]) * (x - xs[a]) / (xs[b] - xs[a])


def beta_prob_above_half(pos, neg):  # stats-compute.js
    N = pos + neg + 1
    term = 0.5 ** N
    total = term
    for j in range(1, pos + 1):
        term *= (N - j + 1) / j
        total += term
    return min(1.0, max(0.0, total))



# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------
def load_correct_embeddings():
    """(tsumego_id, size) -> correct-move axis vector (list of K floats)."""
    out = {}
    with open(PER_PROBLEM_CSV) as f:
        reader = csv.DictReader(f)
        kcols = [c for c in reader.fieldnames if c.startswith("embed_")]
        kcols.sort(key=lambda c: int(c.split("_")[1]))
        for row in reader:
            key = (row["tsumego_id"], int(row["size"]))
            out[key] = [float(row[c]) for c in kcols]
    return out, len(kcols)


def load_ratings():
    """(tsumego_id, size) -> raw rating, from ratings/<size>/tsumego_ratings.csv."""
    out = {}
    for size in (9, 13):
        path = os.path.join(RATINGS_DIR, str(size), "tsumego_ratings.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                out[(row["tsumego_id"], size)] = float(row["rating"])
    return out


def load_curve():
    path = CURVE_JSON if os.path.exists(CURVE_JSON) else CURVE_JSON_BACKEND
    with open(path) as f:
        data = json.load(f)
    # Top level carries size keys ("9"/"13") plus a "time_adjust" block; keep only
    # the numeric size keys. Both size curves are identical (one shared fit).
    return {int(k): (v["ratings"], v["ranks"])
            for k, v in data.items() if k.isdigit()}


def load_labels_and_prior():
    with open(EMBED_META) as f:
        meta = json.load(f)
    labels = meta["labels"]["components"]  # array; component[k] = {positive, negative,...}
    K = meta["K"]
    with open(POP_CONSTANTS) as f:
        pop = json.load(f)
    axis_lambda = pop["weakness_axis_lambda"]
    axis_sum_mean = pop["weakness_axis_mean"]
    return labels, K, axis_lambda, axis_sum_mean


class PickedCellLoader:
    """Lazily loads per-cell npz files and resolves a GTP move to its axis vector."""

    def __init__(self):
        self._cache = {}  # (size, tid) -> {(x,y): np.ndarray}

    def _load(self, size, tid):
        key = (size, tid)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(PER_CELL_DIR, str(size), tid + ".npz")
        table = None
        if os.path.exists(path):
            try:
                with np.load(path) as d:
                    emb = d["embeddings"].astype(np.float32)
                    xys = d["xys"]
                table = {(int(x), int(y)): emb[i] for i, (x, y) in enumerate(xys)}
            except Exception:
                table = None
        self._cache[key] = table
        return table

    def picked(self, size, tid, move):
        """Axis vector (list of floats) at the clicked cell, or None if unmappable."""
        if not move:
            return None
        xy = gtp_to_xy(move)
        if xy is None:
            return None
        table = self._load(size, tid)
        if not table:
            return None
        vec = table.get((int(xy[0]), int(xy[1])))
        return None if vec is None else [float(v) for v in vec]


def build_history(con, user_id, correct_emb, ratings, curves, picked_loader):
    """Assemble the model's per-attempt rows for one user, mirroring the client's
    trainWeakness() filter: non-repeat attempts that carry an embedding + a numeric
    problemRank. Date-ordered (getMergedAnswerHistory is date-ordered), then the
    caller slices the last MODEL_INPUT_HISTORY_LENGTH."""
    rows = con.execute(
        "SELECT problem, size, success, move, is_repeat, created_at "
        "FROM tsumego_reports WHERE user_id = ? ORDER BY created_at ASC, id ASC",
        (user_id,),
    ).fetchall()

    history = []
    for r in rows:
        if r["is_repeat"]:
            continue  # weakness.js: !a.isRepeat (repeats also carry served-frame moves)
        size = r["size"]
        tid = r["problem"]
        key = (tid, size)
        emb = correct_emb.get(key)
        rating = ratings.get(key)
        if emb is None or rating is None:
            continue  # rating-less / embedding-less rows are skipped, as on the client
        curve = curves.get(size) or next(iter(curves.values()))
        problem_rank = interp_extrap(rating, curve[0], curve[1])
        correct = bool(r["success"])
        picked = None
        if not correct:  # picked cell only recorded/needed for wrong attempts
            picked = picked_loader.picked(size, tid, r["move"])
        history.append({
            "problemId": tid,
            "size": size,
            "correct": correct,
            "problemRank": problem_rank,
            "embedding": emb,
            "pickedEmbedding": picked,
            "date": r["created_at"],
        })
    return history


# ---------------------------------------------------------------------------
# Model: trainWeakness() + Laplace posterior (weakness.js).
# ---------------------------------------------------------------------------
def solve_linear(A, b):
    """optimizer.js solveLinear: Gauss-Jordan with partial pivoting; None on a
    singular pivot or a non-finite solution."""
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return x if np.all(np.isfinite(x)) else None


def newton_block_step(params, grad, hess, loss_fn):
    """optimizer.js newtonBlockStep over the whole parameter block.

    Returns (decrement, accepted); `params` holds the accepted point, or the
    starting point when nothing was accepted."""
    step = solve_linear(hess, grad)
    if step is None:
        return 0.0, False
    decrement = float(grad @ step)
    if not decrement > 0:          # curvature not PD in this direction
        return 0.0, False
    loss0 = loss_fn(params)
    start = params.copy()
    t = 1.0
    for _ in range(MAX_BACKTRACKS):
        params[:] = start - t * step
        loss = loss_fn(params)
        if np.isfinite(loss) and loss <= loss0 - ARMIJO_C * t * decrement:
            return decrement, True
        t /= 2.0
    params[:] = start
    return decrement, False


def train_weakness(history, K, axis_lambda, axis_sum_mean):
    """Returns (params, se, axis_cov) or None when there isn't enough signal —
    faithful to weakness.js trainWeakness(): a damped-Newton fit of the penalized
    logistic in RAW SUMS over the window (the scaling the posterior covariance is
    defined in), with the per-axis ridge pulling each of an axis's two slots
    toward HALF the shipped population sum centre — so the SUM is pulled to that
    centre while the CONTRAST is still pulled to zero."""
    hist = history[-MODEL_INPUT_HISTORY_LENGTH:]
    if len(hist) < MIN_TRAINING_ATTEMPTS:
        return None
    if any(len(a["embedding"]) != K for a in hist):
        return None
    if not (isinstance(axis_lambda, list) and len(axis_lambda) == K
            and all(np.isfinite(v) and v > 0 for v in axis_lambda)):
        return None
    if not (isinstance(axis_sum_mean, list) and len(axis_sum_mean) == K
            and all(np.isfinite(v) for v in axis_sum_mean)):
        return None

    num_params = 2 + 2 * K
    N = len(hist)

    # Picked vector per row: sidecar pick for wrong attempts, else the correct
    # embedding (fallback row informs only β_correct + β_picked).
    picked_for = [
        a["pickedEmbedding"] if (a["pickedEmbedding"] is not None
                                 and len(a["pickedEmbedding"]) == K)
        else a["embedding"]
        for a in hist
    ]

    # Design matrix: [1, problemRank, correct_emb, picked_emb].
    X = np.empty((N, num_params))
    for i, a in enumerate(hist):
        X[i, 0] = 1.0
        X[i, 1] = a["problemRank"]
        X[i, 2:2 + K] = a["embedding"]
        X[i, 2 + K:] = picked_for[i]
    y = np.array([1.0 if a["correct"] else 0.0 for a in hist])

    lam_vec = np.full(num_params, RIDGE_GUARD)
    prior_mean = np.zeros(num_params)
    for k in range(K):
        lam_vec[2 + k] = lam_vec[2 + K + k] = axis_lambda[k]
        prior_mean[2 + k] = prior_mean[2 + K + k] = axis_sum_mean[k] / 2.0

    def probs(par):
        return 1.0 / (1.0 + np.exp(-(X @ par)))

    def loss_at(par):
        pr = np.clip(probs(par), 1e-15, 1 - 1e-15)
        d = par - prior_mean
        return float(-(y * np.log(pr) + (1 - y) * np.log(1 - pr)).sum()
                     + 0.5 * (lam_vec * d * d).sum())

    def grad_at(par):
        return X.T @ (probs(par) - y) + lam_vec * (par - prior_mean)

    def hess_at(par):
        pr = probs(par)
        w = pr * (1.0 - pr)
        return X.T @ (w[:, None] * X) + np.diag(lam_vec)

    params = np.zeros(num_params)
    steps = 0
    for _ in range(MAX_NEWTON_ITERS):
        decrement, accepted = newton_block_step(
            params, grad_at(params), hess_at(params), loss_at)
        if not accepted:
            break
        steps += 1
        if decrement / 2 < NEWTON_TOL:
            break
    if steps == 0:
        return None
    if not np.all(np.isfinite(params)):
        return None

    # Laplace posterior: H at the optimum, cov = H^-1.
    H = hess_at(params)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    if not np.all(np.isfinite(se)):
        return None
    axis_cov = [float(cov[2 + k, 2 + K + k]) for k in range(K)]
    return params.tolist(), se.tolist(), axis_cov


# ---------------------------------------------------------------------------
# Two-level detection (stats-weaknesses.js + stats-compute.js).
# ---------------------------------------------------------------------------
def label_for(labels, k, side):
    """The motif name at one tail, or None when that tail is unnamed. Symmetric
    on purpose (stats-weaknesses.js labelFor): an unnamed tail is skipped rather
    than given a placeholder — "style #2" is not a finding a user can act on."""
    entry = labels[k] if k < len(labels) else None
    if not entry:
        return None
    return entry.get("negative") if side == "negative" else entry.get("positive")


def compute_axis_directions(history, flagged_ks, K):
    """stats-compute.js computeAxisDirections: paired sign test over wrong,
    non-repeat attempts with a picked embedding, attributing each error to the
    flagged axis of largest |picked − correct|."""
    out = {k: {"nPos": 0, "nNeg": 0, "pOver": 0.5} for k in flagged_ks}
    if not flagged_ks:
        return out
    rows = history[-MODEL_INPUT_HISTORY_LENGTH:]
    for a in rows:
        if a["correct"]:
            continue
        pick = a["pickedEmbedding"]
        if pick is None:
            continue
        best_k, best_abs, best_d = -1, 0.0, 0.0
        for k in flagged_ks:
            if len(a["embedding"]) <= k or len(pick) <= k:
                continue
            ec, ep = a["embedding"][k], pick[k]
            if not (np.isfinite(ec) and np.isfinite(ep)):
                continue
            d = ep - ec
            if abs(d) > best_abs:
                best_abs, best_k, best_d = abs(d), k, d
        if best_k < 0:
            continue
        s = out[best_k]
        if best_d > 0:
            s["nPos"] += 1
        else:
            s["nNeg"] += 1
    for s in out.values():
        s["pOver"] = beta_prob_above_half(s["nPos"], s["nNeg"])
    return out


def detect_issues(history, params, se, axis_cov, labels, K, axis_sum_mean):
    """Returns the sorted list of {k, side, quadrant, p, sentence} — the exact
    rows the Weaknesses pane renders (buildLikelyIssuesElement)."""
    # Level 1: flag axes whose identified SUM coefficient sits credibly away from
    # the shipped POPULATION CENTRE (not from zero — measuring against zero would
    # report the population's own position as this user's finding).
    candidates = []
    for k in range(K):
        mu_c, mu_p = params[2 + k], params[2 + K + k]
        var_c = se[2 + k] ** 2
        var_p = se[2 + K + k] ** 2
        cov = axis_cov[k] or 0.0
        if not all(np.isfinite(v) for v in (mu_c, mu_p, var_c, var_p, cov)):
            continue
        var_sum = var_c + var_p + 2 * cov
        if not (var_sum > 0):
            continue
        centre = axis_sum_mean[k]
        if not np.isfinite(centre):
            continue
        p_weak_pos = norm_cdf(-(mu_c + mu_p - centre) / math.sqrt(var_sum))
        if p_weak_pos >= WEAK_PROB_THRESHOLD:
            if label_for(labels, k, "positive"):
                candidates.append({"k": k, "p": p_weak_pos, "side": "positive"})
        elif (1 - p_weak_pos) >= WEAK_PROB_THRESHOLD and label_for(labels, k, "negative"):
            candidates.append({"k": k, "p": 1 - p_weak_pos, "side": "negative"})

    # MULTIPLICITY: strongest first, admit while the running sum of (1 − p) —
    # the expected number of spurious issues in the list — stays within the
    # budget. Sorted descending, so a rejected issue is never followed by a
    # cheaper one; stop rather than skip.
    candidates.sort(key=lambda c: c["p"], reverse=True)
    flagged = []
    spent = 0.0
    for c in candidates:
        if spent + (1 - c["p"]) > WEAK_FALSE_FLAG_BUDGET:
            break
        spent += 1 - c["p"]
        flagged.append(c)

    # Level 2: qualify each flagged axis's direction from the paired sign test.
    directions = compute_axis_directions(history, [f["k"] for f in flagged], K)
    issues = []
    for f in flagged:
        k, p, side = f["k"], f["p"], f["side"]
        quadrant = "badAt"
        d = directions.get(k)
        if d and d["nPos"] + d["nNeg"] >= MIN_DIRECTION_N:
            p_over = (1 - d["pOver"]) if side == "negative" else d["pOver"]
            if p_over >= DIRECTION_PROB_THRESHOLD:
                quadrant = "tooMuch"
            elif (1 - p_over) >= DIRECTION_PROB_THRESHOLD:
                quadrant = "notEnough"
        motif = label_for(labels, k, side)
        issues.append({
            "k": k, "side": side, "quadrant": quadrant, "p": p,
            "motif": motif,
            "sentence": PHRASES[quadrant].format(motif=motif),
        })
    # Already strongest-first — the budget loop consumed `candidates` in that
    # order and preserved it.
    return issues


def served_rank_label(history):
    """Mean internal rank of the distinct first-time problems in the training
    window → label. Context only (the served-difficulty band ≈ the user's rank);
    NOT the precise front-page window-MLE headline."""
    window = history[-MODEL_INPUT_HISTORY_LENGTH:]
    seen = {}
    for a in window:
        if a["problemId"] not in seen:
            seen[a["problemId"]] = a["problemRank"]
    if len(seen) < 10:
        return None
    band = int(math.floor(sum(seen.values()) / len(seen)))
    band = max(-30, min(8, band))          # clamp to the labelled ladder (30k..9d)
    return f"{-band}k" if band < 0 else f"{band + 1}d"


# ---------------------------------------------------------------------------
# Report assembly.
# ---------------------------------------------------------------------------
def report_for_user(con, user_id, ctx):
    (correct_emb, ratings, curves, labels, K, axis_lambda, axis_sum_mean,
     picked_loader) = ctx
    history = build_history(con, user_id, correct_emb, ratings, curves, picked_loader)
    used = min(len(history), MODEL_INPUT_HISTORY_LENGTH)
    result = {
        "userId": user_id,
        "usableAttempts": len(history),
        "windowUsed": used,
        "servedRank": served_rank_label(history),
        "status": None,
        "issues": [],
    }
    if len(history) < MIN_TRAINING_ATTEMPTS:
        result["status"] = "not_enough_data"
        return result
    trained = train_weakness(history, K, axis_lambda, axis_sum_mean)
    if trained is None:
        result["status"] = "model_unavailable"
        return result
    params, se, axis_cov = trained
    issues = detect_issues(history, params, se, axis_cov, labels, K, axis_sum_mean)
    result["status"] = "ok"
    result["issues"] = issues
    return result


def egd_users(con):
    """Users with an EGD profile attached (external_ids.egd_pin), joined to their
    claimed invitation for display name / GoR / country."""
    rows = con.execute(
        """
        SELECT e.user_id, e.egd_pin, i.egd_name, i.gor, i.country_code,
               (SELECT COUNT(*) FROM tsumego_reports t WHERE t.user_id = e.user_id) AS attempts
        FROM external_ids e
        LEFT JOIN invitation i
          ON i.claimed_by_user_id = e.user_id AND i.egd_pin = e.egd_pin
        WHERE e.egd_pin IS NOT NULL
        GROUP BY e.user_id
        ORDER BY COALESCE(i.gor, -1) DESC, attempts DESC
        """
    ).fetchall()
    return rows


def print_user_report(result, header=None):
    if header:
        print(header)
    else:
        print(f"User {result['userId']}  ·  {result['usableAttempts']} usable "
              f"attempts ({result['windowUsed']} in window)")
    if result["servedRank"]:
        print(f"  served-difficulty band ≈ {result['servedRank']}")
    if result["status"] == "not_enough_data":
        print("  (not enough data — needs ≥ 20 non-repeat rated attempts)")
        return
    if result["status"] == "model_unavailable":
        print("  (model could not be fit)")
        return
    if not result["issues"]:
        print(f"  no issues above {int(WEAK_PROB_THRESHOLD * 100)}% confidence")
        return
    for it in result["issues"]:
        tail = "" if it["side"] == "positive" else "  [−tail]"
        print(f"  {it['p'] * 100:3.0f}%  {it['sentence']}{tail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--user", help="report the weaknesses of a single userId")
    g.add_argument("--egd", action="store_true",
                   help="report weaknesses for every user with an EGD profile attached")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"local DB (default {DEFAULT_DB})")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        ap.error(f"DB not found: {args.db} (run pipeline.fetch_database first)")

    correct_emb, K_csv = load_correct_embeddings()
    ratings = load_ratings()
    curves = load_curve()
    labels, K, axis_lambda, axis_sum_mean = load_labels_and_prior()
    if K_csv != K:
        print(f"WARNING: per_problem.csv K={K_csv} != embedding_meta K={K}; "
              f"backend/artifacts out of sync", file=sys.stderr)
    ctx = (correct_emb, ratings, curves, labels, K, axis_lambda, axis_sum_mean,
           PickedCellLoader())

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    if args.user:
        result = report_for_user(con, args.user, ctx)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_user_report(result)
        return

    # --egd
    users = egd_users(con)
    results = []
    for u in users:
        result = report_for_user(con, u["user_id"], ctx)
        result["egd"] = {"pin": u["egd_pin"], "name": u["egd_name"],
                         "gor": u["gor"], "country": u["country_code"]}
        result["totalAttempts"] = u["attempts"]
        results.append(result)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    print(f"Weakness results for {len(results)} user(s) with an EGD profile "
          f"(strongest GoR first)\n" + "=" * 72)
    for result in results:
        egd = result["egd"]
        name = egd["name"] or "(unknown)"
        gor = f"GoR {egd['gor']}" if egd["gor"] is not None else "GoR ?"
        country = f", {egd['country']}" if egd["country"] else ""
        header = (f"\n{name} ({gor}{country})  ·  user {result['userId']}  ·  "
                  f"{result['totalAttempts']} attempts, {result['windowUsed']} in window")
        print_user_report(result, header=header)


if __name__ == "__main__":
    main()
