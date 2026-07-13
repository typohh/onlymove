#!/usr/bin/env python3
"""Forward predictive power of the PWA weakness model + a check of the
quality * exp(WEAKNESS_TEMPERATURE * weaknessScore) selection formula.

Offline replication of pwa/js/weakness.js on the full report log — every user,
no timeframe restriction — but ONLY attempts whose reported `move` resolves to a
per-cell embedding, so wrong attempts carry their REAL picked-move embedding
(no picked=correct fallback):

  P(correct) = sigmoid(alpha + gamma*problemRank
                       + beta_correct . correct_emb + beta_picked . picked_emb)

  - correct_emb from embeddings/joint/per_problem.csv (the shipped vector);
  - picked_emb for wrong attempts from embeddings/joint/per_cell/<size>/<tid>.npz,
    row located by utils.coords.gtp_to_xy(move) — the reported move is GTP-style
    board notation in the canonical orientation-0 frame (novel attempts are
    always served at identity, and buildReportPayload canonicalizes since
    2026-07-07), which is exactly the npz `xys` frame;
  - correct attempts use picked = correct by definition (same as the client);
  - is_repeat = 0 only (the client's trainWeakness filter), `move` NOT NULL,
    passes and unresolvable coordinates dropped (counted).

Everything is evaluated FORWARD (walk-forward chaining): at every EVAL_BLOCK-th
attempt, the model is fit on the trailing MODEL_INPUT_HISTORY_LENGTH(=500)
attempts (per-axis empirical-Bayes ridge from population_constants.json, exactly
the client's prior; IRLS instead of the client's time-capped Adam, so this
measures the model class's ceiling) and predicts the NEXT block of attempts it
has never seen.

Report:
  1. FORWARD PREDICTIVE POWER — held-out log-loss / AUC vs the difficulty-only
     baseline sigmoid(alpha + gamma*problemRank) fit on the same window, pooled
     and per user; decile calibration of the predicted weakness logit-shift vs
     realized outcomes. The evaluated prediction uses PRE-ATTEMPT features only:
     sigmoid(alpha + gamma*rank + (beta_correct + beta_picked) . correct_emb) —
     the level-1 candidate view (picked = correct), exactly what selection can
     know before the user moves. Scoring the eval row with its own picked_emb
     would be label leakage: picked != correct if and only if the attempt is
     wrong, so the full feature vector encodes the outcome (a naive full-feature
     "held-out" AUC comes out ~0.95+ on pure reconstruction). The picked
     embeddings' job is in TRAINING, where they separate beta_correct from
     beta_picked and thus identify the contrast score.
  2. SELECTION-SCORE VALIDATION — does the shipped weaknessScore (the CONTRAST
     (beta_picked - beta_correct) . correct_emb) forward-predict UNDERperformance
     on the problems actually attempted next?  kappa from
     success ~ offset(logit p_base) + kappa * score; kappa < 0 = a
     higher-weakness-score problem is failed more than difficulty predicts, and
     |kappa| ~ 1 would mean the score is calibrated in true logit units. The sum
     score -(beta_correct + beta_picked) . emb (the pre-2026-07-10 selection
     form) is fit alongside for comparison.
  3. PICKED-SIDE (CONTRASTIVE) STRUCTURE, FORWARD — the part of the model only
     the real picked embeddings can validate. On held-out WRONG attempts:
       (a) cos(picked_emb - correct_emb, -correct_emb): how much of the error
           direction is the first-order "generic wrong move = -correct_emb"
           assumption weaknessScore rests on;
       (b) alignment of the actual pick (relative to the problem's mean legal
           cell, from the per-cell npz) with the user's trained overplay
           direction -beta_picked, and with the full contrast
           beta_picked - beta_correct: does the model's picked-side direction
           predict WHERE this user errs next?
  4. TEMPERATURE SWEEP — at a subsample of eval points, rebuild the counterfactual
     candidate pool (POOL_N nearest-problemRank same-size corpus problems,
     mirroring the 50-candidate serve pool) and compute, for each T, the
     quality*exp(T*score)-weighted served composition: mean weakness score (in
     pool-SD units) and mean predicted success. This shows what the shipped
     T = 0.5 actually does to the served distribution and what larger T would do.

Usage:
  ./.venv/bin/python3 -m auxiliary.eval_weakness_forward                # full run
  ./.venv/bin/python3 -m auxiliary.eval_weakness_forward --eval-block 100 --pool-every 10
  ./.venv/bin/python3 -m auxiliary.eval_weakness_forward --json out.json
"""

import argparse
import json
import os
import struct
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.coords import gtp_to_xy  # noqa: E402

DB_PATH = os.path.join(PROJECT_ROOT, "attempts", "database.sqlite")
PER_PROBLEM_CSV = os.path.join(PROJECT_ROOT, "embeddings", "joint", "per_problem.csv")
PER_CELL_DIR = os.path.join(PROJECT_ROOT, "embeddings", "joint", "per_cell")
PARAMS_BIN = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "problem_params.bin")
CURVE_JSON = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rating_to_rank.json")
POP_CONSTANTS = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "population_constants.json")

# Client constants mirrored from pwa/js (weakness.js / constants.js).
WEAKNESS_TEMPERATURE = 0.5
MODEL_INPUT_HISTORY_LENGTH = 500
MIN_TRAINING_ATTEMPTS = 20
POOL_N = 50          # the serve pool: 50 nearest-rank candidates
RANK_NORM = 29.0     # client problemRank is applyM(rating)/29 (0..1)


# --------------------------------------------------------------------------- data
def load_problem_data():
    """corpus tables: id -> (correct_emb, prank_norm, quality, size)."""
    import csv
    emb = {}
    size_of = {}
    with open(PER_PROBLEM_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        k = sum(1 for c in header if c.startswith("embed_"))
        for row in reader:
            emb[row[0]] = np.array(row[2:2 + k], dtype=np.float32)
            size_of[row[0]] = int(row[1])

    raw = open(PARAMS_BIN, "rb").read()
    rating, slope = {}, {}
    for off in range(0, len(raw) - 15, 16):
        pid = raw[off:off + 4].hex()
        r, s, _f = struct.unpack_from("<fff", raw, off + 4)
        rating[pid] = r
        slope[pid] = s

    curve = json.load(open(CURVE_JSON))["9"]  # both size keys are identical
    xs = np.array(curve["ratings"]); ys = np.array(curve["rank_indices"])

    def prank_norm(r):
        return float(np.interp(r, xs, ys)) / RANK_NORM

    ids = [i for i in emb if i in rating]
    return (
        {i: emb[i] for i in ids},
        {i: prank_norm(rating[i]) for i in ids},
        {i: max(0.0, slope[i]) ** 2 for i in ids},   # quality = a_p^2 (slopeToQuality)
        {i: size_of[i] for i in ids},
        k,
    )


def load_attempts(db_path):
    """All novel attempts with a reported move, event order, exact dups collapsed."""
    import sqlite3
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """SELECT user_id, problem, success, created_at, size, move
             FROM tsumego_reports
            WHERE is_repeat = 0 AND move IS NOT NULL
            GROUP BY user_id, problem, created_at
            ORDER BY user_id, created_at, id"""
    ).fetchall()
    con.close()
    return rows


def resolve_picked_embeddings(rows, emb, size_of, k):
    """For every WRONG attempt, the picked-move embedding from the per-cell npz.
    Returns (dict (problem, move) -> vec, dict tid -> mean legal-cell embedding,
    drop counters). The cell mean is the null pick for the alignment test in
    report section 3. Correct attempts never need a lookup (picked = correct by
    definition)."""
    need = defaultdict(set)      # tid -> set of moves
    for _u, problem, success, _c, _s, move in rows:
        if not success and problem in emb:
            need[problem].add(move)

    picked = {}
    cell_mean = {}
    dropped = {"pass": 0, "no_npz": 0, "no_cell": 0}
    for tid, moves in need.items():
        path = os.path.join(PER_CELL_DIR, str(size_of[tid]), f"{tid}.npz")
        if not os.path.exists(path):
            dropped["no_npz"] += len(moves)
            continue
        with np.load(path) as data:
            embs = np.asarray(data["embeddings"], dtype=np.float32)
            xys = np.asarray(data["xys"])
        xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}
        if embs.shape[0]:
            cell_mean[tid] = embs.mean(axis=0)
        for move in moves:
            xy = gtp_to_xy(move) if move.lower() != "pass" else None
            if xy is None:
                dropped["pass"] += 1
                continue
            row = xy_to_row.get(xy)
            if row is None:
                dropped["no_cell"] += 1
                continue
            vec = embs[row]
            if vec.shape[0] == k:
                picked[(tid, move)] = vec
    return picked, cell_mean, dropped


# ------------------------------------------------------------------------ fitting
def _pen_deviance(X, y, w, pen):
    z = np.clip(X @ w, -35, 35)
    p = 1.0 / (1.0 + np.exp(-z))
    return logloss(y, p).sum() + 0.5 * float(pen @ (w * w))


def irls(X, y, pen, w0=None):
    """Ridge logistic via IRLS with step-halving; pen is a per-coefficient
    penalty vector. The halving guards against the quasi-separation the picked
    features can create in a small window (without it the weights blow up and
    poison every later warm-started fit)."""
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    P = np.diag(pen + 1e-6)   # tiny floor keeps H invertible for pen == 0 slots
    dev = _pen_deviance(X, y, w, pen)
    for _ in range(100):
        z = np.clip(X @ w, -35, 35)
        p = 1.0 / (1.0 + np.exp(-z))
        Wd = p * (1 - p) + 1e-9
        H = X.T @ (X * Wd[:, None]) + P
        g = X.T @ (y - p) - pen * w
        step = np.linalg.solve(H, g)
        # halve until the penalized deviance stops increasing
        for _h in range(12):
            w_new = w + step
            dev_new = _pen_deviance(X, y, w_new, pen)
            if np.isfinite(dev_new) and dev_new <= dev + 1e-9:
                break
            step *= 0.5
        else:
            break
        if np.max(np.abs(step)) < 1e-8:
            w = w_new
            break
        w, dev = w_new, dev_new
    return w


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--eval-block", type=int, default=50,
                    help="refit every N attempts, predict the next N (default 50)")
    ap.add_argument("--warmup", type=int, default=120,
                    help="attempts before the first evaluation (default 120)")
    ap.add_argument("--pool-every", type=int, default=5,
                    help="run the temperature-sweep pool at every Nth eval attempt")
    ap.add_argument("--temperatures", default="0,0.25,0.5,1,2",
                    help="comma-separated T values for the selection sweep")
    ap.add_argument("--json", help="write full per-user results to this JSON file")
    args = ap.parse_args()

    temps = [float(t) for t in args.temperatures.split(",")]

    emb, prank, quality, size_of, K = load_problem_data()
    lam_axis = np.asarray(json.load(open(POP_CONSTANTS))["weakness_axis_lambda"], dtype=float)
    assert len(lam_axis) == K, f"axis lambda K mismatch ({len(lam_axis)} vs {K})"
    print(f"corpus: {len(emb)} problems with embeddings, K={K}")

    rows = load_attempts(args.db)
    print(f"reports: {len(rows)} novel move-carrying rows")
    picked, cell_mean, dropped = resolve_picked_embeddings(rows, emb, size_of, K)
    print(f"picked-move embeddings resolved: {len(picked)} distinct (problem, move); "
          f"dropped: {dropped['pass']} pass, {dropped['no_cell']} unmapped cell, "
          f"{dropped['no_npz']} missing npz")

    # Assemble per-user matrices.
    by_user = defaultdict(list)
    n_drop_emb = n_drop_pick = 0
    for user, problem, success, created, size, move in rows:
        if problem not in emb:
            n_drop_emb += 1
            continue
        if success:
            pvec = emb[problem]
        else:
            pvec = picked.get((problem, move))
            if pvec is None:
                n_drop_pick += 1
                continue
        by_user[user].append((problem, int(bool(success)), pvec, size))
    print(f"dropped rows: {n_drop_emb} no correct-emb (left corpus), "
          f"{n_drop_pick} wrong-attempt move unresolved")

    # corpus pools per size, sorted by prank (for the temperature sweep)
    pool = {}
    for size in (9, 13):
        pids = [i for i in emb if size_of[i] == size]
        pr = np.array([prank[i] for i in pids])
        o = np.argsort(pr)
        pool[size] = (
            np.array(pids)[o],
            pr[o],
            np.stack([emb[i] for i in np.array(pids)[o]]),
            np.array([quality[i] for i in np.array(pids)[o]]),
        )

    pen = np.concatenate([[0.0, 0.0], lam_axis, lam_axis])  # alpha,gamma free; both beta slots
    pen_base = np.array([0.0, 0.0])

    per_user = []
    pooled = defaultdict(list)
    sweep_acc = {t: defaultdict(list) for t in temps}
    align_acc = defaultdict(list)

    eligible = {u: a for u, a in by_user.items() if len(a) >= args.warmup + args.eval_block}
    print(f"users evaluated: {len(eligible)} (>= {args.warmup + args.eval_block} usable attempts) "
          f"of {len(by_user)} with any usable attempt\n")

    for user, attempts in sorted(eligible.items(), key=lambda kv: -len(kv[1])):
        n = len(attempts)
        tids = [a[0] for a in attempts]
        correct_E = np.stack([emb[a[0]] for a in attempts])
        picked_E = np.stack([a[2] for a in attempts])
        y = np.array([a[1] for a in attempts], dtype=float)
        pr_v = np.array([prank[a[0]] for a in attempts])
        sizes = np.array([a[3] for a in attempts])
        X = np.column_stack([np.ones(n), pr_v, correct_E, picked_E])
        Xb = np.column_stack([np.ones(n), pr_v])

        u_ll_base, u_ll_full = [], []
        u_pbase, u_pfull, u_y = [], [], []
        u_wc, u_ws = [], []
        w_full = w_base = None
        for t0 in range(args.warmup, n, args.eval_block):
            lo = max(0, t0 - MODEL_INPUT_HISTORY_LENGTH)
            if t0 - lo < MIN_TRAINING_ATTEMPTS:
                continue
            hi = min(t0 + args.eval_block, n)
            w_full = irls(X[lo:t0], y[lo:t0], pen, w_full)
            w_base = irls(Xb[lo:t0], y[lo:t0], pen_base, w_base)
            beta_c, beta_p = w_full[2:2 + K], w_full[2 + K:]
            # PRE-ATTEMPT prediction: the level-1 candidate view (picked = correct).
            # Never score the eval row with its own picked_emb — that leaks the label
            # (picked != correct iff the attempt is wrong); see the module docstring.
            z_pre = np.clip(w_full[0] + w_full[1] * pr_v[t0:hi]
                            + correct_E[t0:hi] @ (beta_c + beta_p), -35, 35)
            p_full = 1 / (1 + np.exp(-z_pre))
            p_base = 1 / (1 + np.exp(-np.clip(Xb[t0:hi] @ w_base, -35, 35)))
            # shipped weaknessScore: (beta_picked - beta_correct) . correct_emb
            wc = correct_E[t0:hi] @ (beta_p - beta_c)
            # pre-2026-07-10 sum form: -(beta_correct + beta_picked) . correct_emb
            ws = -(correct_E[t0:hi] @ (beta_c + beta_p))
            u_ll_base.append(logloss(y[t0:hi], p_base))
            u_ll_full.append(logloss(y[t0:hi], p_full))
            u_pbase.append(p_base); u_pfull.append(p_full); u_y.append(y[t0:hi])
            u_wc.append(wc); u_ws.append(ws)

            # picked-side structure on the held-out WRONG attempts of this block
            u_contrast = beta_p - beta_c
            u_over = -beta_p
            nc, no_ = np.linalg.norm(u_contrast), np.linalg.norm(u_over)
            for j in range(t0, hi):
                if y[j] == 1:
                    continue
                e = correct_E[j]
                d = picked_E[j] - e
                nd, ne = np.linalg.norm(d), np.linalg.norm(e)
                if nd > 1e-6 and ne > 1e-6:
                    align_acc["generic"].append(float(d @ (-e) / (nd * ne)))
                base = cell_mean.get(tids[j])
                if base is None:
                    continue
                q = picked_E[j] - base
                nq = np.linalg.norm(q)
                if nq < 1e-6:
                    continue
                if no_ > 1e-6:
                    align_acc["overplay"].append(float(q @ u_over / (nq * no_)))
                if nc > 1e-6:
                    align_acc["contrast"].append(float(q @ u_contrast / (nq * nc)))

            # temperature sweep on the counterfactual candidate pool
            for j in range(t0, hi, args.pool_every):
                pids, prv, pemb, pq = pool[sizes[j]]
                jj = np.searchsorted(prv, pr_v[j])
                lo2, hi2 = max(0, jj - POOL_N), min(len(prv), jj + POOL_N)
                cand = np.arange(lo2, hi2)
                cand = cand[np.argsort(np.abs(prv[cand] - pr_v[j]))[:POOL_N]]
                ce = pemb[cand]
                score = ce @ (beta_p - beta_c)
                sd = score.std() or 1.0
                # candidate predicted success with picked = correct (the level-1 view)
                logits = w_full[0] + w_full[1] * prv[cand] + ce @ (beta_c + beta_p)
                psucc = 1 / (1 + np.exp(-logits))
                qc = pq[cand]
                for T in temps:
                    wgt = qc * np.exp(T * score)
                    if wgt.sum() <= 0:
                        continue
                    wgt = wgt / wgt.sum()
                    sweep_acc[T]["score_sd"].append((wgt @ score - score.mean()) / sd)
                    sweep_acc[T]["psucc"].append(wgt @ psucc)

        if not u_y:
            continue
        ll_b = np.concatenate(u_ll_base); ll_f = np.concatenate(u_ll_full)
        yy = np.concatenate(u_y); pb = np.concatenate(u_pbase); pf = np.concatenate(u_pfull)
        wc = np.concatenate(u_wc); ws = np.concatenate(u_ws)
        per_user.append({
            "user": user, "n_eval": len(yy),
            "ll_base": float(ll_b.mean()), "ll_full": float(ll_f.mean()),
            "dll": float((ll_b - ll_f).mean()),
            "auc_base": float(auc(yy, pb)), "auc_full": float(auc(yy, pf)),
        })
        pooled["y"].append(yy); pooled["pb"].append(pb); pooled["pf"].append(pf)
        pooled["llb"].append(ll_b); pooled["llf"].append(ll_f)
        pooled["wc"].append(wc); pooled["ws"].append(ws)

    # ---------------- 1. forward predictive power ---------------------------------
    y = np.concatenate(pooled["y"]); pb = np.concatenate(pooled["pb"]); pf = np.concatenate(pooled["pf"])
    llb = np.concatenate(pooled["llb"]); llf = np.concatenate(pooled["llf"])
    wc = np.concatenate(pooled["wc"]); ws = np.concatenate(pooled["ws"])
    dll = llb - llf
    # user-level bootstrap for the pooled delta
    rng = np.random.default_rng(0)
    peruser_dll = np.array([u["dll"] for u in per_user])
    peruser_n = np.array([u["n_eval"] for u in per_user])
    boots = [np.average(peruser_dll[idx], weights=peruser_n[idx])
             for idx in (rng.integers(0, len(peruser_dll), len(peruser_dll)) for _ in range(2000))]
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])

    print("=== 1. forward predictive power (held-out, walk-forward) ===")
    print(f"eval attempts: {len(y)}  users: {len(per_user)}")
    print(f"log-loss  baseline {llb.mean():.4f}  full {llf.mean():.4f}  "
          f"delta {dll.mean():+.4f} [{lo_ci:+.4f}, {hi_ci:+.4f}] (user bootstrap)")
    print(f"AUC       baseline {auc(y, pb):.4f}  full {auc(y, pf):.4f}")
    frac_users = (peruser_dll > 0).mean()
    print(f"users with positive delta: {100 * frac_users:.0f}%")
    print("\nper-user (top 15 by eval n):")
    print(f"{'user':<12} {'n_eval':>6} {'dLL':>8} {'AUC base':>9} {'AUC full':>9}")
    for u in sorted(per_user, key=lambda d: -d["n_eval"])[:15]:
        print(f"{u['user']:<12} {u['n_eval']:>6} {u['dll']:>+8.4f} {u['auc_base']:>9.3f} {u['auc_full']:>9.3f}")

    # calibration: deciles of the predicted weakness logit-shift vs realized
    print("\ncalibration of the weakness logit-shift (predicted vs realized, deciles):")
    pb = np.clip(pb, 1e-9, 1 - 1e-9); pf = np.clip(pf, 1e-9, 1 - 1e-9)
    lb = np.log(pb / (1 - pb)); lf = np.log(pf / (1 - pf))
    shift = lf - lb
    qs = np.quantile(shift, np.linspace(0, 1, 11))
    print(f"{'decile':>6} {'pred shift':>11} {'realized':>9} {'n':>7}")
    for d in range(10):
        m = (shift >= qs[d]) & (shift <= qs[d + 1] if d == 9 else shift < qs[d + 1])
        if m.sum() < 30:
            continue
        # realized shift: logit(observed success) - mean baseline logit
        obs = y[m].mean()
        obs = min(max(obs, 1e-6), 1 - 1e-6)
        realized = np.log(obs / (1 - obs)) - lb[m].mean()
        print(f"{d + 1:>6} {shift[m].mean():>+11.3f} {realized:>+9.3f} {m.sum():>7}")

    # ---------------- 2. selection-score validation --------------------------------
    print("\n=== 2. does the selection score forward-predict underperformance? ===")
    for name, s in (("contrast (shipped)", wc), ("sum (pre-07-10)", ws)):
        Xk = np.column_stack([np.ones(len(y)), s])
        off = lb
        # logistic with offset: fit alpha + kappa*s on top of the baseline logit
        w = np.zeros(2)
        for _ in range(100):
            p = 1 / (1 + np.exp(-(off + Xk @ w)))
            Wd = p * (1 - p) + 1e-9
            H = Xk.T @ (Xk * Wd[:, None])
            g = Xk.T @ (y - p)
            step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
            w += step
            if np.max(np.abs(step)) < 1e-10:
                break
        p = 1 / (1 + np.exp(-(off + Xk @ w)))
        Wd = p * (1 - p) + 1e-9
        se = np.sqrt(np.diag(np.linalg.inv(Xk.T @ (Xk * Wd[:, None]))))[1]
        print(f"  {name:<20} kappa = {w[1]:+.3f} +- {se:.3f}  (z = {w[1] / se:+.1f}; "
              f"-1 = perfectly calibrated penalty, 0 = no signal)")

    # per-user contrast kappa, inverse-variance pooled (the single pooled fit above
    # is dominated by the heaviest users)
    def offset_kappa(yv, lbv, sv):
        Xk = np.column_stack([np.ones(len(yv)), sv])
        w = np.zeros(2)
        for _ in range(100):
            p = 1 / (1 + np.exp(-np.clip(lbv + Xk @ w, -35, 35)))
            Wd = p * (1 - p) + 1e-9
            H = Xk.T @ (Xk * Wd[:, None]) + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, Xk.T @ (yv - p))
            w += step
            if np.max(np.abs(step)) < 1e-10:
                break
        p = 1 / (1 + np.exp(-np.clip(lbv + Xk @ w, -35, 35)))
        Wd = p * (1 - p) + 1e-9
        se = np.sqrt(np.diag(np.linalg.inv(Xk.T @ (Xk * Wd[:, None]) + 1e-9 * np.eye(2))))[1]
        return w[1], se

    ks, ses = [], []
    for i in range(len(per_user)):
        yv = pooled["y"][i]
        if len(yv) < 300:
            continue
        pbv = np.clip(pooled["pb"][i], 1e-9, 1 - 1e-9)
        try:
            kap, se_k = offset_kappa(yv, np.log(pbv / (1 - pbv)), pooled["wc"][i])
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(kap) and np.isfinite(se_k) and se_k > 0:
            ks.append(kap); ses.append(se_k)
    if ks:
        ks = np.array(ks); ses = np.array(ses)
        ivw = (ks / ses**2).sum() / (1 / ses**2).sum()
        ivw_se = np.sqrt(1 / (1 / ses**2).sum())
        print(f"  per-user contrast kappa, IVW-pooled over {len(ks)} users (>=300 eval): "
              f"{ivw:+.3f} +- {ivw_se:.3f} (z = {ivw / ivw_se:+.1f}); "
              f"{100 * (ks < 0).mean():.0f}% of users negative")

    # ---------------- 3. picked-side (contrastive) structure ------------------------
    print("\n=== 3. picked-side structure on held-out wrong attempts (cosines, null = 0) ===")
    for key, label in (
        ("generic", "picked-correct vs -correct_emb (weaknessScore's 1st-order assumption)"),
        ("overplay", "pick (vs mean legal cell) vs trained -beta_picked (overplay dir)"),
        ("contrast", "pick (vs mean legal cell) vs trained beta_picked-beta_correct"),
    ):
        v = np.array(align_acc[key])
        if not len(v):
            print(f"  {label}: no data")
            continue
        se = v.std() / np.sqrt(len(v))
        print(f"  {label}:\n"
              f"    mean cos {v.mean():+.4f} +- {se:.4f} (z = {v.mean() / se:+.1f}, "
              f"n = {len(v)}, {100 * (v > 0).mean():.0f}% positive)")

    # ---------------- 4. temperature sweep -----------------------------------------
    print("\n=== 4. selection formula quality*exp(T*score): induced served composition ===")
    print(f"(counterfactual {POOL_N}-candidate pools at every {args.pool_every}th eval attempt; "
          f"score = shipped contrast; T = {temps})")
    print(f"{'T':>6} {'score shift (pool SD)':>22} {'mean pred success':>18}")
    for T in temps:
        sc = np.array(sweep_acc[T]["score_sd"]); ps = np.array(sweep_acc[T]["psucc"])
        if not len(sc):
            continue
        mark = "  <- shipped" if abs(T - WEAKNESS_TEMPERATURE) < 1e-9 else ""
        print(f"{T:>6.2f} {sc.mean():>22.3f} {ps.mean():>18.3f}{mark}")

    if args.json:
        json.dump({"per_user": per_user,
                   "pooled": {"ll_base": float(llb.mean()), "ll_full": float(llf.mean()),
                              "dll": float(dll.mean()), "dll_ci": [float(lo_ci), float(hi_ci)],
                              "auc_base": float(auc(y, pb)), "auc_full": float(auc(y, pf)),
                              "n_eval": int(len(y))},
                   "alignment": {k: {"mean": float(np.mean(v)), "n": len(v)}
                                 for k, v in align_acc.items() if len(v)},
                   "sweep": {str(T): {"score_sd": float(np.mean(sweep_acc[T]["score_sd"])),
                                      "psucc": float(np.mean(sweep_acc[T]["psucc"]))}
                             for T in temps if len(sweep_acc[T]["score_sd"])}},
                  open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
