"""Local active-learning labeler for tsumego motifs.

You tick human motif labels on positions; an elastic-net logistic probe
(multi-label, one-vs-rest) is trained on them and pre-fills its predictions on the
next position so you only correct. It retrains in the background after every save
(coalescing) and re-ranks the queue by prediction uncertainty, so as the model
improves you spend your attention where it's unsure.

The probe reads THE MIRT'S OWN per-problem features (precompute_features.py), which
is what makes a labeled motif convertible into an `axes/axis_bank.npz` column: the
probe is a function of exactly what the trunk reads, so a direction it finds is one
the trunk can represent.

Why elastic net (not ridge, not pure L1): a motif is carried by a small set of the
feature columns, so a sparse prior is statistically efficient; but those columns are
correlated, and pure L1 would flip which correlated column it keeps run-to-run —
jittering your suggestions across the every-save retrain. Elastic net's L2 term
groups correlated features and stabilizes selection. Interpretability isn't the goal
here (labels are supervised); statistical efficiency + stability is.

The pool cache is CHECKED, not trusted: the server refuses to start when the
feature space it was built in no longer matches the artifacts on disk, since a
probe fitted on stale numbers keeps fitting and predicting and nothing downstream
can notice. A problem whose board the shards no longer hold is skipped rather than
served — labels for it stay in labels.json and remain training evidence, because
a vanished board costs the picture, not the feature row.

Note the features carry difficulty — the rating is fit from them. A label that
correlates with difficulty ("end game" being easier, say) will be partly predicted by
that rather than by its motif content, which does no harm to the suggestions but is
worth checking before a probe becomes a bank column: the MIRT's factor 1 already
carries rating, so an axis that is mostly rating teaches the fit nothing new.

Run:
  ./.venv/bin/python3 -m auxiliary.labeler.precompute_features --pool 5000
  ./.venv/bin/python3 -m auxiliary.labeler.server         # http://localhost:8899
"""

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import zlib
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.labeler import render  # noqa: E402
from auxiliary.labeler.fingerprint import (  # noqa: E402
    describe_mismatch, feature_fingerprint)
from utils.rating.model_constants import PCA_DIM, SEED, SIZES  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
LABELS_PATH = os.path.join(HERE, "labels.json")

# Probe hyperparameters. A fresh label has a handful of examples against a feature
# count in the hundreds, so regularization is load-bearing whatever the space. Two
# failure modes shaped these defaults, both structural rather than scale-specific:
#   - A fixed strong C zeroes every coefficient → prob stuck at the base rate, and
#     the tool stops suggesting anything.
#   - CV by neg_log_loss on the tiny per-class sets we actually have (masking +
#     focus keep each label's training set small) reliably PICKS that degenerate C,
#     because a base-rate model never takes a confident-wrong log-loss hit.
# So: a fixed cold C in the small-n regime, and only CV (AUC-scored, so separation
# is what wins, degenerate strengths dropped from the grid) once a label has enough
# per-class data to trust folds. The values below were tuned against the tool's
# previous feature space; they behave, but re-check the spread of the printed
# probabilities if suggestions ever look flat.
MIN_POS = 5          # positives needed before a label switches from kNN to logistic
L1_RATIO = 0.5       # elastic-net mix (0 = ridge, 1 = lasso)
LOG_THRESH = 0.5     # logistic prob >= this counts as "predicted on"
C_COLD = 0.3         # fixed inverse-strength in the small-n regime
C_GRID = [0.15, 0.4, 1.0]   # CV search grid (degenerate <0.1 deliberately excluded)
CV_MIN_CLASS = 15    # minority-class count before CV is trustworthy enough to use
MAX_ITER = 3000
KNN_SUGGEST = 0.7    # cold-start: pre-check a label if pseudo-prob >= this
EPS = 1e-6

# ---- shared state ------------------------------------------------------
F = None             # (N, D) float32 MIRT features, column-standardized
POOL = None          # manifest dict from pool.json
TID2ROW = {}
FN = None            # row-normalized F, for kNN cosine (lazy)
CURVE = None         # rating->rank curve, per board size

STATE_LOCK = threading.Lock()
LABEL_SET = []       # ordered list of label names
ANN = {}             # tid -> {"labels": [...], "known": [...], "ts": float}

MODEL_LOCK = threading.Lock()
MODEL = {"version": 0, "pool_prob": {}, "mode": {}, "pos_counts": {},
         "acquisition": None, "pool_entropy": {}, "label_uncertainty": {}}
DIRTY = threading.Event()
TRAINING = threading.Event()
RECENT = deque(maxlen=30)
UNFETCHABLE = set()  # pool tids whose SGF the shards no longer hold (see render_board)


# ---- persistence -------------------------------------------------------

def load_store():
    global LABEL_SET, ANN
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            data = json.load(f)
        LABEL_SET = data.get("label_set", [])
        ANN = data.get("annotations", {})
    print(f"  loaded {len(ANN)} annotations, {len(LABEL_SET)} labels")


def save_store():
    tmp = LABELS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"label_set": LABEL_SET, "annotations": ANN}, f, indent=2)
    os.replace(tmp, LABELS_PATH)


REBUILD_HINT = ("./.venv/bin/python3 -m auxiliary.labeler.precompute_features "
                "--pool 5000")


def _assert_cache_current(full):
    """REFUSE TO SERVE a pool that was built in a different feature space.

    Every block of load_joint_mirt_features is pooled z-scored over the whole
    corpus and the activation block is a joint PCA over it, so a corpus top-up, a
    re-extraction or a new block rewrites every number in features.npy while
    leaving its shape alone in most cases and merely widening it in the rest. A
    probe fitted on the stale numbers keeps fitting and keeps predicting, so
    nothing downstream can notice — and the whole reason the labeler probes THIS
    space is that a direction found here is one the trunk can represent. Fitted in
    last week's space, it is not.

    labels.json is untouched by any of this: the annotations are keyed by problem
    id, not by feature row, so a rebuild costs only the precompute.
    """
    want = POOL.get("feature_fingerprint")
    have = feature_fingerprint(SIZES, PCA_DIM, SEED)
    if want is None:
        raise SystemExit(
            f"POOL CACHE PREDATES THE FINGERPRINT CHECK and cannot be verified "
            f"against the current feature space. Rebuild it:\n  {REBUILD_HINT}")
    if full.shape[1] != POOL.get("fdim"):
        raise SystemExit(
            f"POOL CACHE IS INTERNALLY INCONSISTENT: features.npy is "
            f"{full.shape[1]} columns wide, pool.json records fdim "
            f"{POOL.get('fdim')}. Rebuild it:\n  {REBUILD_HINT}")
    diff = describe_mismatch(want, have)
    if diff:
        raise SystemExit(
            "POOL CACHE IS STALE — it was built in a different feature space than "
            "the one utils.rating.features assembles now:\n"
            + "\n".join(diff)
            + f"\n\nThe probes would be fitted on numbers the MIRT trunk no longer "
              f"reads, so a direction found in them need not be one the trunk can "
              f"represent. Your labels are keyed by problem id and are NOT affected "
              f"— rebuild the pool and they are pinned straight back in:\n"
              f"  {REBUILD_HINT}")


def load_cache():
    global F, POOL, TID2ROW, FN
    feats_path = os.path.join(CACHE_DIR, "features.npy")
    pool_path = os.path.join(CACHE_DIR, "pool.json")
    if not (os.path.exists(feats_path) and os.path.exists(pool_path)):
        raise SystemExit("missing feature cache; run "
                         "`python -m auxiliary.labeler.precompute_features` first")
    full = np.load(feats_path)
    with open(pool_path) as f:
        POOL = json.load(f)
    _assert_cache_current(full)
    TID2ROW = {tid: i for i, tid in enumerate(POOL["tids"])}
    # Column-standardize, and DON'T reduce. The MIRT features are already a compact
    # trunk input (its activation block is a joint PCA), so there is nothing to
    # compress and a second projection would only put the probe's coefficients in a
    # basis the bank export has to undo. Standardizing matters because the blocks
    # arrive on different scales — a shared elastic-net penalty on unequal scales
    # silently prefers whichever block happens to be largest.
    F = full.astype(np.float32)
    del full
    F = (F - F.mean(0)) / np.maximum(F.std(0), EPS)
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    FN = F / np.maximum(norms, EPS)
    global CURVE
    CURVE = render.load_curve()
    print(f"  feature cache: {F.shape} MIRT features "
          f"(pca_dim {POOL['pca_dim']}, seed {POOL['seed']})")


# ---- training ----------------------------------------------------------

def _entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _fit_logistic(X, y):
    """Elastic-net logistic. CV the strength when the minority class is large
    enough for stable folds; otherwise use the fixed cold C."""
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    min_class = min(int(np.sum(y)), int(len(y) - np.sum(y)))
    if min_class >= CV_MIN_CLASS:
        # AUC (ranking) scoring, not log-loss: with limited data log-loss
        # rewards the degenerate base-rate model; AUC rewards separation.
        clf = LogisticRegressionCV(
            Cs=C_GRID, cv=min(4, min_class), penalty="elasticnet", solver="saga",
            l1_ratios=[L1_RATIO], scoring="roc_auc", max_iter=MAX_ITER, tol=1e-3,
        )
    else:
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=L1_RATIO,
            C=C_COLD, max_iter=MAX_ITER, tol=1e-3,
        )
    clf.fit(X, y)
    return clf


def _acquisition(pool_prob, ann):
    """Per-position uncertainty and the weighted acquisition score, derived from
    predictions plus what has already been considered.

    Split out of the fit because it is CHEAP and depends on the annotations,
    while the fit is expensive and depends on the labels: a model restored from
    disk re-derives this against the current annotations instead of trusting a
    stored copy that the last save may already have invalidated.
    """
    N = F.shape[0]
    # Pool rows already annotated for a label (label was in-scope at annotation
    # time) — excluded from acquisition since labeling them again adds nothing.
    known_rows = {}
    for tid, rec in ann.items():
        r = TID2ROW.get(tid)
        if r is None:
            continue
        for label in rec["known"]:
            known_rows.setdefault(label, set()).add(r)

    pool_entropy, label_uncertainty = {}, {}
    for label, prob in pool_prob.items():
        ent = _entropy(prob)
        kr = known_rows.get(label)
        if kr:
            m = np.ones(N, dtype=bool)
            m[list(kr)] = False
            ent = ent * m
        pool_entropy[label] = ent                       # per-position, masked
        label_uncertainty[label] = float(ent.sum())     # total remaining for the label

    # Weighted acquisition: scale each label's per-position uncertainty by that
    # label's TOTAL remaining uncertainty, so an uncertain position in a broadly
    # unresolved label outranks an equally-uncertain position in a label the
    # model already nails almost everywhere.
    acq = np.zeros(N)
    for label, ent in pool_entropy.items():
        acq += label_uncertainty[label] * ent
    return pool_entropy, label_uncertainty, acq


def train():
    """Fit one probe per label on the current annotations and recompute pool
    predictions + acquisition. Returns a fresh model dict (atomically swapped)."""

    with STATE_LOCK:
        ann = {k: dict(v) for k, v in ANN.items()}
        label_set = list(LABEL_SET)
        version = MODEL["version"] + 1

    pool_prob, mode, pos_counts = {}, {}, {}
    for label in label_set:
        rows, y = [], []
        for tid, rec in ann.items():
            r = TID2ROW.get(tid)
            if r is None or label not in rec["known"]:
                continue  # not in pool, or masked (label post-dates this annotation)
            rows.append(r)
            y.append(1 if label in rec["labels"] else 0)
        pos = int(sum(y))
        pos_counts[label] = pos

        if pos >= MIN_POS and (len(y) - pos) >= 1:
            clf = _fit_logistic(F[rows], np.array(y))
            pool_prob[label] = clf.predict_proba(F)[:, 1]
            mode[label] = "logistic"
        elif pos >= 1:
            pos_rows = [r for r, yy in zip(rows, y) if yy == 1]
            cen = F[pos_rows].mean(axis=0)
            cen = cen / max(np.linalg.norm(cen), EPS)
            sim = FN @ cen
            lo, hi = np.percentile(sim, [10, 99])
            pool_prob[label] = np.clip((sim - lo) / max(hi - lo, EPS), 0, 1)
            mode[label] = "knn"
        else:
            mode[label] = None

    pool_entropy, label_uncertainty, acq = _acquisition(pool_prob, ann)
    return {"version": version, "pool_prob": pool_prob, "mode": mode,
            "pos_counts": pos_counts, "acquisition": acq,
            "pool_entropy": pool_entropy, "label_uncertainty": label_uncertainty}


# ---- model persistence -------------------------------------------------

MODEL_NPZ = os.path.join(CACHE_DIR, "model.npz")
MODEL_META = os.path.join(CACHE_DIR, "model.json")


def _pool_fingerprint():
    """The pool's tid list IN ORDER. Predictions are stored as arrays indexed by
    pool ROW, so a reordered pool silently reassigns every prediction to a
    different problem — the one corruption that leaves a well-formed file."""
    return hashlib.sha256("\n".join(POOL["tids"]).encode()).hexdigest()[:16]


def save_model(m):
    """Persist the fitted predictions so a restart serves them immediately.

    Only the PREDICTIONS are stored. Entropy and acquisition are re-derived at
    load from the annotations as they stand then, because a stored copy goes
    stale the moment a label is saved and a stale acquisition silently re-offers
    positions you have already answered.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    arrays = {f"prob::{k}": v for k, v in m["pool_prob"].items()}
    np.savez_compressed(MODEL_NPZ + ".tmp.npz", **arrays)
    os.replace(MODEL_NPZ + ".tmp.npz", MODEL_NPZ)
    with open(MODEL_META + ".tmp", "w") as f:
        json.dump({"pool": _pool_fingerprint(), "version": m["version"],
                   "mode": m["mode"], "pos_counts": m["pos_counts"]}, f)
    os.replace(MODEL_META + ".tmp", MODEL_META)


def load_model():
    """Restore a persisted model, or None. A pool mismatch discards it rather
    than remapping: there is no correct remap, since a pool rebuild changes which
    problems exist as well as their order."""
    if not (os.path.exists(MODEL_NPZ) and os.path.exists(MODEL_META)):
        return None
    try:
        with open(MODEL_META) as f:
            meta = json.load(f)
        if meta.get("pool") != _pool_fingerprint():
            print("  cached model was fitted on a different pool — discarding")
            return None
        z = np.load(MODEL_NPZ, allow_pickle=False)
        pool_prob = {k[len("prob::"):]: z[k] for k in z.files
                     if k.startswith("prob::")}
        n = F.shape[0]
        if any(v.shape != (n,) for v in pool_prob.values()):
            print("  cached model has the wrong row count — discarding")
            return None
    except Exception as e:  # noqa: BLE001 -- a bad cache must never block startup
        print(f"  cached model unreadable ({type(e).__name__}) — discarding")
        return None
    pe, lu, acq = _acquisition(pool_prob, ANN)
    return {"version": int(meta.get("version", 0)), "pool_prob": pool_prob,
            "mode": meta.get("mode", {}), "pos_counts": meta.get("pos_counts", {}),
            "acquisition": acq, "pool_entropy": pe, "label_uncertainty": lu}


def trainer_loop():
    while True:
        DIRTY.wait()
        DIRTY.clear()
        if not ANN:
            continue
        TRAINING.set()
        try:
            new_model = train()
            with MODEL_LOCK:
                global MODEL
                MODEL = new_model
            save_model(new_model)
            print(f"  [trainer] model v{new_model['version']} "
                  f"({sum(1 for m in new_model['mode'].values() if m=='logistic')} logistic, "
                  f"{sum(1 for m in new_model['mode'].values() if m=='knn')} kNN)", flush=True)
        except Exception as e:  # keep the loop alive on a bad fit
            print(f"  [trainer] ERROR: {type(e).__name__}: {e}", flush=True)
        finally:
            TRAINING.clear()


# ---- position payloads -------------------------------------------------

def render_board(tid):
    """The problem's board, or None when it can no longer be fetched.

    The pool records the shard a problem lived in when the pool was built, and
    `package_for_pwa` reshards; a problem can also leave the corpus outright while
    labels.json still carries its annotation. Both surface here as the SGF simply
    not being in the zip. That must not take out the request: a single unfetchable
    position would otherwise 500 `/api/next` and stall labelling entirely, or drop
    a whole grid page. Misses are remembered so the cost is paid once per id.
    """
    if tid in UNFETCHABLE:
        return None
    r = TID2ROW[tid]
    try:
        return render.build(tid, POOL["zip_ids"][r], POOL["sizes"][r],
                            POOL["ratings"][r], CURVE)
    except (FileNotFoundError, KeyError) as e:
        UNFETCHABLE.add(tid)
        print(f"  [render] {tid} unfetchable ({type(e).__name__}) — skipping; "
              f"its labels are kept", flush=True)
        return None


def render_position(tid):
    r = TID2ROW[tid]
    size = POOL["sizes"][r]
    rating = POOL["ratings"][r]
    rendered = render_board(tid)
    if rendered is None:
        return None

    m = MODEL
    predicted = {}
    for label in LABEL_SET:
        prob = m["pool_prob"].get(label)
        predicted[label] = {
            "prob": float(prob[r]) if prob is not None else None,
            "mode": m["mode"].get(label),
        }
    rec = ANN.get(tid)
    return {
        "tid": tid, "size": size, "rating": round(rating, 2),
        "rank": rendered["rank"], "svg": rendered["svg"],
        "to_move": rendered["to_move"], "has_ko": rendered["has_ko"],
        "annotated": rec is not None,
        "current_labels": rec["labels"] if rec else [],
        "known": rec["known"] if rec else [],
        "predicted": predicted, "label_set": list(LABEL_SET),
        "model_version": m["version"],
    }


def _known_of(tid):
    rec = ANN.get(tid)
    return set(rec["known"]) if rec else set()


def iter_next(mode, current, focus):
    """Yield candidate positions in acquisition order. A GENERATOR rather than one
    pick, so the caller can walk past a position whose board no longer renders
    instead of handing the client an error and stopping."""
    m = MODEL
    exclude = set(RECENT) | UNFETCHABLE
    if current:
        exclude.add(current)
    focus = [f for f in (focus or []) if f in LABEL_SET]
    fset = set(focus)

    if focus:
        pe = m.get("pool_entropy") or {}
        lu = m.get("label_uncertainty") or {}
        vecs = [lu.get(l, 0.0) * pe[l] for l in focus if l in pe]
        acq = np.sum(vecs, axis=0) if vecs else None

        def needs(tid):
            return not fset.issubset(_known_of(tid))
    else:
        acq = m["acquisition"]

        def needs(tid):
            return tid not in ANN

    if mode == "uncertain" and acq is not None:
        for r in np.argsort(-acq):
            tid = POOL["tids"][r]
            if tid in exclude:
                continue
            if focus and not needs(tid):
                continue
            yield tid
        return

    # bootstrap / "all" (also the uncertain fallback when no model yet)
    cand = [r for r, tid in enumerate(POOL["tids"])
            if tid not in exclude and needs(tid)]
    if acq is not None:
        cand.sort(key=lambda r: -acq[r])
    for r in cand:
        yield POOL["tids"][r]


def next_position(mode, current, focus):
    """The next renderable position's payload, or None when the queue is empty."""
    for tid in iter_next(mode, current, focus):
        pos = render_position(tid)
        if pos is not None:
            RECENT.append(tid)
            return pos
    return None


def grid_page(label, pred, include_labeled, sort, offset, limit):
    """A page of positions for one label, filtered by predicted on/off.

    'confidence' is the prob of the predicted class (prob for on, 1-prob for
    off); sort 'desc' = most confident first, 'asc' = nearest the boundary.
    Marking a cell incorrect later flips it: on-grid → negative, off-grid →
    positive (the filter fixes the direction)."""
    if label not in LABEL_SET:
        return {"items": [], "total": 0, "has_more": False, "modeled": False,
                "error": "unknown label"}
    m = MODEL
    prob = m["pool_prob"].get(label)
    if prob is None:
        # No model yet: show random positions so a brand-new label can be
        # bootstrapped straight from the grid. Stable shuffle (seeded by label)
        # keeps pagination consistent across pages and restarts. The client
        # forces "predicted OFF", so marking a cell means "this IS the label".
        idx = [r for r in range(len(POOL["tids"]))
               if include_labeled or label not in _known_of(POOL["tids"][r])]
        random.Random(zlib.crc32(label.encode())).shuffle(idx)
        items = _grid_items(idx[offset:offset + limit], label, None, None)
        return {"items": items, "total": len(idx), "has_more": offset + limit < len(idx),
                "modeled": False, "random": True}
    mode = m["mode"].get(label)
    thr = KNN_SUGGEST if mode == "knn" else LOG_THRESH
    on = prob >= thr
    want = on if pred == "on" else ~on
    idx = [int(r) for r in np.where(want)[0]]
    if not include_labeled:
        idx = [r for r in idx if label not in _known_of(POOL["tids"][r])]
    conf = prob if pred == "on" else (1.0 - prob)
    idx.sort(key=lambda r: float(conf[r]), reverse=(sort != "asc"))

    total = len(idx)
    items = _grid_items(idx[offset:offset + limit], label, prob, mode)
    return {"items": items, "total": total, "has_more": offset + limit < total,
            "modeled": True, "threshold": float(thr)}


def _grid_items(rows, label, prob, mode):
    """Render one page's cells, DROPPING any whose board is gone rather than
    failing the page. A short page is a visible, harmless symptom; a 500 loses the
    label's entire review pass."""
    items = []
    for r in rows:
        tid = POOL["tids"][r]
        rendered = render_board(tid)
        if rendered is None:
            continue
        rec = ANN.get(tid)
        items.append({
            "tid": tid, "size": POOL["sizes"][r], "rank": rendered["rank"],
            "svg": rendered["svg"],
            "prob": None if prob is None else float(prob[r]), "mode": mode,
            "considered": label in _known_of(tid),
            "current": (label in rec["labels"]) if rec else None,
        })
    return items


def state_payload():
    m = MODEL
    return {
        "label_set": list(LABEL_SET),
        "pos_counts": m["pos_counts"],
        "modes": m["mode"],
        "label_uncertainty": m.get("label_uncertainty", {}),
        "n_annotated": len(ANN),
        "n_pool": len(POOL["tids"]),
        "n_unfetchable": len(UNFETCHABLE),
        "model_version": m["version"],
        "training": TRAINING.is_set(),
    }


# ---- HTTP --------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html", "/grid", "/grid.html"):
                page = "grid.html" if u.path.startswith("/grid") else "index.html"
                with open(os.path.join(HERE, page), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/state":
                self._json(state_payload())
            elif u.path == "/api/next":
                mode = q.get("mode", ["all"])[0]
                current = q.get("current", [None])[0]
                focus_q = q.get("focus", [""])[0]
                focus = [s for s in focus_q.split(",") if s]
                pos = next_position(mode, current, focus)
                self._json(pos if pos is not None else {"done": True})
            elif u.path == "/api/position":
                tid = q.get("tid", [None])[0]
                pos = render_position(tid) if tid in TID2ROW else None
                if pos is None:
                    # A 200 with `done` rather than a 404 body the client would try
                    # to render as a position: this is reachable from the history
                    # buttons, where the pool moved under a tid you already saw.
                    self._json({"done": True, "unavailable": tid})
                else:
                    self._json(pos)
            elif u.path == "/api/grid":
                self._json(grid_page(
                    q.get("label", [None])[0],
                    q.get("pred", ["on"])[0],
                    q.get("include_labeled", ["0"])[0] in ("1", "true"),
                    q.get("sort", ["desc"])[0],
                    int(q.get("offset", ["0"])[0]),
                    int(q.get("limit", ["24"])[0]),
                ))
            elif u.path == "/api/export":
                self._json({"label_set": LABEL_SET, "annotations": ANN})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._read_body()
            if u.path == "/api/label":
                tid = body["tid"]
                ticked = [l for l in body.get("labels", []) if l in LABEL_SET]
                focus = [f for f in body.get("focus", []) if f in LABEL_SET]
                with STATE_LOCK:
                    old = ANN.get(tid, {"labels": [], "known": []})
                    if focus:
                        # Merge: keep prior labels outside the focus; within the
                        # focus, the current ticks are authoritative. Mark only
                        # the focus labels as now-considered.
                        fset = set(focus)
                        kept = [l for l in old["labels"] if l not in fset]
                        new_labels = kept + [l for l in ticked if l in fset]
                        new_known = sorted(set(old["known"]) | fset)
                    else:
                        new_labels = ticked
                        new_known = list(LABEL_SET)
                    ANN[tid] = {"labels": new_labels, "known": new_known,
                                "ts": time.time()}
                    save_store()
                DIRTY.set()
                self._json({"ok": True})
            elif u.path == "/api/correct":
                # Batch-set one label across many positions (grid review).
                # value=True → label present, False → absent. Merge semantics
                # identical to focused /api/label: only this label is touched.
                label = body.get("label")
                value = bool(body.get("value"))
                tids = body.get("tids", [])
                if label not in LABEL_SET:
                    self._json({"error": "unknown label"}, 400)
                    return
                with STATE_LOCK:
                    for tid in tids:
                        if tid not in TID2ROW:
                            continue
                        old = ANN.get(tid, {"labels": [], "known": []})
                        kept = [l for l in old["labels"] if l != label]
                        new_labels = kept + ([label] if value else [])
                        new_known = sorted(set(old["known"]) | {label})
                        ANN[tid] = {"labels": new_labels, "known": new_known,
                                    "ts": time.time()}
                    save_store()
                DIRTY.set()
                self._json({"ok": True, "n": len(tids)})
            elif u.path == "/api/new_label":
                name = (body.get("name") or "").strip()
                if not name:
                    self._json({"error": "empty name"}, 400)
                    return
                with STATE_LOCK:
                    if name not in LABEL_SET:
                        LABEL_SET.append(name)
                        save_store()
                DIRTY.set()
                self._json({"label_set": list(LABEL_SET)})
            elif u.path == "/api/delete_label":
                name = body.get("name")
                with STATE_LOCK:
                    if name in LABEL_SET:
                        LABEL_SET.remove(name)
                        for rec in ANN.values():
                            if name in rec["labels"]:
                                rec["labels"].remove(name)
                            if name in rec["known"]:
                                rec["known"].remove(name)
                        save_store()
                DIRTY.set()
                self._json({"label_set": list(LABEL_SET)})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()

    print("Loading labeler...")
    load_cache()
    load_store()
    # Serve the last fit IMMEDIATELY, then refresh in the background. Without
    # this the first minutes after every restart have no model at all, and the
    # grid falls back to random positions — which is the least useful thing it
    # can show and looks like a broken control rather than a cold start.
    restored = load_model()
    if restored is not None:
        global MODEL
        MODEL = restored
        print(f"  restored model v{MODEL['version']} from cache "
              f"({sum(1 for m in MODEL['mode'].values() if m)} labels)")
    threading.Thread(target=trainer_loop, daemon=True).start()
    if ANN:
        DIRTY.set()  # refresh against the annotations as they stand now

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"  serving http://127.0.0.1:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    main()
