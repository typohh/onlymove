"""Reader for the per-intersection ownership cache that
pipeline.extract_ownership_maps writes.

The cache exists so a rule written over KataGo ownership — the two
life-and-death/semeai rules, territory, influence, anything new — can be
re-derived and re-thresholded without re-running the engine. It records, per
problem, the ownership map after each of several candidate first moves rather
than one summary per problem, because a motif is often legible only in the
CONTRAST between candidates: the first and second moves sharing a purpose and
differing slightly in execution, against a third move and a pass that do
something else.

Numpy only, so a rating-side reader loads nothing from utils/katago.

LAYOUT, per board size, in OWNERSHIP_MAPS_DIR/<size>/part_<n>.npz. A problem
sits in exactly one part; a reader concatenates them (`load_size`).

  tid        (N,)                 str
  solver     (N,)                 int8   1 = black to play, 2 = white
  stones     (N, S, S)            int8   0 empty, 1 black, 2 white  — the ROOT board
  legal      (N, S, S)            bool   empty points legal for the solver at the root
  own        (N, SLOTS, S, S)     float16 ownership, BLACK-positive, NaN in an unused slot
  move       (N, SLOTS)           int16  y*S + x, or MOVE_PASS / MOVE_NONE
  sgf_wr     (N, SLOTS)           float32 the SGF comment's winrate (0..1), BLACK, NaN if none
  sgf_score  (N, SLOTS)           float32 the SGF comment's score, BLACK, NaN if none
  sgf_visits (N, SLOTS)           int32  visits behind the comment (parse_eval_visits), 0 if none
  resolved   (N, SLOTS)           bool   the root variation's line reaches a conclusion
  var_idx    (N, SLOTS)           int16  the slot's index among the root variations, -1 if none
                                         (a pass with no pass variation, or an unused slot)
  eng_wr     (N, SLOTS)           float32 the engine's own winrate for the slot's position, BLACK
  eng_score  (N, SLOTS)           float32 the engine's scoreLead for it, BLACK
  meta       ()                   str    JSON provenance: model, visits, rules, komi, layout

Every [y][x] grid indexes y from the BOTTOM (the gtp_to_xy convention), so a
move code y*S + x and a grid cell agree with no flip.

EVERYTHING IS BLACK-POSITIVE — ownership, both winrates, both scores — because
that is the frame the stones are stored in and the frame the SGF comments are
written in, so nothing in the file needs a per-row sign to read. A rule almost
always wants the SOLVER's frame instead; take it through `solver_frame` rather
than flipping by hand, since getting the sign wrong inverts every reading with no
other symptom.

SLOTS — every slot is a position AFTER a solver move; the root itself is not
stored, because the engine prices the root as though the answer will be found
and its map is practically the correct child's (measured: a point neutral at the
root reaches 0.95 after the answer on ~1% of problems, against ~21% measured
from a refutation). The "nothing played" reference is the PASS instead:
  SLOT_PASS     the solver passes — always present, whether or not the SGF has a pass variation
  SLOT_CORRECT  the answer, variation_nodes[0] by construction
  SLOT_ALT0..   the next TOP_ALTS stone-move root variations IN SGF ORDER — the
                order the generator wrote its candidates in, which is its own
                ranking of how good they are, so SLOT_ALT0 is the second branch
                and practically always the best alternative. A pass variation is
                skipped here, since it has its own slot.

Not re-ranked by the comment scores: the document order is the generator's
judgement, made with the search it ran, and a re-ranking by a comment written at
whatever budget its pass used is a second, shallower opinion. A slot can still be
an unresolved stub; its `resolved` and `sgf_visits` say how much to trust it, and
a rule that needs a played-out line filters on them.
"""
import glob
import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OWNERSHIP_MAPS_DIR = os.path.join(PROJECT_ROOT, "activations", "_signal_cache", "ownership_maps")

LAYOUT_VERSION = 2
TOP_ALTS = 4                  # alternatives in SGF order; with the answer, the first 5 moves
SLOT_PASS, SLOT_CORRECT = 0, 1
SLOT_ALT0 = 2
SLOTS = SLOT_ALT0 + TOP_ALTS

MOVE_PASS, MOVE_NONE = -1, -3

EMPTY, BLACK, WHITE = 0, 1, 2

ARRAYS = ("tid", "solver", "stones", "legal", "own", "move", "var_idx", "sgf_wr",
          "sgf_score", "sgf_visits", "resolved", "eng_wr", "eng_score")


def size_dir(size, root=None):
    return os.path.join(root or OWNERSHIP_MAPS_DIR, str(size))


def part_paths(size, root=None):
    return sorted(glob.glob(os.path.join(size_dir(size, root), "part_*.npz")))


def load_size(size, root=None, ids=None):
    """Every part of one board size concatenated into one dict of arrays, plus
    `meta` (parsed). `ids`, if given, keeps only those problems and RAISES on any
    that are missing — a reader asking for a problem the cache lacks is a stale
    cache, not a problem to skip."""
    paths = part_paths(size, root)
    if not paths:
        raise FileNotFoundError(f"no ownership-map parts under {size_dir(size, root)} "
                                "— run pipeline.extract_ownership_maps")
    parts, meta = [], None
    for p in paths:
        with np.load(p) as d:
            m = json.loads(str(d["meta"]))
            if meta is None:
                meta = m
            elif m != meta:
                raise ValueError(f"{p}: provenance {m} differs from {meta} — mixed cache")
            parts.append({k: d[k] for k in ARRAYS})
    out = {k: np.concatenate([q[k] for q in parts]) for k in ARRAYS}
    if len(set(out["tid"].tolist())) != len(out["tid"]):
        raise ValueError(f"size {size}: a problem appears in two parts")
    if ids is not None:
        pos = {t: i for i, t in enumerate(out["tid"].tolist())}
        missing = [t for t in ids if t not in pos]
        if missing:
            raise KeyError(f"size {size}: {len(missing)} ids not in the ownership cache "
                           f"(e.g. {missing[:3]}) — re-run pipeline.extract_ownership_maps")
        idx = np.array([pos[t] for t in ids], dtype=np.int64)
        out = {k: v[idx] for k, v in out.items()}
    out["meta"] = meta
    return out


def solver_sign(solver):
    """+1 for a black solver, -1 for white, shaped to broadcast against (N, ...)."""
    return np.where(np.asarray(solver) == BLACK, 1.0, -1.0).astype(np.float32)


def solver_frame(d):
    """Solver-frame views of the BLACK-positive fields, as float32:
      own        (N, SLOTS, S, S)  + = the solver's
      p_own      (N, SLOTS, S, S)  probability the solver owns the point, (own + 1) / 2
      sgf_score, eng_score  (N, SLOTS)  + = good for the solver
      sgf_wr, eng_wr        (N, SLOTS)  the solver's winrate
      mine, theirs (N, S, S) bool  the solver's / the opponent's stones at the root
    """
    s = solver_sign(d["solver"])
    own = d["own"].astype(np.float32) * s[:, None, None, None]
    flip_wr = (s < 0)[:, None]
    return {
        "own": own,
        "p_own": (own + 1.0) / 2.0,
        "sgf_score": d["sgf_score"] * s[:, None],
        "eng_score": d["eng_score"] * s[:, None],
        "sgf_wr": np.where(flip_wr, 1.0 - d["sgf_wr"], d["sgf_wr"]),
        "eng_wr": np.where(flip_wr, 1.0 - d["eng_wr"], d["eng_wr"]),
        "mine": d["stones"] == np.asarray(d["solver"])[:, None, None],
        "theirs": (d["stones"] != EMPTY) & (d["stones"] != np.asarray(d["solver"])[:, None, None]),
    }


def best_resolved_slot(d):
    """(N,) the slot holding the best RESOLVED refutation by SGF score in the
    solver's frame — extract_ownership_swing's baseline — among the pass and the
    cached alternatives, or -1 where none is resolved. Exact only when that
    refutation lies within the first TOP_ALTS alternatives, which SGF order makes
    the usual case."""
    sc = solver_frame(d)["sgf_score"]
    cand = np.zeros_like(d["resolved"])
    cand[:, SLOT_PASS] = True
    cand[:, SLOT_ALT0:] = True
    ok = cand & d["resolved"] & np.isfinite(sc) & (d["move"] != MOVE_NONE)
    best = np.where(ok, sc, -np.inf).argmax(axis=1)
    return np.where(ok.any(axis=1), best, -1)


def move_xy(code, size):
    """A move code -> (x, y), or None for a pass or an unused slot."""
    code = int(code)
    return None if code < 0 else (code % size, code // size)
