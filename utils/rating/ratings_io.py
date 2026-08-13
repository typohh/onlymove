"""Shared readers for the backend rating artifacts under ratings/<size>/ and the
ratings/rating_to_rank.json curve.

These were copy-pasted across the pipeline (map_rating_to_rank,
fit_population_constants, package_for_pwa) and the auxiliary rank tools — the same
`{tsumego_id: float}` CSV load and the same rating_to_rank.json knot extraction in
seven-plus places. Centralizing them keeps a column rename or curve-shape change
from drifting between consumers. Pure stdlib + numpy (no pandas) so even the
csv-based pipeline steps can share it without pulling in a heavy dependency.
"""

import csv
import json
import os

import numpy as np

# Backend per-size CSVs (written by pipeline.train_mirt, keyed by tsumego_id).
RATINGS_CSV = "tsumego_ratings.csv"   # value col 'rating' — 80%-clear b_p
SLOPE_CSV = "tsumego_slopes.csv"      # value col 'slope'  — 2PL discrimination a_p
FLOOR_CSV = "tsumego_floors.csv"      # value col 'floor'  — 3PL guessing floor c_p


def load_value_map(csv_path, value_col, *, keep=None, missing_ok=False):
    """{tsumego_id: float(value_col)} from a ratings CSV, keyed by its first column
    (conventionally 'tsumego_id').

    keep: optional id allowlist — ids outside it are dropped.
    missing_ok: if True a missing file yields {} instead of raising.
    Rows whose value_col is absent or non-numeric are skipped.
    """
    out = {}
    if missing_ok and not os.path.exists(csv_path):
        return out
    keep_set = None if keep is None else set(keep)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out  # existing-but-empty CSV (no header) -> no rows
        id_col = reader.fieldnames[0]
        for row in reader:
            tid = row[id_col]
            if keep_set is not None and tid not in keep_set:
                continue
            try:
                out[tid] = float(row[value_col])
            except (TypeError, ValueError, KeyError):
                pass
    return out


def load_size_value_maps(ratings_dir, size, *, keep=None, missing_ok=False):
    """(ratings, slopes, floors) {tsumego_id: float} maps for one board size from
    ratings/<size>/ — each CSV's value column is named for what it holds
    (tsumego_ratings' 'rating', tsumego_slopes' 'slope', tsumego_floors' 'floor')."""
    sd = os.path.join(ratings_dir, str(size))
    return (
        load_value_map(os.path.join(sd, RATINGS_CSV), "rating", keep=keep, missing_ok=missing_ok),
        load_value_map(os.path.join(sd, SLOPE_CSV), "slope", keep=keep, missing_ok=missing_ok),
        load_value_map(os.path.join(sd, FLOOR_CSV), "floor", keep=keep, missing_ok=missing_ok),
    )


def load_lapse(ratings_dir):
    """The global 4PL ceiling λ (fat-finger / lapse rate) the shipped ratings were
    derived under and the client uses. train_mirt writes it to rating_constants.json
    (the authoritative source — it learns λ jointly); fit_population_constants only
    reads it THROUGH into population_constants.json. So prefer rating_constants.json,
    fall back to population_constants.json, then 0.0 — otherwise a re-run of
    train_mirt (fresh λ) followed by map_rating_to_rank would anchor the curve under
    a one-cycle-stale λ still sitting in population_constants.json. fit_user_rating's
    4PL must use THIS λ so the anchor matches the client's rank estimator
    (pwa/js/irt.js LAPSE)."""
    for fname in ("rating_constants.json", "population_constants.json"):
        path = os.path.join(ratings_dir, fname)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    v = json.load(f).get("fat_finger_rate")
                if isinstance(v, (int, float)) and 0.0 <= v < 1.0:
                    return float(v)
            except (ValueError, OSError):
                pass
    return 0.0


def load_rating_rank_curve(ratings_dir, *, size="9"):
    """(rating_knots, rank_knots) float arrays from the board-`size` key of
    ratings/rating_to_rank.json. The v2 curve writes both board keys identical, so
    the default '9' stands in for either (falling back to whatever key is present).
    Ranks are on the project scale (pwa/shared/rank-math.js) — there is no second
    coordinate to convert to any more."""
    with open(os.path.join(ratings_dir, "rating_to_rank.json")) as f:
        rr = json.load(f)
    c = rr.get(str(size))
    if c is None:
        # Fall back to any board-size curve — but never the top-level
        # time_adjust dict, which has no ratings/ranks keys.
        c = next((v for v in rr.values()
                  if isinstance(v, dict) and "ratings" in v), None)
    if c is None:
        raise KeyError(f"rating_to_rank.json has no curve for size {size!r}")
    xs = np.asarray(c["ratings"], dtype=float)
    ys = np.asarray(c["ranks"], dtype=float)
    return xs, ys
