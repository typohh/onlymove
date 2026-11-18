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

    A row whose value is absent, non-numeric or non-finite RAISES, as does a file
    with no header: these values are per-problem model parameters, and a problem
    silently missing one is served, rated or packaged on a default nobody fitted.
    """
    out = {}
    if missing_ok and not os.path.exists(csv_path):
        return out
    keep_set = None if keep is None else set(keep)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or value_col not in reader.fieldnames:
            raise ValueError(f"{csv_path}: no '{value_col}' column")
        id_col = reader.fieldnames[0]
        for line, row in enumerate(reader, start=2):
            tid = row[id_col]
            if keep_set is not None and tid not in keep_set:
                continue
            try:
                v = float(row[value_col])
            except (TypeError, ValueError) as e:
                raise ValueError(f"{csv_path}:{line}: {value_col} "
                                 f"{row[value_col]!r} for {tid} is not a number") from e
            if not np.isfinite(v):
                raise ValueError(f"{csv_path}:{line}: {value_col} for {tid} is {v}")
            out[tid] = v
    return out


def load_size_value_maps(ratings_dir, size, *, keep=None, missing_ok=False):
    """(ratings, slopes, floors) {tsumego_id: float} maps for one board size from
    ratings/<size>/ — each CSV's value column is named for what it holds
    (tsumego_ratings' 'rating', tsumego_slopes' 'slope', tsumego_floors' 'floor').

    The three come from ONE fit and must cover the same problems; any id in one
    map but not the others raises rather than leaving a caller to fill the gap."""
    sd = os.path.join(ratings_dir, str(size))
    maps = (
        load_value_map(os.path.join(sd, RATINGS_CSV), "rating", keep=keep, missing_ok=missing_ok),
        load_value_map(os.path.join(sd, SLOPE_CSV), "slope", keep=keep, missing_ok=missing_ok),
        load_value_map(os.path.join(sd, FLOOR_CSV), "floor", keep=keep, missing_ok=missing_ok),
    )
    ids = set(maps[0])
    for name, m in zip(("slope", "floor"), maps[1:]):
        if set(m) != ids:
            odd = sorted(ids ^ set(m))
            raise ValueError(f"{sd}: ratings and {name}s cover different problems "
                             f"({len(odd)} differ, e.g. {', '.join(odd[:5])})")
    return maps


def load_lapse(ratings_dir):
    """The global 4PL ceiling λ (fat-finger / lapse rate) the shipped ratings were
    derived under and the client uses. train_mirt writes it to rating_constants.json
    (the authoritative source — it calibrates λ on its held-aside rows); fit_population_constants only
    reads it THROUGH into population_constants.json. So prefer rating_constants.json,
    read population_constants.json only where rating_constants.json is absent (the
    published pwa/tsumegos/ carries only population_constants.json) — otherwise a re-run of
    train_mirt (fresh λ) followed by map_rating_to_rank would anchor the curve under
    a one-cycle-stale λ still sitting in population_constants.json. fit_user_rating's
    4PL must use THIS λ so the anchor matches the client's rank estimator
    (pwa/js/irt.js LAPSE). Neither file, or a file without a valid λ, raises."""
    for fname in ("rating_constants.json", "population_constants.json"):
        path = os.path.join(ratings_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                v = json.load(f).get("fat_finger_rate")
            if not (isinstance(v, (int, float)) and 0.0 <= v < 1.0):
                raise ValueError(f"{path}: fat_finger_rate {v!r} is not a rate in [0, 1)")
            return float(v)
    raise FileNotFoundError(f"no rating_constants.json or population_constants.json "
                            f"in {ratings_dir} — run train_mirt")


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


def interp_extrap_vec(x, xs, ys):
    """np.interp with linear extrapolation past both ends (vectorized).
    xs strictly ascending, len ≥ 2; x an array."""
    y = np.interp(x, xs, ys)
    sl0 = (ys[1] - ys[0]) / (xs[1] - xs[0])
    sl1 = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    y = np.where(x < xs[0], ys[0] + sl0 * (x - xs[0]), y)
    y = np.where(x > xs[-1], ys[-1] + sl1 * (x - xs[-1]), y)
    return y


def interp_extrap(x, xs, ys):
    """Scalar linear interp with end-slope extrapolation; mirrors the PWA's
    rank-decomposition.js interpExtrap so client and offline tools agree exactly.
    xs ascending, len ≥ 2; x a scalar; xs/ys may be plain lists or arrays."""
    n = len(xs)
    if x <= xs[0]:
        return ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0])
    if x >= xs[n - 1]:
        return ys[n - 1] + (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]) * (x - xs[n - 1])
    lo, hi = 0, n - 1
    while hi - lo > 1:
        m = (lo + hi) >> 1
        if xs[m] <= x:
            lo = m
        else:
            hi = m
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


def load_M(ratings_dir, *, size="9"):
    """The M curve's knots, SORTED: (rating_knots_asc, internal_rank_asc), ready for
    interp_extrap_vec. Same file as load_rating_rank_curve; this adds only the sort
    the interpolators require, and both board keys are identical so `size` rarely
    matters. `M` is the rating→latent-consensus-rank map (pwa/shared/rank-math.js
    applyM); every server rank is a coupling off it, not a second curve."""
    ratings, internal = load_rating_rank_curve(ratings_dir, size=size)
    order = np.argsort(ratings)
    return ratings[order], internal[order]


def apply_M(rating, m_ratings, m_internal):
    """rating → internal rank scalar, through the knots load_M returns."""
    return float(interp_extrap_vec(np.array([rating]), m_ratings, m_internal)[0])


def load_rank_map(ratings_dir):
    """{server: (our_knots_asc, server_knots_asc)} from ratings/rank_map.json — the
    per-server couplings off the internal consensus scale, sorted ascending in `our`
    so either direction can be interpolated. Pair with internal_to_server (forward,
    what we would show a user on that server) or declared_to_internal (inverse, what
    a user declaring a rank there is claiming in internal units)."""
    with open(os.path.join(ratings_dir, "rank_map.json")) as f:
        rm = json.load(f)
    out = {}
    for server, v in rm.items():
        our = np.asarray(v["our"], dtype=float)
        srv = np.asarray(v["server"], dtype=float)
        order = np.argsort(our)
        out[server] = (our[order], srv[order])
    return out


def internal_to_server(server, internal_scalar, rank_map):
    """internal rank scalar → that server's own rank scalar, UNCLAMPED (the ladder
    bounds a label, not a value — mirroring rank-translation.js
    internalRankToServerRank).

    An unknown or degenerate coupling yields NaN, deliberately NOT the identity:
    identity would report the raw internal rank AS a server rank, which is silently
    wrong and indistinguishable from a real result in a plot. The inverse direction
    (declared_to_internal) DOES fall back to identity, because there an unrecognised
    server's declaration is already on the hub scale."""
    c = rank_map.get(server)
    if not c or len(c[0]) < 2:
        return float("nan")
    our, srv = c
    return float(interp_extrap_vec(np.array([internal_scalar]), our, srv)[0])


def declared_to_internal(server, declared_scalar, rank_map):
    """Invert a coupling: a declared SERVER rank scalar → internal rank scalar,
    interpolating through the {server → our} knots (server is the x-axis here). An
    unknown server is the identity hub. `srv` is ascending because it was sorted
    with `our` and the fit is monotone."""
    if server not in rank_map:
        return float(declared_scalar)
    our, srv = rank_map[server]
    return float(interp_extrap_vec(np.array([declared_scalar]), srv, our)[0])


def load_time_adjust(ratings_dir):
    """(alpha, center) from the shipped population_constants.json, or None — the
    log-time dock alpha·(log10 median solve minutes − center) that map_rating_to_rank
    fits and pwa/js/population-constants.js applies. THESE TWO KEYS ARE THE SHIPPED
    FORM: a reader gating on any other field silently gets no dock, which lands a
    raw rating in a comparison that must be time-adjusted (see the consensus-mapping
    note in CLAUDE.md)."""
    with open(os.path.join(ratings_dir, "population_constants.json")) as f:
        ta = json.load(f).get("time_adjust")
    if not ta:
        return None
    return (float(ta["alpha_per_log10_min"]), float(ta["center_log10_min"]))


def read_problem_params(path):
    """The published per-problem parameter table (`pwa/tsumegos/problem_params.bin`):
    16-byte rows of a 4-byte id (the 8-hex tsumego id) followed by little-endian
    float32 (rating, slope, floor), the layout `package_for_pwa.emit_problem_params`
    writes and the Worker and client read into one Map. Returns
    {"id": [hex, ...], "rating": f32[], "slope": f32[], "floor": f32[]}. Raises on a
    missing file or a length that is not a whole number of rows."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {path} — run pipeline.package_for_pwa")
    dt = np.dtype([("id", "V4"), ("rating", "<f4"), ("slope", "<f4"), ("floor", "<f4")])
    size = os.path.getsize(path)
    if size % dt.itemsize:
        raise ValueError(f"{path}: {size} bytes is not a whole number of 16-byte rows")
    arr = np.fromfile(path, dtype=dt)
    raw = arr["id"].tobytes()
    ids = [raw[i:i + 4].hex() for i in range(0, len(raw), 4)]
    return {"id": ids, "rating": arr["rating"].astype(np.float64),
            "slope": arr["slope"].astype(np.float64), "floor": arr["floor"].astype(np.float64)}


def load_declaration_exclusions(ratings_dir):
    """{(user_prefix8, server, declared_band)} — the declarations `map_rating_to_rank`
    screened out before fitting, from ratings/declaration_exclusions.csv.

    Anything scoring the mapping against the same survey has to drop them too: a
    declaration the curve was not fitted to, scored against that curve, measures
    how far an excluded row sits from a fit that excluded it. Returns an empty set
    when the file is absent, so a directory written before screening still reads."""
    path = os.path.join(ratings_dir, "declaration_exclusions.csv")
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["user_prefix"], r["server"], int(r["declared_band"]))
                for r in csv.DictReader(f)}
