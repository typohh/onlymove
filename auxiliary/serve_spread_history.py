"""How wrong were the ratings of each era, measured against today's ratings.

The selector serves the ~50 corpus problems nearest the user's serve rank (a
weighted draw over that pool, `selectChallengeCandidate` in problem-loader.js),
so two adjacent NOVEL problems were, in the rating frame that was live at the
time, near-identical in difficulty. Re-score that same pair with TODAY's ratings
and whatever gap appears is mostly the era's rating ERROR: the pair was chosen
to be equal and only a retroactive re-rating could have pulled it apart.

    Δ_today  =  (err_A − err_B)  +  (design spread in the era's own frame)

with err = today's rating − the era's rating for that problem. The two errors are
independent draws, so the robust scale of Δ over many pairs gives

    σ_err  =  sqrt( max(0, sd(Δ)² − sd(Δ)²_floor) / 2 )

where the floor is the design spread — whatever survives even under perfect
ratings. `--floor design` computes it directly from the corpus (the width of the
50-nearest pool around each observed serve rank) and it is TINY, because the
corpus is dense enough that the 50 nearest problems sit almost on top of each
other; `--floor recent` instead assumes today's ratings are exact and takes the
most recent bucket as the floor. With the design floor σ_err is barely more than
a rescaling of the spread, which is the honest reading: nearly none of the
observed gap is design.

σ_err is an UPPER bound on the era's rating error, because it also absorbs every
other way two adjacent problems can differ. The lag profile is the diagnostic
that separates them: per-problem rating error is independent between any two
problems, so it is FLAT in lag, whereas a target that moves during the session
(serve-rank updates, the UCB bonus relaxing, a nearer shard finishing its load)
accumulates and RISES with lag.

CONFOUNDS, with the direction stated only where it is actually known:
  - Corpus growth. A denser corpus narrows the 50-nearest pool independently of
    rating quality, so the floor was wider in early eras than the value computed
    from today's corpus — this one does flatter the present.
  - Pool depletion. The selector skips the last 3000 problems the user saw, so a
    heavy account parked in one band reaches further for a fresh problem.
    Stratified by tenure.
  - Population mix. Corpus density and the rating→rank curve's local slope both
    vary along the ladder, so a shift in who was playing moves the aggregate on
    its own. Stratified by rank band, and standardized to a fixed mix.
  - Selection regime. Since 2026-07-18 a converged user's pool is weighted by
    weakness rather than information; that re-weights within the pool without
    widening it, so it is second-order here.

Coverage is the other caveat and it is not small: a problem dropped from the
corpus has no current rating, so its attempts leave the sample entirely, and the
early months lose most of theirs. Buckets under --min-coverage are printed but
marked `*` and left out of the trend summary. Direction of that bias: a pair
survives only if BOTH its problems are still in the corpus, and a problem pulled
for being malformed is exactly the kind that carried a wrong rating — so the
early buckets, if anything, understate their era's error. --strict-adjacency is
the sensitivity check, requiring the pair to be consecutive in the raw stream.

Everything is measured in ONE frame (today's shipped pwa/tsumegos, never the
backend ratings/ dir), which is the whole point: the ground truth is assumed to
be current. Primary units are internal rank (stones) via rating_to_rank.json;
raw rating units are printed alongside since that is the model's own frame.

Repeats are excluded by default — a recall is drawn from the user's history, not
from the near-rank pool, so it carries no information about serving accuracy.

    ./.venv/bin/python3 -m auxiliary.serve_spread_history
    ./.venv/bin/python3 -m auxiliary.serve_spread_history --size 9 --bucket week
    ./.venv/bin/python3 -m auxiliary.serve_spread_history --strict-adjacency
    ./.venv/bin/python3 -m auxiliary.serve_spread_history --min-tenure 30
    ./.venv/bin/python3 -m auxiliary.serve_spread_history --csv out.csv --plot out.png
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.fit_rank_curves import interp_extrap_vec  # noqa: E402

DB_PATH = os.path.join(PROJECT_ROOT, "attempts", "database.sqlite")
TSUMEGOS_DIR = os.path.join(PROJECT_ROOT, "pwa", "tsumegos")
PARAMS_PATH = os.path.join(TSUMEGOS_DIR, "problem_params.bin")
CURVE_PATH = os.path.join(TSUMEGOS_DIR, "rating_to_rank.json")
# lookup.txt sits at the served root, one level above tsumegos/.
LOOKUP_PATH = os.path.join(PROJECT_ROOT, "pwa", "lookup.txt")

# sd of a zero-symmetric variable from its median absolute value (Gaussian).
MAD_TO_SD = 1.0 / 0.674489750196
# The selector's candidate pool: the N problems nearest the serve rank.
POOL_SIZE = 50

# Rank bands keyed by internal rank index (low <= ri < high), matching the
# kyu/dan convention of getDescriptiveRank (band < 0 -> kyu, else dan).
RANK_BANDS = [
    (float("-inf"), 5, "16k+weaker"),
    (5, 10, "15k-11k"),
    (10, 15, "10k-6k"),
    (15, 20, "5k-1k"),
    (20, float("inf"), "1d+"),
]
# Prior recorded attempts by the same user, at the time of the pair.
TENURE_BANDS = [
    (0, 50, "<50"),
    (50, 250, "50-249"),
    (250, 1000, "250-999"),
    (1000, float("inf"), "1000+"),
]
MIN_STRATUM_PAIRS = 30
# Pairs a user needs in a bucket before their own median enters the user-weighted
# statistic — below that the per-user median is mostly sampling noise.
MIN_USER_PAIRS = 20


# ---------- loading ----------

def load_problem_ratings():
    """{tsumego_id: rating} from the shipped problem_params.bin (16 bytes each:
    4-byte id + float32 rating/slope/floor, both board sizes in one file)."""
    if not os.path.exists(PARAMS_PATH):
        sys.exit(f"Missing {PARAMS_PATH}. Run `python -m pipeline.package_for_pwa` first.")
    raw = np.frombuffer(open(PARAMS_PATH, "rb").read(), dtype=np.uint8).reshape(-1, 16)
    ratings = raw[:, 4:8].copy().view("<f4").ravel()
    return {raw[i, :4].tobytes().hex(): float(ratings[i]) for i in range(len(raw))}


def load_problem_sizes():
    """{tsumego_id: board size} from lookup.txt — problem_params.bin merges both
    sizes with no size column, and the two corpora have very different density."""
    if not os.path.exists(LOOKUP_PATH):
        sys.exit(f"Missing {LOOKUP_PATH}. Run `python -m pipeline.package_for_pwa` first.")
    out = {}
    with open(LOOKUP_PATH) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                out[parts[0]] = int(parts[2])
    return out


def load_curves():
    if not os.path.exists(CURVE_PATH):
        sys.exit(f"Missing {CURVE_PATH}. Run `python -m pipeline.map_rating_to_rank` "
                 f"then `python -m pipeline.package_for_pwa` first.")
    curves = json.load(open(CURVE_PATH))
    return {k: (np.asarray(v["ratings"], float), np.asarray(v["ranks"], float))
            for k, v in curves.items() if isinstance(v, dict) and "ratings" in v}


def to_rank_index(ratings, size, curves):
    xs, ys = curves[str(size)]
    return interp_extrap_vec(np.asarray(ratings, float), xs, ys)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def bucket_key(epoch, mode):
    d = datetime.fromtimestamp(epoch, timezone.utc)
    if mode == "week":
        iso = d.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return d.strftime("%Y-%m")


def load_attempts(sizes, include_repeats, min_tenure=0):
    """Ordered per-user attempt stream, with tenure counted over EVERY recorded
    row (repeats and un-rated problems included — tenure is app experience, not
    sample membership).

    min_tenure drops each account's first N attempts. It is a PREFIX cut, so it
    cannot open a hole between two surviving attempts: pairs still come from
    genuinely consecutive serves, just later in the account's life."""
    if not os.path.exists(DB_PATH):
        sys.exit(f"Missing {DB_PATH}. Run `python -m pipeline.fetch_database` first.")
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT user_id, created_at, size, problem, is_repeat FROM tsumego_reports "
        "ORDER BY user_id, created_at, id").fetchall()
    con.close()

    out, tenure = [], 0
    prev_user = None
    for user, created, size, problem, is_repeat in rows:
        if user != prev_user:
            tenure, prev_user = 0, user
        tenure += 1
        if size in sizes and (include_repeats or not is_repeat) and tenure - 1 >= min_tenure:
            out.append((user, parse_ts(created), size, problem, tenure - 1))
    return out


# ---------- pair construction ----------

def build_sessions(attempts, ratings, curves, session_gap_s, strict_adjacency):
    """Cut the attempt stream into sessions — same user, same size, no gap longer
    than session_gap_s — and score each attempt in today's frame.

    Returns (runs, n_raw_adjacent, n_rated_rows), where each run is a list of
    scored records the pair builder may draw from. Non-strict runs are the
    session's rated attempts with un-rated ones closed up (a problem that has
    since left the corpus is skipped, and the session-gap bound still limits how
    far the pair reaches); strict runs are maximal stretches of CONSECUTIVE rated
    attempts, so a dropped row disqualifies the pair rather than widening it."""
    rated_rows = 0
    resolved = []
    for user, t, size, problem, tenure in attempts:
        r = ratings.get(problem)
        rated_rows += r is not None
        resolved.append([user, t, size, problem, tenure, r, None])
    # One vectorized curve pass per size rather than a lookup per attempt.
    by_size = defaultdict(list)
    for i, rec in enumerate(resolved):
        if rec[5] is not None:
            by_size[rec[2]].append(i)
    for size, idxs in by_size.items():
        vals = to_rank_index([resolved[i][5] for i in idxs], size, curves)
        for i, v in zip(idxs, vals):
            resolved[i][6] = float(v)

    sessions, cur, prev = [], [], None
    raw_adjacent = 0
    for rec in resolved:
        user, t, size = rec[0], rec[1], rec[2]
        if prev is not None and prev[0] == user and prev[2] == size \
                and 0 <= t - prev[1] <= session_gap_s:
            raw_adjacent += 1
        else:
            if len(cur) > 1:
                sessions.append(cur)
            cur = []
        cur.append(rec)
        prev = rec
    if len(cur) > 1:
        sessions.append(cur)

    runs = []
    for s in sessions:
        if strict_adjacency:
            run = []
            for rec in s:
                if rec[5] is None:
                    if len(run) > 1:
                        runs.append(run)
                    run = []
                else:
                    run.append(rec)
            if len(run) > 1:
                runs.append(run)
        else:
            run = [rec for rec in s if rec[5] is not None]
            if len(run) > 1:
                runs.append(run)
    return runs, raw_adjacent, rated_rows


def pairs_at_lag(runs, lag, bucket_mode):
    """Pairs `lag` novel attempts apart within a run. A pair is
    (bucket, user, size, mean_rank_index, tenure, |Δ rank|, |Δ rating|); the
    bucket is the later attempt's. Repeated serves of the SAME problem are
    dropped — their zero gap measures nothing about serving accuracy."""
    out = []
    for run in runs:
        for i in range(len(run) - lag):
            a, b = run[i], run[i + lag]
            if a[3] == b[3]:
                continue
            out.append((
                bucket_key(b[1], bucket_mode), b[0], b[2],
                0.5 * (a[6] + b[6]), b[4],
                abs(b[6] - a[6]), abs(b[5] - a[5])))
    return out


# ---------- statistics ----------

def band_of(value, bands):
    for lo, hi, label in bands:
        if lo <= value < hi:
            return label
    return bands[-1][2]


def robust_sd(abs_deltas):
    """Scale of a zero-symmetric Δ from its median absolute value. Robust because
    the shipped rating tails are unclipped and a single extrapolated outlier
    would dominate a moment estimator."""
    return MAD_TO_SD * float(np.median(abs_deltas))


def implied_sigma(sd, floor_sd):
    """Per-problem rating error implied by a spread of `sd` once the design floor
    is removed: Var(Δ) = 2σ² + Var(floor)."""
    excess = sd * sd - floor_sd * floor_sd
    return float(np.sqrt(max(0.0, excess) / 2.0))


def pair_weighted(groups):
    return float(np.median(np.concatenate(groups)))


def user_weighted(groups, min_pairs=MIN_USER_PAIRS):
    """Median across users of each user's own median |Δ| — the typical USER's
    spread, where the pair-weighted median is the typical ATTEMPT's and a single
    heavy account can carry a whole bucket."""
    per_user = [np.median(g) for g in groups if len(g) >= min_pairs]
    return float(np.median(per_user)) if per_user else float("nan")


def bootstrap_ci(groups, stat_fn, n_boot, rng, alpha=0.05):
    """Cluster bootstrap over users — attempts within a user are anything but
    independent, so the resampled unit is the user, not the pair."""
    if n_boot <= 0 or len(groups) < 2:
        return (float("nan"), float("nan"))
    n = len(groups)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        stats[b] = stat_fn([groups[j] for j in pick])
    stats = stats[np.isfinite(stats)]
    if len(stats) < 2:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def reference_spreads(ratings, problem_sizes, curves, sizes, serve_ranks_by_size,
                      rng, n_samples=400):
    """The two ends of the scale a measured spread sits between, per size:

      floor  — median |Δ| between two problems drawn from the POOL_SIZE nearest
               the serve rank in TODAY's corpus, i.e. what the selector still
               produces under PERFECT ratings. Optimistic in two ways, both
               stated rather than corrected: it ignores the within-pool
               quality/weakness weighting, and it pools that size's whole corpus
               where the client only sees the shards it has loaded and skips the
               3000 problems it most recently served.
      random — median |Δ| between two problems drawn from the corpus at large,
               i.e. no targeting at all.

    Both in stones and in raw rating units, since the rating→rank curve's local
    slope varies enormously across the ladder."""
    out = {}
    for size in sizes:
        vals = np.array([r for t, r in ratings.items() if problem_sizes.get(t) == size], float)
        serve = serve_ranks_by_size.get(size)
        if len(vals) < POOL_SIZE or serve is None or len(serve) == 0:
            continue
        order = np.argsort(vals)
        vals_sorted = vals[order]
        ranks = to_rank_index(vals_sorted, size, curves)  # monotone curve: stays sorted

        picks = rng.choice(serve, size=min(n_samples, len(serve)), replace=len(serve) < n_samples)
        tri = np.triu_indices(POOL_SIZE, 1)
        gaps_rank, gaps_rating = [], []
        for r in picks:
            j = int(np.searchsorted(ranks, r))
            lo = max(0, min(j - POOL_SIZE // 2, len(ranks) - POOL_SIZE))
            pool_rank = ranks[lo:lo + POOL_SIZE]
            pool_rating = vals_sorted[lo:lo + POOL_SIZE]
            gaps_rank.append(np.median(np.abs(pool_rank[:, None] - pool_rank[None, :])[tri]))
            gaps_rating.append(np.median(np.abs(pool_rating[:, None] - pool_rating[None, :])[tri]))

        n = min(20000, len(vals))
        a = rng.choice(len(vals), n)
        b = rng.choice(len(vals), n)
        out[size] = dict(
            floor_rank=float(np.median(gaps_rank)),
            floor_rating=float(np.median(gaps_rating)),
            random_rank=float(np.median(np.abs(ranks[a] - ranks[b]))),
            random_rating=float(np.median(np.abs(vals_sorted[a] - vals_sorted[b]))),
        )
    return out


def summarize(pairs, buckets, min_coverage, coverage, n_boot, seed):
    """Per-bucket robust spread + cluster-bootstrap CI, keyed by bucket."""
    rng = np.random.default_rng(seed)
    rows = {}
    for b in buckets:
        sel = [p for p in pairs if p[0] == b]
        if not sel:
            continue
        arr_rank = np.array([p[5] for p in sel])
        arr_rating = np.array([p[6] for p in sel])
        by_user = defaultdict(list)
        for p in sel:
            by_user[p[1]].append(p[5])
        groups = [np.array(v) for v in by_user.values()]
        lo, hi = bootstrap_ci(groups, pair_weighted, n_boot, rng)
        u_lo, u_hi = bootstrap_ci(groups, user_weighted, n_boot, rng)
        rows[b] = dict(
            bucket=b, pairs=len(sel), users=len(by_user),
            coverage=coverage.get(b, float("nan")),
            valid=coverage.get(b, 0.0) >= min_coverage,
            med_rank=float(np.median(arr_rank)),
            ci_lo=lo, ci_hi=hi,
            med_user=user_weighted(groups),
            user_ci_lo=u_lo, user_ci_hi=u_hi,
            n_user_qualified=sum(1 for g in groups if len(g) >= MIN_USER_PAIRS),
            p90_rank=float(np.percentile(arr_rank, 90)),
            sd_rank=robust_sd(arr_rank),
            med_rating=float(np.median(arr_rating)),
            sd_rating=robust_sd(arr_rating),
            user_rank=float(np.median([p[3] for p in sel])),
            tenure=float(np.median([p[4] for p in sel])),
        )
    return rows


def standardize(pairs, buckets, strata_of):
    """Direct standardization: per-bucket median within each stratum, recombined
    with FIXED pooled weights, so a shift in who was playing cannot move the
    number. Returns {bucket: (adjusted median, weight fraction covered)}."""
    per_bucket = defaultdict(lambda: defaultdict(list))
    pooled = defaultdict(int)
    for p in pairs:
        s = strata_of(p)
        per_bucket[p[0]][s].append(p[5])
        pooled[s] += 1
    total = sum(pooled.values())
    out = {}
    for b in buckets:
        num = den = 0.0
        for s, w in pooled.items():
            vals = per_bucket[b].get(s)
            if not vals or len(vals) < MIN_STRATUM_PAIRS:
                continue
            num += w * float(np.median(vals))
            den += w
        out[b] = (num / den if den else float("nan"), den / total if total else 0.0)
    return out


# ---------- reporting ----------

def print_main_table(rows, buckets, floor_sd, unit_note):
    print(f"\n{'bucket':>9} {'users':>6} {'pairs':>7} {'cov':>5} "
          f"{'med|Δ|':>7} {'95% CI':>13} {'per-user':>9} {'95% CI':>13} "
          f"{'p90':>6} {'σ_err':>6} {'rank':>6} {'tenure':>7}")
    print("-" * 112)
    for b in buckets:
        r = rows[b]
        ci = f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]" if np.isfinite(r["ci_lo"]) else "-"
        uci = f"[{r['user_ci_lo']:.2f},{r['user_ci_hi']:.2f}]" if np.isfinite(r["user_ci_lo"]) else "-"
        print(f"{b:>9} {r['users']:>6} {r['pairs']:>7} {r['coverage']:>5.2f} "
              f"{r['med_rank']:>7.3f} {ci:>13} {r['med_user']:>9.3f} {uci:>13} "
              f"{r['p90_rank']:>6.2f} {implied_sigma(r['sd_rank'], floor_sd):>6.2f} "
              f"{r['user_rank']:>6.1f} {r['tenure']:>7.0f}"
              + ("" if r["valid"] else "  *"))
    print(f"  {unit_note}")
    print(f"  med|Δ| weights every pair (the typical ATTEMPT); per-user takes each user's own "
          f"median\n  first (the typical USER, ≥{MIN_USER_PAIRS} pairs to qualify) so one heavy "
          f"account cannot carry a bucket.")
    print("  * coverage below the --min-coverage bar; excluded from the trend summary.")


def print_strata_table(pairs, buckets, index, bands, title):
    print(f"\n  median |Δ| (stones) by {title}, pair count in brackets")
    labels = [b[2] for b in bands]
    cells = defaultdict(list)
    for p in pairs:
        cells[(p[0], band_of(p[index], bands))].append(p[5])
    print("    " + f"{'bucket':>9} " + " ".join(f"{l:>13}" for l in labels))
    for b in buckets:
        parts = []
        for l in labels:
            v = cells.get((b, l))
            if v and len(v) >= MIN_STRATUM_PAIRS:
                n = f"{len(v) // 1000}k" if len(v) >= 1000 else str(len(v))
                parts.append(f"{np.median(v):>7.2f}[{n:>4}]")
            else:
                parts.append(f"{'-':>13}")
        print("    " + f"{b:>9} " + " ".join(parts))


def print_lag_profile(runs, buckets, bucket_mode, lags):
    """Median |Δ| against how many attempts apart the two problems were.

    FLAT in lag means each problem carries its own independent offset from the
    others — the signature of per-problem rating error (and of the pool draw),
    since two errors are equally independent one apart or sixteen apart. RISING
    in lag means the target itself is moving during the session — serve-rank
    drift, a shard loading, the UCB bonus relaxing — which accumulates with
    distance. The two have very different fixes, so the shape is worth more than
    any single bucket's level."""
    print("\n  median |Δ| (stones) by lag — attempts apart within a session")
    print("    " + f"{'bucket':>9} " + " ".join(f"{'lag ' + str(l):>9}" for l in lags))
    per_lag = {l: defaultdict(list) for l in lags}
    for l in lags:
        for p in pairs_at_lag(runs, l, bucket_mode):
            per_lag[l][p[0]].append(p[5])
    for b in buckets:
        cells = []
        for l in lags:
            v = per_lag[l].get(b)
            cells.append(f"{np.median(v):>9.2f}" if v and len(v) >= MIN_STRATUM_PAIRS
                         else f"{'-':>9}")
        print("    " + f"{b:>9} " + " ".join(cells))


def print_trend(rows, valid, floor_sd, adjusted):
    if len(valid) < 2:
        print("\nNot enough buckets above the coverage bar for a trend summary.")
        return
    first, last = valid[0], valid[-1]
    a, b = rows[first], rows[last]
    print(f"\nTrend over the {len(valid)} buckets above the coverage bar "
          f"({first} → {last}):")
    print(f"  median |Δ|      {a['med_rank']:.3f} → {b['med_rank']:.3f} stones "
          f"({100 * (b['med_rank'] / a['med_rank'] - 1):+.1f}%)")
    print(f"  per-user        {a['med_user']:.3f} → {b['med_user']:.3f} stones "
          f"({100 * (b['med_user'] / a['med_user'] - 1):+.1f}%)")
    adj_a, adj_b = adjusted[first][0], adjusted[last][0]
    print(f"  mix-adjusted    {adj_a:.3f} → {adj_b:.3f} stones "
          f"({100 * (adj_b / adj_a - 1):+.1f}%)   "
          f"[weight covered {adjusted[first][1]:.2f} → {adjusted[last][1]:.2f}]")
    print(f"  implied σ_err   {implied_sigma(a['sd_rank'], floor_sd):.3f} → "
          f"{implied_sigma(b['sd_rank'], floor_sd):.3f} stones per problem")
    xs = np.arange(len(valid), dtype=float)
    ys = np.array([rows[v]["med_rank"] for v in valid])
    slope = float(np.polyfit(xs, ys, 1)[0])
    print(f"  OLS slope       {slope:+.4f} stones per bucket over the valid span")


def write_csv(path, rows, buckets, adjusted, floor_sd):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "users", "pairs", "coverage", "above_coverage_bar",
                    "median_abs_delta_stones", "ci_lo", "ci_hi",
                    "per_user_median_stones", "per_user_ci_lo", "per_user_ci_hi",
                    "users_qualified", "p90_stones",
                    "robust_sd_stones", "implied_sigma_err_stones",
                    "median_abs_delta_rating", "robust_sd_rating",
                    "mix_adjusted_stones", "mix_weight_covered",
                    "median_user_rank_index", "median_tenure"])
        for b in buckets:
            r = rows[b]
            w.writerow([b, r["users"], r["pairs"], f"{r['coverage']:.4f}", int(r["valid"]),
                        f"{r['med_rank']:.4f}", f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}",
                        f"{r['med_user']:.4f}", f"{r['user_ci_lo']:.4f}", f"{r['user_ci_hi']:.4f}",
                        r["n_user_qualified"],
                        f"{r['p90_rank']:.4f}", f"{r['sd_rank']:.4f}",
                        f"{implied_sigma(r['sd_rank'], floor_sd):.4f}",
                        f"{r['med_rating']:.4f}", f"{r['sd_rating']:.4f}",
                        f"{adjusted[b][0]:.4f}", f"{adjusted[b][1]:.3f}",
                        f"{r['user_rank']:.2f}", f"{r['tenure']:.0f}"])
    print(f"\nwrote {path}")


def write_plot(path, rows, buckets, adjusted, floor_med, floor_sd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(buckets))
    med = np.array([rows[b]["med_rank"] for b in buckets])
    lo = np.array([rows[b]["ci_lo"] for b in buckets])
    hi = np.array([rows[b]["ci_hi"] for b in buckets])
    adj = np.array([adjusted[b][0] for b in buckets])
    rating = np.array([rows[b]["med_rating"] for b in buckets])
    weak = ~np.array([rows[b]["valid"] for b in buckets])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    per_user = np.array([rows[b]["med_user"] for b in buckets])
    ax1.fill_between(x, lo, hi, alpha=0.2, color="C0", label="95% CI (cluster bootstrap)")
    ax1.plot(x, med, "o-", color="C0", label="median |Δ| between adjacent problems")
    ax1.plot(x, per_user, "^-", color="C4", alpha=0.8, label="per-user median")
    ax1.plot(x, adj, "s--", color="C1", alpha=0.8, label="mix-adjusted (fixed rank×tenure weights)")
    if np.isfinite(floor_med):
        ax1.axhline(floor_med, color="C3", ls=":", label="design floor (50-nearest pool, today's corpus)")
    if weak.any():
        ax1.scatter(x[weak], med[weak], marker="x", s=80, color="C3", zorder=5,
                    label="below coverage bar")
    ax1.set_ylabel("stones")
    ax1.set_title("Spread between adjacent novel problems, scored in today's ratings")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Raw rating units rather than a rescaling of panel 1: the rating→rank curve's
    # local slope varies hugely along the ladder, so the two frames move apart
    # whenever the population shifts bands.
    ax2.plot(x, rating, "o-", color="C2")
    ax2.set_ylabel("rating units")
    ax2.set_title("The same gap in the model's own frame (raw rating units)")
    ax2.grid(alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(buckets, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=["9", "13", "all"], default="all",
                    help="board size; pairs always require both problems on the same size")
    ap.add_argument("--bucket", choices=["month", "week"], default="month")
    ap.add_argument("--session-gap-min", type=float, default=30.0,
                    help="max minutes between the two attempts of a pair")
    ap.add_argument("--include-repeats", action="store_true",
                    help="count recall repeats too (they are not near-rank selections)")
    ap.add_argument("--min-tenure", type=int, default=0,
                    help="drop each account's first N attempts, where the rank estimate "
                         "is still far from converged and the UCB serve bonus is largest")
    ap.add_argument("--strict-adjacency", action="store_true",
                    help="require the pair to be consecutive in the RAW stream, so a "
                         "problem that has since left the corpus disqualifies it")
    ap.add_argument("--min-coverage", type=float, default=0.90,
                    help="rating-coverage bar below which a bucket is marked and left "
                         "out of the trend summary")
    ap.add_argument("--floor", default="design",
                    help="what to treat as the irreducible spread when implying σ_err: "
                         "'design' = the computed 50-nearest pool width, 'recent' = the "
                         "most recent bucket above the coverage bar (i.e. assume today's "
                         "ratings are exact), or an explicit robust sd in stones")
    ap.add_argument("--bootstrap", type=int, default=200, help="cluster-bootstrap resamples (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv")
    ap.add_argument("--plot")
    args = ap.parse_args()

    sizes = [9, 13] if args.size == "all" else [int(args.size)]
    rng = np.random.default_rng(args.seed)

    ratings = load_problem_ratings()
    problem_sizes = load_problem_sizes()
    curves = load_curves()
    attempts = load_attempts(sizes, args.include_repeats, args.min_tenure)
    runs, considered, rated_rows = build_sessions(
        attempts, ratings, curves, args.session_gap_min * 60.0, args.strict_adjacency)
    pairs = pairs_at_lag(runs, 1, args.bucket)

    print(f"attempts in scope        {len(attempts):>8}  "
          f"(sizes {sizes}, repeats {'in' if args.include_repeats else 'out'}"
          + (f", first {args.min_tenure} per account dropped" if args.min_tenure else "") + ")")
    print(f"with a current rating    {rated_rows:>8}  ({rated_rows / max(1, len(attempts)):.3f})")
    print(f"adjacent pairs in window {considered:>8}  (≤{args.session_gap_min:g} min apart)")
    print(f"pairs both rated         {len(pairs):>8}  ({len(pairs) / max(1, considered):.3f})"
          + ("  [strict adjacency]" if args.strict_adjacency else ""))
    if not pairs:
        sys.exit("No pairs — nothing to report.")

    # Per-bucket rating coverage of the attempts themselves (not of the pairs),
    # since that is what a reader needs to judge how selected the sample is.
    seen, have = defaultdict(int), defaultdict(int)
    for user, t, size, problem, tenure in attempts:
        b = bucket_key(t, args.bucket)
        seen[b] += 1
        have[b] += problem in ratings
    coverage = {b: have[b] / seen[b] for b in seen}

    buckets = sorted({p[0] for p in pairs})
    rows = summarize(pairs, buckets, args.min_coverage, coverage, args.bootstrap, args.seed)
    buckets = [b for b in buckets if b in rows]
    valid = [b for b in buckets if rows[b]["valid"]]

    serve_by_size = defaultdict(list)
    for p in pairs:
        serve_by_size[p[2]].append(p[3])
    refs = reference_spreads(ratings, problem_sizes, curves, sizes,
                             {k: np.array(v) for k, v in serve_by_size.items()}, rng)
    floor_med = float(np.median([r["floor_rank"] for r in refs.values()])) if refs else float("nan")

    print(f"\n{'reference':>28} {'median |Δ| stones':>18} {'median |Δ| rating':>18}")
    for s in sorted(refs):
        r = refs[s]
        print(f"{f'{s}x{s} perfect targeting':>28} {r['floor_rank']:>18.3f} {r['floor_rating']:>18.3f}")
        print(f"{f'{s}x{s} no targeting':>28} {r['random_rank']:>18.3f} {r['random_rating']:>18.3f}")

    if args.floor == "design":
        floor_sd = MAD_TO_SD * floor_med if refs else 0.0
        floor_sd_rating = (MAD_TO_SD * float(np.median([r["floor_rating"] for r in refs.values()]))
                           if refs else 0.0)
        floor_src = f"the computed {POOL_SIZE}-nearest pool width"
    elif args.floor == "recent":
        floor_sd = rows[valid[-1]]["sd_rank"] if valid else 0.0
        floor_sd_rating = rows[valid[-1]]["sd_rating"] if valid else 0.0
        floor_src = (f"the most recent bucket above the bar ({valid[-1] if valid else 'n/a'}) "
                     f"— i.e. today's ratings taken as exact")
    else:
        floor_sd = float(args.floor)
        floor_sd_rating = 0.0
        floor_src = "--floor"

    print(f"\nfloor used for σ_err: robust sd {floor_sd:.3f} stones, from {floor_src}.")
    print("σ_err is an UPPER bound on the era's per-problem rating error: it also absorbs "
          "serve-rank\nmovement between the two attempts and pool depletion (the selector "
          "skips the last 3000\nproblems the user saw, so a parked heavy user reaches further "
          "for a fresh one).")

    print_main_table(rows, buckets, floor_sd,
                     "med|Δ|, p90 and σ_err in stones; rank = median served rank index; "
                     "tenure = median prior attempts")

    print("\n" + "-" * 96)
    print("Same quantity in raw rating units (the model's own frame):")
    print(f"{'bucket':>9} {'med|Δ|':>8} {'sd':>7} {'σ_err':>7}")
    for b in buckets:
        r = rows[b]
        print(f"{b:>9} {r['med_rating']:>8.3f} {r['sd_rating']:>7.3f} "
              f"{implied_sigma(r['sd_rating'], floor_sd_rating):>7.3f}"
              + ("" if r["valid"] else "  *"))

    adjusted = standardize(pairs, buckets,
                           lambda p: (band_of(p[3], RANK_BANDS), band_of(p[4], TENURE_BANDS)))
    print("\n" + "-" * 96)
    print("Composition control — the same pairs, held at a fixed rank × tenure mix:")
    print(f"{'bucket':>9} {'raw':>7} {'adjusted':>9} {'weight':>7}")
    for b in buckets:
        adj, w = adjusted[b]
        print(f"{b:>9} {rows[b]['med_rank']:>7.3f} {adj:>9.3f} {w:>7.2f}"
              + ("" if rows[b]["valid"] else "  *"))

    print_strata_table(pairs, buckets, 3, RANK_BANDS, "rank band")
    print_strata_table(pairs, buckets, 4, TENURE_BANDS, "tenure band")
    print_lag_profile(runs, buckets, args.bucket, [1, 2, 4, 8, 16])

    print_trend(rows, valid, floor_sd, adjusted)

    if args.csv:
        write_csv(args.csv, rows, buckets, adjusted, floor_sd)
    if args.plot:
        write_plot(args.plot, rows, buckets, adjusted, floor_med, floor_sd)


if __name__ == "__main__":
    main()
