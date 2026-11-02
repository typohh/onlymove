#!/usr/bin/env python3
"""What the CLIENT's own estimators do — the figures docs/methodology.md §2.3, §2.4
and §2.11 quote about serving, the information slot and the community histogram.

Every number here is produced by RUNNING the shipped client modules, not by
restating their formulas: `ablations/client_estimators.mjs` imports
`pwa/shared/rank-window.js`, `pwa/js/irt.js` and `pwa/js/constants.js` unmodified
under node and calls them, so a change to the estimator moves the figure instead
of quietly invalidating it. This module assembles the shipped inputs those modules
need, runs that helper, and turns its output into ledger figures.

  §2.3  the onboarding serve margin, simulated on the shipped corpus
    serve.onboard_over3_pct        share of a simulated 20-kyu's first thirty
                                   first-time attempts served more than three ranks
                                   above their true rank, under the shipped cap
    serve.onboard_over5_pct        the same at more than five ranks
    serve.onboard_const_over3_pct  the same two under a CONSTANT margin of five,
    serve.onboard_const_over5_pct    the standing alternative
    serve.onboard_count_over3_pct  and under the attempt-count margin, the
                                   alternative that matches the shipped rule's speed
    serve.onboard_dan_attempts     first-time attempts an unlimited-strength solver
                                   needs before the SHOWN rank reaches 1 dan
    serve.onboard_dan_attempts_const  the same under the constant margin

  §2.4  the information slot's extreme-value bias
    draw.info_argmax_enrichment    how much more often the top-information candidate
                                   of a difficulty-matched pool sits in a motif tail
                                   than the pool itself does, on the worst tail
    draw.info_best_over_pool       the best candidate's information over the pool mean
    draw.info_fifth_advantage      the fifth-best candidate's share of that advantage
    draw.info_slice_over_best      what a draw from the top fifth carries against the
                                   top candidate itself

  §2.11 the histogram's minimum-attempt gate
    community.gate_shift_ranks     how far the mean shown rank moves when everyone
                                   active is counted instead of ranked players
    community.gate_shift_signed    the same with its sign, for the guard on which
                                   way the sentence says it moves

    ./.venv/bin/python3 -m ablations.client_estimator_figures
"""

import datetime
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import AblationError, Figure, require  # noqa: E402
from utils.constants import not_ignored_sql  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402

TIER = "about"

PWA_ROOT = os.path.join(PROJECT_ROOT, "pwa")
TSUMEGOS = os.path.join(PWA_ROOT, "tsumegos")
HELPER = os.path.join(PROJECT_ROOT, "ablations", "client_estimators.mjs")

# The simulation's seed. Fixed, and passed into the helper's PRNG, so the figures
# reproduce exactly on re-run; nothing here ever reads a clock or Math.random.
SEED = 20260901

# The simulated player §2.3 states its exposure figures for, on the internal scale
# (0 = the 1k/1d boundary, integers are band edges), and the horizon.
TWENTY_KYU = -19.5
HORIZON = 30
SEEDS = 2000
GAP_THRESHOLDS = (3.0, 5.0)
BADGE_RANK = 0.0            # the 1k/1d boundary — the first dan badge
BADGE_MAX_ATTEMPTS = 500
CONSTANT_MARGIN = 5.0       # the standing alternative §2.3 argues against
# 9×9 carries roughly three quarters of the first-time attempt log, so it is the
# board a simulated onboarding should run on.
SIM_SIZE = 9
POOL_SIZE = 50              # selectChallengeCandidate's candidate count
FIFTH_INDEX = 4             # "the fifth-best candidate", 0-based into the sorted pool
TOP_FRACTION = 0.2          # INFO_TOP_FRACTION (problem-loader.js): what the slot draws from

# §2.11's window: the community section aggregates the trailing seven days.
WINDOW_DAYS = 7
# What counts as active, and therefore who the ungated population is: one first-time
# attempt inside the window, the same event that stamps a player's activity on the
# server. Requiring more than that invents a SECOND gate beside the one under test
# and shrinks the contrast the sentence is about, so the comparison is the shipped
# minimum-attempt gate against no gate at all.
ACTIVE_MIN = 1

BOARD_SIZES = (9, 13)


# --- shipped inputs ----------------------------------------------------------------

def _read_json(path):
    require(os.path.exists(path), f"no {path} — run pipeline.package_for_pwa")
    with open(path) as f:
        return json.load(f)


def load_corpus():
    """The published per-problem triple joined to the published motif vectors.

    `problem_params.bin` is 16 bytes per problem (4-byte id + three little-endian
    float32) and is keyed by id; `lookup.embeddings.bin` is row-aligned to
    `lookup.txt`, not to it, so the join is by id and never by position."""
    params_path = os.path.join(TSUMEGOS, "problem_params.bin")
    lookup_path = os.path.join(PWA_ROOT, "lookup.txt")
    emb_path = os.path.join(TSUMEGOS, "lookup.embeddings.bin")
    for p in (params_path, lookup_path, emb_path):
        require(os.path.exists(p), f"no {p} — run pipeline.package_for_pwa")

    raw = np.fromfile(params_path, dtype=np.uint8)
    require(raw.size % 16 == 0, "problem_params.bin is not a multiple of 16 bytes")
    raw = raw.reshape(-1, 16)
    ids = [row.tobytes().hex() for row in raw[:, :4]]
    vals = raw[:, 4:].copy().view("<f4").reshape(-1, 3)

    meta = _read_json(os.path.join(TSUMEGOS, "embedding_meta.json"))
    K = meta.get("K")
    require(isinstance(K, int) and K > 0, "embedding_meta.json has no usable K")
    require(meta.get("dtype") == "float16",
            f"lookup.embeddings.bin dtype is {meta.get('dtype')!r}, not float16")
    emb = np.fromfile(emb_path, dtype="<f2")
    require(emb.size % K == 0, "lookup.embeddings.bin is not a multiple of K")
    emb = emb.reshape(-1, K).astype(np.float32)

    rows = [line.split() for line in open(lookup_path)]
    require(all(len(r) == 3 for r in rows), "lookup.txt is not 'id zipHash size'")
    require(len(rows) == emb.shape[0],
            f"lookup.txt has {len(rows)} lines against {emb.shape[0]} embedding rows")
    pos = {r[0]: i for i, r in enumerate(rows)}
    sizes = np.array([int(r[2]) for r in rows], dtype=np.uint8)
    require(len(pos) == len(rows), "lookup.txt holds a duplicate id")
    require(len(ids) == len(rows),
            f"problem_params.bin holds {len(ids)} problems against "
            f"{len(rows)} lookup rows")
    missing = [t for t in ids if t not in pos]
    require(not missing, f"{len(missing)} problem_params ids are absent from lookup.txt")

    order = np.array([pos[t] for t in ids])
    return {
        "ids": ids, "K": K,
        "rating": vals[:, 0], "slope": vals[:, 1], "floor": vals[:, 2],
        "size": sizes[order], "emb": emb[order],
        "labels": meta.get("labels", {}).get("components", []),
    }


def named_tail_mask(corpus, motif_z):
    """Bit k set when a problem's correct-move loading clears the motif gate on axis
    k's NAMED tail. Only the positive tail carries a name in the shipped labels, and
    an unnamed tail is one the client never prints, so it cannot be a served motif."""
    K = corpus["K"]
    labels = corpus["labels"]
    require(len(labels) == K, f"labels carry {len(labels)} components against K={K}")
    named = [bool(c.get("positive")) for c in labels]
    require(any(named), "no axis carries a named positive tail")
    require(K <= 32, f"K={K} axes do not fit the 32-bit tail mask")
    mask = np.zeros(corpus["emb"].shape[0], dtype=np.uint32)
    for k in range(K):
        if named[k]:
            mask |= (corpus["emb"][:, k] >= motif_z).astype(np.uint32) << np.uint32(k)
    return mask, named


def prior_grid():
    """The shipped population rank prior as (grid, normalized density). It is the
    measured distribution of rank across the app's own users, so weighting the pools
    by it asks what the draw does where players actually are rather than averaging
    over stretches of the ladder nobody occupies."""
    pc = _read_json(os.path.join(TSUMEGOS, "population_constants.json"))
    rp = pc.get("rank_prior")
    require(isinstance(rp, dict) and "log_density" in rp,
            "population_constants.json has no rank_prior")
    lg = np.asarray(rp["log_density"], dtype=float)
    require(lg.size >= 2 and np.all(np.isfinite(lg)), "rank_prior log_density is unusable")
    grid = rp["rank_lo"] + rp["rank_step"] * np.arange(lg.size)
    w = np.exp(lg - lg.max())
    return grid, w / w.sum()


def community_buffers(corpus, buffer_cap):
    """Per active player, the attempts the leaderboard's own rank fit would see.

    `active` is a first-time (novel) attempt inside the window, which is exactly what
    stamps `last_active` on the server. The buffer is the player's last `buffer_cap`
    novel attempts by insert order — the one ordering every writer and rebuild agrees
    on — resolved against the published parameters here, so the helper only has to
    run the estimator. The window ends at the log's own newest row rather than at the
    clock, so the measurement is a function of the snapshot alone."""
    require(os.path.exists(DB_SQLITE), f"no {DB_SQLITE} — run pipeline.fetch_database")
    params = {t: (float(r), float(s), float(f)) for t, r, s, f in
              zip(corpus["ids"], corpus["rating"], corpus["slope"], corpus["floor"])}
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        newest = con.execute("SELECT MAX(created_at) FROM tsumego_reports").fetchone()[0]
        require(isinstance(newest, str) and newest, "tsumego_reports is empty")
        end = datetime.datetime.fromisoformat(newest.replace("Z", "+00:00"))
        cutoff = (end - datetime.timedelta(days=WINDOW_DAYS)) \
            .isoformat().replace("+00:00", "Z")
        active = con.execute(
            "SELECT user_id, COUNT(*) FROM tsumego_reports "
            f" WHERE is_repeat = 0 AND created_at >= ? AND {not_ignored_sql()} GROUP BY user_id",
            (cutoff,)).fetchall()
        require(active, f"no first-time attempts in the {WINDOW_DAYS}-day window")
        users = []
        for user_id, in_window in active:
            rows = con.execute(
                "SELECT problem, success FROM tsumego_reports "
                " WHERE user_id = ? AND is_repeat = 0 ORDER BY id DESC LIMIT ?",
                (user_id, buffer_cap)).fetchall()
            attempts = [[*params[p], 1 if s else 0]
                        for p, s in reversed(rows) if p in params]
            users.append({"id": user_id, "in_window": int(in_window),
                          "attempts": attempts})
    finally:
        con.close()
    return {"window_end": newest, "cutoff": cutoff, "users": users}


# --- the helper --------------------------------------------------------------------

def run_helper(corpus, mask):
    node = shutil.which("node")
    require(node is not None, "no `node` on PATH — the client modules run under node")
    require(os.path.exists(HELPER), f"no {HELPER}")
    grid, weights = prior_grid()
    # The helper needs the shipped constants before it can say what a buffer is, so
    # the community rows are assembled at the shared cap the estimator's window uses,
    # read from the mirror serve_log_figures checks against rank-window.js.
    from ablations.serve_log_figures import WIN_CAP, _check_win_cap
    _check_win_cap()
    from_cap = WIN_CAP
    community = community_buffers(corpus, from_cap)

    n = corpus["rating"].size
    packed = np.zeros(n, dtype=[("rating", "<f4"), ("slope", "<f4"), ("floor", "<f4"),
                                ("size", "u1"), ("motif", "<u4")])
    require(packed.dtype.itemsize == 17, "the corpus record is not 17 bytes")
    packed["rating"] = corpus["rating"]
    packed["slope"] = corpus["slope"]
    packed["floor"] = corpus["floor"]
    packed["size"] = corpus["size"]
    packed["motif"] = mask

    with tempfile.TemporaryDirectory(prefix="client-estimators-") as tmp:
        problems_path = os.path.join(tmp, "problems.bin")
        packed.tofile(problems_path)
        payload = {
            "pwa_root": PWA_ROOT + os.sep,
            "problems_path": problems_path,
            "seed": SEED,
            "board_sizes": list(BOARD_SIZES),
            "pool_size": POOL_SIZE,
            "serve_margin": {
                "true_ranks": [TWENTY_KYU],
                "horizon": HORIZON,
                "seeds": SEEDS,
                "gap_thresholds": list(GAP_THRESHOLDS),
                "constant_margin": CONSTANT_MARGIN,
                "badge_rank": BADGE_RANK,
                "badge_max_attempts": BADGE_MAX_ATTEMPTS,
                "size": SIM_SIZE,
            },
            "info_slot": {
                "axes": corpus["K"],
                "grid": grid.tolist(),
                "weights": weights.tolist(),
                "advantage_index": FIFTH_INDEX,
                "top_fraction": TOP_FRACTION,
            },
            "community": {"users": community["users"]},
        }
        payload_path = os.path.join(tmp, "payload.json")
        with open(payload_path, "w") as f:
            json.dump(payload, f)
        proc = subprocess.run([node, HELPER, payload_path], cwd=PROJECT_ROOT,
                              capture_output=True, text=True)
        require(proc.returncode == 0,
                f"client_estimators.mjs failed ({proc.returncode}):\n{proc.stderr.strip()}")
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise AblationError(f"client_estimators.mjs wrote no JSON: {e}") from None
    out["_community_meta"] = community
    require(out["community"]["buffer_cap"] == from_cap,
            f"the estimator's window cap is {out['community']['buffer_cap']}, not "
            f"{from_cap} — rebuild the buffers at the shipped cap")
    return out


# --- the measurements --------------------------------------------------------------

def measure():
    corpus = load_corpus()
    # MOTIF_Z is the client's own gate and is read from its module, so the tail
    # definition cannot drift from what the app displays. One helper call resolves it
    # and everything else, so it is read from a cheap first pass over the constants.
    motif_z = _motif_z()
    mask, named = named_tail_mask(corpus, motif_z)
    out = run_helper(corpus, mask)

    # §2.3 -------------------------------------------------------------------------
    sm = out["serve_margin"]
    require(len(sm["abilities"]) == 1, "the serve-margin arm list is not the one ability")
    arms = sm["abilities"][0]["arms"]
    for name in ("shipped", "constant", "attempt_count"):
        require(name in arms, f"the simulation ran no {name!r} arm")
        require(arms[name]["attempts"] == SEEDS * HORIZON,
                f"{name}: {arms[name]['attempts']} served attempts, expected "
                f"{SEEDS * HORIZON}")
    for name, reached in sm["badge"].items():
        require(reached is not None,
                f"{name}: an unlimited-strength solver never reached a dan badge in "
                f"{BADGE_MAX_ATTEMPTS} attempts")

    # §2.4 -------------------------------------------------------------------------
    info = out["info_slot"]
    require(abs(info["motif_z"] - motif_z) < 1e-12,
            "the helper and the producer disagree about MOTIF_Z")
    tails = [t for t, is_named in zip(info["tails"], named) if is_named]
    require(tails, "no named tail survived")
    for t in tails:
        require(t["ratio"] is not None, f"axis {t['axis']} is empty in every pool")
    worst = max(tails, key=lambda t: t["ratio"])
    require(info["best_over_pool"] > 1.0,
            "the best candidate does not beat its pool's mean information")

    # §2.11 ------------------------------------------------------------------------
    comm = out["community"]
    gate = comm["rank_start_min"]
    users = comm["users"]
    by_id = {u["id"]: u for u in out["_community_meta"]["users"]}
    ranked = [u["rank"] for u in users
              if u["rank"] is not None and u["usable"] >= gate]
    everyone = [u["rank"] for u in users
                if u["rank"] is not None and by_id[u["id"]]["in_window"] >= ACTIVE_MIN]
    require(len(ranked) >= 2, f"only {len(ranked)} ranked players in the window")
    require(len(everyone) >= len(ranked),
            "the ungated population does not contain the ranked one")
    mean_ranked = float(np.mean(ranked))
    mean_everyone = float(np.mean(everyone))

    return {
        "corpus_problems": corpus["rating"].size,
        "motif_z": motif_z,
        "labels": [c.get("positive") for c in corpus["labels"]],
        "arms": arms,
        "badge": sm["badge"],
        "worst_tail": worst,
        "tails": tails,
        "best_over_pool": info["best_over_pool"],
        "fifth_advantage": info["fifth_advantage"],
        "slice_over_best": info["slice_over_best"],
        "gate": gate,
        "window_end": out["_community_meta"]["window_end"],
        "n_ranked": len(ranked), "n_everyone": len(everyone),
        "n_active": len(users),
        "mean_ranked": mean_ranked, "mean_everyone": mean_everyone,
        "shift": mean_everyone - mean_ranked,
    }


def figures():
    m = measure()
    sim = (f"simulated {_kyu(TWENTY_KYU)}, first {HORIZON} first-time "
           f"attempts, {SEEDS} seeded runs on the shipped {SIM_SIZE}×{SIM_SIZE} corpus")
    worst_label = m["labels"][m["worst_tail"]["axis"]]
    return [
        Figure("serve.onboard_over3_pct", m["arms"]["shipped"]["over_share"][0], "%.0f",
               f"share served more than {GAP_THRESHOLDS[0]:g} ranks above true rank "
               f"under the shipped serve-gap cap ({sim})"),
        Figure("serve.onboard_over5_pct", m["arms"]["shipped"]["over_share"][1], "%.0f",
               f"share served more than {GAP_THRESHOLDS[1]:g} ranks above true rank "
               f"under the shipped serve-gap cap ({sim})"),
        Figure("serve.onboard_const_over3_pct",
               m["arms"]["constant"]["over_share"][0], "%.0f",
               f"share served more than {GAP_THRESHOLDS[0]:g} ranks above true rank "
               f"under a constant margin of {CONSTANT_MARGIN:g} ({sim})"),
        Figure("serve.onboard_const_over5_pct",
               m["arms"]["constant"]["over_share"][1], "%.0f",
               f"share served more than {GAP_THRESHOLDS[1]:g} ranks above true rank "
               f"under a constant margin of {CONSTANT_MARGIN:g} ({sim})"),
        Figure("serve.onboard_count_over3_pct",
               m["arms"]["attempt_count"]["over_share"][0], "%.0f",
               f"share served more than {GAP_THRESHOLDS[0]:g} ranks above true rank "
               f"under a margin growing with the attempt count ({sim})"),
        Figure("serve.onboard_dan_attempts", m["badge"]["shipped"], "d",
               "first-time attempts before an unlimited-strength solver's SHOWN rank "
               "reaches 1 dan, under the shipped serve-gap cap"),
        Figure("serve.onboard_dan_attempts_const", m["badge"]["constant"], "d",
               f"the same under a constant margin of {CONSTANT_MARGIN:g}"),
        Figure("serve.onboard_dan_attempts_count", m["badge"]["attempt_count"], "d",
               "the same under a margin growing with the attempt count"),
        Figure("draw.info_argmax_enrichment", m["worst_tail"]["ratio"], ".1f",
               "on the worst named motif tail "
               f"({worst_label}), how much more often the top-information candidate "
               f"of a pool of {POOL_SIZE} sits in that tail than the pool itself does, "
               "over rank-prior-weighted pools on both board sizes"),
        Figure("draw.info_best_over_pool", m["best_over_pool"], ".1f",
               f"the best candidate's Fisher information over the pool's mean, "
               f"averaged over the same pools"),
        Figure("draw.info_fifth_advantage", m["fifth_advantage"], ".2f",
               "the fifth-best candidate's share of the best candidate's advantage "
               "over the pool mean, over the same pools"),
        Figure("draw.info_slice_over_best", m["slice_over_best"], ".2f",
               f"the information a uniform draw from the top {TOP_FRACTION:.0%} of the "
               "pool carries against the top candidate's own — what the slot gives up "
               "by not argmaxing"),
        Figure("community.gate_shift_ranks", abs(m["shift"]), ".1f",
               "ranks the mean shown rank moves when every player active in the "
               f"{WINDOW_DAYS}-day window is counted instead of those carrying a "
               f"rank ({m['gate']}+ valid buffered attempts)"),
        Figure("community.gate_shift_signed", m["shift"], ".1f",
               "the same shift with its sign — negative when dropping the gate moves "
               "the mean toward the weaker end"),
    ]


def _kyu(rank):
    """A simulated ability's kyu/dan label on the one internal scale (integers are
    band EDGES, so the band is the floor)."""
    band = int(np.floor(rank))
    return f"{-band}k" if band < 0 else f"{band + 1}d"


def _motif_z():
    """MOTIF_Z straight out of the client's constants module — the same gate that
    decides which motifs a problem is shown as having."""
    node = shutil.which("node")
    require(node is not None, "no `node` on PATH — the client modules run under node")
    src = ("import {MOTIF_Z} from '%s';process.stdout.write(String(MOTIF_Z));"
           % os.path.join(PWA_ROOT, "js", "constants.js"))
    proc = subprocess.run([node, "--input-type=module", "-e", src],
                          cwd=PROJECT_ROOT, capture_output=True, text=True)
    require(proc.returncode == 0,
            f"could not read MOTIF_Z from the client constants:\n{proc.stderr.strip()}")
    value = float(proc.stdout.strip())
    require(value > 0, f"MOTIF_Z is {value}, which gates nothing")
    return value


def main():
    m = measure()
    print("\n" + "=" * 78)
    print(f"corpus: {m['corpus_problems']:,} published problems; "
          f"motif gate |loading| >= {m['motif_z']:g}")
    print()
    print(f"§2.3  onboarding serve margin — a {_kyu(TWENTY_KYU)}'s first "
          f"{HORIZON} first-time attempts, {SEEDS} seeded runs, "
          f"{SIM_SIZE}×{SIM_SIZE}")
    print(f"  {'margin rule':<22}{'>3 ranks above':>16}{'>5 ranks above':>16}"
          f"{'dan badge at':>14}")
    names = {"shipped": "shipped (run+1)",
             "constant": f"constant {CONSTANT_MARGIN:g}",
             "attempt_count": "attempt count"}
    for arm, label in names.items():
        o = m["arms"][arm]["over_share"]
        print(f"  {label:<22}{100 * o[0]:>15.1f}%{100 * o[1]:>15.1f}%"
              f"{m['badge'][arm]:>14d}")
    print("  dan badge = first-time attempts before an unlimited-strength solver's")
    print("              SHOWN rank reaches the 1k/1d boundary")
    print()
    print("§2.4  the information slot, over rank-prior-weighted pools of "
          f"{POOL_SIZE} on both sizes")
    print(f"  {'named tail':<20}{'argmax rate':>13}{'pool rate':>11}{'ratio':>8}")
    for t in sorted(m["tails"], key=lambda t: -t["ratio"]):
        print(f"  {m['labels'][t['axis']]:<20}{100 * t['argmax']:>12.1f}%"
              f"{100 * t['pool']:>10.1f}%{t['ratio']:>8.2f}")
    print(f"  best candidate carries {m['best_over_pool']:.2f}x the pool's mean "
          f"information; the fifth-best keeps {m['fifth_advantage']:.2f} of that "
          f"advantage; a draw from the top {TOP_FRACTION:.0%} carries "
          f"{m['slice_over_best']:.2f} of the top candidate's information")
    print()
    print(f"§2.11 the histogram gate — {WINDOW_DAYS} days to {m['window_end']}")
    print(f"  active players in the window          : {m['n_active']}")
    print(f"  carrying a rank ({m['gate']}+ valid attempts) : {m['n_ranked']}"
          f"   mean {m['mean_ranked']:+.2f}")
    print(f"  active in the window, no gate         : {m['n_everyone']}"
          f"   mean {m['mean_everyone']:+.2f}")
    print(f"  shift                                 : {m['shift']:+.2f} ranks")


if __name__ == "__main__":
    main()
