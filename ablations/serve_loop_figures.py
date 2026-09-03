"""How much the serve loop tightens the weakness reading, on a simulated player.

The client flags a motif the player is measurably worse at and then serves more
of it (`docs/methodology.md` section 2.9), so a flagged motif is sampled above
its corpus rate from then on. This producer puts a number on that: a player of
fixed true rank with a PLANTED per-motif effect answers through the shipped 4PL,
is ranked by the shipped estimator, read by the shipped accuracy reading and
served by the shipped slot draw, under three arms — the loop as shipped, the
loop opened (no motif slots, exposure at the pool's own rate), and an oracle
that serves the planted motif's slot from the first attempt (the loop with no
flag latency). The measurement half runs the client modules under node
(`ablations/serve_loop.mjs`); nothing from the attempt log enters, since the
recorded histories were themselves served under the loop.

    ./.venv/bin/python3 -m ablations.serve_loop_figures            # the table
    ./.venv/bin/python3 -m ablations.serve_loop_figures --json OUT # + the raw runs
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import AblationError, Figure, require  # noqa: E402
from ablations.client_estimator_figures import load_corpus, PWA_ROOT  # noqa: E402

TIER = "methodology"   # about twenty minutes of CPU under node, no GPU
HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve_loop.mjs")

SEED = 20260903
POOL_SIZE = 50
SIM_SIZE = 9
BOARD_SIZES = (9, 13)
HORIZON = 500
CHECKPOINTS = (50, 100, 200, 300, 499)
# True ranks on the internal scale (0 = the 1k/1d boundary): 15k, 8k, 3k, 2d.
TRUE_RANKS = (-15.0, -8.0, -3.0, 1.5)
# Planted effect in ranks per unit loading; 0 is the false-flag arm.
DELTAS = (0.0, -0.5, -1.0, -2.0)
ARMS = ("shipped", "neutral", "oracle")
SEEDS = 24


def _info_top_fraction():
    src = open(os.path.join(PWA_ROOT, "js", "problem-loader.js")).read()
    import re
    m = re.search(r"const INFO_TOP_FRACTION = ([0-9.]+);", src)
    require(m is not None, "problem-loader.js has no INFO_TOP_FRACTION")
    return float(m.group(1))


def run_helper(corpus, axes, seeds=SEEDS, true_ranks=TRUE_RANKS, deltas=DELTAS,
               arms=ARMS, horizon=HORIZON, checkpoints=CHECKPOINTS):
    node = shutil.which("node")
    require(node is not None, "no `node` on PATH — the client modules run under node")
    require(os.path.exists(HELPER), f"no {HELPER}")
    K = corpus["K"]
    for k in axes:
        comp = corpus["labels"][k]
        require(comp.get("positive") and not comp.get("negative"),
                f"axis {k} is not named on its positive tail alone; the helper's "
                "category mirror assumes that")
    n = corpus["rating"].size
    packed = np.zeros(n, dtype=[("rating", "<f4"), ("slope", "<f4"), ("floor", "<f4"),
                                ("size", "u1"), ("emb", "<f4", (K,))])
    require(packed.dtype.itemsize == 13 + 4 * K, "the corpus record is not 13+4K bytes")
    packed["rating"] = corpus["rating"]
    packed["slope"] = corpus["slope"]
    packed["floor"] = corpus["floor"]
    packed["size"] = corpus["size"]
    packed["emb"] = corpus["emb"]
    with tempfile.TemporaryDirectory(prefix="serve-loop-") as tmp:
        problems_path = os.path.join(tmp, "problems.bin")
        packed.tofile(problems_path)
        payload = {
            "pwa_root": PWA_ROOT + os.sep,
            "problems_path": problems_path,
            "seed": SEED, "K": K,
            "board_sizes": list(BOARD_SIZES),
            "pool_size": POOL_SIZE,
            "info_top_fraction": _info_top_fraction(),
            "sim": {
                "size": SIM_SIZE, "horizon": horizon, "seeds": seeds,
                "true_ranks": list(true_ranks), "axes": list(axes),
                "deltas": list(deltas), "arms": list(arms),
                "checkpoints": list(checkpoints),
            },
        }
        payload_path = os.path.join(tmp, "payload.json")
        with open(payload_path, "w") as f:
            json.dump(payload, f)
        proc = subprocess.run([node, HELPER, payload_path], cwd=PROJECT_ROOT,
                              capture_output=True, text=True)
        require(proc.returncode == 0,
                f"serve_loop.mjs failed ({proc.returncode}):\n{proc.stderr.strip()}")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise AblationError(f"serve_loop.mjs wrote no JSON: {e}") from None


def summarize(out):
    """Per (rank, axis, delta, arm, checkpoint): seed means of the planted axis's
    reading and of the exposure it was read on."""
    rows = []
    for c in out["cells"]:
        by_t = {}
        for s in c["seeds"]:
            for r in s["rows"]:
                by_t.setdefault(r["t"], []).append(r)
        for t, rs in sorted(by_t.items()):
            def mean(key, only=None):
                v = [r[key] for r in rs if r[key] is not None]
                return float(np.mean(v)) if v else float("nan")
            rows.append({
                "true_rank": c["true_rank"], "axis": c["axis"], "delta": c["delta"],
                "arm": c["arm"], "t": t, "n": len(rs),
                "est": mean("est"), "se": mean("se"), "abs_z": float(np.mean([abs(r["z"]) for r in rs if r["z"] is not None])) if any(r["z"] is not None for r in rs) else float("nan"),
                "flag": mean("flagged_weak"),
                "other_weak": float(np.sum([r["other_weak"] for r in rs]) / max(1, np.sum([r["other_testable"] for r in rs]))),
                "tail": mean("served_tail"), "motif_slot": mean("served_motif"),
                "gap": mean("served_gap"), "rank_err": mean("rank_err"),
                "conv": mean("converged"),
            })
    return rows


def print_table(rows, labels):
    hdr = ("rank  axis           delta  arm      t    est     se   |z|  flag  other  tail  mslot   gap  rkerr")
    print(hdr)
    last = None
    for r in rows:
        key = (r["true_rank"], r["axis"], r["delta"])
        if key != last and last is not None:
            print()
        last = key
        print(f"{r['true_rank']:5.1f}  {labels[r['axis']]['positive'][:13]:13s} {r['delta']:5.1f}  "
              f"{r['arm']:8s}{r['t']:4d}  {r['est']:5.2f}  {r['se']:5.2f}  {r['abs_z']:4.2f}  "
              f"{r['flag']:4.2f}  {r['other_weak']:5.2f}  {r['tail']:4.2f}  {r['motif_slot']:5.2f}  "
              f"{r['gap']:5.2f}  {r['rank_err']:5.2f}")


def _cells(out, delta, arm, t):
    """Every (rank, axis, seed) row of one arm at one checkpoint, in grid order so
    two arms' lists pair row for row on the same seed stream."""
    rows = []
    for c in out["cells"]:
        if c["delta"] != delta or c["arm"] != arm:
            continue
        for s in c["seeds"]:
            r = next(r for r in s["rows"] if r["t"] == t)
            rows.append({**r, "ever_flagged": s["ever_flagged"]})
    return rows


def measure(out=None):
    corpus = load_corpus()
    if out is None:
        out = run_helper(corpus, list(range(corpus["K"])))
    t = CHECKPOINTS[-1]
    require(out["K"] == corpus["K"], "the helper ran on a different axis count")
    unit = -1.0
    require(unit in DELTAS and 0.0 in DELTAS, "the grid needs the unit and the zero effect")

    def arm(delta, name):
        return _cells(out, delta, name, t)

    def mean(rows, key):
        v = [r[key] for r in rows if r[key] is not None]
        require(v, f"no {key} at t={t}")
        return float(np.mean(v))

    neutral, shipped, oracle = (arm(unit, a) for a in ("neutral", "shipped", "oracle"))
    # Paired per seed stream, then averaged: the SE the loop leaves against the
    # SE the opened loop leaves, on the same player and dice.
    def se_ratio(a, b):
        pairs = [(x["se"], y["se"]) for x, y in zip(a, b) if x["se"] and y["se"]]
        require(len(pairs) > len(a) // 2, "too few paired readings")
        return float(np.mean([x for x, _ in pairs]) / np.mean([y for _, y in pairs]))

    # The estimator's offset under the loop: shipped minus neutral, paired, over
    # every non-zero planted size, and separately at zero.
    def offset(delta):
        d = [x["est"] - y["est"] for x, y in zip(arm(delta, "shipped"), arm(delta, "neutral"))
             if x["est"] is not None and y["est"] is not None]
        return float(np.mean(d)), float(np.std(d) / np.sqrt(len(d)))
    nonzero = [offset(d) for d in DELTAS if d != 0.0]
    off_mean = float(np.mean([m for m, _ in nonzero]))
    off_se = float(np.sqrt(np.mean([e * e for _, e in nonzero]) / len(nonzero)))
    zero_mean, zero_se = offset(0.0)
    flagged = [r["est"] for r in shipped if r["ever_flagged"] and r["est"] is not None]

    z0n, z0s = arm(0.0, "neutral"), arm(0.0, "shipped")
    strong = DELTAS[-1]
    return {
        "t": t, "unit": unit, "strong": strong,
        "se_neutral": mean(neutral, "se"), "se_shipped": mean(shipped, "se"),
        "se_oracle": mean(oracle, "se"),
        "se_cut_shipped": 1 - se_ratio(shipped, neutral),
        "se_cut_oracle": 1 - se_ratio(oracle, neutral),
        "tail_neutral": mean(neutral, "served_tail"), "tail_shipped": mean(shipped, "served_tail"),
        "tail_oracle": mean(oracle, "served_tail"),
        "true_flag_neutral": mean(neutral, "flagged_weak"),
        "true_flag_shipped": mean(shipped, "flagged_weak"),
        "false_flag_neutral": mean(z0n, "flagged_weak"), "false_flag_shipped": mean(z0s, "flagged_weak"),
        "offset": off_mean, "offset_se": off_se, "offset_zero": zero_mean, "offset_zero_se": zero_se,
        "flagged_recovery": float(np.mean(flagged)), "flagged_n": len(flagged),
        "rank_absorb_neutral": mean(arm(strong, "neutral"), "rank_err"),
        "rank_absorb_shipped": mean(arm(strong, "shipped"), "rank_err"),
        "rank_absorb_oracle": mean(arm(strong, "oracle"), "rank_err"),
        "K": out["K"],
    }


def figures(out=None):
    m = measure(out)
    sim = (f"simulated players at {len(TRUE_RANKS)} true ranks, one planted axis at a "
           f"time over all {m['K']}, {SEEDS} seeded runs per cell, {HORIZON} first-time "
           f"attempts on the shipped {SIM_SIZE}×{SIM_SIZE} corpus, read at attempt {m['t']}")
    unit = f"a planted weakness of {-m['unit']:g} rank per unit loading"
    return [
        Figure("loop.se_cut_shipped_pct", m["se_cut_shipped"], "%.0f",
               f"share of the accuracy reading's standard error the shipped serve loop "
               f"removes against the loop opened (no motif slots), paired per run, {unit} ({sim})"),
        Figure("loop.se_cut_oracle_pct", m["se_cut_oracle"], "%.0f",
               "the same for a loop serving the planted motif's slot from the first attempt "
               "— zero flag latency"),
        Figure("loop.se_neutral_ranks", m["se_neutral"], ".2f",
               f"the reading's standard error in ranks per unit loading with the loop opened, {unit}"),
        Figure("loop.se_shipped_ranks", m["se_shipped"], ".2f", "the same under the shipped loop"),
        Figure("loop.tail_neutral_pct", m["tail_neutral"], "%.0f",
               "share of served problems in the planted motif's tail with the loop opened — "
               "the pool's own rate"),
        Figure("loop.tail_shipped_pct", m["tail_shipped"], "%.0f", "the same under the shipped loop"),
        Figure("loop.tail_oracle_pct", m["tail_oracle"], "%.0f", "the same under the zero-latency loop"),
        Figure("loop.true_flag_neutral_pct", m["true_flag_neutral"], "%.0f",
               f"share of runs flagging {unit} at the gate, loop opened"),
        Figure("loop.true_flag_shipped_pct", m["true_flag_shipped"], "%.0f",
               "the same under the shipped loop"),
        Figure("loop.false_flag_neutral_pct", m["false_flag_neutral"], "%.0f",
               "share of runs flagging a weakness on an axis with NO planted effect, loop opened"),
        Figure("loop.false_flag_shipped_pct", m["false_flag_shipped"], "%.0f",
               "the same under the shipped loop"),
        Figure("loop.offset_ranks", m["offset"], ".2f",
               "the reading under the shipped loop minus the reading with it opened, paired per "
               "run and averaged over the non-zero planted sizes — positive is optimistic"),
        Figure("loop.offset_se", m["offset_se"], ".2f", "standard error of that offset"),
        Figure("loop.offset_zero_ranks", m["offset_zero"], ".2f",
               "the same offset with no planted effect"),
        Figure("loop.offset_zero_se", m["offset_zero_se"], ".2f", "its standard error"),
        Figure("loop.flagged_recovery_ranks", m["flagged_recovery"], ".2f",
               f"mean reading under the shipped loop over the runs that ever flagged the planted "
               f"axis, {unit} (n={m['flagged_n']})"),
        Figure("loop.rank_absorb_neutral_ranks", m["rank_absorb_neutral"], ".1f",
               f"rank-estimate error for a planted weakness of {-m['strong']:g} ranks per unit "
               "loading, loop opened"),
        Figure("loop.rank_absorb_shipped_ranks", m["rank_absorb_shipped"], ".1f",
               "the same under the shipped loop"),
        Figure("loop.rank_absorb_oracle_ranks", m["rank_absorb_oracle"], ".1f",
               "the same under the zero-latency loop"),
        # Guard figures: a guard compares one key against literals, so every
        # between-key claim the passage makes is emitted as its own number.
        Figure("loop.oracle_cuts_more", float(m["se_cut_oracle"] > m["se_cut_shipped"]), "d",
               "1 when the zero-latency loop cuts the standard error more than the shipped one"),
        Figure("loop.false_flag_falls", float(m["false_flag_shipped"] < m["false_flag_neutral"]), "d",
               "1 when the shipped loop leaves fewer false weakness flags than the opened loop"),
        Figure("loop.true_flag_gap_pct", m["true_flag_shipped"] - m["true_flag_neutral"], "%.1f",
               "shipped minus opened share of runs flagging the planted weakness"),
        Figure("loop.offset_z", m["offset"] / m["offset_se"], ".1f",
               "the offset over its standard error"),
        Figure("loop.offset_zero_z", m["offset_zero"] / m["offset_zero_se"], ".1f",
               "the zero-effect offset over its standard error"),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the helper's raw output here")
    ap.add_argument("--axes", default=None, help="comma-separated axis indices (default all)")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--ranks", default=None, help="comma-separated true ranks")
    ap.add_argument("--deltas", default=None)
    ap.add_argument("--from-json", help="re-read a saved raw output instead of running")
    args = ap.parse_args(argv)
    corpus = load_corpus()
    axes = [int(a) for a in args.axes.split(",")] if args.axes else list(range(corpus["K"]))
    ranks = [float(x) for x in args.ranks.split(",")] if args.ranks else TRUE_RANKS
    deltas = [float(x) for x in args.deltas.split(",")] if args.deltas else DELTAS
    if args.from_json:
        out = json.load(open(args.from_json))
    else:
        out = run_helper(corpus, axes, seeds=args.seeds, true_ranks=ranks, deltas=deltas)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f)
    print_table(summarize(out), corpus["labels"])
    if not args.axes and not args.ranks and not args.deltas and args.seeds == SEEDS:
        print()
        for f in figures(out):
            print(f"{f.key:36s} {f.text:>8s}   {f.note}")


if __name__ == "__main__":
    main()
