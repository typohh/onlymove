#!/usr/bin/env python3
"""Run-to-run determinism (methodology §1.8), read off repeats the panels
already contain.

§1.8 records that the fixed seed is sufficient only because nothing else in the
stack varies, and that this was checked rather than assumed: a repeated run on
one machine reproduces every per-problem output BITWISE. That check needs two
independent invocations of one configuration on one machine, and the refit
panels supply them for free — several panels carry a `shipped` arm, each run in
its own process, and any two arms anywhere under `dev_runs/` whose EFFECTIVE
configuration is identical are exactly such a repeat.

EFFECTIVE means the recorded config together with the budget and the schedule
the arm actually ran, which are properties of the run rather than of the
trainer's constants at the moment this is read. Two arms of one config at two
budgets are not a repeat, and nothing here is measured against the shipped
budget: a pair keeps its meaning after the baseline moves, and what such a move
costs is only that the pairs on disk describe the PREVIOUS regime until a panel
is re-measured. The printout says which regime each pair ran under for that
reason.

So this producer trains nothing and adds no arms. It groups every arm record by
its effective configuration, keeps the groups holding more than one arm, and
compares those arms' written artifacts byte for byte. The evidence grows on its
own as panels are added: a second configuration repeated across two panels
becomes a second pair with no manifest change anywhere.

WHAT A DIFFERENCE WOULD MEAN. The comparison is between processes on one
machine at one revision, which is the only claim §1.8 makes — across machines
or library builds nothing is pinned, and the section says so. A pair that
disagrees therefore reports a real defect (an unseeded draw, a nondeterministic
kernel reached by a new code path, an artifact carrying a timestamp), never an
expected environmental difference, which is why the count of differing files
backs a guard set to zero rather than a quoted number.

    ./.venv/bin/python3 -m ablations.refit_determinism_figures
"""

import json
import hashlib
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
# Imported as a MODULE and read in `main` alone, for one line of printed
# context. Nothing here selects, groups or checks an arm against the trainer's
# current constants — a repeat is a repeat under the regime it ran.
from pipeline import train_mirt  # noqa: E402

TIER = "refit"

DEV_RUNS = os.path.join(PROJECT_ROOT, "ablations", "dev_runs")

# The per-problem outputs §1.8's claim is about, relative to an arm's directory.
# Named rather than globbed: a glob would quietly report "0 files differ" if an
# arm's artifacts were absent, which is the one answer this producer must never
# give by accident.
PER_PROBLEM = tuple(
    os.path.join("ratings", str(size), name)
    for size in (9, 13)
    for name in ("tsumego_ratings.csv", "tsumego_slopes.csv",
                 "tsumego_floors.csv")
) + (os.path.join("axes", "per_problem.csv"),)
# The rest of a run's output. Held apart from the per-problem files because the
# claim names those, and because a model checkpoint differing while the derived
# per-problem values agree would be a different (and milder) finding.
OTHER = (os.path.join("ratings", "user_theta.csv"),
         os.path.join("ratings", "mirt_model.pt"),
         os.path.join("ratings", "rating_constants.json"),
         os.path.join("axes", "axes.npz"))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(path):
    """Data rows in a CSV, header excluded."""
    with open(path, "rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def regime_of(record):
    """The (budget, schedule) an arm actually ran — what a repeat has to match
    beyond its config, since one config at two budgets is two experiments."""
    metrics = record["metrics"]
    return int(metrics["epochs"]), str(metrics.get("lr_schedule", "constant"))


def load_arms():
    """Every arm record under dev_runs, as (panel, arm, record, arm_dir)."""
    out = []
    for panel in sorted(os.listdir(DEV_RUNS)) if os.path.isdir(DEV_RUNS) else []:
        panel_dir = os.path.join(DEV_RUNS, panel)
        if not os.path.isdir(panel_dir):
            continue
        for entry in sorted(os.listdir(panel_dir)):
            if not entry.endswith(".json"):
                continue
            arm = entry[:-len(".json")]
            record = json.load(open(os.path.join(panel_dir, entry)))
            out.append((panel, arm, record,
                        os.path.join(panel_dir, arm)))
    return out


def repeat_pairs(arms):
    """Pairs of arms that ran the SAME configuration in separate processes.

    Grouped by the EFFECTIVE configuration — the recorded config plus the budget
    and schedule the run actually used — and by the pins that decide whether two
    runs are even comparable: one machine, one revision, one attempt-log
    frontier, because §1.8's claim is scoped to exactly that. A smoke arm groups
    with nothing, its short budget being part of what it is."""
    groups = {}
    for panel, arm, record, arm_dir in arms:
        pins = record["pins"]
        key = (json.dumps(record["config"], sort_keys=True), regime_of(record),
               pins["gpu"], pins["git"], pins["db_max_id"],
               pins["blocks"], pins["pca_dim"], pins["k_axes"],
               pins["labels_sha"], pins["proxies_sha"], pins["torch"])
        groups.setdefault(key, []).append((panel, arm, record, arm_dir))
    pairs = []
    for members in groups.values():
        # Consecutive members only: n runs of one configuration give n-1
        # independent comparisons, and every pair beyond that is implied.
        for a, b in zip(members, members[1:]):
            pairs.append((a, b))
    return pairs


def measure():
    arms = load_arms()
    require(arms,
            f"no arm records under {DEV_RUNS} — this producer reads "
            f"repeats out of panels that have already run")
    pairs = repeat_pairs(arms)
    require(pairs,
            "no two arms under dev_runs share a configuration, so the panels "
            "contain no repeat to read determinism off. A panel carrying the "
            "shipped configuration is what supplies one; run a second such "
            "panel rather than adding an arm for this producer alone.")

    compared = differing = 0
    per_problem_compared = per_problem_differing = 0
    problems = None
    traces_identical = True
    for (pa, aa, ra, da), (pb, ab, rb, db) in pairs:
        for rel in PER_PROBLEM + OTHER:
            fa, fb = os.path.join(da, rel), os.path.join(db, rel)
            require(os.path.exists(fa) and os.path.exists(fb),
                    f"{rel} is missing from {pa}/{aa} or {pb}/{ab} — an arm "
                    f"whose artifacts were cleaned up cannot be compared, and "
                    f"skipping it would understate the check")
            same = _sha(fa) == _sha(fb)
            compared += 1
            differing += 0 if same else 1
            if rel in PER_PROBLEM:
                per_problem_compared += 1
                per_problem_differing += 0 if same else 1
        n = sum(_rows(os.path.join(da, rel)) for rel in PER_PROBLEM
                if rel.endswith("tsumego_ratings.csv"))
        if problems is None:
            problems = n
        require(problems == n,
                f"{pa}/{aa} and an earlier pair cover different problem counts "
                f"({n} against {problems}) — the arms are not describing one "
                f"corpus and the comparison is not a repeat")
        ta = [r["stop_bce"] for r in ra["metrics"]["trace"]]
        tb = [r["stop_bce"] for r in rb["metrics"]["trace"]]
        traces_identical = traces_identical and ta == tb

    return {"pairs": len(pairs), "compared": compared, "differing": differing,
            "pp_compared": per_problem_compared,
            "pp_differing": per_problem_differing,
            "problems": problems, "traces_identical": traces_identical,
            "gpu": pairs[0][0][2]["pins"]["gpu"],
            "regimes": sorted({regime_of(a[2]) for a, _ in pairs}),
            "examples": [(a[0] + "/" + a[1], b[0] + "/" + b[1],
                          regime_of(a[2])) for a, b in pairs]}


def figures():
    m = measure()
    return [
        Figure("determinism.repeat_pairs", m["pairs"], "d",
               "independent repeats found among the refit panels' arms — pairs "
               "of separate processes that ran one configuration on one machine "
               "at one revision"),
        Figure("determinism.repeat_problem_files_differing",
               m["pp_differing"], "d",
               "per-problem output files that differ between the two runs of a "
               "repeat. §1.8's bitwise claim is this being zero, which is what "
               "a fig-guard holds it to, not a number worth quoting"),
        Figure("determinism.repeat_files_differing", m["differing"], "d",
               "the same over EVERY output a run writes, the model checkpoint "
               "and the axis artifacts included"),
        Figure("determinism.repeat_files_compared", m["compared"], "d",
               "output files compared across all repeats — the size of the "
               "check behind the zero above"),
        Figure("determinism.repeat_problems", float(m["problems"]), "~1000",
               "problems whose full parameter row was compared in each repeat"),
        Figure("determinism.repeat_traces_identical",
               1 if m["traces_identical"] else 0, "d",
               "1 when every repeat's held-out convergence trace matches "
               "value for value, 0 otherwise — the training path itself "
               "repeating, not only what it wrote"),
    ]


def main():
    m = measure()
    print(f"repeats found: {m['pairs']}  (one machine: {m['gpu']})")
    for a, b, (budget, schedule) in m["examples"]:
        print(f"  {a}  vs  {b}   [{budget} epochs, {schedule} rate]")
    current = (train_mirt.EPOCHS, train_mirt.LR_SCHEDULE)
    if current not in m["regimes"]:
        print(f"  NOTE: no repeat runs the trainer's current regime "
              f"({current[0]} epochs, {current[1]} rate) — the bitwise claim is "
              f"so far verified on the previous regime only.")
    print(f"per-problem files: {m['pp_compared'] - m['pp_differing']}"
          f"/{m['pp_compared']} identical  "
          f"({m['problems']} problems per repeat)")
    print(f"all outputs:       {m['compared'] - m['differing']}"
          f"/{m['compared']} identical")
    print(f"held-out traces:   "
          f"{'identical' if m['traces_identical'] else 'DIFFER'}")


if __name__ == "__main__":
    main()
