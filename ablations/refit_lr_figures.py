#!/usr/bin/env python3
"""The learning-rate split's figures (methodology §1.8, repeated in §3.5's
constants table), read from the `lr` refit panel.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, internally inconsistent, or STALE against the current attempt log —
the pins rule `docs/mirt_benchmark.md` states, mechanized. Guards and spans keep
evaluating against the last-measured ledger values, so the panel's cost is paid
only when it is deliberately re-run.

WHAT THE PANEL VARIES. Production trains the shared network at LR_NET and the
free per-user fields (the abilities, the axis positions, the lapse) at LR_FIELD,
a ratio of about thirty-fold. The arms move one rate at a time: `ratio1x` pulls
the field rate down ONTO the network's, which is the single-shared-rate
configuration §1.8 says the split exists to avoid; `ratio10x` sits at the bottom
of the saturating region; `net_slow` keeps the shipped field rate and thirds the
network's, which is the "two values a factor of three apart" comparison.

DIVERGENCE IS A MEASUREMENT HERE, NOT AN INPUT ERROR. The shared-rate arm is
expected to blow up — that IS the claim — so a non-finite or grossly risen
held-out trace is read out as `lr.shared_rate_diverged` rather than raised. Only
the SHIPPED arm is required to be finite throughout: it is the baseline every
paired delta is taken against, so a diverging baseline voids the panel.

  lr.shared_rate_diverged      1 when the single-shared-rate arm diverged, 0
                               when it did not. A 1/0 indicator, meant for a
                               `<!-- fig-guard: lr.shared_rate_diverged == 1 -->`
                               beside §1.8's "diverges late and in evaluation
                               mode" rather than for a span.
  lr.shared_rate_diverge_epoch the epoch at which it first did so (0 = never) —
                               §1.8's "late" as a number.
  lr.shared_rate_cost_mnat     its held-out (stop-split) cost against the shipped
                               arm in millinats, at the last checkpoint both
                               traces are finite.
  lr.ratio10x_cost_mnat        the same cost at a ten-fold ratio — §1.8's "below
                               ten-fold the fit loses several millinats".
  lr.net_slow_cost_mnat        the same cost for the three-times-slower network
                               at the shipped ratio; the claim is a TIE, so this
                               is the figure that is meant to be small, and it is
                               signed (positive = the slower net is worse).
  lr.shipped_converge_epoch    where each of those two arms reaches its own
  lr.net_slow_converge_epoch   plateau, defined below.
  lr.net_slow_converge_ratio   the second over the first — §1.8's "reaches it in
                               half the epochs", as the factor it actually is.

Two definitions, both stated here because both are choices:

CONVERGENCE. §1.8 records that the held-out score reaches a band a thousandth
wide and then wanders inside it, so an arm's plateau is its own final value plus
that band: `converge_epoch` is the FIRST checkpoint from which the arm's
held-out score never again rises more than CONVERGED_MARGIN_MNAT millinats above
its own final value. Sustained rather than first-touch, so a transient dip on the
way down cannot claim the plateau.

DIVERGENCE. Non-finite at any checkpoint, or risen more than
DIVERGENCE_RISE_MNAT millinats above the arm's OWN RUNNING BEST — running rather
than final, so the test is causal and the epoch it reports is the epoch a watcher
would have seen it at. The threshold is twenty times the width of the settled
band, which is what separates a blow-up from the wandering §1.8 describes.

    ./.venv/bin/python3 -m ablations.refit_lr_figures
"""

import json
import math
import os
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from pipeline.train_mirt import EPOCHS, LR_FIELD, LR_NET  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402

TIER = "refit"

PANEL = "lr"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
ARMS = ("shipped", "ratio1x", "ratio10x", "net_slow")

# The settled band §1.8 describes is a thousandth of a nat wide. An arm has
# reached its plateau once it stays inside one band of its own final value...
CONVERGED_MARGIN_MNAT = 1.0
# ...and has diverged once it stands twenty bands above its own running best,
# which no amount of wandering inside that band can reach.
DIVERGENCE_RISE_MNAT = 20.0


# --- the panel -------------------------------------------------------------

def load_panel():
    require(os.path.exists(MANIFEST),
            f"no {MANIFEST} — the panel manifest is what the arm records are "
            f"checked against, so it cannot be read without it")
    manifest = json.load(open(MANIFEST))
    require(set(manifest["arms"]) == set(ARMS),
            f"{MANIFEST} declares arms {sorted(manifest['arms'])}, this "
            f"producer reads {sorted(ARMS)}")

    arms = {}
    for arm in ARMS:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the lr panel has not run. Run it with "
                f"`python -m ablations.run_panel lr` (or "
                f"`vast_orchestration.py ablate lr`) first.")
        arms[arm] = json.load(open(path))

    for arm, r in arms.items():
        require(r["metrics"]["epochs"] == EPOCHS,
                f"{arm} ran {r['metrics']['epochs']} epochs, not the production "
                f"budget of {EPOCHS} — a smoke arm is not quotable")
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == EPOCHS,
                f"{arm}'s convergence trace does not reach its budget")
        # The record has to be the arm this producer thinks it is: an arm run by
        # hand with a different flag is a well-formed record of a different
        # experiment, and every figure below would be attributed to the wrong
        # configuration.
        require(r["config"] == manifest["arms"][arm],
                f"{arm} was run with {r['config']!r}, but the manifest declares "
                f"{manifest['arms'][arm]!r} — the record is not this arm")

    base = arms["shipped"]["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha",
                    "blocks", "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} ({r['pins'][pin]!r}) differs from the "
                    f"shipped arm's ({base[pin]!r}) — the arms are not one "
                    f"panel and their comparison is void")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"the panel was measured at attempt-log frontier {base['db_max_id']} "
            f"but the log now ends at {now} — the stop-row draw moved, the "
            f"paired comparison is void. Re-run the panel.")

    # What each arm's rates actually WERE, and the ratios the claims are about.
    # Stated here rather than only in the manifest, because a producer that
    # cannot say which ratio it measured cannot be checked.
    rates = {arm: (float(r["config"].get("lr_net", LR_NET)),
                   float(r["config"].get("lr_field", LR_FIELD)))
             for arm, r in arms.items()}
    for arm, want in (("ratio1x", 1.0), ("ratio10x", 10.0)):
        net, field = rates[arm]
        require(math.isclose(field / net, want, rel_tol=1e-9),
                f"{arm} runs at a field/net ratio of {field / net:.4g}, not "
                f"{want:g} — the arm does not measure the point it is named for")
    require(math.isclose(rates["net_slow"][0], LR_NET / 3.0, rel_tol=1e-9)
            and rates["net_slow"][1] == LR_FIELD,
            f"net_slow runs at {rates['net_slow']}, not the shipped field rate "
            f"with a third of the shipped net rate")
    return arms, rates


# --- reading a convergence trace -------------------------------------------

def stop_curve(record):
    """(epochs, held-out BCE) from an arm's convergence trace."""
    trace = record["metrics"]["trace"]
    return ([int(r["epoch"]) for r in trace],
            [float(r["stop_bce"]) for r in trace])


def diverge_epoch(epochs, bce):
    """The first checkpoint at which the held-out score is non-finite or stands
    DIVERGENCE_RISE_MNAT millinats above the arm's own RUNNING best; 0 = never.
    Running rather than final, so the test only ever looks backwards."""
    best = math.inf
    for e, v in zip(epochs, bce):
        if not math.isfinite(v):
            return e
        if v > best + DIVERGENCE_RISE_MNAT / 1e3:
            return e
        best = min(best, v)
    return 0


def converge_epoch(epochs, bce):
    """The first checkpoint from which the score never again rises more than
    CONVERGED_MARGIN_MNAT millinats above the arm's own FINAL value."""
    require(all(math.isfinite(v) for v in bce),
            "converge_epoch on a trace that is not finite throughout — an arm "
            "that diverged has no plateau to reach")
    bar = bce[-1] + CONVERGED_MARGIN_MNAT / 1e3
    i = len(epochs) - 1
    while i >= 0 and bce[i] <= bar:
        i -= 1
    return epochs[i + 1]


def paired_cost_mnat(arm_bce, ship_bce, epochs):
    """An arm's held-out cost against the shipped arm, in millinats, at the LAST
    checkpoint both traces are finite. For an arm that stayed finite that is the
    final epoch; for one that blew up past representability it is the last epoch
    the comparison still exists at, which is reported alongside."""
    ok = [i for i in range(len(epochs))
          if math.isfinite(arm_bce[i]) and math.isfinite(ship_bce[i])]
    require(ok, "no checkpoint has both arms finite — nothing is comparable")
    i = ok[-1]
    return (arm_bce[i] - ship_bce[i]) * 1e3, epochs[i]


# --- the measurement -------------------------------------------------------

def measure():
    arms, rates = load_panel()
    curves = {arm: stop_curve(r) for arm, r in arms.items()}
    grid = curves["shipped"][0]
    for arm, (ep, _) in curves.items():
        require(ep == grid,
                f"{arm}'s trace is scored at different epochs than the shipped "
                f"arm's — the deltas would not be like for like")

    ship_bce = curves["shipped"][1]
    require(all(math.isfinite(v) for v in ship_bce),
            "the shipped arm's held-out trace is not finite throughout — the "
            "baseline every paired delta is taken against diverged, so nothing "
            "in this panel is quotable. Re-run it.")
    require(all(math.isfinite(v) for v in curves["net_slow"][1]),
            "the net_slow arm diverged. §1.8's claim about it is a TIE reached "
            "more slowly, so a diverged trace is not a smaller version of that "
            "finding — it is a different one, and this producer has no key for "
            "it.")

    d_epoch = diverge_epoch(grid, curves["ratio1x"][1])
    shared_cost, shared_at = paired_cost_mnat(curves["ratio1x"][1], ship_bce, grid)
    tenx_cost, _ = paired_cost_mnat(curves["ratio10x"][1], ship_bce, grid)
    slow_cost, _ = paired_cost_mnat(curves["net_slow"][1], ship_bce, grid)

    ship_conv = converge_epoch(grid, ship_bce)
    slow_conv = converge_epoch(grid, curves["net_slow"][1])
    return {
        "rates": rates,
        "shared_diverged": 1 if d_epoch else 0,
        "shared_diverge_epoch": d_epoch,
        "shared_cost": shared_cost,
        "shared_cost_at": shared_at,
        "tenx_cost": tenx_cost,
        "slow_cost": slow_cost,
        "ship_conv": ship_conv,
        "slow_conv": slow_conv,
        "slow_conv_ratio": slow_conv / ship_conv,
        "ship_final": ship_bce[-1],
        "gpu": arms["shipped"]["pins"]["gpu"],
    }


def figures():
    m = measure()
    return [
        Figure("lr.shared_rate_diverged", m["shared_diverged"], "d",
               "1 when the single-shared-rate arm (field rate pulled onto the "
               f"network's) diverged, 0 when it did not — non-finite, or "
               f"{DIVERGENCE_RISE_MNAT:g} millinats above its own running best "
               f"held-out score. An indicator for a fig-guard, not a span"),
        Figure("lr.shared_rate_diverge_epoch", m["shared_diverge_epoch"], "d",
               "the epoch at which the single-shared-rate arm first met that "
               "test; 0 means it never did"),
        Figure("lr.shared_rate_cost_mnat", m["shared_cost"], ".0f",
               "held-out (stop-split) cost of one shared rate against the "
               "shipped split, millinats, at the last checkpoint both traces "
               "are finite — PAIRED: every arm of this panel runs at the "
               "production seed and so holds aside the same stop rows"),
        Figure("lr.ratio10x_cost_mnat", m["tenx_cost"], ".1f",
               "held-out cost at a ten-fold field/net ratio against the shipped "
               "~33-fold, millinats, paired stop rows — the bottom of the "
               "saturating region"),
        Figure("lr.net_slow_cost_mnat", m["slow_cost"], ".2f",
               "held-out cost of a three-times-slower network at the shipped "
               "field rate, millinats, paired stop rows; signed, positive means "
               "the slower network is the worse fit. The claim is a tie"),
        Figure("lr.shipped_converge_epoch", m["ship_conv"], "d",
               "first epoch from which the shipped arm's held-out score never "
               f"again rises {CONVERGED_MARGIN_MNAT:g} millinat above its own "
               f"final value — the band §1.8 says the fit settles into"),
        Figure("lr.net_slow_converge_epoch", m["slow_conv"], "d",
               "the same epoch for the three-times-slower network"),
        Figure("lr.net_slow_converge_ratio", m["slow_conv_ratio"], ".1f",
               "how many times longer the slower network takes to reach its own "
               "plateau, at the same held-out margin"),
    ]


def main():
    m = measure()
    print(f"lr panel  (all arms full budget, production seed, one machine: "
          f"{m['gpu']})")
    print(f"  convergence = first epoch from which the held-out score never "
          f"again rises {CONVERGED_MARGIN_MNAT:g} mnat above its own final "
          f"value")
    print(f"  divergence  = non-finite, or {DIVERGENCE_RISE_MNAT:g} mnat above "
          f"the arm's own running best")
    print()
    print(f"{'arm':>10}{'lr_net':>10}{'lr_field':>10}{'ratio':>8}"
          f"{'cost mnat':>12}{'converged':>11}")
    for arm in ARMS:
        net, field = m["rates"][arm]
        cost = {"shipped": 0.0, "ratio1x": m["shared_cost"],
                "ratio10x": m["tenx_cost"], "net_slow": m["slow_cost"]}[arm]
        # Convergence is scored only for the two arms §1.8's "half the epochs"
        # claim compares; the ratio arms are about the fit reached, not the
        # epoch it was reached at.
        conv = {"shipped": m["ship_conv"], "net_slow": m["slow_conv"]}.get(arm)
        print(f"{arm:>10}{net:>10.1e}{field:>10.1e}{field / net:>7.0f}x"
              f"{cost:>12.2f}{(str(conv) if conv else 'not scored'):>11}")
    print()
    print(f"  cost is against the shipped arm on the SAME held-out rows "
          f"(shipped final {m['ship_final']:.4f}); positive = worse")
    if m["shared_diverged"]:
        print(f"  the shared-rate arm DIVERGED at epoch "
              f"{m['shared_diverge_epoch']}, and its cost above is read at "
              f"epoch {m['shared_cost_at']}, the last checkpoint both traces "
              f"are finite")
    else:
        print(f"  the shared-rate arm did NOT diverge — §1.8's stated reason "
              f"for the split is not reproduced on this panel")
    print(f"  the slower network reaches its plateau "
          f"{m['slow_conv_ratio']:.1f}x later than the shipped rate")


if __name__ == "__main__":
    main()
