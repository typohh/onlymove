#!/usr/bin/env python3
"""What the availability null changes about the tendency reading — the figures
behind the fixed-even-split comparison the methodology quotes.

Drives `ablations.eval_error_null_models` in full (its default production arms:
the shipped per-cell store, the shipped fit, the whole recorded-error log) and
reads its phase-2 statistics at the null the client ships, the VISITS-LINEAR
availability null:

  motif.null_axes_measured    the shipped axis count the measurement ran over,
                              read from the same labels file the fit is taught
                              from — the count the sentences below are stated per.
  motif.null_sign_reversals   axes whose population lean (the pooled toward-share
                              of recorded errors) sits on one side of the fixed
                              even split and the OTHER side of the shipped null's
                              balance point — the menu, not the players, decided
                              the direction.
  motif.menu_worst_toward_pct on the most menu-skewed axis (the shipped null's
                              balance point farthest from an even split), the
                              share of the offered alternatives' weight on the
                              toward side.
  motif.fixed_null_worst_flag_pct   on that same axis, the share of testable
                              users (>= the client's minimum scored errors) the
                              retired fixed-50% test would flag.
  motif.shipped_null_worst_flag_pct on that same axis, the share the shipped
                              null's test flags.

    ./.venv/bin/python3 -m ablations.motif_null_figures
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations import eval_error_null_models as nulls  # noqa: E402
from ablations._figures import Figure, require  # noqa: E402

TIER = "methodology"

# The null the client ships (CLAUDE.md, methodology §2.9): selected by tag, never
# by column position — column 0 is the unweighted reference arm.
SHIPPED_TAG = "visits-lin"


def measure():
    stats = nulls.main([])
    require(isinstance(stats, dict), "eval_error_null_models returned no phase-2 statistics")
    labels, tags = stats["labels"], stats["tags"]
    require(SHIPPED_TAG in tags, f"phase-2 panel carries no {SHIPPED_TAG!r} column")
    j = tags.index(SHIPPED_TAG)
    obs = np.asarray(stats["obs"], dtype=float)
    ship = np.asarray(stats["pnull"], dtype=float)[j]
    K = len(labels)
    require(obs.shape == (K,) and ship.shape == (K,), "phase-2 statistics have the wrong shape")
    require(np.all(np.isfinite(obs)) and np.all(np.isfinite(ship)),
            "an axis has no finite toward-share — the per-cell store is behind the corpus")

    reversals = int(np.sum(np.sign(obs - 0.5) != np.sign(obs - ship)))
    worst = int(np.argmax(np.abs(ship - 0.5)))
    tested0 = float(stats["tested"][0][worst])
    tested_j = float(stats["tested"][j][worst])
    require(tested0 > 0 and tested_j > 0, "no testable users on the most menu-skewed axis")
    return {
        "axes": K, "reversals": reversals,
        "worst_axis": labels[worst],
        "menu_worst": float(ship[worst]),
        "fixed_flag_share": float(stats["flag_legacy"][worst]) / tested0,
        "shipped_flag_share": float(stats["flag"][j][worst]) / tested_j,
    }


def figures():
    m = measure()
    return [
        Figure("motif.null_axes_measured", m["axes"], "d",
               "shipped axes the availability-null measurement runs over"),
        Figure("motif.null_sign_reversals", m["reversals"], "d",
               "axes whose population error lean changes sign between the fixed even "
               "split and the shipped visits-linear availability null"),
        Figure("motif.menu_worst_toward_pct", m["menu_worst"], "%.0f",
               f"toward-side share of offered-alternative weight on the most "
               f"menu-skewed axis ({m['worst_axis']})"),
        Figure("motif.fixed_null_worst_flag_pct", m["fixed_flag_share"], "%.0f",
               "share of testable users the retired fixed-50% test flags on that axis"),
        Figure("motif.shipped_null_worst_flag_pct", m["shipped_flag_share"], "%.0f",
               "share of testable users the shipped null's test flags on that axis"),
    ]


def main():
    m = measure()
    print("\n" + "=" * 78)
    print(f"axes measured                     : {m['axes']}")
    print(f"population-lean sign reversals    : {m['reversals']}   "
          f"(fixed 0.5 vs the shipped {SHIPPED_TAG} null)")
    print(f"most menu-skewed axis             : {m['worst_axis']} "
          f"(menu toward-share {100 * m['menu_worst']:.1f}%)")
    print(f"  flagged there by fixed 0.5      : {100 * m['fixed_flag_share']:.1f}% of testable users")
    print(f"  flagged there by the shipped null: {100 * m['shipped_flag_share']:.1f}%")


if __name__ == "__main__":
    main()
