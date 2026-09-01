#!/usr/bin/env python3
"""One arm of a refit-tier MIRT panel: the PRODUCTION training loop with a named
knob turned, its outputs and metrics written under `ablations/dev_runs/`.

This is not a second trainer. It calls `pipeline.train_mirt.run(...)` — the
shipped loop itself, whose every keyword defaults to the production constant —
with output dirs redirected into the panel's own directory, so an arm can never
touch `ratings/` or `axes/` and the loop cannot drift from what ships (the seam
was verified by a bitwise pre/post comparison of a short-budget run, and any
future drift shows up as the `shipped` arm of a panel disagreeing with a
production run at the same snapshot).

KNOBS ARE CURATED, one flag per quantity a refit-tier claim in
`docs/methodology.md` turns on, added cluster by cluster as their producers are
written — never a general override surface. Present clusters: the seed and the
budget, the slope barrier (§1.7), the learning-rate split (§1.8), the
orthogonality penalty and the taught pull (§1.5, §1.14), and the feature layout
(§1.3: the activation width, which blocks the trunk sees, and whether the
held-aside fraction is drawn over rows or over whole problems).

An arm writes:

  dev_runs/<panel>/<arm>/ratings/...   the run's full artifact set (redirected)
  dev_runs/<panel>/<arm>/axes/...
  dev_runs/<panel>/<arm>.json          metrics + config + snapshot pins + env

The JSON's `pins` block is what makes a panel COMPARABLE and lets a producer
refuse a stale one: the attempt-log frontier (`db_max_id`), the git revision,
digests of the taught bank inputs, and the machine identity — §1.8's
cross-environment caveat means arm comparisons are valid within one panel on
one machine, and the producer checks the pins agree across the arms it reads.

    ./.venv/bin/python3 -m ablations.ablation_mirt --panel barrier --arm shipped
    ./.venv/bin/python3 -m ablations.ablation_mirt --panel barrier \
        --arm barrier_off --shallow-w 0.0

Run a whole panel (skip-existing, optional concurrency) with
`ablations.run_panel`; run it on a rented box with
`vast_orchestration.py ablate`.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pipeline import train_mirt  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import (  # noqa: E402
    K_AXES, PCA_DIM, PRODUCTION_BLOCKS)

DEV_RUNS = os.path.join(PROJECT_ROOT, "ablations", "dev_runs")


def _sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def snapshot_pins(cfg=None):
    """What an arm was measured against. A producer compares these across the
    arms of a panel and against the current tree: a `fetch_database` between
    arms changes the attempt log and the stop-row draw, which voids every
    paired comparison — the same rule `docs/mirt_benchmark.md` states for its
    own tables.

    The feature-layout pins carry the arm's EFFECTIVE values, not the tree's
    constants, so an arm that sweeps one of them is pinned to what it ran with.
    A producer for such a panel is then the thing that decides which pin is
    allowed to differ across its arms — the default of requiring them all equal
    stays right for every panel that sweeps something else."""
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        db_max_id = int(con.execute(
            "SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=PROJECT_ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    import torch
    cfg = cfg or {}
    return {
        "db_max_id": db_max_id,
        "git": git,
        "labels_sha": _sha16(os.path.join(PROJECT_ROOT, "axes", "labels.json")),
        "proxies_sha": _sha16(os.path.join(PROJECT_ROOT, "axes",
                                           "motif_proxies.npz")),
        "blocks": str(cfg.get("blocks", PRODUCTION_BLOCKS)),
        "pca_dim": int(cfg.get("pca_dim", PCA_DIM)),
        "k_axes": int(cfg.get("k_axes", K_AXES)),
        "torch": torch.__version__,
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"),
    }


def run_arm(panel, arm, cfg, force=False):
    panel_dir = os.path.join(DEV_RUNS, panel)
    arm_dir = os.path.join(panel_dir, arm)
    meta_path = os.path.join(panel_dir, f"{arm}.json")
    if os.path.exists(meta_path) and not force:
        raise SystemExit(
            f"{meta_path} exists — this arm has run. An arm re-run under the "
            f"same name silently replaces the panel's record; pass --force to "
            f"mean that, or use a new arm name.")
    os.makedirs(arm_dir, exist_ok=True)

    pins = snapshot_pins(cfg)
    metrics = train_mirt.run(out_dir=os.path.join(arm_dir, "ratings"),
                             axes_dir=os.path.join(arm_dir, "axes"), **cfg)
    record = {"panel": panel, "arm": arm, "config": cfg,
              "pins": pins, "metrics": metrics}
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, meta_path)
    print(f"\nARM DONE {panel}/{arm} -> {meta_path}", flush=True)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True,
                    help="panel name; outputs go under dev_runs/<panel>/")
    ap.add_argument("--arm", required=True, help="arm name within the panel")
    ap.add_argument("--force", action="store_true",
                    help="re-run an arm whose record exists, replacing it")
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed (default: the production seed)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="epoch budget. SMOKE ONLY when not the production "
                         "budget — recorded in the arm JSON, and producers "
                         "refuse to quote a short-budget arm")
    # --- the barrier cluster (methodology §1.7) ---
    ap.add_argument("--shallow-w", type=float, default=None,
                    help="the one-sided slope barrier's weight (production "
                         f"{train_mirt.SHALLOW_W}); 0 disables it")
    ap.add_argument("--shallow-tau", type=float, default=None,
                    help="the barrier's pre-softplus threshold (production "
                         f"{train_mirt.SHALLOW_TAU})")
    # --- the learning-rate cluster (methodology §1.8) ---
    ap.add_argument("--lr-net", type=float, default=None,
                    help=f"the network's rate (production {train_mirt.LR_NET})")
    ap.add_argument("--lr-field", type=float, default=None,
                    help="the free per-user fields' rate (production "
                         f"{train_mirt.LR_FIELD})")
    # --- the penalty/identity clusters (methodology §1.5, §1.14) ---
    ap.add_argument("--ortho-rating", type=float, default=None,
                    help="the rating-orthogonality weight (production "
                         f"{train_mirt.ORTHO_RATING}); 0 disables it")
    ap.add_argument("--ortho-mode", choices=("per_tail", "whole", "off"),
                    default=None,
                    help="per_tail is what ships; whole is the linear-statistic "
                         "comparison arm; off removes the term")
    ap.add_argument("--teach-weight", type=float, default=None,
                    help="the pull toward the taught rule columns (production "
                         f"{train_mirt.TEACH_WEIGHT}); 0 frees the basis")
    # --- the feature cluster (methodology §1.3) ---
    ap.add_argument("--pca-dim", type=int, default=None,
                    help="joint-PCA activation width (production "
                         f"{PCA_DIM})")
    ap.add_argument("--blocks", default=None,
                    help="which per-problem blocks the trunk sees, as "
                         f"utils.rating.features names them (production "
                         f"{PRODUCTION_BLOCKS!r}; 'all' adds the "
                         f"answer-derived blocks)")
    ap.add_argument("--k-axes", type=int, default=None,
                    help=f"how many axes the fit carries (production {K_AXES}). "
                         f"The bank is NESTED: a smaller count keeps the first k "
                         f"components of axes/labels.json, so a sweep varies "
                         f"only the count. A count above what labels.json lists "
                         f"halts, since there is no rule to teach the extra axis")
    ap.add_argument("--stop-unit", choices=("row", "problem"), default=None,
                    help="what the held-aside fraction is drawn over. 'row' "
                         "ships and is a convergence detector; 'problem' holds "
                         "whole problems out, which is what a claim about the "
                         "FEATURE side has to be scored on")
    args = ap.parse_args(argv)

    cfg = {}
    for flag in ("seed", "epochs", "shallow_w", "shallow_tau", "lr_net",
                 "lr_field", "ortho_rating", "ortho_mode", "teach_weight",
                 "pca_dim", "blocks", "stop_unit", "k_axes"):
        v = getattr(args, flag)
        if v is not None:
            cfg[flag] = v
    run_arm(args.panel, args.arm, cfg, force=args.force)


if __name__ == "__main__":
    main()
