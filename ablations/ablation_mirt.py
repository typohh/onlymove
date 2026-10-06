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
held-aside fraction is drawn over rows or over whole problems, or is the
per-problem spread holdout of §1.5), the axis count
(§1.14), the two dropout sites with the weight decay §1.5 rejects as their
substitute, the schedule (§1.8: the decay against the constant rate, and
the opening ramp the axis-space panel measures), and the ability prior and
the base-rate start (§1.5, §1.8).

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
`auxiliary/vast_orchestration.py ablate`.
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
# What every arm holds aside to be scored on — rows or whole problems, as its
# stop unit says. The shipped fit holds nothing aside, so this is the arm
# runner's own and is not part of an arm's config: a shipped arm's config stays
# empty while it is still measured.
HELD_ASIDE_FRAC = 0.03


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
        "act_source": str(cfg.get("act_source", "store")),
        "k_axes": int(cfg.get("k_axes", K_AXES)),
        # THE REGIME: what a baseline change moves. Effective values, so an arm
        # that sweeps either is pinned to what it ran with, and a record written
        # under a previous baseline carries the one it was measured against —
        # its config is `{}` either way, which is exactly why the config alone
        # cannot tell a stale shipped arm from a current one. The regime's third
        # component, the mutual term's weight and detachment, is read off the
        # metrics (`panel_regime`), where run() records every arm's effective
        # value.
        "epochs": int(cfg.get("epochs", train_mirt.EPOCHS)),
        "lr_schedule": str(cfg.get("lr_schedule", train_mirt.LR_SCHEDULE)),
        "floor_mode": str(cfg.get("floor_mode", train_mirt.FLOOR_MODE)),
        "theta_laplace": bool(cfg.get("theta_laplace", train_mirt.THETA_LAPLACE)),
        "init_base_rate": bool(cfg.get("init_base_rate", train_mirt.INIT_BASE_RATE)),
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
                             axes_dir=os.path.join(arm_dir, "axes"),
                             stop_row_frac=HELD_ASIDE_FRAC, **cfg)
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
                    help=f"epoch budget (production {train_mirt.EPOCHS}). A "
                         f"panel is one budget: every producer requires the "
                         f"arms it reads to share one, so a short arm is a "
                         f"smoke arm and is never quotable beside full ones; "
                         f"a budget SWEEP is a panel whose manifest declares "
                         f"the budgets, read by its own producer")
    # --- the barrier cluster (methodology §1.7) ---
    ap.add_argument("--shallow-w", type=float, default=None,
                    help="the one-sided slope barrier's weight (production "
                         f"{train_mirt.SHALLOW_W}); 0 disables it")
    ap.add_argument("--shallow-tau", type=float, default=None,
                    help="the barrier's pre-softplus threshold (production "
                         f"{train_mirt.SHALLOW_TAU})")
    ap.add_argument("--random-user-weight", type=float, default=None,
                    help="share of the synthetic RANDOM user's rows drawn into "
                         "each epoch (production "
                         f"{train_mirt.RANDOM_USER_WEIGHT}): an ordinary user "
                         "who answers every problem at one over its legal moves")
    ap.add_argument("--strong-user-weight", type=float, default=None,
                    help="the same for the synthetic STRONG user (production "
                         f"{train_mirt.STRONG_USER_WEIGHT}), who answers every "
                         "problem at the stored evaluations' probability of "
                         "choosing the answer")
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
    ap.add_argument("--ortho-mode",
                    choices=("per_tail", "whole", "off", "rating_only"),
                    default=None,
                    help="per_tail is what ships; whole is the "
                         "linear-statistic comparison arm; rating_only keeps "
                         "the per-tail form but narrows the target to the "
                         "rating alone, which is the displacement arm; off "
                         "removes the term")
    ap.add_argument("--ortho-axes", type=float, default=None,
                    help="the MUTUAL orthogonality penalty on the axes — the "
                         "squared off-diagonal of the loadings' batch "
                         f"correlation matrix (production {train_mirt.ORTHO_AXES}, "
                         "on the detached readout); 0 removes the term. Part "
                         "of the regime: its weight and detachment are pinned "
                         "on every record")
    ap.add_argument("--teach-detach", type=int, default=None,
                    help="1 = the teaching pull acts on loadings computed from "
                         "a detached copy of the trunk output, so it steers the "
                         "axis readout alone and never the network the rating "
                         "shares (production 0: the pull reaches the trunk)")
    ap.add_argument("--ortho-detach", type=int, default=None,
                    help="1 = the rating-orthogonality penalty's loadings side "
                         "is detached from the trunk the same way (production 0)")
    ap.add_argument("--ortho-axes-detach", type=int, default=None,
                    help="1 = the mutual penalty (--ortho-axes) is detached from "
                         "the trunk the same way (production "
                         f"{int(train_mirt.ORTHO_AXES_DETACH)}); 0 lets its "
                         "gradient reach the trunk")
    ap.add_argument("--v-ridge-ladder", type=float, default=None,
                    help="ratio between consecutive axes' ridge weights on the "
                         "user positions (production 1.0, isotropic); above 1 "
                         "the prior is anisotropic and the objective is no "
                         "longer indifferent to rotating the axis space")
    ap.add_argument("--teach-weight", type=float, default=None,
                    help="the pull toward the taught rule columns (production "
                         f"{train_mirt.TEACH_WEIGHT}); 0 frees the basis")
    # --- the ability prior and the starting point (methodology §1.5, §1.8) ---
    ap.add_argument("--theta-laplace", type=int, default=None,
                    help="1 = ability carries the population prior whose "
                         "variance is re-estimated each epoch from the "
                         "abilities and their Fisher information (production "
                         f"{int(train_mirt.THETA_LAPLACE)}); 0 leaves ability "
                         "free. Part of the regime")
    ap.add_argument("--init-base-rate", type=int, default=None,
                    help="1 = every ability is shifted by one amount before "
                         "training so the initial mean prediction equals the "
                         "observed clearance (production "
                         f"{int(train_mirt.INIT_BASE_RATE)}); 0 starts from the "
                         "network's initial output. Part of the regime")
    # --- the feature cluster (methodology §1.3) ---
    ap.add_argument("--pca-dim", type=int, default=None,
                    help="joint-PCA activation width (production "
                         f"{PCA_DIM})")
    ap.add_argument("--blocks", default=None,
                    help="which per-problem blocks the trunk sees, as "
                         f"utils.rating.features names them (production "
                         f"{PRODUCTION_BLOCKS!r}; 'all' adds the "
                         f"answer-derived blocks)")
    ap.add_argument("--act-source", choices=("store", "trunk_final", "last_block",
                             "trunk_final_tf3b10c512"),
                    default=None,
                    help="which correct-cell activation roll-up the joint PCA "
                         "reads (production 'store': the activation store's own "
                         "roll-up). 'trunk_final' is the "
                         "trunk state the heads read and 'last_block' the "
                         "final block's increment alone; both need "
                         "auxiliary/rating_dev/extract_last_block.py run "
                         "first; with --pca-dim 0 they enter unrotated")
    ap.add_argument("--k-axes", type=int, default=None,
                    help=f"how many axes the fit carries (production {K_AXES}). "
                         f"The bank is NESTED: a smaller count keeps the first k "
                         f"components of axes/labels.json, so a sweep varies "
                         f"only the count. A count above what labels.json lists "
                         f"halts, since there is no rule to teach the extra axis")
    ap.add_argument("--dropout", type=float, default=None,
                    help="the TRUNK's dropout rate (production "
                         f"{train_mirt.DROPOUT}); 0 removes it")
    ap.add_argument("--input-dropout", type=float, default=None,
                    help="COLUMN dropout over the wide input region "
                         f"(production {train_mirt.INPUT_DROPOUT}); 0 removes "
                         f"it. Separate from the trunk's because 1.5 holds the "
                         f"two sites to suppress different failure modes")
    ap.add_argument("--weight-decay", type=float, default=None,
                    help="Adam weight decay (production 0 — none ships); the "
                         "knob exists to re-measure whether a norm penalty "
                         "substitutes for dropout")
    ap.add_argument("--weight-decay-fields", type=float, default=None,
                    help="weight decay for the FREE PER-USER FIELDS alone "
                         "(default: follow --weight-decay). 0 with a nonzero "
                         "--weight-decay decays the network only, separating "
                         "a regularization gain from the ability-ridge "
                         "re-gauging that decaying the fields causes")
    ap.add_argument("--lr-schedule", choices=("constant", "cosine"),
                    default=None,
                    help=f"{train_mirt.LR_SCHEDULE} ships: cosine decays both "
                         f"rates to zero over the budget, holding their ratio "
                         f"fixed; constant is the comparison arm. A schedule "
                         f"is only comparable at matched WORK (the recorded "
                         f"learning-rate integral), not at matched epochs")
    ap.add_argument("--lr-warmup-epochs", type=int, default=None,
                    help="linear ramp of both rates from near zero over this "
                         "many opening epochs, multiplied into the schedule "
                         "(production 0 — no warmup ships). The knob exists "
                         "for the axis-space panel: whether a gentler opening "
                         "makes a free axis fit land on the same basis twice")
    ap.add_argument("--pca-seed", type=int, default=None,
                    help="seed for the randomized joint-PCA FEATURE BASIS alone "
                         "(default: the one seed). Pinning it while --seed "
                         "varies separates a spread caused by the basis "
                         "re-rolling from one caused by the network's own "
                         "initialization")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="seed for the HELD-ASIDE DRAW alone (default: the one "
                         "seed). Pinning it across arms while --seed varies "
                         "holds the scored rows fixed and leaves the "
                         "initialization as the only thing moving")
    ap.add_argument("--stop-unit", choices=("row", "problem", "spread"),
                    default=None,
                    help="what the held-aside fraction is drawn over. 'row' "
                         "ships and is a convergence detector; 'problem' holds "
                         "whole problems out, which is what a claim about the "
                         "FEATURE side has to be scored on; 'spread' holds out "
                         "often-attempted problems, some whole and some half, "
                         "and writes their rows for the per-problem spread "
                         "(methodology 1.5)")
    args = ap.parse_args(argv)

    cfg = {}
    for flag in ("seed", "epochs", "shallow_w", "shallow_tau", "lr_net",
                 "random_user_weight", "strong_user_weight",
                 "lr_field", "ortho_rating", "ortho_mode", "ortho_axes",
                 "teach_weight", "teach_detach", "ortho_detach",
                 "ortho_axes_detach", "v_ridge_ladder",
                 "pca_dim", "blocks", "act_source", "stop_unit", "k_axes",
                 "dropout", "input_dropout", "weight_decay",
                 "weight_decay_fields",
                 "lr_schedule", "lr_warmup_epochs", "split_seed", "pca_seed",
                 "theta_laplace", "init_base_rate"):
        v = getattr(args, flag)
        if v is not None:
            cfg[flag] = v
    run_arm(args.panel, args.arm, cfg, force=args.force)


if __name__ == "__main__":
    main()
