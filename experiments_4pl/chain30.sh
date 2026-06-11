#!/bin/bash
# Chain 30: free slope under the mechanical floor — the deconfounding test.
# Every historical free-slope verdict (rating blowup +289 R15-b, assignment-is-noise Finding I,
# learned-global-scalar drifting to ~4.3 R4/R17-C) was measured with the learned floor head and
# trained lapse ALSO moving. M0b's structure table showed the slope head picks up real assignment
# signal (slope<->bot_steep +0.17..+0.22 mid-deciles) once relieved of floor duty — so retest
# slope freedom with floor measured (0.70*bot) and lapse pinned (0.02), everything else = the
# minimal family (ERM, 20k+9d anchors, detached dsf residuals l2=4, eval mode, seed 0).
# F1: fully free per-problem slope head (only the 0.01 softplus floor remains).
#     Watch: rating range (blowup channel still open in principle), slope p1/p50/p99,
#     slope<->bot_steep isn't available in eval mode — judge spread/tails + the usual panel.
# F2: --global-slope, ONE learned scalar. Does the MLE still drift to ~4.3 when the floor
#     can't trade against it, or was that level a compensation artifact?
# Refs: M1b seed-0 0.4146/0.5613/0.7890/hv .0062/fb .0183 (pinned 2.2);
#       M0b seed-0 0.4160/0.5385/0.7977/hv .0109/fb .0161 (hier ~2.2).
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
M="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --mech-floor --mech-floor-scale-fixed 0.70 --lapse-fixed 0.02"
$P -m pipeline.train_rating_v2 $M > experiments_4pl/f1_freeslope_mech.log 2>&1
$P -m pipeline.train_rating_v2 $M --global-slope > experiments_4pl/f2_globalslope_mech.log 2>&1
echo CHAIN30_DONE
