#!/bin/bash
# Chain 27: settle the minimal-model ship question.
# (1) M1b multi-seed (seeds 1,2; seed-0 = m1b_hybrid_sfix.log): rho −0.015 and transfer +0.011
#     vs A are inside/near A's own seed spread (A: rho .8038/.7954/.7885, transfer
#     .5498/.5602/.5624) — decide on means, not seed-0 draws.
# (2) M0b: mech-floor + FIXED s 0.70 but KEEP hier-slope (μ.8/σ.8) — attribution: M0→M1 lost
#     0.010 transfer when the slope pinned; does near-constant-but-free slope + honest fixed
#     floor keep M0's transfer (0.5433) AND M1b's calibration (0.0062)?
# (3) M3c: bw-minimal, looser δ ridge (l2 1.0 vs 4.0) — the placement-flaw probe: heavy users
#     grind SEEN problems, so data-attested δ_diff could substitute for the activation
#     fingerprint exactly where the bw backbone misplaces. Watch fb-sliced/hv vs M3b
#     (0.0675/0.0292) and the Finding-W rho cost of loose ridge.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
M="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-mode detached --mech-floor --mech-floor-scale-fixed 0.70 --lapse-fixed 0.02"
$P -m auxiliary.train_rating_v2_research $M --floor-resid-l2 4.0 --global-slope-fixed 2.2 --seed 1 > experiments_4pl/m1b_seed1.log 2>&1
$P -m auxiliary.train_rating_v2_research $M --floor-resid-l2 4.0 --global-slope-fixed 2.2 --seed 2 > experiments_4pl/m1b_seed2.log 2>&1
$P -m auxiliary.train_rating_v2_research $M --floor-resid-l2 4.0 --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8 > experiments_4pl/m0b_hier_sfix.log 2>&1
$P -m auxiliary.train_rating_v2_research $M --floor-resid-l2 1.0 --global-slope-fixed 2.2 --features bw > experiments_4pl/m3c_bw_l2loose.log 2>&1
echo CHAIN27_DONE
