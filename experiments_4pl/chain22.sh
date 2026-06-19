#!/bin/bash
# Chain 22: H4 — bw-only backbone + detached residuals with the FIXED slope target (0.8/0.8,
# slope level ~2.2) instead of bw-only's own low pass-1 level (0.40 -> slopes 1.49, which
# flattened curves: lambda died at 0.0046, hv calib 0.0267, floors p5=0.001 — H1's wrinkle).
# If rho ~0.82 holds while lambda/calibration recover, this is the new one-stage candidate
# (also: ONE pass again, since the target is fixed). H5: same + bw-dropout 0.2.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --features bw --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8"
$P -m auxiliary.train_rating_v2_research $R > experiments_4pl/onestage_bwonly_mu08.log 2>&1
$P -m auxiliary.train_rating_v2_research $R --bw-dropout 0.2 > experiments_4pl/onestage_bwonly_mu08_bwdrop.log 2>&1
echo CHAIN22_DONE
