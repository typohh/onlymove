#!/bin/bash
# Chain 26: fix the mech-floor scale at its externally MEASURED value (0.70) instead of
# learning it. Chain25 diagnosis: the learned s is contaminated wherever the features can't
# place difficulty precisely — hybrid learns 0.757-0.771 (≈ the measured 0.70-0.75 ratio,
# calibrated far-below), bw-only drags s to 0.830-0.879 to absorb difficulty misplacement,
# which IS the residual Finding-KK signed far-below over-prediction (+0.05..+0.10 ≈
# (s-0.75)·botf). If fixing s closes it: the floor needs ZERO learned parameters and the
# minimal bw model (M4b: 62 cols, no anchors, one head, three constants) is complete. If the
# misfit resurfaces in difficulty/rho instead: bw-only genuinely can't place those problems
# and the hybrid backbone stays.
# Refs: M3 0.4445/0.5172/0.8064 hv .0289 fb .0647 s .879 | M4 0.4460/0.5183/0.7970 hv .0297
# fb .0545 s .830 | M1 0.4157/0.5533/0.7798 hv .0095 fb .0210 s .771 | A .4164/.5498/.8038/.0070/.0217.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --global-slope-fixed 2.2 --mech-floor --mech-floor-scale-fixed 0.70 --lapse-fixed 0.02"
$P -m pipeline.train_rating_v2 $R --features bw > experiments_4pl/m3b_bw_sfix.log 2>&1
$P -m pipeline.train_rating_v2 $R --features bw --no-anchors > experiments_4pl/m4b_bw_noanchor_sfix.log 2>&1
$P -m pipeline.train_rating_v2 $R > experiments_4pl/m1b_hybrid_sfix.log 2>&1
echo CHAIN26_DONE
