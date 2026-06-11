#!/bin/bash
# Chain 23: act-dropout dose response (0.5 won rho 0.8308/transfer 0.5019 but broke curve
# shape: hv 0.0349, +0.07 far-below over-prediction). 0.25 probes for a middle point.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
A="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8"
$P -m pipeline.train_rating_v2 $A --act-dropout 0.25 > experiments_4pl/onestage_actdrop025.log 2>&1
echo CHAIN23_DONE
