#!/bin/bash
# Chain 21: head-pin dropout (user compensation-learning test) on the one-stage candidate.
# Per batch with prob p, pin slope OR floor head to its corpus-median EMA; other heads
# compensate. Predictions on record (push-back): behaves like shrinkage — floor spread and
# floor<->bot_floor degrade toward the smooth-family equilibrium, transfer may tick up,
# two-candidate hard case worsens; no assignment gain. Dose: 0.3 and 0.6.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
A="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8"
$P -m pipeline.train_rating_v2 $A --head-pin-dropout 0.3 > experiments_4pl/onestage_headpin03.log 2>&1
$P -m pipeline.train_rating_v2 $A --head-pin-dropout 0.6 > experiments_4pl/onestage_headpin06.log 2>&1
echo CHAIN21_DONE
