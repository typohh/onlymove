#!/bin/bash
# Chain 28: multi-seed the M0b ship candidate (= candidate A with the floor head replaced by
# the MEASURED 0.70·bot_floor and λ pinned 0.02; hier slope kept — chain27 attribution showed
# the hard slope pin was the rho cost). Seed-0: 0.4160/0.5385/0.7977/hv .0109/fb .0161.
# A 3-seed means: 0.4166/0.5575/0.7959/hv ~.0095.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
M="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8 --mech-floor --mech-floor-scale-fixed 0.70 --lapse-fixed 0.02"
$P -m auxiliary.train_rating_v2_research $M --seed 1 > experiments_4pl/m0b_seed1.log 2>&1
$P -m auxiliary.train_rating_v2_research $M --seed 2 > experiments_4pl/m0b_seed2.log 2>&1
echo CHAIN28_DONE
