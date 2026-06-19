#!/bin/bash
# Chain 29: complete the M0b candidacy — full artifact -> structure table + strong-user lens
# (the Round-17c checklist). M0b = A minus the learned floor head (-> fixed 0.70·bot_floor)
# minus trained λ (-> 0.02), multi-seed-confirmed (chain28): transfer better by 0.019 on
# means (every seed), rho/by-row par, fb-sliced par.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
M="--no-dro-skill --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8 --mech-floor --mech-floor-scale-fixed 0.70 --lapse-fixed 0.02"
$P -m auxiliary.rating_dev.train_rating_v2_research $M --out-dir ratings_v2_mechfloor > experiments_4pl/full_mechfloor.log 2>&1
$P -m auxiliary.diag_4pl_structure --dir ratings_v2_mechfloor > experiments_4pl/diag_mechfloor.log 2>&1
$P -m auxiliary.eval_strong_user_error mechfloor=ratings_v2_mechfloor onestage=ratings_v2_onestage > experiments_4pl/strong_mechfloor.log 2>&1
echo CHAIN29_DONE
