#!/bin/bash
# Chain 14: ridge sweep for the multi-column frozen residuals (find the by-row vs rho/
# calibration knee; l2 0.2 ⇒ prior σ≈1.0 was too loose: rho 0.803→0.772, hv|Δ| 0.0126→0.020).
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1 --hier-slope --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor"
$P -m auxiliary.rating_dev.train_rating_v2_research $E --floor-resid-l2 1.0 > experiments_4pl/erm_hs_resid_dsf_l1.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $E --floor-resid-l2 4.0 > experiments_4pl/erm_hs_resid_dsf_l4.log 2>&1
echo CHAIN14_DONE
