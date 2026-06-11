#!/bin/bash
# Chain 13: extract the per-problem reference model's ~0.06-nat local headroom SAFELY —
# frozen-backbone residuals generalized to difficulty(+slope) columns (legacy δ_p/η_p on
# the measured-best backbone). Refs: floor-only frozen 0.4174/0.5509/0.8029/0.0126;
# per-problem free 3PL 0.3550 by-row but rho 0.51 (geometry destroyed).
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1 --hier-slope --slope-floor 0.01 --floor-residual"
$P -m pipeline.train_rating_v2 $E --resid-cols "diff,floor" > experiments_4pl/erm_hs_resid_df.log 2>&1
$P -m pipeline.train_rating_v2 $E --resid-cols "diff,slope,floor" > experiments_4pl/erm_hs_resid_dsf.log 2>&1
echo CHAIN13_DONE
