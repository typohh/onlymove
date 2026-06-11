#!/bin/bash
# Chain 19: single-20k anchor under the one-stage config (anchor-count reduction fallback
# after full removal failed on floor structure — diag ratings_v2_onestage_noanchor).
# Fixed hier target interpolated between the 20k+9d (0.81/0.78) and no-anchor (0.63/0.65)
# regimes. Eval first, then full artifact for the structure table.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--no-dro-skill --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --hier-slope --hier-slope-mu 0.7 --hier-slope-sigma 0.7 --anchor-ranks 20k"
$P -m pipeline.train_rating_v2 $R --holdout-problem-frac 0.1 > experiments_4pl/onestage_hier_20k.log 2>&1
$P -m pipeline.train_rating_v2 $R --out-dir ratings_v2_onestage_20k > experiments_4pl/full_onestage_20k.log 2>&1
echo CHAIN19_DONE
