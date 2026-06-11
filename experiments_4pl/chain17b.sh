#!/bin/bash
# Chain 17b: follow-ups to the consolidation matrix.
# B2: was cell B's no-anchor degradation caused by the anchored-regime fixed target (0.8/0.8)?
#     Use the no-anchor pass-1 moments (0.632/0.651, from abl_noanchor_hier_resid pass-1).
# B3: best-shot no-anchor = B2 + plateau λ folded in.
# F:  champion (cell A) + plateau λ folded in — the λ open item on the anchored config.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached"
$P -m pipeline.train_rating_v2 $R --hier-slope --hier-slope-mu 0.63 --hier-slope-sigma 0.65 --no-anchors > experiments_4pl/onestage_hier_noanchor_mu063.log 2>&1
$P -m pipeline.train_rating_v2 $R --hier-slope --hier-slope-mu 0.63 --hier-slope-sigma 0.65 --no-anchors --lapse-fixed 0.0225 > experiments_4pl/onestage_hier_noanchor_mu063_lfix.log 2>&1
$P -m pipeline.train_rating_v2 $R --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8 --lapse-fixed 0.02 > experiments_4pl/onestage_hier_lfix.log 2>&1
echo CHAIN17B_DONE
