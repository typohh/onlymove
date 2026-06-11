#!/bin/bash
# Chain 18: full artifacts + multi-seed for the Round-17 winners.
# 1) full one-stage candidate (cell A config) → ratings_v2_onestage
# 2) full best no-anchor (cell B2 config)     → ratings_v2_onestage_noanchor
# 3-4) seeds 1,2 eval of cell A (rho noise ~0.01 — Finding Z protocol)
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--no-dro-skill --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached"
A="--hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8"
B="--hier-slope --hier-slope-mu 0.63 --hier-slope-sigma 0.65 --no-anchors"
$P -m pipeline.train_rating_v2 $R $A --out-dir ratings_v2_onestage > experiments_4pl/full_onestage.log 2>&1
$P -m pipeline.train_rating_v2 $R $B --out-dir ratings_v2_onestage_noanchor > experiments_4pl/full_onestage_noanchor.log 2>&1
$P -m pipeline.train_rating_v2 $R $A --holdout-problem-frac 0.1 --seed 1 > experiments_4pl/onestage_hier_seed1.log 2>&1
$P -m pipeline.train_rating_v2 $R $A --holdout-problem-frac 0.1 --seed 2 > experiments_4pl/onestage_hier_seed2.log 2>&1
echo CHAIN18_DONE
