#!/bin/bash
# Answer-free identification matrix (see docs/rating_4pl_structure_log.md):
# ERM, anchors 20k+9d (champion), holdout 0.1, seed 0. New metrics: gap-binned calibration.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1"
$P -m pipeline.train_rating_v2 $E > experiments_4pl/erm_20k9d_calib.log 2>&1
$P -m pipeline.train_rating_v2 $E --hier-slope --hier-floor --slope-floor 0.01 > experiments_4pl/erm_20k9d_hier.log 2>&1
$P -m pipeline.train_rating_v2 $E --gap-balance > experiments_4pl/erm_20k9d_gapbal.log 2>&1
$P -m pipeline.train_rating_v2 $E --hier-slope --hier-floor --slope-floor 0.01 --gap-balance > experiments_4pl/erm_20k9d_hier_gapbal.log 2>&1
echo PRINCIPLED_CHAIN_DONE
