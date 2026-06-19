#!/bin/bash
# Chain 9: (1) hier-slope-only rerun to get its residual-association table (reference for the
# θ-selection confound); (2) + per-problem floor residual — the Finding-O test: should win
# by-row AND held-out-problem simultaneously if floors are locally-identifiable-but-not-
# feature-predictable.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1"
$P -m auxiliary.train_rating_v2_research $E --hier-slope --slope-floor 0.01 > experiments_4pl/erm_20k9d_hierslope2b.log 2>&1
$P -m auxiliary.train_rating_v2_research $E --hier-slope --slope-floor 0.01 --floor-residual > experiments_4pl/erm_20k9d_hierslope_fresid.log 2>&1
echo CHAIN9_DONE
