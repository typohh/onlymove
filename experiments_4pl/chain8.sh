#!/bin/bash
# Chain 8: conditional-μ hier shrinkage (principled continuous alternative) vs the
# conditional diversity+decorrelation objective (user proposal, falsification test).
# Judge by: objectives + heavy-val calibration (eval cells) AND slope↔bot_steep +
# structure tables on the full artifacts (diag_4pl_structure).
cd /home/ootakamoku/Repositories/onlymove
until ! pgrep -f "pipeline.train_rating_[v]2" >/dev/null \
      && grep -q "wrote user_theta" experiments_4pl/full_20k9d_hierslope.log 2>/dev/null; do sleep 60; done
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1"
$P -m auxiliary.rating_dev.train_rating_v2_research $E --hier-slope --hier-floor --hier-conditional --slope-floor 0.01 > experiments_4pl/erm_20k9d_hiercond.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $E --cond-diversity 1.0 --slope-floor 0.01 > experiments_4pl/erm_20k9d_conddiv.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research --no-dro-skill --hier-slope --hier-floor --hier-conditional --slope-floor 0.01 --out-dir ratings_v2_20k9d_hiercond > experiments_4pl/full_20k9d_hiercond.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research --no-dro-skill --cond-diversity 1.0 --slope-floor 0.01 --out-dir ratings_v2_conddiv > experiments_4pl/full_conddiv.log 2>&1
echo CHAIN8_DONE
