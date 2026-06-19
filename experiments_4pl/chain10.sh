#!/bin/bash
# Chain 10: bot-column feature ablation (user hypothesis: the humanv0 winrate columns are a
# convenient-but-flawed shortcut for the shape heads). Same problem set, same anchors
# (20k+9d rows stay), only the fed feature columns change.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1"
$P -m auxiliary.rating_dev.train_rating_v2_research $E --features act > experiments_4pl/erm_20k9d_actonly.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $E --features act --hier-slope --slope-floor 0.01 --floor-residual > experiments_4pl/erm_20k9d_actonly_candidate.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $E --features bw > experiments_4pl/erm_20k9d_bwonly.log 2>&1
echo CHAIN10_DONE
