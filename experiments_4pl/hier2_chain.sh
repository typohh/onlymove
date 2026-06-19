#!/bin/bash
# Fixed (two-pass frozen-moment) hier cells + softened gap-balance + baseline heavy-user tables.
cd /home/ootakamoku/Repositories/onlymove
until ! pgrep -f "pipeline.train_rating_[v]2" >/dev/null \
      && grep -q "survey rho" experiments_4pl/erm_20k9d_hier_slopeonly.log 2>/dev/null; do sleep 60; done
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1"
$P -m auxiliary.rating_dev.train_rating_v2_research $E > experiments_4pl/erm_20k9d_calib2.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $E --hier-slope --hier-floor --slope-floor 0.01 > experiments_4pl/erm_20k9d_hier2.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $E --gap-balance --gap-balance-power 0.5 --gap-balance-max 3 > experiments_4pl/erm_20k9d_gapbal_soft.log 2>&1
echo HIER2_CHAIN_DONE
