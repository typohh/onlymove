#!/bin/bash
# Chain 4: explicit global-slope reference, the INTENDED fixed hier-slope-only (the logged
# hier_slopeonly cell accidentally ran the per-step-moment code = constant slope), and the
# spectrum+regime combo (dist-prior with soft gap-balance).
cd /home/ootakamoku/Repositories/onlymove
until ! pgrep -f "pipeline.train_rating_[v]2" >/dev/null \
      && grep -q "survey rho" experiments_4pl/erm_20k9d_gapbal_soft.log 2>/dev/null; do sleep 60; done
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1"
$P -m auxiliary.rating_dev.train_rating_v2_research $E --global-slope > experiments_4pl/erm_20k9d_globalslope.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $E --hier-slope --slope-floor 0.01 > experiments_4pl/erm_20k9d_hierslope2.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $E --slope-dist-prior 1.0 --slope-floor 0.01 --gap-balance --gap-balance-power 0.5 --gap-balance-max 3 > experiments_4pl/erm_20k9d_distprior_gapbalsoft.log 2>&1
echo CHAIN4_DONE
