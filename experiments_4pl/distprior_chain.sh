#!/bin/bash
# Spectral-prior test cells (user: still worth testing whether pinning the slope SPECTRUM
# alone holds floor+rating in check — it provides information even if legacy-anchored).
# Like-for-like vs the hier cells: same ERM 20k+9d anchors, slope-floor 0.01.
cd /home/ootakamoku/Repositories/onlymove
# wait for the principled chain (by9ey1rnq) to finish
until ! pgrep -f "pipeline.train_rating_[v]2" >/dev/null \
      && grep -q "survey rho" experiments_4pl/erm_20k9d_hier_gapbal.log 2>/dev/null; do sleep 60; done
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1"
$P -m auxiliary.rating_dev.train_rating_v2_research $E --slope-dist-prior 1.0 --slope-floor 0.01 > experiments_4pl/erm_20k9d_distprior.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $E --hier-slope --slope-floor 0.01 > experiments_4pl/erm_20k9d_hier_slopeonly.log 2>&1
echo DISTPRIOR_CHAIN_DONE
