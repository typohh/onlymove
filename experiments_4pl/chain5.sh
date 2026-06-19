#!/bin/bash
# Chain 5: full-run artifacts for structure diagnostics (CSV outputs, no holdout):
# the ERM 20k+9d baseline champion and the dist-prior candidate.
cd /home/ootakamoku/Repositories/onlymove
until ! pgrep -f "pipeline.train_rating_[v]2" >/dev/null \
      && grep -q "survey rho" experiments_4pl/erm_20k9d_distprior_gapbalsoft.log 2>/dev/null; do sleep 60; done
P=./.venv/bin/python3
$P -m auxiliary.rating_dev.train_rating_v2_research --no-dro-skill --out-dir ratings_v2_20k9d_erm > experiments_4pl/full_20k9d_erm.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research --no-dro-skill --slope-dist-prior 1.0 --slope-floor 0.01 --out-dir ratings_v2_20k9d_distprior > experiments_4pl/full_20k9d_distprior.log 2>&1
echo CHAIN5_DONE
