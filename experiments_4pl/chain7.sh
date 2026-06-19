#!/bin/bash
# Chain 7: full artifact for the current best package (fixed hier-slope-only) → diagnostic.
cd /home/ootakamoku/Repositories/onlymove
until ! pgrep -f "pipeline.train_rating_[v]2" >/dev/null \
      && grep -q "wrote user_theta" experiments_4pl/full_dense_disc_prior.log 2>/dev/null; do sleep 60; done
P=./.venv/bin/python3
$P -m auxiliary.train_rating_v2_research --no-dro-skill --hier-slope --slope-floor 0.01 --out-dir ratings_v2_20k9d_hierslope > experiments_4pl/full_20k9d_hierslope.log 2>&1
echo CHAIN7_DONE
