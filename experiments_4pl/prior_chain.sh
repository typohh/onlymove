#!/bin/bash
cd /home/ootakamoku/Repositories/onlymove
# wait for the discount chain (b8dmllda5) to finish — bracket trick avoids pgrep self-match
until ! pgrep -f "pipeline.train_rating_[v]2" >/dev/null && grep -q "wrote user_theta" experiments_4pl/full_dense_disc.log 2>/dev/null; do sleep 60; done
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1"
$P -m pipeline.train_rating_v2 $E --anchor-ranks 5k --slope-dist-prior 1.0 --slope-floor 0.01 > experiments_4pl/erm_5k_prior.log 2>&1
$P -m pipeline.train_rating_v2 $E --no-anchors      --slope-dist-prior 1.0 --slope-floor 0.01 > experiments_4pl/erm_noanchor_prior.log 2>&1
$P -m pipeline.train_rating_v2 --no-dro-skill --anchor-ranks 5k --slope-dist-prior 1.0 --slope-floor 0.01 --out-dir ratings_v2_5k_slopeprior > experiments_4pl/full_5k_prior.log 2>&1
echo PRIOR_CHAIN_DONE
