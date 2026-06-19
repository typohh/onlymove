#!/bin/bash
# Chain 6: can the dense discounted ladder + spectral prior pin slope ASSIGNMENT?
# (Old combo-chain cell, now better motivated: every artifact's slope ordering is
# ~uncorrelated with bot-ladder steepness; dense+discount teaches relative shape,
# the prior supplies the hard-tail constraint that killed plain dense runs.)
cd /home/ootakamoku/Repositories/onlymove
until ! pgrep -f "pipeline.train_rating_[v]2" >/dev/null \
      && grep -q "wrote user_theta" experiments_4pl/full_20k9d_distprior.log 2>/dev/null; do sleep 60; done
DENSE="20k,18k,16k,14k,12k,10k,8k,6k,4k,2k,1k,1d,2d,3d,4d,5d,6d,7d,8d,9d"
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1"
$P -m auxiliary.rating_dev.train_rating_v2_research $E --anchor-ranks "$DENSE" --anchor-slope-discount --slope-dist-prior 1.0 --slope-floor 0.01 > experiments_4pl/erm_dense_disc_prior.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research --no-dro-skill --anchor-ranks "$DENSE" --anchor-slope-discount --slope-dist-prior 1.0 --slope-floor 0.01 --out-dir ratings_v2_dense_disc_prior > experiments_4pl/full_dense_disc_prior.log 2>&1
echo CHAIN6_DONE
