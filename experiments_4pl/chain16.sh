#!/bin/bash
# Chain 16: multi-seed the two stack finalists (hier vs dist-prior; Round 15 = tie within
# single-seed noise). Seed-0 refs: hier 0.4172/0.5471/0.7936/0.0088; dist 0.4165/0.5489/
# 0.8027/0.0105.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--no-dro-skill --holdout-problem-frac 0.1 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --slope-floor 0.01"
for S in 1 2; do
  $P -m pipeline.train_rating_v2 $R --hier-slope --seed $S > experiments_4pl/seed${S}_stack_hier.log 2>&1
  $P -m pipeline.train_rating_v2 $R --slope-dist-prior 1.0 --seed $S > experiments_4pl/seed${S}_stack_dist.log 2>&1
done
echo CHAIN16_DONE
