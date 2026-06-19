#!/bin/bash
# Chain 12: per-problem free 3PL + global HUMAN-only ceiling, all-61-bot anchors (user request).
# The maximal-local reference: the bot battery IS the model, humans correct where attested.
# Ignore metric 2 (held-out-problem) — meaningless without features.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
A="--per-problem --lapse-human-only --anchor-ranks all --batch 262144 --no-dro-skill"
$P -m auxiliary.train_rating_v2_research $A --holdout-problem-frac 0.1 > experiments_4pl/erm_perproblem.log 2>&1
$P -m auxiliary.train_rating_v2_research $A --out-dir ratings_v2_perproblem > experiments_4pl/full_perproblem.log 2>&1
$P -m auxiliary.diag_4pl_structure --dir ratings_v2_perproblem > experiments_4pl/diag_perproblem.log 2>&1
echo CHAIN12_DONE
