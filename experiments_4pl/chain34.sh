#!/bin/bash
# Chain 34 (user proposal): NO anchors + mixed-outcome-only training corpus (--mixed-only:
# keep only train rows whose problem AND user each have both successes and failures, fixed-
# point iterated; eval rows unfiltered so metrics stay comparable). Rationale: one-sided score
# patterns have monotone likelihood (all-success problem ⇒ difficulty→−∞ pressure) — the
# classical Rasch perfect/zero-score drop — and that outward pull is a candidate cause of the
# no-anchor difficulty-axis warp (Finding OO: M2 rho 0.69). If filtering rescues it, the
# anchors' location job was an artifact of training on uninformative rows.
# I1: minimal no-anchor (M2 config) + mixed-only — the direct OO rescue test.
#     Ref M2: 0.4265/0.5601/0.6909/fb 0.0434. Success = rho back near 0.78+.
# I2: FREE slope, no anchors + mixed-only — does the filter also close the Finding-WW blowup?
#     All-failure hard problems (where 9d fails and no human succeeded) drop out of training,
#     so every TRAINED problem has an observed transition; unattempted problems still get
#     feature-extrapolated slopes — open whether the head stays sane off-support.
#     Refs: F1 (anchored free) 0.4070/0.5619/0.7814 ratings +255; M2 above.
# Predictions on record: the filter removes the one-sided PULL but also removes information
# (the kept corpus shrinks and selection biases base rates toward 0.5); partial rho rescue
# plausible on I1; I2's slope head still blows on off-support problems (the net extrapolates
# where no trained problem constrains it) — rating range is the referee.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
B="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --mech-floor --mech-floor-scale-fixed 0.70 --lapse-fixed 0.02 --no-anchors --mixed-only"
$P -m pipeline.train_rating_v2 $B --global-slope-fixed 2.2 > experiments_4pl/i1_mixed_noanchor_pin22.log 2>&1
$P -m pipeline.train_rating_v2 $B                          > experiments_4pl/i2_mixed_noanchor_free.log 2>&1
echo CHAIN34_DONE
