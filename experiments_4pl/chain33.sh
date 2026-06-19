#!/bin/bash
# Chain 33 (user proposal, third const-anchor iteration): TWO synthetic rungs const0.9 +
# const0.8 on the free-slope mech-floor config — targets chosen INSIDE the reachable band
# (above floor ~0.41, below ceiling 0.98) so the Finding-XX parking escape is priced out,
# and the PAIR constrains slope: P(θa)=0.9, P(θb)=0.8 on the same curve implies
# slope = Δz/(θa−θb) — a slope-consistency constraint whose LEVEL is self-calibrating
# (the learned rung spacing replaces hier's cached μ/σ). The 0.8 rung additionally pins the
# curve at the shipped 80%-clear point — the most direct rating-bounding pull possible.
# Escape economics (BCE vs on-curve entropy, per row): const0.5 parked for ~0.02 nats
# (floor≈0.41≈0.5 — why chain31 failed); const0.8 parking costs ≥0.30 nats either side
# (trapped); const0.9 parking ABOVE costs only ~0.09 (may half-escape, watch its θ).
# Predictions on record:
#  (1) blowup bounded (ratings single/low-double digits) — the 0.8 rung can't be evaded;
#  (2) slope rail closes (p1 >> 0.01), spread shrinks toward one implied level — the pair is
#      a constant-slope prior in data space (aligned with Finding I: assignment is noise);
#  (3) hard-rating compression toward the 0.8-rung's ability (2 equations/problem ⇒ universal
#      template in the no-data band — bounded ≠ ordered above 9d coverage);
#  (4) far-below calibration NOT damaged for once (no targets in the floor regime, mech floor
#      intact) — should hold near F1's 0.028, unlike const0.5 (0.045+) and const4 (0.052+);
#  (5) θ_80 in-population; θ_90 drifts high if anywhere.
# Refs: F1 0.4070/0.5619/0.7814/hv .0135/fb .0280 ratings +255; M0b .4160/.5385/.7977/.0109/.0161.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
F="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --mech-floor --mech-floor-scale-fixed 0.70 --lapse-fixed 0.02 --anchor-ranks const0.9,const0.8 --no-anchor-fixed-mass"
$P -m auxiliary.rating_dev.train_rating_v2_research $F --anchor-weight 0.2  > experiments_4pl/h1_const9080_w20.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $F --anchor-weight 0.05 > experiments_4pl/h2_const9080_w05.log 2>&1
echo CHAIN33_DONE
