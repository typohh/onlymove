#!/bin/bash
# Chain 31 (user proposal): SINGULAR synthetic anchor — an artificial examinee with target
# pass rate 0.5 on EVERY problem (--anchor-ranks const0.5), replacing 20k+9d, on the free-slope
# mech-floor config (F1). Idea: one mid-curve point per problem = exactly the data regime
# Finding WW showed missing above 9d coverage; should bound the rating blowup in BOTH
# directions (z≈0 has healthy sigmoid gradient, so the a→0 escape is blocked; b drift gets a
# restoring pull toward the pivot).
# Weight sweep per user direction (singular rung probably needs LESS than a 20k/9d rung's 0.2):
# --no-anchor-fixed-mass --anchor-weight {0.2, 0.05, 0.0125}.
# Predictions on record:
#  (1) blowup closes at every weight (user's claim — expect ratings back to single digits);
#  (2) the above-9d-coverage tail doesn't just get bounded, it gets PINNED: fixed point is the
#      pivot manifold difficulty = θ_A + g(floor)/slope, so former-blowup problems compress
#      into a narrow band (watch top rating-bin edges) — runaway traded for collapse, hard-tail
#      ordering dies; lower weight softens the pin but also the bound (same dial);
#  (3) unreachable-target wrinkle: wherever mech floor > 0.5 (floor median 0.41, p95 0.64 — a
#      sizable easy slice) P ≥ floor > 0.5 ALWAYS, the row is a constant "harder" push there,
#      resisted by dense human data, bias ∝ weight;
#  (4) at 0.2 the false-fact bias shows in rho/hv vs F1 (0.4070/0.5619/0.7814/hv 0.0135,
#      ratings +255); at 0.0125 numbers ≈ F1's with (hopefully) a sane rating range — the
#      question is whether any weight gets BOTH.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
F="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --mech-floor --mech-floor-scale-fixed 0.70 --lapse-fixed 0.02 --anchor-ranks const0.5 --no-anchor-fixed-mass"
$P -m auxiliary.rating_dev.train_rating_v2_research $F --anchor-weight 0.2    > experiments_4pl/f3a_const50_w20.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $F --anchor-weight 0.05   > experiments_4pl/f3b_const50_w05.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $F --anchor-weight 0.0125 > experiments_4pl/f3c_const50_w0125.log 2>&1
echo CHAIN31_DONE
