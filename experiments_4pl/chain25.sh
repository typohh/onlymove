#!/bin/bash
# Chain 25: the MINIMAL-MODEL ladder (user direction: keep simplifying). Idea: the hand-holding
# count scales with the number of under-identified LEARNED shape parameters — so stop learning
# them. Floor becomes MEASURED (--mech-floor: c_p = s·bot_floor_p, one global scalar; legacy
# validated 0.70·bot, floors track 0.7·botf in every healthy config anyway), slope becomes a
# PINNED constant (--global-slope-fixed 2.2 = the hier equilibrium level; assignment is noise,
# Finding I), λ fixed 0.02 (Finding EE: a wash). One learned feature head (rating) + detached
# dsf residuals. Predictions on record:
#   M0 ≈ cell A on objectives, easy-end floor wart FIXED (p95 ~0.7 not 0.93), swap impossible;
#   M1 ≈ M0 with slightly worse rho if the slope-level effect bites (Finding K: 2.26 kept rho);
#   M2 (no anchors) — THE dissolution test: if the anchors' job was babysitting the learned
#      floor, M2 ≈ M1 (vs B2's far-below failure −0.05 signed, |Δ| 0.045); if they also pin
#      difficulty location, M2 degrades — that residual gap is the anchors' irreducible value;
#   M3 (bw-only) — the Finding-KK fix test: floor no longer needs the activation fingerprint
#      (it's measured), so the signed far-below over-prediction (+0.05..+0.14) should close
#      while keeping bw transfer ~0.51 and rho ~0.81+;
#   M4 (bw-only, no anchors) — the maximally simple model: 62-dim input, one learned head,
#      one learned scalar (s), two constants (slope, λ), detached residuals. If M4 holds A's
#      numbers the entire scaffolding inventory reduces to "measure the floor, pin the rest".
# Cell A reference: 0.4164 / 0.5498 / 0.8038 / hv|Δ| 0.0070, far-below sliced |Δ| 0.0217.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached"
$P -m auxiliary.rating_dev.train_rating_v2_research $R --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8 --mech-floor > experiments_4pl/m0_A_mechfloor.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $R --global-slope-fixed 2.2 --mech-floor --lapse-fixed 0.02 > experiments_4pl/m1_minimal.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $R --global-slope-fixed 2.2 --mech-floor --lapse-fixed 0.02 --no-anchors > experiments_4pl/m2_minimal_noanchor.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $R --global-slope-fixed 2.2 --mech-floor --lapse-fixed 0.02 --features bw > experiments_4pl/m3_minimal_bw.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $R --global-slope-fixed 2.2 --mech-floor --lapse-fixed 0.02 --features bw --no-anchors > experiments_4pl/m4_minimal_bw_noanchor.log 2>&1
echo CHAIN25_DONE
