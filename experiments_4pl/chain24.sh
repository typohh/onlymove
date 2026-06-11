#!/bin/bash
# Chain 24: (1) anchor question on PREDICTIVE grounds (user correction: floor↔bot_floor is an
# indicator, not the failure — the failure would be predicted≠observed success in the far-below
# regime, conditioned on rating). New gap×rating-decile calibration table; cells A (anchored
# one-stage candidate) vs B2 (no-anchor, matched μ.63/σ.65 target). Decision rule: if B2's
# signed Δ concentrates in far-below cells at mid/hard deciles (where human rows exist),
# anchors do observable predictive work; if B2 is flat everywhere with support, Finding GG
# lives in the unobservable region and anchor value reduces to by-row + λ + an unfalsifiable
# prior.
# (2) bw-only + floor-only head-pin (Finding KK candidate fix via Finding JJ's λ-honesty
# effect): dose 0.3 floor-only = JJ's effective floor-pin rate at its 0.6 both-heads dose.
# Watch: trained λ vs plateau λ̂ convergence AND whether the signed far-below over-prediction
# (+0.04…+0.09, the 0.70×bot floor mismatch half of KK) moves. Refs: H4 0.4415/0.5098/0.8095/
# hv 0.0365; candidate A 0.4164/0.5498/0.8038/0.0070.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached"
$P -m pipeline.train_rating_v2 $R --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8 > experiments_4pl/onestage_hier_ratingslice.log 2>&1
$P -m pipeline.train_rating_v2 $R --hier-slope --hier-slope-mu 0.63 --hier-slope-sigma 0.65 --no-anchors > experiments_4pl/onestage_noanchor_ratingslice.log 2>&1
$P -m pipeline.train_rating_v2 $R --features bw --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8 --head-pin-dropout 0.3 --head-pin-heads floor > experiments_4pl/onestage_bwonly_floorpin03.log 2>&1
echo CHAIN24_DONE
