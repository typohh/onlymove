#!/bin/bash
# Chain 17: STAGE-CONSOLIDATION matrix — collapse the 4-stage final stack (free pass-1 →
# hier pass-2 → frozen residual pass → posthoc plateau λ) into ONE optimization pass, and
# test anchor removal under it. New mechanisms: --hier-slope-mu/sigma (fixed shrinkage
# target, pass-1 skipped), --floor-resid-mode detached (gradient-isolated one-stage
# residuals; same fixed point as frozen), --lapse-fixed (plateau λ folded into the fit).
# References: 4-stage stack 0.4172/0.5471/0.7936/hv0.0088 (erm_hs_resid_dsf_l4);
#             4-stage no-anchor 0.4270/0.5535/0.7881/hv0.0108 (abl_noanchor_hier_resid).
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached"
H="--hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8"
# A: one-stage hier-fixed, anchors 20k+9d (the direct 1-stage rebuild of the stack)
$P -m pipeline.train_rating_v2 $R $H > experiments_4pl/onestage_hier.log 2>&1
# B: A without anchors
$P -m pipeline.train_rating_v2 $R $H --no-anchors > experiments_4pl/onestage_hier_noanchor.log 2>&1
# C: one-stage global-slope (no hier constants at all; slope varies only via detached δ_slope)
$P -m pipeline.train_rating_v2 $R --global-slope > experiments_4pl/onestage_gslope.log 2>&1
# D: C without anchors
$P -m pipeline.train_rating_v2 $R --global-slope --no-anchors > experiments_4pl/onestage_gslope_noanchor.log 2>&1
# E: the full ship-in-one-command no-anchor candidate (B + plateau λ folded in)
$P -m pipeline.train_rating_v2 $R $H --no-anchors --lapse-fixed 0.0225 > experiments_4pl/onestage_hier_noanchor_lfix.log 2>&1
echo CHAIN17_DONE
