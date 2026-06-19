#!/bin/bash
# Chain 15: leave-one-out ablations on the FINAL stack (user challenge: components were
# validated against their round's baseline, not against the finished stack — check for
# redundancy). Stack = ERM 20k+9d + hier-slope (sf 0.01) + frozen dsf-residuals l2=4.
# Reference (full stack, l2=4): 0.4172 / 0.5471 / 0.7936 / hv 0.0088 (erm_hs_resid_dsf_l4).
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0"
E="--no-dro-skill --holdout-problem-frac 0.1"
# (a) drop hier-slope, keep old hard clamp 0.2
$P -m auxiliary.rating_dev.train_rating_v2_research $E $R > experiments_4pl/abl_clamp02_resid.log 2>&1
# (b) drop slope constraint entirely (free at 0.01) — is hier redundant given residuals?
$P -m auxiliary.rating_dev.train_rating_v2_research $E $R --slope-floor 0.01 > experiments_4pl/abl_free001_resid.log 2>&1
# (c) swap hier for the legacy spectral prior under the stack
$P -m auxiliary.rating_dev.train_rating_v2_research $E $R --slope-dist-prior 1.0 --slope-floor 0.01 > experiments_4pl/abl_distprior_resid.log 2>&1
# (d) drop anchors under the stack
$P -m auxiliary.rating_dev.train_rating_v2_research $E $R --hier-slope --slope-floor 0.01 --no-anchors > experiments_4pl/abl_noanchor_hier_resid.log 2>&1
# (e) single-5k anchor under the stack
$P -m auxiliary.rating_dev.train_rating_v2_research $E $R --hier-slope --slope-floor 0.01 --anchor-ranks 5k > experiments_4pl/abl_5k_hier_resid.log 2>&1
echo CHAIN15_DONE
