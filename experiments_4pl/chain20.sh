#!/bin/bash
# Chain 20 (REDIRECTED by user): activations were already established to add no transferable
# value (Finding S: bw-only matches hybrid rho, beats transfer 0.5229 vs 0.5585 and
# calibration; act gains are seen-problem-only = the delta-residual category). So test REMOVING
# them from the backbone outright, with residuals carrying local info, and put dropout where
# the noise actually is (per-bot overdispersion ~50%):
#   H1 bw-only backbone + detached dsf residuals (+ two-pass hier: bw-only spectrum unknown,
#      Finding DD — if it wins, freeze its pass-1 target for the one-stage production config)
#   H2 = H1 + --bw-dropout 0.2 (per-column consensus regularizer)
#   H3 hybrid + --act-dropout 0.5 (soft-removal comparison, deprioritized last)
# Refs: bw-only no-resid 0.4772/0.5229/0.8186/hv0.0092; candidate A 0.4164/0.5498/0.8038/0.0070.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
R="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached"
$P -m auxiliary.train_rating_v2_research $R --features bw --hier-slope > experiments_4pl/onestage_bwonly_resid.log 2>&1
$P -m auxiliary.train_rating_v2_research $R --features bw --hier-slope --bw-dropout 0.2 > experiments_4pl/onestage_bwonly_resid_bwdrop.log 2>&1
$P -m auxiliary.train_rating_v2_research $R --hier-slope --hier-slope-mu 0.8 --hier-slope-sigma 0.8 --act-dropout 0.5 > experiments_4pl/onestage_actdrop05.log 2>&1
echo CHAIN20_DONE
