#!/bin/bash
# Chain 35 (user request): Round-28 rerun on bw-only features (62 dims, no PCA activations).
# Same cells as chain34 (no anchors + --mixed-only), --features bw. Rationale for a different
# outcome: on bw features the ladder IS the input — difficulty location enters through the
# features (Finding PP: anchors near-redundant on bw; M4b no-anchor full-corpus rho 0.8065) —
# so the question is whether feature directness preserves ordering when the human supervision
# is outcome-conditioned noise (Finding AAA), or whether the gutted training signal flattens
# the net regardless. User expectation on record: no great results, looking for HOW it changes.
# Refs: I1 (bw_act, pin 2.2) 0.5876/0.5995/rho 0.0350; I2 (bw_act, free) 0.5819/0.5703/0.0886,
#       slopes railed 0.01, ratings ±1000; M4b (bw, no-anchor, FULL corpus, s=.70, pin 2.2)
#       0.4432/0.5183/0.8065/fb 0.0526.
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
B="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --mech-floor --mech-floor-scale-fixed 0.70 --lapse-fixed 0.02 --no-anchors --mixed-only --features bw"
$P -m auxiliary.train_rating_v2_research $B --global-slope-fixed 2.2 > experiments_4pl/i3_mixed_noanchor_bw_pin22.log 2>&1
$P -m auxiliary.train_rating_v2_research $B                          > experiments_4pl/i4_mixed_noanchor_bw_free.log 2>&1
echo CHAIN35_DONE
