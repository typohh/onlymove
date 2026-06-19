#!/bin/bash
# Chain 32 (user proposal): FOUR constant synthetic anchors (const0.95, const0.65, const0.35,
# const0.05) with ALL four 4PL parameters freed (no mech-floor, no slope constraint, trained λ)
# — the idea: each parameter regime gets its own pivot row, so the model is "motivated to
# constrain each one" and the per-problem heads can be released.
# Push-back filed in advance (the in-principle limit): constant targets carry ZERO per-problem
# information — four of them per problem are 4 equations through the SAME four (ability, target)
# points for every problem, i.e. in the hard limit every problem gets ONE identical universal
# curve. Soft version = data-space shrinkage of every head toward one global template — the
# strongest form of the hier-everything/smooth-family direction Finding G already measured
# (global-target shrinkage crushes real floor spread to a constant). Also two targets are
# UNSATISFIABLE on most of the corpus: 0.05 sits below the real guess floor almost everywhere
# (human floor ≈ 0.7·bot, median ≈ 0.43), so the 5% row is a constant floors→0 push (the
# Finding-V/R failure injected as supervision); 0.95 sits above the hard tail's reality, a
# constant hard-difficulty→down push.
# Predictions on record:
#  (1) rating range bounded both ways at both weights (boundedness claim holds, as const0.5);
#  (2) floors collapse low corpus-wide at weight 0.1 (5%-row pressure + freed floor head) —
#      far-below calibration tanks (systematic under-prediction, B2's signature but stronger);
#  (3) hard tail compresses below the 95%-anchor's learned ability — top rating bins narrow;
#  (4) rho ≤ F1's 0.7814 (the four rows add no per-problem ordering information, only bias);
#  (5) the weight dial trades F1's blowup against template collapse with no per-problem
#      identification anywhere on it; "all four parameters freed" buys spread that is template-
#      shaped, not problem-shaped.
# Refs: F1 free-slope 0.4070/0.5619/0.7814/hv 0.0135/fb 0.0280, ratings +255;
#       M0b 0.4160/0.5385/0.7977/hv 0.0109/fb 0.0161; chain31 const0.5 sweep (running).
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
G="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --anchor-ranks const0.95,const0.65,const0.35,const0.05 --no-anchor-fixed-mass"
$P -m auxiliary.rating_dev.train_rating_v2_research $G --anchor-weight 0.1   > experiments_4pl/g1_const4_w10.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research $G --anchor-weight 0.025 > experiments_4pl/g2_const4_w025.log 2>&1
echo CHAIN32_DONE
