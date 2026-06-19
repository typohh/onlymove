#!/bin/bash
# Chain 36 (user request): M0b on ACT-ONLY features (129 dims: 128 PCA + size flag, no bot
# columns), with and without a third anchor rung = the near-argmax engine kata1 t0.02.
# Two regime adaptations from the M0b recipe, both forced by prior findings:
#  - hier slope runs TWO-PASS (no fixed mu/sigma): the cached 0.8/0.8 target was derived on
#    the anchored bw_act regime and is feature-set-sensitive (Finding DD; chain20 precedent).
#  - Finding T's kata-anchor lambda-poisoning channel (slip-free engine drags ceiling to 0) is
#    CLOSED by construction here — M0b pins lambda at 0.02 — so this is the first clean test of
#    a superhuman rung without its known failure mode.
# Why act-only changes the anchor question: act features contain NO ladder — anchor rows are
# the model's ONLY difficulty-location signal (Finding PP says anchor value is feature-set-
# conditional: bw ≈ redundant, hybrid irreducible; act-only should be the extreme case), and
# the t0.02 rung adds a hard-tail point that act features have no proxy for at all.
# Refs: M0b bw_act 0.4160/0.5385/0.7977/hv .0109/fb .0161; R11 act-only baseline (clamp .2)
#   0.4460/0.5587/0.7864/.0153; act-only candidate 0.4257/0.5519/0.7545/.0138;
#   R12 third-rung-on-bw_act 0.4258/0.5821/0.7919/.0193 (lambda died 0.0046 — now pinned).
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
KATA="kata1-b18c384nbt-s9996604416-d4316597426_t0.02"
J="--no-dro-skill --holdout-problem-frac 0.1 --slope-floor 0.01 --hier-slope --floor-residual --resid-cols diff,slope,floor --floor-resid-l2 4.0 --floor-resid-mode detached --mech-floor --mech-floor-scale-fixed 0.70 --lapse-fixed 0.02 --features act"
$P -m auxiliary.train_rating_v2_research $J                                      > experiments_4pl/j1_m0b_act.log 2>&1
$P -m auxiliary.train_rating_v2_research $J --anchor-ranks "20k,9d,$KATA"        > experiments_4pl/j2_m0b_act_kata.log 2>&1
echo CHAIN36_DONE
