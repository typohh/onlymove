#!/bin/bash
# Chain 11: general-KataGo-engine anchor A/B (user question: replace 9d humanv0?).
# Trade on record: a superhuman rung pins hard-tail difficulty ordering (the strongest
# examinee owns it; 9d saturates there) BUT is a more alien examinee than humanv0 (curve-
# shape mismatch is the measured bot-anchor failure mode). Baseline ref: erm_20k9d_calib2.
cd /home/ootakamoku/Repositories/onlymove
KATA="kata1-b18c384nbt-s9996604416-d4316597426"
P=./.venv/bin/python3; E="--no-dro-skill --holdout-problem-frac 0.1"
$P -m auxiliary.train_rating_v2_research $E --anchor-ranks "20k,${KATA}_t0.02" > experiments_4pl/erm_20k_kata002.log 2>&1
$P -m auxiliary.train_rating_v2_research $E --anchor-ranks "20k,9d,${KATA}_t0.02" > experiments_4pl/erm_20k9d_kata002.log 2>&1
$P -m auxiliary.train_rating_v2_research $E --anchor-ranks "20k,${KATA}_t1" > experiments_4pl/erm_20k_kata1.log 2>&1
echo CHAIN11_DONE
