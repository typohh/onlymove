"""Constants and the clear-point helper shared across the rating pipeline.

These describe the FEATURE LAYOUT and the 4PL parameterization, so every stage
that touches per-problem parameters needs them: the trainer, the per-cell axis
fanout, the rank mapping, the population constants, packaging, and the feature
loaders themselves. They live here rather than inside any one trainer so that a
trainer can be replaced without every other stage following it.

`PCA_DIM` = 128 is MEASURED-OPTIMAL and should not be swept again (2026-07-27):
8/16/32/64/128/256 were each run end to end, 128 wins from both directions, and
256 is refuted, carrying the memorization signature.
"""

import numpy as np

from utils.constants import ALL_BOARD_SIZES
from utils.rating.fit_rank_curves import CALIBRATION_P

K_AXES = 3                       # skill axes fitted alongside the rating
EPS = 1e-6
SIZES = list(ALL_BOARD_SIZES)
SEED = 0
PCA_DIM = 128                    # joint-PCA activation feature dimensionality
DIFF_HIDDEN = (128, 64)          # feature trunk hidden layers
DROPOUT = 0.3                    # trunk dropout
INPUT_DROPOUT = 0.3              # COLUMN dropout over the wide input region
SLOPE_FLOOR = 0.01               # lower clamp on the per-problem slope a_p
LAPSE_CAP = 0.05                 # upper bound on the global ceiling slip lambda
LAPSE_INIT = 0.02                # lambda init (lambda is learned)
GAUGE_MOM = 0.01                 # EMA momentum of the gauge's whitening buffers


def clear_logit_vec(c, lapse=0.0):
    """Per-problem 80%-clear logit anchor under the 4PL.

    At the rating P(clear) = CALIBRATION_P, so sigmoid(a*(theta-b)) =
    (P* - c)/(1 - c - lambda) and rating = b + logit(...)/a. Vector over
    problems. `lapse` is the global 4PL slip rate and must match the client's
    clearKappa, or the shipped rating is not the 80%-clear point it claims."""
    c = np.asarray(c, dtype=np.float64)
    q = np.clip((CALIBRATION_P - c) / (1.0 - c - lapse), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))
