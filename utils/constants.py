import functools
import os
from datetime import datetime

from utils.katago.models import MODEL_FULL
from utils.paths import PROJECT_ROOT

DEBUG = False
ALL_BOARD_SIZES = (9, 13)
DEFAULT_MODEL = MODEL_FULL
KOMI = 6.5
RULES = "japanese"

# Ids of problems removed from the corpus that still carry attempt rows in the
# DB. Lives here because both sides of the id invariant need it and they import
# nothing from each other: the WRITER (pipeline.generate_tsumegos, which must
# never reissue one) and the READERS (utils.rating.features, which must never
# find one back in the corpus).
RETIRED_IDS_PATH = os.path.join(PROJECT_ROOT, "tsumegos", "retired_ids.txt")


@functools.lru_cache(maxsize=1)
def read_retired_ids():
    """The retired id set, read once per process.

    A missing file yields an empty set: this backs a guard against reuse, and a
    generator run should not be blocked because the list has not been produced
    yet."""
    retired = set()
    try:
        with open(RETIRED_IDS_PATH) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    retired.add(line)
    except FileNotFoundError:
        pass
    return retired


def debug(message: str):
    if DEBUG:
        print(datetime.now().strftime("%H:%M:%S"), message)
