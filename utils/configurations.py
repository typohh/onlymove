from datetime import datetime

from utils.models import MODEL_FULL

DEBUG = False
BOARD_SIZE = 13
ALL_BOARD_SIZES = (9, 13)
DEFAULT_MODEL = MODEL_FULL
KOMI = 6.5
RULES = "japanese"


def debug(message: str):
    if DEBUG:
        print(datetime.now().strftime("%H:%M:%S"), message)
