from datetime import datetime

DEBUG = False
BOARD_SIZE = 9
KOMI = 6.5
RULES = "japanese"


def debug(message: str):
    if DEBUG:
        print(datetime.now().strftime("%H:%M:%S"), message)
