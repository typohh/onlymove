import time

DEBUG = True

class Stats:
    def __init__(self):
        self.games_processed = 0
        self.tsumego_creation_attempts = 0
        self.problems_found = 0
        self.total_winrate_corrections = 0
        self.total_score_corrections = 0
        self.failed_tsumego_creation = 0
        self.start_time = time.time()

    def add_game_processed(self):
        self.games_processed += 1

    def add_tsumego_creation_attempt(self):
        self.tsumego_creation_attempts += 1

    def add_problem_found(self):
        self.problems_found += 1

    def add_winrate_correction(self):
        self.total_winrate_corrections += 1

    def add_score_correction(self):
        self.total_score_corrections += 1

    def add_failed_tsumego_creation(self):
        self.failed_tsumego_creation += 1

    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        tsumegos_per_minute = (self.problems_found / elapsed_time) * 60 if elapsed_time > 0 else 0
        stats = {
            "Running Time": f"{elapsed_time:.2f}s",
            "Tsumego Creation Attempts": self.tsumego_creation_attempts,
            "Tsumegos Creations Failed": self.failed_tsumego_creation,
            "Problems Found": self.problems_found,
            "Avg Winrate adjustments per Attempt": f"{((
                                                             self.total_winrate_corrections / self.tsumego_creation_attempts
            ) if self.tsumego_creation_attempts > 0 else 0):.2f}",
            "Avg Score adjustments per Attempt": f"{((
                                                             self.total_score_corrections / self.tsumego_creation_attempts
            ) if self.tsumego_creation_attempts > 0 else 0):.2f}",
            "Tsumegos per Minute": f"{tsumegos_per_minute:.2f}",
        }
        return stats

STATS = Stats()