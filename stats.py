import time

DEBUG = False


class Stats:
    def __init__(self):
        self.problems_found = 0
        self.failures_to_create = 0
        self.total_number_of_moves = 0
        self.start_time = time.time()

    def add_problem_found(self, number_of_moves, move_rank, game_rank):
        self.problems_found += 1
        self.total_number_of_moves += number_of_moves

    def add_failure_to_create_problem(self):
        self.failures_to_create += 1

    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        tsumegos_per_minute = (self.problems_found / elapsed_time) * 60 if elapsed_time > 0 else 0
        average_move_length = self.total_number_of_moves / max(1, self.problems_found)
        stats = {
            "Running Time": f"{elapsed_time:.2f}s",
            "Games processed": self.problems_found + self.failures_to_create,
            "Success rate": f"{(100*self.problems_found/max(1,self.problems_found + self.failures_to_create)):.1f}%",
            "Avg moves": f"{average_move_length:.1f}",
            "Tsumegos per Minute": f"{tsumegos_per_minute:.2f}",
        }
        return stats


STATS = Stats()
