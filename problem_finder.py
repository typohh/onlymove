import os
import uuid
import random

from sgfmill import sgf

from Stats import STATS, DEBUG
from difficulty_evaluator import DifficultyEvaluator
from katago_queries import KatagoQueries, katago_to_sgf_coords

MIN_SCORE = -2
MAX_SCORE = 0
MIN_WINRATE_CORRECT = 0.6
MAX_WINRATE_ALTERNATIVE = 0.4


class ProblemFinder:
    """
    Assesses the difficulty of a move.
    """

    def __init__(self, queries: KatagoQueries):
        """
        Initializes the DifficultyAssessor.

        Args:
            queries (KatagoQueries): The KataGo controller for the SL network.
        """
        self.queries = queries
        self.difficulty = DifficultyEvaluator(queries)

    def generate_initial_position(self):
        """
        Generates a single game by starting with random moves and then balancing the game.

        Returns:
            list: A list of moves representing the game sequence.
        """
        moves = []

        # 1. Play 4 random moves
        for _ in range(4):
            response = self.queries.query_policy(moves)
            # Filter out pass moves and moves with very low policy values
            legal_moves = [
                (i, p)
                for i, p in enumerate(response["policy"])
                if p > 0 and self.queries.get_move_from_idx(i) != "pass"
            ]

            # Weighted random sampling based on policy
            move_indices, probabilities = zip(*legal_moves)
            total_prob = sum(probabilities)
            normalized_probs = [p / total_prob for p in probabilities]
            chosen_idx = random.choices(move_indices, weights=normalized_probs)[0]
            move = self.queries.get_move_from_idx(chosen_idx)

            moves.append(move)

        if DEBUG:
            move, winrate, score = self.queries.get_best_move(self.queries.query_policy(moves))
            print(f"status after 4 random moves: {(100 * winrate):.1f}% {score:.1f}")

        # 2. Play 2 moves to balance the game
        for _ in range(2):
            response = self.queries.query_children_low(moves)
            if not response.get("moveInfos"):
                break  # No further moves possible

            # Find the move that brings the score closest to 0
            best_move = min(
                response["moveInfos"],
                key=lambda x: abs(x["scoreLead"]) * abs(x["winrate"] - 0.5),
            )["move"]

            moves.append(best_move)

        if DEBUG:
            move, winrate, score = self.queries.get_best_move(self.queries.query_policy(moves))
            print(f"status after rebalance: {(100 * winrate):.1f}% {score:.1f}")

        return moves

    def adjust_tsumego(self, moves):
        """
        play additional exchanges in an attempt to get score for BLACK within MIN_SCORE and MAX_SCORE. and so there is only one move with winrate abeove MAX_WINRATE_ALTERNATIVE and it has winrate above MIN_WINRATE_CORRECT.
        """

        # We want the tsumego to have large gap in winrate between best and second best move. We want the score of the second best move to be close to -0.5 as possible.

        if DEBUG: print("adjusting tsumego")

        # if multiple moves above MAX_WINRATE_ALTERNATIVE, remove all but one.
        while True:
            best_move, best_winrate, best_score = self.queries.get_best_move(self.queries.query(moves))
            second_move, second_winrate, second_score = self.queries.get_best_move(self.queries.query(moves, exclude_move=best_move))

            if DEBUG: print(f"{(100 * best_winrate):.1f}%, {(100 * second_winrate):.1f}%, {best_score:.1f}, {second_score:.1f}")

            # do we meet all criteria?
            if best_winrate > MIN_WINRATE_CORRECT and second_winrate < MAX_WINRATE_ALTERNATIVE and MIN_SCORE < second_score < MAX_SCORE:
                if DEBUG: print("valid tsumego!")
                return moves, best_move

            if best_move == "pass" or second_move == "pass":
                if DEBUG: print("terminating due to pass")
                return None, None

            # black plays second best move
            if DEBUG: print(f"B {second_move} {(second_winrate * 100):.1f}% {second_score:.1f}")
            moves.append(second_move)

            response = self.queries.query_children_low(moves)

            # Find move that makes score difference closest to target
            best_response = None
            best_response_penalty = float('inf')
            best_score = 0
            best_winrate = 0.5
            target_score = best_score - second_score + (MIN_SCORE + MAX_SCORE) / 2

            for move_info in response["moveInfos"]:
                penalty = abs(move_info["scoreLead"] - target_score)
                if penalty < best_response_penalty:
                    best_response_penalty = penalty
                    best_response = move_info["move"]
                    best_score = move_info["scoreLead"]
                    best_winrate = move_info["winrate"]

            if DEBUG: print(f"W {best_response} {(best_winrate * 100):.1f}% {best_score:.1f}")
            moves.append(best_response)

    def create_problem(self, output_dir):

        moves_so_far = self.generate_initial_position()

        adjusted_moves, solution_move = self.adjust_tsumego(moves_so_far.copy())
        if not solution_move:
            STATS.add_failure_to_create_problem()
            return False

        move_mean, move_stddev = self.difficulty.assess_move_difficulty(adjusted_moves, solution_move)
        game_mean, game_stddev = self.difficulty.assess_game_difficulty(adjusted_moves, move_mean)

        STATS.add_problem_found(len(adjusted_moves), move_mean, game_mean)
        self.save_problem_as_sgf({
            "moves": adjusted_moves,
            "solution": solution_move,
            "move_rank": self.difficulty.get_rank_str(move_mean, move_stddev),
            "game_rank": self.difficulty.get_rank_str(game_mean, game_stddev)
        }, output_dir)
        return True

    @staticmethod
    def save_problem_as_sgf(problem, output_dir):
        """
        Saves a single problem as an SGF file with annotations.

        Args:
            problem (dict): The problem dictionary.
            output_dir (str): The name of the directory for SGF files.
        """

        # print(problem)

        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{uuid.uuid4()}.sgf")

        game = sgf.Sgf_game(size=9)
        game.root.set("KM", str(7.5))
        game.root.set("RU", "tromp-taylor")
        game.root.set("GN", f"Last move difficulty: {problem["move_rank"]}, Last move + remaining game difficulty: {problem['game_rank']}")

        player = "b"

        # Add initial moves
        for move in problem["moves"]:
            node = game.extend_main_sequence()
            node.set_move(player, katago_to_sgf_coords(move, 9))
            player = "w" if player == "b" else "b"

        # Add the solution move and annotations
        solution_move = problem["solution"]
        solution_node = game.extend_main_sequence()
        solution_node.set_move(player, katago_to_sgf_coords(solution_move, 9))
        solution_node.set("C", f"This move difficulty: {problem["move_rank"]}, This move + Remaining game difficulty: {problem['game_rank']}")
        solution_node.set("MA", [katago_to_sgf_coords(solution_move, 9)])

        with open(filename, "wb") as f:
            f.write(game.serialise())
