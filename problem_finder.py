import os
import uuid

from sgfmill import sgf

from Stats import STATS
from difficulty_evaluator import DifficultyEvaluator
from game_generator import katago_to_sgf_coords
from katago_queries import KatagoQueries

MIN_SCORE = 0
MAX_SCORE = 2
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

    def adjust_tsumego(self, moves):
        """
        play additional exchanges in an attempt to get score for BLACK within MIN_SCORE and MAX_SCORE. and so there is only one move with winrate abeove MAX_WINRATE_ALTERNATIVE and it has winrate above MIN_WINRATE_CORRECT.
        """

        # We want the tsumego to have large gap in winrate between best and second best move. We want the score of the second best move to be close to -0.5 as possible.

        # if multiple moves above MAX_WINRATE_ALTERNATIVE, remove all but one.
        while True:
            best_move, best_winrate, best_score = self.queries.get_best_move(self.queries.query(moves))
            second_move, second_winrate, second_score = self.queries.get_best_move(self.queries.query(moves, exclude_move=best_move))

            # do we meet all criteria?
            if best_winrate > MIN_WINRATE_CORRECT and second_winrate < MAX_WINRATE_ALTERNATIVE and MIN_SCORE < second_score < MAX_SCORE :
                return moves,best_move

            if best_move == "pass":
                STATS.add_problems_found( "black best move was pass")
                return None, None


            # black plays second best move
            moves.append(second_move)

            response = self.queries.query_children_low(moves)

            if best_winrate < MIN_WINRATE_CORRECT or second_winrate > MAX_WINRATE_ALTERNATIVE:
                STATS.add_winrate_correction()
                # Find a move that gets black's winrate just above MIN_WINRATE_CORRECT
                best_response = None
                best_response_penalty = float('inf')

                for move_info in response["moveInfos"]:
                    test_winrate = move_info["winrate"]

                    penalty = abs(test_winrate - MIN_WINRATE_CORRECT)
                    if test_winrate > MIN_WINRATE_CORRECT:
                        penalty *= 10
                    if penalty < best_response_penalty:
                        best_response_penalty = penalty
                        best_response = move_info["move"]

                moves.append(best_response)
            else:
                STATS.add_score_correction()
                # Find move that makes score difference closest to target
                best_response = None
                best_response_penalty = float('inf')
                target_score = best_score - second_score - 1

                for move_info in response["moveInfos"]:
                    penalty = abs(move_info["scoreLead"] - target_score)
                    if penalty < best_response_penalty:
                        best_response_penalty = penalty
                        best_response = move_info["move"]

                moves.append(best_response)

    def validate_tsumego(self, moves):
        """
        Validates position to to ensure its score 0-1 for black, only one move with positive score and winrate above 50%, and correct move is in range 19kyu-8dan.

        Args:
            moves (list): The list of moves in the game.

        Returns:
            solution_move or None if not valid tsumego
        """
        # Now verify the solution move
        solution_response = self.queries.query(moves, True)
        solution_move_info = solution_response["moveInfos"][0]

        solution_move = solution_move_info["move"]
        solution_winrate = solution_move_info["winrate"]
        solution_score = solution_move_info["scoreLead"]
        if solution_winrate < MIN_WINRATE_CORRECT:
            STATS.add_rejection_reason("Winrate too low")
            return None
        if solution_score < MIN_SCORE:
            STATS.add_rejection_reason("Score too low")
            return None
        if not solution_score > MAX_SCORE:
            STATS.add_rejection_reason("Score too high")
            return None

        # Check alternative moves
        exclude_response = self.queries.query(moves, True, solution_move)
        for move_info in exclude_response["moveInfos"]:
            if move_info["winrate"] > MAX_WINRATE_ALTERNATIVE:
                STATS.add_rejection_reason("Alternative too good")
                return None

        return solution_move

    def _adjust_score(self, moves):
        """
        Adjusts position to ensure solution leads to ~(MIN_SCORE+MAX_SCORE)/2 win.
        Performs a single exchange (player move + opponent response) if needed.
        Returns original sequence if no adjustments were needed or possible.

        Args:
            moves (list): Current move sequence

        Returns:
            list: Adjusted move sequence
        """
        # Query for moves in current position (player's turn)
        response = self.queries.query_children_low(moves)

        best_candidate = None
        best_candidate_penalty = float('inf')
        best_candidate_winrate = None
        best_candidate_score = None

        # Find best candidate move for player
        for move_info in response["moveInfos"]:
            candidate_move = move_info["move"]
            if "scoreLead" not in move_info or "visits" not in move_info:
                continue

            candidate_score = move_info["scoreLead"]
            candidate_winrate = move_info["winrate"]
            penalty = abs(candidate_score - (MAX_SCORE + MIN_SCORE) / 2) + 0.25 * (1 - candidate_winrate)
            # penalty = abs(candidate_score - 0.5)

            if penalty < best_candidate_penalty:
                best_candidate_penalty = penalty
                best_candidate = candidate_move
                best_candidate_winrate = candidate_winrate
                best_candidate_score = candidate_score

        # Apply player's candidate move
        adjusted_moves = moves + [best_candidate]

        return adjusted_moves, best_candidate, best_candidate_winrate, best_candidate_score

    def find_problems(self, game_moves, output_dir):
        """
        Finds problems in a game sequence.

        Args:
            game_moves (list): The list of moves in the game.
            output_dir (str): The directory to save problems to.

        """
        # Analyze all moves except first 10 and last 10
        for i in list(reversed(range(10, len(game_moves) - 10))):
            if game_moves[i][0] != "B":
                continue
            STATS.add_tsumego_creation_attempt()
            moves_so_far = game_moves[:i]

            adjusted_moves,solution_move = self.adjust_tsumego(moves_so_far.copy())
            if not solution_move:
                continue

            move_mean, move_stddev = self.difficulty.assess_move_difficulty(adjusted_moves, solution_move)
            game_mean, game_stddev = self.difficulty.assess_game_difficulty(adjusted_moves)

            STATS.add_problems_found(1)
            self.save_problem_as_sgf({
                "moves": adjusted_moves,
                "solution": solution_move,
                "move_rank": self.difficulty.get_rank_str(move_mean, move_stddev),
                "game_rank": self.difficulty.get_rank_str(game_mean, game_stddev)
            }, output_dir)
        STATS.add_game_processed()

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
        game.root.set("GN", f"Next move difficulty: {problem["move_rank"]}, Remaining game difficulty: {problem['game_rank']}")

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
        solution_node.set("C", f"Next move difficulty: {problem["move_rank"]}, Remaining game difficulty: {problem['game_rank']}")
        solution_node.set("MA", [katago_to_sgf_coords(solution_move, 9)])

        with open(filename, "wb") as f:
            f.write(game.serialise())
