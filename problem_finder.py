import random

from sgfmill import sgf

from stats import STATS, DEBUG
from katago_queries import KatagoQueries, katago_to_sgf_coords, get_rank_str

MIN_WINRATE_CORRECT = 0.7
MAX_WINRATE_ALTERNATIVE = 0.3
GOAL_SCORE = -3

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

    def generate_initial_position(self):
        """
        Generates a single game by starting with random moves and then balancing the game.

        Returns:
            list: A list of moves representing the game sequence.
        """
        moves = []

        def is_valid(move):
            if move == "pass":
                return False
            if move[0] == "A" or move[0] == "J":
                return False
            if move[1] == "1" or move[1] == "9":
                return False
            return True

        # Play 6 random moves, anywhere except 1st line
        for _ in range(6):
            response = self.queries.query_policy(moves)
            legal_moves = [
                (i, p)
                for i, p in enumerate(response["policy"])
                if p > 0 and is_valid(self.queries.get_move_from_idx(i))
            ]

            # Weighted random sampling based on policy
            move_indices, probabilities = zip(*legal_moves)
            # total_prob = sum(probabilities)
            # normalized_probs = [p / total_prob for p in probabilities]
            chosen_idx = random.choices(move_indices)[0]
            move = self.queries.get_move_from_idx(chosen_idx)

            moves.append(move)

        if DEBUG:
            move, winrate, score = self.queries.get_best_move(self.queries.query_policy(moves))
            print(f"status after 6 random moves: {(100 * winrate):.1f}% {score:.1f}")

        # 2. Play 2 moves to balance the game
        for _ in range(2):
            response = self.queries.query_children(moves)
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
        play additional exchanges in an attempt to get winrate above WINRATE_CORRECT and second best move below WINRATE_ALTERNATIVE and score within MIN_SCORE, MAX_SCORE for second best move.
        """

        # We want the tsumego to have large gap in winrate between best and second best move. We want the score of the second best move to be close to -0.5 as possible.

        if DEBUG: print("adjusting tsumego")

        while True:
            best_move, best_winrate, best_score = self.queries.get_best_move(self.queries.query(moves))
            second_move, second_winrate, second_score = self.queries.get_best_move(self.queries.query(moves, exclude_move=best_move))

            if DEBUG: print(f"{(100 * best_winrate):.1f}%, {(100 * second_winrate):.1f}%, {best_score:.1f}, {second_score:.1f}")

            # do we meet all criteria?
            if best_winrate > MIN_WINRATE_CORRECT and second_winrate < MAX_WINRATE_ALTERNATIVE and second_score < 0:
                if DEBUG: print("valid tsumego!")
                return moves, best_move

            if best_move == "pass" or second_move == "pass":
                if DEBUG: print("terminating due to pass")
                return None, None

            # black plays second best move
            if DEBUG: print(f"B {second_move} {(second_winrate * 100):.1f}% {second_score:.1f}")
            moves.append(second_move)

            response = self.queries.query_children(moves)

            # Find move that makes score difference closest to target
            best_response = None
            best_response_penalty = float('inf')
            best_score = 0
            best_winrate = 0.5
            target_score = best_score - second_score + GOAL_SCORE

            for move_info in response["moveInfos"]:
                penalty = abs(move_info["scoreLead"] - target_score)
                if penalty < best_response_penalty:
                    best_response_penalty = penalty
                    best_response = move_info["move"]
                    best_score = move_info["scoreLead"]
                    best_winrate = move_info["winrate"]

            if DEBUG: print(f"W {best_response} {(best_winrate * 100):.1f}% {best_score:.1f}")
            moves.append(best_response)

    def create_problem(self):

        moves_so_far = self.generate_initial_position()

        adjusted_moves, solution_move = self.adjust_tsumego(moves_so_far.copy())
        if not solution_move:
            STATS.add_failure_to_create_problem()
            return None

        # Generate variations data - all legal moves and their optimal responses
        variations = self._generate_variations(adjusted_moves)

        STATS.add_problem_found(len(adjusted_moves))
        return {
            "moves": adjusted_moves,
            "solution": solution_move,
            "variations": variations
        }

    def _generate_variations(self, moves):
        """
        Generate variations for all legal moves from the current position and their optimal responses.
        
        Args:
            moves (list): The sequence of moves leading to the current position.
            
        Returns:
            list: A list of variations, each containing move, winrate, score, and optional response.
        """
        variations = []

        # Get all legal moves from the current position
        response = self.queries.query_children(moves)

        # For each legal move, get its evaluation and optimal response
        if DEBUG: print("creating", len(response["moveInfos"]), "variations")
        for move_info in response["moveInfos"]:
            move = move_info["move"]
            # Get the optimal response to this move
            response_moves = moves + [move]
            response_result = self.queries.query(response_moves)
            response_move, winrate, score = self.queries.get_best_move(response_result)
            variation = {
                "move": move,
                "response": response_move,
                "winrate": winrate * 100,  # Convert to percentage
                "score": score
            }
            variations.append(variation)

        return variations


def problem_to_sgf(problem):
    """
    Saves a single problem as an SGF file with annotations.

    Args:
        problem (dict): The problem dictionary.
        output_dir (str): The name of the directory for SGF files.
    """

    # print(problem)

    game = sgf.Sgf_game(size=9)
    game.root.set("KM", "7.5")
    game.root.set("RU", "chinese")

    player = "b"

    # Add initial moves
    for move in problem["moves"]:
        node = game.extend_main_sequence()
        node.set_move(player, katago_to_sgf_coords(move, 9))
        player = "w" if player == "b" else "b"

    # Get the node where variations should be attached (the position where player needs to move)
    parent_node = game.get_last_node()

    # Add variations for all legal moves and their responses
    if problem.get("variations"):
        for variation in problem["variations"]:
            # First move in variation (by the same player as the solution)
            var_player = player
            var_node = parent_node.new_child()
            var_node.set_move(var_player, katago_to_sgf_coords(variation["move"], 9))

            # Add comment with winrate and score
            text = "Correct move!" if variation["move"] == problem.get("solution") else "Losing move!"
            var_node.set("C", f"{text} Winrate: {variation['winrate']:.1f}%, Score: {variation['score']:.1f}")

            response_player = "b" if var_player == "w" else "w"
            response_node = var_node.new_child()
            response_node.set_move(response_player, katago_to_sgf_coords(variation["response"], 9))
            response_node.set("C", f"Winrate: {variation['winrate']:.1f}%, Score: {variation['score']:.1f}")

    return game.serialise().decode()
