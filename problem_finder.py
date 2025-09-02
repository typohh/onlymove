import math
import random

from configurations import debug, DEBUG
from katago_queries import KatagoQueries

MIN_WINRATE_CORRECT = 0.6
MAX_WINRATE_ALTERNATIVE = 0.4
MIN_WINRATE_CORRECT_VERIFY = 0.8
MAX_WINRATE_ALTERNATIVE_VERIFY = 0.2
RANDOM_MOVE_COUNT = 4
PLAYOUTS = 10
PLAYOUTS_VERIFY = 1000


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
            if move[0] == "A" or move[0] == "B" or move[0] == "H" or move[0] == "J":
                return False
            if move[1] == "1" or move[1] == "2" or move[1] == "8" or move[1] == "9":
                return False
            return True

        # Play random moves, anywhere except 1st line
        for _ in range(RANDOM_MOVE_COUNT):
            response = self.queries.query(moves)
            legal_moves = [
                (i, p)
                for i, p in enumerate(response["policy"])
                if p > 0 and is_valid(self.queries.get_move_from_idx(i))
            ]

            move_indices, probabilities = zip(*legal_moves)
            chosen_idx = random.choices(move_indices)[0]
            move = self.queries.get_move_from_idx(chosen_idx)

            moves.append(move)

        if DEBUG:
            _, winrate, score, _ = self.queries.get_best_move(self.queries.query(moves))
            debug(f"status after {RANDOM_MOVE_COUNT} random moves: {(100 * winrate):.1f}% {score:.1f}")

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
            _, winrate, score, _ = self.queries.get_best_move(self.queries.query(moves))
            debug(f"status after rebalance: {(100 * winrate):.1f}% {score:.1f}")

        return moves

    def is_valid(self, moves):
        steps = 20
        for i in range(steps):
            p = (i + 1) / steps
            playouts = int(PLAYOUTS + (PLAYOUTS_VERIFY - PLAYOUTS) * p)
            min_winrate_correct = MIN_WINRATE_CORRECT + (MIN_WINRATE_CORRECT_VERIFY - MIN_WINRATE_CORRECT) * p
            max_winrate_alternative = MAX_WINRATE_ALTERNATIVE + (MAX_WINRATE_ALTERNATIVE_VERIFY - MAX_WINRATE_ALTERNATIVE) * p

            best_move, best_winrate, best_score, _ = self.queries.get_best_move(self.queries.query(moves, playouts=playouts))
            _, second_winrate, second_score, _ = self.queries.get_best_move(self.queries.query(moves, exclude_move=best_move, playouts=playouts))
            debug(f"verify: {i}/{steps} {(100 * best_winrate):.1f}%, {(100 * second_winrate):.1f}%, {best_score:.1f}, {second_score:.1f}")
            if best_winrate < min_winrate_correct or second_winrate > max_winrate_alternative or second_score > 0:
                debug("looked promising, but failed after more playouts")
                return False
        return True

    def _rebalance_game(self, moves):
        response = self.queries.query_children(moves)

        # Find move that makes score difference closest to target
        best_move = None
        best_response_penalty = float('inf')
        best_score = 0
        best_winrate = 0.5
        best_policy = 1

        for move_info in response["moveInfos"]:
            move = move_info["move"]
            policy = response["policy"][self.queries.get_idx_from_move(move)]
            winrate = move_info["winrate"]
            score = move_info["scoreLead"]
            penalty_winrate = (abs(winrate - MIN_WINRATE_CORRECT) * 2) ** 2
            penalty_policy = -math.log(policy)
            penalty = penalty_winrate + penalty_policy * 0.05 + random.gauss(0,0.05)
            if penalty < best_response_penalty:
                best_response_penalty = penalty
                best_move = move
                best_policy = policy
                best_score = score
                best_winrate = winrate

        return best_move, best_score, best_winrate, best_policy

    def adjust_tsumego(self, moves):
        """
        play additional exchanges in an attempt to get winrate above WINRATE_CORRECT and second best move below WINRATE_ALTERNATIVE and score within MIN_SCORE, MAX_SCORE for second best move.
        """

        debug("adjusting tsumego")
        policies = [ 0 for _ in range(len(moves))]

        while True:
            best_move, best_winrate, best_score, best_policy = self.queries.get_best_move(self.queries.query(moves, playouts=PLAYOUTS))
            second_move, second_winrate, second_score, second_policy = self.queries.get_best_move(
                self.queries.query(moves, exclude_move=best_move, playouts=PLAYOUTS))

            debug(
                f"{(100 * best_winrate):.1f}% vs {(100 * second_winrate):.1f}%, {best_score:.1f} vs {second_score:.1f} , {100*best_policy:.1f}% vs {100*second_policy:.1f}%")

            # do we meet all criteria?
            if best_winrate > MIN_WINRATE_CORRECT and second_winrate < MAX_WINRATE_ALTERNATIVE and second_score < 0:
                if self.is_valid(moves):
                    debug("found tsumego!")
                    debug( moves )
                    debug( policies )
                    return moves, best_move

            if best_move == "pass" or second_move == "pass":
                debug("terminating due to pass")
                return None, None

            if second_winrate > MIN_WINRATE_CORRECT_VERIFY:  # we are winning too much, need to lose points..
                balanced_move, balanced_score, balanced_winrate, balanced_policy = self._rebalance_game(moves)
                debug(f"B {balanced_move} {(balanced_winrate * 100):.1f}% {balanced_score:.1f} {100 * balanced_policy:.1f}% (rebalancing game)")
                moves.append(balanced_move)
                policies.append(balanced_policy)
            elif second_policy * 5 > best_policy and second_winrate > MIN_WINRATE_CORRECT:  # second best move is good enough and natural enough
                debug(f"B {second_move} {(second_winrate * 100):.1f}% {second_score:.1f} {100 * second_policy:.1f}% (increasing the gap)")
                moves.append(second_move)
                policies.append(second_policy)
            elif second_score > best_score - 0.5 and second_winrate > MIN_WINRATE_CORRECT:  # second best move is likely just a ko threat, lets play it
                debug(f"B {second_move} {(second_winrate * 100):.1f}% {second_score:.1f} {100 * second_policy:.1f}% (removing irrelevant sente)")
                moves.append(second_move)
                policies.append(second_policy)
            else:  # we are behind too much, lets just play the best move
                debug(f"B {best_move} {(best_winrate * 100):.1f}% {best_score:.1f} {(100 * best_policy):.1f}%")
                moves.append(best_move)
                policies.append(best_policy)

            balanced_move, balanced_score, balanced_winrate, balanced_policy = self._rebalance_game(moves)
            if balanced_move == 'pass':
                return None,None
            debug(f"W {balanced_move} {(balanced_winrate * 100):.1f}% {balanced_score:.1f} {100 * balanced_policy:.2f}%")
            moves.append(balanced_move)
            policies.append(balanced_policy)

    def create_problem(self):

        intial_position = self.generate_initial_position()

        moves, solution = self.adjust_tsumego(intial_position)
        if not solution:
            return None, None

        return moves, solution
