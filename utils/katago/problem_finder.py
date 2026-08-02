import math
import random

from utils.constants import debug, DEBUG
from utils.katago.queries import KatagoQueries, GTP_COLS

MIN_WINRATE_CORRECT = 0.6
MAX_WINRATE_ALTERNATIVE = 0.4
MIN_WINRATE_CORRECT_VERIFY = 0.8
MAX_WINRATE_ALTERNATIVE_VERIFY = 0.2
RANDOM_MOVE_COUNT = 4
PLAYOUTS = 10
PLAYOUTS_VERIFY = 1000
PLAYOUTS_GAME = 30


class ProblemFinder:
    """
    Generates tsumego problems and validates their difficulty.
    """

    def __init__(self, queries: KatagoQueries):
        """
        Args:
            queries (KatagoQueries): The KataGo controller for the SL network.
        """
        self.queries = queries
        self.board_size = queries.board_size
        # First two and last two columns/rows of this board are off-limits for
        # the random opening moves (1st/2nd line). Derived from the board size
        # the queries object is bound to.
        self.edge_cols = {
            GTP_COLS[0], GTP_COLS[1],
            GTP_COLS[self.board_size - 2], GTP_COLS[self.board_size - 1],
        }
        self.edge_rows = {"1", "2", str(self.board_size - 1), str(self.board_size)}

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
            if move[0] in self.edge_cols:
                return False
            if move[1:] in self.edge_rows:
                return False
            return True

        # Play random moves, anywhere except 1st and 2nd lines. The draw is
        # deliberately UNIFORM over legal non-edge cells (not policy-weighted)
        # to maximize opening-position diversity in the generated corpus.
        for _ in range(RANDOM_MOVE_COUNT):
            response = self.queries.query(moves)
            legal_idxs = [
                i
                for i, p in enumerate(response["policy"])
                if p > 0 and is_valid(self.queries.get_move_from_idx(i))
            ]

            chosen_idx = random.choices(legal_idxs)[0]
            move = self.queries.get_move_from_idx(chosen_idx)

            moves.append(move)

        if DEBUG:
            _, winrate, score, _ = self.queries.get_best_move(self.queries.query(moves))
            debug(f"status after {RANDOM_MOVE_COUNT} random moves: {(100 * winrate):.1f}% {score:.1f}")

        return moves

    def is_valid(self, moves):
        """Verify the position is a sound tsumego and RETURN the move that wins it.

        Escalates from PLAYOUTS to PLAYOUTS_VERIFY playouts and plays both the
        winning and losing lines to the end. On success returns the deep-read
        best move (the final, highest-playout iteration's best_move); returns
        None on any failure. The caller's cheap 10-playout best_move is only a
        pre-filter — it can disagree with this deep-read move, which is precisely
        the case we want to keep (a policy-tempting move refuted by reading), so
        the validated move, not the shallow one, becomes the solution.
        """
        best_move = None
        steps = 20
        for i in range(steps):
            p = (i + 1) / steps
            playouts = int(PLAYOUTS + (PLAYOUTS_VERIFY - PLAYOUTS) * p)
            min_winrate_correct = MIN_WINRATE_CORRECT + (MIN_WINRATE_CORRECT_VERIFY - MIN_WINRATE_CORRECT) * p
            max_winrate_alternative = MAX_WINRATE_ALTERNATIVE + (MAX_WINRATE_ALTERNATIVE_VERIFY - MAX_WINRATE_ALTERNATIVE) * p

            best_move, best_winrate, best_score, _ = self.queries.get_best_move(self.queries.query(moves, playouts=playouts))
            second_move, second_winrate, second_score, _ = self.queries.get_best_move(self.queries.query(moves, exclude_move=best_move, playouts=playouts))
            debug(f"verify: {i}/{steps} {(100 * best_winrate):.1f}%, {(100 * second_winrate):.1f}%, {best_score:.1f}, {second_score:.1f}")
            if best_winrate < min_winrate_correct or second_winrate > max_winrate_alternative or second_score > 0:
                debug("looked promising, but failed after more playouts")
                return None

        # and lets play the game to the end, just to make sure we arrive at the same conclusion..
        if not self._play_line_to_end(moves, best_move, expect_win=True):
            return None
        if not self._play_line_to_end(moves, second_move, expect_win=False):
            return None

        # best_move here is the final (highest-playout) iteration's value — the move
        # deep reading is confident wins, which may differ from the shallow policy move.
        return best_move

    def _play_line_to_end(self, moves, first_move, expect_win):
        """Play `first_move`, then best-move responses to the double-pass end, and
        confirm black stays on the expected side at every step AND at the final
        position. expect_win=True requires black to stay ahead (bail if
        winrate < 0.5 or score < 0); expect_win=False requires black to stay
        behind (bail if winrate > 0.5 or score > 0). Returns True iff the whole
        line confirms the expectation (the caller maps False → None)."""
        game_moves = moves.copy()
        game_moves.append(first_move)
        # Depth cap: KataGo can refuse to pass in a settled position and play
        # on indefinitely (add_variations guards the same failure mode with
        # MAX_VARIATION_DEPTH). Bail rather than loop forever — a line this long
        # is not a clean tsumego resolution anyway.
        max_line_len = len(moves) + 2 * self.board_size * self.board_size
        while game_moves[-1] != 'pass' or game_moves[-2] != 'pass':
            if len(game_moves) > max_line_len:
                debug(f"line exceeded {max_line_len} moves without a double pass; bailing")
                return False
            move, winrate, score, _ = self.queries.get_best_move(self.queries.query(game_moves, playouts=PLAYOUTS_GAME))
            fell_off = (winrate < 0.5 or score < 0) if expect_win else (winrate > 0.5 or score > 0)
            if fell_off:
                n = len(game_moves) - len(moves)
                if expect_win:
                    debug(f"looked promising, but black fell behind after {n} moves.")
                else:
                    debug(f"looked promising, but despite wrong move black got ahead after {n} moves.")
                return False
            game_moves.append(move)

        score, winrate = self.queries.get_position_info(game_moves)
        debug(f"playing {'correct' if expect_win else 'incorrect'} move to the end : {(100 * winrate):.1f}% {score:.1f}")
        fell_off = (winrate < 0.5 or score < 0) if expect_win else (winrate > 0.5 or score > 0)
        if fell_off:
            if expect_win:
                debug("looked promising, but playing to the end didnt give conclusive win")
            else:
                debug("looked promising, but playing to the end didnt give conclusive loss despite second best move")
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
            penalty = penalty_winrate + penalty_policy * 0.1 + random.gauss(0, 0.05)
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
        policies = [0 for _ in range(len(moves))]
        # potential = {}

        while True:
            best_move, best_winrate, best_score, best_policy = self.queries.get_best_move(self.queries.query(moves, playouts=PLAYOUTS))
            second_move, second_winrate, second_score, second_policy = self.queries.get_best_move(
                self.queries.query(moves, exclude_move=best_move, playouts=PLAYOUTS))

            debug(
                f"{(100 * best_winrate):.1f}% vs {(100 * second_winrate):.1f}%, {best_score:.1f} vs {second_score:.1f} , {100 * best_policy:.1f}% vs {100 * second_policy:.1f}%")

            if best_move == "pass" or second_move == "pass":
                debug("terminating due to pass")
                return None, None

            # do we meet all criteria? The 10-playout best_move/second_move here are a
            # cheap pre-filter; is_valid re-reads to PLAYOUTS_VERIFY and returns the move
            # it actually validated to the end, which is the solution we keep — even when
            # it differs from the shallow policy move (the case we want: a tempting move
            # refuted by deeper reading).
            if best_winrate > MIN_WINRATE_CORRECT and second_winrate < MAX_WINRATE_ALTERNATIVE and best_score > 0 and second_score < 0:
                solution = self.is_valid(moves)
                if solution:
                    debug("found tsumego!")
                    debug(moves)
                    debug(str(policies))
                    return moves, solution

            if second_winrate > MIN_WINRATE_CORRECT_VERIFY and second_score > 0.5:  # we are winning too much, need to lose points..
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
                return None, None
            debug(f"W {balanced_move} {(balanced_winrate * 100):.1f}% {balanced_score:.1f} {100 * balanced_policy:.2f}%")
            moves.append(balanced_move)
            policies.append(balanced_policy)

            if policies[-1] < 0.01 or policies[-2] < 0.01:
                debug ("recent moves are too ugly, discarding game..")
                return None, None

    def create_problem(self):

        intial_position = self.generate_initial_position()

        moves, solution = self.adjust_tsumego(intial_position)
        if not solution:
            return None, None

        return moves, solution
