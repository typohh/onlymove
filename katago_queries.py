import math
import random
from pprint import pprint

import numpy as np

PLAYOUTS = 30

_SUPPORTED_RANKS = [
    "rank_20k", "rank_19k", "rank_18k", "rank_17k", "rank_16k",
    "rank_15k", "rank_14k", "rank_13k", "rank_12k", "rank_11k",
    "rank_10k", "rank_9k", "rank_8k", "rank_7k", "rank_6k",
    "rank_5k", "rank_4k", "rank_3k", "rank_2k", "rank_1k",
    "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d",
    "rank_6d", "rank_7d", "rank_8d", "rank_9d"
]


def get_all_ranks():
    return [(i + 0.5) / len(_SUPPORTED_RANKS) for i in range(len(_SUPPORTED_RANKS))]


class KatagoQueries:
    """
    A class to manage the KataGo analysis engine process.
    """

    def __init__(self, katago_controller):
        """
        Initializes the KatagoController.

        Args:
            katago_path (str): The path to the KataGo executable.
            model_path (str): The path to the KataGo model file (for the RL network).
            human_model_path (str, optional): The path to the KataGo human SL model file. Defaults to None.
        """
        self._katago_controller = katago_controller

    @staticmethod
    def _include_player_in_move_sequence(moves):
        player_and_move = []
        for move in moves:
            player_and_move.append([_get_player_to_move_next(player_and_move), move])
        return player_and_move

    def query_policy(self, moves, human_rank=None, komi=7.5):
        """
        Sends a standardized query for root position analysis (1 visit).
        Returns rootInfo but no moveInfos.
        """
        query = {
            "moves": self._include_player_in_move_sequence(moves),
            "initialStones": [],
            "rules": "chinese",
            "komi": komi,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 1,
            "includePolicy": True,
        }
        if human_rank is not None:
            # noinspection PyTypeChecker
            query["overrideSettings"] = {
                "humanSLProfile": _get_rank_str(human_rank),
                "ignorePreRootHistory": False,
                "rootNumSymmetriesToSample": 1
            }
        return self._katago_controller.send_query(query)

    def query_weighted_random_move(self, moves, human_rank):
        """
        Queries the policy for a given rank and returns a move based on a weighted random sample.
        """

        response = self.query_policy(moves, human_rank=human_rank)

        if not response.get("humanPolicy"):
            print("moves:", moves)
            print("human_rank:", human_rank, _get_rank_str(human_rank))
            print("response:")
            pprint(response)
            response = self.query_policy(moves, human_rank=human_rank)
            if not response.get("humanPolicy"):
                raise Exception("Human policy not found in response.")
            else:
                raise Exception("Human policy found in repeat query.")
            # return "pass"

        human_policy = response["humanPolicy"]
        strong_policy = response["policy"]

        human_legal_indices = {i for i, p in enumerate(human_policy) if p > 0}
        strong_legal_indices = {i for i, p in enumerate(strong_policy) if p > 0}

        if human_legal_indices != strong_legal_indices:
            raise Exception("Human and strong policies disagree on legal moves.")

        legal_moves_map = {
            i: p for i, p in enumerate(human_policy) if i in strong_legal_indices
        }

        pass_idx = len(strong_policy) - 1
        if pass_idx in legal_moves_map:
            legal_moves_map[pass_idx] = strong_policy[pass_idx]

        legal_moves = list(legal_moves_map.items())

        if not legal_moves:
            return "pass"

        move_indices, probabilities = zip(*legal_moves)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        chosen_index = np.random.choice(move_indices, p=probabilities)

        return self.get_move_from_idx(chosen_index)

    def query_children(self, moves):
        """
        Sends a standardized query for children analysis (visits adjusted by the number of legal moves).
        Returns moveInfos but not necessarily accurate rootInfo.
        """
        # Get root policy to count legal moves
        root_response = self.query_policy(moves)
        num_legal_moves = sum(1 for p in root_response["policy"] if p >= 0)

        query = {
            "moves": self._include_player_in_move_sequence(moves),
            "initialStones": [],
            "rules": "chinese",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": num_legal_moves * int(PLAYOUTS ** 0.5),
            "rootPolicyTemperature": 100,
            "overrideSettings": {
                "wideRootNoise": 5.0
            }
        }
        r = self._katago_controller.send_query(query)
        if len(r["moveInfos"]) < num_legal_moves:
            move_info_count = len(r["moveInfos"])
            print("moves:", moves)
            print(f"WARNING: Not enough moveInfos found in response: {move_info_count} vs {num_legal_moves}")
        return r

    def query_human(self, moves, human_rank, playouts):
        """
        Provides a high playout query, where specific first move may be excluded.
        """

        utility_limit = 0.5

        query = {
            "moves": self._include_player_in_move_sequence(moves),
            "initialStones": [],
            "rules": "chinese",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": playouts,
            "includePolicy": True,
            "overrideSettings": {
                "humanSLProfile": _get_rank_str(human_rank),
                "ignorePreRootHistory": False,
                "humanSLPlaExploreProbWeightful": 1.0,
                "humanSLOppExploreProbWeightful": 1.0,
                "humanSLCpuctPermanent": utility_limit,
                "useNoisePruning": False,
                "useUncertainty": False,
                "subtreeValueBiasFactor": 0,
            }
        }
        response = self._katago_controller.send_query(query)

        human_policy = response["humanPolicy"]

        # pprint(response)
        # exit(-1)

        moves = {
            self.get_move_from_idx(i) : p * math.exp(response["rootInfo"]["utility"] / utility_limit) for i, p in enumerate(human_policy) if p > 0
        }

        for move_info in response["moveInfos"]:
            weight = move_info["humanPrior"] * math.exp(move_info["utility"] / utility_limit)
            moves[move_info["move"]] = weight

        return random.choices(list(moves.keys()), weights=list(moves.values()))[0]


    def query(self, moves, exclude_move=None):
        """
        Provides a high playout query, where specific first move may be excluded.
        """
        query = {
            "moves": self._include_player_in_move_sequence(moves),
            "initialStones": [],
            "rules": "chinese",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": PLAYOUTS,
        }
        if exclude_move:
            query["avoidMoves"] = [
                {
                    "player": _get_player_to_move_next(moves),
                    "moves": [exclude_move],
                    "untilDepth": 1
                }
            ]

        return self._katago_controller.send_query(query)

    def get_move_from_idx(self, idx):
        """
        Converts a KataGo policy index to a Go move string (e.g., "A1", "pass").
        Assumes a 9x9 board
        """
        board_x_size = 9
        board_y_size = 9

        if idx == board_x_size * board_y_size:
            return "pass"

        row = idx // board_x_size
        col = idx % board_x_size

        # Convert to Go board coordinates (A1 is bottom-left, I9 is top-right)
        # KataGo uses 0-indexed from top-left, so A1 is (0,0) in KataGo's internal representation
        # and I9 is (8,8). We need to convert to standard Go notation.
        # Column: A=0, B=1, ..., I=8
        # Row: 1=8, 2=7, ..., 9=0

        col_char = chr(ord('A') + col)
        if col >= 8:  # Skip 'I'
            col_char = chr(ord('A') + col + 1)
        row_char = str(board_y_size - row)

        return f"{col_char}{row_char}"

    def get_idx_from_move(self, move):
        """
        Converts a Go move string (e.g., "A1", "pass") to a KataGo policy index.
        Assumes a 9x9 board.
        """
        board_x_size = 9
        board_y_size = 9

        if move == "pass":
            return board_x_size * board_y_size

        col_char = move[0].upper()
        row_str = move[1:]

        # Convert column character to index
        col = ord(col_char) - ord('A')
        if col_char > 'H':  # Skip 'I'
            col -= 1

        # Convert row string to index
        row_num = int(row_str)
        row = board_y_size - row_num

        return row * board_x_size + col

    def get_position_info(self, moves):
        """
        Helper to get the current position score and winrate from Black's perspective.
        Returns (score, winrate) or (None, None) if failed.
        """
        response = self.query_policy(moves)

        score_lead = response["rootInfo"]["scoreLead"]
        winrate = response["rootInfo"]["winrate"]

        return score_lead, winrate

    def get_best_move(self, response):
        """
        Helper to get the the best policy move
        Returns move,winrate,score
        """

        if response.get("humanPolicy"):
            policy = response["humanPolicy"]
            score = response["rootInfo"]["scoreLead"]
            winrate = response["rootInfo"]["winrate"]
            return self.get_move_from_idx(policy.index(max(policy))), winrate, score
        elif response.get("moveInfos"):
            move_info = response["moveInfos"][0]
            score = move_info["scoreLead"]
            winrate = move_info["winrate"]
            return move_info["move"], winrate, score
        else:
            policy = response["policy"]
            score = response["rootInfo"]["scoreLead"]
            winrate = response["rootInfo"]["winrate"]
            return self.get_move_from_idx(policy.index(max(policy))), winrate, score


def _get_player_to_move_next(moves):
    """
    returns "B" or "W" corresponding to whos turn it is to move, assuming black started game and alternating turns.
    """
    if len(moves) % 2 == 0:
        return "B"
    else:
        return "W"


def katago_to_sgf_coords(katago_move, board_size):
    """
    Converts KataGo coordinates to SGF coordinates.

    Args:
        katago_move (str): The move in KataGo's coordinate format (e.g., 'A1', 'B2').
        board_size (int): The size of the board.

    Returns:
        tuple: The move in SGF coordinate format (row, col).
    """
    if katago_move == "pass":
        return None

    col_str = katago_move[0]
    row_str = katago_move[1:]

    col = ord(col_str.upper()) - ord('A')
    if col_str.upper() >= 'I':
        col -= 1
    row = board_size - int(row_str)

    return row, col


def sgf_to_katago_coords(sgf_move, board_size):
    """
    Converts SGF coordinates to KataGo coordinates.

    Args:
        sgf_move (tuple): The move in SGF coordinate format (row, col).
        board_size (int): The size of the board.

    Returns:
        str: The move in KataGo's coordinate format (e.g., 'A1', 'B2').
    """
    if sgf_move is None:
        return "pass"

    row, col = sgf_move
    col_str = chr(ord('A') + col)
    if col >= 8:
        col_str = chr(ord('A') + col + 1)
    row_str = str(board_size - row)
    return col_str + row_str


def _get_rank_str(rank):
    return _SUPPORTED_RANKS[max(0, min(int(rank * len(_SUPPORTED_RANKS)), len(_SUPPORTED_RANKS) - 1))]


def get_rank_str(rank, stddev=None):
    if stddev is None:
        return _get_rank_str(rank)[5:]
    else:
        return f"{_get_rank_str(rank)[5:]}±{stddev * len(_SUPPORTED_RANKS):.1f}"


def get_rank_number(rank_str):
    return _SUPPORTED_RANKS.index("rank_" + rank_str) / (len(_SUPPORTED_RANKS) - 1)
