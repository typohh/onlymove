from utils.configurations import RULES, KOMI
# GTP_COLS re-exported here for back-compat (utils.problem_finder and others
# import it from this module); the coordinate primitives live in utils.coords.
from utils.coords import GTP_COLS, col_index_to_gtp_letter, gtp_letter_to_col_index

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

    def __init__(self, katago_controller, board_size):
        """
        Initializes the KatagoController.

        Args:
            katago_controller (KatagoController): The controller wrapping the engine.
            board_size (int): The board size all queries/coordinate math use.
        """
        self._katago_controller = katago_controller
        self.board_size = board_size

    def get_legal_moves(self, moves):
        """
        Gets all legal moves for the current position.
        """
        response = self.query(moves)
        legal_moves = []
        for i, p in enumerate(response["policy"]):
            if p > 0:
                legal_moves.append(self.get_move_from_idx(i))
        return legal_moves

    @staticmethod
    def _include_player_in_move_sequence(moves):
        player_and_move = []
        for move in moves:
            player_and_move.append([_get_player_to_move_next(player_and_move), move])
        return player_and_move

    def query_children(self, moves, playouts=1):
        """
        Gets information about all child nodes by querying each legal move individually.
        """
        legal_moves = self.get_legal_moves(moves)

        move_infos = []

        for move in legal_moves:
            # For each legal move, query the position after that move is made
            response = self.query(moves + [move], playouts=playouts)

            move_info = response.get("rootInfo")
            move_info["move"] = move
            move_infos.append(move_info)

        # Let's get the root info for the original position
        root_response = self.query(moves, playouts=playouts)

        # The response should be sorted by something, probably visits or winrate.
        # query_children seems to return it sorted by visits.
        # Here, all visits are the same (playouts). Let's sort by winrate descending.
        move_infos = sorted(move_infos, key=lambda x: x.get("winrate", 0.0), reverse=True)
        root_response["moveInfos"] = move_infos
        return root_response


    @staticmethod
    def adjust_temperature(policy, temperature):
        heated = [(p ** (1 / temperature) if p > 0 else p) for p in policy]
        sum_heated = sum([max(0, p) for p in heated])
        normalized = [p / sum_heated if p > 0 else 0 for p in heated]
        return normalized

    def query(self, moves, human_rank=None, exclude_move=None, playouts=1, override_settings=None):
        """
        Provides a high playout query, where specific first move may be excluded.
        """
        query = {
            "moves": self._include_player_in_move_sequence(moves),
            "initialStones": [],
            "rules": RULES,
            "komi": KOMI,
            "boardXSize": self.board_size,
            "boardYSize": self.board_size,
            "includePolicy": playouts == 1,
            "maxVisits": playouts,
        }
        if exclude_move:
            query["avoidMoves"] = [
                {
                    "player": _get_player_to_move_next(moves),
                    "moves": [exclude_move],
                    "untilDepth": 1
                }
            ]
        settings = {}
        if human_rank is not None:
            settings.update({
                "humanSLProfile": _get_rank_str(human_rank),
                "ignorePreRootHistory": False,
                "rootNumSymmetriesToSample": 1
            })
        if override_settings:
            settings.update(override_settings)
        if settings:
            query["overrideSettings"] = settings

        return self._katago_controller.send_query(query)

    def get_move_from_idx(self, idx):
        """
        Converts a KataGo policy index to a Go move string (e.g., "A1", "pass").
        Board size is taken from self.board_size.
        """

        if idx == self.board_size * self.board_size:
            return "pass"

        row = idx // self.board_size
        col = idx % self.board_size

        # Convert to Go board coordinates (A1 is bottom-left, I9 is top-right)
        # KataGo uses 0-indexed from top-left, so A1 is (0,0) in KataGo's internal representation
        # and I9 is (8,8). We need to convert to standard Go notation.
        # Column: A=0, B=1, ..., I=8
        # Row: 1=8, 2=7, ..., 9=0

        col_char = col_index_to_gtp_letter(col)
        row_char = str(self.board_size - row)

        return f"{col_char}{row_char}"

    def get_idx_from_move(self, move):
        """
        Converts a Go move string (e.g., "A1", "pass") to a KataGo policy index.
        Board size is taken from self.board_size.
        """

        if move == "pass":
            return self.board_size * self.board_size

        col_char = move[0].upper()
        row_str = move[1:]

        # Convert column character to index (I-skipped)
        col = gtp_letter_to_col_index(col_char)

        # Convert row string to index
        row_num = int(row_str)
        row = self.board_size - row_num

        return row * self.board_size + col

    def get_position_info(self, moves):
        """
        Helper to get the current position score and winrate from Black's perspective.
        Returns (score, winrate) or (None, None) if failed.
        """
        response = self.query(moves)

        score_lead = response["rootInfo"]["scoreLead"]
        winrate = response["rootInfo"]["winrate"]

        return score_lead, winrate

    def get_move_probability(self, policy, move):
        return policy[self.get_idx_from_move(move)]

    def get_best_move(self, response):
        """
        Helper to get the the best policy move
        Returns move,winrate,score
        """

        if response.get("humanPolicy"):
            policy = response["humanPolicy"]
            score = response["rootInfo"]["scoreLead"]
            winrate = response["rootInfo"]["winrate"]
            max_policy = max(policy)
            return self.get_move_from_idx(policy.index(max_policy)), winrate, score, max_policy
        elif response.get("moveInfos"):
            move_info = response["moveInfos"][0]
            score = move_info["scoreLead"]
            winrate = move_info["winrate"]
            policy_prior = move_info["prior"]
            return move_info["move"], winrate, score, policy_prior
        else:
            policy = response["policy"]
            score = response["rootInfo"]["scoreLead"]
            winrate = response["rootInfo"]["winrate"]
            max_policy = max(policy)
            return self.get_move_from_idx(policy.index(max_policy)), winrate, score, max_policy


def _get_player_to_move_next(moves):
    """
    returns "B" or "W" corresponding to whos turn it is to move, assuming black started game and alternating turns.
    """
    if len(moves) % 2 == 0:
        return "B"
    else:
        return "W"


def _get_rank_str(rank):
    return _SUPPORTED_RANKS[max(0, min(int(rank * len(_SUPPORTED_RANKS)), len(_SUPPORTED_RANKS) - 1))]


def get_rank_str(rank, stddev=None):
    if stddev is None:
        return _get_rank_str(rank)[5:]
    else:
        return f"{_get_rank_str(rank)[5:]}±{stddev * len(_SUPPORTED_RANKS):.1f}"


def get_rank_number(rank_str):
    return (_SUPPORTED_RANKS.index("rank_" + rank_str) + 0.5) / len(_SUPPORTED_RANKS)


