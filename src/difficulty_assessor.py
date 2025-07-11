from src.katago_controller import KatagoController

class DifficultyAssessor:
    """
    Assesses the difficulty of a move.
    """
    def __init__(self, human_controller: KatagoController, strong_controller: KatagoController):
        """
        Initializes the DifficultyAssessor.

        Args:
            human_controller (KatagoController): The KataGo controller for the SL network.
            strong_controller (KatagoController): The KataGo controller for the RL network.
        """
        self.human_controller = human_controller
        self.strong_controller = strong_controller
        self._SUPPORTED_RANKS = [
            "rank_20k", "rank_19k", "rank_18k", "rank_17k", "rank_16k",
            "rank_15k", "rank_14k", "rank_13k", "rank_12k", "rank_11k",
            "rank_10k", "rank_9k", "rank_8k", "rank_7k", "rank_6k",
            "rank_5k", "rank_4k", "rank_3k", "rank_2k", "rank_1k",
            "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d",
            "rank_6d", "rank_7d", "rank_8d", "rank_9d"
        ]

    def assess_move_difficulty(self, moves, player_to_move, correct_move):
        """
        Finds the precise rank that is the threshold for solving the problem.

        Args:
            moves (list): The list of moves in the game.
            player_to_move (str): The player to move ("B" or "W").
            correct_move (str): The correct move.

        Returns:
            str: The best found rank.
        """
        low = 0
        high = len(self._SUPPORTED_RANKS) - 1
        best_found_rank = None

        while low <= high:
            mid = (low + high) // 2
            rank_string = self._SUPPORTED_RANKS[mid]

            if self._check_rank_solves(rank_string, moves, player_to_move, correct_move):
                best_found_rank = rank_string
                high = mid - 1  # Try to find an even weaker rank
            else:
                low = mid + 1  # Need a stronger rank

        return best_found_rank

    def _check_rank_solves(self, rank_string, moves, player, correct_move):
        """
        Checks if a single, specific rank can solve the problem.

        Args:
            rank_string (str): The rank to check.
            moves (list): The list of moves in the game.
            player (str): The player to move.
            correct_move (str): The correct move.

        Returns:
            bool: True if the rank solves the problem, False otherwise.
        """
        query = {
            "id": "check_rank",
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5, # Assuming standard komi
            "boardXSize": 9, # Assuming 9x9 board
            "boardYSize": 9,
            "maxVisits": 1, # Pure intuition
            "includePolicy": True,
            "overrideSettings": {
                "humanSLProfile": rank_string
            }
        }

        sl_response = self.human_controller.send_query(query)
        if not sl_response or "policy" not in sl_response:
            return False

        # Check if the correct move is in the top few proposals
        proposed_moves = [move_info[0] for move_info in sl_response["policy"][:3]]
        if correct_move not in proposed_moves:
            return False

        # Verify with the strong network that the move is not a blunder
        strong_query = {
            "id": "verify_move",
            "moves": moves + [[player, correct_move]],
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 100 # Decent search to verify
        }
        rl_response = self.strong_controller.send_query(strong_query)
        if not rl_response or "rootInfo" not in rl_response:
            return False

        # A simple check: if the winrate doesn't drop too much, it's a good move.
        # This can be made more sophisticated.
        initial_winrate_query = strong_query.copy()
        initial_winrate_query["id"] = "initial_winrate"
        initial_winrate_query["moves"] = moves
        initial_winrate_response = self.strong_controller.send_query(initial_winrate_query)
        if not initial_winrate_response or "rootInfo" not in initial_winrate_response:
            return False

        initial_winrate = initial_winrate_response["rootInfo"]["winrate"]
        new_winrate = rl_response["rootInfo"]["winrate"]

        # Allow a small drop in winrate
        return new_winrate >= (initial_winrate - 0.1)
