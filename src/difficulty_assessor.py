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

        # Get the policy from the response
        policy = sl_response["humanPolicy"] if "humanPolicy" in sl_response else sl_response["policy"]

        # Get the top N moves based on policy probabilities
        top_n = 3 # Check top 3 moves
        # Create a list of (probability, index) tuples
        indexed_policy = []
        for i, prob in enumerate(policy):
            indexed_policy.append((prob, i))

        # Sort by probability in descending order
        indexed_policy.sort(key=lambda x: x[0], reverse=True)

        # Get the top N move indices
        top_move_indices = [idx for prob, idx in indexed_policy[:top_n]]

        # Convert indices to Go coordinates
        proposed_moves = [self.human_controller.get_move_from_idx(idx) for idx in top_move_indices]

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

    def is_valid_tsumego(self, moves, player_to_move, correct_move):
        """
        Checks if a position is a valid tsumego according to the new criteria.

        Args:
            moves (list): The list of moves in the game.
            player_to_move (str): The player to move ("B" or "W").
            correct_move (str): The correct move.

        Returns:
            bool: True if the position is a valid tsumego, False otherwise.
        """
        # 1. Analyze the current position with the strong RL network
        query = {
            "id": "tsumego_check",
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 1000, # High visits for accurate analysis
            "analysisPVLen": 10
        }
        response = self.strong_controller.send_query(query)
        if not response or "rootInfo" not in response or "moveInfos" not in response:
            return False

        # 2. Check the score condition
        score = response["rootInfo"]["scoreLead"]
        if player_to_move == "W":
            score = -score
        if not (-5 <= score <= 5):
            return False

        # 3. Verify the winrate conditions
        initial_winrate = response["rootInfo"]["winrate"]
        if initial_winrate <= 0.1:
            return False

        # Check the correct move
        correct_move_info = None
        for move_info in response["moveInfos"]:
            if move_info["move"] == correct_move:
                correct_move_info = move_info
                break
        
        if correct_move_info is None:
            return False # Should not happen if correct_move is from RL self-play

        # Check winrate after the correct move
        # This requires a new query, as the analysis is from the current position
        winrate_after_correct_move_query = query.copy()
        winrate_after_correct_move_query["id"] = "winrate_after_correct"
        winrate_after_correct_move_query["moves"] = moves + [[player_to_move, correct_move]]
        winrate_after_correct_move_response = self.strong_controller.send_query(winrate_after_correct_move_query)
        if not winrate_after_correct_move_response or "rootInfo" not in winrate_after_correct_move_response:
            return False
        
        winrate_after_correct_move = winrate_after_correct_move_response["rootInfo"]["winrate"]
        if winrate_after_correct_move < 0.5:
            return False

        # 4. Check that all other moves lead to a winrate below 0.5
        # We only need to check the top few alternatives
        for move_info in response["moveInfos"][:5]: # Check top 5 alternatives
            if move_info["move"] != correct_move:
                # Query for winrate after this alternative move
                winrate_after_alternative_query = query.copy()
                winrate_after_alternative_query["id"] = "winrate_after_alternative"
                winrate_after_alternative_query["moves"] = moves + [[player_to_move, move_info["move"]]]
                winrate_after_alternative_response = self.strong_controller.send_query(winrate_after_alternative_query)
                if not winrate_after_alternative_response or "rootInfo" not in winrate_after_alternative_response:
                    continue # Or handle error appropriately

                winrate_after_alternative = winrate_after_alternative_response["rootInfo"]["winrate"]
                if winrate_after_alternative >= 0.5:
                    return False # Found an alternative move that doesn't lose

        return True
