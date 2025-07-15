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
        Checks if a position is a valid tsumego and adjusts it to ensure:
        1. Only one correct first move (others lose)
        2. Correct move leads to exactly 0.5 point win
        
        Args:
            moves (list): The list of moves in the game.
            player_to_move (str): The player to move ("B" or "W").
            correct_move (str): The correct move.

        Returns:
            bool: True if position meets criteria after adjustment, False otherwise
        """
        MAX_ATTEMPTS = 5  # Limit adjustment attempts
        adjusted_moves = moves.copy()
        
        for attempt in range(MAX_ATTEMPTS):
            # 1. Check if we have exactly one winning move (the solution)
            query = {
                "id": "tsumego_check",
                "moves": adjusted_moves,
                "initialStones": [],
                "rules": "tromp-taylor",
                "komi": 7.5,
                "boardXSize": 9,
                "boardYSize": 9,
                "maxVisits": 1000,
                "analysisPVLen": 10
            }
            response = self.strong_controller.send_query(query)
            if not response or "rootInfo" not in response or "moveInfos" not in response:
                return False

            # Find all moves with winrate >= 0.5
            winning_moves = []
            for move_info in response["moveInfos"][:5]:  # Check top 5 alternatives
                if move_info["winrate"] >= 0.5:
                    winning_moves.append(move_info["move"])
            
            # If only solution move wins, check 0.5 point condition
            if winning_moves == [correct_move]:
                if self._verify_0_5_win(adjusted_moves, player_to_move, correct_move):
                    return True
                else:
                    # Adjust position to reach 0.5 point win
                    adjusted_moves = self._adjust_to_0_5(adjusted_moves, player_to_move, correct_move)
                    continue
            
            # If multiple winning moves, make others lose
            adjusted_moves = self._make_alternatives_lose(adjusted_moves, player_to_move, correct_move, winning_moves)
        
        return False  # Failed after max attempts

    def _adjust_to_0_5(self, moves, player, correct_move):
        """
        Adjusts position by having player play suboptimal moves until solution leads to 0.5 win.
        
        Args:
            moves (list): Current move sequence
            player (str): Player to move
            correct_move (str): Solution move
            
        Returns:
            list: Adjusted move sequence
        """
        current_moves = moves.copy()
        MAX_ADJUSTMENTS = 3
        
        for _ in range(MAX_ADJUSTMENTS):
            # Get current score after solution
            score = self._get_score_after_move(current_moves, player, correct_move)
            if abs(score - 0.5) < 0.1:
                return current_moves
            
            # Find a move that reduces the score toward 0.5
            query = {
                "id": "find_adjustment",
                "moves": current_moves,
                "initialStones": [],
                "rules": "tromp-taylor",
                "komi": 7.5,
                "boardXSize": 9,
                "boardYSize": 9,
                "maxVisits": 1000
            }
            response = self.strong_controller.send_query(query)
            if not response or "moveInfos" not in response:
                return current_moves
                
            # Try moves that reduce the score
            for move_info in response["moveInfos"]:
                if move_info["move"] != correct_move:
                    test_score = self._get_score_after_move(
                        current_moves + [[player, move_info["move"]]],
                        "W" if player == "B" else "B",  # Opponent responds
                        "pass"  # Assume opponent passes after adjustment
                    )
                    if abs(test_score - 0.5) < abs(score - 0.5):
                        # This move improves score toward 0.5
                        return current_moves + [
                            [player, move_info["move"]],
                            ["W" if player == "B" else "B", "pass"]
                        ]
        
        return current_moves

    def _make_alternatives_lose(self, moves, player, correct_move, winning_moves):
        """
        Adjusts position by having player play alternatives until only solution wins.
        
        Args:
            moves (list): Current move sequence
            player (str): Player to move
            correct_move (str): Solution move
            winning_moves (list): Current winning alternative moves
            
        Returns:
            list: Adjusted move sequence
        """
        current_moves = moves.copy()
        
        for alt_move in winning_moves:
            if alt_move == correct_move:
                continue
                
            # Have player play this alternative, then opponent responds optimally
            current_moves = current_moves + [
                [player, alt_move],
                ["W" if player == "B" else "B", self._get_best_response(current_moves + [[player, alt_move]])]
            ]
        
        return current_moves

    def _get_score_after_move(self, moves, player, move):
        """Helper to get score after playing a move"""
        query = {
            "id": "get_score",
            "moves": moves + [[player, move]],
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 1000
        }
        response = self.strong_controller.send_query(query)
        if not response or "rootInfo" not in response:
            return None
        score = response["rootInfo"]["scoreLead"]
        return -score if player == "W" else score

    def _get_best_response(self, moves):
        """Helper to get best response to a move"""
        query = {
            "id": "best_response",
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 100
        }
        response = self.strong_controller.send_query(query)
        if not response or "moveInfos" not in response:
            return "pass"
        return response["moveInfos"][0]["move"]
