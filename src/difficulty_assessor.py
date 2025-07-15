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
        Checks if position is valid tsumego and returns (is_valid, final_score)
        """
        """
        Checks if a position is a valid tsumego and adjusts it to ensure:
        1. Only one correct first move (others lose)
        2. Correct move leads to score in 0-1 range (rounds to 0.5)
        
        Args:
            moves (list): The list of moves in the game.
            player_to_move (str): The player to move ("B" or "W").
            correct_move (str): The correct move.

        Returns:
            bool: True if position meets criteria after adjustment, False otherwise
        """
        print(f"\n  Verifying tsumego criteria for {player_to_move} at move {len(moves)+1}...")
        initial_score = self._get_score_after_move(moves, player_to_move, correct_move)
        print(f"  Initial score after solution: {initial_score:.1f}")
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
                    final_score = self._get_score_after_move(adjusted_moves, player_to_move, correct_move)
                    return True, final_score
                else:
                    # Adjust position to reach 0.5 point win
                    adjusted_moves = self._adjust_to_0_5(adjusted_moves, player_to_move, correct_move)
                    continue
            
            # If multiple winning moves, make others lose
            adjusted_moves = self._make_alternatives_lose(adjusted_moves, player_to_move, correct_move, winning_moves)
        
        return False, None  # Failed after max attempts

    def _adjust_to_0_5(self, moves, player, correct_move):
        """
        Adjusts position by having player play suboptimal moves until solution leads to 0.5 win.
        Tries multiple rounds of adjustments if needed.
        
        Args:
            moves (list): Current move sequence
            player (str): Player to move ("B" or "W")
            correct_move (str): Solution move
            
        Returns:
            list: Adjusted move sequence
        """
        current_moves = moves.copy()
        initial_score = self._get_score_after_move(current_moves, player, correct_move)
        MAX_ADJUSTMENTS = 5  # Increased from 3 to allow more refinement
        TARGET_RANGE = (0.0, 1.0)  # Target range that will round to 0.5
        
        print(f"\n  Adjusting position to reach 0.5 point win...")
        
        for adjustment_num in range(1, MAX_ADJUSTMENTS + 1):
            # Get current score after solution
            score = self._get_score_after_move(current_moves, player, correct_move)
            if score is None:
                return current_moves
                
            print(f"  Adjustment {adjustment_num}: Current score after solution: {score:.1f}")
            
            # Check if score is in target range (0-1 for Black)
            if (player == "B" and TARGET_RANGE[0] <= score <= TARGET_RANGE[1]) or \
               (player == "W" and -TARGET_RANGE[1] <= score <= -TARGET_RANGE[0]):
                print(f"  ✅ Reached target score range (will round to 0.5)")
                return current_moves
            
            # Find adjustment moves
            query = {
                "id": f"adjust_{adjustment_num}",
                "moves": current_moves,
                "initialStones": [],
                "rules": "tromp-taylor",
                "komi": 7.5,
                "boardXSize": 9,
                "boardYSize": 9,
                "maxVisits": 1000,
                "analysisPVLen": 5
            }
            response = self.strong_controller.send_query(query)
            if not response or "moveInfos" not in response:
                return current_moves
                
            # Try moves that move score toward target
            best_adjustment = None
            best_score_diff = float('inf')
            
            for move_info in response["moveInfos"][:5]:  # Try top 5 alternatives
                if move_info["move"] == correct_move:
                    continue
                    
                # Have player play this move
                test_moves = current_moves + [[player, move_info["move"]]]
                
                # Have opponent respond (optimally if we're helping Black, suboptimally if helping White)
                if player == "B":
                    # For Black problems, opponent responds optimally
                    opponent_move = self._get_best_response(test_moves)
                else:
                    # For White problems, opponent responds suboptimally to help Black
                    opponent_move = self._get_worst_response(test_moves)
                
                test_moves.append(["W" if player == "B" else "B", opponent_move])
                
                # Check resulting score
                test_score = self._get_score_after_move(test_moves, player, correct_move)
                if test_score is None:
                    continue
                    
                # Calculate how much closer this gets us to target
                if player == "B":
                    target = 0.5
                    score_diff = abs(test_score - target)
                else:
                    target = -0.5
                    score_diff = abs(test_score - target)
                
                if score_diff < best_score_diff:
                    best_score_diff = score_diff
                    best_adjustment = test_moves
                    print(f"    Found adjustment: {move_info['move']} -> {opponent_move} "
                          f"(new score: {test_score:.1f})")
            
            if best_adjustment:
                current_moves = best_adjustment
            else:
                break
                
        final_score = self._get_score_after_move(current_moves, player, correct_move)
        print(f"\n  Score adjustment results:")
        print(f"  - Initial score: {initial_score:.1f}")
        print(f"  - Final score: {final_score:.1f}")
        print(f"  - Target range: {'0.0-1.0' if player == 'B' else '-1.0-0.0'}")
        if (player == "B" and 0 <= final_score <= 1) or (player == "W" and -1 <= final_score <= 0):
            print("  ✅ Score within target range (will round to 0.5)")
        return current_moves

    def _get_worst_response(self, moves):
        """Helper to get worst reasonable response to a move (for helping Black)"""
        query = {
            "id": "worst_response",
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor", 
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 100
        }
        response = self.strong_controller.send_query(query)
        if not response or "moveInfos" not in response or len(response["moveInfos"]) < 2:
            return "pass"
        # Return the worst move among the top 5 candidates
        return response["moveInfos"][-1]["move"]

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
