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

    def _build_query(self, moves, **kwargs):
        """Helper to build standardized query object"""
        query = {
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 49,
            "rootPolicyTemperature": 49
        }
        query.update(kwargs)
        return query

    def validate_and_adjust_tsumego(self, moves, player_to_move, correct_move):
        """
        Validates and adjusts position to meet tsumego criteria.
        Returns (is_valid, final_score, adjusted_moves)
        """
        print(f"\n  Validating tsumego at move {len(moves)+1}...")
        MAX_ATTEMPTS = 5
        adjusted_moves = moves.copy()
        
        # Get and print initial score before adjustments
        initial_score = self._get_score_after_move(adjusted_moves, player_to_move, correct_move)
        if initial_score is None:
            print("    Initial score: None")
        else:
            print(f"    Initial score: {initial_score:.1f}")
        
        # Phase 1: Adjust score to 0-1 range
        print(f"  Phase 1: Adjusting score to 0-1 range...")
        for attempt in range(MAX_ATTEMPTS):
            print(f"    Score adjustment attempt {attempt+1}/{MAX_ATTEMPTS}")
            adjusted_moves = self._adjust_to_0_5(adjusted_moves, player_to_move, correct_move)
            score = self._get_score_after_move(adjusted_moves, player_to_move, correct_move)
            if score is None:
                print("    Score is None, aborting adjustment")
                return False, None, None
            print(f"    Current score after solution: {score:.1f}")
            if 0 <= score <= 1:
                print("    ✅ Score within target range")
                break
            else:
                print("    ❌ Score still outside target range")
        else:
            print("    Failed to adjust score after max attempts")
            return False, None, None
        
        # Phase 2: Ensure only one winning move
        print(f"  Phase 2: Ensuring single winning move...")
        for attempt in range(MAX_ATTEMPTS):
            print(f"    Winning move adjustment attempt {attempt+1}/{MAX_ATTEMPTS}")
            query = self._build_query(
                adjusted_moves,
                id="tsumego_check",
                analysisPVLen=10
            )
            response = self.strong_controller.send_query(query)
            if not response or "rootInfo" not in response or "moveInfos" not in response:
                print("    Error in tsumego check query")
                return False, None, None

            # Find all moves with winrate >= 0.5
            winning_moves = []
            for move_info in response["moveInfos"]:
                if move_info["winrate"] >= 0.5:
                    winning_moves.append(move_info["move"])
            
            print(f"    Winning moves: {winning_moves} (correct: {correct_move})")
            
            # If only solution move wins, we're done
            if winning_moves == [correct_move]:
                print("    ✅ Only solution move wins")
                return True, score, adjusted_moves
            
            # If multiple winning moves, make others lose
            print("    Multiple winning moves - making alternatives lose...")
            adjusted_moves = self._make_alternatives_lose(adjusted_moves, player_to_move, correct_move, winning_moves)
        
        print("    Failed to achieve single winning move after max attempts")
        return False, None, None

    def _get_quick_position_score(self, moves):
        """Helper to get the current position score with maxVisits=1 for stability"""
        query = {
            "id": "position_score_quick",
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 1   # Only one visit for stability
        }
        response = self.strong_controller.send_query(query)
        if response and "rootInfo" in response:
            return response["rootInfo"]["scoreLead"]
        return None

    def _get_current_score_black(self, moves):
        """
        Helper to get the current position score from Black's perspective.
        Returns None if failed.
        """
        score_lead = self._get_quick_position_score(moves)
        if score_lead is None:
            return None

        # Determine current player to move
        if not moves:
            current_player = "B"
        else:
            last_player = moves[-1][0]
            current_player = "W" if last_player == "B" else "B"

        # Convert to Black's perspective
        if current_player == "B":
            return score_lead
        else:
            return -score_lead

    def _adjust_to_0_5(self, moves, player, correct_move):
        """
        Adjusts position to ensure solution leads to ~0.5 win.
        Always picks the move that brings score closest to 0.5.
        
        Args:
            moves (list): Current move sequence
            player (str): Player to move ("B" or "W")
            correct_move (str): Solution move
            
        Returns:
            list: Adjusted move sequence
        """
        current_moves = moves.copy()
        MAX_ADJUSTMENTS = 5
        
        for adjustment_num in range(1, MAX_ADJUSTMENTS + 1):
            # First, check current score and break if in range
            current_score = self._get_current_score_black(current_moves)
            if current_score is None:
                print("    Failed to get current score")
                break
            if 0 <= current_score <= 1:
                print(f"    Current score {current_score:.1f} already in [0,1] - no adjustment needed")
                break

            print(f"  Adjustment round {adjustment_num}:")
            # Query for moves in current position (player's turn)
            query = self._build_query(
                current_moves,
                id=f"adjust_{adjustment_num}",
                analysisPVLen=5
            )
            response = self.strong_controller.send_query(query)
            if not response or "moveInfos" not in response:
                print("    No response or no moveInfos in response")
                return current_moves
                
            best_candidate = None
            best_candidate_score = None
            best_candidate_diff = float('inf')
            
            # Find best candidate move for player
            for move_info in response["moveInfos"]:
                candidate_move = move_info["move"]
                if candidate_move == correct_move:
                    continue
                if "scoreLead" not in move_info:
                    continue
                    
                # Convert score to Black's perspective
                candidate_score = move_info["scoreLead"] if player == "B" else -move_info["scoreLead"]
                score_diff = abs(candidate_score - 0.5)
                
                if score_diff < best_candidate_diff:
                    best_candidate_diff = score_diff
                    best_candidate = candidate_move
                    best_candidate_score = candidate_score
            
            if not best_candidate:
                print("    No suitable candidate move found")
                return current_moves
                
            print(f"    Selected {player} move: {best_candidate} (expected score: {best_candidate_score:.1f})", flush=True)
            # Apply player's candidate move
            current_moves = current_moves + [[player, best_candidate]]
            opponent = "W" if player == "B" else "B"
            
            # Query for opponent's moves
            opponent_query = self._build_query(
                current_moves,
                id=f"adjust_{adjustment_num}_opponent",
                analysisPVLen=5
            )
            opponent_response = self.strong_controller.send_query(opponent_query)
            if not opponent_response or "moveInfos" not in opponent_response:
                print("    No response or no moveInfos for opponent")
                return current_moves
                
            best_opponent = None
            best_opponent_score = None
            best_opponent_diff = float('inf')
            
            # Find best opponent move
            for move_info in opponent_response["moveInfos"]:
                opponent_move = move_info["move"]
                if "scoreLead" not in move_info:
                    continue
                    
                # Convert score to Black's perspective
                opponent_score = move_info["scoreLead"] if opponent == "B" else -move_info["scoreLead"]
                score_diff = abs(opponent_score - 0.5)
                
                if score_diff < best_opponent_diff:
                    best_opponent_diff = score_diff
                    best_opponent = opponent_move
                    best_opponent_score = opponent_score
            
            if not best_opponent:
                print("    No suitable opponent move found")
                return current_moves
                
            print(f"    Selected {opponent} move: {best_opponent} (expected score: {best_opponent_score:.1f})", flush=True)
            # Apply opponent's move
            current_moves = current_moves + [[opponent, best_opponent]]
            print(f"  Applied adjustment: {player} {best_candidate} -> {opponent} {best_opponent}")
                
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
        query = self._build_query(
            moves + [[player, move]],
            id="get_score"
        )
        response = self.strong_controller.send_query(query)
        if not response or "rootInfo" not in response:
            return None
        score_lead = response["rootInfo"]["scoreLead"]
        # Always return score from Black's perspective
        return -score_lead

    def _get_best_response(self, moves):
        """Helper to get best response to a move"""
        query = self._build_query(
            moves,
            id="best_response"
        )
        response = self.strong_controller.send_query(query)
        if not response or "moveInfos" not in response:
            return "pass"
        return response["moveInfos"][0]["move"]
