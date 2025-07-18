import numpy as np
from scipy.stats import norm
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
        Assesses difficulty by sampling all ranks and fitting a normal distribution.

        Args:
            moves (list): The list of moves in the game.
            player_to_move (str): The player to move ("B" or "W").
            correct_move (str): The correct move.

        Returns:
            str: String representation of mean and standard deviation.
        """
        probabilities = []
        for rank in self._SUPPORTED_RANKS:
            prob = self._get_probability_for_rank(rank, moves, correct_move)
            probabilities.append(prob)
        
        # Convert probabilities to z-scores
        clamped_probs = np.clip(probabilities, 0.01, 0.99)
        z_scores = norm.ppf(clamped_probs)
        
        # Create numerical rank values (0 to 28)
        numerical_ranks = np.arange(len(self._SUPPORTED_RANKS))
        
        # Solve linear regression: rank = mu + sigma * z_score
        A = np.vstack([np.ones(len(z_scores)), z_scores]).T
        mu, sigma = np.linalg.lstsq(A, numerical_ranks, rcond=None)[0]
        
        # Ensure positive standard deviation
        sigma = abs(sigma)
        
        # Convert numerical mean to rank string
        rank_idx = int(round(mu))
        if rank_idx < 0: rank_idx = 0
        if rank_idx >= len(self._SUPPORTED_RANKS): rank_idx = len(self._SUPPORTED_RANKS)-1
        rank_str = self._SUPPORTED_RANKS[rank_idx].replace("rank_", "").replace("_", " ").upper()
        
        # Debug: show probabilities
        print("    Probabilities by rank:")
        for i, rank in enumerate(self._SUPPORTED_RANKS):
            r = rank.replace("rank_", "").replace("_", " ").upper()
            print(f"      {r}: {probabilities[i]:.3f}")
        
        return f"mean={mu:.2f} ({rank_str}), sd={sigma:.2f}"

    def _get_probability_for_rank(self, rank_string, moves, correct_move):
        """
        Gets the probability of correct move for a given rank.

        Args:
            rank_string (str): The rank to check.
            moves (list): The list of moves in the game.
            correct_move (str): The correct move.

        Returns:
            float: Probability of the correct move.

        Raises:
            Exception: If any step fails unexpectedly.
        """
        response = self.human_controller.query_root(
            moves,
            id="get_prob",
            humanRank=rank_string
        )
        if not response:
            raise Exception(f"No response for rank: {rank_string}")
            
        # Use humanPolicy if available, else policy
        policy = response.get("humanPolicy", response.get("policy", []))
        if not policy:
            raise Exception(f"Empty policy in response for rank: {rank_string}")
            
        # Convert move to index
        idx = self.human_controller.get_idx_from_move(correct_move)
        if idx is None:
            raise Exception(f"Invalid move: {correct_move}")
        if idx >= len(policy):
            raise Exception(f"Move index {idx} out of bounds for policy length {len(policy)}")
            
        return policy[idx]


    def _print_position_info(self, moves, label):
        """
        Helper to get and print position info (score, winrate, num_moves_above_50).
        """
        score, winrate = self._get_position_info(moves)
        num_above = self._get_num_moves_above_50(moves)
        print(f"    {label}: {score:.1f}, {winrate:.3f}, {num_above}")

    def validate_and_adjust_tsumego(self, moves, player_to_move):
        """
        Validates and adjusts position to meet tsumego criteria.
        Returns (is_valid, final_score, adjusted_moves)
        """
        move_num = len(moves)+1
        print(f"\n  Validating tsumego at move {move_num}...")
        adjusted_moves = moves.copy()
        self._print_position_info(adjusted_moves, "Initial")

        # Phase: Adjusting Score
        print(f"    Adjusting Score")
        adjusted_moves, candidate_move, opponent_move, candidate_winrate, opponent_winrate, candidate_score, opponent_score, candidate_visits, opponent_visits = self._adjust_to_0_5(adjusted_moves, player_to_move)
        
        if candidate_move and opponent_move:
            # Compute opponent color
            opponent = "W" if player_to_move == "B" else "B"
            # Print candidate move with score/winrate/visits
            print(f"    {player_to_move} {candidate_move} (score: {candidate_score:.1f}, winrate: {candidate_winrate:.3f}, visits: {candidate_visits})")
            
            # Print opponent response with score/winrate/visits
            print(f"    {opponent} {opponent_move} (score: {opponent_score:.1f}, winrate: {opponent_winrate:.3f}, visits: {opponent_visits})")
        
        # Get status after adjustment
        self._print_position_info(adjusted_moves, "After adjustment")

        # Print "Removing choices" unconditionally as per user request
        print(f"    Removing choices")

        # Early termination if score not in target range
        post_score, post_winrate = self._get_position_info(adjusted_moves)
        if not (0 <= post_score <= 1):
            return False, None, None, None

        # After score adjustment, check winrate
        if post_winrate < 0.5:
            return False, None, None, None

        # Now that the position is adjusted, determine the solution move
        solution_move = self._get_best_move(adjusted_moves, player_to_move)
        if not solution_move:
            print("    ❌ Failed to determine solution move")
            return False, None, None, None

        print(f"    Solution move determined: {solution_move}")

        # Run 100-visit analysis to confirm solution move meets criteria
        print("    Running 100-visit analysis for solution move...")
        response = self.strong_controller.query_children_100_visits(
            adjusted_moves,
            id="confirm_solution"
        )
        if not response or "moveInfos" not in response:
            print("    ❌ Failed to get 100-visit response for solution move")
            return False, None, None, None

        solution_found = False
        for move_info in response["moveInfos"]:
            if move_info["move"] == solution_move:
                winrate = move_info["winrate"]
                score_lead = move_info["scoreLead"]
                print(f"      Solution move: winrate={winrate:.3f}, score_lead={score_lead:.1f}")
                if winrate > 0.5 and 0 <= score_lead <= 1:
                    solution_found = True
                break

        if not solution_found:
            print("    ❌ Solution move doesn't meet win criteria")
            return False, None, None, None

        # Run 100-visit analysis with solution move excluded
        print("    Running 100-visit analysis excluding solution move...")
        exclude_response = self.strong_controller.query_children_exclude_move(
            adjusted_moves,
            solution_move,
            player_to_move,
            id="exclude_solution"
        )
        if not exclude_response or "moveInfos" not in exclude_response:
            print("    ❌ Failed to get response when excluding solution move")
            return False, None, None, None

        # Check if any alternative move meets win criteria
        invalid_alternative = False
        for move_info in exclude_response["moveInfos"]:
            winrate = move_info["winrate"]
            score_lead = move_info["scoreLead"]
            if winrate > 0.5 and score_lead > 0:
                print(f"      Alternative move {move_info['move']} meets win criteria: winrate={winrate:.3f}, score_lead={score_lead:.1f}")
                invalid_alternative = True
                break

        if invalid_alternative:
            print("    ❌ Found alternative move that meets win criteria")
            return False, None, None, None

        MAX_ATTEMPTS = 5
        # Phase: Adjusting Moves
        for attempt in range(MAX_ATTEMPTS):
            response = self.strong_controller.query_children(
                adjusted_moves,
                id="tsumego_check"
            )
            if not response or "rootInfo" not in response or "moveInfos" not in response:
                raise Exception("Error in tsumego check query")

            # Find best winrate
            best_winrate = -1
            for move_info in response["moveInfos"]:
                if move_info["winrate"] > best_winrate:
                    best_winrate = move_info["winrate"]

            # Use a fixed relative threshold of 0.01 (1%) for all positions
            threshold = 0.01

            best_moves = []
            for move_info in response["moveInfos"]:
                if best_winrate - move_info["winrate"] <= threshold:
                    best_moves.append(move_info["move"])

            # If only solution move is best, we're done
            if best_moves == [solution_move]:
                final_score, final_winrate = self._get_position_info(adjusted_moves)
                if final_score is None or final_winrate is None:
                    raise Exception("Failed to get final position info")
                return True, final_score, adjusted_moves, solution_move

            # If multiple best moves, remove one alternative at a time
            # Find the first alternative move that is not the solution move
            alt_played = None
            response_played = None
            for alt_move in best_moves:
                if alt_move != solution_move:
                    adjusted_moves, alt_played, response_played = self._remove_alternative(adjusted_moves, player_to_move, solution_move, alt_move)
                    break
            
            # Get status after removal
            self._print_position_info(adjusted_moves, "After removal")

        print("    Failed to achieve single best move after max attempts")
        return False, None, None, None

    def _get_position_info(self, moves):
        """
        Helper to get the current position score and winrate from Black's perspective.
        Returns (score, winrate) or (None, None) if failed.
        Uses maxVisits=10 for a quick but reliable read.
        """
        response = self.strong_controller.query_root(
            moves,
            id="position_info"
        )
        if not response or "rootInfo" not in response:
            raise Exception("Failed to get position info from KataGo")

        score_lead = response["rootInfo"]["scoreLead"]
        winrate = response["rootInfo"]["winrate"]

        # Determine current player to move
        if not moves:
            current_player = "B"
        else:
            last_player = moves[-1][0]
            current_player = "W" if last_player == "B" else "B"

        return score_lead, winrate

    def _get_num_moves_above_50(self, moves):
        """
        Helper to get the number of moves with winrate above 50% for the current player.
        Returns None if failed.
        """
        response = self.strong_controller.query_children(
            moves,
            id="moves_above_50"
        )
        if not response:
            raise Exception("No response from KataGo for moves above 50%")
        if "moveInfos" not in response:
            # For debugging, print the moves and the response
            print(f"Debug: Moves: {moves}")
            print(f"Debug: Response: {response}")
            raise Exception("Response missing 'moveInfos' for moves above 50%")

        num_moves_above_50 = 0
        for move_info in response["moveInfos"]:
            if move_info["winrate"] > 0.5:
                num_moves_above_50 += 1

        return num_moves_above_50

    def _adjust_to_0_5(self, moves, player):
        """
        Adjusts position to ensure solution leads to ~0.5 win.
        Performs a single exchange (player move + opponent response) if needed.
        Returns original sequence if no adjustments were needed or possible.

        Args:
            moves (list): Current move sequence
            player (str): Player to move ("B" or "W")
            correct_move (str): Solution move

        Returns:
            list: Adjusted move sequence
        """
        # First, check if the score is already in [0,1] for Black
        score, _ = self._get_position_info(moves)
        if score is not None and 0 <= score <= 1:
            print("    Current score already in [0,1] - no adjustment needed")
            return moves, None, None

        # Query for moves in current position (player's turn)
        response = self.strong_controller.query_children(
            moves,
            id="adjust_to_0_5"
        )
        if not response or "moveInfos" not in response:
            print("    No response or no moveInfos in response")
            return moves, None, None


        best_candidate = None
        best_candidate_score = None
        best_candidate_winrate = None
        best_candidate_visits = None
        best_candidate_penalty = float('inf')

        # Find best candidate move for player
        for move_info in response["moveInfos"]:
            candidate_move = move_info["move"]
            if "scoreLead" not in move_info or "visits" not in move_info:
                continue

            candidate_score = move_info["scoreLead"]
            candidate_winrate = move_info["winrate"]
            penalty = (0.5 + abs(candidate_score - 0.5)) * (1 - candidate_winrate)

            if penalty < best_candidate_penalty:
                best_candidate_penalty = penalty
                best_candidate = candidate_move
                best_candidate_score = candidate_score
                best_candidate_winrate = candidate_winrate
                best_candidate_visits = move_info["visits"]

        if not best_candidate:
            # If no candidate found, use the first non-best move we can find
            for move_info in response["moveInfos"]:
                candidate_move = move_info["move"]
                if "scoreLead" not in move_info or "visits" not in move_info:
                    continue
                best_candidate = candidate_move
                best_candidate_score = move_info["scoreLead"] if player == "B" else -move_info["scoreLead"]
                best_candidate_winrate = move_info["winrate"]
                best_candidate_visits = move_info["visits"]
                break
            if not best_candidate:
                raise Exception("No suitable candidate move found")

        # Apply player's candidate move
        adjusted_moves = moves + [[player, best_candidate]]
        opponent = "W" if player == "B" else "B"

        # Query for opponent's moves
        opponent_response = self.strong_controller.query_children(
            adjusted_moves,
            id="adjust_to_0_5_opponent"
        )
        if not opponent_response or "moveInfos" not in opponent_response:
            raise Exception("No response or no moveInfos for opponent")


        best_opponent = None
        best_opponent_score = None
        best_opponent_winrate = None
        best_opponent_visits = None
        best_opponent_penalty = float('inf')

        # Find best opponent move
        for move_info in opponent_response["moveInfos"]:
            opponent_move = move_info["move"]
            if "scoreLead" not in move_info or "winrate" not in move_info or "visits" not in move_info:
                continue

            opponent_score = move_info["scoreLead"]
            opponent_winrate = move_info["winrate"]
            penalty = (0.5 + abs(opponent_score - 0.5)) * (1 - opponent_winrate)

            if penalty < best_opponent_penalty:
                best_opponent_penalty = penalty
                best_opponent = opponent_move
                best_opponent_score = opponent_score
                best_opponent_winrate = opponent_winrate
                best_opponent_visits = move_info["visits"]

        if not best_opponent:
            raise Exception("No suitable opponent move found")

        # Apply opponent's move
        adjusted_moves = adjusted_moves + [[opponent, best_opponent]]

        return adjusted_moves, best_candidate, best_opponent, best_candidate_winrate, best_opponent_winrate, best_candidate_score, best_opponent_score, best_candidate_visits, best_opponent_visits



    def _remove_alternative(self, moves, player, correct_move, alternative_move):
        """
        Adjusts position by having player play one alternative move and then opponent responds optimally.
        Returns original sequence if no adjustments were needed or possible.

        Args:
            moves (list): Current move sequence
            player (str): Player to move
            correct_move (str): Solution move
            alternative_move (str): One alternative move to remove

        Returns:
            list: Adjusted move sequence
        """
        # First, check if alternative is already removed
        # We'll let the caller decide if removal is needed

        # Get move info for alternative move
        alt_response = self.strong_controller.query_children(
            moves,
            id="alt_move_info"
        )
        if not alt_response or "moveInfos" not in alt_response:
            raise Exception("Failed to get move info for alternative move")
    
        alt_move_info = None
        for move_info in alt_response["moveInfos"]:
            if move_info["move"] == alternative_move:
                alt_move_info = move_info
                break
    
        if not alt_move_info:
            raise Exception(f"Alternative move {alternative_move} not found in moveInfos")

        # Apply player's alternative move
        adjusted_moves = moves + [[player, alternative_move]]
        opponent = "W" if player == "B" else "B"

        # Get best response and its info
        resp_response = self.strong_controller.query_children(
            adjusted_moves,
            id="resp_move_info"
        )
        if not resp_response or "moveInfos" not in resp_response or not resp_response["moveInfos"]:
            raise Exception("Failed to get response move info")
    
        resp_move_info = resp_response["moveInfos"][0]
        response_played = resp_move_info["move"]

        # Print alternative move with score/winrate/visits
        alt_winrate = alt_move_info["winrate"] if player == "B" else 1 - alt_move_info["winrate"]
        alt_score = alt_move_info["scoreLead"] if player == "B" else -alt_move_info["scoreLead"]
        alt_visits = alt_move_info["visits"]
        print(f"    {player} {alternative_move} (score: {alt_score:.1f}, winrate: {alt_winrate:.3f}, visits: {alt_visits})")
        
        # Print opponent response with score/winrate/visits
        resp_winrate = resp_move_info["winrate"] if opponent == "B" else 1 - resp_move_info["winrate"]
        resp_score = resp_move_info["scoreLead"] if opponent == "B" else -resp_move_info["scoreLead"]
        resp_visits = resp_move_info["visits"]
        print(f"    {opponent} {response_played} (score: {resp_score:.1f}, winrate: {resp_winrate:.3f}, visits: {resp_visits})")

        # Apply opponent's response
        adjusted_moves = adjusted_moves + [[opponent, response_played]]
        return adjusted_moves, alternative_move, response_played

    def _get_score_after_move(self, moves, player, move):
        """Helper to get score after playing a move"""
        query = self.strong_controller.create_root_query(
            moves + [[player, move]],
            id="get_score"
        )
        response = self.strong_controller.send_query(query)
        if not response or "rootInfo" not in response:
            raise Exception("Failed to get score after move")
        score_lead = response["rootInfo"]["scoreLead"]
        # Always return score from Black's perspective
        return -score_lead

    def _get_best_move(self, moves, player):
        """Helper to get best move for player in current position"""
        response = self.strong_controller.query_children(
            moves,
            id="get_best_move"
        )
        if not response or "moveInfos" not in response or not response["moveInfos"]:
            return None
        return response["moveInfos"][0]["move"]

    def _get_best_response(self, moves):
        """Helper to get best response to a move"""
        query = self.strong_controller.create_children_query(
            moves,
            id="best_response"
        )
        response = self.strong_controller.send_query(query)
        if not response or "moveInfos" not in response:
            raise Exception("Failed to get best response")
        return response["moveInfos"][0]["move"]
