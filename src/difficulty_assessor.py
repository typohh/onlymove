import numpy as np
from scipy.stats import norm
from src.katago_controller import KatagoController, other_player


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
        response = self.human_controller.query_root(moves, humanRank=rank_string)
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

    def _debug(self,move_info):
        for mi in move_info:
            candidate_move = mi['move']
            candidate_score = mi['scoreLead']
            candidate_winrate = mi['winrate']
            print(f" DEBUG:    {candidate_move} (score: {candidate_score:.1f}, winrate: {candidate_winrate:.3f})")

    def _penalty(self,score,winrate):
        return abs(score - 0.5) + 0.25 * (1 - winrate)

    def validate_and_adjust_tsumego(self, moves, player_to_move):
        """
        Validates and adjusts position to meet tsumego criteria.
        Returns (is_valid, final_score, adjusted_moves, solution_move)

        Args:
            moves (list): The list of moves in the game.
            player_to_move (str): The player to move ("B" or "W").

        Returns:
            tuple: (is_valid, final_score, adjusted_moves, solution_move)
        """
        move_num = len(moves)+1
        print(f"\n  Validating tsumego at move {move_num}...")
        adjusted_moves = moves.copy()
        self._print_position_info(adjusted_moves, "Initial")

        # First, check if the score is already in [0,1] for Black
        while True:
            score,winrate = self._get_position_info(adjusted_moves)
            if 0 < score < 1 and winrate > 0.5:
                break
            print(f"    Adjusting score because (score: {score:.1f}, winrate: {winrate:.3f})")
            adjusted_moves, candidate_move, candidate_winrate, candidate_score = self._adjust_to_0_5(adjusted_moves, player_to_move)
            print(f"    {player_to_move} {candidate_move} (score: {candidate_score:.1f}, winrate: {candidate_winrate:.3f})")
            if candidate_move == "pass":
                return False,None,None,None
            adjusted_moves, candidate_move, candidate_winrate, candidate_score = self._adjust_to_0_5(adjusted_moves, other_player(player_to_move))
            print(f"    {other_player(player_to_move)} {candidate_move} (score: {candidate_score:.1f}, winrate: {candidate_winrate:.3f})")
            if candidate_move == "pass":
                return False,None,None,None

        self._print_position_info(adjusted_moves, "After adjustment")

        while True:
            response = self.strong_controller.query_children_100_visits(adjusted_moves)
            # Count moves above 50% winrate
            winning_moves = [mi for mi in response["moveInfos"] if mi['winrate'] > 0.5]
            num_winning_moves = len(winning_moves)
            if num_winning_moves == 1:
                print( "only 1 winning move, ready for check" )
                break
            if num_winning_moves == 0:
                print( "no winning moves" )
                return False, None, None, None
            print(f"    Eliminating moves because {num_winning_moves} excess winning moves." )

            candidate_move = winning_moves[1]['move']
            candidate_score = winning_moves[1]['scoreLead']
            candidate_winrate = winning_moves[1]['winrate']
            if candidate_move == "pass":
                return False,None,None,None
            print(f"    {player_to_move} {candidate_move} (score: {candidate_score:.1f}, winrate: {candidate_winrate:.3f})")

            # Add candidate move to adjusted_moves
            adjusted_moves.append([player_to_move, candidate_move])

            # Query for opponent's best response
            response_after_candidate = self.strong_controller.query_root(adjusted_moves)

            # Extract policy and find best move index (excluding -1 for illegal moves)
            policy = response_after_candidate["policy"]
            best_idx = max(range(len(policy)), key=lambda i: policy[i] if policy[i] != -1 else -999)
            opponent_move = self.strong_controller.get_move_from_idx(best_idx)
            print(f"    {other_player(player_to_move)} {candidate_move} (score: ?, winrate: ?)")

            # Add opponent's response to adjusted_moves
            adjusted_moves.append([other_player(player_to_move), opponent_move])

        # Now verify the solution move
        solution_response = self.strong_controller.query_children_100_visits(adjusted_moves)
        solution_move_info = solution_response["moveInfos"][0]

        solution_move = solution_move_info["move"]
        solution_winrate = solution_move_info["winrate"]
        solution_score = solution_move_info["scoreLead"]
        if solution_winrate < 0.5:
            print(f"    ❌ Solution move winrate too low: {solution_winrate:.3f}")
            return False, None, None, None

        # Check alternative moves
        exclude_response = self.strong_controller.query_children_exclude_move(adjusted_moves, solution_move, player_to_move)
        for move_info in exclude_response["moveInfos"]:
            if move_info["winrate"] > 0.5 or move_info["scoreLead"] > 0 :
                print(f"    ❌ Alternative move {move_info['move']} has winrate {move_info['winrate']:.3f} and score {move_info["scoreLead"]}")
                return False, None, None, None

        # All checks passed
        return True, solution_score, adjusted_moves, solution_move

    def _get_position_info(self, moves):
        """
        Helper to get the current position score and winrate from Black's perspective.
        Returns (score, winrate) or (None, None) if failed.
        Uses maxVisits=10 for a quick but reliable read.
        """
        response = self.strong_controller.query_root(moves)
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
        response = self.strong_controller.query_children(moves)
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
        # Query for moves in current position (player's turn)
        response = self.strong_controller.query_children(moves)

        best_candidate = None
        best_candidate_penalty = float('inf')
        best_candidate_winrate = None
        best_candidate_score = None

        # Find best candidate move for player
        for move_info in response["moveInfos"]:
            candidate_move = move_info["move"]
            if "scoreLead" not in move_info or "visits" not in move_info:
                continue

            candidate_score = move_info["scoreLead"]
            candidate_winrate = move_info["winrate"]
            penalty = self._penalty(candidate_score,candidate_winrate)
            # penalty = abs(candidate_score - 0.5)

            if penalty < best_candidate_penalty:
                best_candidate_penalty = penalty
                best_candidate = candidate_move
                best_candidate_winrate = candidate_winrate
                best_candidate_score = candidate_score

        # Apply player's candidate move
        adjusted_moves = moves + [[player, best_candidate]]

        return adjusted_moves,best_candidate,best_candidate_winrate,best_candidate_score
