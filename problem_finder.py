import numpy as np
from scipy.stats import norm
from sgfmill import sgf

from katago_queries import KatagoQueries, other_player
from game_generator import katago_to_sgf_coords

SUPPORTED_RANKS = [
    "rank_20k", "rank_19k", "rank_18k", "rank_17k", "rank_16k",
    "rank_15k", "rank_14k", "rank_13k", "rank_12k", "rank_11k",
    "rank_10k", "rank_9k", "rank_8k", "rank_7k", "rank_6k",
    "rank_5k", "rank_4k", "rank_3k", "rank_2k", "rank_1k",
    "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d",
    "rank_6d", "rank_7d", "rank_8d", "rank_9d"
]

MIN_SCORE = 0
MAX_SCORE = 2
MIN_WINRATE_CORRECT = 0.6
MAX_WINRATE_ALTERNATIVE = 0.4

class ProblemFinder:
    """
    Assesses the difficulty of a move.
    """
    def __init__(self, queries: KatagoQueries):
        """
        Initializes the DifficultyAssessor.

        Args:
            human_controller (KatagoQueries): The KataGo controller for the SL network.
        """
        self.queries = queries

    def assess_move_difficulty(self, moves, player_to_move, correct_move):
        """
        Assesses difficulty by sampling all ranks determinig CDF for probability of move being first choice.

        Args:
            moves (list): The list of moves in the game.
            player_to_move (str): The player to move ("B" or "W").
            correct_move (str): The correct move.

        Returns:
            String: closest mean rank as a string
            Float: mean rank of the distribution
            Float: stdev rank of the distribution.

        """
        probabilities = []
        for rank in SUPPORTED_RANKS:
            rank_order = self._get_order_for_rank(rank, moves, correct_move)
            probabilities.append(1.0 if rank_order == 0 else 0.0)

        # Convert probabilities to z-scores
        clamped_probs = np.clip(probabilities, 0.01, 0.99)
        z_scores = norm.ppf(clamped_probs)

        # Create numerical rank values (0 to 28)
        numerical_ranks = np.arange(len(SUPPORTED_RANKS))

        # Solve linear regression: rank = mu + sigma * z_score
        A = np.vstack([np.ones(len(z_scores)), z_scores]).T
        mu, sigma = np.linalg.lstsq(A, numerical_ranks, rcond=None)[0]

        # Ensure positive standard deviation 
        sigma = abs(sigma)

        # Convert numerical mean to rank string
        rank_idx = int(round(mu))
        if rank_idx < 0: rank_idx = 0
        if rank_idx >= len(SUPPORTED_RANKS): rank_idx = len(SUPPORTED_RANKS) - 1
        rank_str = SUPPORTED_RANKS[rank_idx].replace("rank_", "").replace("_", " ").upper()

        return rank_str, mu, sigma


    def _get_order_for_rank(self, rank_string, moves, correct_move):
        """
        Gets the move rank of moves by plicy of the correct move for a given strength rank.

        Args:
            rank_string (str): The rank to check.
            moves (list): The list of moves in the game.
            correct_move (str): The correct move.

        Returns:
            float: index of the correct move when sorted by policy, 0 indicating first choice, and ascending from there.

        Raises:
            Exception: If any step fails unexpectedly.
        """
        response = self.queries.query_root(moves, human_rank=rank_string)
        if not response:
            raise Exception(f"No response for rank: {rank_string}")

        # Use humanPolicy if available, else policy
        policy = response.get("humanPolicy", response.get("policy", []))
        if not policy:
            raise Exception(f"Empty policy in response for rank: {rank_string}")

        # Get index of correct move and its policy probability
        idx = self.queries.get_idx_from_move(correct_move)
        if idx is None:
            raise Exception(f"Invalid move: {correct_move}")
        if idx >= len(policy):
            raise Exception(f"Move index {idx} out of bounds for policy length {len(policy)}")

        # Return rank by sorting policy probabilities
        prob_with_idx = [(p, i) for i, p in enumerate(policy) if p > 0]
        prob_with_idx.sort(reverse=True)
        rank = next(i for i, (p, move_idx) in enumerate(prob_with_idx) if move_idx == idx)
        return rank

    def _print_position_info(self, moves, label):
        """
        Helper to get and print position info (score, winrate, num_moves_above_50).
        """
        score, winrate = self.queries.get_position_info(moves)
        num_above = self._get_num_moves_above_50(moves)
        print(f"    {label}: {score:.1f}, {winrate:.3f}, {num_above}")

    def adjust_tsumego(self,moves,player_to_move):
        """
        """
        adjusted_moves = moves.copy()
        move_num = len(moves)+1

        print(f"\n  Adjusting tsumego at move {move_num}...")
        self._print_position_info(adjusted_moves, "Initial")

        # First, check if the score is already in [MIN_SCORE-MAX_SCORE] for Black
        while True:
            score,winrate = self.queries.get_position_info(adjusted_moves)
            if MIN_SCORE < score < MAX_SCORE and winrate > MIN_WINRATE_CORRECT:
                break
            print(f"    Adjusting score because (score: {score:.1f}, winrate: {winrate:.3f})")
            adjusted_moves, candidate_move, candidate_winrate, candidate_score = self._adjust_score(adjusted_moves, player_to_move)
            print(f"    {player_to_move} {candidate_move} (score: {candidate_score:.1f}, winrate: {candidate_winrate:.3f})")
            if candidate_move == "pass":
                return None
            adjusted_moves, candidate_move, candidate_winrate, candidate_score = self._adjust_score(adjusted_moves, other_player(player_to_move))
            print(f"    {other_player(player_to_move)} {candidate_move} (score: {candidate_score:.1f}, winrate: {candidate_winrate:.3f})")
            if candidate_move == "pass":
                return None

        self._print_position_info(adjusted_moves, "After adjustment")

        while True:
            response = self.queries.query_children_100_visits(adjusted_moves)
            # Count moves above 50% winrate
            winning_moves = [mi for mi in response["moveInfos"] if mi['winrate'] > MAX_WINRATE_ALTERNATIVE]
            num_winning_moves = len(winning_moves)
            if num_winning_moves == 1:
                print( "only 1 winning move, ready for check" )
                break
            if num_winning_moves == 0:
                print( "no winning moves" )
                return None
            print(f"    Eliminating moves because {num_winning_moves} excess winning moves." )

            candidate_move = winning_moves[1]['move']
            candidate_score = winning_moves[1]['scoreLead']
            candidate_winrate = winning_moves[1]['winrate']
            if candidate_move == "pass":
                return None
            print(f"    {player_to_move} {candidate_move} (score: {candidate_score:.1f}, winrate: {candidate_winrate:.3f})")

            # Add candidate move to adjusted_moves
            adjusted_moves.append([player_to_move, candidate_move])

            # Query for opponent's best response
            response_after_candidate = self.queries.query_root(adjusted_moves)

            # Extract policy and find best move index (excluding -1 for illegal moves)
            policy = response_after_candidate["policy"]
            best_idx = max(range(len(policy)), key=lambda i: policy[i] if policy[i] != -1 else -999)
            opponent_move = self.queries.get_move_from_idx(best_idx)
            print(f"    {other_player(player_to_move)} {opponent_move} (score: ?, winrate: ?)")

            # Add opponent's response to adjusted_moves
            adjusted_moves.append([other_player(player_to_move), opponent_move])

        return adjusted_moves


    def validate_tsumego(self, moves, player_to_move):
        """
        Validates position to to ensure its score 0-1 for black, only one move with positive score and winrate above 50%, and correct move is in range 19kyu-8dan.

        Args:
            moves (list): The list of moves in the game.
            player_to_move (str): The player to move ("B" or "W").

        Returns:
            tuple: (is_valid, solution_move, rank_str, mean rank, std rank)
        """
        move_num = len(moves)+1
        print(f"\n  Validating tsumego...")

        # Now verify the solution move
        solution_response = self.queries.query_children_100_visits(moves)
        solution_move_info = solution_response["moveInfos"][0]

        solution_move = solution_move_info["move"]
        solution_winrate = solution_move_info["winrate"]
        solution_score = solution_move_info["scoreLead"]
        if solution_winrate < MIN_WINRATE_CORRECT:
            print(f"    ❌ Solution move winrate of {solution_winrate:.3f} is below {MIN_WINRATE_CORRECT}:.3f")
            return False, None, None, None, None
        if not MIN_SCORE < solution_score < MAX_SCORE:
            print(f"    ❌ Solution move score: {solution_score:.3f} is not in range [{MIN_SCORE}-{MAX_SCORE}]")
            return False, None, None, None, None

        # Check alternative moves
        exclude_response = self.queries.query_children_exclude_move(moves, solution_move, player_to_move)
        for move_info in exclude_response["moveInfos"]:
            if move_info["winrate"] > MAX_WINRATE_ALTERNATIVE:
                print(f"    ❌ Alternative move {move_info['move']} has winrate {move_info['winrate']:.3f}")
                return False, None, None, None, None

        rank_str,mu,sigma = self.assess_move_difficulty(moves, player_to_move, solution_move)
        print(f"  Difficulty: {mu:.2f}({rank_str}) ± {sigma:.3f} ")
        if not 0 < mu < len(SUPPORTED_RANKS) - 1 :
            print(f"    ❌ Tsumego too easy or hard [19k-8d]")
            return False, None, None, None, None

        return True, solution_move, rank_str, mu, sigma

    def _get_num_moves_above_50(self, moves):
        """
        Helper to get the number of moves with winrate above 50% for the current player.
        Returns None if failed.
        """
        response = self.queries.query_children(moves)
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

    def _adjust_score(self, moves, player):
        """
        Adjusts position to ensure solution leads to ~(MIN_SCORE+MAX_SCORE)/2 win.
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
        response = self.queries.query_children(moves)

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
            penalty = abs(candidate_score - (MAX_SCORE + MIN_SCORE) / 2) + 0.25 * (1 - candidate_winrate)
            # penalty = abs(candidate_score - 0.5)

            if penalty < best_candidate_penalty:
                best_candidate_penalty = penalty
                best_candidate = candidate_move
                best_candidate_winrate = candidate_winrate
                best_candidate_score = candidate_score

        # Apply player's candidate move
        adjusted_moves = moves + [[player, best_candidate]]

        return adjusted_moves,best_candidate,best_candidate_winrate,best_candidate_score


    def find_problems(self, game_moves):
        """
        Finds problems in a game sequence.

        Args:
            game_moves (list): The list of moves in the game.

        Returns:
            list: A list of identified problems.
        """
        # Pre-allocate difficulties array
        tsumegos = []

        # Analyze all moves except first 10 and last 10
        for i in range(10, len(game_moves) - 10):
            if game_moves[i][0] != "B":
                continue
            moves_so_far = game_moves[:i]
            player_to_move = game_moves[i][0]

            print(f"\nValidating tsumego at move {i+1}/{len(game_moves)}...")
            adjusted_moves = self.adjust_tsumego(moves_so_far,player_to_move)
            if adjusted_moves is None:
                continue
            is_valid, solution_move, rank_str, mu, sigma = self.validate_tsumego(adjusted_moves, player_to_move)
            if not is_valid:
                continue
            print( "Valid tsumego" )
            tsumegos.append({
                "moves":adjusted_moves,
                "solution":solution_move,
                "rank":rank_str,
                "mean":mu,
                "stddev":sigma
            })
        return tsumegos

    def save_problem_as_sgf(self, problem, filename, board_size=9, komi=7.5):
        """
        Saves a single problem as an SGF file with annotations.

        Args:
            problem (dict): The problem dictionary.
            filename (str): The name of the SGF file.
            board_size (int): The size of the board.
            komi (float): The komi for the game.
        """
        game = sgf.Sgf_game(size=board_size)
        game.root.set("KM", str(komi))
        game.root.set("RU", "tromp-taylor")
        game.root.set("GN", f"Problem Difficulty: {problem["rank"]}±{int(round(problem["stddev"]))}")

        # Add initial moves
        for player, move in problem["moves"]:
            node = game.extend_main_sequence()
            node.set_move(player.lower(), katago_to_sgf_coords(move, board_size))

        # Add the solution move and annotations
        solution_player, solution_move = problem["solution"]
        solution_node = game.extend_main_sequence()
        solution_node.set_move(solution_player.lower(), katago_to_sgf_coords(solution_move, board_size))
        solution_node.set("C", f"Solution move for {problem["rank"]}±{int(round(problem["stddev"]))} problem.")
        solution_node.set("MA", [katago_to_sgf_coords(solution_move, board_size)])

        with open(filename, "wb") as f:
            f.write(game.serialise())
