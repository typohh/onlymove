import numpy as np
from scipy.stats import norm

from katago_queries import KatagoQueries

SUPPORTED_RANKS = [
    "rank_20k", "rank_19k", "rank_18k", "rank_17k", "rank_16k",
    "rank_15k", "rank_14k", "rank_13k", "rank_12k", "rank_11k",
    "rank_10k", "rank_9k", "rank_8k", "rank_7k", "rank_6k",
    "rank_5k", "rank_4k", "rank_3k", "rank_2k", "rank_1k",
    "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d",
    "rank_6d", "rank_7d", "rank_8d", "rank_9d"
]

PREAZ_RANKS = [
    "preaz_20k", "preaz_19k", "preaz_18k", "preaz_17k", "preaz_16k",
    "preaz_15k", "preaz_14k", "preaz_13k", "preaz_12k", "preaz_11k",
    "preaz_10k", "preaz_9k", "preaz_8k", "preaz_7k", "preaz_6k",
    "preaz_5k", "preaz_4k", "preaz_3k", "preaz_2k", "preaz_1k",
    "preaz_1d", "preaz_2d", "preaz_3d", "preaz_4d", "preaz_5d",
    "preaz_6d", "preaz_7d", "preaz_8d", "preaz_9d"
]

SIMULATION_ROUNDS = 10


class DifficultyEvaluator:
    """
    Assesses the difficulty of a move or game.
    """

    def __init__(self, queries: KatagoQueries):
        """
        Initializes the DifficultyAssessor.

        Args:
            queries (KatagoQueries): The KataGo controller for the SL network.
        """
        self.queries = queries

    def assess_move_difficulty(self, moves, correct_move):
        """
        Assesses difficulty by sampling all ranks determinig CDF for probability of move being played.

        Args:
            moves (list): The list of moves in the game.
            correct_move (str): The correct move.

        Returns:
            String: closest mean rank as a string
            Float: mean rank of the distribution
            Float: stdev rank of the distribution.

        """
        rank_mu, rank_sigma = self._assess_move_difficulty(moves, correct_move, SUPPORTED_RANKS)
        preaz_mu, preaz_sigma = self._assess_move_difficulty(moves, correct_move, PREAZ_RANKS)
        average_mu = (rank_mu + preaz_mu) / 2
        average_sigma = (rank_sigma + preaz_sigma) / 2

        return average_mu, average_sigma

    def _assess_move_difficulty(self, moves, correct_move, ranks):
        """
        Assesses difficulty by sampling all ranks determinig CDF for probability of move being played.

        Args:
            moves (list): The list of moves in the game.
            correct_move (str): The correct move.
            ranks (list): The list of ranks to assess against.

        Returns:
            Float: mean rank of the distribution
            Float: stdev rank of the distribution.

        """
        probabilities = []
        for rank in ranks:
            probability = self._get_probability_for_rank(rank, moves, correct_move)
            probabilities.append(probability)

        # Convert probabilities to z-scores
        clamped_probs = np.clip(probabilities, 0.01, 0.99)
        z_scores = norm.ppf(clamped_probs)

        # Create numerical rank values
        numerical_ranks = np.arange(len(ranks))

        # Solve linear regression: rank = mu + sigma * z_score
        a = np.vstack([np.ones(len(z_scores)), z_scores]).T
        mu, sigma = np.linalg.lstsq(a, numerical_ranks)[0]

        # Ensure positive standard deviation
        sigma = abs(sigma)

        return mu, sigma

    def _get_probability_for_rank(self, rank_string, moves, correct_move):
        """
        Gets the move probability of the correct move for a given strength rank.

        Args:
            rank_string (str): The rank to check.
            moves (list): The list of moves in the game.
            correct_move (str): The correct move.

        Returns:
            float: probability of the correct move being played.

        Raises:
            Exception: If any step fails unexpectedly.
        """
        response = self.queries.query_policy(moves, human_rank=rank_string)

        # Use humanPolicy if available, else policy
        policy = response["humanPolicy"]

        # Get index of correct move and its policy probability
        idx = self.queries.get_idx_from_move(correct_move)

        return policy[idx]

    def _playout_game(self, moves, rank):
        simulation_moves = moves.copy()
        while True:
            # Human player's turn
            human_move = self.queries.query_weighted_random_move(simulation_moves, rank)
            simulation_moves.append(human_move)

            # Check if game is over
            if len(simulation_moves) > 1 and simulation_moves[-1] == "pass" and simulation_moves[-2] == "pass":
                break

            # KataGo's turn
            # response = self.queries.query_low(simulation_moves)
            # katago_move = response["moveInfos"][0]["move"]
            katago_move = self.queries.get_best_move(self.queries.query_policy(simulation_moves))
            simulation_moves.append(katago_move)

            # Check if game is over
            if len(simulation_moves) > 1 and simulation_moves[-1] == "pass" and simulation_moves[-2] == "pass":
                break

        score, _ = self.queries.get_position_info(simulation_moves)
        return score > 0  # Black wins

    def assess_game_difficulty(self, moves):
        """
        Assesses the difficulty of winning the game from the current position.
        This is done by simulating games against the full-strength KataGo engine,
        with the human player choosing moves based on a weighted random sample
        from the policy network of different ranks.

        Args:
            moves (list): The list of moves in the game.

        Returns:
            str: The rank at which the human player has a 50% chance of winning.
        """
        best_rank = None
        closest_winrate_diff = float('inf')

        for rank in SUPPORTED_RANKS:
            wins = 0
            for _ in range(SIMULATION_ROUNDS):
                if self._playout_game(moves, rank):
                    wins += 1
            winrate = wins / SIMULATION_ROUNDS
            winrate_diff = abs(winrate - 0.5)

            if winrate_diff < closest_winrate_diff:
                closest_winrate_diff = winrate_diff
                best_rank = rank

        return best_rank[5:]

    @staticmethod
    def get_rank_str(mean, stddev=None):
        """
        returns rank as string
        """
        mean_str = SUPPORTED_RANKS[int(round(mean))][5:]
        if stddev:
            return f"{mean_str}±{stddev:.1f}"
        else:
            return mean_str
