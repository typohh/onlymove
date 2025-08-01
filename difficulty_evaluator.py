from collections import Counter

import numpy as np
from scipy.stats import norm, stats

from katago_queries import KatagoQueries, get_all_ranks, get_rank_number
from stats import DEBUG

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

    def assess_game_difficulty(self, moves, move_rank, correct_move):
        current_rank = move_rank
        number_of_ranks = len(get_all_ranks())
        games = []
        # games playes, continue until the number of games is more than SIMULATIONS ROUNDS, ignoring the first game of each rank
        while len(games) - len(set(games)) < SIMULATION_ROUNDS:
            current_rank = max(0, min(current_rank, 1))
            games.append(current_rank)

            if self._playout_game(moves + [correct_move], current_rank):
                current_rank -= 1 / number_of_ranks
            else:
                current_rank += 1 / number_of_ranks

        if DEBUG: print("ranks played:", games)

        # ignore first game of each rank
        counter = Counter(games)
        relevant_games = list(Counter({k: v - 1 for k, v in counter.items() if v > 1}).elements())
        return float(stats.tmean(relevant_games)), 0.0 if len(set(relevant_games)) == 1 else float(stats.tstd(relevant_games))

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
        probabilities = []
        for rank in get_all_ranks():
            probability = self.get_probability_for_rank(rank, moves, correct_move)
            probabilities.append(probability)

        mu, sigma = rank_probabilities_into_distribution(probabilities)

        if DEBUG: print("move difficulty:", mu, sigma, probabilities)

        return mu, sigma

    def assess_move_difficulty_alternative(self, moves, correct_move):
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
        most_likely = None
        most_likely_twice_over = None
        for rank in get_all_ranks():
            response = self.queries.query_policy(moves, human_rank=rank)
            policy = response["humanPolicy"]
            best_move_idx = self.queries.get_idx_from_move(correct_move)
            second_best_move_idx = 0
            for i in range(len(policy)):
                if i != best_move_idx:
                    second_best_move_idx = i
            correct_policy = policy[best_move_idx]
            altenative_policy = policy[second_best_move_idx]

            if most_likely is None and correct_policy > altenative_policy:
                most_likely = rank

            if most_likely_twice_over is None and correct_policy > 2 * altenative_policy:
                most_likely_twice_over = rank

            if most_likely is not None and most_likely_twice_over is not None:
                break

        return get_rank_number(most_likely), get_rank_number(most_likely_twice_over)

    def get_probability_for_rank(self, rank, moves, correct_move):
        """
        Gets the move probability of the correct move for a given strength rank.

        Args:
            rank (float): The rank to check.
            moves (list): The list of moves in the game.
            correct_move (str): The correct move.

        Returns:
            float: probability of the correct move being played.

        Raises:
            Exception: If any step fails unexpectedly.
        """
        response = self.queries.query_policy(moves, human_rank=rank)

        # Use humanPolicy if available, else policy
        policy = response["humanPolicy"]

        # Get index of correct move and its policy probability
        idx = self.queries.get_idx_from_move(correct_move)

        return policy[idx]

    def _playout_game(self, moves, rank_str):
        simulation_moves = moves.copy()
        while True:
            # KataGo's turn
            katago_move, winrate, _ = self.queries.get_best_move(self.queries.query_policy(simulation_moves))

            if winrate < 0.01:  # katago thinks 99% chance of it winning
                if DEBUG: print(rank_str, "resigns after", len(simulation_moves) - len(moves), "moves");
                return False

            simulation_moves.append(katago_move)

            # Check if game is over
            if len(simulation_moves) > 1 and simulation_moves[-1] == "pass" and simulation_moves[-2] == "pass":
                break

            # Human player's turn
            human_move = self.queries.query_weighted_random_move(simulation_moves, rank_str)
            simulation_moves.append(human_move)

            # Check if game is over
            if len(simulation_moves) > 1 and simulation_moves[-1] == "pass" and simulation_moves[-2] == "pass":
                break

        score, _ = self.queries.get_position_info(simulation_moves)
        return score > 0  # Black wins


def rank_probabilities_into_distribution(probabilities):
    # Convert probabilities to z-scores
    clamped_probs = np.clip(probabilities, 0.01, 0.99)
    z_scores = norm.ppf(clamped_probs)

    # Create numerical rank values
    numerical_ranks = np.arange(len(probabilities))

    # Solve linear regression: rank = mu + sigma * z_score
    a = np.vstack([np.ones(len(z_scores)), z_scores]).T
    mu, sigma = np.linalg.lstsq(a, numerical_ranks)[0]

    # Ensure positive standard deviation
    sigma = abs(sigma)

    return mu / len(probabilities), sigma / len(probabilities)
