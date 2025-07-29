# ranking.py
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class ProbitRanking:
    """
    Calculates bot ranks using a probit model on aggregated pair data.
    The likelihood of observing k wins in n games for a pair is modeled
    using the binomial distribution.
    """

    def __init__(self, persistence, bot_names: list):
        self.persistence = persistence
        self.bot_names = sorted(bot_names)
        self.bot_map = {name: i for i, name in enumerate(self.bot_names)}
        self.ranks = np.zeros(len(self.bot_names))
        self.uncertainties = np.inf * np.ones(len(self.bot_names))

    def _negative_log_likelihood(self, skills, aggregated_games):
        """
        Binomial negative log-likelihood function.
        """
        log_likelihood = 0.0

        for pair_data in aggregated_games.values():
            bot_a, bot_b = pair_data['bot_a'], pair_data['bot_b']
            idx_a = self.bot_map[bot_a]
            idx_b = self.bot_map[bot_b]

            skill_a, skill_b = skills[idx_a], skills[idx_b]

            wins_a = pair_data['wins_a']
            wins_b = pair_data['wins_b']

            # P(A wins) = Φ(skill_a - skill_b)
            prob_a_wins = norm.cdf(skill_a - skill_b)

            # Add epsilon to prevent log(0)
            epsilon = 1e-9
            prob_a_wins = np.clip(prob_a_wins, epsilon, 1 - epsilon)

            # Binomial log-likelihood: k*log(p) + (n-k)*log(1-p)
            log_likelihood += wins_a * np.log(prob_a_wins) + wins_b * np.log(1 - prob_a_wins)

        return -log_likelihood

    def update_ranks(self):
        """
        Recalculates all bot ranks based on the entire aggregated battle history.
        """
        aggregated_games = self.persistence.get_aggregated_pair_results()
        if not aggregated_games:
            print("No games played yet. Ranks are at their initial state.")
            return

        initial_skills = self.ranks
        constraints = ({'type': 'eq', 'fun': lambda skills: np.mean(skills)})

        result = minimize(
            fun=self._negative_log_likelihood,
            x0=initial_skills,
            args=(aggregated_games,),
            method='SLSQP',
            constraints=constraints,
        )

        if result.success:
            self.ranks = result.x
            # The uncertainty calculation remains valid for the estimated parameters
            try:
                hess_inv_matrix = result.hess_inv.todense()
                self.uncertainties = np.sqrt(np.diag(hess_inv_matrix))
            except Exception:
                self.uncertainties = np.full(len(self.bot_names), np.inf)
        else:
            print(f"Warning: Optimizer failed to converge. {result.message}")

    def get_ranks(self) -> dict:
        return {
            name: {'rank': self.ranks[i], 'uncertainty': self.uncertainties[i]}
            for name, i in self.bot_map.items()
        }

    def print_leaderboard(self):
        print("\n--- Probit Model Leaderboard (Aggregated) ---")
        ranks = self.get_ranks()
        sorted_bots = sorted(ranks.items(), key=lambda item: item[1]['rank'], reverse=True)

        print(f"{'#':<4} {'Bot Name':<25} {'Skill (Rank)':<15} {'Uncertainty':<15}")
        print("-" * 60)
        for i, (name, stats) in enumerate(sorted_bots):
            uncertainty_str = f"{stats['uncertainty']:.4f}" if stats['uncertainty'] != np.inf else 'N/A'
            print(f"{i + 1:<4} {name:<25} {stats['rank']:<15.4f} {uncertainty_str:<15}")
        print("-" * 60)