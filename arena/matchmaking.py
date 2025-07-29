# matchmaking.py
import itertools


class Matchmaking:
    """Selects the bot pairing with the highest informational value based on aggregated data."""

    def get_next_battle(self, bot_names: list, aggregated_results: dict) -> tuple:
        """
        Determines the best pair to battle next.
        Strategy:
        1. Prioritize pairs with the minimum number of games played.
        2. Among those, choose the pair whose win ratio is closest to 50/50.
        """
        if len(bot_names) < 2:
            return None

        all_possible_pairs = list(itertools.combinations(sorted(bot_names), 2))

        pair_stats = []
        for bot_a, bot_b in all_possible_pairs:
            pair_key = frozenset({bot_a, bot_b})
            data = aggregated_results.get(pair_key)

            if data:
                total_games = data['total_games']
                win_ratio = data['wins_a'] / total_games if total_games > 0 else 0.5
            else:
                total_games = 0
                win_ratio = 0.5  # Assume 50/50 for pairs that haven't played

            # Score is how far from 50/50 the match is (lower is better)
            contention_score = abs(win_ratio - 0.5)
            pair_stats.append(((bot_a, bot_b), total_games, contention_score))

        # Sort by total_games (ascending), then by contention_score (ascending)
        pair_stats.sort(key=lambda x: (x[1], x[2]))

        return pair_stats[0][0]  # Return the best pair