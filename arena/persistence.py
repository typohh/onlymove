# persistence.py
import sqlite3
from datetime import datetime
from collections import defaultdict


class Persistence:
    """Handles persistence and aggregation of battle results."""

    def __init__(self, db_path='bot_battles_aggregated.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_a_name TEXT NOT NULL,
                bot_b_name TEXT NOT NULL,
                winner_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        ''')
        self.conn.commit()

    def record_battle(self, bot_a_name: str, bot_b_name: str, winner_name: str):
        """Records the outcome of a single battle."""
        timestamp = datetime.utcnow()
        # Ensure consistent ordering for aggregation
        bot1, bot2 = sorted([bot_a_name, bot_b_name])
        self.cursor.execute('''
            INSERT INTO results (bot_a_name, bot_b_name, winner_name, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (bot1, bot2, winner_name, timestamp))
        self.conn.commit()

    def get_aggregated_pair_results(self) -> dict:
        """
        Aggregates results for every pair of bots.

        Returns:
            A dictionary where key is a frozenset({bot_a, bot_b}) and
            value is {'wins_a': count, 'total_games': count}.
            'bot_a' is always the alphabetically first name.
        """
        self.cursor.execute("""
            SELECT
                bot_a_name,
                bot_b_name,
                SUM(CASE WHEN winner_name = bot_a_name THEN 1 ELSE 0 END) as wins_a,
                SUM(CASE WHEN winner_name = bot_b_name THEN 1 ELSE 0 END) as wins_b,
                COUNT(*) as total_games
            FROM results
            WHERE winner_name != 'draw'
            GROUP BY bot_a_name, bot_b_name
        """)

        aggregated_data = {}
        for bot_a, bot_b, wins_a, wins_b, total in self.cursor.fetchall():
            pair_key = frozenset({bot_a, bot_b})
            aggregated_data[pair_key] = {
                'bot_a': bot_a,
                'bot_b': bot_b,
                'wins_a': wins_a,
                'wins_b': wins_b,
                'total_games': total
            }
        return aggregated_data

    def close(self):
        self.conn.close()