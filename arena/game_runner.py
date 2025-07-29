# game_runner.py

# A placeholder Game Logic class.
# In a real application, this would contain your game's rules.
class SimpleGame:
    def get_winner(self, move_a, move_b, name_a, name_b):
        # Example: Rock-Paper-Scissors
        rules = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
        if move_a == move_b: return "draw"
        if rules.get(move_a) == move_b: return name_a
        return name_b


class GameRunner:
    """Orchestrates a single battle and verifies the winner."""

    def __init__(self, bot_a, bot_b):
        self.bot_a = bot_a
        self.bot_b = bot_b
        # The game state can be as complex as needed
        self.game_state = {'history': [], 'winner': None}
        self.game_logic = SimpleGame()  # Replace with your actual game logic

    def run_battle(self) -> str:
        """Executes a complete battle and returns the verified winner's name."""
        self.bot_a.start_bot(self.game_state)
        self.bot_b.start_bot(self.game_state)

        # In a turn-based game, this would be a loop.
        # For a simple one-shot game:
        move_a = self.bot_a.get_next_move(self.game_state)
        move_b = self.bot_b.get_next_move(self.game_state)

        # Update bots with opponent's move
        self.bot_a.add_move(move_b, self.bot_b.name())
        self.bot_b.add_move(move_a, self.bot_a.name())

        # The game runner determines the official winner
        official_winner = self.game_logic.get_winner(move_a, move_b, self.bot_a.name(), self.bot_b.name())
        self.game_state['winner'] = official_winner

        # CRITICAL: Check that both bots agree on the winner
        winner_from_a = self.bot_a.get_winner(self.game_state)
        winner_from_b = self.bot_b.get_winner(self.game_state)

        if not (winner_from_a == winner_from_b == official_winner):
            print("\n!!! WINNER DISAGREEMENT !!!")
            print(f"Official Winner: {official_winner}")
            print(f"{self.bot_a.name()} declares: {winner_from_a}")
            print(f"{self.bot_b.name()} declares: {winner_from_b}")
            # This result should be considered invalid
            return "disagreement"

        self.bot_a.stop_bot(self.game_state)
        self.bot_b.stop_bot(self.game_state)

        print(f"Battle: {self.bot_a.name()} vs {self.bot_b.name()} -> Winner: {official_winner}")
        return official_winner