# bot_interface.py
from abc import ABC, abstractmethod

class BotInterface(ABC):
    """
    Defines the standard interface that all bots must implement.
    """

    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of the bot."""
        pass

    @abstractmethod
    def start_bot(self, game_state: dict):
        """Initializes the bot for a new game."""
        pass

    @abstractmethod
    def add_move(self, move: any, player_name: str):
        """Informs the bot of a move made in the game."""
        pass

    @abstractmethod
    def get_next_move(self, game_state: dict) -> any:
        """Requests the bot's next move."""
        pass

    @abstractmethod
    def stop_bot(self, game_state: dict):
        """Called at the end of a game for cleanup."""
        pass

    @abstractmethod
    def get_winner(self, game_state: dict) -> str:
        """
        Asks the bot to declare the winner from its perspective.
        Should return the name of the winning bot or "draw".
        """
        pass