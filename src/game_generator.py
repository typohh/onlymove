from src.katago_controller import KatagoController
from sgfmill import sgf

class GroundTruthGenerator:
    """
    Generates optimal game sequences using the RL network.
    """
    def __init__(self, controller: KatagoController):
        """
        Initializes the GroundTruthGenerator.

        Args:
            controller (KatagoController): The KataGo controller for the RL network.
        """
        self.controller = controller

    def generate_game(self, board_size=9, komi=7.5):
        """
        Generates a single optimal game.

        Args:
            board_size (int): The size of the board.
            komi (float): The komi for the game.

        Returns:
            list: A list of moves representing the game sequence.
        """
        moves = []
        player = "B"
        consecutive_passes = 0

        while consecutive_passes < 2:
            query = {
                "id": "game_gen",
                "moves": moves,
                "initialStones": [],
                "rules": "tromp-taylor",
                "komi": komi,
                "boardXSize": board_size,
                "boardYSize": board_size,
                "maxVisits": 1000  # High visits for optimal play
            }

            response = self.controller.send_query(query)
            best_move = response["moveInfos"][0]["move"]

            if best_move == "pass":
                consecutive_passes += 1
            else:
                consecutive_passes = 0

            moves.append([player, best_move])
            player = "W" if player == "B" else "B"

        return moves

    def save_sgf(self, moves, filename, board_size=9, komi=7.5):
        """
        Saves a game to an SGF file.

        Args:
            moves (list): The list of moves in the game.
            filename (str): The name of the SGF file.
            board_size (int): The size of the board.
            komi (float): The komi for the game.
        """
        game = sgf.Sgf_game(size=board_size)
        game.root.set("KM", str(komi))
        game.root.set("RU", "tromp-taylor")

        for player, move in moves:
            node = game.extend_main_sequence()
            node.set_move(player.lower(), self._katago_to_sgf_coords(move, board_size))

        with open(filename, "wb") as f:
            f.write(game.serialise())

    def _katago_to_sgf_coords(self, katago_move, board_size):
        """
        Converts KataGo coordinates to SGF coordinates.

        Args:
            katago_move (str): The move in KataGo's coordinate format (e.g., 'A1', 'B2').
            board_size (int): The size of the board.

        Returns:
            tuple: The move in SGF coordinate format (row, col).
        """
        if katago_move == "pass":
            return None

        col_str = katago_move[0]
        row_str = katago_move[1:]

        col = ord(col_str.upper()) - ord('A')
        if col_str.upper() >= 'I':
            col -= 1
        row = board_size - int(row_str)

        return row, col
