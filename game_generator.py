from katago_controller import KatagoController
from sgfmill import sgf

def generate_game(controller:KatagoController):
    """
    Generates a single game.

    Args:

    Returns:
        list: A list of moves representing the game sequence.
    """
    moves = []
    player = "B"
    consecutive_passes = 0

    while consecutive_passes < 2:
        response = controller.query_root()
        if response and "policy" in response and response["policy"]:
            best_move_idx = response['policy'].index(max(response['policy']))
            best_move = controller.get_move_from_idx(best_move_idx)
        elif response and "moveInfos" in response and response["moveInfos"]:
            best_move = response["moveInfos"][0]["move"]
        else:
            best_move = "pass"

        if best_move == "pass":
            consecutive_passes += 1
        else:
            consecutive_passes = 0

        moves.append([player, best_move])
        player = "W" if player == "B" else "B"

    return moves

def save_sgf(moves, filename, board_size=9, komi=7.5):
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


def load_sgf(filename):
    """
    Loads a game from an SGF file.

    Args:
        filename (str): The name of the SGF file.

    Returns:
        list: A list of moves representing the game sequence.
    """
    with open(filename, 'rb') as f:
        sgf_game = sgf.Sgf_game.from_bytes(f.read())

    moves = []
    for node in sgf_game.get_main_sequence():
        player, move = node.get_move()
        if player is not None:
            moves.append([player.upper(), self._sgf_to_katago_coords(move, sgf_game.get_size())])
    return moves


def katago_to_sgf_coords(katago_move, board_size):
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


def sgf_to_katago_coords(sgf_move, board_size):
    """
    Converts SGF coordinates to KataGo coordinates.

    Args:
        sgf_move (tuple): The move in SGF coordinate format (row, col).
        board_size (int): The size of the board.

    Returns:
        str: The move in KataGo's coordinate format (e.g., 'A1', 'B2').
    """
    if sgf_move is None:
        return "pass"

    row, col = sgf_move
    col_str = chr(ord('A') + col)
    if col >= 8:
        col_str = chr(ord('A') + col + 1)
    row_str = str(board_size - row)
    return col_str + row_str
