from katago_queries import KatagoQueries, sgf_to_katago_coords, katago_to_sgf_coords
from sgfmill import sgf

def generate_game(queries:KatagoQueries):
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
        response = queries.query_root(moves)
        if response and "policy" in response and response["policy"]:
            best_move_idx = response['policy'].index(max(response['policy']))
            best_move = queries.get_move_from_idx(best_move_idx)
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

def save_game_sgf(moves, filename, board_size=9, komi=7.5):
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
        node.set_move(player.lower(), katago_to_sgf_coords(move, board_size))

    with open(filename, "wb") as f:
        f.write(game.serialise())


def load_game_sgf(filename):
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
    board_size = sgf_game.get_size()
    for node in sgf_game.get_main_sequence():
        player, move = node.get_move()
        if player is not None:
            moves.append([player.upper(), sgf_to_katago_coords(move, board_size)])
    return moves
