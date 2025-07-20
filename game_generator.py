import random
from katago_queries import KatagoQueries, sgf_to_katago_coords, katago_to_sgf_coords
from sgfmill import sgf

def generate_game(queries: KatagoQueries):
    """
    Generates a single game by starting with random moves and then balancing the game.

    Returns:
        list: A list of moves representing the game sequence.
    """
    moves = []

    # 1. Play 4 random moves
    for _ in range(4):
        response = queries.query_policy(moves)
        # Filter out pass moves and moves with very low policy values
        legal_moves = [
            (i, p)
            for i, p in enumerate(response["policy"])
            if p > 1e-5 and queries.get_move_from_idx(i) != "pass"
        ]

        # Weighted random sampling based on policy
        move_indices, probabilities = zip(*legal_moves)
        total_prob = sum(probabilities)
        normalized_probs = [p / total_prob for p in probabilities]
        chosen_idx = random.choices(move_indices, weights=normalized_probs, k=1)[0]
        move = queries.get_move_from_idx(chosen_idx)

        moves.append(move)

    # 2. Play 2 moves to balance the game
    for _ in range(2):
        response = queries.query_children_low(moves)
        if not response.get("moveInfos"):
            break  # No further moves possible

        # Find the move that brings the score closest to 0
        best_move = min(
            response["moveInfos"],
            key=lambda x: abs(x["scoreLead"]) * abs(x["winrate"]-0.5),
        )["move"]

    # 3. Continue with normal game generation
    consecutive_passes = 0
    while consecutive_passes < 2:
        response = queries.query_policy(moves)
        best_move,_,_ = queries.get_best_move(response)

        if best_move == "pass":
            consecutive_passes += 1
        else:
            consecutive_passes = 0

        moves.append(best_move)

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

    player = "b"

    for move in moves:
        node = game.extend_main_sequence()
        node.set_move(player, katago_to_sgf_coords(move, board_size))
        player = "w" if player == "b" else "b"

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
            moves.append(sgf_to_katago_coords(move, board_size))
    return moves
