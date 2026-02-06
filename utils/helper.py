from sgfmill import sgf

from utils.configurations import BOARD_SIZE, KOMI, RULES
from utils.katago_queries import katago_to_sgf_coords


def construct_initial_sgf(moves, solution):
    game = sgf.Sgf_game(size=BOARD_SIZE)
    game.root.set("KM", KOMI)
    game.root.set("RU", RULES)
    player = "b"
    node = None
    for move in moves:
        node = game.extend_main_sequence()
        node.set_move(player, katago_to_sgf_coords(move, BOARD_SIZE))
        player = "w" if player == "b" else "b"
    if solution:
        answer = game.extend_main_sequence()
        answer.set_move(player, katago_to_sgf_coords(solution, BOARD_SIZE))
    else:
        answer = None
    return game, node, answer
