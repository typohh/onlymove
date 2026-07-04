from sgfmill import sgf

from utils.configurations import KOMI, RULES
from utils.coords import gtp_to_sgfmill_point


def construct_initial_sgf(moves, solution, board_size):
    game = sgf.Sgf_game(size=board_size)
    game.root.set("KM", KOMI)
    game.root.set("RU", RULES)
    player = "b"
    node = None
    for move in moves:
        node = game.extend_main_sequence()
        node.set_move(player, gtp_to_sgfmill_point(move, board_size))
        player = "w" if player == "b" else "b"
    if solution:
        answer = game.extend_main_sequence()
        answer.set_move(player, gtp_to_sgfmill_point(solution, board_size))
    else:
        answer = None
    return game, node, answer
