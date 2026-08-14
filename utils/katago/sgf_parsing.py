"""SGF helpers shared by the pipeline stages that read the tsumego corpus.

They live in utils because three stages need them — activation extraction, the
move-eval cache and packaging — and none of those is driving the bot benchmark
that first defined them. A pipeline stage importing another pipeline stage for a
parser couples two schedules that have nothing else to do with each other.

The coordinate conversion `sgf_to_gtp` is NOT here — it already lives in
`utils.katago.coords`, and `pipeline.benchmark_bots` was only re-exporting it.
Import it from coords directly.
"""

import sgfmill.sgf

from utils.katago.coords import sgf_to_gtp, sgfmill_point_to_sgf


def parse_tsumego_sgf(game: sgfmill.sgf.Sgf_game):
    """
    Correctly parses a tsumego SGF file by finding the node where the path branches.
    Uses the correct, list-like API for sgfmill Tree_node objects.
    """
    setup_moves = []
    root_node = game.get_root()
    size = game.get_size()

    for prop, color_char in [("AB", "B"), ("AW", "W")]:
        if root_node.has_property(prop):
            for point_tuple in root_node.get(prop):
                sgf_string = sgfmill_point_to_sgf(point_tuple, size)
                setup_moves.append((color_char, sgf_to_gtp(sgf_string, size)))

    current_node = root_node
    while len(current_node) == 1:
        if current_node.has_property("B") or current_node.has_property("W"):
            color, move_bytes = current_node.get_raw_move()
            move_sgf = move_bytes.decode('utf-8')
            setup_moves.append((color, sgf_to_gtp(move_sgf, size)))
        current_node = current_node[0]

    if current_node.has_property("B") or current_node.has_property("W"):
        color, move_bytes = current_node.get_raw_move()
        move_sgf = move_bytes.decode('utf-8')
        setup_moves.append((color, sgf_to_gtp(move_sgf, size)))

    variation_nodes = list(current_node)
    if not variation_nodes:
        raise ValueError("SGF file is linear and contains no variations to analyze.")

    return setup_moves, variation_nodes
