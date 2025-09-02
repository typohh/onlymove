from configurations import BOARD_SIZE
from katago_queries import KatagoQueries, katago_to_sgf_coords
from sgfmill import sgf

PLAYOUTS_SHALLOW = 30
PLAYOUTS = 30
MIN_PLAYOUT_PERC = 0.01


def get_child_move(node: sgf.Tree_node, color, move: str, has_to_exist=False):
    """
    Find the child node with a given move. If no such node exists, create a new one.
    - node: The parent sgf.Tree_node.
    - color: 'b' or 'w' for the player making the move.
    - move: The move in KataGo coordinate format (e.g., "D4", "pass").
    """
    target_move = katago_to_sgf_coords(move, BOARD_SIZE)

    # Look for an existing child node with the same move
    for child in node:
        child_color, child_move = child.get_move()
        if child_color == color.lower() and child_move == target_move:
            return child

    if has_to_exist:
        raise Exception("could not find node!")

    # If no matching child is found, create a new one
    new_node = node.new_child()
    new_node.set_move(color.lower(), target_move)
    return new_node


def add_shallow_variations(queries: KatagoQueries, node: sgf.Tree_node, moves):
    """
    Generate shallow variations for all legal moves from the current position and their optimal responses.
    This function modifies the SGF game object in place.
    """

    # Traverse the game tree to the node where variations should be attached.

    # Now, determine whose turn it is to play from the current position.
    player = "b" if (len(moves) % 2 == 0) else "w"

    # Get all legal moves from the current position
    legal_moves = queries.get_legal_moves(moves)

    # For each legal move, get its evaluation and optimal response
    for move in legal_moves:
        # Get the optimal response to this move
        response_moves = moves + [move]
        response_result = queries.query(response_moves, playouts=PLAYOUTS_SHALLOW)
        response_move, winrate, score, _ = queries.get_best_move(response_result)

        # Add variation to SGF
        var_node = get_child_move(node, player, move)
        var_node.set("C", f"{winrate * 100:.1f}%/{score:.1f}")

        response_player = "b" if player == "w" else "w"
        response_node = get_child_move(var_node, response_player, response_move)
        response_node.set("C", f"{winrate * 100:.1f}%/{score:.1f}")

    return


def generate_variations_directly_to_sgf(
        queries: KatagoQueries,
        parent_sgf_node: sgf.Tree_node,
        current_moves: list,
        cumulative_playout_perc: float,
        exclude_move=None
):
    """
    Recursively generates variations and adds them directly as child nodes to the parent_sgf_node.
    'is_first_level_branch_node' indicates if the node being added is the first move in a new variation branch.
    """
    if cumulative_playout_perc < MIN_PLAYOUT_PERC:
        return

    current_pos_analysis = queries.query(current_moves, playouts=PLAYOUTS, exclude_move=exclude_move)
    if not current_pos_analysis or "moveInfos" not in current_pos_analysis:
        return

    total_visits = sum(info.get("visits", 0) for info in current_pos_analysis["moveInfos"])
    if total_visits == 0:
        return

    sorted_move_infos = sorted(current_pos_analysis["moveInfos"], key=lambda x: x.get("visits", 0), reverse=True)

    for move_info in sorted_move_infos:
        visits = move_info.get("visits", 0)
        if visits == 0:
            continue

        current_branch_perc = visits / total_visits
        new_cumulative_perc = cumulative_playout_perc * current_branch_perc

        # Create a new child node for this variation
        var_node = get_child_move(parent_sgf_node, "b" if (len(current_moves) % 2 == 0) else "w", move_info["move"])

        comment = f"{move_info['winrate'] * 100:.1f}%/{move_info['scoreLead']:.1f}"
        var_node.set("C", comment)

        # Recursively add deeper variations from this new node
        generate_variations_directly_to_sgf(
            queries,
            var_node,  # The new child node becomes the parent for the next level
            current_moves + [move_info["move"]],  # Extend the move list for deeper analysis
            new_cumulative_perc
        )
