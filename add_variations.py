import glob
import os
import traceback

from sgfmill import sgf

# Assuming these are available in the same context
# You might need to adjust imports based on your project structure
from katago_controller import KatagoController
from katago_queries import KatagoQueries, katago_to_sgf_coords, sgf_to_katago_coords
# Removed _get_player_to_move_next from import as we are replacing its logic here
from models import MODEL_9X9, get_model

# Testing values
PLAYOUTS = 10
MIN_PLAYOUT_PERC = 0.1  # Minimum playout percentage for a branch to be explored

APP_NAME = "add_variations.py"
APP_VERSION = "5.1"


# Helper function for determining the next player based on a move list and the game's initial player
def _get_player_for_next_move(moves_played: list, game_starting_player: str) -> str:
    """
    Determines the player whose turn it is for the next move based on the number of
    moves already played and the actual starting player of the game.
    `moves_played` is a list of KataGo coordinates.
    `game_starting_player` is 'b' or 'w'.
    """
    num_moves = len(moves_played)

    if game_starting_player == 'b':
        # If Black starts: 0 moves -> B, 1 move -> W, 2 moves -> B, etc.
        return 'b' if num_moves % 2 == 0 else 'w'
    else:  # game_starting_player == 'w'
        # If White starts: 0 moves -> W, 1 move -> B, 2 moves -> W, etc.
        return 'w' if num_moves % 2 == 0 else 'b'


def generate_variations_directly_to_sgf(
        queries: KatagoQueries,
        parent_sgf_node,
        current_moves: list,
        cumulative_playout_perc: float,
        sgf_board_size: int,
        is_first_level_branch_node: bool,  # Indicates if this is the initial node of a new branch
        game_initial_player: str
):
    """
    Recursively generates variations and adds them directly as child nodes to the parent_sgf_node.
    'is_first_level_branch_node' indicates if the node being added is the first move in a new variation branch.
    """
    if cumulative_playout_perc < MIN_PLAYOUT_PERC:
        return

    current_pos_analysis = queries.query(current_moves, playouts=PLAYOUTS)
    if not current_pos_analysis or "moveInfos" not in current_pos_analysis:
        return

    total_visits = sum(info.get("visits", 0) for info in current_pos_analysis["moveInfos"])
    if total_visits == 0:
        return

    sorted_move_infos = sorted(current_pos_analysis["moveInfos"], key=lambda x: x.get("visits", 0), reverse=True)

    # Determine the player for the move to be added *to* the variation node using our helper
    player_to_move_next = _get_player_for_next_move(current_moves, game_initial_player)

    for move_info in sorted_move_infos:
        visits = move_info.get("visits", 0)
        if visits == 0:
            continue

        current_branch_perc = visits / total_visits
        new_cumulative_perc = cumulative_playout_perc * current_branch_perc

        # Create a new child node for this variation
        var_node = parent_sgf_node.new_child()
        var_node.set_move(player_to_move_next, katago_to_sgf_coords(move_info["move"], sgf_board_size))

        comment = f"Winrate: {move_info['winrate'] * 100:.1f}%, Score: {move_info['scoreLead']:.1f}"
        var_node.set("C", comment)

        # Recursively add deeper variations from this new node
        generate_variations_directly_to_sgf(
            queries,
            var_node,  # The new child node becomes the parent for the next level
            current_moves + [move_info["move"]],  # Extend the move list for deeper analysis
            new_cumulative_perc,
            sgf_board_size,
            is_first_level_branch_node=False,  # Subsequent variations are always not first-level branches
            game_initial_player=game_initial_player
        )


def add_variations_to_existing_sgf_direct(game: sgf.Sgf_game, queries: KatagoQueries):
    """
    Adds variations directly to an existing SGF game, preserving all original data.
    This function modifies the SGF game object in place.
    """
    root_node = game.root
    sgf_board_size = game.size

    # Update AP property (Application Property)
    new_ap_entry = (APP_NAME, APP_VERSION)
    current_ap = None
    if root_node.has_property("AP"):
        current_ap = root_node.get("AP")

    if current_ap is None:
        root_node.set("AP", new_ap_entry)
    elif isinstance(current_ap, tuple):
        if current_ap[0] == APP_NAME:
            root_node.set("AP", new_ap_entry)  # Update version if already our app
        else:
            # Convert to list and add ours. Ensure existing tuple is preserved in list.
            root_node.set("AP", [current_ap, new_ap_entry])
    elif isinstance(current_ap, list):
        found = False
        updated_list = []
        for item in current_ap:
            if isinstance(item, (tuple, list)) and item[0] == APP_NAME:
                updated_list.append(new_ap_entry)  # Update version
                found = True
            else:
                updated_list.append(item)
        if not found:
            updated_list.append(new_ap_entry)  # Add if not found
        game.root.set("AP", updated_list)
    else:  # Fallback for other unexpected types, just overwrite
        root_node.set("AP", new_ap_entry)

    # Get the moves in the main sequence up to the last node
    main_sequence_moves = []
    # Iterate through the main sequence to get all moves played so far
    for node in game.get_main_sequence():
        move_prop = node.get_move()
        if move_prop and move_prop[0] is not None:
            _color, move_tuple = move_prop
            # Convert SGF coordinates to KataGo coordinates
            main_sequence_moves.append(sgf_to_katago_coords(move_tuple, sgf_board_size))

    # The point where variations will branch from is the last node in the main sequence
    parent_node_for_variations = game.get_last_node()

    # Determine the initial player for the entire game from the 'PL' property
    initial_player_for_game = "b"  # Default to black if 'PL' is not present, as per SGF standard
    if game.root.has_property("PL"):
        initial_player_for_game = game.root.get("PL").lower()  # SGF uses 'B'/'W', convert to 'b'/'w'

    # Perform a root analysis to get top-level variations for the current position (after main sequence)
    root_analysis = queries.query(main_sequence_moves, playouts=PLAYOUTS)
    if not root_analysis or "moveInfos" not in root_analysis:
        return game.serialise().decode()

    sorted_move_infos = sorted(root_analysis["moveInfos"], key=lambda x: x.get("visits", 0), reverse=True)
    top_3_move_infos = sorted_move_infos[:3]  # Consider only top 3 moves for initial branches

    for move_info in top_3_move_infos:
        # For each top move, create a new child variation branch
        # Determine the player for this top move using our helper function
        player_for_this_top_move = _get_player_for_next_move(main_sequence_moves, initial_player_for_game)

        var_node = parent_node_for_variations.new_child()
        var_node.set_move(player_for_this_top_move, katago_to_sgf_coords(move_info["move"], sgf_board_size))

        comment = f"Winrate: {move_info['winrate'] * 100:.1f}%, Score: {move_info['scoreLead']:.1f}"
        # For the very top-level variations directly branching from the main line, we don't include playout percentage in their comments.
        var_node.set("C", comment)

        # Recursively generate deeper variations from this first level node
        generate_variations_directly_to_sgf(
            queries,
            var_node,  # The new child node is now the parent for deeper variations
            main_sequence_moves + [move_info["move"]],  # The move list extends to include this variation move
            1.0,  # Initial cumulative playout percentage for this specific branch
            sgf_board_size,
            is_first_level_branch_node=True,  # This node is the first move in a new top-level branch
            game_initial_player=initial_player_for_game
        )

    return game.serialise().decode()


def simple_sgf_to_moves(sgf_content):
    """
    Converts SGF file content to a list of moves in the main variation.
    Also checks if the SGF already contains variations added by this app based on the AP property.
    Returns (moves, already_processed_by_app_flag).
    """
    game = sgf.Sgf_game.from_string(sgf_content)

    # Check for our APP_NAME in the AP property to avoid reprocessing
    if game.root.has_property("AP"):
        ap_value = game.root.get("AP")
        if isinstance(ap_value, tuple) and ap_value[0] == APP_NAME:
            return None, True  # Already processed
        elif isinstance(ap_value, list):
            for item in ap_value:
                if isinstance(item, (tuple, list)) and item[0] == APP_NAME:
                    return None, True  # Already processed

    moves = []
    # sgfmill's game.get_main_sequence() iterates through the nodes in the main branch
    for node in game.get_main_sequence():
        move_prop = node.get_move()
        if move_prop and move_prop[0] is not None:
            _color, move_tuple = move_prop
            # Convert SGF coordinates to KataGo coordinates
            moves.append(sgf_to_katago_coords(move_tuple, game.size))

    return moves, False


def main():
    controller = None
    try:
        print("Starting KataGo engine...")
        controller = KatagoController()
        controller.start_engine(get_model(MODEL_9X9))
        print("KataGo engine started.")

        queries = KatagoQueries(controller)

        tsumegos_dir = os.path.join(os.path.dirname(__file__), "tsumegos")
        # Ensure the 'tsumegos' directory exists or handle its absence
        if not os.path.exists(tsumegos_dir):
            print(f"Error: 'tsumegos' directory not found at {tsumegos_dir}. Please create it and add SGF files.")
            return

        sgf_files = glob.glob(os.path.join(tsumegos_dir, "*.sgf"))

        if not sgf_files:
            print(f"No SGF files found in {tsumegos_dir}. Please add SGF files to the directory.")
            return

        for sgf_file in sgf_files:
            print(f"Processing {sgf_file}...")

            with open(sgf_file, "r", encoding='utf-8', errors='ignore') as f:
                sgf_content = f.read()

            _temp_moves_for_check, already_processed = simple_sgf_to_moves(sgf_content)

            if already_processed:
                print("  Skipping (already has variations from this app).")
                continue

            print("  Loading SGF game object...")
            game = sgf.Sgf_game.from_string(sgf_content)

            print("  Generating variations and adding directly to SGF...")
            new_sgf_content = add_variations_to_existing_sgf_direct(game, queries)

            print("\n--- Original SGF ---")
            print(sgf_content)
            print("\n--- New SGF with Variations ---")
            print(new_sgf_content)
            exit()

            with open(sgf_file, "w", encoding='utf-8') as f:
                f.write(new_sgf_content)
            print(f"  Successfully updated {sgf_file}")

        print("\nNo new SGF files were processed (either none or all already had variations).")


    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if controller:
            print("Stopping KataGo engine...")
            controller.stop_engine()
            print("KataGo engine stopped.")


if __name__ == "__main__":
    main()