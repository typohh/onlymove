from sgfmill import sgf

# This is a temporary script to debug the problem_to_sgf_with_variations function.

APP_NAME = "test"
APP_VERSION = "1.0"

# Copied from add_variations.py
def katago_str_to_sgf_tuple(katago_move_str, board_size=9):
    """Converts a KataGo coordinate string (eg 'D4') to an SGF (row, col) tuple."""
    if katago_move_str.lower() == "pass":
        return None
    col_char = katago_move_str[0].upper()
    row_str = katago_move_str[1:]
    col = ord(col_char) - ord('A')
    if col_char > 'H':  # Skip 'I' in Go coordinates
        col -= 1
    row = board_size - int(row_str)
    return (row, col)

def raw_sgf_to_tuple(raw_move_bytes):
    """Converts raw SGF coordinate bytes (e.g., b'pd') to a (row, col) tuple."""
    if not raw_move_bytes:
        return None  # Pass move
    col = ord(raw_move_bytes[0:1].decode('ascii')) - ord('a')
    row = ord(raw_move_bytes[1:2].decode('ascii')) - ord('a')
    return (row, col)

# Copied from add_variations.py
def add_variations_to_node(node, variations, player_code):
    """Recursively adds variations to an SGF node."""
    for variation in variations:
        var_player_code = player_code
        var_node = node.new_child()
        var_node.set_move(var_player_code, katago_str_to_sgf_tuple(variation["move"]))
        var_node.set("C", f"Playouts: {variation['playout_perc']:.1f}%, Winrate: {variation['winrate']:.1f}%, Score: {variation['score']:.1f}")
        if "variations" in variation and variation["variations"]:
            next_player_code = "w" if var_player_code == "b" else "b"
            add_variations_to_node(var_node, variation["variations"], next_player_code)

# This is the function under test
def problem_to_sgf_with_variations(problem):
    """
    Saves a problem with deep variations as an SGF file.
    """
    game = sgf.Sgf_game(size=9)
    game.root.set("KM", "6.5")
    game.root.set("RU", "japanese")
    game.root.set("AP", (APP_NAME, APP_VERSION))

    node = game.root
    player_code = 'b'
    for move in problem["moves"]:
        new_node = node.new_child()
        
        # The conversion and write
        sgf_tuple = katago_str_to_sgf_tuple(move)
        new_node.set_move(player_code, sgf_tuple)
        
        # The read-after-write verification
        read_back_color, read_back_raw = new_node.get_raw_move()
        read_back_tuple = raw_sgf_to_tuple(read_back_raw)
        
        print(f"Move: {move} | Sent to set_move: {sgf_tuple} | Read back from get_raw_move: {read_back_tuple}")

        node = new_node
        player_code = 'w' if player_code == 'b' else 'b'

    if problem.get("variations"):
        add_variations_to_node(node, problem["variations"], player_code)

    return game.serialise().decode()

def main():
    # Test with a simple, valid sequence from the actual data.
    test_moves = ['G5', 'G6']
    problem = {
        "moves": test_moves,
        "variations": []
    }
    
    sgf_output = problem_to_sgf_with_variations(problem)
    print(sgf_output)

if __name__ == "__main__":
    main()
