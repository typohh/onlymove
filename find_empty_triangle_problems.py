import os
import sys
import math
import pandas as pd

# Add sgfmill to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'docs'))

import sgfmill.sgf
import sgfmill.boards

TSUMEGO_DIR = "tsumegos"

def get_descriptive_rank(numeric_rank):
    """
    Converts a numeric rank (0-1) to a descriptive rank (e.g., '10 kyu', '3 dan').
    Logic is ported from pwa/js/utils.js.
    """
    if numeric_rank < 0: numeric_rank = 0
    if numeric_rank > 1: numeric_rank = 1

    total_ranks = 29
    rank_index = math.floor(numeric_rank * total_ranks)

    if rank_index > 29:
        rank_index = 29

    if rank_index < 20:
        return f"{20 - rank_index} kyu"
    else:
        # This can result in "10 dan" if rank_index is 29, consistent with the JS code.
        return f"{rank_index - 19} dan"


def load_rating_data():
    """
    Loads and processes tsumego ratings and quality from CSV files,
    returning maps from tsumego_id to rank and quality.
    The rank calculation is ported from prepare_tsumegos.py.
    """
    # Load and process ratings
    tsumego_ratings_df = pd.read_csv("ratings/tsumego_ratings_final.csv")
    tsumego_ratings_df = tsumego_ratings_df.sort_values(by='rating', ascending=True).reset_index(drop=True)

    total = len(tsumego_ratings_df)
    bottom_1_count = int(0.01 * total)
    middle_end = int(0.90 * total)
    top_count = total - middle_end

    if bottom_1_count > 1:
        tsumego_ratings_df.loc[:bottom_1_count - 1, 'rating'] = (tsumego_ratings_df.index[:bottom_1_count] / (
                    bottom_1_count - 1)) * (1 / 29)
    elif bottom_1_count > 0:
        tsumego_ratings_df.loc[:bottom_1_count - 1, 'rating'] = 0.0

    min_middle = tsumego_ratings_df.loc[bottom_1_count:middle_end - 1, 'rating'].min()
    max_middle = tsumego_ratings_df.loc[bottom_1_count:middle_end - 1, 'rating'].max()
    if max_middle > min_middle:
        tsumego_ratings_df.loc[bottom_1_count:middle_end - 1, 'rating'] = (tsumego_ratings_df.loc[
                                                                    bottom_1_count:middle_end - 1, 'rating'] - min_middle) / (
                                                                                       max_middle - min_middle) * (
                                                                                       28 / 29 - 1 / 29) + 1 / 29
    else:
        tsumego_ratings_df.loc[bottom_1_count:middle_end - 1, 'rating'] = 1 / 29

    if top_count > 1:
        tsumego_ratings_df.loc[middle_end:, 'rating'] = ((tsumego_ratings_df.index[middle_end:] - middle_end) / (
                    top_count - 1)) * (1 - 28 / 29) + 28 / 29
    elif top_count > 0:
        tsumego_ratings_df.loc[middle_end:, 'rating'] = 28 / 29

    tsumego_ratings_map = dict(zip(tsumego_ratings_df['tsumego_id'], tsumego_ratings_df['rating']))

    # Load and process quality
    tsumego_quality_df = pd.read_csv("ratings/tsumego_quality_final.csv")
    min_quality = tsumego_quality_df['quality'].min()
    max_quality = tsumego_quality_df['quality'].max()
    if max_quality > min_quality:
        tsumego_quality_df['quality'] = (tsumego_quality_df['quality'] - min_quality) / (max_quality - min_quality)
    else:
        tsumego_quality_df['quality'] = 0.5  # Default value if all qualities are the same

    tsumego_quality_map = dict(zip(tsumego_quality_df['tsumego_id'], tsumego_quality_df['quality']))

    return tsumego_ratings_map, tsumego_quality_map


def get_branching_node(game):
    """
    Finds the branching node in the main variation.
    """
    current = game.get_root()
    while len(current) == 1:
        current = current[0]
    return current


def get_board_at_node(game, node):
    """
    Gets the board position at the given node.
    """
    board = sgfmill.boards.Board(9)
    root = game.get_root()
    # Apply setup stones
    if root.has_property('AB'):
        for stone in root.get('AB'):
            board.play(stone[0], stone[1], 'b')
    if root.has_property('AW'):
        for stone in root.get('AW'):
            board.play(stone[0], stone[1], 'w')
    # Replay moves to node
    path = []
    current = node
    while current != root:
        path.insert(0, current)
        current = current.parent
    for n in path:
        if n.has_property('B'):
            mv = n.get('B')
            if mv is not None:
                board.play(mv[0], mv[1], 'b')
        elif n.has_property('W'):
            mv = n.get('W')
            if mv is not None:
                board.play(mv[0], mv[1], 'w')
    return board


# Define transformations for all orientations
transforms = [
    lambda dx, dy: (dx, dy),  # 0 degrees
    lambda dx, dy: (dy, 2 - dx),  # 90 degrees
    lambda dx, dy: (2 - dx, 2 - dy),  # 180 degrees
    lambda dx, dy: (2 - dy, dx),  # 270 degrees
    lambda dx, dy: (dx, 2 - dy),  # reflect horizontal
    lambda dx, dy: (2 - dx, dy),  # reflect vertical
    lambda dx, dy: (dy, dx),  # reflect diagonal
    lambda dx, dy: (2 - dy, 2 - dx),  # reflect anti-diagonal
]

base_req_b = [(0, 0), (1, 0), (0, 1)]
base_req_e = [(1, 1)]
base_forb = [(0, 2), (1, 2), (2, 0), (2, 1)]
base_opt = [(2, 2)]

patterns = []
for f in transforms:
    req_b = [f(*p) for p in base_req_b]
    req_e = [f(*p) for p in base_req_e]
    forb = [f(*p) for p in base_forb]
    opt = [f(*p) for p in base_opt]
    patterns.append((req_b, req_e, forb, opt))


def get_liberty_count(board, x, y):
    color = board.board[x][y]
    if color is None:
        return 0
    visited = set()
    group = set()
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))
        group.add((cx, cy))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < board.side and 0 <= ny < board.side:
                if board.board[nx][ny] == color and (nx, ny) not in visited:
                    stack.append((nx, ny))
    liberties = set()
    for gx, gy in group:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < board.side and 0 <= ny < board.side and board.board[nx][ny] is None:
                liberties.add((nx, ny))
    return len(liberties)


def matches_pattern(board, tx, ty, req_b, req_e, forb):
    for dx, dy in req_b:
        x, y = tx + dx, ty + dy
        if not (0 <= x < 9 and 0 <= y < 9 and board.board[x][y] == 'b'):
            return False
    for dx, dy in req_e:
        x, y = tx + dx, ty + dy
        if not (0 <= x < 9 and 0 <= y < 9 and board.board[x][y] is None):
            return False
    for dx, dy in forb:
        x, y = tx + dx, ty + dy
        if 0 <= x < 9 and 0 <= y < 9 and board.board[x][y] == 'b':
            return False
    return True


def print_debug_info(tsumego_id, mx, my, tx, ty, orientation, req_b, req_e, forb, before_board, after_board):
    print(f"Detected empty triangle creation for {tsumego_id}")
    print(f"Move at {mx},{my}")
    print(f"At tx={tx}, ty={ty}, orientation={orientation}")
    print("Required blacks:", req_b)
    print("Required empty:", req_e)
    print("Forbidden:", forb)
    print("Before 3x3:")
    for i in range(3):
        row = ''
        for j in range(3):
            x, y = tx + i, ty + j
            if 0 <= x < 9 and 0 <= y < 9:
                stone = before_board.board[x][y]
                if stone == 'b': row += 'B '
                elif stone == 'w': row += 'W '
                else: row += '. '
            else:
                row += '? '
        print(row)
    print("After 3x3:")
    for i in range(3):
        row = ''
        for j in range(3):
            x, y = tx + i, ty + j
            if 0 <= x < 9 and 0 <= y < 9:
                stone = after_board.board[x][y]
                if stone == 'b': row += 'B '
                elif stone == 'w': row += 'W '
                else: row += '. '
            else:
                row += '? '
        print(row)


def find_and_print_empty_triangle_problems():
    """
    Goes through all SGF files, finds problems where the correct move creates an empty triangle,
    and prints their rank, quality, and a link.
    """
    if not os.path.isdir(TSUMEGO_DIR):
        print(f"Error: Directory '{TSUMEGO_DIR}' not found.", file=sys.stderr)
        return

    try:
        tsumego_ratings_map, tsumego_quality_map = load_rating_data()
    except FileNotFoundError as e:
        print(f"Error loading rating/quality data: {e}", file=sys.stderr)
        return

    for filename in sorted(os.listdir(TSUMEGO_DIR)):
        if not filename.endswith(".sgf"):
            continue

        file_path = os.path.join(TSUMEGO_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        game = sgfmill.sgf.Sgf_game.from_string(content)
        branching = get_branching_node(game)
        if len(branching) == 0:
            continue

        correct_node = branching[0]
        if not correct_node.has_property('B'):
            continue

        move = correct_node.get('B')
        before_board = get_board_at_node(game, branching)
        after_board = before_board.copy()
        captures = after_board.play(move[0], move[1], 'b')
        if captures:
            continue

        mx, my = move
        # Check if any adjacent black groups are in atari before the move
        adjacent_in_atari = False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ax, ay = mx + dx, my + dy
            if 0 <= ax < 9 and 0 <= ay < 9 and before_board.board[ax][ay] == 'b':
                if get_liberty_count(before_board, ax, ay) == 1:
                    adjacent_in_atari = True
                    break
        if adjacent_in_atari:
            continue
        found = False
        for tx in range(max(0, mx - 2), min(7, mx + 1)):
            for ty in range(max(0, my - 2), min(7, my + 1)):
                for req_b, req_e, forb, opt in patterns:
                    if matches_pattern(after_board, tx, ty, req_b, req_e, forb) and not matches_pattern(before_board, tx, ty, req_b, req_e, forb):
                        rel_x, rel_y = mx - tx, my - ty
                        if (rel_x, rel_y) in req_b:
                            found = True
                            break
                if found:
                    break
            if found:
                break

        if found:
            tsumego_id = filename[:-4]
            # print_debug_info(tsumego_id, mx, my, tx, ty, patterns.index((req_b, req_e, forb, opt)), req_b, req_e, forb, before_board, after_board)
            numeric_rank = tsumego_ratings_map.get(tsumego_id)
            quality = tsumego_quality_map.get(tsumego_id)

            if numeric_rank is not None and quality is not None:
                descriptive_rank = get_descriptive_rank(numeric_rank)
                print(
                    f"{descriptive_rank}, quality: {quality:.2f} - https://onlymove.net/?id={tsumego_id}")
            else:
                print(f"Rank/quality not found for {tsumego_id} - https://onlymove.net/?id={tsumego_id}")


if __name__ == "__main__":
    find_and_print_empty_triangle_problems()