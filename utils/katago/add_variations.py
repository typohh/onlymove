"""SGF tree construction: build a problem's initial SGF from its move sequence,
then hang the analysed variations off it.

construct_initial_sgf lived in utils/helper.py — a one-function module whose name
said nothing about its contents — and is the step that produces the tree the rest
of this module annotates, so it belongs beside it.
"""
import re

from sgfmill import sgf

from utils.constants import KOMI, RULES
from utils.katago.coords import gtp_to_sgfmill_point
from utils.katago.queries import KatagoQueries


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


# Parses a variation comment written as "{winrate}%/{score}" (e.g. "60.6%/-0.5")
# back into its two numbers. Both are stored from BLACK's perspective
# (bots/analysis.cfg sets reportAnalysisWinratesAs = BLACK), matching the format
# set on every var_node's "C" property below.
_EVAL_COMMENT_RE = re.compile(r"(-?[\d.]+)%/(-?[\d.]+)")


def parse_eval_comment(comment):
    """(winrate, score) from a "{winrate}%/{score}" comment, or None if unparseable."""
    if not comment:
        return None
    m = _EVAL_COMMENT_RE.search(comment)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _candidate_evals(node):
    """[(child, mover_winrate, mover_score)] for every candidate first move under
    `node` that carries a parseable evaluation comment, in variation order.

    Winrate/score comments are black-perspective, so they are flipped when White
    is the side to move. Moves without a parseable comment (and property-only
    nodes) are dropped, matching the corpus audit that defined this invariant.
    """
    out = []
    for child in node:
        colour, _ = child.get_move()
        if colour is None:  # pass or a property-only node
            continue
        comment = child.get("C") if child.has_property("C") else None
        ev = parse_eval_comment(comment)
        if ev is None:
            continue
        winrate_black, score_black = ev
        black_to_move = colour == "b"
        out.append((child,
                    winrate_black if black_to_move else 100.0 - winrate_black,
                    score_black if black_to_move else -score_black))
    return out


def first_move_is_well_formed(node):
    """True iff the problem's first-move decision has exactly ONE canonical answer.

    Among the candidate first moves (the children of the problem position `node`
    that carry an evaluation comment), the position must have exactly one clean
    WIN for the side to move — winrate > 50% AND score > 0 from the mover's
    perspective — and every other candidate must be a clean LOSS (winrate < 50%
    AND score < 0). Anything else is discarded:
      - two or more winners  → the problem has multiple solutions,
      - zero winners         → seki / half-point / dame (no move both-positive),
      - a winrate/score disagreement on any move (e.g. 60.6%/-0.5) → ambiguous.

    This says nothing about WHERE the winner sits in the variation list; that is
    promote_winning_first_move's job. Keeping the two separate is deliberate —
    a problem whose solution is merely listed late is perfectly well-posed and
    should be reordered, not discarded.
    """
    if node is None:
        return False
    winners = 0
    ambiguous = 0
    for _child, mover_winrate, mover_score in _candidate_evals(node):
        if mover_winrate > 50 and mover_score > 0:
            winners += 1
        elif mover_winrate < 50 and mover_score < 0:
            pass  # clean loss
        else:
            ambiguous += 1
    return winners == 1 and ambiguous == 0


def promote_winning_first_move(node):
    """Reorder `node`'s children so the winning first move is variation index 0.

    The corpus convention is that variation_nodes[0] IS the solution, and every
    backend consumer relies on it — pipeline.extract_activations takes its
    correct_idx from it, pipeline.benchmark_bots grades against it. The generator
    can break that convention: the solution it writes first is the move
    ProblemFinder's is_valid re-read verified, while the candidates' comments are
    written afterwards by their own passes at their own playout budgets, so when
    the two disagree the first variation is a move the SHIPPED evaluations call a
    loss. The client has never trusted the position (pwa/js/sgf.js
    findBestMoveFromChildren picks by evaluation), so the file is reordered here
    to agree with it rather than discarding an otherwise well-posed problem.

    Selection is the client's rule: highest winrate for the side to move, score
    breaking exact ties, first-listed winning an exact tie on both. Returns True
    if a node was moved.
    """
    if node is None:
        return False
    candidates = _candidate_evals(node)
    if not candidates:
        return False
    # max() returns the first maximal element, matching the client's strict-`>`
    # scan, so an exact tie on both axes leaves the earlier variation in front.
    best = max(candidates, key=lambda c: (c[1], c[2]))[0]
    if best is node[0]:
        return False
    best.reparent(node, 0)
    return True


PLAYOUTS_SHALLOW = 30
PLAYOUTS = 30
MIN_PLAYOUT_PERC = 0.01

# Stay well under Python's default recursion limit (1000). KataGo under Japanese
# rules will sometimes refuse to pass in a settled position; this guard prevents
# the primary line from blowing the stack.
MAX_VARIATION_DEPTH = 800


class VariationDepthExceeded(Exception):
    """Raised when generate_variations_directly_to_sgf goes too deep without terminating."""
    def __init__(self, moves):
        self.moves = list(moves)
        super().__init__(f"variation depth {len(self.moves)} exceeded; sequence: {self.moves}")

# Utility reweighting that makes KataGo prefer preserving the score over flipping
# the win/loss outcome. Used when generating the post-solution continuation so the
# losing side plays calm capitulation moves instead of swindle attempts.
CALM_UTILITY_OVERRIDE = {
    "winLossUtilityFactor": 0.1,
    "staticScoreUtilityFactor": 0.5,
    "dynamicScoreUtilityFactor": 0.5,
}


def get_child_move(node: sgf.Tree_node, color, move: str, board_size, has_to_exist=False):
    """
    Find the child node with a given move. If no such node exists, create a new one.
    - node: The parent sgf.Tree_node.
    - color: 'b' or 'w' for the player making the move.
    - move: The move in KataGo coordinate format (e.g., "D4", "pass").
    - board_size: Board size, for KataGo→SGF coordinate conversion.
    """
    target_move = gtp_to_sgfmill_point(move, board_size)

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
        var_node = get_child_move(node, player, move, queries.board_size)
        var_node.set("C", f"{winrate * 100:.1f}%/{score:.1f}")

        response_player = "b" if player == "w" else "w"
        response_node = get_child_move(var_node, response_player, response_move, queries.board_size)
        response_node.set("C", f"{winrate * 100:.1f}%/{score:.1f}")

    return


def generate_variations_directly_to_sgf(
        queries: KatagoQueries,
        parent_sgf_node: sgf.Tree_node,
        current_moves: list,
        cumulative_playout_perc: float,
        is_primary: bool,
        exclude_move=None,
        override_settings=None,
        mark_children=True,
):
    """
    Recursively generates variations and adds them directly as child nodes to the parent_sgf_node.
    'is_first_level_branch_node' indicates if the node being added is the first move in a new variation branch.

    override_settings is forwarded to KataGo (e.g. utility reweighting for the calm post-solution line).
    Existing comments are left untouched so the first pass's numbers win on nodes shared between passes.

    Returns True when some line at or under this node was read all the way to a
    DOUBLE PASS, and marks every child that was, with `TE[1]`.

    That mark is the whole point: it records which branches this function
    actually finished, so no consumer has to guess afterwards. Guessing is what
    the client used to do — it treated "a pass appears somewhere in the subtree"
    as "this line was pursued to a conclusion", which is false for every branch
    abandoned at the MIN_PLAYOUT_PERC cutoff below, and let it walk a solver
    into a lost position. Only lines that reach the double pass are marked; the
    abandoned ones are left bare precisely so they can be told apart.

    mark_children=False marks nothing at THIS level while still marking deeper
    ones, and the caller that builds the refuted first moves passes it. A mark
    on a decision-node child reads as "this is the move to find", so a refuted
    first move must never carry one: were the answer's own lines to fail to
    resolve, the client would find only refuted moves marked and offer one of
    them as the solution. The answer is marked by the caller instead.
    """
    if len(current_moves) >= MAX_VARIATION_DEPTH:
        raise VariationDepthExceeded(current_moves)

    if current_moves[-1] == 'pass' and current_moves[-2] == 'pass':
        return True

    if cumulative_playout_perc < MIN_PLAYOUT_PERC and not is_primary:
        return False

    current_pos_analysis = queries.query(
        current_moves,
        playouts=PLAYOUTS,
        exclude_move=exclude_move,
        override_settings=override_settings,
    )
    if not current_pos_analysis or "moveInfos" not in current_pos_analysis:
        return False

    total_visits = sum(info.get("visits", 0) for info in current_pos_analysis["moveInfos"])
    if total_visits == 0:
        return False

    reached_end = False

    for move_info in current_pos_analysis["moveInfos"]:
        visits = move_info.get("visits", 0)
        if visits == 0:
            continue

        current_branch_perc = visits / total_visits
        new_cumulative_perc = cumulative_playout_perc * current_branch_perc

        # Create a new child node for this variation
        var_node = get_child_move(parent_sgf_node, "b" if (len(current_moves) % 2 == 0) else "w", move_info["move"], queries.board_size)

        if not var_node.has_property("C"):
            var_node.set("C", f"{move_info['winrate'] * 100:.1f}%/{move_info['scoreLead']:.1f}")

        # Recursively add deeper variations from this new node
        child_reached_end = generate_variations_directly_to_sgf(
            queries,
            var_node,  # The new child node becomes the parent for the next level
            current_moves + [move_info["move"]],  # Extend the move list for deeper analysis
            new_cumulative_perc,
            is_primary,
            override_settings=override_settings,
        )

        if child_reached_end:
            if mark_children:
                var_node.set("TE", 1)
            reached_end = True

        is_primary = False

        if cumulative_playout_perc < MIN_PLAYOUT_PERC:
            break

    return reached_end
