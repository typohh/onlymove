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


# Parses the first two fields of a variation comment written as
# "{winrate}%/{score}/{visits}" (e.g. "60.6%/-0.5/1000") back into their numbers;
# the third field is read separately by parse_eval_visits. Both are stored from
# BLACK's perspective
# (bots/analysis.cfg sets reportAnalysisWinratesAs = BLACK), matching the format
# set on every var_node's "C" property below.
_EVAL_COMMENT_RE = re.compile(r"(-?[\d.]+)%/(-?[\d.]+)")


def parse_eval_comment(comment):
    """(winrate, score) from a "{winrate}%/{score}/{visits}" comment, or None if
    unparseable. UNANCHORED, so a comment carrying text before the fields still
    reads, and the trailing visits field simply falls outside the match."""
    if not comment:
        return None
    m = _EVAL_COMMENT_RE.search(comment)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


# A THIRD field records the VISIT COUNT the estimate rests on: "96.4%/1.3/1000".
# It exists because the first two numbers alone cannot say how much search is
# behind them, which is how a corpus-wide 30-visit annotation went unnoticed —
# every value looked exactly like a deep one.
#
# Appending rather than restructuring is deliberate: every parser in the repo
# (pipeline/benchmark_bots, pipeline/cache_rating_features, utils/rating/features,
# pwa/js/sgf.js) matches the first two fields with an UNANCHORED search, so a
# comment written before this field existed still reads identically.
_EVAL_VISITS_RE = re.compile(r"-?[\d.]+%/-?[\d.]+/(\d+)")


def format_eval_comment(winrate, score, visits):
    """The "C" property for a node evaluated at `visits`, e.g. "96.4%/1.3/1000"."""
    return f"{winrate * 100:.1f}%/{score:.1f}/{int(visits)}"


def parse_eval_visits(comment):
    """How many visits back a comment's estimate.

    A comment carrying no third field predates the field, and nothing has ever
    been written below the 30-visit floor, so it reports PLAYOUTS_SHALLOW — that
    is a fact about the corpus, not a fallback. Anything that is not an
    evaluation at all reports 0, so any real evaluation outranks it."""
    if not comment:
        return 0
    m = _EVAL_VISITS_RE.search(comment)
    if m:
        return int(m.group(1))
    return PLAYOUTS_SHALLOW if _EVAL_COMMENT_RE.search(comment) else 0


def _is_pass(node):
    """A pass node: sgfmill gives (colour, None) for a pass and (None, None) for
    a node carrying no move at all, so the COLOUR is what tells them apart."""
    colour, point = node.get_move()
    return colour is not None and point is None


def subtree_reaches_conclusion(node):
    """True when THIS branch is one the writer played out to a DOUBLE PASS.

    Follows `children[0]` — the writer appends children in `moveInfos` order and
    always recurses into the first one, so the first child IS the continuation it
    followed — and asks whether that chain ends in two consecutive passes.
    Measured on rebuilt 13x13 files, it holds for 120/120 answers and 120/120
    best refutations (the decision node's `children[1]`), and rejects the side
    branches, which stop at the visit floor.

    Two cheaper-looking tests are both wrong, and were measured to be:

      * "a pass-pair appears anywhere in the subtree" over-accepts, because
        add_shallow_variations writes a move AND its best response, so a pass
        answered by a pass manufactures the shape without anything being played
        out. Against the old TE marks it admitted 32,445 extra children.
      * "follow the highest-visit child" drifts off the line entirely. KataGo
        ranks moveInfos by LCB, not raw visits, so the followed move is not the
        highest-visit one at 20.8% of multi-child nodes, and that walk found only
        42.5% of the answer lines this one finds.

    NOTE what this does NOT say: that the line resolves the way the problem
    claims. 7.5% of principal lines end with the SOLVER LOSING — a tsumego whose
    answer is +4.6 at 1000 visits can drift to -27.0 over a 108-ply tail played
    at the 30-visit floor. Whether a resolved line is a WIN is a separate
    question, answered by the evaluation the terminal node already carries.
    """
    n = node
    prev_was_pass = _is_pass(node)
    for _ in range(MAX_VARIATION_DEPTH + 2):
        kids = [c for c in n if c.get_move()[0] is not None]
        if not kids:
            return False
        n = kids[0]
        is_pass = _is_pass(n)
        if is_pass and prev_was_pass:
            return True
        prev_was_pass = is_pass
    return False


def set_eval_comment(node, winrate, score, visits):
    """Write an evaluation onto `node`, but NEVER DOWNGRADE ONE.

    A comment is replaced only by an estimate resting on MORE visits than the one
    already there. That single rule replaces the pass-ordering the module used to
    depend on: the deep passes overwrite the shallow values they find, the shallow
    pass cannot undo them whatever order it runs in, and re-running any pass over
    its own output is a no-op rather than a coin flip. Returns True if it wrote."""
    existing = node.get("C") if node.has_property("C") else None
    if int(visits) <= parse_eval_visits(existing):
        return False
    node.set("C", format_eval_comment(winrate, score, visits))
    return True


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


def winning_first_move(node):
    """The child the client would accept as the answer, or None if nothing here
    can be ranked.

    The client's rule: highest winrate for the side to move, score breaking exact
    ties, first-listed winning an exact tie on both. Split out from
    promote_winning_first_move so a caller can ASK the question without the
    answer being reordered as a side effect — a repair must never reorder, since
    the id, the ratings and the attempt history are all tied to the answer the
    file already has.
    """
    if node is None:
        return None
    candidates = _candidate_evals(node)
    if not candidates:
        return None
    # max() returns the first maximal element, matching the client's strict-`>`
    # scan, so an exact tie on both axes leaves the earlier variation in front.
    return max(candidates, key=lambda c: (c[1], c[2]))[0]


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

    Selection is winning_first_move's. Returns True if a node was moved.
    """
    best = winning_first_move(node)
    if best is None or best is node[0]:
        return False
    best.reparent(node, 0)
    return True


PLAYOUTS_SHALLOW = 30
PLAYOUTS = 30
# The bar for "the engine already read this branch deeply enough to be worth
# recording". SEPARATE FROM `PLAYOUTS` on purpose, though they share a value
# today: PLAYOUTS is the default BUDGET and this is a QUALITY threshold, and they
# only coincide under a flat budget. Tying them together makes a raised budget
# widen the tree as a side effect — measured, a v+30 policy took the tree from
# 146 to 764 queries because nearly every child then cleared a bar meant to admit
# only well-searched ones.
FREE_BRANCH_VISITS = 30

# What a deep pass spends at the position it is ENTERED at — the answer's
# continuation, or the decision node with the answer excluded. Everything below
# that entry is budgeted by deep_follow_playouts off the parent's own visits.
DEEP_ENTRY_PLAYOUTS = 1000


def deep_follow_playouts(parent_visits):
    """max(30, v) — the budget for a node the parent's search already spent v
    visits on.

    Never search a node less deeply than the engine has already read it: those
    visits explored this very subtree, so their neural-net evaluations are still
    cached and asking for them again is close to free. The floor is what does the
    work; a sweep could not separate this from v+10 or v+20, so the increment is
    dropped and the plain floor kept."""
    return max(FREE_BRANCH_VISITS, int(parent_visits))

# Stay well under Python's default recursion limit (1000). KataGo under Japanese
# rules will sometimes refuse to pass in a settled position; this guard prevents
# the primary line from blowing the stack.
MAX_VARIATION_DEPTH = 800


class VariationDepthExceeded(Exception):
    """Raised when generate_variations_directly_to_sgf goes too deep without terminating."""
    def __init__(self, moves):
        self.moves = list(moves)
        super().__init__(f"variation depth {len(self.moves)} exceeded; sequence: {self.moves}")

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
        var_node = get_child_move(node, player, move, queries.board_size)

        # DECIDE BEFORE PAYING FOR THE QUERY. This pass never overwrites, so a
        # move that already carries an evaluation AND a response under it has
        # nothing to gain from a 30-visit search — running one and discarding it
        # is what made this pass cost a query per legal move (a median 13x13
        # problem has 96) when the deep passes had already covered the ones worth
        # covering. What is left for it is exactly the moves the search declined
        # to read to the floor, which is the whole point of a shallow sweep.
        if var_node.has_property("C") and any(c.has_property("C") for c in var_node):
            continue

        # Get the optimal response to this move
        response_moves = moves + [move]
        response_result = queries.query(response_moves, playouts=PLAYOUTS_SHALLOW)
        response_move, winrate, score, _ = queries.get_best_move(response_result)

        # Both nodes record PLAYOUTS_SHALLOW, because that is the depth the
        # estimate rests on, and set_eval_comment will not let it displace a
        # deeper one. This pass ran LAST and used to write unconditionally, so
        # its 30-visit estimate replaced whatever the deep passes had found —
        # including the answer's, which is a child of `node` like any other and
        # which the deep pass never writes (that call passes exclude_move).
        # Depth is the whole difference: measured at one decision node, the
        # answer reads 87.4% at 30 visits and 96.6% at the 1000 generation
        # verifies with.
        set_eval_comment(var_node, winrate, score, PLAYOUTS_SHALLOW)

        response_player = "b" if player == "w" else "w"
        response_node = get_child_move(var_node, response_player, response_move, queries.board_size)
        set_eval_comment(response_node, winrate, score, PLAYOUTS_SHALLOW)

    return


def generate_variations_directly_to_sgf(
        queries: KatagoQueries,
        parent_sgf_node: sgf.Tree_node,
        current_moves: list,
        is_primary: bool,
        exclude_move=None,
        playouts=None,
        playouts_fn=None,
        write_parent_eval=False,
):
    """
    Recursively generates variations and adds them directly as child nodes to the parent_sgf_node.

    WHAT GETS RECORDED is every move the search read to FREE_BRANCH_VISITS,
    whether or not it lies on the line being followed to a conclusion. Those
    visits explored that subtree already and their neural-net evaluations are
    cached, so keeping the branch costs a query that is served from cache —
    measured, a re-query of an already-searched position costs 0.06s against
    18.75s cold. Below the floor there is nothing worth keeping, and since
    `moveInfos` is ordered best-first, the first move under it ends the loop.

    HOW DEEP each branch is then searched is `playouts_fn(visits)` — a child's
    budget follows from what the parent already spent on it, so a branch is never
    re-read at less depth than the engine has already read it to. The PRIMARY
    line is followed regardless of its share, because it is what carries the
    variation to a conclusion.

    `write_parent_eval` writes this search's ROOT evaluation onto the node the
    variations hang off. It is how the answer gets an honest number: the answer
    is nobody's `moveInfos` entry (the decision-node pass excludes it), so its
    comment can only come from a search of the position it leads to — which is
    what "how good is this move" means, and what the shallow pass was
    approximating at 30 visits.

    Returns True when some line at or under this node was read all the way to a
    DOUBLE PASS.

    NOTHING IS MARKED IN THE FILE, because the STRUCTURE already says it. A
    branch is followed to a conclusion exactly when it is the primary line, and
    the primary line is `moveInfos[0]` — the FIRST child appended here, and the
    one this function recurses into. So "which branch was played out" is answered
    by walking first children to a double pass, which is exactly what
    `subtree_reaches_conclusion` above does.

    Do NOT restate that as "the highest-visit child". KataGo ranks `moveInfos` by
    LCB rather than raw visits, so the followed move is frequently not the
    highest-visit one, and a max-visit walk drifts off the line — measured, in
    `subtree_reaches_conclusion`'s own docstring.

    The one case that could separate the primary line from a conclusion does not
    reach a file: a line the engine will not play out raises
    VariationDepthExceeded, and every caller discards the problem rather than
    writing a truncated one. Note this is a statement about the PRIMARY LINE, not
    a licence to treat "a pass appears somewhere in the subtree" as a conclusion:
    that is false for every branch abandoned at the visit floor, and it caused
    two corpus purges.
    """
    if len(current_moves) >= MAX_VARIATION_DEPTH:
        raise VariationDepthExceeded(current_moves)

    if current_moves[-1] == 'pass' and current_moves[-2] == 'pass':
        return True

    current_pos_analysis = queries.query(
        current_moves,
        playouts=PLAYOUTS if playouts is None else playouts,
        exclude_move=exclude_move,
    )
    if not current_pos_analysis or not current_pos_analysis.get("moveInfos"):
        return False

    if write_parent_eval:
        root_info = current_pos_analysis.get("rootInfo") or {}
        if "winrate" in root_info:
            set_eval_comment(parent_sgf_node, root_info["winrate"],
                             root_info["scoreLead"], root_info.get("visits", 0))

    reached_end = False

    for move_info in current_pos_analysis["moveInfos"]:
        visits = move_info.get("visits", 0)
        if visits < FREE_BRANCH_VISITS and not is_primary:
            break

        # Create a new child node for this variation
        var_node = get_child_move(parent_sgf_node, "b" if (len(current_moves) % 2 == 0) else "w", move_info["move"], queries.board_size)
        set_eval_comment(var_node, move_info["winrate"], move_info["scoreLead"],
                         visits)

        # Recursively add deeper variations from this new node
        child_reached_end = generate_variations_directly_to_sgf(
            queries,
            var_node,  # The new child node becomes the parent for the next level
            current_moves + [move_info["move"]],  # Extend the move list for deeper analysis
            is_primary,
            playouts=(None if playouts_fn is None else playouts_fn(visits)),
            playouts_fn=playouts_fn,
            # Let the child's own search upgrade the comment just written from
            # the parent's view of it. On the thin end of the primary line the
            # parent may have spent only a handful of visits on a move that is
            # then searched at the floor, so the better estimate is the one the
            # recursion is about to produce anyway. set_eval_comment refuses a
            # downgrade, so where the two are equal this writes nothing.
            write_parent_eval=True,
        )

        if child_reached_end:
            reached_end = True

        is_primary = False

    return reached_end

