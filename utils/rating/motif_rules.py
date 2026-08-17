"""The motif PROXY rules — hand-written criteria that define a direction by construction.

A proxy is not a discovered motif and does not try to be. `pipeline.train_mirt`
teaches its axes onto these, and the reason production works this way is measured:
a FREE fit lands on a different basis every seed (0.34 agreement), and a fit taught
from previously-FITTED directions holds them at 0.43-0.73, which inspection showed
is below where a name survives — four of six poles at those figures had mutated or
become unnameable. A rule does not have that failure mode. It says what the
direction IS, so the axis it seeds carries the same meaning every refit, and the
only thing a refit can change is how well the model expresses it.

The counter-evidence for rules is thin and points the same way: `opening` and
`counting` are the two motifs that reproduced across two entirely different model
generations, and both are rule-shaped concepts.

Every rule evaluates EVERY problem, so each label is decided both ways. That is
what hand labelling cannot supply cheaply and what lets a column be fitted at all:
without negatives there is nothing to separate from.

Thresholds are chosen to put each rule in a comparable share of the corpus, since a
rule that fires on 1% seeds a direction almost nothing constrains and one that
fires on 60% seeds a direction that says little. They are sampled rather than swept
over the whole corpus — 10k gives +/-0.6pp at 95%, ample for a target quoted in
whole percent.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# The two engine-side helpers are imported INSIDE the rules that use them. This
# module is the authority on what the motifs are AND on what they are called, and
# packaging reads the naming half; importing it must not drag in the KataGo side,
# which also keeps `utils/rating` and `utils/katago` free of each other as the
# split intends.

SIGNAL_CACHE = os.path.join(PROJECT_ROOT, "activations", "_signal_cache")
OWNERSHIP_CSV = os.path.join(SIGNAL_CACHE, "ownership_swing.csv")

LD_FOCUS = ("own_dead_revived", "opp_alive_killed")   # fail-to-live, fail-to-kill
LD_OTHER = ("own_alive_killed", "opp_dead_revived")   # sacrifice, let-out
LD_MIN_STONES = 5.0    # the winning part must move at least this many stones
# Trades are judged RELATIVE to the outcome, not against a fixed stone count. A
# 30-stone group living or dying is not a trade because a quarter-stone drifted
# somewhere else, and an absolute cap says it is — it gets stricter the larger
# the problem, which is backwards. What "no trade" means is that nothing else
# comes close to the thing being decided.
LD_TRADE_RATIO = 0.20  # every OTHER directional part, as a fraction of the winner

# counting: the top two first moves finish within a hair of each other on score
# while the win probability separates hard — the position turns on a margin you
# have to count rather than on a status you can see.
# The winrate bar has to be HIGH to select anything. Admission requires exactly
# one winning reply, played out, so the corpus has almost no ambiguous-outcome
# positions: 99.5% of problems clear a 50-point winrate gap and the clause passes
# everything. At 90 it admits 55%, which is a filter.
COUNT_SCORE_MAX = 5.0   # points between best and second best
COUNT_WR_MIN = 90.0     # percentage points of winrate between the same two

# ko: replay each resolved line this many plies and ask whether a ko shape stands.
# Depth 0 is the root position itself.
KO_DEPTH = 2   # covers the relevant scenarios; depth 3 adds 3.5 points of corpus for little

# opening: few moves led to the position, per board size.
# OPENING — an ABSOLUTE prefix length per board, not a percentile.
#
# A percentile is population-dependent: the cutoff moves when the corpus grows and
# flips the label on problems that have not changed, which makes an incremental
# labelling quietly wrong and every stored column a different question from the
# last one. These values ARE the percentile as measured — sampled at 10k, the 5th
# percentile of game-phase prefix sits at 12 moves on 9x9 (5.39% of problems) and
# 28 on 13x13 (5.33%) — frozen so they stay put. Re-measure and re-set them
# deliberately if the corpus ever shifts enough to matter; do not make them
# floating again.
OPENING_MAX_PREFIX = {9: 12, 13: 28}

# SHAPE — the two best resolved first moves are CLOSE in winrate, so the position
# does not announce which is right and the choice rests on judging the shape.
# Sampled, the gap has median 91.2 and only its low tail is uncertain at all: p5
# is 64.1, so a ceiling at 67 takes roughly the bottom 7%.
SHAPE_WR_GAP = 67.0   # MAXIMUM gap; a wide gap means the answer is obvious

# TERRITORY — the score swing outruns what the stones changing hands can account
# for, by a wide MULTIPLE rather than a fixed margin.
TERR_BASE = 0.5      # points of swing required when no stone changes hands at all
TERR_PER_STONE = 4.0 # points of swing each flipped stone must be beaten by

# The rule set, and its ORDER is the column order of axes/motif_proxies.npz. Names
# are the key: `components` in axes/labels.json names rules, and readers look them
# up in the npz's stored `rules` array rather than assuming a position. So a name
# must identify exactly one rule — a repeat would make the second column
# unreachable, teaching an axis from whichever copy came first with nothing
# reporting the other was ignored. (This is the property the DEV motif store does
# not have: it holds several directions called `ko` from different fits, which is
# why that path is keyed by an `ax_<hex>` content hash instead of by name.)
LABEL_SET = ["life and death", "counting", "ko", "opening", "territory",
             "shape"]
if len(set(LABEL_SET)) != len(LABEL_SET):
    raise ValueError(f"duplicate rule name in LABEL_SET: {LABEL_SET}")

# What the client calls each motif. THIS MODULE IS THE ONLY SOURCE — packaging
# resolves a shipped tail through here, never through the `bank` in
# axes/labels.json, which is the dev store and carries several directions under
# each of these names (three `counting`, three `opening`, and a `ko` that is one
# entry's POSITIVE tail and another's NEGATIVE). A name lookup there cannot pick
# among them and cannot even fix the sign, whereas a rule name is unique by the
# check above and means exactly one thing: the criterion written below it.
#
# The English key is the rule name, so a rule cannot ship under a name no rule
# has. Translations are the standard Go terms; a tail with no entry here ships
# unlabeled and the client stays silent about it, which is the intended state for
# an axis a human has not inspected yet.
RULE_LABELS = {
    "life and death": {"zh": "死活", "ja": "死活", "ko": "사활"},
    "counting":       {"zh": "数目", "ja": "目算", "ko": "계가"},
    "ko":             {"zh": "打劫", "ja": "コウ", "ko": "패"},
    "opening":        {"zh": "布局", "ja": "序盤", "ko": "포석"},
    "territory":      {"zh": "实地", "ja": "地",   "ko": "실리"},
    "shape":          {"zh": "棋形", "ja": "形",   "ko": "모양"},
}
_unknown = sorted(set(RULE_LABELS) - set(LABEL_SET))
if _unknown:
    raise ValueError(f"RULE_LABELS names no rule defines: {_unknown}")


# ---------------------------------------------------------------- helpers

def _subtree_has_te(node):
    """Was this root variation RESOLVED — i.e. does anything BELOW it carry TE.

    The node's own mark is deliberately not counted. At the decision node a TE
    mark means "this is the move to find", not "this line was played out to a
    conclusion"; only the answer is marked there, and marking a refuted move
    would read as naming it the solution. So a root variation's own TE says
    nothing about whether its continuation was resolved, and the question here is
    only ever about the continuation."""
    stack = list(node)
    while stack:
        n = stack.pop()
        if n.has_property("TE"):
            return True
        stack.extend(list(n))
    return False


def _mover_eval(node):
    """(winrate, score) in the MOVER's frame, or None. Comments are stored from
    BLACK's perspective, so a white mover flips both."""
    colour, _ = node.get_raw_move()
    if colour is None:
        return None
    from utils.katago.add_variations import parse_eval_comment
    c = node.get("C") if node.has_property("C") else None
    parsed = parse_eval_comment(c)
    if parsed is None:
        return None
    wr_black, score_black = parsed
    return ((wr_black, score_black) if colour == "b"
            else (100.0 - wr_black, -score_black))


def _line_moves(node, depth):
    """The first `depth` plies at and under `node`, as (COLOR, sgf_coord) pairs in
    board_render's setup format.

    Follows the TE-MARKED child at each step, falling back to the first child only
    when no child carries a mark. TE is what says a branch was driven to a double
    pass, so it names the line the generator actually read; the first child is
    merely the first one written and is not guaranteed to be that line. Where a
    node has several marked children the first marked one is taken — they are
    alternatives the generator resolved, and one line has to be picked."""
    out = []
    cur = node
    while cur is not None and len(out) < depth:
        colour, raw = cur.get_raw_move()
        if colour is not None and raw and raw != b"tt":
            out.append((colour.upper(), raw.decode("utf-8")))
        kids = list(cur)
        if not kids:
            break
        marked = [k for k in kids if k.has_property("TE")]
        cur = marked[0] if marked else kids[0]
    return out


# ---------------------------------------------------------------- the rules

def rule_life_and_death(own, min_stones=None, trade_ratio=None):
    """ONE status change, big enough to be the point of the problem, and nothing
    else close to it.

    `own` is the raw ownership row (stone counts, NOT the z-scored feature
    columns — a threshold in stones is only a threshold in stones before the
    pooled standardization). The dominant directional part must be a fail-to-live
    or a fail-to-kill; the other three are judged as a FRACTION of it, so a
    problem where you save one group by giving up another is excluded while
    incidental drift beside a large group is not.
    """
    min_stones = LD_MIN_STONES if min_stones is None else min_stones
    trade_ratio = LD_TRADE_RATIO if trade_ratio is None else trade_ratio
    parts = {k: abs(float(own[k])) for k in (*LD_FOCUS, *LD_OTHER)}
    top = max(parts, key=parts.get)
    if top not in LD_FOCUS or parts[top] < min_stones:
        return False
    bar = trade_ratio * parts[top]
    return all(v <= bar for k, v in parts.items() if k != top)


def rule_territory(own, base=None, per_stone=None):
    """Score swings by more than the stones changing hands can pay for.

        (score_correct - score_alt)  >  base + per_stone * stone_swing_abs

    MULTIPLICATIVE in the stones, not a fixed remainder. A flipped stone is worth
    two points on the board, so requiring the swing to beat FOUR times the stones
    demands the swing be at least twice what capture explains — the position has
    to be decided by territory, not merely to contain some. A fixed remainder
    instead admitted any problem with a large enough absolute swing however much
    of it came from stones, which on a big capture is most of it.

    `stone_swing_abs` is the ABSOLUTE swing summed over occupied points, so a TRADE
    does not cancel. That is deliberate: stones dying on both sides is a trade
    whatever the net comes to, and a trade is not a territory problem. Netting them
    would let a big two-sided capture pass as territory because the sides happened
    to balance.

    Empty-point swings are NOT subtracted — they are the territory itself, and
    charging for them would leave nothing to detect.

    `own` is the raw ownership row, in stones (see rule_life_and_death on why a
    threshold in stones needs the pre-standardization values).
    """
    base = TERR_BASE if base is None else base
    per_stone = TERR_PER_STONE if per_stone is None else per_stone
    swing = float(own["score_correct"]) - float(own["score_alt"])
    return swing > base + per_stone * float(own["stone_swing_abs"])


def rule_shape(evals, max_gap=None):
    """The two best RESOLVED first moves are CLOSE, so which is right is a
    judgement rather than a calculation.

    `evals` is the mover-framed (winrate, score) of every root variation whose
    continuation the generator resolved — an abandoned line is not a rival, it is
    a line nobody played out. Ranked by WINRATE, which is what names the answer
    (findBestMoveFromChildren).

    A MAXIMUM gap, not a minimum: a wide gap means one move wins and the other
    loses, which the position announces. The uncertain end is the narrow one, and
    it is narrow — the gap's median is 91 and its p5 is 64, so almost every
    problem in this corpus has an unambiguous best move and the ones that do not
    are the tail this rule takes.

    Distinct from `rule_counting`, which wants a LARGE winrate gap alongside a
    small score gap: there the position is decided by a margin and the engine is
    sure which move takes it."""
    if len(evals) < 2:
        return False
    max_gap = SHAPE_WR_GAP if max_gap is None else max_gap
    top = sorted(evals, key=lambda t: (t[0], t[1]), reverse=True)[:2]
    return (top[0][0] - top[1][0]) <= max_gap


def rule_counting(evals):
    """Score says the two best first moves are equivalent; winrate says they are
    not. That gap is the signature of a position decided by a margin.

    `evals` is [(winrate, score)] over the root variations, mover-framed. Ranked
    by WINRATE, because that is what names the answer (findBestMoveFromChildren),
    so "best and second best" is the answer and its closest rival."""
    if len(evals) < 2:
        return False
    top = sorted(evals, key=lambda t: (t[0], t[1]), reverse=True)[:2]
    (w1, s1), (w2, s2) = top
    return (s1 - s2) <= COUNT_SCORE_MAX and (w1 - w2) >= COUNT_WR_MIN


def rule_ko(base, variation_nodes, size):
    """A ko shape stands within KO_DEPTH plies of a line the generator resolved.

    Checked at the ROOT first, then along each TE-marked root variation — the
    root case is the position that already sits in a ko, the line case is the
    position that becomes one once the fight starts. Only resolved branches are
    walked, for the same reason the ownership baseline only uses them: an
    abandoned stub is a line nobody played out.

    `base` is the setup in SGF COORDINATES ("pd"), which is what `simulate`
    parses — NOT the GTP form ("D4") that `parse_tsumego_sgf` hands back. Feeding
    it GTP fails silently and completely: every stone misses the range check, the
    board stays empty, and no position ever holds a ko."""
    from utils.katago.board_render import simulate
    if simulate(base, size)[2] is not None:
        return True
    if KO_DEPTH <= 0:
        return False
    for node in variation_nodes:
        if not _subtree_has_te(node):
            continue
        moves = _line_moves(node, KO_DEPTH)
        for i in range(1, len(moves) + 1):
            if simulate(base + moves[:i], size)[2] is not None:
                return True
    return False


def rule_opening(prefix, size):
    """Few moves led to the contested position.

    The bar is an ABSOLUTE prefix length per board (OPENING_MAX_PREFIX), not a
    percentile of the corpus. A percentile moves as the corpus grows and re-labels
    problems that did not change, which is the one thing an incremental labelling
    cannot absorb."""
    return prefix <= OPENING_MAX_PREFIX[int(size)]


# ---------------------------------------------------------------- driver

