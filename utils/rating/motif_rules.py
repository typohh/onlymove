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

# semeai: BOTH focus parts at once. The answer saves your group AND kills theirs,
# and neither is a footnote to the other, so the position is a capturing race
# rather than one status change with drift beside it.
#
# The bar is on the pair's SPLIT, not on either part, so the rule does not ask
# the two groups to be the same SIZE — a race is about both being at stake, and
# 0.25 admits up to a 1:3 mismatch between them.
#
# STAYING UNDER 1/3 IS WHAT KEEPS THIS DISJOINT FROM `life and death`, and it is
# LD_TRADE_RATIO that puts the boundary there: a rule judging every other part
# against a fraction of the largest one holds the smaller focus part at or under
# ratio/(1+ratio) of the pair, i.e. a split at least 1/3 away from even. So the
# two rules cannot both fire while this bar sits below that, and the gap between
# them — a split between 1/6 and 1/4 — deliberately labels as NEITHER rather than
# letting the two meet at a shared edge, where a hair of drift decides which
# motif a problem seeds. Moving LD_TRADE_RATIO moves that boundary.
SEMEAI_BALANCE = 0.25
# The PAIR analogue of LD_MIN_STONES: life and death asks one decisive group to
# be worth 5 stones, and a race is decided by two, so the bar goes on their sum.
# With the balance bar it also floors the smaller side at (0.5 - balance) * this,
# which is what stops a one-stone throw-in beside a large group reading as a
# race. 10 takes ~9% of the corpus; 8 -> 11.3%, 6 -> 14.9%, 12 -> 6.7%. Narrow is
# the safe side: the fit SPREADS a narrow rule outward from its core and takes a
# ~20% rule as written (see rule_territory, narrowed for the same reason).
SEMEAI_MIN_STONES = 10.0

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
# last one. So these are frozen numbers, re-set DELIBERATELY when there is a
# reason — which is what the caps below are: aimed at the ~20% share the rule set
# targets, measured over the whole phase table (18 -> 18.8% on 9x9, 46 -> 19.8% on
# 13x13). They previously sat at the 5th percentile (12 and 28), which put the rule
# on ~5% of the corpus and left it seeding a direction almost nothing constrains.
# Per board because game length scales with the board, unlike the tenuki and sente
# distances below.
OPENING_MAX_PREFIX = {9: 18, 13: 46}

# TENUKI / SENTE — Chebyshev distance bars, set by SAMPLING 4k problems per board
# for the ~20% share every rule is aimed at.
#
# NOT per board size, unlike OPENING_MAX_PREFIX, and that is measured rather than
# assumed: both distance distributions come out nearly identical on the two boards
# (median 1, p75 3, p85 4, p90 5 on each), because a tsumego is a local position
# whose geometry does not scale with the board it sits on. A per-board bar would
# be fitting the sampling noise, and worse, it would make one name mean two
# different criteria — the 20%-closest bars are 4 on 9x9 and 3 on 13x13, i.e. the
# BIGGER board would demand LESS distance to count as playing away.
#
# The share lands at 17.2%/15.3% rather than 20% because the distance is a whole
# number of lines: >=3 gives 25.9%/23.4% and >=4 gives 17.2%/15.3%, so 4 is the
# integer nearest the target.
#
# ONE bar serves BOTH rules, and rule_sente has none of its own. Tenuki is a
# single notion — the opponent played away — and both rules are statements about
# it, so a second constant would only create the possibility of the two drifting
# apart while each kept the name. rule_tenuki asks it of the answer against the
# opponent's LAST move; rule_sente asks it twice, of the reply each first move
# draws, and wants the answer below the bar and the refuted try at or above it.
TENUKI_MIN_DIST = 4    # lines between a move and the opponent's play

# SHAPE — the two best resolved first moves are CLOSE in winrate, so the position
# does not announce which is right and the choice rests on judging the shape.
# 67, NOT the 82 that a ~20% share target implies. Widening it was measured and
# it broke the rule: N50 went 1982 -> 6692 over four seeds and the fitted column
# came back ANTI-correlated between halves at short histories (r@125 of -0.72 on
# every seed). The reason is that the threshold carries the meaning here — at a
# ceiling of 82 the two best moves are no longer close, so the criterion stops
# denoting what its name says. Contrast `opening`, where widening a prefix cutoff
# keeps the concept ("an early position") and merely admits more of it.
#
# So a share target does not apply to this rule. Sampled at 2.5k per board the gap
# has median 90.8 and p5 64.6; 67 takes ~5%. Note ~20% of problems have fewer than
# two RESOLVED first moves and can never fire whatever the ceiling.
SHAPE_WR_GAP = 67.0   # MAXIMUM gap; a wide gap means the answer is obvious

# TERRITORY — the score swing outruns what the stones changing hands can account
# for, by a wide MULTIPLE rather than a fixed margin.
TERR_BASE = 0.5      # points of swing required when no stone changes hands at all
# 8.0, taking ~9% of the corpus. Swept: 4 -> 30.4%, 5 -> 21.5%, 6 -> 15.8%,
# 8 -> 9.4%, 10 -> 6.2%.
#
# Narrowed rather than widened, against the share target, because of what the
# fitted axes do with a rule of each width: the fit SPREADS a narrow rule (mass
# participation ratio 1.5-1.85x its share) and leaves a ~20% rule essentially
# as-stated (1.03-1.20x). So a narrow rule only needs its CORE to be right and the
# model elaborates outward, while a wide rule has to already BE the concept. At
# 21.5% territory was the broadest rule, carried the lowest mass lift of the eight
# (3.2x), and reached N50 `inf` on two seeds — the signature of a rule too vague
# to pin a direction, which more coverage cannot fix.
TERR_PER_STONE = 8.0 # points of swing each flipped stone must be beaten by

# The rule set, and its ORDER is the column order of axes/motif_proxies.npz. Names
# are the key: `components` in axes/labels.json names rules, and readers look them
# up in the npz's stored `rules` array rather than assuming a position. So a name
# must identify exactly one rule — a repeat would make the second column
# unreachable, teaching an axis from whichever copy came first with nothing
# reporting the other was ignored. (This is the property the DEV motif store does
# not have: it holds several directions called `ko` from different fits, which is
# why that path is keyed by an `ax_<hex>` content hash instead of by name.)
LABEL_SET = ["life and death", "counting", "ko", "opening", "territory",
             "shape", "tenuki", "sente", "semeai"]
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
    "tenuki":         {"zh": "脱先", "ja": "手抜き", "ko": "손빼기"},
    "sente":          {"zh": "先手", "ja": "先手", "ko": "선수"},
    "semeai":         {"zh": "对杀", "ja": "攻め合い", "ko": "수상전"},
}
_unknown = sorted(set(RULE_LABELS) - set(LABEL_SET))
if _unknown:
    raise ValueError(f"RULE_LABELS names no rule defines: {_unknown}")


# ---------------------------------------------------------------- helpers

def _subtree_is_resolved(node):
    """Was this root variation played out to a CONCLUSION.

    `subtree_reaches_conclusion` follows the writer's own first-child chain and
    asks whether it ends in two consecutive passes. That is exact rather than
    heuristic: the writer creates the second pass only where its recursion
    reached the terminal, so the tree carries the fact and nothing in the file is
    annotated. Don't weaken it to "a pass anywhere in the subtree" — that
    over-accepts on refuted first moves, which is how a solver ends up offered a
    line that loses."""
    from utils.katago.add_variations import subtree_reaches_conclusion
    return subtree_reaches_conclusion(node)


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

    Follows the FIRST child at each step, which is the line the generator
    actually read: the writer appends children in `moveInfos` order and always
    recurses into the first one, so the first child IS the continuation it
    followed. This is the same walk `subtree_reaches_conclusion` makes, and the
    two must stay on the same child for a line judged resolved to be the line
    measured here. Don't switch to the highest-visit child — KataGo ranks
    `moveInfos` by LCB, not raw visits, so that walk drifts off the line."""
    out = []
    cur = node
    while cur is not None and len(out) < depth:
        colour, raw = cur.get_raw_move()
        if colour is not None and raw and raw != b"tt":
            out.append((colour.upper(), raw.decode("utf-8")))
        kids = list(cur)
        if not kids:
            break
        cur = kids[0]
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


def rule_semeai(own, balance=None, min_stones=None, trade_ratio=None):
    """A capturing race: the answer saves your own group AND kills the
    opponent's, and neither of the two is a footnote to the other.

    The COMPLEMENT of rule_life_and_death over the same two ownership parts, and
    disjoint from it by construction (see SEMEAI_BALANCE). Where that rule wants
    one status change with nothing near it, this one wants both, which is what a
    semeai is: two groups whose fates are the same question. A rule that only
    asks for a dominant part cannot express that — it reads the second group as
    the trade that disqualifies the first.

    Judged on the pair's SPLIT and on the pair's SIZE, never on either part
    alone, so the two groups need not match in size. The trade bar is inherited
    from rule_life_and_death and referred to the PAIR, since the pair is what is
    being decided here: a race you win by letting a third group die is a trade
    like any other.

    `own` is the raw ownership row, in stones (see rule_life_and_death on why a
    threshold in stones needs the pre-standardization values).
    """
    balance = SEMEAI_BALANCE if balance is None else balance
    min_stones = SEMEAI_MIN_STONES if min_stones is None else min_stones
    trade_ratio = LD_TRADE_RATIO if trade_ratio is None else trade_ratio
    save, kill = (abs(float(own[k])) for k in LD_FOCUS)
    pair = save + kill
    if pair <= 0.0 or pair < min_stones:
        return False
    if abs(save / pair - 0.5) > balance:
        return False
    bar = trade_ratio * pair
    return all(abs(float(own[k])) <= bar for k in LD_OTHER)


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

    Checked at the ROOT first, then along each RESOLVED root variation — the
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
        if not _subtree_is_resolved(node):
            continue
        moves = _line_moves(node, KO_DEPTH)
        for i in range(1, len(moves) + 1):
            if simulate(base + moves[:i], size)[2] is not None:
                return True
    return False


def _xy(sgf_coord):
    return ord(sgf_coord[0]) - 97, ord(sgf_coord[1]) - 97


def _dist(a, b):
    """Chebyshev (king-move) distance between two SGF coordinates.

    "Two lines away" in Go is a statement about either axis, not about their sum,
    and not about a norm that penalises the diagonal — a move one right and one
    up is adjacent, not 1.41 away."""
    (ax, ay), (bx, by) = _xy(a), _xy(b)
    return max(abs(ax - bx), abs(ay - by))


def answer_opponent_distance(setup, correct):
    """Distance from the ANSWER to the opponent's last move, or None.

    `setup` is parse_sgf's ordered move chain, so the opponent's move is simply
    the last entry of the other colour. Constructed positions with no preceding
    move of that colour return None and the rule declines to fire, rather than
    guessing at a move that was never played."""
    if not correct:
        return None
    acol, amove = correct
    for col, mv in reversed(setup):
        if col.upper() != acol.upper():
            return _dist(amove, mv)
    return None


def reply_distance(node):
    """How close the opponent's answer lands to the move that provoked it.

    ONE measurement, taken from a root variation's OWN move to the opponent's
    reply to it, so the same question can be asked of the answer and of a refuted
    try without either needing a notion of its own. Returns None when the line
    has no reply to measure.

    The reply is taken along the line the generator read (see _line_moves),
    which is the first-child chain."""
    if node is None:
        return None
    colour, raw = node.get_raw_move()
    if colour is None or not raw or raw == b"tt":
        return None
    mv = _line_moves(node, 2)
    if len(mv) != 2 or mv[1][0].upper() == colour.upper():
        return None
    return _dist(mv[0][1], mv[1][1])


def best_refuted_node(variation_nodes, correct):
    """The strongest first move that is NOT the answer, or None.

    "Strongest" is the client's own rule — highest winrate in the mover's frame,
    score breaking ties — over the RESOLVED alternatives only. An abandoned stub
    has a continuation the generator never read, so what the opponent "does"
    about it is not evidence of anything; rule_shape ranks only resolved moves
    for the same reason."""
    if not correct:
        return None
    _acol, amove = correct
    best, best_eval = None, None
    for n in variation_nodes:
        colour, raw = n.get_raw_move()
        if colour is None or not raw or raw == b"tt":
            continue
        if raw.decode("utf-8") == amove:
            continue                      # the answer, not a refuted move
        if not _subtree_is_resolved(n):
            continue
        ev = _mover_eval(n)
        if ev is None:
            continue
        if best_eval is None or ev > best_eval:
            best, best_eval = n, ev
    return best


def rule_tenuki(opp_distance, root_ko, min_dist=None):
    """The answer is played AWAY from what the opponent just did.

    A KO STANDING ON THE BOARD DISQUALIFIES THE PROBLEM OUTRIGHT, whatever the
    distance. Taking a ko, or answering one, is a forced local exchange that
    happens to be recorded far from the opponent's last move — the geometry says
    tenuki and the go does not. Excluding those is the difference between a rule
    about playing elsewhere and a rule about ko notation.

    The bar is per board size: a fixed number of lines is a different fraction of
    a 9x9 board than of a 13x13 one, so a shared threshold would fire on a
    different concept for each — the same reason OPENING_MAX_PREFIX is per
    board."""
    if opp_distance is None or root_ko:
        return False
    return opp_distance >= (TENUKI_MIN_DIST if min_dist is None else min_dist)


def rule_sente(answer_reply, refuted_reply, min_dist=None):
    """The opponent MUST answer the correct move — and demonstrably so.

    A CONTRAST between two moves rather than a property of one, and the SAME
    question of each (reply_distance): how close does the opponent play to the
    move just made. The answer draws a reply that is not a tenuki; the best
    refuted try draws one that is.

    Asking it only of the answer does not identify sente. Most of this corpus is
    a local fight, and a local fight produces a local reply whatever was played,
    so a near reply on its own says the position is busy. It is the refuted move
    being ignored that says the reply to the answer was compelled — a move both
    tries get is a busy corner, a move only the right one gets is sente.

    ONE bar, TENUKI_MIN_DIST, because there is one notion here and it is tenuki:
    the rule reads "not a tenuki, against a tenuki". A second constant for the
    answer side would let the two drift apart while both kept the name.

    Declines when either distance is missing, rather than reading an absent
    refutation as a tenuki — which would fire the rule where the corpus knows
    least. In practice that costs almost nothing: sampled at 4k per board, both
    distances are measurable on 99.6% of 9x9 and 100% of 13x13, so the rule is
    decided on essentially the whole corpus rather than on a subset.

    The share lands at 20.4%/23.8%, on the ~20% target the rule set aims for, and
    the CONTRAST is what puts it there: the answer-side test alone fires on
    61.7%/48.4%, far too broad to name anything. The refuted side is doing most
    of the work, which is the measured form of the argument above.

    Board-independent, for the reason given on rule_tenuki."""
    if answer_reply is None or refuted_reply is None:
        return False
    bar = TENUKI_MIN_DIST if min_dist is None else min_dist
    return answer_reply < bar and refuted_reply >= bar


def rule_opening(prefix, size):
    """Few moves led to the contested position.

    The bar is an ABSOLUTE prefix length per board (OPENING_MAX_PREFIX), not a
    percentile of the corpus. A percentile moves as the corpus grows and re-labels
    problems that did not change, which is the one thing an incremental labelling
    cannot absorb."""
    return prefix <= OPENING_MAX_PREFIX[int(size)]


# ---------------------------------------------------------------- driver

