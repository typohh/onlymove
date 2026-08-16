"""Measure, per tsumego, how many stones' worth of ownership the correct first
move swings against the best alternative first move.

The question this answers: does a problem involve life and death, and whose? A
point's KataGo ownership is the expected final owner on [-1, +1]. Read that at
the intersections that ALREADY CARRY A STONE and compare the map after the
correct move against the map after the best refuted move: a stone whose
ownership flips is a stone that only the correct move kills or saves. A problem
with no such flip is decided by territory or capture-race counting, not by life
and death.

Method, per problem:
  - Parse the SGF; the correct move is variation_nodes[0] (generator convention,
    enforced by promote_winning_first_move) and its colour is the solver.
  - Collect the RESOLVED refutations: root variations whose subtree carries a
    TE[1] mark, i.e. lines the generator drove to a double pass. Everything else
    is an abandoned stub, and an ownership map read off one of those is the net's
    opinion about a line nobody finished rather than a scored result.

    A PASS counts like any other resolved variation. The baseline asks what the
    position is worth when the solver MISSES the answer, and a pass answers that
    question: where every resolved stone move loses more than doing nothing, the
    pass IS the strongest refutation, and ranking it out leaves the baseline on a
    move the solver would never prefer, which overstates the swing. It is also
    sometimes the ONLY refutation the generator resolved, and excluding it there
    leaves no measurement at all rather than a slightly wrong one. That the
    client offers no pass button does not bear on this: these columns price the
    POSITION, not the set of clickable cells.
  - Evaluate the answer and those refutations, and read each ownership map.
  - Baseline := the RESOLVED refutation with the highest SCORE for the solver.
  - Score the swing at the ROOT board's occupied points (the root board is the
    common reference; a child board differs from it wherever the move captured).

Restricting to TE-marked refutations is what makes the baseline well defined.
Ranking all ~50 legal moves instead is an argmax over near-ties — most
refutations lose by indistinguishable margins, and which one wins is then
numerical noise (measured: the two backends agreed on the move only 25 times in
40). Measured on the corpus, a problem has a median of ONE resolved refutation,
so the ranking rarely has to break a tie at all, and only the handful of
positions that need evaluating get evaluated.

RANKED BY THE SGF'S OWN SCORE, not by a winrate and not by a fresh query. Two
separate reasons, and the second is the one that survives if the first is ever
fixed.

SCORE RATHER THAN WINRATE, because every candidate here is a LOSING move. The
solver's win probability is compressed against zero across the whole candidate
set (median best 0.065), so an ordering read off it is largely the value head's
noise, while score keeps its resolution in exactly that regime. Measured over
every problem carrying two or more resolved refutations: the criteria disagree
on 6% of them, the winrate gap between the top two is a median 0.001 wherever
they disagree against 0.061 overall, and re-running flips 0.5% of winrate-chosen
baselines from engine nondeterminism alone — against score margins of which only
0.1% fall inside 0.01. Score is also the RIGHT UNIT: what gets read off the
baseline is an ownership map denominated in points, so ranking it by a
probability puts the selection and the measurement in different currencies.

THE SGF'S SCORE RATHER THAN A QUERY OF OUR OWN, because the generator already
evaluated every root variation with a real search and wrote the result into the
node's "C" comment, while anything this module asks for costs a query and gets
answered at maxVisits=1 — a raw net eval, which is the under-resolved instrument
the paragraph above is complaining about. The corpus's own numbers are both
better and free. They are also what the CLIENT displays and selects on, so the
baseline agrees with what the problem asserts about itself rather than with a
second opinion nobody else can see.

Comments are BLACK-perspective (bots/analysis.cfg sets reportAnalysisWinratesAs
= BLACK) and these variations are the SOLVER's moves, so a white solver flips the
sign — the same flip `add_variations._candidate_evals` does. Get it wrong and the
WORST refutation is picked with no other symptom. One caveat worth stating: a
comment is written at whatever budget the pass that produced it used, and
`add_variations` leaves an existing comment untouched, so budgets are not
uniform across a problem's variations. That is still a search per candidate
against one visit per candidate.

Note this is NOT the client's rule for naming the answer
(`findBestMoveFromChildren`: winrate first, score breaking exact ties) and must
not be conformed to it — the client decides which move a player should find,
where win probability is what a player maximizes; this module decides which
position to measure against.

Units: a delta of 2.0 in raw ownership is one point going from wholly the
opponent's to wholly ours, so every swing column is HALVED to read as stones.
`opp_stones_won` = 1.0 means the correct move kills exactly one opponent stone
that the best refutation leaves alive.

TWO BACKENDS, same output schema, so either can check the other:
  - `engine` (default): the KataGo analysis binary, one maxVisits=1 query per
    child, issued concurrently. Faster end to end because it builds the input
    features in C++ — the gap widens with the candidate count, so most of it is
    on the larger board.
  - `torch`: the vendored PyTorch net, all children in one batched forward pass.

Their conventions DIFFER and each was verified rather than assumed — get one
wrong and the sign silently inverts:
  - Torch ownership is positive for the PLAYER TO MOVE, and a child's player to
    move is the solver's OPPONENT, so the map is negated once. Its value head is
    likewise mover-framed, so the solver's winrate is softmax(value)[1].
  - Engine ownership and `rootInfo.winrate` are BLACK-absolute regardless of who
    is to move, so both are flipped by the SOLVER'S COLOUR instead. The engine's
    array is row-major from the TOP-left while this module indexes y from the
    bottom (the gtp_to_xy convention), so it is also flipped vertically.
Verified equal on real problems: ownership correlation 0.999+ per child, and the
same choice of best alternative. Neither backend supplies the RANKING any more —
that comes from the SGF — so a backend now only has to agree about the map it is
asked for, and the two cannot disagree about which map that is.

RULES ARE JAPANESE AT KOMI 6.5 (utils.constants RULES/KOMI, GameState.RULES_JAPANESE)
— the conditions the corpus was generated and evaluated under, so a winrate here
is comparable to the SGF's own branch comments. Every step that reads a position
off these SGFs takes its rules from the same place (pipeline.extract_activations
maps the same constants onto the vendored net's dict form), because reading a
corpus under rules it was not evaluated with changes the numbers while leaving
every shape intact.

TERRITORY SCORING IS HISTORY-DEPENDENT, and that dictates how positions are fed
to the engine: the full MOVE SEQUENCE, never `initialStones`. Reconstructing a
child from stones alone discards the prisoner count territory scoring needs, and
the engine then scores the position wrongly — measured, it called the SGF's own
verified-winning move a loss (0.002 against the SGF's 99.9%), while the same
position fed as moves returns 0.996. Do not "simplify" this back to
initialStones; extract_activations replays setup stones as moves for the same
reason, under the same rules.

The winrate and score columns are WHOLE-BOARD, not a verdict on the tsumego. The
score pair is the ranking key, kept so the file records what its baseline was
chosen on; the winrate pair is carried alongside as the independent reading, and
`wr_correct` below `wr_alt` is meaningful in itself — it marks a problem the net
cannot solve at one ply.

Output: ONE csv for both boards beside the other feature-block caches
(activations/_signal_cache/ownership_swing.csv), keyed by its `size` column the
way axes/per_problem.csv is. Appended and fsynced every --flush-every rows, so a
killed run loses at most that many. RESUMABLE by the tid column — ids are unique
across the whole namespace, so a tid already present is done whatever board it
came from. A partial trailing line from a kill mid-write is truncated on restart.

Measured-rejected, do not re-add: a multiprocessing pool for the torch backend's
child-feature build. It is about half that backend's wall time, so parallelising
it looks free, but shipping the input tensors back through a pipe costs more
than the cores save — measured SLOWER on both boards. Moving that work into the
engine is what actually removes it.

`n_te_alts` = 0 marks a problem with no resolved refutation of ANY kind — every
refuted first move, the pass included, was abandoned at the generator's playout
cutoff. That is a real gap rather than a failure: the swing columns are left
empty instead of being filled against an unfinished line, and such a problem is
not unmarked (its answer carries TE like every other). Coverage is complete on
9x9 (auxiliary/add_losing_lines.py gave every problem a played-out losing line)
and a floor on 13x13, where the generator pursues refutations less often, so the
count there tracks whatever it happened to resolve.

Launch it detached or it dies with the shell:

    setsid nohup ./.venv/bin/python3 -u -m auxiliary.extract_ownership_swing \\
        > /tmp/ownership_swing.log 2>&1 < /dev/null &
"""

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sgfmill.sgf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import utils.katago.torch_model  # noqa: F401  -- adds vendored katago/ to sys.path

from katago.game.board import Board
from katago.game.gamestate import GameState

from utils.constants import ALL_BOARD_SIZES, DEFAULT_MODEL, KOMI, RULES
from utils.katago.add_variations import parse_eval_comment
from utils.katago.controller import KatagoController
from utils.katago.coords import gtp_to_xy, sgf_to_gtp, xy_to_gtp
from utils.katago.models import get_model
from utils.katago.sgf_parsing import parse_tsumego_sgf

DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT, "models",
    "kata1-b18c384nbt-s9996604416-d4316597426",
    "model.ckpt",
)

# Beside the other per-problem feature caches the rating loaders read, not in a
# directory of its own at the repo root.
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "activations", "_signal_cache",
                           "ownership_swing.csv")

# The torch-side mirror of utils.constants RULES/KOMI. Asserted rather than
# assumed: the two describe the same ruleset through different interfaces, and a
# silent disagreement would price the two backends differently.
TORCH_RULES = GameState.RULES_JAPANESE
assert RULES == "japanese" and TORCH_RULES["scoringRule"] == "SCORING_TERRITORY", RULES
assert TORCH_RULES["whiteKomi"] == KOMI, (TORCH_RULES["whiteKomi"], KOMI)

FIELDS = [
    "tid", "size", "solver", "n_root_vars", "n_te_alts",
    "correct_move", "alt_move",
    # The SGF's own evaluations, solver-framed. score_alt is the RANKING KEY the
    # baseline was chosen on, recorded so the file says what it was chosen by;
    # the winrate pair is the independent reading and ranks nothing.
    "score_correct", "score_alt", "wr_correct", "wr_alt",
    "n_stones_solver", "n_stones_opp",
    # THE EIGHT PARTS: fractional stones, split by whose stones, by whether they
    # were alive or dead if the solver MISSES the move, and by which way they go.
    # The two the whole exercise is about are own_dead_revived (the group you
    # save) and opp_alive_killed (the group you kill).
    "own_dead_revived", "own_alive_secured", "own_alive_killed", "own_dead_deadened",
    "opp_alive_killed", "opp_dead_deadened", "opp_dead_revived", "opp_alive_secured",
    # Roll-ups: each is the sum of two parts above, kept for a coarse read.
    "opp_stones_won",    # opponent stones the correct move takes that the best alternative does not
    "own_stones_saved",  # our stones the correct move keeps that the best alternative loses
    "own_stones_lost",   # our stones the correct move gives up that the best alternative keeps
    "opp_stones_freed",  # opponent stones the correct move lets live that the best alternative kills
    "stone_swing_abs",   # total |swing| over occupied points — the life-and-death magnitude
    "empty_swing_abs",   # total |swing| over empty points — territory, for contrast
    "max_stone_delta", "max_stone_pt",
]

HALF = 0.5  # 2.0 of ownership spans opponent-owned to ours, i.e. one stone

# A candidate first move is an (x, y) cell or this sentinel. Passing is always
# available, so it carries no coordinate to check against the legal set.
PASS = "pass"


def cand_to_gtp(cand):
    """Candidate -> the GTP string both the engine and the csv want."""
    return PASS if cand is PASS else xy_to_gtp(*cand)


# ---------------------------------------------------------------- board setup

def build_state(setup_moves, size):
    """Replay the SGF's setup into the position the solver faces. The replay is
    move by move, not a stone placement, so the prisoner count territory scoring
    needs is present — see the rules note in the module docstring."""
    state = GameState(size, TORCH_RULES.copy())
    for color_char, gtp in setup_moves:
        xy = gtp_to_xy(gtp)
        if xy is None:
            continue
        x, y = xy
        state.play(Board.BLACK if color_char.lower() == "b" else Board.WHITE,
                   state.board.loc(x, y))
    return state


def enumerate_legal_cells(board, pla, size):
    """Every empty point legal for `pla`, in board-scan order (y outer, x inner).
    Identical rule to pipeline.extract_activations, so candidate indices match
    that corpus."""
    return [(x, y)
            for y in range(size)
            for x in range(size)
            if board.board[board.loc(x, y)] == Board.EMPTY
            and board.would_be_legal(pla, board.loc(x, y))]


def root_masks(board, pla, size):
    """Boolean (size, size) masks indexed [y][x] with y from the BOTTOM. Taken on
    the ROOT board so both children are scored against one reference set — a
    child board differs wherever its move captured."""
    opp = Board.get_opp(pla)
    ours = np.zeros((size, size), dtype=bool)
    theirs = np.zeros((size, size), dtype=bool)
    for y in range(size):
        for x in range(size):
            c = board.board[board.loc(x, y)]
            if c == pla:
                ours[y, x] = True
            elif c == opp:
                theirs[y, x] = True
    return ours, theirs


def _subtree_has_te(node):
    """TE[1] anywhere at or under `node`."""
    stack = [node]
    while stack:
        n = stack.pop()
        if n.has_property("TE"):
            return True
        stack.extend(list(n))
    return False


def variation_move(node, size):
    """The candidate a root variation plays: an (x, y) cell, PASS, or None when
    the node carries no move at all.

    The two spellings of a pass are both read. sgfmill returns (None, None) for a
    node with no B/W property and (colour, b"") for the empty-value pass, so an
    absent move is told from a pass by the COLOUR, not by the value being falsy.
    This corpus writes its passes as `tt`, which sgf_to_gtp resolves; both are
    legal SGF, and reading either as "no move" would silently drop a refutation.
    """
    colour, move_bytes = node.get_raw_move()
    if colour is None:
        return None
    if not move_bytes:
        return PASS
    gtp = sgf_to_gtp(move_bytes.decode("utf-8"), size)
    return PASS if gtp == PASS else gtp_to_xy(gtp)


def variation_score(node):
    """The SGF's own score for a root variation, in the SOLVER's frame, or None
    when the node carries no parseable evaluation comment.

    Comments are BLACK-perspective, and a root variation is the SOLVER's move, so
    the solver is the mover and a white solver flips the sign. This is the same
    flip `add_variations._candidate_evals` applies; it is repeated here rather
    than imported because that helper drops nodes this module must keep.
    """
    colour, _ = node.get_raw_move()
    if colour is None:
        return None
    comment = node.get("C") if node.has_property("C") else None
    parsed = parse_eval_comment(comment)
    if parsed is None:
        return None
    _, score_black = parsed
    return score_black if colour == "b" else -score_black


def te_marked_alternatives(variation_nodes, size):
    """Refuted first moves the generator READ OUT TO SCORING, as [(move, score)] —
    root variations (excluding the answer at index 0) whose subtree contains a
    TE[1] mark, i.e. a line driven to a double pass. A resolved PASS is one of
    them; see the module docstring for why it is not held out.

    A variation with no parseable evaluation comment is DROPPED rather than
    ranked last: the baseline is the best refutation by score, and a candidate
    with no score cannot be compared to one that has it. Ranking it last would
    quietly make "unannotated" mean "worst"; dropping it makes the problem fall
    back to the candidates that can be ranked, and if none can the row is emitted
    baseline-less exactly as it is for a problem with no resolved refutation.

    Why the mark must be looked for DEEPER and never on the variation itself:
    at the decision node the generator marks only the ANSWER (`mark_children=False`
    on the refuted-first-move call), because a root mark reads as "this is the
    move to find". So a refutation is recognised by a resolved CONTINUATION,
    which is the property wanted here anyway — an unresolved stub is a line
    nobody played out, and its ownership map is an opinion rather than a result.
    """
    out = []
    for node in variation_nodes[1:]:
        mv = variation_move(node, size)
        if mv is None or not _subtree_has_te(node):
            continue
        sc = variation_score(node)
        if sc is None:
            continue
        out.append((mv, sc))
    return out


def setup_move_sequence(setup_moves):
    """The SGF's setup as an analysis-engine `moves` list. Deliberately NOT a
    board snapshot: territory scoring reads the prisoner count, which only the
    sequence carries."""
    return [[color.upper(), gtp] for color, gtp in setup_moves]


# ---------------------------------------------------------------- backends
# Both expose evaluate_children(state, pla, cands) -> (ownership, winrate)
# with ownership (n, size, size) indexed [y][x] from the BOTTOM and positive for
# the SOLVER, and winrate (n,) the solver's.

class EngineBackend:
    """KataGo analysis binary: one maxVisits=1 query per child, concurrently."""

    def __init__(self, size, nn_batch, threads):
        self.size = size
        self.threads = threads
        self.controller = KatagoController()
        self.controller.start_engine(
            get_model(DEFAULT_MODEL), size,
            extra_overrides=f"nnMaxBatchSize={nn_batch},numAnalysisThreads={threads}")
        self.pool = ThreadPoolExecutor(max_workers=threads)

    def evaluate_children(self, state, pla, cands, setup_moves):
        size = self.size
        seq = setup_move_sequence(setup_moves)
        color = "B" if pla == Board.BLACK else "W"

        def one(cand):
            r = self.controller.send_query({
                "initialStones": [],
                "moves": seq + [[color, cand_to_gtp(cand)]],
                "rules": RULES, "komi": KOMI,
                "boardXSize": size, "boardYSize": size,
                "maxVisits": 1,
                "includeOwnership": True,
                "includePolicy": False,
            })
            # Engine ownership is BLACK-positive and row-major from the TOP;
            # flip vertically to this module's y-from-bottom indexing.
            own = np.asarray(r["ownership"], dtype=np.float32).reshape(size, size)[::-1]
            return own, float(r["rootInfo"]["winrate"])

        results = list(self.pool.map(one, cands))
        # Black-absolute -> solver frame.
        sign = 1.0 if pla == Board.BLACK else -1.0
        own = np.stack([sign * o for o, _ in results])
        wr = np.array([w if pla == Board.BLACK else 1.0 - w for _, w in results],
                      dtype=np.float32)
        return own, wr

    def close(self):
        self.pool.shutdown(wait=False)
        self.controller.stop_engine()


class TorchBackend:
    """Vendored PyTorch net: every child in one batched forward pass."""

    def __init__(self, size, ckpt, device, max_batch):
        import torch
        from katago.game.features import Features
        from katago.train.load_model import load_model
        self.torch = torch
        self.size = size
        self.max_batch = max_batch
        self.model, _, _ = load_model(ckpt, use_swa=False, device=device,
                                      pos_len=size, verbose=False)
        self.model.eval()
        self.features = Features(self.model.config, self.model.pos_len)

    def _child_features(self, state, pla, cands):
        """play/undo keeps the move history correct (the net reads recent moves),
        which replaying the setup per candidate would too, at ~50x the cost."""
        bins, globs = [], []
        for cand in cands:
            loc = (Board.PASS_LOC if cand is PASS
                   else state.board.loc(cand[0], cand[1]))
            state.play(pla, loc)
            b, g = state.get_input_features(self.features)
            state.undo()
            bins.append(b)
            globs.append(g)
        return np.concatenate(bins, axis=0), np.concatenate(globs, axis=0)

    def evaluate_children(self, state, pla, cands, setup_moves):
        # setup_moves is unused: the state already carries the replayed history.
        torch = self.torch
        bin_in, glob_in = self._child_features(state, pla, cands)
        owns, wrs = [], []
        for i in range(0, bin_in.shape[0], self.max_batch):
            b = torch.tensor(bin_in[i:i + self.max_batch], dtype=torch.float32,
                             device=self.model.device)
            g = torch.tensor(glob_in[i:i + self.max_batch], dtype=torch.float32,
                             device=self.model.device)
            with torch.no_grad():
                out = self.model.postprocess_output(self.model(b, g, input_meta=None))[0]
                # Mover-framed: the child's mover is the solver's opponent.
                own = -torch.tanh(out[4])[:, 0]
                wr = torch.softmax(out[1], dim=1)[:, 1]   # opponent's loss = solver's win
            owns.append(own.cpu().numpy())
            wrs.append(wr.cpu().numpy())
        return np.concatenate(owns, axis=0), np.concatenate(wrs, axis=0)

    def close(self):
        pass


# ---------------------------------------------------------------- per problem

def process_one_sgf(sgf_path, backend):
    with open(sgf_path, "rb") as f:
        game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    size = game.get_size()
    setup_moves, variation_nodes = parse_tsumego_sgf(game)

    correct_color, correct_mb = variation_nodes[0].get_raw_move()
    pla = Board.BLACK if correct_color.lower() == "b" else Board.WHITE
    correct_xy = gtp_to_xy(sgf_to_gtp(correct_mb.decode("utf-8"), size))
    if correct_xy is None:
        raise ValueError("correct variation is a pass")

    state = build_state(setup_moves, size)
    legal = set(enumerate_legal_cells(state.board, pla, size))
    if correct_xy not in legal:
        raise ValueError(f"correct move {correct_xy} is not a legal cell")

    # Only refutations the generator resolved are eligible, so exactly the
    # candidates that need evaluating are evaluated — the answer plus, typically,
    # one alternative, rather than every legal cell. A pass needs no legality
    # check; a stone move is dropped if the SGF disagrees with the board.
    alts = [(m, sc) for m, sc in te_marked_alternatives(variation_nodes, size)
            if m is PASS or m in legal]

    ours, theirs = root_masks(state.board, pla, size)
    row = {
        "tid": os.path.basename(sgf_path)[:-4],
        "size": size,
        "solver": "B" if pla == Board.BLACK else "W",
        "n_root_vars": len(variation_nodes) - 1,
        "n_te_alts": len(alts),
        "correct_move": xy_to_gtp(*correct_xy),
        "score_correct": variation_score(variation_nodes[0]),
        "n_stones_solver": int(ours.sum()),
        "n_stones_opp": int(theirs.sum()),
    }
    if not alts:
        # No refutation was read to a conclusion, so there is nothing sound to
        # compare against. Emit the row (n_te_alts = 0 says why) so a resume does
        # not retry it, and so the gap is visible rather than silently absent.
        return row

    # The baseline is chosen from the SGF BEFORE anything is evaluated, so the
    # engine is asked only for the maps that get used: the answer and the one
    # refutation the corpus's own scores name as strongest.
    best_alt, best_alt_score = max(alts, key=lambda t: t[1])
    cands = [correct_xy, best_alt]
    correct_idx, alt_idx = 0, 1
    own, wr = backend.evaluate_children(state, pla, cands, setup_moves)

    base, good = own[alt_idx], own[correct_idx]   # solver frame, both in [-1, +1]
    delta = good - base                           # [-2, +2], + = correct move gives us more
    stones = ours | theirs

    # A stone's ALIVENESS is (ownership in ITS OWN colour's frame + 1) / 2, so a
    # swing of 2.0 in ownership is one whole stone changing hands and every
    # column below is a fractional count of stones. For the solver's stones
    # aliveness rises with `delta`; for the opponent's it falls, so their sign
    # conventions are mirrored throughout.
    #
    # BASELINE IS THE BEST ALTERNATIVE, not the root position: the root already
    # assumes the solver finds the answer (measured: root sits ~2.4x closer to
    # the correct child than to the best alternative), so "currently dead" has
    # to mean "dead if you miss it", which is what the alternative is.
    gain, loss = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    dead_ours, alive_ours = ours & (base < 0), ours & (base > 0)
    alive_theirs, dead_theirs = theirs & (base < 0), theirs & (base > 0)

    def s(mask, arr):
        return round(float(arr[mask].sum() * HALF), 4)

    parts = {
        # solver's stones
        "own_dead_revived": s(dead_ours, gain),     # dead if missed -> alive: the group you save
        "own_alive_secured": s(alive_ours, gain),   # alive either way, made safer
        "own_alive_killed": s(alive_ours, loss),    # alive if missed -> dead: self-inflicted
        "own_dead_deadened": s(dead_ours, loss),    # dead either way, made deader
        # opponent's stones (aliveness moves opposite to delta)
        "opp_alive_killed": s(alive_theirs, gain),  # alive if missed -> dead: the group you kill
        "opp_dead_deadened": s(dead_theirs, gain),  # dead either way, made deader
        "opp_dead_revived": s(dead_theirs, loss),   # dead if missed -> alive: you let it out
        "opp_alive_secured": s(alive_theirs, loss), # alive either way, made safer
    }
    row.update(parts)
    row.update({
        "alt_move": cand_to_gtp(best_alt),
        "score_alt": best_alt_score,
        "wr_correct": round(float(wr[correct_idx]), 4),
        "wr_alt": round(float(wr[alt_idx]), 4),
        # The four totals are the sums of the eight parts by construction, so a
        # reader can use either granularity without them disagreeing.
        "opp_stones_won": round(parts["opp_alive_killed"] + parts["opp_dead_deadened"], 4),
        "own_stones_saved": round(parts["own_dead_revived"] + parts["own_alive_secured"], 4),
        "own_stones_lost": round(parts["own_alive_killed"] + parts["own_dead_deadened"], 4),
        "opp_stones_freed": round(parts["opp_dead_revived"] + parts["opp_alive_secured"], 4),
        "stone_swing_abs": round(float(np.abs(delta[stones]).sum() * HALF), 4),
        "empty_swing_abs": round(float(np.abs(delta[~stones]).sum() * HALF), 4),
    })
    if stones.any():
        masked = np.where(stones, np.abs(delta), -1.0)
        my, mx = np.unravel_index(int(np.argmax(masked)), masked.shape)
        row["max_stone_delta"] = round(float(delta[my, mx] * HALF), 4)
        row["max_stone_pt"] = xy_to_gtp(int(mx), int(my))
    return row


# ---------------------------------------------------------------- CSV state

def repair_trailing_line(path):
    """Drop a partial final line left by a kill mid-write, so the append that
    follows starts on a clean record boundary."""
    if not os.path.exists(path):
        return
    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return
        chunk = min(1 << 16, size)
        f.seek(size - chunk)
        data = f.read(chunk)
        idx = data.rfind(b"\n")
        f.truncate(size - chunk + idx + 1 if idx != -1 else 0)


def load_done(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return done
        for rec in reader:
            if len(rec) == len(header) and rec[0]:
                done.add(rec[0])
    return done


# ---------------------------------------------------------------- driver

def make_backend(size, args):
    if args.backend == "engine":
        return EngineBackend(size, args.nn_batch, args.engine_threads)
    import torch
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    return TorchBackend(size, args.ckpt, device, args.max_batch)


def run_size(size, args):
    in_dir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
    if not os.path.isdir(in_dir):
        print(f"\n[size {size}] no tsumegos dir at {in_dir}; skipping", flush=True)
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_path = args.out
    err_path = args.out + ".errors.log"

    repair_trailing_line(out_path)
    done = load_done(out_path)

    sgf_files = sorted(f for f in os.listdir(in_dir) if f.endswith(".sgf"))
    if args.limit is not None:
        sgf_files = sgf_files[: args.limit]
    todo = [f for f in sgf_files if f[:-4] not in done]
    print(f"\n[size {size}] {len(sgf_files)} SGFs, {len(done)} already done, "
          f"{len(todo)} to do -> {out_path}", flush=True)
    if not todo:
        return

    print(f"[size {size}] starting {args.backend} backend", flush=True)
    backend = make_backend(size, args)

    new_file = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    n_done = n_fail = 0
    t0 = time.time()
    try:
        with open(out_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="raise")
            if new_file:
                writer.writeheader()
            for i, fname in enumerate(todo, 1):
                try:
                    row = process_one_sgf(os.path.join(in_dir, fname), backend)
                except Exception as e:  # noqa: BLE001
                    # A diagnostic sweep, not a pipeline artifact: nothing
                    # downstream is keyed to completeness, so one bad SGF must not
                    # cost the run. Recorded and counted, never silently dropped.
                    if args.strict:
                        raise RuntimeError(f"[size {size}] failed on {fname}") from e
                    n_fail += 1
                    with open(err_path, "a") as ef:
                        ef.write(f"{fname}\t{type(e).__name__}: {e}\n")
                    continue
                writer.writerow({k: row.get(k, "") for k in FIELDS})
                n_done += 1
                if n_done % args.flush_every == 0:
                    fh.flush()
                    os.fsync(fh.fileno())
                if i % args.progress_every == 0 or i == len(todo):
                    rate = i / (time.time() - t0)
                    print(f"[size {size}] {i}/{len(todo)}  {rate:.2f} probs/s  "
                          f"ETA {(len(todo) - i) / rate / 3600:.2f} h  failures {n_fail}",
                          flush=True)
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        backend.close()

    print(f"[size {size}] wrote {n_done} rows in {(time.time()-t0)/3600:.2f} h", flush=True)
    if n_fail:
        print(f"[size {size}] *** {n_fail} FAILURES — see {err_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("engine", "torch"), default="engine",
                        help="engine = KataGo analysis binary (default, ~2x faster); "
                             "torch = vendored PyTorch net.")
    parser.add_argument("--size", type=int, choices=list(ALL_BOARD_SIZES), default=None,
                        help="Board size (default: every size in ALL_BOARD_SIZES).")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output CSV (default: {os.path.relpath(DEFAULT_OUT, PROJECT_ROOT)}).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N SGFs per size (debugging).")
    parser.add_argument("--engine-threads", type=int, default=64,
                        help="engine: concurrent queries / numAnalysisThreads (default: 64).")
    parser.add_argument("--nn-batch", type=int, default=64,
                        help="engine: nnMaxBatchSize override. The cfg's tuned value is "
                             "sized for interactive search and throttles a bulk sweep.")
    parser.add_argument("--max-batch", type=int, default=96,
                        help="torch: cap on child positions per forward pass (default: 96).")
    parser.add_argument("--flush-every", type=int, default=200,
                        help="fsync the CSV every N rows (default: 200).")
    parser.add_argument("--progress-every", type=int, default=200,
                        help="Print a progress line every N problems (default: 200).")
    parser.add_argument("--strict", action="store_true",
                        help="Abort on the first failed problem instead of logging it.")
    parser.add_argument("--device", default=None,
                        help="torch: cuda or cpu (default: cuda if available).")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="torch: PyTorch checkpoint.")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)  # silence KataGo's config dump

    sizes = [args.size] if args.size else list(ALL_BOARD_SIZES)
    print(f"Ownership swing extraction, {args.backend} backend; sizes {sizes}", flush=True)
    for size in sizes:
        run_size(size, args)


if __name__ == "__main__":
    main()
