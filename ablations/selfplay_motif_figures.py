#!/usr/bin/env python3
"""How often each shipped motif rule fires on the positions of ordinary play,
against how often it fires on the problem set — split into the problems rated
at kyu level and at dan level.

**Self-play.** KataGo plays itself on 9×9 from the problem generator's own
random opening (a fixed number of uniform non-edge moves), then both sides play
the top move of a VISITS-visit search until two consecutive passes. Every
position after the random opening is treated as a problem whose ANSWER is the
move actually played — there is no single-winning-move gate, which is exactly
the difference from the problem set this measures. Each rule is the shipped one,
fed its usual quantities with every problem ingredient mapped onto its game
analogue:

  answer            the played move
  refuted try       the best other move of the position's own search (a pass
                    is never taken), falling back to the top policy move
  answer's line     the game continuation (replies, ko plies)
  refuted line      a search of the position after the refuted try; its top
                    move is the reply
  root variations   {answer, refuted try}: their child searches supply the
                    ownership maps life and death / semeai compare
  pass baseline     the position after the mover passes. Territory and
                    influence compare the ownership after the played move with
                    the ownership after that pass, and yose the two scores,
                    both read the way the problem
                    set's are (pipeline.extract_ownership_maps: one visit, all
                    eight symmetries averaged), so the two sides of the contrast
                    are the same measurement
  prefix            moves played before the position, random opening included

N_GAMES is sized to be barely sufficient: it puts every rule's share inside
about a percentage point at 95% (clustered by game, since positions of one game
are not independent) and the rarest rule's inside about a fifth of itself. The
interval is emitted beside the shares. Game `i` draws its opening from
`Random(SEED + i)`, so the game set is fixed; the engine's batched evaluation
is the one remaining source of run-to-run wander, and the interval covers it.

**Corpus.** The 9×9 problem set's shipped rule labels, split at the 1k/1d
boundary by the rank each problem's raw rating maps to through the shipped
rating→rank curve.

GPU (KataGo). Minutes at 9×9.

    ./.venv/bin/python3 -m ablations.selfplay_motif_figures
    ./.venv/bin/python3 -m ablations --only selfplay_motif_figures
"""

import argparse
import json
import os
import random
import sys
import threading
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from utils.constants import KOMI, RULES  # noqa: E402
from utils.katago.board_render import simulate  # noqa: E402
from utils.katago.controller import KatagoController  # noqa: E402
from utils.katago.coords import gtp_to_sgfmill_point, sgfmill_point_to_sgf  # noqa: E402
from utils.katago.models import DEFAULT_MODEL, get_model  # noqa: E402
from utils.katago.problem_finder import RANDOM_MOVE_COUNT, ProblemFinder  # noqa: E402
from pipeline.extract_ownership_maps import SYMMETRIES, VISITS as MAP_VISITS  # noqa: E402
from utils.katago.queries import KatagoQueries  # noqa: E402
from utils.rating import motif_rules as mr  # noqa: E402
from utils.rating.ratings_io import interp_extrap_vec, load_M, load_size_value_maps  # noqa: E402

TIER = "methodology"

SIZE = 9
N_GAMES = 1000
VISITS = 3
SEED = 0
THREADS = 64       # games in flight; the engine batches their queries
NN_BATCH = 64

LABELS_PATH = os.path.join(PROJECT_ROOT, "axes", "labels.json")
PROXIES_PATH = os.path.join(PROJECT_ROOT, "axes", "motif_proxies.npz")
RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")

HALF = 0.5  # an ownership delta of 2.0 is one stone changing hands


def slug(rule):
    return rule.replace(" ", "_")


def colour_at(i):
    return "B" if i % 2 == 0 else "W"


def to_sgf(gtp):
    return sgfmill_point_to_sgf(gtp_to_sgfmill_point(gtp, SIZE), SIZE)


def dist(a, b):
    return mr._dist(to_sgf(a), to_sgf(b))


# ---------------------------------------------------------------- self-play

class Engine:
    def __init__(self):
        self.controller = KatagoController()
        self.controller.start_engine(
            get_model(DEFAULT_MODEL), SIZE,
            extra_overrides=f"nnMaxBatchSize={NN_BATCH},numAnalysisThreads={THREADS}")
        self.queries = KatagoQueries(self.controller, SIZE)

    def search(self, moves):
        """A VISITS search after `moves`, with ownership and policy. Everything
        comes back BLACK-absolute (bots/analysis.cfg)."""
        r = self.controller.send_query({
            "moves": [[colour_at(i), m] for i, m in enumerate(moves)],
            "initialStones": [], "rules": RULES, "komi": KOMI,
            "boardXSize": SIZE, "boardYSize": SIZE,
            "maxVisits": VISITS,
            "includeOwnership": True, "includePolicy": True,
        })
        require("rootInfo" in r and "ownership" in r, f"engine error: {r.get('error', r)}")
        return r

    def ownership_map(self, moves):
        """(map, score) after `moves`, as the pass-referenced rules read them:
        one visit, symmetries averaged, BLACK-absolute, the map [y][x] from the
        top."""
        r = self.controller.send_query({
            "moves": [[colour_at(i), m] for i, m in enumerate(moves)],
            "initialStones": [], "rules": RULES, "komi": KOMI,
            "boardXSize": SIZE, "boardYSize": SIZE,
            "maxVisits": MAP_VISITS, "includeOwnership": True,
            "overrideSettings": {"rootNumSymmetriesToSample": SYMMETRIES},
        })
        require("ownership" in r, f"engine error: {r.get('error', r)}")
        return (np.asarray(r["ownership"], dtype=np.float32).reshape(SIZE, SIZE),
                r["rootInfo"]["scoreLead"])


def play_game(engine, finder, rng):
    """(moves, searches, alt, area): searches[i] is the search of the position
    before moves[i]; alt[i] is (refuted try, the search after it) per scored
    position; area[i] is the (after the played move, after a pass) ownership
    pair the area rules compare, then the BLACK-absolute score of the played
    move less that of the pass, which yose reads."""
    moves = finder.generate_initial_position(rng)
    searches, alt, area = {}, {}, {}
    cap = RANDOM_MOVE_COUNT + 2 * SIZE * SIZE
    while len(moves) < cap:
        r = engine.search(moves)
        i = len(moves)
        searches[i] = r
        infos = r.get("moveInfos") or []
        a = infos[0]["move"] if infos else "pass"
        if a != "pass":
            b = next((mi["move"] for mi in infos[1:] if mi["move"] not in (a, "pass")), None)
            if b is None:
                pol = r["policy"]
                for k in sorted(range(SIZE * SIZE), key=lambda k: pol[k], reverse=True):
                    mv = engine.queries.get_move_from_idx(k)
                    if pol[k] > 0 and mv != a:
                        b = mv
                        break
            if b is not None:
                alt[i] = (b, engine.search(moves + [b]))
                (own_a, score_a) = engine.ownership_map(moves + [a])
                (own_p, score_p) = engine.ownership_map(moves + ["pass"])
                area[i] = (own_a, own_p, score_a - score_p)
        moves.append(a)
        if len(moves) >= 2 and moves[-1] == "pass" and moves[-2] == "pass":
            break
    return moves, searches, alt, area


def ownership_parts(board, good, base, mover):
    """The ownership-swing parts in the solver's frame, in stones: `good` is the
    map after the answer, `base` after the refuted try, both [y][x] from the top
    like simulate's board."""
    arr = np.array(board)
    ours, theirs = arr == mover, arr == ("W" if mover == "B" else "B")
    delta = good - base
    gain, loss = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    dead_ours, alive_ours = ours & (base < 0), ours & (base > 0)
    alive_theirs, dead_theirs = theirs & (base < 0), theirs & (base > 0)

    def s(mask, x):
        return float(x[mask].sum() * HALF)
    return {
        "own_dead_revived": s(dead_ours, gain), "own_alive_secured": s(alive_ours, gain),
        "own_alive_killed": s(alive_ours, loss), "own_dead_deadened": s(dead_ours, loss),
        "opp_alive_killed": s(alive_theirs, gain), "opp_dead_deadened": s(dead_theirs, gain),
        "opp_dead_revived": s(dead_theirs, loss), "opp_alive_secured": s(alive_theirs, loss),
    }


def area_verdict(board, a, own_answer, own_pass, sign):
    """rule_area on one position: "territory", "influence" or None."""
    arr = np.array(board)
    answer = np.zeros((SIZE, SIZE), bool)
    x, y = mr._xy(to_sgf(a))
    answer[y, x] = True
    q = mr.area_quantities((sign * own_answer + 1) / 2, (sign * own_pass + 1) / 2,
                           arr != ".", answer)
    return mr.rule_area(*(np.asarray(v).item() for v in q))


def label_game(moves, searches, alt, area, rules):
    """Per rule, how many of the game's scored positions it fires on; and the
    number of scored positions."""
    sgf_moves = [(colour_at(i), to_sgf(m)) for i, m in enumerate(moves) if m != "pass"]
    nonpass_before = np.cumsum([0] + [m != "pass" for m in moves])
    hits = dict.fromkeys(rules, 0)
    n = 0
    for i in sorted(alt):
        child_a = searches.get(i + 1)
        if child_a is None:
            continue
        a, mover = moves[i], colour_at(i)
        b, child_b = alt[i]
        setup = sgf_moves[:nonpass_before[i]]
        board, _, ko = simulate(setup, SIZE)
        root_ko = ko is not None

        # The answer's line is the game continuation with passes skipped, as the
        # problem-side line walk skips them.
        a_line = [(colour_at(j), moves[j]) for j in range(i, len(moves))
                  if moves[j] != "pass"][:mr.KO_DEPTH]
        reply_b = (child_b.get("moveInfos") or [{"move": "pass"}])[0]["move"]
        b_line = ([(mover, b)] + ([(colour_at(i + 1), reply_b)] if reply_b != "pass" else []))
        b_line = b_line[:mr.KO_DEPTH]
        ko_line = any(simulate(setup + [(c, to_sgf(m)) for c, m in line[:k]], SIZE)[2] is not None
                      for line in (a_line, b_line) for k in range(1, len(line) + 1))

        opp_last = next((m for j, m in reversed(list(enumerate(moves[:i])))
                         if m != "pass" and colour_at(j) != mover), None)
        a_reply = a_line[1] if len(a_line) > 1 else None
        sign = 1.0 if mover == "B" else -1.0
        own = ownership_parts(
            board,
            sign * np.asarray(child_a["ownership"], dtype=np.float32).reshape(SIZE, SIZE),
            sign * np.asarray(child_b["ownership"], dtype=np.float32).reshape(SIZE, SIZE),
            mover)
        own_a, own_p, gain = area[i]
        region = area_verdict(board, a, own_a, own_p, sign)
        verdict = {
            "life and death": mr.rule_life_and_death(own),
            "yose": bool(mr.rule_yose(sign * gain)),
            "ko": root_ko or (mr.KO_DEPTH > 0 and ko_line),
            "opening": mr.rule_opening(i, SIZE),
            "tenuki": mr.rule_tenuki(dist(a, opp_last) if opp_last else None, root_ko),
            "sente": mr.rule_sente(
                dist(a, a_reply[1]) if a_reply and a_reply[0] != mover else None,
                dist(b, reply_b) if reply_b != "pass" else None),
            "semeai": mr.rule_semeai(own),
            "territory": region == "territory",
            "influence": region == "influence",
        }
        require(set(rules) <= set(verdict),
                f"no game analogue written for rule(s) {sorted(set(rules) - set(verdict))}")
        n += 1
        for r in rules:
            hits[r] += bool(verdict[r])
    return n, hits


def selfplay(rules):
    """Play the fixed game set; per-game (positions, hits) in game order."""
    engine = Engine()
    finder = ProblemFinder(engine.queries)
    results = [None] * N_GAMES
    lock = threading.Lock()
    next_game = [0]
    errors = []
    t0 = time.time()

    def worker():
        while True:
            with lock:
                g = next_game[0]
                next_game[0] += 1
            if g >= N_GAMES or errors:
                return
            try:
                moves, searches, alt, area = play_game(engine, finder, random.Random(SEED + g))
                results[g] = label_game(moves, searches, alt, area, rules)
            except Exception as e:
                errors.append(f"game {g}: {type(e).__name__}: {e}")
                return
            done = sum(r is not None for r in results)
            if done % 100 == 0:
                print(f"  {done}/{N_GAMES} games, {time.time() - t0:.0f}s", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(THREADS)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        engine.controller.stop_engine()
    require(not errors, f"self-play failed: {errors[:1]}")
    require(all(r is not None for r in results), "self-play left games unplayed")
    return results


def clustered_share(n, h):
    """Share of positions and its 95% half-width, clustered by game (a ratio
    estimator over games)."""
    n, h = np.asarray(n, float), np.asarray(h, float)
    g = len(n)
    p = h.sum() / n.sum()
    se = np.sqrt(((h - p * n) ** 2).sum() / (g - 1) / g) / n.mean()
    return p, 1.96 * se


# ---------------------------------------------------------------- corpus

def corpus_bands(rules):
    """{band: {rule: share}} over the SIZE problem set, and the band counts."""
    mp = np.load(PROXIES_PATH)
    require({"tid", "size", "labels", "rules"} <= set(mp.files),
            f"{PROXIES_PATH}: unexpected fields {mp.files}")
    stored = list(mp["rules"])
    require(set(rules) <= set(stored), f"motif proxies lack rule(s) {set(rules) - set(stored)}")
    m = mp["size"] == SIZE
    tids, lab = mp["tid"][m], mp["labels"][m]
    ratings = load_size_value_maps(RATINGS_DIR, SIZE)[0]
    require(set(tids) == set(ratings),
            f"motif proxies and ratings cover different {SIZE}×{SIZE} problems "
            f"({len(set(tids) ^ set(ratings))} differ)")
    rank = interp_extrap_vec(np.array([ratings[t] for t in tids]), *load_M(RATINGS_DIR))
    out, counts = {}, {}
    for band, sel in (("kyu", rank < 0), ("dan", rank >= 0)):
        require(sel.sum() > 0, f"no {band} problems")
        counts[band] = int(sel.sum())
        out[band] = {r: float(lab[sel, stored.index(r)].mean()) for r in rules}
    return out, counts


# ---------------------------------------------------------------- figures

def measure():
    with open(LABELS_PATH) as f:
        rules = json.load(f)["components"]
    require(rules and all("/" not in r for r in rules),
            f"expected one rule per axis in {LABELS_PATH}, got {rules}")
    bands, counts = corpus_bands(rules)
    games = selfplay(rules)
    n = [g[0] for g in games]
    sp = {r: clustered_share(n, [g[1][r] for g in games]) for r in rules}
    return {"rules": rules, "bands": bands, "counts": counts, "selfplay": sp,
            "positions": int(sum(n))}


def figures(m=None):
    m = m or measure()
    figs = [
        Figure("motifs.selfplay_games", N_GAMES, "d",
               f"{SIZE}×{SIZE} self-play games, {VISITS} visits a move on both sides"),
        Figure("motifs.selfplay_positions", m["positions"], "~1000",
               "self-play positions scored, the random opening excluded"),
        Figure("motifs.selfplay_ci95_max_pp", 100 * max(h for _, h in m["selfplay"].values()), ".1f",
               "largest 95% half-width over the rules' self-play shares, in percentage "
               "points, clustered by game"),
        Figure("motifs.corpus_kyu_problems", m["counts"]["kyu"], "~1000",
               f"{SIZE}×{SIZE} problems rated at kyu level"),
        Figure("motifs.corpus_dan_problems", m["counts"]["dan"], "~1000",
               f"{SIZE}×{SIZE} problems rated at dan level"),
    ]
    for r in m["rules"]:
        s = slug(r)
        figs += [
            Figure(f"motifs.selfplay.{s}_pct", m["selfplay"][r][0], "%.1f",
                   f"share of {SIZE}×{SIZE} self-play positions the '{r}' rule fires on, "
                   f"the played move standing in for the answer"),
            Figure(f"motifs.corpus_kyu.{s}_pct", m["bands"]["kyu"][r], "%.1f",
                   f"share of kyu-rated {SIZE}×{SIZE} problems the '{r}' rule fires on"),
            Figure(f"motifs.corpus_dan.{s}_pct", m["bands"]["dan"][r], "%.1f",
                   f"share of dan-rated {SIZE}×{SIZE} problems the '{r}' rule fires on"),
        ]
    return figs


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    m = measure()
    print(f"\n{SIZE}×{SIZE}: share of positions each rule fires on")
    print(f"  corpus kyu = {m['counts']['kyu']} problems rated below 1d; corpus dan = "
          f"{m['counts']['dan']} rated 1d and up;")
    print(f"  self-play = {m['positions']} positions from {N_GAMES} games at {VISITS} visits, "
          f"played move as the answer, ± = 95% half-width clustered by game\n")
    print(f"  {'rule':16s} {'corpus kyu':>11s} {'corpus dan':>11s} {'self-play':>16s}")
    for r in m["rules"]:
        p, h = m["selfplay"][r]
        print(f"  {r:16s} {m['bands']['kyu'][r]:11.1%} {m['bands']['dan'][r]:11.1%} "
              f"{p:9.1%} ±{100 * h:4.1f}")


if __name__ == "__main__":
    main()
