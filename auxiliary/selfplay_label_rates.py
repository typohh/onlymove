"""How often each shipped motif label fires on the positions of real games —
KataGo self-play — against how often it fires on the tsumego corpus.

Every position of every game is treated as a problem whose ANSWER is the move
actually played; there is no single-winning-move gate. The rules are the
shipped ones (utils.rating.motif_rules) fed the same quantities the corpus
pipeline feeds them, with each tsumego ingredient mapped onto its game analogue:

  answer            the played move
  refuted try       the best other move of the position's own search (a pass
                    is never taken), falling back to the top policy move
  answer's line     the game continuation (replies, ko plies)
  refuted line      the refuted try's child search: its top move is the reply
  root variations   {answer, refuted try} — their child searches supply the
                    (winrate, score) pairs yose ranks and the ownership maps
                    life and death / semeai compare
  prefix            moves played before the position (random opening included)

Games open with the generator's own RANDOM_MOVE_COUNT uniform non-edge moves
(ProblemFinder.generate_initial_position), so the opening mix matches the
corpus; those moves are not scored as positions. Both sides then play the top
move of a `--visits` search until two consecutive passes.

Throughput comes from the engine's batching: `--threads` games run at once,
each blocking on its own query, so the analysis engine always has a batch to
fill.

Runs until killed. Every `--every` seconds a snapshot is APPENDED to
<out>/summary.txt and the per-position rows gathered since the last snapshot
are appended to <out>/positions.jsonl, so a kill loses at most one interval.

    setsid nohup ./.venv/bin/python3 -u -m auxiliary.selfplay_label_rates 9 \\
        > /tmp/selfplay_labels.log 2>&1 < /dev/null &
"""

import argparse
import json
import os
import random
import signal
import sys
import threading
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.constants import KOMI, RULES  # noqa: E402
from utils.katago.board_render import simulate  # noqa: E402
from utils.katago.controller import KatagoController  # noqa: E402
from utils.katago.coords import gtp_to_sgfmill_point, sgfmill_point_to_sgf  # noqa: E402
from utils.katago.models import DEFAULT_MODEL, get_model  # noqa: E402
from utils.katago.problem_finder import RANDOM_MOVE_COUNT, ProblemFinder  # noqa: E402
from utils.katago.queries import KatagoQueries  # noqa: E402
from utils.rating import motif_rules as mr  # noqa: E402

HALF = 0.5  # ownership delta of 2.0 is one stone changing hands (extract_ownership_swing)
OWN_PARTS = ("own_dead_revived", "own_alive_secured", "own_alive_killed", "own_dead_deadened",
             "opp_alive_killed", "opp_dead_deadened", "opp_dead_revived", "opp_alive_secured")
# The corpus-mix column reweights by EXACT move number, not a bucket: every
# corpus problem is black to move (an even prefix), so a bucket straddling
# parities mixes in white-to-move positions the corpus never holds, and the
# opening rule's cutoff falls inside a bucket.
BUCKET = {9: 1, 13: 1}

stop_event = threading.Event()


def colour_at(i):
    return "B" if i % 2 == 0 else "W"


def to_sgf(gtp, size):
    return sgfmill_point_to_sgf(gtp_to_sgfmill_point(gtp, size), size)


def dist(a, b, size):
    return mr._dist(to_sgf(a, size), to_sgf(b, size))


class Engine:
    def __init__(self, size, visits, threads, batch):
        self.size, self.visits = size, visits
        self.controller = KatagoController()
        self.controller.start_engine(
            get_model(DEFAULT_MODEL), size,
            extra_overrides=f"nnMaxBatchSize={batch},numAnalysisThreads={threads}")
        self.queries = KatagoQueries(self.controller, size)

    def search(self, moves):
        """A `visits` search of the position after `moves`, with ownership and
        policy. Everything is BLACK-absolute (bots/analysis.cfg)."""
        return self.controller.send_query({
            "moves": [[colour_at(i), m] for i, m in enumerate(moves)],
            "initialStones": [], "rules": RULES, "komi": KOMI,
            "boardXSize": self.size, "boardYSize": self.size,
            "maxVisits": self.visits,
            "includeOwnership": True, "includePolicy": True,
        })


def play_game(engine, finder):
    """One self-play game: (moves, searches, alt) where searches[i] is the search
    of the position before moves[i] (and after the last move), and alt[i] is
    (refuted_move, its child search) for each scored position."""
    size = engine.size
    moves = finder.generate_initial_position()
    searches, alt = {}, {}
    cap = RANDOM_MOVE_COUNT + 2 * size * size
    while len(moves) < cap and not stop_event.is_set():
        r = engine.search(moves)
        i = len(moves)
        searches[i] = r
        infos = r.get("moveInfos") or []
        a = infos[0]["move"] if infos else "pass"
        if a != "pass":
            b = next((mi["move"] for mi in infos[1:] if mi["move"] not in (a, "pass")), None)
            if b is None:
                pol = r["policy"]
                order = sorted(range(size * size), key=lambda k: pol[k], reverse=True)
                b = next((engine.queries.get_move_from_idx(k) for k in order
                          if pol[k] > 0 and engine.queries.get_move_from_idx(k) != a), None)
            if b is not None:
                alt[i] = (b, engine.search(moves + [b]))
        moves.append(a)
        if len(moves) >= 2 and moves[-1] == "pass" and moves[-2] == "pass":
            searches[len(moves)] = None
            break
    return moves, searches, alt


def mover_eval(search, mover):
    ri = search["rootInfo"]
    wr, sc = 100.0 * ri["winrate"], ri["scoreLead"]
    return (wr, sc) if mover == "B" else (100.0 - wr, -sc)


def own_grid(search, size):
    """Black-positive ownership, [y][x] with y from the TOP — the same indexing
    as simulate's board, which reads SGF coordinates."""
    return np.asarray(search["ownership"], dtype=np.float32).reshape(size, size)


def ownership_parts(board, good, base, mover):
    """extract_ownership_swing's eight parts, solver frame, in stones."""
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


def label_game(moves, searches, alt, size):
    """One row per scored position: the rule inputs and the verdicts."""
    sgf_moves = [(colour_at(i), to_sgf(m, size)) for i, m in enumerate(moves) if m != "pass"]
    # sgf_moves index for the position before moves[i]: count of non-pass moves before i
    nonpass_before = np.cumsum([0] + [m != "pass" for m in moves])
    rows = []
    for i in sorted(alt):
        a = moves[i]
        mover = colour_at(i)
        child_a = searches.get(i + 1)
        if child_a is None:
            continue
        b, child_b = alt[i]
        setup = sgf_moves[:nonpass_before[i]]
        board, _, ko = simulate(setup, size)
        root_ko = ko is not None

        # answer line: the game continuation, passes skipped (mr._line_moves)
        a_line = [(colour_at(j), moves[j]) for j in range(i, len(moves)) if moves[j] != "pass"][:mr.KO_DEPTH]
        reply_b = (child_b.get("moveInfos") or [{"move": "pass"}])[0]["move"]
        b_line = [(mover, b)] + ([(colour_at(i + 1), reply_b)] if reply_b != "pass" else [])
        b_line = b_line[:mr.KO_DEPTH]
        ko_line = False
        for line in (a_line, b_line):
            for k in range(1, len(line) + 1):
                if simulate(setup + [(c, to_sgf(m, size)) for c, m in line[:k]], size)[2] is not None:
                    ko_line = True
        ko_hit = root_ko or (mr.KO_DEPTH > 0 and ko_line)

        opp_last = next((m for j, m in reversed(list(enumerate(moves[:i])))
                         if m != "pass" and colour_at(j) != mover), None)
        opp_dist = dist(a, opp_last, size) if opp_last else None
        a_reply = (a_line[1] if len(a_line) > 1 else None)
        ans_reply = dist(a, a_reply[1], size) if a_reply and a_reply[0] != mover else None
        ref_reply = dist(b, reply_b, size) if reply_b != "pass" else None

        evals = [mover_eval(child_a, mover), mover_eval(child_b, mover)]
        sign = 1.0 if mover == "B" else -1.0
        own = ownership_parts(board, sign * own_grid(child_a, size),
                              sign * own_grid(child_b, size), mover)
        labels = {
            "life and death": mr.rule_life_and_death(own),
            "yose": mr.rule_yose(evals),
            "ko": ko_hit,
            "opening": mr.rule_opening(i, size),
            "tenuki": mr.rule_tenuki(opp_dist, root_ko),
            "sente": mr.rule_sente(ans_reply, ref_reply),
            "semeai": mr.rule_semeai(own),
        }
        rows.append({"move": i, "answer": a, "refuted": b, "reply_b": reply_b,
                     "evals": [[round(w, 2), round(s, 2)] for w, s in evals],
                     "own": {k: round(v, 3) for k, v in own.items()},
                     "opp_dist": opp_dist, "ans_reply": ans_reply, "ref_reply": ref_reply,
                     "root_ko": root_ko, "labels": [k for k, v in labels.items() if v]})
    return rows


def corpus_reference(size, comps):
    """Corpus label shares and the corpus's move-number mix, for the snapshot."""
    mp = np.load(os.path.join(PROJECT_ROOT, "axes", "motif_proxies.npz"))
    ph = json.load(open(os.path.join(PROJECT_ROOT, "activations", "_signal_cache",
                                     f"game_phase_{size}.json")))
    m = mp["size"] == size
    rules = list(mp["rules"])
    share = {c: float(mp["labels"][m, rules.index(c)].mean()) for c in comps}
    pre = np.array([ph[t][1] for t in mp["tid"][m]])
    mix = np.bincount(pre // BUCKET[size]).astype(float)
    return share, mix / mix.sum()


class Tally:
    def __init__(self, comps, size):
        self.comps, self.size = comps, size
        self.lock = threading.Lock()
        self.games = self.positions = 0
        self.nb = 512
        self.pos_b = np.zeros(self.nb)
        self.hit_b = {c: np.zeros(self.nb) for c in comps}
        self.pending = []

    def add(self, rows):
        with self.lock:
            self.games += 1
            for r in rows:
                b = min(r["move"] // BUCKET[self.size], self.nb - 1)
                self.positions += 1
                self.pos_b[b] += 1
                for c in r["labels"]:
                    if c in self.hit_b:
                        self.hit_b[c][b] += 1
            self.pending.append(rows)


def worker(engine, finder, tally):
    while not stop_event.is_set():
        try:
            moves, searches, alt = play_game(engine, finder)
            if stop_event.is_set():
                return
            tally.add(label_game(moves, searches, alt, engine.size))
        except Exception as e:  # a failed game is dropped, the worker carries on
            print(f"game failed: {type(e).__name__}: {e}", flush=True)


def snapshot(tally, ref_share, ref_mix, t0, out_dir):
    with tally.lock:
        games, positions = tally.games, tally.positions
        pos_b = tally.pos_b.copy()
        hit_b = {c: v.copy() for c, v in tally.hit_b.items()}
        pending, tally.pending = tally.pending, []
    with open(os.path.join(out_dir, "positions.jsonl"), "a") as fh:
        for rows in pending:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    el = time.time() - t0
    lines = [f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  games {games}  positions {positions}  "
             f"({positions / max(el, 1):.1f} pos/s, {games / max(el, 1) * 60:.1f} games/min)"]
    if positions:
        n = min(len(ref_mix), len(pos_b))
        mix = np.zeros(len(pos_b)); mix[:n] = ref_mix[:n]
        cov = (pos_b > 0) & (mix > 0)
        lines.append(f"{'label':16s} {'selfplay':>9s} {'at corpus move mix':>19s} {'corpus':>7s}")
        for c in tally.comps:
            sp = hit_b[c].sum() / positions
            rw = ((hit_b[c][cov] / pos_b[cov]) * mix[cov]).sum() / mix[cov].sum()
            lines.append(f"{c:16s} {sp:9.1%} {rw:19.1%} {ref_share[c]:7.1%}")
        lines.append(f"(corpus move-mix mass with no self-play positions: {mix[~cov].sum():.1%})")
    with open(os.path.join(out_dir, "summary.txt"), "a") as fh:
        fh.write("\n".join(lines) + "\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("size", type=int, choices=(9, 13))
    ap.add_argument("--visits", type=int, default=3, help="search visits per move, both sides")
    ap.add_argument("--threads", type=int, default=64, help="games in flight = numAnalysisThreads")
    ap.add_argument("--batch", type=int, default=64, help="nnMaxBatchSize")
    ap.add_argument("--every", type=float, default=15.0, help="seconds between snapshots")
    ap.add_argument("--out", default=None, help="default dev_runs/selfplay_labels_<size>_v<visits>")
    args = ap.parse_args()

    comps = json.load(open(os.path.join(PROJECT_ROOT, "axes", "labels.json")))["components"]
    out_dir = args.out or os.path.join(PROJECT_ROOT, "dev_runs",
                                       f"selfplay_labels_{args.size}_v{args.visits}")
    os.makedirs(out_dir, exist_ok=True)
    ref_share, ref_mix = corpus_reference(args.size, comps)

    def _stop(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGHUP, _stop)

    engine = Engine(args.size, args.visits, args.threads, args.batch)
    finder = ProblemFinder(engine.queries)
    tally = Tally(comps, args.size)
    with open(os.path.join(out_dir, "summary.txt"), "a") as fh:
        fh.write(f"##### start {time.strftime('%Y-%m-%d %H:%M:%S')}  size {args.size}  "
                 f"visits {args.visits}  threads {args.threads}  batch {args.batch}  "
                 f"model {DEFAULT_MODEL}\n\n")
    t0 = time.time()
    for _ in range(args.threads):
        threading.Thread(target=worker, args=(engine, finder, tally), daemon=True).start()
    try:
        while True:
            time.sleep(args.every)
            snapshot(tally, ref_share, ref_mix, t0, out_dir)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        snapshot(tally, ref_share, ref_mix, t0, out_dir)
        engine.controller.stop_engine()


if __name__ == "__main__":
    main()
