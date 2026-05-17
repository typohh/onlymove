#!/usr/bin/env python3
"""Calibrate humanv0 rank Elo scale via adjacent-rank self-play on 19x19.

19x19 counterpart of gorm/calibrate_ranks.py (which is 9x9). We use this to
verify whether gorm's 9x9 finding — that humanv0 adjacent ranks are spaced
at a roughly constant ~45 Elo per nominal stone across the full 20k → 9d
range — is a property of the model labels or a board-size artifact.
statistics.txt (real human 19x19 play) shows Elo/stone widening from ~45
in low kyu to ~190 at high dan; if that widening is real-rank behavior we
expect humanv0 on 19x19 to reproduce some of it. If 9x9-style flatness
persists, the bots' label-to-strength mapping is the constraint, not the
board size.

Both sides sample from KataGo's humanPolicy at temp 1.0 with maxVisits=1 —
identical recipe to the 9x9 script and to the rank opponents in
auxiliary/calibrate_handicap_ranks.py.

Restartable: SGFs land in calibration_19x19_sgf/ and the next run scans
them to rebuild the win/loss table before resuming.

Usage:
    python -m auxiliary.calibrate_ranks_19x19 [--threads N] [--games-per-pair N]
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.models import MODEL_FULL, MODEL_HUMAN, get_model  # noqa: E402

# --- Paths ---
KATAGO_BIN = os.environ.get("KATAGO_PATH") or os.path.join(
    PROJECT_ROOT, "bots", "katago")
SGF_DIR = os.path.join(PROJECT_ROOT, "calibration_19x19_sgf")
CFG_PATH = os.path.join(PROJECT_ROOT, "bots", "calibration_19x19.cfg")

BOARD_SIZE = 19
KOMI = 6.5
RULES = "japanese"
MAX_MOVES = 500

# --- Opponent strength scale.
# The humanv0 rank-string → Elo mapping is a property of how that model was
# trained (one network, conditioned on a rank label); it doesn't depend on
# board size. Same constants as gorm/calibrate_ranks.py and the rank
# opponents in eval_remote.py. ---
_ELO_E0 = 2363.8
_ELO_A = 12.0
_ELO_TAU = 14.3

_RANK_STR = {
    "rank_20k": 91.2, "rank_19k": 87.8, "rank_18k": 84.3, "rank_17k": 80.8,
    "rank_16k": 77.4, "rank_15k": 74.0, "rank_14k": 70.5, "rank_13k": 66.0,
    "rank_12k": 61.5, "rank_11k": 57.0, "rank_10k": 52.5, "rank_9k": 48.0,
    "rank_8k": 45.6, "rank_7k": 43.1, "rank_6k": 40.7, "rank_5k": 38.3,
    "rank_4k": 35.8, "rank_3k": 33.4, "rank_2k": 31.0, "rank_1k": 28.5,
    "rank_1d": 26.1, "rank_2d": 17.7, "rank_3d": 15.6, "rank_4d": 13.4,
    "rank_5d": 13.0, "rank_6d": 12.6, "rank_7d": 12.2, "rank_8d": 11.8,
    "rank_9d": 11.4,
}


def _str_to_elo(s):
    if s <= 0:
        return _ELO_E0 + abs(s) * 100.0
    return _ELO_E0 - _ELO_A * s - (100 - _ELO_A) * _ELO_TAU * (
        1 - math.exp(-s / _ELO_TAU))


def _human9d_temp_elo(temp):
    return _str_to_elo(6.6 + 6.6 * temp ** 4.8)


def _spec_temp_elo(temp):
    if temp <= 0.1:
        return _ELO_E0
    return _ELO_E0 - 515.5 * (temp - 0.1) ** 1.5


# Ordered opponent list: (name, elo, type, param)
OPPONENTS = [(_r, _str_to_elo(_s), "rank", _r) for _r, _s in _RANK_STR.items()]
for _t in [0.8, 0.6]:
    OPPONENTS.append((f"human9d_t{_t:.1f}", _human9d_temp_elo(_t), "human_temp", _t))
for _t in [1.0, 0.8, 0.6, 0.4]:
    OPPONENTS.append((f"spec_t{_t:.1f}", _spec_temp_elo(_t), "spec_temp", _t))
OPPONENTS.append(("spec_t0.1", _ELO_E0, "spec_temp", 0.1))
OPPONENTS.sort(key=lambda x: x[1])

OPP_NAMES = [o[0] for o in OPPONENTS]
OPP_INDEX = {o[0]: i for i, o in enumerate(OPPONENTS)}
OPP_INFO = {o[0]: o for o in OPPONENTS}
PAIRS = [(OPP_NAMES[i], OPP_NAMES[i + 1]) for i in range(len(OPP_NAMES) - 1)]

_BASE_ELO = OPPONENTS[0][1]
CURRENT_ELOS = {name: elo - _BASE_ELO for name, elo, _, _ in OPPONENTS}


def opponent_display(name):
    if name.startswith("rank_"):
        return name.replace("rank_", "")
    return name


def display_to_opponent(disp):
    if disp in OPP_INDEX:
        return disp
    rname = f"rank_{disp}"
    if rname in OPP_INDEX:
        return rname
    return None


# --- Config -------------------------------------------------------------

def generate_config(num_threads):
    cfg = (
        "logSearchInfo = false\n"
        "logToStderr = true\n"
        "numSearchThreads = 1\n"
        f"numAnalysisThreads = {num_threads}\n"
        f"nnMaxBatchSize = {num_threads}\n"
        "reportAnalysisWinratesAs = BLACK\n"
    )
    os.makedirs(os.path.dirname(CFG_PATH), exist_ok=True)
    with open(CFG_PATH, "w") as f:
        f.write(cfg)


# --- SGF helpers --------------------------------------------------------

def _to_sgf_coord(action):
    """Action int → SGF coord. SGF uses 'a'..'s' for 19x19 columns/rows."""
    if action == BOARD_SIZE * BOARD_SIZE:
        return ""
    return (chr(action % BOARD_SIZE + ord('a'))
            + chr(action // BOARD_SIZE + ord('a')))


def _make_sgf(moves, black_name, white_name, result_tag):
    sgf = (f"(;GM[1]FF[4]SZ[{BOARD_SIZE}]KM[{KOMI}]RU[{RULES.capitalize()}]"
           f"RE[{result_tag}]PB[{black_name}]PW[{white_name}]\n")
    for i, a in enumerate(moves):
        color = "B" if i % 2 == 0 else "W"
        sgf += f";{color}[{_to_sgf_coord(a)}]"
    sgf += ")"
    return sgf


# --- Recovery from existing SGFs ----------------------------------------

def recover_results(sgf_dir):
    results = {}
    for w, s in PAIRS:
        results[(w, s)] = {"wins_stronger": 0, "wins_weaker": 0,
                           "ties": 0, "total": 0}

    if not os.path.isdir(sgf_dir):
        return results

    for fname in os.listdir(sgf_dir):
        if not fname.endswith(".sgf"):
            continue
        try:
            with open(os.path.join(sgf_dir, fname)) as f:
                content = f.read(2000)  # only need header

            pb_m = re.search(r'PB\[([^\]]+)\]', content)
            pw_m = re.search(r'PW\[([^\]]+)\]', content)
            re_m = re.search(r'RE\[([^\]]+)\]', content)
            if not (pb_m and pw_m and re_m):
                continue

            pb = display_to_opponent(pb_m.group(1))
            pw = display_to_opponent(pw_m.group(1))
            res = re_m.group(1)
            if pb is None or pw is None:
                continue

            pb_i, pw_i = OPP_INDEX[pb], OPP_INDEX[pw]
            if abs(pb_i - pw_i) != 1:
                continue

            weaker = OPP_NAMES[min(pb_i, pw_i)]
            stronger = OPP_NAMES[max(pb_i, pw_i)]
            key = (weaker, stronger)
            results[key]["total"] += 1

            if res in ("Void", "0", "?"):
                results[key]["ties"] += 1
            elif res.startswith("B+"):
                if pb_i > pw_i:
                    results[key]["wins_stronger"] += 1
                else:
                    results[key]["wins_weaker"] += 1
            elif res.startswith("W+"):
                if pw_i > pb_i:
                    results[key]["wins_stronger"] += 1
                else:
                    results[key]["wins_weaker"] += 1
        except Exception:
            continue

    return results


# --- KataGo controller --------------------------------------------------

class KataGoController:
    """Single shared KataGo analysis process, thread-safe via per-query events."""

    def __init__(self, model_path, human_model_path):
        N = BOARD_SIZE
        cmd = [
            KATAGO_BIN, "analysis",
            "-config", CFG_PATH,
            "-model", model_path,
            "-human-model", human_model_path,
            "-override-config",
            (f"maxBoardXSizeForNNBuffer={N},"
             f"maxBoardYSizeForNNBuffer={N},"
             f"requireMaxBoardSize=true"),
        ]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._stdin_lock = threading.Lock()
        self._pending = {}
        self._pending_lock = threading.Lock()

        threading.Thread(target=self._drain_stderr, daemon=True).start()
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _drain_stderr(self):
        start = time.time()
        for line in self.process.stderr:
            if time.time() - start < 5:
                print(f"  KataGo: {line.strip()}")

    def _read_stdout(self):
        for line in self.process.stdout:
            try:
                resp = json.loads(line)
            except Exception:
                continue
            qid = resp.get("id")
            with self._pending_lock:
                entry = self._pending.get(qid)
            if entry:
                entry["result"] = resp
                entry["event"].set()

    def _action_to_gtp(self, action):
        N = BOARD_SIZE
        if action == N * N:
            return "pass"
        r, c = action // N, action % N
        # GTP columns skip I, so anything at index >= 8 shifts up by 1.
        col = chr(ord("A") + c + (1 if c >= 8 else 0))
        return f"{col}{N - r}"

    def query(self, moves, extra):
        with self._counter_lock:
            self._counter += 1
            qid = str(self._counter)

        event = threading.Event()
        with self._pending_lock:
            self._pending[qid] = {"event": event, "result": None}

        gtp_moves = [
            ["B" if i % 2 == 0 else "W", self._action_to_gtp(a)]
            for i, a in enumerate(moves)
        ]
        q = {
            "id": qid,
            "moves": gtp_moves,
            "initialStones": [],
            "rules": RULES,
            "komi": KOMI,
            "boardXSize": BOARD_SIZE,
            "boardYSize": BOARD_SIZE,
            **extra,
        }
        with self._stdin_lock:
            self.process.stdin.write(json.dumps(q) + "\n")
            self.process.stdin.flush()

        if not event.wait(timeout=120):
            with self._pending_lock:
                self._pending.pop(qid, None)
            raise TimeoutError(f"KataGo query {qid} timed out")

        with self._pending_lock:
            entry = self._pending.pop(qid, None)
        if entry is None or entry["result"] is None:
            raise RuntimeError(f"KataGo query {qid} returned no result")
        return entry["result"]

    def get_move(self, moves, opp_type, opp_param, rng):
        """Sample one move for the given opponent.

        opp_type / opp_param:
          "rank"       / rank string -> humanPolicy at rank, temp 1.0
          "human_temp" / float temp  -> humanPolicy 9d at given temp
          "spec_temp"  / float temp  -> strong-net policy at given temp
        """
        N = BOARD_SIZE

        if opp_type in ("rank", "human_temp"):
            rank = opp_param if opp_type == "rank" else "rank_9d"
            resp = self.query(moves, {
                "includePolicy": True,
                "maxVisits": 1,
                "overrideSettings": {
                    "humanSLProfile": rank,
                    "ignorePreRootHistory": False,
                    "rootNumSymmetriesToSample": 1,
                },
            })
            if "error" in resp:
                return N * N

            # Strong-policy pass detection: if the strong net wants to pass,
            # the position is finished and we honor that regardless of what
            # the humanPolicy would do.
            strong = resp.get("policy", [])
            if strong and int(np.argmax(strong)) == N * N:
                return N * N

            policy = resp.get("humanPolicy")
            if not policy:
                return N * N
            if int(np.argmax(policy)) == N * N:
                return N * N

            probs = np.maximum(np.array(policy, dtype=np.float64), 0.0)
            temp = opp_param if opp_type == "human_temp" else 1.0
        else:  # spec_temp
            resp = self.query(moves, {
                "includePolicy": True,
                "maxVisits": 1,
            })
            if "error" in resp:
                return N * N
            policy = resp.get("policy", [])
            if not policy:
                return N * N
            if int(np.argmax(policy)) == N * N:
                return N * N
            probs = np.maximum(np.array(policy, dtype=np.float64), 0.0)
            temp = opp_param

        if temp != 1.0 and temp > 0:
            probs = np.maximum(probs, 1e-30)
            log_p = np.log(probs)
            log_p /= temp
            log_p -= log_p.max()
            probs = np.exp(log_p)

        s = probs.sum()
        if s <= 0:
            return N * N
        probs /= s
        return int(rng.choice(len(probs), p=probs))

    def score_game(self, moves):
        resp = self.query(moves, {"includeOwnership": False, "maxVisits": 10})
        if "error" in resp:
            return 0.0, 0.5
        root = resp.get("rootInfo", {})
        return root.get("scoreLead", 0.0), root.get("winrate", 0.5)

    def close(self):
        try:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


# --- Play one game ------------------------------------------------------

def play_game(katago, black_opp, white_opp, rng):
    """black_opp, white_opp: (name, elo, type, param) tuples from OPP_INFO."""
    N = BOARD_SIZE
    moves = []
    consec_passes = 0

    for _ in range(MAX_MOVES):
        opp = black_opp if len(moves) % 2 == 0 else white_opp
        try:
            action = katago.get_move(moves, opp[2], opp[3], rng)
        except Exception:
            action = N * N  # treat as pass on transient failure
        moves.append(action)

        if action == N * N:
            consec_passes += 1
            if consec_passes >= 2:
                break
        else:
            consec_passes = 0

    score, winrate = katago.score_game(moves)
    disagree = (score != 0.0 and winrate != 0.5
                and (score > 0) != (winrate > 0.5))
    score_rounded = round(score * 2) / 2

    if disagree:
        return moves, score_rounded, "TIE"
    if score > 0:
        return moves, score_rounded, "B"
    if score < 0:
        return moves, score_rounded, "W"
    return moves, score_rounded, "DRAW"


# --- Elo computation ----------------------------------------------------

def compute_elo_scales(results):
    """Return (raw_elos, smoothed_elos, raw_gaps, smoothed_gaps)."""

    raw_gaps = []
    for weaker, stronger in PAIRS:
        r = results[(weaker, stronger)]
        ws, ww = r["wins_stronger"], r["wins_weaker"]
        decisive = ws + ww
        if decisive == 0:
            raw_gaps.append(None)
            continue
        wr = max(0.02, min(0.98, ws / decisive))
        raw_gaps.append(400 * math.log10(wr / (1 - wr)))

    sigma = 3.0
    smoothed_gaps = []
    for i in range(len(raw_gaps)):
        num, den = 0.0, 0.0
        for j in range(len(raw_gaps)):
            if raw_gaps[j] is None:
                continue
            r = results[PAIRS[j]]
            games = r["wins_stronger"] + r["wins_weaker"]
            w = math.exp(-0.5 * ((i - j) / sigma) ** 2) * games
            num += w * raw_gaps[j]
            den += w
        smoothed_gaps.append(num / den if den > 0 else 0.0)

    raw_elos = {OPP_NAMES[0]: 0.0}
    smoothed_elos = {OPP_NAMES[0]: 0.0}
    raw_cum, smooth_cum = 0.0, 0.0
    for i, (gap, sgap) in enumerate(zip(raw_gaps, smoothed_gaps)):
        raw_cum += gap if gap is not None else 0.0
        smooth_cum += sgap
        raw_elos[OPP_NAMES[i + 1]] = raw_cum
        smoothed_elos[OPP_NAMES[i + 1]] = smooth_cum

    return raw_elos, smoothed_elos, raw_gaps, smoothed_gaps


def print_report(results, verbose=False):
    raw_elos, smoothed_elos, raw_gaps, smoothed_gaps = compute_elo_scales(results)
    total = sum(r["total"] for r in results.values())
    decisive = sum(r["wins_stronger"] + r["wins_weaker"] for r in results.values())

    print(f"\n{'=' * 110}")
    print(f"OPPONENT CALIBRATION 19x19  "
          f"({decisive} decisive / {total} total games)")
    print(f"Ranks: humanPolicy sampling; human9d_tX: humanPolicy 9d at temp X; "
          f"spec_tX: strong policy at temp X")
    print(f"{'=' * 110}")

    if verbose:
        print(f"\n{'Matchup':<28} {'Games':>6} {'Str W':>6} {'Wk W':>6} "
              f"{'Ties':>5} {'Str%':>6} {'Raw gap':>8} {'Sm gap':>8}")
        print("-" * 81)
        for i, (weaker, stronger) in enumerate(PAIRS):
            r = results[(weaker, stronger)]
            ws, ww = r["wins_stronger"], r["wins_weaker"]
            dec = ws + ww
            label = (f"{opponent_display(weaker):>12s} v "
                     f"{opponent_display(stronger):<12s}")
            if dec > 0:
                pct = f"{ws / dec * 100:.0f}%"
                rg = f"{raw_gaps[i]:+.0f}" if raw_gaps[i] is not None else "-"
            else:
                pct, rg = "-", "-"
            sg = f"{smoothed_gaps[i]:+.0f}"
            print(f"{label:<28} {r['total']:>6} {ws:>6} {ww:>6} "
                  f"{r['ties']:>5} {pct:>6} {rg:>8} {sg:>8}")

    print(f"\n{'Opponent':<13} {'Raw':>8} {'Smoothed':>10} {'Current':>10}  "
          f"{'Raw-Cur':>8} {'Sm-Cur':>8}")
    print("-" * 63)
    for name in OPP_NAMES:
        rd = opponent_display(name)
        raw = raw_elos[name]
        smooth = smoothed_elos[name]
        cur = CURRENT_ELOS[name]
        print(f"{rd:<13} {raw:>8.0f} {smooth:>10.0f} {cur:>10.0f}  "
              f"{raw - cur:>+8.0f} {smooth - cur:>+8.0f}")

    print(f"{'=' * 110}\n")


# --- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate humanv0 rank Elo via 19x19 adjacent-rank self-play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--threads", type=int, default=128,
                        help="Parallel games / KataGo analysis threads. "
                             "19x19 NN forward passes are heavier than 9x9 so "
                             "default is lower than gorm's 9x9 default of 512.")
    parser.add_argument("--games-per-pair", type=int, default=0,
                        help="Stop after N decisive games per pair (0 = run forever)")
    parser.add_argument("--report-interval", type=int, default=28,
                        help="Print full report every N games")
    parser.add_argument("--report-only", action="store_true",
                        help="Only print report from existing SGFs, then exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(SGF_DIR, exist_ok=True)

    print(f"Recovering results from {SGF_DIR}...")
    results = recover_results(SGF_DIR)
    recovered = sum(r["total"] for r in results.values())
    if recovered:
        print(f"Recovered {recovered} games")
        print_report(results, verbose=args.verbose)
    else:
        print("No existing games found, starting fresh.")

    if args.report_only:
        return

    model_path = get_model(MODEL_FULL)
    human_path = get_model(MODEL_HUMAN)
    for path, label in [(KATAGO_BIN, "KataGo binary"),
                        (model_path, "strong model"),
                        (human_path, "humanv0 model")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} missing at {path}", file=sys.stderr)
            sys.exit(1)

    generate_config(args.threads)
    print(f"Starting KataGo ({args.threads} analysis threads)...")
    katago = KataGoController(model_path, human_path)
    time.sleep(3)
    if katago.process.poll() is not None:
        print(f"ERROR: KataGo exited (code {katago.process.returncode})",
              file=sys.stderr)
        sys.exit(1)

    try:
        resp = katago.query([], {
            "maxVisits": 1, "includePolicy": True,
            "overrideSettings": {"humanSLProfile": "rank_15k"},
        })
        if "humanPolicy" not in resp:
            print("ERROR: humanPolicy missing from response. "
                  "Is -human-model wired up?", file=sys.stderr)
            katago.close()
            sys.exit(1)
        print("KataGo ready.")
    except Exception as e:
        print(f"ERROR: warmup failed: {e}", file=sys.stderr)
        katago.close()
        sys.exit(1)

    lock = threading.Lock()
    games_since_report = [0]

    def select_pair():
        best, best_n = None, float("inf")
        for w, s in PAIRS:
            decisive = (results[(w, s)]["wins_stronger"]
                        + results[(w, s)]["wins_weaker"])
            if args.games_per_pair > 0 and decisive >= args.games_per_pair:
                continue
            if decisive < best_n:
                best_n = decisive
                best = (w, s)
        return best

    def worker():
        rng = np.random.default_rng()
        while True:
            with lock:
                pair = select_pair()
                if pair is None:
                    return
                weaker, stronger = pair
                key = (weaker, stronger)
                game_num = results[key]["total"]

            # Alternate colors each game per pair.
            if game_num % 2 == 0:
                br, wr = stronger, weaker
            else:
                br, wr = weaker, stronger

            try:
                moves, score, result = play_game(
                    katago, OPP_INFO[br], OPP_INFO[wr], rng)
            except Exception as e:
                print(f"  game error ({opponent_display(weaker)} v "
                      f"{opponent_display(stronger)}): {e}")
                time.sleep(1)
                continue

            if result == "B":
                re_tag = f"B+{abs(score)}"
            elif result == "W":
                re_tag = f"W+{abs(score)}"
            else:
                re_tag = "Void"

            sgf = _make_sgf(moves,
                            opponent_display(br), opponent_display(wr), re_tag)
            sgf_path = os.path.join(
                SGF_DIR,
                f"{opponent_display(weaker)}_vs_{opponent_display(stronger)}"
                f"_{uuid.uuid4().hex[:8]}.sgf")
            with open(sgf_path, "w") as f:
                f.write(sgf)

            with lock:
                results[key]["total"] += 1
                if result == "B":
                    if OPP_INDEX[br] > OPP_INDEX[wr]:
                        results[key]["wins_stronger"] += 1
                    else:
                        results[key]["wins_weaker"] += 1
                elif result == "W":
                    if OPP_INDEX[wr] > OPP_INDEX[br]:
                        results[key]["wins_stronger"] += 1
                    else:
                        results[key]["wins_weaker"] += 1
                else:
                    results[key]["ties"] += 1

                r = results[key]
                total_all = sum(d["total"] for d in results.values())
                if args.verbose:
                    print(f"[{total_all:5d}] "
                          f"{opponent_display(weaker):>12s} v "
                          f"{opponent_display(stronger):<12s}: "
                          f"{result:4s} ({score:+.1f})  "
                          f"{r['wins_stronger']}W {r['wins_weaker']}L "
                          f"{r['ties']}T / {r['total']}")

                games_since_report[0] += 1
                if games_since_report[0] >= args.report_interval:
                    games_since_report[0] = 0
                    print_report(results, verbose=args.verbose)

    try:
        threads = [threading.Thread(target=worker, daemon=True)
                   for _ in range(args.threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        katago.close()

    print_report(results, verbose=args.verbose)
    print("Done.")


if __name__ == "__main__":
    main()
