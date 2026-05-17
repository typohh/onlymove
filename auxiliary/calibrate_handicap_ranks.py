#!/usr/bin/env python3
"""Calibrate humanv0 rank gaps via standard-handicap games on 19x19.

Standard handicap (Japanese rules, komi 0.5 to white, N stones placed):
  - h=0: 0 stones, komi 6.5, B first (regular even game)
  - h=1: 0 stones, komi 0.5, B first (reduced-komi handicap)
  - h=N for N>=2: N stones placed by B at star points, komi 0.5, W first

We don't assume a linear "1 stone = 1 rank" mapping. Instead h_eff[h] is a
free parameter for h=1..9 (with h_eff[0]=0 fixed), inferred jointly with
ranks from game outcomes. A scalar α captures how the curve scales with
absolute rank: h_eff_eff(h, r_mid) = h_eff[h] · exp(α · r_mid), where
r_mid = (r_strong + r_weak) / 2. α=0 means rank-independent curvature;
α≠0 means each handicap stone buys more (or less) rank as players get
stronger.

Anchor: rank[humanv0_3k] = 0 (translation only — the bot's "3k" label is
not used as evidence of rank). Calibration set: humanv0_20k through
humanv0_9d. A weak N(nominal_label_index, σ=3) prior on rank exists purely
as a cold-start anchor; data dominates once each bot has enough games.

Per game:
  1. Pair selection prioritized by information value. Only pairs whose
     current |rank gap| fits within the largest playable handicap *at that
     pair's mid-rank* (h_eff[9]·exp(α·r_mid)) are candidates. Among those,
     each pair gets weight
         w_pair = (SE_a^2 + SE_b^2 + eps) / (games_on_pair + alpha)
     and one is sampled proportionally.
  2. Predicted gap g = |rank[A] - rank[B]| from current estimates.
  3. Handicap level h: bracket g on the per-pair h_eff_local grid
     (h_eff_arr scaled by exp(α·r_mid)) and stochastically round.
  4. Configure board per h above; weaker side plays black for h>=1; h=0
     alternates colors per pair.
  5. Each move sampled from humanv0 (humanPolicy temp=1.0, maxVisits=1),
     same recipe as gorm/calibrate_ranks.py.
  6. Score with strong policy (maxVisits=10), record SGF.
  7. Periodically refit ranks + h_eff[h] + α + c jointly via logistic MAP:
        P(strong wins) = σ(c · (r_s - r_w - h_eff[h]·exp(α·r_mid)))
     Weak Gaussian priors keep the fit identified at cold start; they
     decay with data.

Restartable: re-parsing SGFs in calibration_handicap_sgf/ recovers everything.
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
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.models import MODEL_FULL, MODEL_HUMAN, get_model  # noqa: E402

# --- Paths ---
KATAGO_BIN = os.environ.get("KATAGO_PATH") or os.path.join(
    PROJECT_ROOT, "bots", "katago")
SGF_DIR = os.path.join(PROJECT_ROOT, "calibration_handicap_sgf")
CFG_PATH = os.path.join(PROJECT_ROOT, "bots", "calibration_handicap.cfg")

BOARD_SIZE = 19
EVEN_KOMI = 6.5      # h=0
HANDICAP_KOMI = 0.5  # h>=1 (standard handicap)
RULES = "japanese"
MAX_MOVES = 500


def komi_for(h):
    return EVEN_KOMI if h == 0 else HANDICAP_KOMI


def first_color(h):
    """Who moves first. h=0 and h=1 are no-setup-stone games so B leads.
    h>=2 has B's stones already placed, so W moves first."""
    return "B" if h <= 1 else "W"


def h_eff_linear(h):
    """Theoretical "1 stone = 1 rank, standard handicap costs ½ stone" curve.
    Used as the prior mean and initial value for the empirical fit; the
    model itself no longer assumes this form."""
    return 0.0 if h == 0 else h - 0.5


def h_eff_eff(h, r_mid, h_eff_arr, alpha):
    """Per-game effective rank gap: h_eff_arr[h] · exp(α · r_mid).

    h_eff_arr[h] is the curve value at the anchor mid-rank (r_mid=0). α is
    the multiplicative rank-coupling: α>0 means each handicap stone buys
    more rank when both players are strong, α<0 means less. α=0 collapses
    to a rank-independent curve."""
    return float(h_eff_arr[h]) * math.exp(alpha * r_mid)


H_LEVELS = list(range(10))  # 0..9

# Calibration bots. 3k is the anchor (rank = 0); the rest are fit. The
# full range 20k..9d gives the chain enough overlap that within-9-stones
# pair selection still connects every bot to its neighbours.
BOT_LABELS = (
    [f"{n}k" for n in range(20, 0, -1)]   # 20k..1k
    + [f"{n}d" for n in range(1, 10)]     # 1d..9d
)
ANCHOR_BOT = "3k"
ANCHOR_IDX = BOT_LABELS.index(ANCHOR_BOT)
KATAGO_PROFILE = {label: f"rank_{label}" for label in BOT_LABELS}

# 19x19 standard handicap stone placements (GTP coords, I skipped).
# h=1 places no stone — standard handicap 1 is just reduced komi with B
# moving first.
HANDICAP_STONES = {
    0: [],
    1: [],
    2: ["Q4", "D16"],
    3: ["Q4", "D16", "D4"],
    4: ["D4", "Q4", "D16", "Q16"],
    5: ["D4", "Q4", "D16", "Q16", "K10"],
    6: ["D4", "Q4", "D16", "Q16", "D10", "Q10"],
    7: ["D4", "Q4", "D16", "Q16", "D10", "Q10", "K10"],
    8: ["D4", "Q4", "D16", "Q16", "D10", "Q10", "K4", "K16"],
    9: ["D4", "Q4", "D16", "Q16", "D10", "Q10", "K4", "K16", "K10"],
}


# --- Config + path helpers -------------------------------------------------

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


# --- SGF helpers ----------------------------------------------------------

def gtp_to_sgf(gtp, size=BOARD_SIZE):
    if gtp == "pass" or gtp == "":
        return ""
    col_letter = gtp[0].upper()
    row_num = int(gtp[1:])
    col_idx = ord(col_letter) - ord('A')
    if col_letter > 'I':
        col_idx -= 1  # GTP skips I; SGF doesn't
    return chr(ord('a') + col_idx) + chr(ord('a') + (size - row_num))


def make_sgf(black_label, white_label, handicap, moves, result_tag):
    sgf = (f"(;GM[1]FF[4]SZ[{BOARD_SIZE}]KM[{komi_for(handicap)}]"
           f"RU[Japanese]")
    if handicap >= 1:
        sgf += f"HA[{handicap}]"
    sgf += f"RE[{result_tag}]PB[{black_label}]PW[{white_label}]\n"
    if HANDICAP_STONES[handicap]:
        sgf += "AB" + "".join(f"[{gtp_to_sgf(s)}]" for s in HANDICAP_STONES[handicap])
        sgf += "\n"
    for color, coord in moves:
        sgf += f";{color}[{gtp_to_sgf(coord)}]"
    sgf += ")"
    return sgf


_PROP_RE = re.compile(r'(HA|KM|PB|PW|RE)\[([^\]]*)\]')


def parse_sgf(path):
    """Return (black, white, handicap, result) or None if malformed."""
    try:
        with open(path) as f:
            head = f.read(2000)  # only need header
    except Exception:
        return None
    props = dict(_PROP_RE.findall(head))
    if not all(k in props for k in ("PB", "PW", "RE")):
        return None
    handicap = int(props.get("HA", "0")) if props.get("HA") else 0
    return props["PB"], props["PW"], handicap, props["RE"]


# --- Recovery -------------------------------------------------------------

def recover_games(sgf_dir):
    """Return list of game dicts {bot_strong, bot_weak, handicap, stronger_won, draw}.

    Bots whose labels aren't in BOT_LABELS are skipped (legacy data, manual
    games). "Stronger" is determined by current nominal rank order.
    """
    if not os.path.isdir(sgf_dir):
        return []

    nominal = {l: i for i, l in enumerate(BOT_LABELS)}
    games = []
    for fname in os.listdir(sgf_dir):
        if not fname.endswith(".sgf"):
            continue
        parsed = parse_sgf(os.path.join(sgf_dir, fname))
        if parsed is None:
            continue
        black, white, handicap, result = parsed
        if black not in nominal or white not in nominal:
            continue
        # By construction, black is the weaker side when handicap >= 1.
        # For h=0 we still need a canonical "stronger" choice for the MLE;
        # nominal label order is fine because the model is symmetric in
        # which side we call stronger as long as we're consistent.
        if nominal[black] > nominal[white]:
            stronger, weaker = black, white
            black_is_stronger = True
        else:
            stronger, weaker = white, black
            black_is_stronger = False

        draw = result.startswith("Void") or result == "0" or result == "?"
        if draw:
            games.append({"bot_strong": stronger, "bot_weak": weaker,
                          "handicap": handicap, "stronger_won": None,
                          "draw": True})
            continue

        black_won = result.startswith("B+")
        stronger_won = (black_won == black_is_stronger)
        games.append({"bot_strong": stronger, "bot_weak": weaker,
                      "handicap": handicap, "stronger_won": stronger_won,
                      "draw": False})
    return games


# --- KataGo controller ----------------------------------------------------

class KataGoController:
    """Single shared KataGo analysis process, thread-safe via per-query events."""

    def __init__(self, model_path, human_model_path):
        cmd = [
            KATAGO_BIN, "analysis",
            "-config", CFG_PATH,
            "-model", model_path,
            "-human-model", human_model_path,
            "-override-config",
            (f"maxBoardXSizeForNNBuffer={BOARD_SIZE},"
             f"maxBoardYSizeForNNBuffer={BOARD_SIZE},"
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

    def query(self, initial_stones, moves, komi, extra):
        with self._counter_lock:
            self._counter += 1
            qid = str(self._counter)
        event = threading.Event()
        with self._pending_lock:
            self._pending[qid] = {"event": event, "result": None}

        q = {
            "id": qid,
            "moves": moves,
            "initialStones": initial_stones,
            "rules": RULES,
            "komi": komi,
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

    def get_move(self, initial_stones, moves, komi, bot_label, rng):
        """Sample one move from humanv0 conditioned on bot_label's rank."""
        resp = self.query(initial_stones, moves, komi, {
            "includePolicy": True,
            "maxVisits": 1,
            "overrideSettings": {
                "humanSLProfile": KATAGO_PROFILE[bot_label],
                "ignorePreRootHistory": False,
                "rootNumSymmetriesToSample": 1,
            },
        })
        if "error" in resp:
            return "pass"

        N = BOARD_SIZE
        pass_idx = N * N

        # Pass-detection: if strong policy puts argmax on pass, the position is
        # finished — respect it regardless of humanPolicy.
        strong = resp.get("policy", [])
        if strong and int(np.argmax(strong)) == pass_idx:
            return "pass"

        policy = resp.get("humanPolicy")
        if not policy:
            return "pass"
        if int(np.argmax(policy)) == pass_idx:
            return "pass"

        probs = np.maximum(np.array(policy, dtype=np.float64), 0.0)
        s = probs.sum()
        if s <= 0:
            return "pass"
        probs /= s
        action = int(rng.choice(len(probs), p=probs))

        if action == pass_idx:
            return "pass"
        # KataGo policy layout: row-major from top-left, row 0 = top of board.
        # Action index = row * N + col, with col 0 = A and row 0 = top (row N).
        row = action // N
        col = action % N
        # Convert col index to GTP letter (skip I).
        col_letter = chr(ord('A') + col + (1 if col >= 8 else 0))
        row_num = N - row  # row 0 (top) -> row_num N (top of board in GTP)
        return f"{col_letter}{row_num}"

    def score_game(self, initial_stones, moves, komi):
        resp = self.query(initial_stones, moves, komi,
                          {"includeOwnership": False, "maxVisits": 10})
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


# --- Game play ------------------------------------------------------------

def play_game(kc, black_label, white_label, handicap, rng):
    """Return (moves, score, result_tag).

    moves: list of [color, gtp_coord]
    result_tag: SGF RE value (e.g. "B+12.5", "W+3.5", "Void")
    """
    initial_stones = [["B", s] for s in HANDICAP_STONES[handicap]]
    komi = komi_for(handicap)
    next_color = first_color(handicap)

    moves = []
    consec_passes = 0
    for _ in range(MAX_MOVES):
        bot_label = white_label if next_color == "W" else black_label
        try:
            coord = kc.get_move(initial_stones, moves, komi, bot_label, rng)
        except Exception:
            coord = "pass"
        moves.append([next_color, coord])

        if coord == "pass":
            consec_passes += 1
            if consec_passes >= 2:
                break
        else:
            consec_passes = 0
        next_color = "B" if next_color == "W" else "W"

    score, winrate = kc.score_game(initial_stones, moves, komi)
    disagree = (score != 0.0 and winrate != 0.5
                and (score > 0) != (winrate > 0.5))
    score_rounded = round(score * 2) / 2

    if disagree or score == 0:
        result_tag = "Void"
    elif score > 0:
        result_tag = f"B+{abs(score_rounded)}"
    else:
        result_tag = f"W+{abs(score_rounded)}"
    return moves, score_rounded, result_tag


# --- MLE fit --------------------------------------------------------------

def fit_ranks(games, init_ranks=None, init_c=1.0, init_h_eff=None,
              init_alpha=0.0,
              sigma_rank=3.0, sigma_c=0.5,
              sigma_heff=1.0, sigma_alpha=0.05):
    """MAP fit of (ranks, c, h_eff[1..9], α) under
        P(stronger wins | h, r_s, r_w)
            = sigmoid(c · (r_s − r_w − h_eff[h] · exp(α · r_mid)))
    with r[ANCHOR_BOT] = 0, r_mid = (r_s + r_w) / 2, h_eff[0] = 0 fixed,
    and weakly-informative Gaussian priors:
        rank[b] ~ N(nominal_label_index(b), sigma_rank)  (cold-start anchor)
        c ~ N(1, sigma_c)
        h_eff[h] ~ N(h − 0.5, sigma_heff)  (centered at the old linear form)
        α ~ N(0, sigma_alpha)              (centered at "no rank effect")
    All priors are dominated by data once each parameter has enough games.
    Returns (ranks, c, std_err, h_eff_arr, h_eff_se, alpha, alpha_se).
    h_eff_arr is a length-10 array with h_eff_arr[0] = 0; h_eff_se is the
    matching SE dict keyed by h (0 = 0.0 since fixed).
    """
    bots_free = [b for b in BOT_LABELS if b != ANCHOR_BOT]
    n_free = len(bots_free)
    idx = {b: i for i, b in enumerate(bots_free)}

    # Nominal positions are anchor-relative: 3k -> 0, 4k -> -1, 2k -> +1,
    # ..., 20k -> -17, 9d -> +11. Used both as initial guess and as a weak
    # Gaussian prior pulling each rank toward its label-implied position.
    # The prior decays with data; it only matters when a bot is sparsely
    # connected to the rest of the chain.
    nominal_for = {b: float(BOT_LABELS.index(b) - ANCHOR_IDX)
                   for b in BOT_LABELS}
    heff_linear_arr = np.array([h_eff_linear(h) for h in range(10)])

    def empty_return():
        ranks = dict(nominal_for)
        ranks[ANCHOR_BOT] = 0.0
        h_eff_se = {h: float("inf") for h in range(10)}
        h_eff_se[0] = 0.0
        return (ranks, init_c, {b: float("inf") for b in BOT_LABELS},
                heff_linear_arr.copy(), h_eff_se,
                init_alpha, float("inf"))

    decisive = [g for g in games if not g["draw"]]
    if not decisive:
        return empty_return()

    strong_idx = np.array(
        [idx[g["bot_strong"]] if g["bot_strong"] != ANCHOR_BOT else -1
         for g in decisive])
    weak_idx = np.array(
        [idx[g["bot_weak"]] if g["bot_weak"] != ANCHOR_BOT else -1
         for g in decisive])
    handicap_arr = np.array([g["handicap"] for g in decisive], dtype=int)
    y = np.array([1.0 if g["stronger_won"] else 0.0 for g in decisive])

    nominal_vec = np.array([nominal_for[b] for b in bots_free])
    heff_prior_mean = heff_linear_arr[1:]  # priors only on h=1..9
    inv_2var_rank = 1.0 / (2.0 * sigma_rank * sigma_rank)
    inv_2var_c = 1.0 / (2.0 * sigma_c * sigma_c)
    inv_2var_heff = 1.0 / (2.0 * sigma_heff * sigma_heff)
    inv_2var_alpha = 1.0 / (2.0 * sigma_alpha * sigma_alpha)

    # Param layout: [c, alpha, h_eff[1..9], rank_b1..rank_bN]
    N_FIXED = 2 + 9

    def unpack(params):
        c = params[0]
        alpha = params[1]
        h_eff_free = params[2:N_FIXED]
        ranks_vec = params[N_FIXED:]
        h_eff_full = np.concatenate(([0.0], h_eff_free))
        return c, alpha, h_eff_full, ranks_vec

    def gap_array(alpha, h_eff_full, ranks_vec):
        r_strong = np.where(strong_idx >= 0, ranks_vec[strong_idx], 0.0)
        r_weak = np.where(weak_idx >= 0, ranks_vec[weak_idx], 0.0)
        r_mid = 0.5 * (r_strong + r_weak)
        heff_per_game = h_eff_full[handicap_arr] * np.exp(alpha * r_mid)
        return r_strong - r_weak - heff_per_game

    def neg_ll(params):
        c, alpha, h_eff_full, ranks_vec = unpack(params)
        z = c * gap_array(alpha, h_eff_full, ranks_vec)
        log_p_strong = -np.logaddexp(0.0, -z)
        log_p_weak = -np.logaddexp(0.0, z)
        nll = -float(np.sum(y * log_p_strong + (1 - y) * log_p_weak))
        nll += inv_2var_rank * float(np.sum((ranks_vec - nominal_vec) ** 2))
        nll += inv_2var_c * (c - 1.0) ** 2
        nll += inv_2var_heff * float(
            np.sum((params[2:N_FIXED] - heff_prior_mean) ** 2))
        nll += inv_2var_alpha * alpha ** 2
        return nll

    if init_ranks is None:
        init_ranks = dict(nominal_for)
    if init_h_eff is None:
        init_h_eff_arr = heff_linear_arr
    else:
        init_h_eff_arr = np.asarray(init_h_eff, dtype=float)
    x0 = np.array(
        [init_c, init_alpha]
        + list(init_h_eff_arr[1:10])
        + [init_ranks.get(b, nominal_for[b]) for b in bots_free]
    )

    bounds = (
        [(0.1, 3.0), (-0.2, 0.2)]
        + [(0.0, 12.0)] * 9
        + [(-25.0, 20.0)] * n_free
    )
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)

    c_fit, alpha_fit, h_eff_full_fit, ranks_vec_fit = unpack(res.x)
    ranks = {ANCHOR_BOT: 0.0}
    for b in bots_free:
        ranks[b] = float(ranks_vec_fit[idx[b]])

    std_err = {b: float("nan") for b in BOT_LABELS}
    h_eff_se = {h: float("nan") for h in range(10)}
    alpha_se = float("nan")
    try:
        eps = 1e-3
        D = len(res.x)
        H = np.zeros((D, D))
        f0 = neg_ll(res.x)
        for i in range(D):
            for j in range(i, D):
                x_pp = res.x.copy(); x_pp[i] += eps; x_pp[j] += eps
                x_pm = res.x.copy(); x_pm[i] += eps; x_pm[j] -= eps
                x_mp = res.x.copy(); x_mp[i] -= eps; x_mp[j] += eps
                x_mm = res.x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                if i == j:
                    H[i, i] = (neg_ll(x_pp) - 2 * f0 + neg_ll(x_mm)) / (
                        4 * eps * eps)
                else:
                    H[i, j] = H[j, i] = (
                        neg_ll(x_pp) - neg_ll(x_pm)
                        - neg_ll(x_mp) + neg_ll(x_mm)) / (4 * eps * eps)
        cov = np.linalg.pinv(H)
        diag = np.diag(cov)
        alpha_se = float(math.sqrt(max(diag[1], 0.0)))
        h_eff_se[0] = 0.0
        for h in range(1, 10):
            h_eff_se[h] = float(math.sqrt(max(diag[1 + h], 0.0)))
        std_err[ANCHOR_BOT] = 0.0
        for b in bots_free:
            v = diag[N_FIXED + idx[b]]
            std_err[b] = float(math.sqrt(max(v, 0.0)))
    except Exception:
        pass

    return (ranks, c_fit, std_err, h_eff_full_fit, h_eff_se,
            alpha_fit, alpha_se)


# --- Pair / handicap selection -------------------------------------------

def _safe_se(value, default=1.0):
    """Return a finite, non-NaN SE; substitute `default` otherwise.

    Empty / freshly-initialized states report SE as NaN (no data) or inf
    (singular Hessian). Treat both as "max uncertainty" so the pair still
    competes for selection, but with a finite weight that np.random.choice
    can normalize."""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or not math.isfinite(v):
        return default
    return v


def pick_pair_and_handicap(ranks, std_err, pair_counts,
                           h_eff_arr, alpha, rng,
                           se_floor=0.1, count_alpha=5.0):
    """Pick (stronger, weaker, handicap_level), prioritized by information.

    Candidate pool: all unordered bot pairs whose current estimated rank
    gap is within the largest playable handicap *at that pair's mid-rank*,
    i.e. `h_eff_arr[9] · exp(α · r_mid)`. Pairs further apart can't be
    played at a balanced handicap.

    Each candidate's weight is
        w = (SE_a^2 + SE_b^2 + se_floor) / (n_pair + count_alpha)
    so (i) high-uncertainty bots get sampled first, and (ii) any single
    pair has diminishing returns once it has accumulated games.

    Handicap is then a stochastic round of the gap onto the per-pair
    h_eff_local grid (h_eff_arr scaled by exp(α · r_mid))."""
    candidates = []
    weights = []
    n = len(BOT_LABELS)
    max_h_eff = float(h_eff_arr[9])
    for i in range(n):
        a = BOT_LABELS[i]
        ra = ranks.get(a, 0.0)
        sa = _safe_se(std_err.get(a))
        for j in range(i + 1, n):
            b = BOT_LABELS[j]
            rb = ranks.get(b, 0.0)
            r_mid = 0.5 * (ra + rb)
            max_gap_pair = max_h_eff * math.exp(alpha * r_mid)
            if abs(ra - rb) > max_gap_pair:
                continue
            sb = _safe_se(std_err.get(b))
            n_pair = pair_counts.get(frozenset((a, b)), 0)
            w = (sa * sa + sb * sb + se_floor) / (n_pair + count_alpha)
            candidates.append((a, b))
            weights.append(w)

    if not candidates:
        # Pathological: priors should keep some adjacent pair in range.
        # Fall back to a random pair so the worker keeps progress.
        pick_idx = rng.choice(n, size=2, replace=False)
        a, b = BOT_LABELS[int(pick_idx[0])], BOT_LABELS[int(pick_idx[1])]
    else:
        w_arr = np.asarray(weights, dtype=float)
        w_arr /= w_arr.sum()
        pick = int(rng.choice(len(candidates), p=w_arr))
        a, b = candidates[pick]

    ra, rb = ranks[a], ranks[b]
    if ra >= rb:
        stronger, weaker = a, b
    else:
        stronger, weaker = b, a
    gap = abs(ra - rb)

    r_mid = 0.5 * (ra + rb)
    scale = math.exp(alpha * r_mid)
    h_eff_local = [float(h_eff_arr[h]) * scale for h in H_LEVELS]

    if gap <= h_eff_local[0]:
        return stronger, weaker, H_LEVELS[0]
    if gap >= h_eff_local[-1]:
        return stronger, weaker, H_LEVELS[-1]
    for k in range(len(h_eff_local) - 1):
        lo, hi = h_eff_local[k], h_eff_local[k + 1]
        if lo <= gap <= hi:
            frac = (gap - lo) / (hi - lo) if hi > lo else 0.0
            h = H_LEVELS[k + 1] if rng.random() < frac else H_LEVELS[k]
            return stronger, weaker, h
    return stronger, weaker, H_LEVELS[-1]


# --- Report ---------------------------------------------------------------

def aggregate_pair_stats(games):
    """{(strong, weak, h): {'wins_strong': n, 'wins_weak': n, 'draws': n}}"""
    out = defaultdict(lambda: {"wins_strong": 0, "wins_weak": 0, "draws": 0})
    for g in games:
        key = (g["bot_strong"], g["bot_weak"], g["handicap"])
        if g["draw"]:
            out[key]["draws"] += 1
        elif g["stronger_won"]:
            out[key]["wins_strong"] += 1
        else:
            out[key]["wins_weak"] += 1
    return out


def print_report(games, ranks, c, std_err, h_eff_arr, h_eff_se,
                 alpha, alpha_se, verbose=False):
    total = len(games)
    decisive = sum(1 for g in games if not g["draw"])
    print(f"\n{'=' * 90}")
    print(f"HANDICAP CALIBRATION  ({decisive} decisive / {total} total games)")
    print(f"  slope c = {c:.3f}  "
          f"(P(stronger wins) at +1 stone imbalance = {expit(c):.3f})")
    alpha_se_str = (f"{alpha_se:.4f}" if not math.isnan(alpha_se)
                    and math.isfinite(alpha_se) else "  -")
    factor_neg = math.exp(alpha * -10) if math.isfinite(alpha) else float("nan")
    factor_pos = math.exp(alpha * +10) if math.isfinite(alpha) else float("nan")
    print(f"  α       = {alpha:+.4f} ± {alpha_se_str}   "
          f"(handicap-value factor: ×{factor_neg:.2f} at mid-rank -10, "
          f"×{factor_pos:.2f} at +10)")
    print(f"{'=' * 90}")

    # h_eff curve table (curve at anchor mid-rank r_mid=0)
    print(f"\n{'h':>3} {'h_eff_base':>11} {'± SE':>8} {'linear':>8} {'drift':>8}")
    print("-" * 44)
    for h in H_LEVELS:
        base = float(h_eff_arr[h])
        se = h_eff_se.get(h, float("nan"))
        if h == 0:
            se_str = " fixed"
        elif not math.isnan(se) and math.isfinite(se):
            se_str = f"{se:.3f}"
        else:
            se_str = "  -"
        lin = h_eff_linear(h)
        drift = base - lin
        print(f"{h:>3} {base:>+11.3f} {se_str:>8} {lin:>+8.2f} {drift:>+8.2f}")

    # h_eff curve at sample mid-ranks (eyeball curvature shift with rank)
    print(f"\nh_eff(h) · exp(α · r_mid) at sample mid-ranks:")
    sample_mids = [-10, 0, 5]
    header = "  h  " + "  ".join(f"r_mid={m:+d}" for m in sample_mids)
    print(header)
    print("-" * len(header))
    for h in H_LEVELS:
        vals = [float(h_eff_arr[h]) * math.exp(alpha * m) for m in sample_mids]
        row = f"  {h}  " + "  ".join(f"{v:>+8.2f}" for v in vals)
        print(row)

    # Rank table
    print(f"\n{'Bot':<6} {'Rank':>8} {'Nominal':>9} {'Drift':>8} {'± SE':>8}")
    print("-" * 44)
    for b in BOT_LABELS:
        nominal = float(BOT_LABELS.index(b) - ANCHOR_IDX)
        drift = ranks[b] - nominal
        se = std_err.get(b, float("nan"))
        se_str = f"{se:.2f}" if not math.isnan(se) and math.isfinite(se) else "  -"
        marker = "  (anchor)" if b == ANCHOR_BOT else ""
        print(f"{b:<6} {ranks[b]:>+8.2f} {nominal:>+9.2f} {drift:>+8.2f} "
              f"{se_str:>8}{marker}")

    # Multi-hop sanity check: for each triple, compare direct gap from MLE
    # vs sum of adjacent gaps. Since MLE enforces consistency by construction,
    # the more informative check is *raw* per-pair gap (from win rate at the
    # most-played handicap) vs the chained-through-intermediate path.
    pair_stats = aggregate_pair_stats(games)

    def raw_gap(strong, weak):
        """Best per-pair rank-gap estimate from games at any handicap.
        gap ≈ h_eff_eff(h, r_mid) + (1/c) * logit(p_strong)."""
        if c <= 0:
            return None
        hits = [(h, v) for (s, w, h), v in pair_stats.items()
                if s == strong and w == weak]
        if not hits:
            return None
        best = max(hits, key=lambda kv: kv[1]["wins_strong"] + kv[1]["wins_weak"])
        h, v = best
        n = v["wins_strong"] + v["wins_weak"]
        if n == 0:
            return None
        p = max(0.05, min(0.95, v["wins_strong"] / n))
        r_mid = 0.5 * (ranks.get(strong, 0.0) + ranks.get(weak, 0.0))
        return h_eff_eff(h, r_mid, h_eff_arr, alpha) + math.log(p / (1 - p)) / c

    if verbose:
        print(f"\nMulti-hop sanity (raw gaps; direct vs chained through intermediate):")
        print(f"{'A':<5} {'B':<5} {'C':<5} {'AB+BC':>7} {'AC direct':>10} "
              f"{'diff':>7} {'games':>7}")
        print("-" * 50)
        triples_shown = 0
        for i, a in enumerate(BOT_LABELS):
            for j, b in enumerate(BOT_LABELS[i + 1:], start=i + 1):
                for c_ in BOT_LABELS[j + 1:]:
                    gab = raw_gap(b, a)
                    gbc = raw_gap(c_, b)
                    gac = raw_gap(c_, a)
                    if gab is None or gbc is None or gac is None:
                        continue
                    chained = gab + gbc
                    nab = sum(v["wins_strong"] + v["wins_weak"]
                              for (s, w, _h), v in pair_stats.items()
                              if {s, w} == {a, b})
                    nbc = sum(v["wins_strong"] + v["wins_weak"]
                              for (s, w, _h), v in pair_stats.items()
                              if {s, w} == {b, c_})
                    nac = sum(v["wins_strong"] + v["wins_weak"]
                              for (s, w, _h), v in pair_stats.items()
                              if {s, w} == {a, c_})
                    print(f"{a:<5} {b:<5} {c_:<5} {chained:>+7.2f} "
                          f"{gac:>+10.2f} {chained - gac:>+7.2f} "
                          f"{min(nab, nbc, nac):>7d}")
                    triples_shown += 1
                    if triples_shown >= 20:
                        break
                if triples_shown >= 20:
                    break
            if triples_shown >= 20:
                break

        # Per-pair-handicap table
        print(f"\nPer-pair games (most-played handicap shown):")
        print(f"{'Strong':<6} {'Weak':<6} {'h':>3} {'StrW':>5} {'WkW':>5} "
              f"{'Dr':>4} {'StrW%':>6} {'all-h n':>8}")
        print("-" * 50)
        by_pair = defaultdict(list)
        for (s, w, h), v in pair_stats.items():
            by_pair[(s, w)].append((h, v))
        for pair, lst in sorted(by_pair.items()):
            best = max(lst, key=lambda kv: kv[1]["wins_strong"] + kv[1]["wins_weak"])
            h, v = best
            n = v["wins_strong"] + v["wins_weak"]
            pct = (v["wins_strong"] / n * 100) if n else 0
            n_all = sum(vv["wins_strong"] + vv["wins_weak"] + vv["draws"]
                        for _, vv in lst)
            print(f"{pair[0]:<6} {pair[1]:<6} {h:>3} {v['wins_strong']:>5} "
                  f"{v['wins_weak']:>5} {v['draws']:>4} {pct:>5.1f}% "
                  f"{n_all:>8}")

    print(f"{'=' * 90}\n")


# --- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate humanv0 handicap ranks via 19x19 self-play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--threads", type=int, default=128,
                        help="Parallel games / KataGo analysis threads")
    parser.add_argument("--max-games", type=int, default=0,
                        help="Stop after N games total (0 = run forever)")
    parser.add_argument("--refit-interval", type=int, default=30,
                        help="Refit ranks every N decisive games")
    parser.add_argument("--report-interval", type=int, default=60,
                        help="Print full report every N games")
    parser.add_argument("--prior-rank-sigma", type=float, default=3.0,
                        help="Gaussian prior σ pulling each rank to its "
                             "nominal label (looser = more data-driven, "
                             "tighter = more stable on sparse data)")
    parser.add_argument("--prior-c-sigma", type=float, default=0.5,
                        help="Gaussian prior σ on the slope c (centered at 1)")
    parser.add_argument("--prior-heff-sigma", type=float, default=1.0,
                        help="Gaussian prior σ on h_eff[h] (centered at the "
                             "h − 0.5 linear assumption)")
    parser.add_argument("--prior-alpha-sigma", type=float, default=0.05,
                        help="Gaussian prior σ on the rank-coupling α "
                             "(centered at 0; α=0 means handicap value is "
                             "independent of absolute rank)")
    parser.add_argument("--report-only", action="store_true",
                        help="Refit + print from existing SGFs and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(SGF_DIR, exist_ok=True)

    print(f"Recovering games from {SGF_DIR} ...")
    games = recover_games(SGF_DIR)
    print(f"  recovered {len(games)} games "
          f"({sum(1 for g in games if not g['draw'])} decisive)")

    (ranks, c, std_err, h_eff_arr, h_eff_se,
     alpha, alpha_se) = fit_ranks(
        games, sigma_rank=args.prior_rank_sigma,
        sigma_c=args.prior_c_sigma,
        sigma_heff=args.prior_heff_sigma,
        sigma_alpha=args.prior_alpha_sigma)
    print_report(games, ranks, c, std_err, h_eff_arr, h_eff_se,
                 alpha, alpha_se, verbose=args.verbose)

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
    kc = KataGoController(model_path, human_path)
    time.sleep(3)
    if kc.process.poll() is not None:
        print(f"ERROR: KataGo exited (code {kc.process.returncode})",
              file=sys.stderr)
        sys.exit(1)

    # Warmup
    try:
        resp = kc.query([], [], EVEN_KOMI, {
            "maxVisits": 1, "includePolicy": True,
            "overrideSettings": {"humanSLProfile": KATAGO_PROFILE[ANCHOR_BOT]},
        })
        if "humanPolicy" not in resp:
            print("ERROR: humanPolicy missing from response. "
                  "Is -human-model wired up?", file=sys.stderr)
            kc.close()
            sys.exit(1)
        print("KataGo ready.")
    except Exception as e:
        print(f"ERROR: warmup failed: {e}", file=sys.stderr)
        kc.close()
        sys.exit(1)

    state_lock = threading.Lock()
    pair_counts = defaultdict(int)
    for g in games:
        pair_counts[frozenset((g["bot_strong"], g["bot_weak"]))] += 1
    state = {
        "games": games,
        "ranks": ranks,
        "c": c,
        "std_err": std_err,
        "h_eff_arr": h_eff_arr,
        "h_eff_se": h_eff_se,
        "alpha": alpha,
        "alpha_se": alpha_se,
        "pair_counts": pair_counts,
        "since_refit": 0,
        "since_report": 0,
        "color_alt": defaultdict(int),  # h=0 games: alternate black/white assignment
    }

    def worker():
        rng = np.random.default_rng()
        while True:
            with state_lock:
                if args.max_games > 0 and len(state["games"]) >= args.max_games:
                    return
                stronger, weaker, h = pick_pair_and_handicap(
                    state["ranks"], state["std_err"],
                    state["pair_counts"],
                    state["h_eff_arr"], state["alpha"], rng)
                if h == 0:
                    # Alternate who plays black across h=0 games for this pair.
                    pair_key = tuple(sorted([stronger, weaker]))
                    flip = state["color_alt"][pair_key] % 2 == 1
                    state["color_alt"][pair_key] += 1
                else:
                    flip = False
                if h >= 1:
                    black, white = weaker, stronger
                else:
                    black, white = (stronger, weaker) if flip else (weaker, stronger)

            try:
                moves, score, result_tag = play_game(kc, black, white, h, rng)
            except Exception as e:
                print(f"  game error ({black} v {white} h={h}): {e}")
                time.sleep(1)
                continue

            sgf = make_sgf(black, white, h, moves, result_tag)
            fname = f"{stronger}_v_{weaker}_h{h}_{uuid.uuid4().hex[:8]}.sgf"
            with open(os.path.join(SGF_DIR, fname), "w") as f:
                f.write(sgf)

            draw = result_tag.startswith("Void")
            if not draw:
                black_won = result_tag.startswith("B+")
                stronger_won = (black == stronger and black_won) or \
                               (white == stronger and not black_won)
            else:
                stronger_won = None
            game_rec = {"bot_strong": stronger, "bot_weak": weaker,
                        "handicap": h, "stronger_won": stronger_won,
                        "draw": draw}

            with state_lock:
                state["games"].append(game_rec)
                state["pair_counts"][frozenset((stronger, weaker))] += 1
                state["since_refit"] += 0 if draw else 1
                state["since_report"] += 1

                if args.verbose:
                    outcome = "Dr" if draw else (
                        "S+" if stronger_won else "W+")
                    print(f"[{len(state['games']):5d}] "
                          f"{stronger:>4} v {weaker:<4} h={h} -> "
                          f"{outcome} ({result_tag})")

                if state["since_refit"] >= args.refit_interval:
                    state["since_refit"] = 0
                    try:
                        (new_ranks, new_c, new_se, new_heff, new_heff_se,
                         new_alpha, new_alpha_se) = fit_ranks(
                            state["games"], init_ranks=state["ranks"],
                            init_c=state["c"],
                            init_h_eff=state["h_eff_arr"],
                            init_alpha=state["alpha"],
                            sigma_rank=args.prior_rank_sigma,
                            sigma_c=args.prior_c_sigma,
                            sigma_heff=args.prior_heff_sigma,
                            sigma_alpha=args.prior_alpha_sigma)
                        state["ranks"] = new_ranks
                        state["c"] = new_c
                        state["std_err"] = new_se
                        state["h_eff_arr"] = new_heff
                        state["h_eff_se"] = new_heff_se
                        state["alpha"] = new_alpha
                        state["alpha_se"] = new_alpha_se
                    except Exception as e:
                        print(f"  refit failed: {e}")

                if state["since_report"] >= args.report_interval:
                    state["since_report"] = 0
                    print_report(state["games"], state["ranks"], state["c"],
                                 state["std_err"], state["h_eff_arr"],
                                 state["h_eff_se"], state["alpha"],
                                 state["alpha_se"], verbose=args.verbose)

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
        kc.close()

    print_report(state["games"], state["ranks"], state["c"],
                 state["std_err"], state["h_eff_arr"],
                 state["h_eff_se"], state["alpha"],
                 state["alpha_se"], verbose=args.verbose)
    print("Done.")


if __name__ == "__main__":
    main()
