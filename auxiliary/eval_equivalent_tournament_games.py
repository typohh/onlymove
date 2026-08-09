"""How many tournament games would it take to rank a player as well as onlymove does?

The claim we want to be able to make is "our rank is about as accurate as an EGF
rating built from x tournament games". This script produces x by simulating the
tournament side from scratch and matching its accuracy to the measured onlymove
accuracy (`auxiliary.check_mapped_vs_declared_rank --csv`), on the same three
statistics, in EGF rank units:

  Spearman rho, % within 1 rank, % within 2 ranks.

  1. LINK. `data/reference/even_game_winrates.txt` first table gives, per declared
     grade G and stone gap d, how often the WEAKER player won an even game. That is
     the win-probability law, and it is strongly rank-dependent: a stone buys far
     more at dan than at kyu, so a dan game carries several times the information of
     a kyu one. It is fitted here as a STONE-VALUE curve beta[b] — the logit gap
     bought by crossing from band b into band b+1 — so that

         P(stronger wins) = sigmoid(phi(r_strong) - phi(r_weak)),  phi' = beta.

     phi is the latent Elo-like scale the stone value varies along; on it the model
     is plain Bradley-Terry, and every rank quoted below is phi mapped back through
     phi^-1, i.e. an EGF grade. Fitting beta jointly across all (G, d) cells rather
     than a separate logistic slope per grade is what lets the d=1..4 columns bend:
     a 4-stone gap spans four different stone values, not four copies of one.

  2. POPULATION. Tournament entrants are drawn from the EGF table's own
     game-weighted grade distribution (`--population egf`), or from our fitted user
     distribution (`--population prior`). Population matters twice: it sets who is
     available to be paired with, and — because Spearman is range-restricted — it
     sets what rho even means. The EVALUATION sample is therefore resampled to match
     the rank distribution of the onlymove cohort being matched, at the same n, so
     the simulated rho and the measured rho are the same statistic.

  3. TOURNAMENT. Every round is a perfect matching, so "the last x games" and "the
     last x rounds" are the same window for everyone and no player's window can
     straddle a game another player has already dropped. Pairing is near-neighbour on
     the current estimates, which is where Fisher information peaks, with recent
     opponents excluded as a real Swiss does. Note that pairing STRICTLY by
     information (`--pair-mixing 0`) is measurably the wrong choice: it maximizes
     what each game says about a local comparison, and leaves the field a chain that
     pins order without pinning distance, so absolute rank error stops falling with
     more games while rho carries on rising. Some long-range mixing is required, and
     a Swiss delivers it by pairing on score rather than on rating.

  4. ESTIMATOR. Bradley-Terry MAP in phi-space over the trailing window only, which
     is what makes "x games of information" a steady state rather than something
     that keeps improving forever. The population prior mirrors what onlymove's own
     estimator does, and is not optional: over a trailing window some player is
     always separated (every game won, or every one lost) and the unpenalized MLE
     sends them to infinity. Renormalizing the mean does not rescue it — that fixes
     the location, and the divergence is in the spread. `--anchor-width` widens the
     prior toward a free fit, and measurably makes absolute rank worse while leaving
     rho alone, for the same reason the pairing does: the prior is carrying part of
     the ladder's absolute scale.

  5. MATCH. Sweep x, measure in steady state, interpolate to the x whose rho / w1 /
     w2 equal the measured onlymove values.

WHAT THE ANSWER IS AND IS NOT. Our measured error is (our estimate - declaration),
and a declaration carries its own error: most are self-reports, quantized to a whole
band. So `--compare truth`, which scores the simulated rating against the ground
truth it was generated from, is matching our error against something strictly
cleaner, and returns FEWER games than we deserve — read it as a floor.
`--compare peer` (default) instead scores one x-game rating against a SECOND,
independent x-game rating of the same player, which is the like-for-like statement:
"our rank agrees with a player's declared rank as closely as two independent x-game
tournament ratings agree with each other". It assumes nothing about which side of
our measurement carries the noise, which is why it is the headline. The two windows
share no games, but the later one's pairings were chosen using the earlier one's
result, so a player who ran hot early meets stronger opponents later: the two errors
are mildly anti-correlated and their difference is if anything slightly wider than
two independent draws.

Caveats worth keeping attached to any number this prints. (a) The first table is
indexed by DECLARED grade, so both players' grade errors flatten the observed curve;
the flattening is severe in the kyu range, where declarations are loosest, and it
makes kyu games look less informative than they are. `--link se-blocks` refits from
the same file's Se-bucketed blocks, which are indexed by RATING instead, and is the
sensitivity check on exactly that. (b) The 20k row is the entry grade and absorbs
everyone weaker, so its stone value is inflated; `--drop-grades 20k` removes it.
(c) The simulated estimator knows the true link and the true population, and
abilities do not drift; all three flatter the tournament side, so the matched x is
if anything conservative.

Usage:
  ./.venv/bin/python3 -m auxiliary.check_mapped_vs_declared_rank --csv /tmp/decl.csv
  ./.venv/bin/python3 -m auxiliary.eval_equivalent_tournament_games --targets /tmp/decl.csv
"""

import argparse
import os
import re
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.rank_surveys import rank_to_value, value_to_rank  # noqa: E402

TABLE_PATH = os.path.join(PROJECT_ROOT, "data", "reference", "even_game_winrates.txt")
POP_CONSTANTS = os.path.join(PROJECT_ROOT, "ratings", "population_constants.json")

# Stone values are fitted on band indices (band b spans [b, b+1); 0 is the 1k/1d
# boundary). beta[b] is the gap from band b to band b+1, so a table cell at grade G
# with a d-stone gap consumes beta[G..G+d-1] and the fit needs d-1 bands of headroom
# past the strongest grade in the table.
MAX_GAP = 4


# ---------------------------------------------------------------------------
# The reference table
# ---------------------------------------------------------------------------

def parse_even_game_table(path):
    """First table -> [(band, gap, n_win_weak, n_games)].

    Field spans come from the table's own dashed rule, so a group whose win count is
    blank (a 0% row prints only games and percentage) still lands in the right
    column."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    rule = next(i for i, ln in enumerate(lines) if ln.startswith("--- ---"))
    spans = [(m.start(), m.end()) for m in re.finditer(r"-{5,}", lines[rule])]
    cells = []
    for ln in lines[rule + 1:]:
        label = ln[:spans[0][0]].strip() if ln[:3].strip() else ""
        if not label:
            break
        band = rank_to_value(label)
        if band is None:
            break
        band = int(np.floor(band))
        for gap, (lo, hi) in enumerate(spans, start=1):
            tok = ln[lo:hi].split()
            if len(tok) == 3:
                nw, ng = int(tok[0]), int(tok[1])
            elif len(tok) == 2:                      # win count blank == zero wins
                ng, nw = int(tok[0]), 0
            else:
                continue
            if ng > 0:
                cells.append((band, gap, nw, ng))
    return cells


def parse_se_blocks(path):
    """The Se-bucketed even-game blocks -> {'kyu': [...], 'dan': [...]}.

    Rows are (rating gap in grades, wins by the weaker side, games). Unlike the first
    table these are indexed by the players' RATINGS, so they are not flattened by
    grade-declaration error — which is the whole reason they are here."""
    with open(path) as fh:
        text = fh.read()
    out = {}
    for key, header in (("kyu", "kyu players"), ("dan", "dan players")):
        start = text.index(header)
        rows = []
        for ln in text[start:].split("\n")[4:]:
            tok = ln.split()
            if len(tok) < 6:
                if rows:
                    break
                continue
            adr, nw, ng = float(tok[2]), int(tok[3]), int(tok[4])
            if ng > 0:
                rows.append((adr / 100.0, nw, ng))     # 100 GoR per grade
        out[key] = rows
    return out


# ---------------------------------------------------------------------------
# Stone-value curve: the win-probability law as a latent scale
# ---------------------------------------------------------------------------

class StoneScale:
    """phi, the latent scale on which one logit is one logit.

    Holds beta per band and the cumulative phi at every band EDGE, so phi and its
    inverse are both piecewise-linear interpolations and monotone by construction
    (beta > 0). Outside the fitted range beta is held at its end value, which extends
    phi linearly rather than letting it flatten or fold."""

    def __init__(self, beta_by_band, lo=-40, hi=15):
        self.lo, self.hi = lo, hi
        self.edges = np.arange(lo, hi + 1, dtype=float)
        fitted = sorted(beta_by_band)
        beta = np.array([beta_by_band[min(max(b, fitted[0]), fitted[-1])]
                         for b in range(lo, hi)], float)
        self.beta = beta
        self.phi_edges = np.concatenate([[0.0], np.cumsum(beta)])
        self.phi_edges -= np.interp(0.0, self.edges, self.phi_edges)

    def phi(self, rank):
        return np.interp(rank, self.edges, self.phi_edges)

    def rank(self, phi):
        return np.interp(phi, self.phi_edges, self.edges)

    def stone_value(self, rank):
        """Logits per grade at this rank — the local information density."""
        idx = np.clip(np.floor(rank).astype(int) - self.lo, 0, len(self.beta) - 1)
        return self.beta[idx]


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def fit_stone_values(cells, smooth=8.0):
    """Binomial MLE for beta over all (grade, gap) cells at once.

    log beta is the free parameter so beta stays positive, and the penalty is on its
    second difference, so thin cells at the ends extrapolate the trend instead of
    running off. One shared curve across all gaps is what makes the d=1..4 columns
    bend: a 4-stone gap crosses four different stone values."""
    from scipy.optimize import minimize

    bands = sorted({b for b, _, _, _ in cells})
    lo, hi = bands[0], bands[-1] + MAX_GAP
    index = {b: i for i, b in enumerate(range(lo, hi))}
    n = len(index)

    rows = []
    for band, gap, nw, ng in cells:
        mask = np.zeros(n)
        for k in range(band, band + gap):
            mask[index[k]] = 1.0
        rows.append((mask, nw, ng))
    design = np.array([r[0] for r in rows])
    n_win = np.array([r[1] for r in rows], float)
    n_all = np.array([r[2] for r in rows], float)

    def nll(theta):
        gap = design @ np.exp(theta)
        p_weak = _sigmoid(-gap)
        p_weak = np.clip(p_weak, 1e-9, 1 - 1e-9)
        ll = n_win * np.log(p_weak) + (n_all - n_win) * np.log1p(-p_weak)
        d2 = np.diff(theta, 2)
        return -ll.sum() + smooth * float(d2 @ d2) * n_all.sum() / 1000.0

    res = minimize(nll, np.full(n, np.log(0.3)), method="L-BFGS-B",
                   options={"maxiter": 4000})
    beta = np.exp(res.x)
    return {b: float(beta[i]) for b, i in index.items()}, (design, n_win, n_all, beta)


def fit_stone_values_se_blocks(blocks, bands):
    """One stone value for kyu and one for dan, from the rating-indexed blocks.

    Coarse by construction — the blocks pool every rating level inside each half of
    the ladder — so it is a sensitivity check on the declared-grade flattening, not a
    replacement curve. The two constants are blended across the 1k/1d boundary so the
    scale has no step in it."""
    from scipy.optimize import minimize

    fitted = {}
    for key, rows in blocks.items():
        gaps = np.array([r[0] for r in rows])
        n_win = np.array([r[1] for r in rows], float)
        n_all = np.array([r[2] for r in rows], float)

        def nll(theta):
            p_weak = np.clip(_sigmoid(-np.exp(theta[0]) * gaps), 1e-9, 1 - 1e-9)
            return -float((n_win * np.log(p_weak)
                           + (n_all - n_win) * np.log1p(-p_weak)).sum())

        fitted[key] = float(np.exp(minimize(nll, [np.log(0.4)],
                                            method="Nelder-Mead").x[0]))
    blend = 2.0
    out = {}
    for b in bands:
        w = float(np.clip((b + 0.5 + blend) / (2 * blend), 0.0, 1.0))
        out[b] = (1 - w) * fitted["kyu"] + w * fitted["dan"]
    return out, fitted


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

def egf_population(cells):
    """Grade -> share of tournament PLAYERS, proxied by games played.

    A game at gap d is one game for the weak grade and one for the grade d stronger,
    so both sides are counted. Game-weighted rather than head-count on purpose: the
    people who can be paired with are the people who turn up. The table tabulates
    only games across a gap of 1..4, so same-grade games — the commonest kind — are
    not in this proxy at all; `--population prior` is the check on how much that
    shape matters, and with a dense field it is mostly the pairing pool, since the
    evaluation sample is resampled to the target cohort regardless."""
    weight = {}
    for band, gap, _, ng in cells:
        weight[band] = weight.get(band, 0.0) + ng
        weight[band + gap] = weight.get(band + gap, 0.0) + ng
    bands = np.array(sorted(weight))
    w = np.array([weight[b] for b in bands], float)
    return bands, w / w.sum()


def prior_population(path):
    """Our own fitted user distribution, on its shipped rank grid."""
    import json
    with open(path) as fh:
        prior = json.load(fh)["rank_prior"]
    log_density = np.asarray(prior["log_density"], float)
    grid = prior["rank_lo"] + prior["rank_step"] * np.arange(len(log_density))
    p = np.exp(log_density - log_density.max())
    bands = np.array(sorted({int(np.floor(v)) for v in grid}))
    w = np.array([p[(np.floor(grid) == b)].sum() for b in bands], float)
    return bands, w / w.sum()


def draw_population(bands, weights, n, rng):
    """Continuous true ranks: a band by its share, then uniform inside it."""
    picked = rng.choice(bands, size=n, p=weights)
    return picked + rng.random(n)


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

def pair_round(order_key, recent, rng, jitter=0.0):
    """Perfect matching, near-neighbour on the estimate, no recent rematches.

    Fisher information for a Bradley-Terry comparison peaks where the two players are
    equal, so pairing the closest available opponent maximizes information PER GAME —
    and on its own it is the wrong objective. Everyone playing only their own
    neighbourhood makes the comparison graph a chain, and a chain pins local order
    while leaving its own length free: absolute rank error then stops falling with
    more games even though rho keeps rising. `jitter` on the pairing key is the
    long-range mixing that repairs it, and is what a real Swiss supplies by pairing
    on score rather than on rating. The rematch bar serves the same end at short
    range, by stopping a window from spending itself on one comparison."""
    n = len(order_key)
    key = order_key + (rng.normal(0, jitter, n) if jitter > 0 else 0.0)
    order = np.argsort(key)
    taken = np.zeros(n, bool)
    pairs = []
    for pos, i in enumerate(order):
        if taken[i]:
            continue
        taken[i] = True
        best = -1
        for j in order[pos + 1:]:
            if taken[j]:
                continue
            if best < 0:
                best = j
            if j not in recent[i]:
                best = j
                break
        if best < 0:
            taken[i] = False
            continue
        taken[best] = True
        pairs.append((i, best))
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def play(pairs, phi_true, rng):
    """Outcome of every game in a round: did the FIRST player of the pair win?"""
    p = _sigmoid(phi_true[pairs[:, 0]] - phi_true[pairs[:, 1]])
    return rng.random(len(pairs)) < p


def bt_map(games, n, phi_init, prior, newton=6, cg_iters=400, tol=1e-7):
    """Bradley-Terry MAP over a set of rounds, by Newton with a CG inner solve.

    The Hessian is a weighted graph Laplacian plus the prior's diagonal, so it is
    only ever applied, never formed. CG rather than a per-player (diagonal) step
    because information-optimal pairing builds a BAND graph — everyone plays their
    own neighbourhood — and a Laplacian's slowest modes are exactly the long-range
    stretch along that band, which is the error this whole exercise measures. A
    diagonal step damps those modes last, so it stops LOOKING like it is moving long
    before it has stopped.

    `prior` is (mu, sd) and is required: a trailing window under information-optimal
    pairing routinely separates somebody completely (every game won, or lost), and
    an unpenalized MLE sends them to infinity. Renormalizing the mean afterwards does
    not fix that — it fixes the location, and the divergence is in the spread. To
    approximate a free fit, widen the prior instead of removing it."""
    if not games:
        return phi_init.copy()
    a = np.concatenate([g[0][:, 0] for g in games])
    b = np.concatenate([g[0][:, 1] for g in games])
    y = np.concatenate([g[1] for g in games]).astype(float)
    mu, sd = prior
    phi = phi_init.copy()

    for _ in range(newton):
        p = _sigmoid(phi[a] - phi[b])
        resid = y - p
        grad = np.bincount(a, weights=resid, minlength=n) \
            - np.bincount(b, weights=resid, minlength=n) + (mu - phi) / sd ** 2
        w = p * (1.0 - p)

        def apply_h(v):
            d = w * (v[a] - v[b])
            return (np.bincount(a, weights=d, minlength=n)
                    - np.bincount(b, weights=d, minlength=n) + v / sd ** 2)

        step = np.zeros(n)
        r = grad.copy()
        d = r.copy()
        rr = float(r @ r)
        target = rr * 1e-12
        for _ in range(cg_iters):
            if rr <= target:
                break
            hd = apply_h(d)
            alpha = rr / max(float(d @ hd), 1e-300)
            step += alpha * d
            r -= alpha * hd
            rr_new = float(r @ r)
            d = r + (rr_new / rr) * d
            rr = rr_new
        phi = phi + step
        if float(np.max(np.abs(step))) < tol:
            break
    return phi


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def band_centre(rank):
    """A declaration is a band, not a number — collapse to the band's centre."""
    return np.floor(rank) + 0.5


def score(estimate, reference):
    from scipy.stats import spearmanr
    err = estimate - reference
    return {
        "n": len(err),
        "rho": float(spearmanr(estimate, reference).statistic),
        "within1": float(np.mean(np.abs(err) <= 1.0)),
        "within2": float(np.mean(np.abs(err) <= 2.0)),
        "mae": float(np.mean(np.abs(err))),
        "sd": float(np.std(err)),
    }


def cohort_sampler(cohort_ranks, true_ranks, rng):
    """Draw an evaluation sample whose TRUE ranks look like the cohort's.

    Spearman is range-restricted, so a simulated rho is only the same statistic as
    the measured one if the two are computed over the same spread of players at the
    same n. Matching is done band by band; a band the simulation has too few players
    in is sampled with replacement."""
    bands, counts = np.unique(np.floor(cohort_ranks).astype(int), return_counts=True)
    pools = {b: np.flatnonzero(np.floor(true_ranks).astype(int) == b) for b in bands}
    nearest = np.floor(true_ranks).astype(int)

    def draw():
        picked = []
        for b, c in zip(bands, counts):
            pool = pools[b]
            if len(pool) == 0:                      # band unpopulated: nearest one
                pool = np.flatnonzero(np.abs(nearest - b) == np.min(np.abs(nearest - b)))
            picked.append(rng.choice(pool, size=c, replace=len(pool) < c))
        return np.concatenate(picked)
    return draw


# ---------------------------------------------------------------------------
# One steady-state run at a given window
# ---------------------------------------------------------------------------

def run_window(x, scale, true_ranks, sampler, args, rng):
    """Play until the estimator is in steady state at a window of x games, then score.

    The window is what makes a steady state exist at all: without it the estimate
    keeps sharpening with every extra round and "x games of information" has no fixed
    point. Burn-in has to exceed the window itself, since until then the estimator
    has not yet seen a full one."""
    n = len(true_ranks)
    phi_true = scale.phi(true_ranks)
    spread = float(phi_true.std())
    prior = (float(phi_true.mean()), args.anchor_width * spread)
    jitter = args.pair_mixing * spread

    window = x * (2 if args.compare == "peer" else 1)
    # One estimate per half-window, each warm-started from its own previous value and
    # run to convergence, so the pairing key IS the x-game estimate a real organizer
    # would hold — not something smoothed over more history than the window allows.
    phi_hat = np.full(n, float(phi_true.mean()))
    phi_old = phi_hat.copy()
    rounds = []
    recent = [set() for _ in range(n)]
    recent_rounds = []

    total = window + args.burn_in + args.measure_rounds
    measured = []
    for t in range(total):
        pairs = pair_round(phi_hat, recent, rng, jitter)
        wins = play(pairs, phi_true, rng)
        rounds.append((pairs, wins))
        recent_rounds.append(pairs)
        for i, j in pairs:
            recent[i].add(int(j))
            recent[j].add(int(i))
        if len(rounds) > window:
            rounds.pop(0)
        if len(recent_rounds) > args.no_rematch:
            old = recent_rounds.pop(0)
            for i, j in old:
                recent[i].discard(int(j))
                recent[j].discard(int(i))

        phi_hat = bt_map(rounds[-x:], n, phi_hat, prior, newton=args.newton_steps)
        if args.compare == "peer":
            phi_old = bt_map(rounds[:x], n, phi_old, prior, newton=args.newton_steps)

        if t < window + args.burn_in:
            continue

        est = scale.rank(phi_hat)
        if args.compare == "peer":
            peer = scale.rank(phi_old)
            ref_full = band_centre(peer) if args.quantize_ref else peer
        else:
            ref_full = band_centre(true_ranks) if args.quantize_ref else true_ranks
        idx = sampler()
        measured.append(score(est[idx], ref_full[idx]))

    out = {k: float(np.mean([m[k] for m in measured])) for k in measured[0]}
    out["trend_mae"] = float(np.polyfit(np.arange(len(measured)),
                                        [m["mae"] for m in measured], 1)[0]) \
        if len(measured) > 2 else 0.0
    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def load_declarations(csv_path, scope):
    """The declarations we are matching, straight from the check script's CSV.

    Read rather than hardcoded so a refit moves the target with it. `scope` picks the
    declarations; EGF keeps everything on one ladder (self-reported EGF plus the
    verified EGD heartbeats, which is the same coupling), ALL pools every server's own
    units — and note that pooling raises rho by adding the between-server spread to
    it, so a pooled rho is not the same statistic as a per-server one."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if scope != "all":
        df = df[df["system"] == scope]
    if df.empty:
        sys.exit(f"No declarations in scope {scope!r} in {csv_path}")
    return df[["our_internal_adj", "declared_internal", "err_server"]].to_numpy(float)


def target_stats(rows):
    """onlymove's measured accuracy over a set of declaration rows.

    Columns are (our rank, declared rank, error in the server's own rank units).
    Taking rows as an array rather than a frame is what lets the same function score
    a bootstrap resample."""
    from scipy.stats import spearmanr
    err = rows[:, 2]
    return {
        "n": len(rows),
        "rho": float(spearmanr(rows[:, 0], rows[:, 1]).statistic),
        "within1": float(np.mean(np.abs(err) <= 1.0)),
        "within2": float(np.mean(np.abs(err) <= 2.0)),
        "mae": float(np.mean(np.abs(err))),
        "sd": float(np.std(err)),
    }


def match_x(xs, values, target, increasing):
    """The x whose simulated statistic first reaches the measured one.

    Interpolated in log x, since accuracy improves with the square root of games and
    the sweep is geometric. FIRST crossing, walking the sweep in x order, rather than
    inverting the whole curve: a statistic can flatten against its own floor (a
    reference quantized to a band cannot be matched more closely than the band), and
    over a flat stretch the inverse is not a number. Off the end of the swept range it
    says so rather than extrapolating something nothing supports."""
    xs = np.asarray(xs, float)
    v = np.asarray(values, float) * (1.0 if increasing else -1.0)
    target = target * (1.0 if increasing else -1.0)
    if v[0] >= target:
        return None, "<"
    for i in range(1, len(xs)):
        if v[i] >= target:
            span = v[i] - v[i - 1]
            frac = (target - v[i - 1]) / span if span > 0 else 0.0
            log_x = np.log(xs[i - 1]) + frac * (np.log(xs[i]) - np.log(xs[i - 1]))
            return float(np.exp(log_x)), "="
    return None, ">"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=None,
                    help="CSV from `check_mapped_vs_declared_rank --csv`. Without it "
                         "the sweep still prints, but nothing is matched to it.")
    ap.add_argument("--scope", default="EGF",
                    help="Which declarations to match: a system name (EGF, OGS, ...) "
                         "or 'all' to pool every server in its own units.")
    ap.add_argument("--table", default=TABLE_PATH)
    ap.add_argument("--link", choices=["table", "se-blocks"], default="table",
                    help="'table': the declared-grade first table (as published). "
                         "'se-blocks': the rating-indexed Se blocks, which are not "
                         "flattened by grade-declaration error.")
    ap.add_argument("--drop-grades", default="",
                    help="Comma-separated grades to exclude from the link fit; 20k is "
                         "the entry grade and absorbs everyone weaker than it.")
    ap.add_argument("--smooth", type=float, default=8.0,
                    help="Second-difference penalty on the stone-value curve.")
    ap.add_argument("--population", choices=["egf", "prior"], default="egf",
                    help="Who enters the tournament. 'egf': the reference table's own "
                         "game-weighted grade distribution. 'prior': our fitted users.")
    ap.add_argument("--n-players", type=int, default=1200)
    ap.add_argument("--compare", choices=["peer", "truth"], default="peer",
                    help="'peer': one x-game rating against a second independent one "
                         "(like-for-like with our measurement). 'truth': against the "
                         "ground truth, which understates x.")
    ap.add_argument("--quantize-ref", dest="quantize_ref", action="store_true",
                    default=True, help="Collapse the reference to its band centre, as "
                                       "a declared rank is (default).")
    ap.add_argument("--no-quantize-ref", dest="quantize_ref", action="store_false")
    ap.add_argument("--anchor-width", type=float, default=1.0,
                    help="Prior SD as a multiple of the population's own, mirroring "
                         "onlymove's population-anchored estimator at 1. Raise it to "
                         "approach a free MLE; a genuinely free one does not exist "
                         "here, since a trailing window separates some players "
                         "completely.")
    ap.add_argument("--games", default="2,3,5,8,12,20,30,50,80,130,200,320",
                    help="Window sizes x to sweep, in games per player.")
    ap.add_argument("--burn-in", type=int, default=25,
                    help="Rounds after the window is full before measuring.")
    ap.add_argument("--measure-rounds", type=int, default=12)
    ap.add_argument("--reps", type=int, default=3,
                    help="Independent populations per window size.")
    ap.add_argument("--no-rematch", type=int, default=8,
                    help="Rounds an opponent stays excluded from re-pairing.")
    ap.add_argument("--pair-mixing", type=float, default=0.4,
                    help="Noise on the pairing key as a fraction of the population's "
                         "spread — how far outside their own neighbourhood players "
                         "are matched. 0 is per-game information-optimal and is NOT "
                         "best for absolute rank: it leaves the ladder a chain whose "
                         "own length is unmeasured.")
    ap.add_argument("--newton-steps", type=int, default=6,
                    help="Newton steps per estimate, from the previous round's warm "
                         "start; each solves the Hessian by CG to tolerance.")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="Resamples of the declarations behind the target, for the "
                         "sampling margin on the matched game count. 0 to skip.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # --- the win-probability law ---
    cells = parse_even_game_table(args.table)
    dropped = {int(np.floor(rank_to_value(g.strip())))
               for g in args.drop_grades.split(",") if g.strip()}
    if dropped:
        cells = [c for c in cells if c[0] not in dropped
                 and c[0] + c[1] not in dropped]
    bands_needed = range(min(c[0] for c in cells), max(c[0] for c in cells) + MAX_GAP)

    if args.link == "table":
        beta, diag = fit_stone_values(cells, smooth=args.smooth)
        scale = StoneScale(beta)
        design, n_win, n_all, beta_vec = diag
        print(f"stone-value curve from the declared-grade table "
              f"({len(cells)} cells, {int(n_all.sum())} games, smooth={args.smooth})")
    else:
        beta, halves = fit_stone_values_se_blocks(parse_se_blocks(args.table),
                                                  bands_needed)
        scale = StoneScale(beta)
        print(f"stone-value curve from the rating-indexed Se blocks  "
              f"(kyu {halves['kyu']:.3f}, dan {halves['dan']:.3f} logits/grade)")

    print(f"{'grade':>7}{'logits/stone':>14}{'P(stronger wins) 1 stone':>26}")
    for b in sorted(beta):
        if b % 3 == 0 or b == 0:
            print(f"{value_to_rank(b + 0.5):>7}{beta[b]:>14.3f}"
                  f"{_sigmoid(beta[b]):>26.3f}")

    if args.link == "table":
        print("\nfit check — observed vs modelled % wins for the WEAKER player")
        print(f"{'grade':>7}" + "".join(f"{'+' + str(d):>14}" for d in range(1, 5)))
        by_cell = {(b, g): (nw, ng) for b, g, nw, ng in cells}
        for b in sorted({c[0] for c in cells}):
            row = f"{value_to_rank(b + 0.5):>7}"
            for d in range(1, 5):
                if (b, d) not in by_cell:
                    row += f"{'-':>14}"
                    continue
                nw, ng = by_cell[(b, d)]
                gap = sum(beta[k] for k in range(b, b + d))
                row += f"{100 * nw / ng:>7.1f}{100 * _sigmoid(-gap):>7.1f}"
            print(row)

    # --- who plays ---
    if args.population == "egf":
        pop_bands, pop_w = egf_population(cells)
    else:
        pop_bands, pop_w = prior_population(POP_CONSTANTS)

    # --- what we are matching ---
    targets, cohort_ranks, decl = (None, None, None)
    if args.targets:
        decl = load_declarations(args.targets, args.scope)
        targets = target_stats(decl)
        cohort_ranks = decl[:, 1]
        print(f"\nonlymove measured accuracy  [scope={args.scope}, "
              f"n={targets['n']}]")
        print(f"  rho {targets['rho']:.3f}   within1 {100*targets['within1']:.0f}%"
              f"   within2 {100*targets['within2']:.0f}%"
              f"   mae {targets['mae']:.2f}   sd {targets['sd']:.2f}")
    else:
        cohort_ranks = draw_population(pop_bands, pop_w, 200,
                                       np.random.default_rng(args.seed))

    # --- sweep ---
    xs = [int(v) for v in args.games.split(",")]
    print(f"\ntournament simulation  [link={args.link}, population={args.population}, "
          f"n={args.n_players}, compare={args.compare}, "
          f"anchor width={args.anchor_width:g}, "
          f"reps={args.reps}]")
    print(f"{'games/player':>13}{'rho':>8}{'within1':>10}{'within2':>10}"
          f"{'mae':>8}{'sd':>8}{'mae trend/round':>17}")
    curves = {k: [] for k in ("rho", "within1", "within2", "mae", "sd")}
    for x in xs:
        reps = []
        for r in range(args.reps):
            rng = np.random.default_rng(args.seed + 1009 * r + 7 * x)
            true_ranks = draw_population(pop_bands, pop_w, args.n_players, rng)
            sampler = cohort_sampler(cohort_ranks, true_ranks, rng)
            reps.append(run_window(x, scale, true_ranks, sampler, args, rng))
        agg = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
        for k in curves:
            curves[k].append(agg[k])
        print(f"{x:>13}{agg['rho']:>8.3f}{100*agg['within1']:>9.0f}%"
              f"{100*agg['within2']:>9.0f}%{agg['mae']:>8.2f}{agg['sd']:>8.2f}"
              f"{agg['trend_mae']:>+17.4f}")

    if targets is None:
        return

    # --- match, with the declarations' own sampling margin carried through ---
    # The simulated curves are held fixed and the DECLARATIONS are resampled, because
    # that is where the uncertainty a reader cares about lives: the target statistics
    # rest on a few dozen declarations, while a curve point is an average over
    # independent populations and measurement rounds. Reported the same way as the
    # accuracy figures the match is against, so the two can be read together.
    stats = (("rho", True), ("within1", True), ("within2", True),
             ("mae", False), ("sd", False))
    boot = {k: [] for k, _ in stats}
    boot["headline"] = []
    censored = 0
    rng = np.random.default_rng(args.seed)
    for _ in range(args.bootstrap):
        draw = target_stats(decl[rng.integers(0, len(decl), len(decl))])
        got = {}
        for key, increasing in stats:
            x_hat, _ = match_x(xs, curves[key], draw[key], increasing)
            if x_hat is not None:
                boot[key].append(x_hat)
                got[key] = x_hat
        # The headline pools the three statistics the accuracy claim is stated in,
        # within a draw — they share the same rows, so pooling across draws instead
        # would count that correlation as independent evidence.
        named = [got[k] for k in ("rho", "within1", "within2") if k in got]
        if len(named) == 3:
            boot["headline"].append(float(np.mean(named)))
        else:
            censored += 1

    def _margin(key):
        v = boot[key]
        if len(v) < max(20, args.bootstrap // 4):
            return "  (too many draws off the swept range)"
        return f" ± {float(np.std(v)):.0f}"

    print(f"\nequivalent tournament games  [{args.compare} comparison"
          + (f", ± over {args.bootstrap} declaration bootstraps]" if args.bootstrap
             else "]"))
    print(f"{'statistic':>12}{'onlymove':>11}{'games':>10}")
    for key, increasing in stats:
        x_hat, rel = match_x(xs, curves[key], targets[key], increasing)
        shown = f"{x_hat:.0f}" if x_hat is not None else \
            f"{rel}{xs[-1] if rel == '>' else xs[0]}"
        fmt = f"{targets[key]:.3f}" if key == "rho" else (
            f"{100*targets[key]:.0f}%" if key.startswith("within")
            else f"{targets[key]:.2f}")
        print(f"{key:>12}{fmt:>11}{shown:>10}"
              + (_margin(key) if args.bootstrap and x_hat is not None else ""))

    named = [match_x(xs, curves[k], targets[k], True)[0]
             for k in ("rho", "within1", "within2")]
    if args.bootstrap and boot["headline"] and all(p is not None for p in named):
        v = np.array(boot["headline"])
        point = float(np.mean(named))
        print(f"\n  headline (rho + within1 + within2 pooled): "
              f"{point:.0f} ± {float(v.std()):.0f} games"
              f"   [bootstrap mean {float(v.mean()):.1f}, "
              f"{np.percentile(v, 2.5):.0f}–{np.percentile(v, 97.5):.0f} at 95%]")
        if censored:
            print(f"  {censored} of {args.bootstrap} draws dropped: at least one "
                  f"statistic matched outside the swept range.")
    print("\n  '>N' / '<N' mean the match lies outside the swept range.")


if __name__ == "__main__":
    main()
