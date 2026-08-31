#!/usr/bin/env python3
"""How the shown rank compares with what players declare elsewhere — the figures
the About page quotes, measured from the SHIPPED mapping.

Runs `check_mapped_vs_declared_rank` (each surveyed player's 80%-clear rating,
time-adjusted, through the shipped curve and their server's coupling, against the
rank they declared there) and reads its per-declaration table, then:

  rank.rho            per-server Spearman between the mapped rating and the
                      declared rank, n-weighted over servers — the survey-rho
                      definition the rating validation uses, so this is the same
                      statistic as the benchmark's, read off the shipped fit
                      rather than a cross-fitted fold. Self-declarations only.
  rank.rho_se         its standard error under a bootstrap over PLAYERS (a
                      player's declarations on several servers are not
                      independent draws).
  rank.within1_pct    share of self-declarations whose shown server rank lands
  rank.within2_pct    within one / two ranks of the declared one, in that
                      server's own units (`err_server`; `err_internal` is not
                      quotable — see CLAUDE.md).
  rank.within_se_pts  the larger of their two bootstrap standard errors, in
                      percentage points.
  rank.games_*        `eval_equivalent_tournament_games` matched on the EGF
                      ladder — every declaration in EGF units, self-reported
                      plus the machine-verified EGD heartbeats, which share one
                      coupling: how many tournament games two independent
                      ratings would need before they agree as closely as the
                      shown rank agrees with a declared one. Restricting to the
                      verified rows alone is the like-for-like comparison, its
                      other side carrying no declaration error, but two dozen of
                      them cannot pin the match: it lands inside its own margin
                      and a large share of bootstrap draws falls off the swept
                      range entirely.

Every input is checked on the way: the table has the columns and the row kinds
this reads, enough servers clear the per-server minimum, the bootstrap is
well-formed, and every tournament statistic matched inside the swept range.

    ./.venv/bin/python3 -m ablations.rank_calibration_figures
"""

import os
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations import check_mapped_vs_declared_rank as calibration  # noqa: E402
from ablations import eval_equivalent_tournament_games as tournament  # noqa: E402
from ablations._figures import Figure, require  # noqa: E402
from utils.rating.rank_surveys import (  # noqa: E402
    SURVEY_RHO_MIN_SERVER, server_weighted_spearman)

TIER = "about"

BOOT = 2000
SEED = 0
# The population the games figure is matched on: the EGF ladder, every declaration
# in EGF units (self-reported plus the verified EGD heartbeats, one coupling).
GAMES_SCOPE = "EGF"
GAMES_KIND = "all"

_COLUMNS = ("prefix", "kind", "server", "system", "our_internal_adj",
            "declared_internal", "err_server")


def _pairs_by_server(df):
    by = {}
    for r in df.itertuples(index=False):
        by.setdefault(r.server, []).append((float(r.our_internal_adj), float(r.declared_internal)))
    return by


def _within(df):
    e = np.abs(df["err_server"].to_numpy(float))
    return float(np.mean(e <= 1.0)), float(np.mean(e <= 2.0))


def measure():
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "declarations.csv")
        calibration.main(["--csv", csv_path])
        require(os.path.exists(csv_path), "check_mapped_vs_declared_rank wrote no CSV")
        df = pd.read_csv(csv_path)
    missing = [c for c in _COLUMNS if c not in df.columns]
    require(not missing, f"declaration table lacks columns {missing}")
    require(set(df["kind"].unique()) <= {"survey", "verified"},
            f"unexpected row kinds {sorted(df['kind'].unique())}")
    surv = df[df["kind"] == "survey"].reset_index(drop=True)
    require(len(surv) > 0, "no self-declared rows")
    require(np.all(np.isfinite(surv[["our_internal_adj", "declared_internal", "err_server"]].to_numpy(float))),
            "non-finite entries in the self-declared rows")

    rho, per_server = server_weighted_spearman(_pairs_by_server(surv), SURVEY_RHO_MIN_SERVER)
    require(np.isfinite(rho) and len(per_server) >= 2,
            f"survey rho needs at least two servers with >= {SURVEY_RHO_MIN_SERVER} declarations")
    within1, within2 = _within(surv)

    # Bootstrap over players: resample users with replacement, take all of each
    # drawn user's declarations. Fixed seed, so a re-run reproduces the margins.
    users = surv["prefix"].unique()
    rows_by_user = {u: surv.index[surv["prefix"] == u].to_numpy() for u in users}
    rng = np.random.default_rng(SEED)
    boot = np.empty((BOOT, 3))
    for b in range(BOOT):
        drawn = rng.choice(users, size=len(users), replace=True)
        idx = np.concatenate([rows_by_user[u] for u in drawn])
        sample = surv.iloc[idx]
        r, _ = server_weighted_spearman(_pairs_by_server(sample), SURVEY_RHO_MIN_SERVER)
        boot[b, 0] = r
        boot[b, 1:] = _within(sample)
    require(np.all(np.isfinite(boot)), "a bootstrap draw produced a non-finite statistic")
    se = boot.std(axis=0, ddof=1)

    scoped = df[df["system"] == GAMES_SCOPE] if GAMES_KIND == "all" else \
        df[(df["kind"] == GAMES_KIND) & (df["system"] == GAMES_SCOPE)]
    require(len(scoped) > 0, f"no {GAMES_KIND} {GAMES_SCOPE} rows to match the games figure on")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "declarations.csv")
        df.to_csv(csv_path, index=False)
        games = tournament.main(["--targets", csv_path, "--scope", GAMES_SCOPE,
                                 "--kind", GAMES_KIND, "--seed", str(SEED)])
    require(games is not None and games.get("headline") is not None,
            "tournament match: at least one statistic fell outside the swept range")
    require(games["n"] == len(scoped), "tournament matched a different row set")

    return {
        "n_declarations": int(len(surv)),
        "n_players": int(len(users)),
        "servers": {s: n for s, (n, _) in per_server.items()},
        "rho": float(rho), "rho_se": float(se[0]),
        "within1": within1, "within2": within2,
        "within_se": float(max(se[1], se[2])),
        "games": float(games["headline"]), "games_sd": float(games["headline_sd"]),
        "games_n": int(games["n"]), "games_censored": int(games["censored"]),
    }


def figures():
    m = measure()
    return [
        Figure("rank.rho", m["rho"], ".2f",
               "per-server Spearman(mapped rating, declared rank), n-weighted, self-declarations"),
        Figure("rank.rho_se", m["rho_se"], ".2f", "bootstrap SE of rank.rho over players"),
        Figure("rank.n_declarations", m["n_declarations"], "~10", "self-declared (player, server) ranks"),
        Figure("rank.n_players", m["n_players"], "~10", "players behind rank.n_declarations"),
        Figure("rank.within1_pct", m["within1"], "%.0f",
               "self-declarations with the shown server rank within one rank of the declared"),
        Figure("rank.within2_pct", m["within2"], "%.0f", "…within two ranks"),
        Figure("rank.within_se_pts", m["within_se"], "%.0f",
               "larger bootstrap SE of the two within-shares, percentage points"),
        Figure("rank.games_equivalent", m["games"], ".0f",
               "tournament games per player at which two independent ratings agree as well as "
               "the shown rank agrees with a declared EGF rank"),
        Figure("rank.games_equivalent_sd", m["games_sd"], ".0f",
               "its bootstrap SD over the declarations behind the target"),
        Figure("rank.games_n", m["games_n"], "d", "EGF declarations the match rests on"),
    ]


def main():
    m = measure()
    print("\n" + "=" * 78)
    print(f"self-declarations: {m['n_declarations']} from {m['n_players']} players; "
          f"per-server rho over {', '.join(f'{s} ({n})' for s, n in sorted(m['servers'].items()))}")
    print(f"{'rho':>14}{m['rho']:>8.3f}  ± {m['rho_se']:.3f}  (bootstrap over players)")
    print(f"{'within 1 rank':>14}{100 * m['within1']:>7.1f}%")
    print(f"{'within 2 ranks':>14}{100 * m['within2']:>7.1f}%  (both ± {100 * m['within_se']:.1f} pts)")
    print(f"{'games':>14}{m['games']:>8.1f}  ± {m['games_sd']:.1f}  "
          f"({GAMES_KIND} {GAMES_SCOPE}, n={m['games_n']}, "
          f"{m['games_censored']} bootstrap draws off the swept range)")


if __name__ == "__main__":
    main()
