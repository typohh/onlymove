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
  rank.stone_value_*  the logit gap one band buys, averaged over the middle kyu
                      ranks and over the dan ranks, out of the same joint fit
                      the games figure's win-probability law comes from. It is
                      what makes an answer in games a statement about a cohort:
                      a dan game is several times as informative as a kyu one.

DECLARATION HONESTY. The same run carries both routes to the one question
anchoring on declarations raises — whether a self-report is displaced from what
the player's rating actually is:

  rank.honesty_*      the rank-matched contrast. A verified user is differenced
                      against self-reporters the fit places at the same rank, so
                      the fit's own residual and any bias it absorbed cancel and
                      the displacement is what remains. `_lo`/`_hi` bracket the
                      estimate over BOTH routes; the `_se`/`_ci_hi` pair is the
                      widest the matched route's own uncertainty reaches.
  rank.exact_match_*  its exact-match limit: the users holding both a
                      self-report and a verified rating on the same ladder, with
                      our estimate out of the comparison entirely. An upper
                      bound on honesty rather than an estimate of it, since
                      these are exactly the players whose declaration is
                      publicly checkable — hence also the `rank.route_gap`
                      between the two.
  rank.spread_*       precision rather than level, over the verified rows' own
  rank.declaration_*  fitted-rank range: the residual's SPREAD on self-reports
                      against its spread on verified ratings, and the
                      declaration noise that difference implies.
  rank.server_*       the per-server residual means, which are the natural thing
                      to reach for here and cannot answer the question at all —
                      a coupling's thresholds are free in location, so a bias
                      shared by a server's declarers leaves no residual behind.
                      Emitted so the claim that none of them is even nominally
                      distinguishable from zero is checked rather than assumed.
  rank.deep_kyu_*     what bounds the contrast: the linkable cohort's median
  rank.*_median_rank  fitted rank against the declaring population's, and the
                      mapping's own residual in the deep-kyu bands the contrast
                      therefore barely reaches.

Every input is checked on the way: the table has the columns and the row kinds
this reads, enough servers clear the per-server minimum, the bootstrap is
well-formed, both honesty routes formed, and every tournament statistic matched
inside the swept range.

    ./.venv/bin/python3 -m ablations.rank_calibration_figures
"""

import os
import shutil
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations import check_mapped_vs_declared_rank as calibration  # noqa: E402
from ablations import eval_equivalent_tournament_games as tournament  # noqa: E402
from ablations._figures import Figure, require  # noqa: E402
from pipeline import map_rating_to_rank as mapping  # noqa: E402
from utils.rating.rank_surveys import (  # noqa: E402
    SURVEY_RHO_MIN_SERVER, server_weighted_spearman, value_to_rank)
from utils.rating.ratings_io import load_declaration_exclusions  # noqa: E402

TIER = "about"

BOOT = 2000
SEED = 0
# The population the games figure is matched on: the EGF ladder, every declaration
# in EGF units (self-reported plus the verified EGD heartbeats, one coupling).
GAMES_SCOPE = "EGF"
GAMES_KIND = "all"

_COLUMNS = ("prefix", "kind", "server", "system", "our_internal_adj",
            "declared_internal", "err_server")

# The two rank bands the deep-kyu/dan residual comparison is quoted over, named
# exactly as the calibration script's own band table labels them.
DEEP_KYU_BAND = "≤15k"
DAN_BAND = "1d–4d"
# Which verified ladder the exact-match route runs on: the one where a player can
# hold both a self-report and a machine-written rating on the same scale.
EXACT_MATCH_SERVER = "EGD"


def _pairs_by_server(df):
    by = {}
    for r in df.itertuples(index=False):
        by.setdefault(r.server, []).append((float(r.our_internal_adj), float(r.declared_internal)))
    return by


def _within(df):
    e = np.abs(df["err_server"].to_numpy(float))
    return float(np.mean(e <= 1.0)), float(np.mean(e <= 2.0))


def _honesty(stats):
    """The two displacement routes, the precision comparison and what bounds them.

    Both routes estimate the same quantity — how far a self-report sits from what
    the player's rating says — and both are quoted together because they fail in
    opposite directions: the rank-matched contrast is representative but rests on
    however many verified ratings exist, while the exact-match limit is tight but
    drawn from the players whose declaration is publicly checkable and so is an
    upper bound on honesty rather than an estimate of it."""
    matched, exact_by_server = stats["matched"], stats["exact_match"]
    require(matched is not None, "no verified ratings: the rank-matched contrast did not form")
    cals = matched["calipers"]
    require(len(cals) == len(calibration.MATCH_CALIPERS),
            "a matching tolerance produced too few pairs to estimate")
    require(exact_by_server is not None and EXACT_MATCH_SERVER in exact_by_server,
            f"no {EXACT_MATCH_SERVER} rows paired with a self-report on the same ladder")
    exact = exact_by_server[EXACT_MATCH_SERVER]
    require(exact["n"] >= 3 and exact["sd"] > 0 and exact["widest_share"] is not None,
            "too few users hold both a self-report and a verified rating to read the limit")

    # The tolerance nearest one rank is the route's reference reading; the others
    # exist to show the answer does not depend on the tolerance.
    mid = min(cals, key=lambda c: abs(c["caliper"] - 1.0))
    est = [c["estimate"] for c in cals]
    spreads = stats["spreads"]
    require(spreads is not None, "too few verified rows to compare residual spreads")
    for k in ("verified", "survey_egf", "survey_pooled"):
        require(spreads[k] is not None and spreads[k] > 0,
                f"residual spread {k} is not estimable over the verified rank range")
    # Independence of the two error sources is what lets one spread be subtracted
    # from the other; it is an assumption, and the document says so.
    noise = {k: float(np.sqrt(max(spreads[k] ** 2 - spreads["verified"] ** 2, 0.0)))
             for k in ("survey_egf", "survey_pooled")}
    require(min(noise.values()) > 0,
            "self-reports are no wider than the verified rows — nothing to subtract")

    servers = stats["servers"]
    require(len(servers) >= 2, "too few servers carry a residual mean")
    band = stats["band"]
    for label in (DEEP_KYU_BAND, DAN_BAND):
        require(band.get(label) is not None, f"no declarations in the {label} band")
    require(band[DAN_BAND]["mae"] > 0, "the dan band has a zero residual")
    for who, value in stats["median_rank"].items():
        require(value is not None and np.isfinite(value), f"no {who} users to take a median rank over")

    return {
        "verified_users": matched["n_verified"],
        "selfreporters": matched["n_selfreport"],
        "matched_lo": min(est), "matched_hi": max(est),
        "matched_mid": mid["estimate"],
        "se": max(c["se"] for c in cals),
        "ci_hi": max(c["ci_hi"] for c in cals),
        "exact_n": exact["n"], "exact_mean": exact["mean"], "exact_sd": exact["sd"],
        "exact_in_band": exact["in_band"],
        "exact_beyond1": exact["n"] - exact["within1"],
        "exact_out_of_band": len(exact["misses"]),
        "exact_miss_narrowest": min(exact["misses"]) if exact["misses"] else 0.0,
        "exact_miss_widest": max(exact["misses"]) if exact["misses"] else 0.0,
        "exact_widest_share": exact["widest_share"],
        "route_gap": exact["mean"] - mid["estimate"],
        # both routes together — what the document brackets the displacement by
        "lo": min(min(est), exact["mean"]), "hi": max(max(est), exact["mean"]),
        "spread_verified": spreads["verified"],
        "spread_survey_egf": spreads["survey_egf"],
        "spread_survey_pooled": spreads["survey_pooled"],
        "noise_lo": noise["survey_egf"], "noise_hi": noise["survey_pooled"],
        "noise_ratio_lo": noise["survey_egf"] / exact["sd"],
        "noise_ratio_hi": noise["survey_pooled"] / exact["sd"],
        "servers_scored": len(servers),
        "server_max_t": max(abs(s["t"]) for s in servers),
        "server_worst": max(servers, key=lambda s: abs(s["t"]))["server"],
        "linkable_median_rank": stats["median_rank"]["verified"],
        "declaring_median_rank": stats["median_rank"]["survey"],
        "deep_kyu_mae": band[DEEP_KYU_BAND]["mae"],
        "dan_mae": band[DAN_BAND]["mae"],
        "deep_kyu_ratio": band[DEEP_KYU_BAND]["mae"] / band[DAN_BAND]["mae"],
    }


def declarations_only_stats():
    """The calibration statistics against a mapping fitted WITHOUT the verified
    ratings. The honesty contrast differences a verified rating's residual against
    self-reports' from one fitted prediction, which is only a contrast if the verified
    rating did not help make that prediction; the shipped mapping anchors on them, so
    the contrast runs against a declarations-only refit built here in a scratch copy
    of ratings/ (its own curve, couplings and exclusions), never against the shipped
    one."""
    with tempfile.TemporaryDirectory() as tmp:
        scratch = os.path.join(tmp, "ratings")
        shutil.copytree(os.path.join(PROJECT_ROOT, "ratings"), scratch)
        mapping.main(["--dir", scratch, "--no-verified"])
        stats = calibration.main(["--dir", scratch, "--csv", os.path.join(tmp, "d.csv")])
    require(isinstance(stats, dict), "declarations-only calibration returned no statistics")
    return stats


def measure():
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "declarations.csv")
        stats = calibration.main(["--csv", csv_path])
        require(isinstance(stats, dict), "check_mapped_vs_declared_rank returned no statistics")
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
    stone = games.get("stone") or {}
    for part in ("kyu", "dan"):
        require(stone.get(part) is not None and stone[part] > 0,
                f"the stone-value fit covers no {part} bands")
    require(stone["dan"] > stone["kyu"],
            "the stone-value curve does not rise up the ladder — check the link fit")

    return {
        "n_screened": len(load_declaration_exclusions(os.path.join(PROJECT_ROOT, "ratings"))),
        "honesty": _honesty(declarations_only_stats()),
        "stone_kyu": float(stone["kyu"]), "stone_dan": float(stone["dan"]),
        "stone_kyu_bands": stone["kyu_bands"], "stone_dan_bands": stone["dan_bands"],
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
        Figure("rank.declarations_screened", m["n_screened"], "d",
               "declarations the mapping screened out as junk before its final fit, "
               "and which nothing here scores"),
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
        Figure("rank.stone_value_kyu_logit", m["stone_kyu"], ".2f",
               f"logits one band buys, averaged over "
               f"{value_to_rank(m['stone_kyu_bands'][0] + 0.5)}–"
               f"{value_to_rank(m['stone_kyu_bands'][1] + 0.5)}, from the even-game table's "
               f"joint stone-value fit"),
        Figure("rank.stone_value_dan_logit", m["stone_dan"], ".2f",
               f"the same over "
               f"{value_to_rank(m['stone_dan_bands'][0] + 0.5)}–"
               f"{value_to_rank(m['stone_dan_bands'][1] + 0.5)}"),
    ] + _honesty_figures(m["honesty"])


def _honesty_figures(h):
    return [
        Figure("rank.honesty_verified", h["verified_users"], "d",
               "users with a machine-written federation rating, the contrast's binding count"),
        Figure("rank.honesty_selfreporters", h["selfreporters"], "d",
               "users they are matched against: self-reporters holding no verified rating"),
        Figure("rank.honesty_lo", h["lo"], "+.2f",
               "displacement of a self-report from a verified rating at the same fitted "
               "rank, lowest reading over both routes; + = self-reports claim stronger"),
        Figure("rank.honesty_hi", h["hi"], "+.2f", "…highest reading over both routes"),
        Figure("rank.honesty_matched_lo", h["matched_lo"], "+.2f",
               "the rank-matched route alone, lowest over the matching tolerances"),
        Figure("rank.honesty_matched_hi", h["matched_hi"], "+.2f", "…highest over them"),
        Figure("rank.honesty_se", h["se"], ".2f",
               "widest standard error the rank-matched route carries"),
        Figure("rank.honesty_ci_hi", h["ci_hi"], ".2f",
               "highest 95% upper bound it reaches — what the data licenses as an "
               "upper limit on inflation"),
        Figure("rank.exact_match_users", h["exact_n"], "d",
               "users holding both a self-report and a verified rating on that same ladder"),
        Figure("rank.exact_match_displacement", h["exact_mean"], "+.2f",
               "their mean self-report minus verified rating, in ranks"),
        Figure("rank.exact_match_spread", h["exact_sd"], ".2f", "its spread over those users"),
        Figure("rank.exact_match_in_band", h["exact_in_band"], "d",
               "of them declaring exactly the band their verified rating falls in"),
        Figure("rank.exact_match_out_of_band", h["exact_out_of_band"], "d",
               "declaring a neighbouring band instead"),
        Figure("rank.exact_match_beyond1", h["exact_beyond1"], "d",
               "of them sitting more than a rank from their verified rating"),
        Figure("rank.exact_match_miss_narrowest", h["exact_miss_narrowest"], ".2f",
               "narrowest miss among the out-of-band ones, as a distance to the "
               "declared band's edge"),
        Figure("rank.exact_match_miss_widest", h["exact_miss_widest"], ".2f", "…widest"),
        Figure("rank.exact_match_widest_share", h["exact_widest_share"], "%.0f",
               "share of the exact-match mean the two widest-missing rows carry between them"),
        Figure("rank.route_gap", h["route_gap"], ".2f",
               "exact-match reading minus the rank-matched one at the one-rank tolerance: "
               "what being publicly checkable is worth"),
        Figure("rank.spread_verified", h["spread_verified"], ".1f",
               "spread of the residual against verified ratings, over their own fitted-rank range"),
        Figure("rank.spread_survey_egf", h["spread_survey_egf"], ".1f",
               "…against self-reports on the federation coupling, same range"),
        Figure("rank.spread_survey_pooled", h["spread_survey_pooled"], ".1f",
               "…against self-reports pooled over every server, same range"),
        Figure("rank.declaration_noise_lo", h["noise_lo"], ".1f",
               "declaration noise the federation spread implies once the verified spread "
               "is subtracted in quadrature"),
        Figure("rank.declaration_noise_hi", h["noise_hi"], ".1f", "…from the pooled spread"),
        Figure("rank.declaration_noise_ratio_lo", h["noise_ratio_lo"], ".1f",
               "that noise against the spread the publicly-checkable users show"),
        Figure("rank.declaration_noise_ratio_hi", h["noise_ratio_hi"], ".1f", "…at the pooled end"),
        Figure("rank.servers_scored", h["servers_scored"], "d",
               "servers carrying enough declarations for a residual mean and its error"),
        Figure("rank.server_mean_max_t", h["server_max_t"], ".2f",
               "largest per-server residual mean in units of its own standard error — "
               "nominal significance is 1.96, before any correction for looking at all of them"),
        Figure("rank.linkable_median_rank", h["linkable_median_rank"], "rank",
               "median fitted rank of the users holding a verified rating"),
        Figure("rank.declaring_median_rank", h["declaring_median_rank"], "rank",
               "…of everyone who declared a rank anywhere"),
        Figure("rank.deep_kyu_mae", h["deep_kyu_mae"], ".1f",
               "mean absolute residual against declarations at 15 kyu and weaker, in ranks"),
        Figure("rank.dan_mae", h["dan_mae"], ".1f", "…through the dan range"),
        Figure("rank.deep_kyu_mae_ratio", h["deep_kyu_ratio"], ".1f",
               "the first over the second"),
    ]


def main():
    m = measure()
    print("\n" + "=" * 78)
    print(f"{m['n_screened']} declarations screened out by the mapping")
    print(f"self-declarations: {m['n_declarations']} from {m['n_players']} players; "
          f"per-server rho over {', '.join(f'{s} ({n})' for s, n in sorted(m['servers'].items()))}")
    print(f"{'rho':>14}{m['rho']:>8.3f}  ± {m['rho_se']:.3f}  (bootstrap over players)")
    print(f"{'within 1 rank':>14}{100 * m['within1']:>7.1f}%")
    print(f"{'within 2 ranks':>14}{100 * m['within2']:>7.1f}%  (both ± {100 * m['within_se']:.1f} pts)")
    print(f"{'games':>14}{m['games']:>8.1f}  ± {m['games_sd']:.1f}  "
          f"({GAMES_KIND} {GAMES_SCOPE}, n={m['games_n']}, "
          f"{m['games_censored']} bootstrap draws off the swept range)")
    print(f"{'stone value':>14}{m['stone_kyu']:>8.2f}  logits/band over "
          f"{value_to_rank(m['stone_kyu_bands'][0] + 0.5)}–"
          f"{value_to_rank(m['stone_kyu_bands'][1] + 0.5)}, "
          f"{m['stone_dan']:.2f} over "
          f"{value_to_rank(m['stone_dan_bands'][0] + 0.5)}–"
          f"{value_to_rank(m['stone_dan_bands'][1] + 0.5)}")

    h = m["honesty"]
    print("\ndeclaration honesty  (+ = a self-report claims more strength than a "
          "verified rating at the same fitted rank)")
    print(f"  rank-matched   {h['matched_lo']:+.2f} to {h['matched_hi']:+.2f} ranks over the "
          f"matching tolerances, se ≤ {h['se']:.2f}, 95% upper bound {h['ci_hi']:.2f}"
          f"  [{h['verified_users']} verified vs {h['selfreporters']} self-reporters]")
    print(f"  exact match    {h['exact_mean']:+.2f} ± {h['exact_sd']:.2f} ranks over "
          f"{h['exact_n']} users holding both; {h['exact_in_band']} declare their verified "
          f"rating's own band, {h['exact_out_of_band']} miss by "
          f"{h['exact_miss_narrowest']:.2f}–{h['exact_miss_widest']:.2f} and carry "
          f"{100 * h['exact_widest_share']:.0f}% of the mean; {h['exact_beyond1']} sit "
          f"over a rank out")
    print(f"  route gap      {h['route_gap']:+.2f} ranks — publicly-checkable users "
          f"against everyone else")
    print(f"  spread         self-reports {h['spread_survey_egf']:.2f} on the federation "
          f"coupling, {h['spread_survey_pooled']:.2f} pooled, against "
          f"{h['spread_verified']:.2f} verified → {h['noise_lo']:.2f}–{h['noise_hi']:.2f} "
          f"ranks of declaration noise, {h['noise_ratio_lo']:.1f}–{h['noise_ratio_hi']:.1f}× "
          f"the exact-match spread")
    print(f"  per-server     largest residual mean {h['server_max_t']:.2f} standard errors "
          f"from zero ({h['server_worst']}) over {h['servers_scored']} servers; "
          f"nominal significance is 1.96")
    print(f"  what bounds it linkable cohort median "
          f"{value_to_rank(h['linkable_median_rank'])} against the declaring population's "
          f"{value_to_rank(h['declaring_median_rank'])}; residual "
          f"{h['deep_kyu_mae']:.2f} ranks at 15k and weaker against {h['dan_mae']:.2f} "
          f"through the dans ({h['deep_kyu_ratio']:.1f}×)")


if __name__ == "__main__":
    main()
