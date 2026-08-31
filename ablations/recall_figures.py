#!/usr/bin/env python3
"""The recall scheduler's measured figures — what the methodology quotes about
retention, the first-success step, and the clean-chain filter behind it.

Drives `ablations.eval_recall_model` on its production defaults (the shipped
retention constants, the shipped per-problem triple, the local attempt log) and
reads the statistics its checks return:

  recall.clean_chain_excluded_pct   share of repeat attempts the clean-chain
                                    filter removes, i.e. the ones sitting in
                                    chains that begin before repeats were
                                    reliably logged. The price of insisting an
                                    elapsed gap really is the whole time since
                                    the last exposure.
  recall.rung0_clear_low/_high      the extremes of the post-lapse clear rate
                                    across elapsed-gap bins from half a day to a
                                    fortnight — the flatness claim's two ends.
  recall.rung0_gap_trend_points     clear-rate points per DOUBLING of the gap
                                    over those same bins; positive means the
                                    rate drifts UP with elapsed time, which is
                                    the opposite of decay.
  recall.rung0_clear_rate           the pooled post-lapse clear rate, against
                                    the 0.80 window-open ceiling.
  recall.recalibration_slope        held-out Platt slope on the retention
                                    model's own log-odds, and
  recall.recalibration_overspread   its reciprocal: how many times too wide the
                                    model's spread is.
  recall.step_form_gain_nats        held-out nats a first-success step buys,
  recall.free_rung_gain_nats        ... and a free effect per rung (the upper
                                    bound on any reshaping of the streak term),
  recall.constant_deficit_nats      ... against the deficit to a constant rate.
  recall.pooled_or_min{5,10,20}     the within-user post-lapse/post-success odds
                                    ratio at three minimum-events-per-cell cuts,
  recall.direction_share_min{5,20}_pct  the share of users showing it in the
                                    expected direction at the loosest and
                                    tightest of those cuts,
  recall.cochran_q / _df            Cochran's heterogeneity statistic and its
                                    degrees of freedom at the loosest cut.
  recall.no_post_success_users_pct  users with no post-success rows at all
  recall.post_success_rows_median   ... and the median count, both inside the
                                    window the on-device model trains on.
  recall.rung_step_logodds_rise     the cold-solve log-odds' rise from the
                                    post-lapse to the post-success state,
  recall.rung_step_model_visible_stability  what the difficulty term already
                                    turns that into, and
  recall.rung_step_model_visible_share      that as a share of the interval
                                    correction the realized rates require.
  recall.mh_step_or_sd / _plateau_or_sd  separation of the first step and of the
                                    one above it, in sd.
  recall.rung0_composition_points   post-lapse clear rate among users who climb
                                    minus among users who never do; negative
                                    means composition suppresses the step
                                    rather than creating it.

    ./.venv/bin/python3 -m ablations.recall_figures
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations import eval_recall_model as recall  # noqa: E402
from ablations._figures import Figure, require  # noqa: E402

TIER = "about"

# The elapsed-gap span the flatness claim is stated over: half a day out to a
# fortnight. Bins are read whole, so the upper edge is the bin boundary past
# two weeks rather than 14 exactly.
GAP_LOW_DAYS, GAP_HIGH_DAYS = 0.5, 16.0

# The minimum-events-per-cell cuts the per-user step is estimated at. These are
# the script's own thresholds; a change there must reach the keys below.
CUTS = (5, 10, 20)


def measure():
    stats = recall.main([])
    require(isinstance(stats, dict), "eval_recall_model returned no statistics")

    # --- the clean-chain filter (methodology 1.13) ---------------------------
    totals = stats["totals"]
    n_repeat = totals.get("repeat attempts", 0)
    n_pre = totals.get("repeat attempts in pre-cutoff chains", 0)
    require(n_repeat > 0, "the attempt log holds no repeat attempts")
    require(0 < n_pre < n_repeat,
            f"clean-chain exclusion is degenerate: {n_pre} of {n_repeat} repeats")

    # --- the post-lapse rate against elapsed time ---------------------------
    decay = stats["decay"]
    require(0 in decay, "no post-lapse (rung 0) decay table — the log is too thin")
    span = [b for b in decay[0]
            if b["lo"] >= GAP_LOW_DAYS and b["hi"] <= GAP_HIGH_DAYS]
    require(len(span) >= 4,
            f"only {len(span)} gap bins between {GAP_LOW_DAYS}d and {GAP_HIGH_DAYS}d "
            "— the range the flatness claim is stated over is not covered")
    trend = recall.gap_trend(span)
    require(trend is not None, "no elapsed-gap trend over the post-lapse bins")

    correction = stats["correction"]
    require(correction is not None, "the interval correction did not run")
    for field in ("rung0_rate", "dz_rung0_to_1", "model_visible_stability"):
        require(field in correction, f"the interval correction reports no {field}")
    require(1 in correction["parity"],
            "no rung-1 interval multiplier — the post-success cell is too thin")
    parity1 = correction["parity"][1]
    require(parity1 > 1.0,
            f"the post-success rung needs no stretching (multiplier {parity1:.2f}) "
            "— the step the methodology describes is not present")
    visible_share = (correction["model_visible_stability"] - 1.0) / (parity1 - 1.0)

    # --- held-out value of the retention computation ------------------------
    heldout = stats["heldout"]
    require(heldout is not None, "the held-out split did not run")
    slope = heldout["platt_slope"]
    require(0.0 < slope < 1.0,
            f"recalibration slope {slope:.3f} is not inside (0, 1) — the model's "
            "spread is not merely too wide, which is what the text claims")

    forms = stats["growth_forms"]
    require(forms is not None, "the streak-shape comparison did not run")
    for form in ("linear", "step", "free", "constant"):
        require(form in forms, f"the streak-shape comparison has no {form!r} arm")

    # --- the within-user step and its consistency ---------------------------
    within = stats["within_user"]
    for cell in ("mh_1_vs_0", "mh_2_vs_1"):
        require(cell in within, f"the within-user comparison has no {cell} cell")
    require("rung0_climbers" in within and "rung0_non_climbers" in within,
            "the within-user comparison reports no composition split")

    step = stats["per_user_step"]
    require(step is not None, "the per-user step decomposition did not run")
    for cut in CUTS:
        require(cut in step["cuts"],
                f"the per-user step was not estimated at a {cut}-events-per-cell cut")
    loosest = step["cuts"][CUTS[0]]
    require(loosest["df"] > 0, "Cochran's statistic has no degrees of freedom")

    return {
        "clean_excluded": n_pre / n_repeat,
        "repeat_attempts": n_repeat,
        "gap_bins": span,
        "rung0_low": min(b["rate"] for b in span),
        "rung0_high": max(b["rate"] for b in span),
        "gap_trend_points": trend[0],
        "gap_trend_sd": trend[1],
        "rung0_rate": correction["rung0_rate"],
        "parity1": parity1,
        "dz": correction["dz_rung0_to_1"],
        "visible_stability": correction["model_visible_stability"],
        "visible_share": visible_share,
        "slope": slope,
        "overspread": 1.0 / slope,
        "step_gain": forms["linear"] - forms["step"],
        "free_gain": forms["linear"] - forms["free"],
        "constant_deficit": forms["linear"] - forms["constant"],
        "mh_step": within["mh_1_vs_0"],
        "mh_plateau": within["mh_2_vs_1"],
        "composition_points": 100.0 * (within["rung0_climbers"]
                                       - within["rung0_non_climbers"]),
        "cuts": step["cuts"],
        "density": step["density"],
    }


def figures():
    m = measure()
    cuts, density = m["cuts"], m["density"]
    figs = [
        Figure("recall.clean_chain_excluded_pct", m["clean_excluded"], "%.0f",
               "share of repeat attempts excluded by the clean-chain filter "
               "(chains beginning before repeats were reliably logged)"),
        Figure("recall.rung0_clear_low", m["rung0_low"], "%.0f",
               f"lowest post-lapse clear rate across elapsed-gap bins from "
               f"{GAP_LOW_DAYS:g} to {GAP_HIGH_DAYS:g} days"),
        Figure("recall.rung0_clear_high", m["rung0_high"], "%.0f",
               f"highest post-lapse clear rate across those same bins"),
        Figure("recall.rung0_gap_trend_points", m["gap_trend_points"], ".1f",
               "post-lapse clear-rate points per doubling of the elapsed gap "
               "over those bins; positive is a rise, not decay"),
        Figure("recall.rung0_clear_rate", m["rung0_rate"], "%.0f",
               "pooled post-lapse clear rate, against the 0.80 window-open ceiling"),
        Figure("recall.recalibration_slope", m["slope"], ".2f",
               "held-out Platt slope on the retention model's own log-odds, "
               "split by user"),
        Figure("recall.recalibration_overspread", m["overspread"], ".1f",
               "reciprocal of that slope: how many times too wide the model's "
               "spread is"),
        Figure("recall.step_form_gain_nats", m["step_gain"], ".3f",
               "held-out nats a first-success step buys over the shipped linear "
               "streak term"),
        Figure("recall.free_rung_gain_nats", m["free_gain"], ".3f",
               "held-out nats a free effect per rung buys — the upper bound on "
               "any reshaping of the streak term"),
        Figure("recall.constant_deficit_nats", m["constant_deficit"], ".2f",
               "held-out nats the retention computation trails a constant rate by"),
        Figure("recall.mh_step_or_sd", m["mh_step"][1], ".1f",
               "separation of the post-success from the post-lapse state, in sd, "
               "within user"),
        Figure("recall.mh_plateau_or_sd", m["mh_plateau"][1], ".1f",
               "separation of the second success from the first, in sd, within user"),
        Figure("recall.rung0_composition_points", m["composition_points"], ".1f",
               "post-lapse clear-rate points, users who reach the next rung minus "
               "users who never do; negative means composition suppresses the step"),
        Figure("recall.cochran_q", cuts[CUTS[0]]["q"], ".0f",
               f"Cochran's heterogeneity statistic for the per-user step at the "
               f"{CUTS[0]}-events-per-cell cut"),
        Figure("recall.cochran_df", cuts[CUTS[0]]["df"], "d",
               "its degrees of freedom"),
        Figure("recall.no_post_success_users_pct", density["none_share"], "%.0f",
               "users with no post-success rows inside the window the on-device "
               "model trains on"),
        Figure("recall.post_success_rows_median", density["median"], "d",
               "median post-success rows per user inside that same window"),
        Figure("recall.rung_step_logodds_rise", m["dz"], ".2f",
               "rise in mean cold-solve log-odds from the post-lapse to the "
               "post-success state"),
        Figure("recall.rung_step_model_visible_stability", m["visible_stability"],
               ".2f",
               "stability multiplier the difficulty term already draws from that rise"),
        Figure("recall.rung_step_model_visible_share", m["visible_share"], ".2f",
               "that multiplier as a share of the interval correction the realized "
               "rates require at the first success"),
    ]
    for cut in CUTS:
        figs.append(Figure(f"recall.pooled_or_min{cut}", cuts[cut]["pooled_or"], ".2f",
                           f"pooled within-user post-lapse/post-success odds ratio, "
                           f"users with at least {cut} events in each cell"))
    for cut in (CUTS[0], CUTS[-1]):
        figs.append(Figure(f"recall.direction_share_min{cut}_pct",
                           cuts[cut]["direction_share"], "%.0f",
                           f"share of those users whose own odds ratio runs in the "
                           f"expected direction, at the {cut}-events cut"))
    return figs


def main():
    m = measure()
    print("\n" + "=" * 78)
    print("RECALL FIGURES")
    print("=" * 78)
    print(f"clean-chain filter          : {100 * m['clean_excluded']:.1f}% of "
          f"{m['repeat_attempts']:,} repeat attempts excluded")
    print(f"post-lapse clear rate       : {100 * m['rung0_rate']:.1f}% pooled "
          f"(ceiling {100 * recall.WINDOW_OPEN_R:.0f}%)")
    print(f"  across gap bins {GAP_LOW_DAYS:g}-{GAP_HIGH_DAYS:g}d  : "
          f"{100 * m['rung0_low']:.1f}% to {100 * m['rung0_high']:.1f}%   "
          f"trend {m['gap_trend_points']:+.2f} points per doubling "
          f"({m['gap_trend_sd']:.1f} sd)")
    for b in m["gap_bins"]:
        print(f"    {b['lo']:>5g}-{b['hi']:<5g}d  n={b['n']:>5}   {100 * b['rate']:5.1f}%")
    print(f"held-out recalibration      : slope {m['slope']:.3f} "
          f"({m['overspread']:.1f}x too wide)")
    print(f"streak-shape gains (nats)   : step {m['step_gain']:+.4f}   "
          f"free per rung {m['free_gain']:+.4f}   "
          f"deficit to a constant {m['constant_deficit']:.4f}")
    print(f"within-user separation (sd) : first step {m['mh_step'][1]:.1f} "
          f"(OR {m['mh_step'][0]:.2f})   next step {m['mh_plateau'][1]:.1f} "
          f"(OR {m['mh_plateau'][0]:.2f})")
    print(f"composition on the step     : {m['composition_points']:+.1f} points "
          f"(climbers minus non-climbers, post-lapse)")
    print("per-user step, by minimum events per cell:")
    for cut in CUTS:
        c = m["cuts"][cut]
        print(f"    >= {cut:>2}   {c['users']:>4} users   pooled OR {c['pooled_or']:.2f}   "
              f"in-direction {100 * c['direction_share']:.0f}%   "
              f"Q={c['q']:.0f} on {c['df']} df   I^2={100 * c['i2']:.0f}%")
    d = m["density"]
    print(f"post-success rows per user  : {100 * d['none_share']:.0f}% have none, "
          f"median {d['median']:.0f}, {100 * d['twenty_plus_share']:.0f}% have 20+")
    print(f"first-step selection        : cold-solve log-odds {m['dz']:+.2f} -> "
          f"x{m['visible_stability']:.2f} of stability, "
          f"{100 * m['visible_share']:.0f}% of the x{m['parity1']:.2f} the realized "
          f"rates require")


if __name__ == "__main__":
    main()
