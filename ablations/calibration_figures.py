#!/usr/bin/env python3
"""The shipped response curve's levels, as the held-aside rows fitted them.

Reads `ratings/rating_constants.json` — the `calibration` object the rating fit
writes after training. A share of the attempt rows never trains the fit; on
them, with every position frozen, five numbers are fitted: one guessing floor,
the lapse, a multiplier and an offset on the slope term, and a multiplier on
each player's ability uncertainty. The floor and the lapse are what ship; the
multiplier and the offset are folded into the shipped slopes and ratings.

What this reports is that record and nothing recomputed from it beyond
differences: the shipped levels beside the ones the fit trained and beside the
ones a curve that ignores ability uncertainty reads (the "plain" curve), the
held-row likelihood each of the three scores, and how the shipped curve's two
ends compare with what the held rows did there.

The read checks what every consumer assumes: the lapse a reader is handed is the
calibrated one, no multiplier ships beside the curve, every problem in
`ratings/<size>/tsumego_floors.csv` carries the one calibrated floor, and the
three held losses are nested the way the three curves are — the plain curve
contains the trained one and the shipped curve contains the plain one, so
neither step can score worse. The held share is the held rows over the rows in
`ratings/user_theta.csv`, which counts every row a real user contributed, held
or not.

    ./.venv/bin/python3 -m ablations.calibration_figures
"""

import csv
import json
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from utils.constants import ALL_BOARD_SIZES  # noqa: E402
from utils.rating.model_constants import CALIBRATION_P  # noqa: E402

TIER = "methodology"

RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")
RATING_CONSTANTS = os.path.join(RATINGS_DIR, "rating_constants.json")
USER_THETA = os.path.join(RATINGS_DIR, "user_theta.csv")

SCALARS = ("slope_mult", "offset", "floor", "lapse", "kappa", "held_loss_plain",
           "held_loss", "plain_floor", "plain_lapse", "ability_sigma",
           "held_loss_as_trained")
PAIRS = ("top10_fail", "bottom5_success")
# The floors file is float text of a float32 column; the lapse is rounded to
# four places on the way into `fat_finger_rate`.
FLOOR_TOL = 1e-5
LAPSE_TOL = 5.1e-5
# A nested curve cannot score worse than the one it contains, up to the
# optimizer's own tolerance.
LOSS_TOL = 1e-6


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def measure():
    require(os.path.exists(RATING_CONSTANTS),
            f"no {RATING_CONSTANTS} — run pipeline.train_mirt")
    with open(RATING_CONSTANTS) as f:
        consts = json.load(f)
    cal = consts.get("calibration")
    require(isinstance(cal, dict),
            f"{RATING_CONSTANTS} carries no calibration — it was written by a "
            f"fit that did not calibrate its curve on held rows")
    for k in SCALARS:
        require(_number(cal.get(k)), f"calibration.{k} is not a finite number: {cal.get(k)!r}")
    for k in PAIRS:
        v = cal.get(k)
        require(isinstance(v, list) and len(v) == 2 and all(_number(x) and 0.0 <= x <= 1.0 for x in v),
                f"calibration.{k} is not an [actual, predicted] pair of rates: {v!r}")
    require(isinstance(cal.get("held_rows"), int) and cal["held_rows"] > 0,
            f"calibration.held_rows is not a positive count: {cal.get('held_rows')!r}")
    trained = cal.get("as_trained")
    require(isinstance(trained, dict) and all(_number(trained.get(k)) for k in ("floor", "lapse")),
            f"calibration.as_trained is not a (floor, lapse) pair: {trained!r}")

    # The levels are a curve's levels.
    for name, c, lam in (("shipped", cal["floor"], cal["lapse"]),
                         ("plain", cal["plain_floor"], cal["plain_lapse"]),
                         ("trained", trained["floor"], trained["lapse"])):
        require(0.0 <= c < 1.0 and 0.0 <= lam < 0.5 and c + lam < 1.0,
                f"the {name} levels are not a response curve's: floor {c}, lapse {lam}")
    require(cal["floor"] < CALIBRATION_P,
            f"the shipped floor {cal['floor']} is not below the calibration "
            f"probability {CALIBRATION_P}, so no rating is defined")
    require(cal["slope_mult"] > 0.0, f"slope multiplier {cal['slope_mult']} is not positive")
    require(cal["kappa"] >= 0.0, f"uncertainty multiplier {cal['kappa']} is negative")

    # What a reader is handed is this calibration, and nothing beside it.
    require(_number(consts.get("fat_finger_rate"))
            and abs(consts["fat_finger_rate"] - cal["lapse"]) <= LAPSE_TOL,
            f"fat_finger_rate {consts.get('fat_finger_rate')!r} is not the "
            f"calibrated lapse {cal['lapse']}")
    require(consts.get("slope_scale") == 1.0,
            f"slope_scale is {consts.get('slope_scale')!r}; the calibrated "
            f"scale is folded into the slopes and none ships beside them")
    require(consts.get("scale_fit") == "held_row_calibration_folded_in",
            f"scale_fit is {consts.get('scale_fit')!r}")

    # Every problem carries the one calibrated floor.
    n_problems = 0
    for size in ALL_BOARD_SIZES:
        path = os.path.join(RATINGS_DIR, str(size), "tsumego_floors.csv")
        require(os.path.exists(path), f"no {path}")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            require(reader.fieldnames == ["tsumego_id", "floor"],
                    f"{path}: unexpected columns {reader.fieldnames}")
            for row in reader:
                n_problems += 1
                require(abs(float(row["floor"]) - cal["floor"]) <= FLOOR_TOL,
                        f"{path}: {row['tsumego_id']} carries floor {row['floor']}, "
                        f"not the calibrated {cal['floor']}")
    require(n_problems > 0, "the floors files are empty")

    # The three curves are nested, so the three losses are ordered.
    require(cal["held_loss_plain"] <= cal["held_loss_as_trained"] + LOSS_TOL,
            f"the plain calibrated curve scores {cal['held_loss_plain']} on the "
            f"held rows against {cal['held_loss_as_trained']} as trained, which "
            f"it contains")
    require(cal["held_loss"] <= cal["held_loss_plain"] + LOSS_TOL,
            f"the shipped curve scores {cal['held_loss']} against "
            f"{cal['held_loss_plain']} under the plain one, which it contains")

    # Rows a real user contributed, held or not (synthetic users carry none).
    require(os.path.exists(USER_THETA), f"no {USER_THETA}")
    with open(USER_THETA, newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == ["profile", "theta", "n_attempts"],
                f"{USER_THETA}: unexpected columns {reader.fieldnames}")
        rows = sum(int(r["n_attempts"]) for r in reader)
    require(rows > cal["held_rows"],
            f"{USER_THETA} counts {rows} rows against {cal['held_rows']} held")

    return {
        "cal": cal, "trained": trained, "rows": rows, "n_problems": n_problems,
        "held_share": cal["held_rows"] / rows,
        "gain_mnat": 1000.0 * (cal["held_loss_as_trained"] - cal["held_loss"]),
        "plain_gain_mnat": 1000.0 * (cal["held_loss_as_trained"] - cal["held_loss_plain"]),
        "aware_gain_mnat": 1000.0 * (cal["held_loss_plain"] - cal["held_loss"]),
    }


def figures():
    m = measure()
    cal, trained = m["cal"], m["trained"]
    figs = [
        Figure("calibration.floor", cal["floor"], ".3f",
               "the one guessing floor every problem ships with, fitted on the held rows"),
        Figure("calibration.lapse", cal["lapse"], ".3f",
               "the shipped lapse, fitted on the held rows"),
        Figure("calibration.held_rows", cal["held_rows"], "~1000",
               "attempt rows held out of the fit and used to fit the curve's levels"),
        Figure("calibration.held_share_pct", m["held_share"], "%.0f",
               "held rows as a share of every row a user contributed"),
        Figure("calibration.trained_floor", trained["floor"], ".3f",
               "the floor the fit itself trained, on the rows it was fitted to"),
        Figure("calibration.trained_lapse", trained["lapse"], ".3f",
               "the lapse the fit itself trained"),
        Figure("calibration.plain_floor", cal["plain_floor"], ".3f",
               "the held-row floor under a curve that takes each player's fitted "
               "ability as exact"),
        Figure("calibration.plain_lapse", cal["plain_lapse"], ".3f",
               "the held-row lapse under that curve"),
        Figure("calibration.gain_mnat", m["gain_mnat"], ".1f",
               "held-row log-loss of the fit as trained minus the shipped curve's, "
               "millinats per attempt, rating factor alone"),
        Figure("calibration.plain_gain_mnat", m["plain_gain_mnat"], ".1f",
               "the same against the plain calibrated curve"),
        Figure("calibration.aware_gain_mnat", m["aware_gain_mnat"], ".2f",
               "held-row log-loss of the plain calibrated curve minus the "
               "shipped (uncertainty-averaged) one, millinats per attempt"),
        Figure("calibration.top10_fail_actual_pct", cal["top10_fail"][0], "%.1f",
               "failure rate on the tenth of held rows the shipped curve rates most "
               "likely to succeed"),
        Figure("calibration.top10_fail_predicted_pct", cal["top10_fail"][1], "%.1f",
               "the shipped curve's mean predicted failure rate on those rows"),
        Figure("calibration.bottom5_success_actual_pct", cal["bottom5_success"][0], "%.1f",
               "success rate on the twentieth of held rows the shipped curve rates "
               "least likely to succeed"),
        Figure("calibration.bottom5_success_predicted_pct", cal["bottom5_success"][1], "%.1f",
               "the shipped curve's mean predicted success rate on those rows"),
        Figure("calibration.kappa", cal["kappa"], ".2f",
               "fitted multiplier on each player's ability uncertainty"),
        Figure("calibration.slope_mult", cal["slope_mult"], ".2f",
               "fitted multiplier on the slope term, folded into the shipped slopes"),
        Figure("calibration.plain_lapse_ratio", cal["plain_lapse"] / cal["lapse"]
               if cal["lapse"] > 0 else 0.0, ".1f",
               "plain-curve lapse over the shipped lapse (0 where the shipped lapse is zero)"),
        # Guard-only: comparisons between two figures, which a guard cannot state.
        Figure("calibration.lapse_above_trained", int(cal["lapse"] > trained["lapse"]), "d",
               "1 when the held rows put the lapse above the one the fit trained"),
        Figure("calibration.plain_levels_higher",
               int(cal["plain_floor"] > cal["floor"] and cal["plain_lapse"] > cal["lapse"]), "d",
               "1 when the plain curve reads both a higher floor and a higher lapse "
               "than the shipped one"),
        Figure("calibration.ends_within_point",
               int(abs(cal["top10_fail"][0] - cal["top10_fail"][1]) < 0.01
                   and abs(cal["bottom5_success"][0] - cal["bottom5_success"][1]) < 0.01), "d",
               "1 when the shipped curve's predicted rate is within one percentage "
               "point of the held rows' at both of its ends"),
    ]
    return figs


def main():
    m = measure()
    cal, trained = m["cal"], m["trained"]
    print(f"{cal['held_rows']:,d} held rows of {m['rows']:,d} ({100 * m['held_share']:.1f}%), "
          f"{m['n_problems']:,d} problems on one floor\n")
    print(f"{'curve':<34}{'floor':>8}{'lapse':>8}{'held loss':>12}{'gain mnat':>11}")
    print(f"{'as trained (levels fitted in-sample)':<34}{trained['floor']:>8.3f}"
          f"{trained['lapse']:>8.3f}{cal['held_loss_as_trained']:>12.5f}{0.0:>11.2f}")
    print(f"{'plain (ability taken as exact)':<34}{cal['plain_floor']:>8.3f}"
          f"{cal['plain_lapse']:>8.3f}{cal['held_loss_plain']:>12.5f}{m['plain_gain_mnat']:>11.2f}")
    print(f"{'shipped (ability uncertainty)':<34}{cal['floor']:>8.3f}"
          f"{cal['lapse']:>8.3f}{cal['held_loss']:>12.5f}{m['gain_mnat']:>11.2f}")
    print("  held loss: mean log-loss per held row, rating factor alone; gain is "
          "against the fit as trained")
    print(f"\nslope multiplier {cal['slope_mult']:.3f}, offset {cal['offset']:+.3f} "
          f"(both folded into the shipped slopes and ratings); ability-uncertainty "
          f"multiplier {cal['kappa']:.2f}; population ability sd {cal['ability_sigma']:.3f}")
    print("\nheld rows at the shipped curve's ends, actual / predicted")
    print(f"{'top 10% by prediction, failed':>34}{100 * cal['top10_fail'][0]:>8.1f}%"
          f"{100 * cal['top10_fail'][1]:>8.1f}%")
    print(f"{'bottom 5% by prediction, cleared':>34}{100 * cal['bottom5_success'][0]:>8.1f}%"
          f"{100 * cal['bottom5_success'][1]:>8.1f}%")


if __name__ == "__main__":
    main()
