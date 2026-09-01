#!/usr/bin/env python3
"""The rating-orthogonality penalty's figures (methodology §1.5), read from the
`penalty` refit panel.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, running a config other than the manifest's, internally inconsistent,
or STALE against the current attempt log — the pins rule `docs/mirt_benchmark.md`
states, mechanized. Guards and spans keep evaluating against the last-measured
ledger values, so the panel's cost is paid only when it is deliberately re-run.

The four arms are the shipped per-tail penalty, the whole-axis form, the penalty
removed, and a free control carrying neither the penalty nor the teaching pull.

  penalty.whole_axis_reported      the whole-axis arm's worst WHOLE-AXIS |corr|
                                   against the shipped triple — the statistic
                                   that arm optimises, and the one it would be
                                   read as rating-orthogonal on.
  penalty.whole_axis_tail_leak     the SAME arm's worst PER-TAIL |cosine|. The
                                   two together are §1.5's understatement pair:
                                   a linear statistic averages an axis whose two
                                   ends are laden in opposite directions to zero
                                   and is satisfied completely, while both named
                                   ends stay laden.
  penalty.whole_axis_understatement  their ratio, the fold-understatement.
  penalty.per_tail_leak            the shipped arm's worst per-tail |cosine| —
                                   what scoring the ends separately cuts the
                                   worst of them to. This is the panel's own
                                   measurement of the quantity the live fit
                                   reports as `axes.rating_leak_worst`; main()
                                   prints the two side by side as context.
  penalty.unpenalized_tail_leak    the same statistic with the penalty removed:
                                   what the axes absorb when nothing holds them
                                   off the rating side.
  penalty.free_slope_corr          the free control's worst whole-axis |corr|
                                   against SLOPE specifically — §1.5's signature
                                   that an unheld axis is spent on a shipped
                                   response parameter, and on slope rather than
                                   on difficulty. main() prints its sign and
                                   which axis carries it.
  penalty.free_stop_cost_mnat      the free control's held-out (stop-split) cost
                                   against the shipped arm, in millinats —
                                   PAIRED: every arm holds out the same rows,
                                   drawn from the same seed at the same log,
                                   which the record's row/user/stop-row counts
                                   are checked against.

ONE STATISTIC ACROSS THE LIVE FIT AND THE PANEL. Both leakage quantities are
computed by `ablations.axes_fit_figures`'s own `leakage` and `whole_axis`,
imported rather than restated, so the hinge (`utils.rating.tail_agreement`), the
centring and the normalisation are single-sourced and a panel arm and the
shipped fit cannot drift into being scored by two different numbers.

TWO POPULATION CHOICES, both inherited from that producer rather than re-decided
here. Leakage runs over EVERY fitted problem of the arm, unweighted — what a fit
ships is one loading per problem over the whole corpus, and that is the
population a client draws from; the trainer optimises the same quantity
attempt-weighted over attempted problems, which is a different number. And a
"worst" is the maximum over both ends of all K axes against all three shipped
quantities (rating, slope, floor), the whole grid the penalty is a sum of squares
over — never one axis or one quantity, except `penalty.free_slope_corr`, which
is restricted to the slope column on purpose because that restriction IS the
claim.

The held-out column is reported for every arm and only the free control's is a
figure. §1.5 states outright that the per-end form's held-out cost against the
whole-axis form is NOT established — one seed cannot separate them — so a key
for that pair would back a claim the document declines to make.

    ./.venv/bin/python3 -m ablations.refit_penalty_figures
"""

import json
import os
import sqlite3
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, load_ledger, require, panel_budget  # noqa: E402
from ablations.axes_fit_figures import leakage, whole_axis  # noqa: E402
from pipeline.train_mirt import EPOCHS  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.features import load_backend_problem_params  # noqa: E402
from utils.rating.model_constants import K_AXES, SIZES  # noqa: E402
from utils.rating.tail_agreement import TAIL_HINGE  # noqa: E402

TIER = "refit"

PANEL = "penalty"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
ARMS = ("shipped", "ortho_whole", "ortho_off", "free_control")
# The column order `load_backend_problem_params` returns, and the order the
# penalty's own target matrix is built in.
QUANTITIES = ("rating", "slope", "floor")
SLOPE_COL = QUANTITIES.index("slope")
# The live fit's own reading of `penalty.per_tail_leak`'s quantity. Printed as
# context, never required: the two are separate fits of the same configuration
# and §1.14's seed spread is real, so agreement is reassurance and disagreement
# is a question rather than a failure.
LIVE_KEY = "axes.rating_leak_worst"


# --- the panel record ---------------------------------------------------------

def load_panel():
    """Every arm's record, checked for the four things that make the panel
    quotable: production budget, the manifest's own config, pins agreeing across
    arms, and a snapshot still matching the current attempt log."""
    require(os.path.exists(MANIFEST),
            f"no {MANIFEST} — the panel manifest is what defines these arms")
    with open(MANIFEST) as f:
        manifest = json.load(f)
    require(set(manifest["arms"]) == set(ARMS),
            f"{MANIFEST} names arms {sorted(manifest['arms'])}, this producer "
            f"reads {sorted(ARMS)} — the manifest and the claims have separated")

    arms = {}
    for arm in ARMS:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the penalty panel has not run. Run it with "
                f"`python -m ablations.run_panel penalty` (or "
                f"`vast_orchestration.py ablate penalty`) first.")
        with open(path) as f:
            arms[arm] = json.load(f)

    budget = panel_budget(arms)
    for arm, r in arms.items():
        require(r["metrics"]["epochs"] == budget,
                f"{arm} ran {r['metrics']['epochs']} epochs against the panel's "
                f"{budget} — one panel is one budget")
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == EPOCHS,
                f"{arm}'s convergence trace does not reach its budget")

    # THE ARM RAN THE MANIFEST'S CONFIG. Every claim below is about a named knob
    # setting, so an arm whose record disagrees with the manifest it is named
    # after — a manifest edited after the run, or a hand-invoked arm dropped into
    # the panel directory — is measuring something else under this arm's name.
    for arm, r in arms.items():
        want, got = manifest["arms"][arm], r["config"]
        for k, v in want.items():
            require(k in got,
                    f"{arm} was run without {k}, which its manifest entry sets "
                    f"to {v!r} — the record is not this arm")
            same = (float(got[k]) == float(v)
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    else got[k] == v)
            require(same,
                    f"{arm} ran with {k}={got[k]!r}, the manifest says {v!r}")
        extra = [k for k in got if k not in want and k != "epochs"]
        require(not extra,
                f"{arm} ran with {extra} beyond its manifest entry — the arm "
                f"carries a knob the panel's claims do not account for")

    base = arms["shipped"]["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha",
                    "blocks", "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} ({r['pins'][pin]!r}) differs from the "
                    f"shipped arm's ({base[pin]!r}) — the arms are not one "
                    f"panel and their comparison is void")
    require(int(base["k_axes"]) == K_AXES,
            f"the panel was fitted at K={base['k_axes']}, this tree ships "
            f"K_AXES={K_AXES} — the axis grid the worsts range over has changed")

    # THE PAIRING, CHECKED RATHER THAN ASSUMED. The stop split is drawn from the
    # seed over the attempt frame, so identical seeds at an identical log give
    # identical held-out rows — which is what makes the millinat column a paired
    # comparison instead of two independent estimates. Equal row, user and
    # stop-row counts are what that identity looks like from the record.
    ship = arms["shipped"]["metrics"]
    for arm, r in arms.items():
        for field in ("rows", "users", "stop_rows", "base_rate"):
            require(r["metrics"][field] == ship[field],
                    f"{arm} fitted {field}={r['metrics'][field]!r} against the "
                    f"shipped arm's {ship[field]!r} — the arms do not hold out "
                    f"the same rows and the millinat column is not paired")
        require(r["metrics"]["bank_spec"] == ship["bank_spec"],
                f"{arm} was taught a different rule bank from the shipped arm — "
                f"its axes are not the same axes by index")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"the panel was measured at attempt-log frontier {base['db_max_id']} "
            f"but the log now ends at {now} — the stop-row draw moved, the "
            f"paired comparison is void. Re-run the panel.")
    return arms


# --- one arm's fit, as artifacts ----------------------------------------------

def load_arm_fit(arm):
    """The arm's own loadings and its own shipped triple, joined BY ID.

    Both files are the arm's, never production's: `<arm>/axes/per_problem.csv`
    and `<arm>/ratings/<size>/tsumego_*.csv`. The three triple CSVs are each
    sorted by their own value, so they are read through
    `load_backend_problem_params` — the shared loader, keyed by tsumego id —
    rather than stacked positionally, which would pair a problem's rating with a
    stranger's slope and be well-formed the whole way."""
    import pandas as pd
    per = os.path.join(PANEL_DIR, arm, "axes", "per_problem.csv")
    require(os.path.exists(per),
            f"no {per} — the {arm} arm's record exists but its axes do not")
    df = pd.read_csv(per)
    cols = [f"embed_{k}" for k in range(K_AXES)]
    missing = [c for c in cols + ["tsumego_id", "size"] if c not in df.columns]
    require(not missing, f"{per} lacks {missing}")
    extra = [c for c in df.columns if c.startswith("embed_") and c not in cols]
    require(not extra, f"{per} carries {extra} past K_AXES={K_AXES}")
    G = df[cols].to_numpy(np.float64)
    require(np.all(np.isfinite(G)), f"{per}: non-finite loading")
    # The gauge divides every loading by the corpus RMS of its own axis, so a
    # loading is in axis-standard-deviation units. Checked per arm, because a
    # collapsed axis under a freed arm is exactly the thing that would silently
    # rescale the correlations below.
    rms = np.sqrt((G ** 2).mean(axis=0))
    require(np.all(np.abs(rms - 1.0) < 0.01),
            f"{arm}: axis loadings are not standardized: per-axis RMS "
            f"{rms.round(4)}")

    keys = list(zip(df["size"].astype(int), df["tsumego_id"].astype(str)))
    ratings_dir = os.path.join(PANEL_DIR, arm, "ratings")
    require(os.path.isdir(ratings_dir),
            f"no {ratings_dir} — the {arm} arm wrote no per-problem triple")
    params = load_backend_problem_params(tuple(SIZES), ratings_dir=ratings_dir)
    absent = [t for _, t in keys if t not in params]
    require(not absent,
            f"{arm}: {len(absent)} fitted problems have no triple (e.g. "
            f"{absent[:3]}) — the arm's ratings/ and axes/ are from different "
            f"runs")
    tri = np.array([params[t] for _, t in keys], dtype=np.float64)
    require(np.all(np.isfinite(tri)), f"{arm}: a rating/slope/floor is not finite")
    require(np.all(tri[:, 1] > 0.0), f"{arm}: a slope is not positive")
    require(np.all((tri[:, 2] >= 0.0) & (tri[:, 2] < 1.0)),
            f"{arm}: a floor is outside [0, 1)")
    return {"keys": keys, "G": G, "tri": tri}


# --- the measurement ----------------------------------------------------------

def opposite_end_pair(ends):
    """The (axis, quantity) whose two ends lean OPPOSITE ways hardest — the cell
    where the whole-axis statistic's blindness is visible outright, since the two
    are what it averages toward zero. None when no cell has opposing ends."""
    best = None
    for k in range(ends["+"].shape[0]):
        for j in range(len(QUANTITIES)):
            p, n = float(ends["+"][k, j]), float(ends["-"][k, j])
            if p * n >= 0:
                continue
            if best is None or abs(p) + abs(n) > best[2]:
                best = (k, j, abs(p) + abs(n), p, n)
    return best


def measure():
    arms = load_panel()
    fits = {arm: load_arm_fit(arm) for arm in ARMS}
    keys = fits["shipped"]["keys"]
    for arm in ARMS:
        require(fits[arm]["keys"] == keys,
                f"{arm} fitted a different problem set from the shipped arm — "
                f"the leakage populations are not comparable")

    out = {}
    for arm in ARMS:
        G, tri = fits[arm]["G"], fits[arm]["tri"]
        ends = leakage(G, tri)
        tails = np.stack([ends["+"], ends["-"]])          # (2, K, 3)
        w = whole_axis(G, tri)                            # (K, 3)
        out[arm] = {
            "ends": ends, "tails": tails, "whole": w,
            "tail_worst": float(np.abs(tails).max()),
            "tail_where": np.unravel_index(int(np.abs(tails).argmax()), tails.shape),
            "whole_worst": float(np.abs(w).max()),
            "whole_where": np.unravel_index(int(np.abs(w).argmax()), w.shape),
            "stop_bce": float(arms[arm]["metrics"]["trace"][-1]["stop_bce"]),
        }
    for arm in ARMS:
        out[arm]["cost_mnat"] = (out[arm]["stop_bce"]
                                 - out["shipped"]["stop_bce"]) * 1e3

    wa, sh = out["ortho_whole"], out["shipped"]
    require(wa["whole_worst"] > 0.0,
            "the whole-axis arm reports a worst whole-axis correlation of "
            "exactly zero — the understatement ratio has no denominator")
    # The two claims the panel exists to establish, as opposed to the magnitudes
    # it exists to measure. Either failing refutes the reason the per-tail form
    # is what ships, not merely the size of the number.
    require(wa["tail_worst"] > wa["whole_worst"],
            f"the whole-axis arm's ends ({wa['tail_worst']:.3f}) carry no more "
            f"rating than its whole-axis statistic ({wa['whole_worst']:.3f}) — "
            f"§1.5's claim that the linear form hides the dependence is "
            f"refuted, not just its magnitude")
    require(sh["tail_worst"] < wa["tail_worst"],
            f"the per-tail arm's worst end ({sh['tail_worst']:.3f}) is no better "
            f"than the whole-axis arm's ({wa['tail_worst']:.3f}) — the per-tail "
            f"form's stated purpose is refuted")

    free = out["free_control"]
    slope_col = free["whole"][:, SLOPE_COL]
    k_slope = int(np.abs(slope_col).argmax())

    led = load_ledger()["figures"].get(LIVE_KEY)
    return {
        "arms": arms, "out": out, "n_problems": len(keys),
        "gpu": arms["shipped"]["pins"]["gpu"],
        "bank_spec": list(arms["shipped"]["metrics"]["bank_spec"]),
        "whole_reported": wa["whole_worst"],
        "whole_tail": wa["tail_worst"],
        "understatement": wa["tail_worst"] / wa["whole_worst"],
        "whole_opposite": opposite_end_pair(wa["ends"]),
        "per_tail": sh["tail_worst"],
        "unpenalized": out["ortho_off"]["tail_worst"],
        "free_slope_corr": float(abs(slope_col[k_slope])),
        "free_slope_signed": float(slope_col[k_slope]),
        "free_slope_axis": k_slope,
        "free_slope_rating": float(free["whole"][k_slope, 0]),
        "free_cost_mnat": free["cost_mnat"],
        "live_leak": None if led is None else float(led["value"]),
        "live_measured": None if led is None else led["measured"],
    }


def figures():
    m = measure()
    return [
        Figure("penalty.whole_axis_reported", m["whole_reported"], ".2f",
               "worst whole-axis |correlation| between an axis and one of the "
               "shipped per-problem quantities (rating, slope, floor) in the "
               "arm penalized on the whole axis — the statistic that arm "
               "optimises, over every fitted problem"),
        Figure("penalty.whole_axis_tail_leak", m["whole_tail"], ".2f",
               "the same arm's worst PER-TAIL |cosine|: one hinged axis END "
               "against one centred shipped quantity, the ends being the only "
               "part of an axis a name is read off"),
        Figure("penalty.whole_axis_understatement", m["understatement"], ".1f",
               "how many fold the whole-axis statistic understates what the "
               "ends of the same fit carry — the second figure over the first"),
        Figure("penalty.per_tail_leak", m["per_tail"], ".2f",
               "worst per-tail |cosine| under the SHIPPED penalty, which scores "
               "the two ends separately — what penalizing per tail cuts the "
               "worst end to, measured on the panel's own shipped arm"),
        Figure("penalty.unpenalized_tail_leak", m["unpenalized"], ".2f",
               "worst per-tail |cosine| with the rating-orthogonality penalty "
               "removed — what the axes absorb when nothing holds them off the "
               "shipped quantities"),
        Figure("penalty.free_slope_corr", m["free_slope_corr"], ".2f",
               "largest |whole-axis correlation| with SLOPE in the free control "
               "(no rating-orthogonality penalty, no teaching pull) — an "
               "unheld axis spent on a shipped response parameter, and on the "
               "slope rather than on difficulty"),
        Figure("penalty.free_stop_cost_mnat", m["free_cost_mnat"], ".1f",
               "the free control's held-out (stop-split) cost against the "
               "shipped arm, millinats, paired rows at one seed"),
    ]


# --- the table ----------------------------------------------------------------

def _axis_names(spec):
    return [s[:20] for s in spec]


def _tail_table(name, arm, names):
    print(f"\n  {name}")
    print(f"    {'axis':<22}{'+rating':>9}{'+slope':>8}{'+floor':>8}"
          f"{'-rating':>10}{'-slope':>8}{'-floor':>8}   {'whole-axis r,s,f':>22}")
    for k in range(K_AXES):
        print(f"    {names[k]:<22}"
              + "".join(f"{v:>9.3f}" for v in arm["ends"]["+"][k])
              + "".join(f"{v:>{10 if j == 0 else 8}.3f}"
                        for j, v in enumerate(arm["ends"]["-"][k]))
              + "   " + " ".join(f"{v:+.3f}" for v in arm["whole"][k]))


def main():
    m = measure()
    out = m["out"]
    names = _axis_names(m["bank_spec"])
    free_names = [f"axis {k}" for k in range(K_AXES)]

    print("\npenalty panel — methodology §1.5, the rating-orthogonality term")
    print(f"four arms at the production budget of {EPOCHS} epochs on one "
          f"machine ({m['gpu']}),")
    print(f"{m['n_problems']:,d} fitted problems per arm, identical problem set "
          f"and identical held-out rows.")
    print("\nLeakage is measured over EVERY fitted problem, unweighted, by the "
          "same two\nfunctions ablations.axes_fit_figures uses on the shipped "
          f"fit: a per-tail entry is\nthe cosine between one hinged axis end "
          f"(z >= {TAIL_HINGE}) and one centred shipped\nquantity, a whole-axis "
          "entry the plain correlation. A 'worst' is the maximum\nover both "
          f"ends of all {K_AXES} axes against all three quantities "
          f"({', '.join(QUANTITIES)}).")

    print(f"\n{'arm':<14}{'worst per-tail':>16}{'worst whole-axis':>18}"
          f"{'ratio':>8}{'stop bce':>11}{'vs shipped':>13}")
    for arm in ARMS:
        a = out[arm]
        ratio = a["tail_worst"] / a["whole_worst"] if a["whole_worst"] > 0 else float("nan")
        cost = "" if arm == "shipped" else f"{a['cost_mnat']:+.2f} mnat"
        print(f"{arm:<14}{a['tail_worst']:>16.3f}{a['whole_worst']:>18.3f}"
              f"{ratio:>8.1f}{a['stop_bce']:>11.4f}{cost:>13}")
    print("  ratio = worst per-tail / worst whole-axis, both over the same "
          "axis x quantity grid.")
    print("  The millinat column is PAIRED (same stop rows, one seed) and is a "
          "single-seed\n  reading: §1.5 states the per-end form's held-out cost "
          "against the whole-axis\n  form is not established, so only the free "
          "control's cost is a figure.")

    print("\n=== the understatement: what a whole-axis penalty reports vs what "
          "its ends carry ===")
    wa = out["ortho_whole"]
    print(f"  reported (whole-axis) {m['whole_reported']:.3f} at "
          f"{names[wa['whole_where'][0]]} / {QUANTITIES[wa['whole_where'][1]]}; "
          f"ends carry {m['whole_tail']:.3f} at "
          f"{names[wa['tail_where'][1]]} / "
          f"{QUANTITIES[wa['tail_where'][2]]} "
          f"({'+' if wa['tail_where'][0] == 0 else '-'} end)"
          f" — {m['understatement']:.1f}-fold.")
    op = m["whole_opposite"]
    if op is None:
        print("  no (axis, quantity) cell has ends leaning opposite ways in "
              "this fit — the\n  mechanism is present as magnitude only.")
    else:
        k, j, _, p, n = op
        print(f"  the mechanism outright: {names[k]} carries {p:+.3f} on its "
              f"+ end and {n:+.3f} on\n  its - end against the same quantity "
              f"({QUANTITIES[j]}), which a linear statistic\n  averages toward "
              f"zero (whole-axis {wa['whole'][k, j]:+.3f}).")
    _tail_table("the whole-axis arm, per axis:", wa, names)
    _tail_table("the shipped per-tail arm, per axis:", out["shipped"], names)

    print("\n=== with the penalty removed ===")
    off = out["ortho_off"]
    s, k, j = off["tail_where"]
    print(f"  worst end {off['tail_worst']:.3f} — {names[k]}'s "
          f"{'+' if s == 0 else '-'} end against {QUANTITIES[j]}, against "
          f"{m['per_tail']:.3f} under the shipped penalty.")

    print("\n=== the free control (no penalty, no teaching) ===")
    print("  Its axes are taught nothing, so they carry no rule names; they are "
          "indexed.")
    _tail_table("whole-axis correlations against the shipped triple:", out["free_control"],
                free_names)
    print(f"  worst against SLOPE: axis {m['free_slope_axis']} at "
          f"{m['free_slope_signed']:+.3f}, the same axis sitting at "
          f"{m['free_slope_rating']:+.3f}\n  against the rating — the axis is "
          f"spent on a shipped response parameter, and on\n  the slope rather "
          f"than on difficulty.")
    print(f"  held-out cost against the shipped arm: "
          f"{m['free_cost_mnat']:+.2f} mnat.")

    print("\n=== context: the panel's shipped arm against the live fit ===")
    if m["live_leak"] is None:
        print(f"  the ledger holds no {LIVE_KEY} to compare against.")
    else:
        print(f"  {LIVE_KEY} (the production fit, read back by "
              f"ablations.axes_fit_figures)\n  is {m['live_leak']:.3f}; this "
              f"panel's shipped arm reads {m['per_tail']:.3f} on the same "
              f"statistic.")
        print(f"  Measured at db max id {m['live_measured']['db_max_id']} "
              f"(git {m['live_measured']['git']}) against this panel's "
              f"{m['arms']['shipped']['pins']['db_max_id']}.")
        print("  These are two fits of one configuration, so they are compared "
              "and never\n  required to agree — the seed spread between runs of "
              "this fit is real.")


if __name__ == "__main__":
    main()
