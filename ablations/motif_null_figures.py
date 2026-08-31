#!/usr/bin/env python3
"""What the availability null changes about the tendency reading, and what the
null's own reference costs — the figures behind the methodology's motif-null
claims.

Two independent measurements, both against what ships.

FIRST, it drives `ablations.eval_error_null_models` in full (its default
production arms: the shipped per-cell store, the shipped fit, the whole recorded
-error log) and reads its phase-2 and phase-2b statistics at the null the client
ships, the VISITS-LINEAR availability null:

  motif.null_axes_measured    the shipped axis count the measurement ran over,
                              read from the same labels file the fit is taught
                              from — the count the sentences below are stated per.
  motif.null_sign_reversals   axes whose population lean (the pooled toward-share
                              of recorded errors) sits on one side of the fixed
                              even split and the OTHER side of the shipped null's
                              balance point — the menu, not the players, decided
                              the direction.
  motif.menu_worst_toward_pct on the most menu-skewed axis (the shipped null's
                              balance point farthest from an even split), the
                              share of the offered alternatives' weight on the
                              toward side.
  motif.fixed_null_worst_flag_pct   on that same axis, the share of testable
                              users (>= the client's minimum scored errors) the
                              retired fixed-50% test would flag.
  motif.shipped_null_worst_flag_pct on that same axis, the share the shipped
                              null's test flags.
  motif.depth_null_axes_better      axes where the shipped depth-weighted null
                              predicts the DIRECTION of a recorded error better
                              than the equal-count one with the paired interval
                              clear of zero — the adjudicating table's own
                              criterion, per axis.
  motif.depth_null_pooled_gain      that same win pooled over axes, in nats of
                              directional log-loss per scored error.
  motif.fitted_null_gain_share_pct  what a null FITTED to the recorded errors
                              adds on top of the depth-weighted one, as a share
                              of the pooled gain above.
  motif.tendency_rank_users   accounts carrying both a fitted ability and enough
                              scored errors to read on the axis below.
  motif.tendency_rank_share_population_pct  the largest per-axis share of the
                              between-user variance in the tendency reading that
                              the user's RANK alone explains, when that reading
                              is scored against a population average — the
                              standing alternative to a per-problem null.
  motif.tendency_rank_share_shipped_pct     the same share for the reading that
                              ships, scored against each problem's own null.

SECOND, from the published artifacts alone, the cost of taking the availability
null's reference vector from anywhere but the problem's own per-cell block:

  motif.altstat_crossproj_setwide_pct   share of a seeded corpus sample whose
                              published masses would move if the reference came
                              from the set-wide correct-move sidecar.
  motif.altstat_crossproj_manifest_pct  the same against each shard's manifest.

    ./.venv/bin/python3 -m ablations.motif_null_figures
"""

import json
import os
import sys
import zipfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations import eval_error_null_models as nulls  # noqa: E402
from ablations._figures import Figure, require  # noqa: E402
# The packaging step's own readers and its own mass computation, imported rather
# than restated: the cross-projection measurement below has to be the SHIPPED
# arithmetic with one input swapped, or it measures a second implementation.
from pipeline import package_for_pwa as pkg  # noqa: E402
from utils.rating.ratings_io import apply_M, load_M, load_value_map  # noqa: E402

TIER = "methodology"

# The null the client ships (CLAUDE.md, methodology §2.9): selected by tag, never
# by column position — column 0 is the unweighted reference arm.
SHIPPED_TAG = "visits-lin"
# The one FITTED arm in phase 2's panel. It is a reference bound rather than a
# shippable candidate — it would have to ship its parameters with the corpus —
# so what it buys over the shipped null is the price of the parameter-free rule.
FITTED_TAG = "zipf+wr*"

PWA_DIR = os.path.join(PROJECT_ROOT, "pwa")
RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")

# The cross-projection sample. Every problem in it costs an SGF parse out of its
# shard, so the whole corpus would be an hour; this is a rate of a few percent,
# and at this size its binomial standard error is around a tenth of a percentage
# point — finer than the figure it backs is quoted at.
CROSSPROJ_SAMPLE = 20000
CROSSPROJ_SEED = 0


def _column(stats, tag):
    tags = stats["tags"]
    require(tag in tags, f"phase-2 panel carries no {tag!r} column")
    return tags.index(tag)


def _adjudication(stats):
    """Phase 2b read at the two columns §2.9's sentences are about.

    Phase 2b scores every null on the ONE number the tendency test consumes —
    the probability that a direction-blind error lands above the answer — so it
    is what adjudicates between them. Column 0 is the equal-count (uniform) arm
    there as everywhere, so a null's gain over equal counting is minus its delta,
    and the fitted arm's gain over the shipped one is the difference of the two
    deltas on the same resample.
    """
    adj = stats["adjudication"]
    ship, fit = _column(stats, SHIPPED_TAG), _column(stats, FITTED_TAG)
    ll = np.asarray(adj["ll"], dtype=float)              # (n_null, K)
    ll_all = np.asarray(adj["ll_all"], dtype=float)      # (n_null,)
    b_d = np.asarray(adj["b_d"], dtype=float)            # (B, n_null, K)
    b_d_all = np.asarray(adj["b_d_all"], dtype=float)    # (B, n_null)
    K = len(stats["labels"])
    require(ll.shape == (len(stats["tags"]), K) and b_d.shape[1:] == ll.shape,
            "phase 2b returned a panel of the wrong shape")
    require(np.all(np.isfinite(ll)) and np.all(np.isfinite(ll_all)),
            "an axis has no finite directional log-loss")

    # An axis counts as won only when the interval is clear of zero, which is the
    # claim's own wording — the point estimate alone would count a tie.
    hi = np.percentile(b_d[:, ship, :], 97.5, axis=0)
    better = int(np.sum((ll[ship] < ll[0]) & (hi < 0.0)))

    gain = float(ll_all[0] - ll_all[ship])
    require(gain > 0.0,
            "the depth-weighted null does not beat the equal-count one pooled "
            f"({gain:+.5f} nats) — the weighting choice §2.9 states is refuted, "
            "and the share below would divide by a loss")
    lo, hi_all = np.percentile(b_d_all[:, ship], [2.5, 97.5])
    return {
        "depth_axes_better": better,
        "depth_gain": gain,
        "depth_gain_ci": (-hi_all, -lo),       # gain is minus the delta
        "fitted_gain": float(ll_all[ship] - ll_all[fit]),
        "fitted_share": float(ll_all[ship] - ll_all[fit]) / gain,
        "scored_errors": float(np.asarray(adj["n"], dtype=float).sum()),
    }


def _rank_share(stats):
    """How much of the between-user spread in a tendency reading is the user's
    RANK wearing a different name.

    TWO ARMS, and they answer different sentences. The SHIPPED one is the
    per-user excess toward-share against the shipped availability null — observed
    leans minus the null's own expected count, over that user's scored errors on
    the axis, which is the quantity the client reports. The POPULATION one is the
    standing alternative: the same user's raw toward-share measured against the
    average over users instead of against each problem's own neutral point. That
    arm is the alternative EXACTLY rather than as a stand-in — the scheme differs
    from the raw share by one constant per axis, and a constant cannot move a
    regression's R², so what such a scheme would report is measurable without
    estimating the population number it would subtract.

    The user's rank is their FITTED ABILITY carried through the shipped
    rating→rank curve, on the internal scale; no time adjustment enters, because
    nothing here is compared against a rank declared on another server. The share
    is the R² of a simple linear regression of the reading on that rank, per
    axis, over the accounts holding both — and each arm's figure is the largest
    of its axes, the strongest-affected motif of the sentence it backs.
    """
    ship = _column(stats, SHIPPED_TAG)
    labels = stats["labels"]
    K = len(labels)
    min_errors = int(stats["min_errors"])
    theta = load_value_map(os.path.join(RATINGS_DIR, "user_theta.csv"), "theta")
    require(theta, f"no fitted abilities in {RATINGS_DIR}/user_theta.csv")
    m_ratings, m_internal = load_M(RATINGS_DIR)

    ranks, shipped, share = [], [], []
    n_no_ability = 0
    for uid, acc in stats["users"].items():
        th = theta.get(uid)
        if th is None:
            n_no_ability += 1
            continue
        ranks.append(apply_M(float(th), m_ratings, m_internal))
        with np.errstate(invalid="ignore", divide="ignore"):
            n_s = acc["n"][ship]
            shipped.append(np.where(n_s >= min_errors,
                                    (acc["tw"][ship] - acc["p"][ship])
                                    / np.maximum(n_s, 1), np.nan))
            # The raw toward-share, on the errors the reference arm admits --
            # the same errors, since an error is directional or it is not.
            n_u = acc["n"][0]
            share.append(np.where(n_u >= min_errors,
                                  acc["tw"][0] / np.maximum(n_u, 1), np.nan))
    require(ranks, "no account carries both a recorded error and a fitted ability")
    rank = np.asarray(ranks, dtype=float)
    share = np.asarray(share, dtype=float)
    # The population scheme subtracts the average over users, per axis. Doing it
    # explicitly rather than leaving the raw share in place, so the column is the
    # alternative itself; the R² is the same either way.
    with np.errstate(invalid="ignore"):
        population = share - np.nanmean(share, axis=0, keepdims=True)
    arms = {"shipped null": np.asarray(shipped, dtype=float),
            "population mean": population}

    out = {"n_no_ability": n_no_ability, "min_errors": min_errors, "arms": {}}
    for name, reading in arms.items():
        r2 = np.full(K, np.nan)
        n_u = np.zeros(K, dtype=int)
        for k in range(K):
            sel = np.isfinite(reading[:, k])
            n_u[k] = int(sel.sum())
            if n_u[k] < 3:
                continue
            x, y = rank[sel], reading[sel, k]
            if np.std(x) <= 0 or np.std(y) <= 0:
                continue
            r2[k] = float(np.corrcoef(x, y)[0, 1] ** 2)
        require(np.isfinite(r2).any(),
                f"no axis has enough readable accounts for the {name} arm")
        kmax = int(np.nanargmax(r2))
        out["arms"][name] = {"r2": r2, "users": n_u, "worst_axis": kmax}
    for name, arm in out["arms"].items():
        k = arm["worst_axis"]
        arm["share_max"] = float(arm["r2"][k])
        arm["share_axis"] = labels[k]
        arm["share_users"] = int(arm["users"][k])
    return out


def measure():
    """Everything that comes out of one full eval_error_null_models run."""
    stats = nulls.main([])
    require(isinstance(stats, dict), "eval_error_null_models returned no phase-2 statistics")
    labels = stats["labels"]
    j = _column(stats, SHIPPED_TAG)
    obs = np.asarray(stats["obs"], dtype=float)
    ship = np.asarray(stats["pnull"], dtype=float)[j]
    K = len(labels)
    require(obs.shape == (K,) and ship.shape == (K,), "phase-2 statistics have the wrong shape")
    require(np.all(np.isfinite(obs)) and np.all(np.isfinite(ship)),
            "an axis has no finite toward-share — the per-cell store is behind the corpus")

    reversals = int(np.sum(np.sign(obs - 0.5) != np.sign(obs - ship)))
    worst = int(np.argmax(np.abs(ship - 0.5)))
    tested0 = float(stats["tested"][0][worst])
    tested_j = float(stats["tested"][j][worst])
    require(tested0 > 0 and tested_j > 0, "no testable users on the most menu-skewed axis")
    out = {
        "axes": K, "reversals": reversals,
        "labels": labels,
        "worst_axis": labels[worst],
        "menu_worst": float(ship[worst]),
        "fixed_flag_share": float(stats["flag_legacy"][worst]) / tested0,
        "shipped_flag_share": float(stats["flag"][j][worst]) / tested_j,
    }
    out.update(_adjudication(stats))
    out["rank"] = _rank_share(stats)
    return out


def _shard_manifest_vectors(size, zip_hash, K):
    """{tsumego id: correct-move vector} out of one shard's manifest — the
    per-problem twin of the set-wide sidecar, carried inside the archive the
    client downloads. Plain JSON numbers, which is what a device reads them as."""
    path = os.path.join(PWA_DIR, "tsumegos", str(size), f"{zip_hash}.zip")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("metadata.json"))
    out = {}
    for rec in manifest:
        name = rec.get("filename", "")
        vec = rec.get("embedding")
        if not name.endswith(".sgf") or vec is None:
            continue
        require(len(vec) == K,
                f"{name}: manifest vector has {len(vec)} axes, sidecar has {K}")
        out[name[:-4]] = np.asarray(vec, dtype=np.float64)
    return out


def cross_projection(n_sample=CROSSPROJ_SAMPLE, seed=CROSSPROJ_SEED):
    """How often the published availability null would MOVE if its reference
    vector came from somewhere other than the problem's own per-cell block.

    The masses are strict per-axis comparisons of every alternative against the
    correct move's loading, so the reference decides every one of them. Two
    nominally equivalent vectors for that same correct move exist beside the
    block's own first row — the set-wide correct-move sidecar and each shard's
    manifest — and both come from a separate computation, so referencing one
    compares cells drawn from two different projections.

    This runs the SHIPPED computation twice per problem with the reference row
    swapped and nothing else touched: replacing row 0 in the block leaves the
    alternatives, their weights and the tie rule exactly as packaging has them,
    and the substituted row still ties with itself and drops, so the two runs
    differ in the reference and in nothing else. A rate is the share of the
    sample where any of the 2K masses changes.

    Reads the published artifacts only; it never re-runs packaging.
    """
    rows = pkg._lookup_rows(PWA_DIR)
    require(rows, f"{PWA_DIR}/lookup.txt is empty — run pipeline.package_for_pwa")
    setwide_path = os.path.join(PWA_DIR, "tsumegos", "lookup.embeddings.bin")
    require(os.path.exists(setwide_path), f"no {setwide_path}")
    flat = np.fromfile(setwide_path, dtype=pkg.EMBEDDING_DTYPE)
    require(flat.size % len(rows) == 0,
            f"lookup.embeddings.bin holds {flat.size} floats, not a whole number "
            f"of rows for {len(rows)} lookup lines")
    K = flat.size // len(rows)
    setwide = flat.reshape(len(rows), K)

    rng = np.random.default_rng(seed)
    take = set(int(i) for i in rng.choice(len(rows), size=min(n_sample, len(rows)),
                                          replace=False))
    by_shard = {}
    for key, items in pkg._group_rows_by_shard(rows).items():
        kept = [(i, tid) for i, tid in items if i in take]
        if kept:
            by_shard[key] = kept

    names = ("set-wide sidecar", "shard manifest")
    moved = {n: 0 for n in names}
    compared = {n: 0 for n in names}
    absent_ref = {n: 0 for n in names}
    n_problems = 0
    for (size, zip_hash), items in sorted(by_shard.items()):
        index = pkg._read_shard_index(PWA_DIR, size, zip_hash)
        bin_path = os.path.join(PWA_DIR, "tsumegos", str(size),
                                f"{zip_hash}.embeddings.bin")
        require(index is not None and os.path.exists(bin_path),
                f"shard {size}/{zip_hash} carries no per-cell blocks — its "
                f"published masses are zero-filled and this rate would be "
                f"measured on a corpus the sidecar does not describe")
        blob = np.fromfile(bin_path, dtype=pkg.EMBEDDING_DTYPE)
        tids = [tid for _, tid in items]
        for tid in tids:
            require(tid in index, f"{tid}: not in its own shard's block index")
        visits_by_tid = pkg._read_shard_variation_visits(PWA_DIR, size, zip_hash, tids)
        manifest = _shard_manifest_vectors(size, zip_hash, K)
        for row_i, tid in items:
            entry = index[tid]
            start = int(entry["offset"]) // pkg.EMBEDDING_BYTES_PER_FLOAT
            n_cells = int(entry["n_cells"])
            block = blob[start:start + n_cells * K]
            require(block.size == n_cells * K,
                    f"{tid}: shard block is short for its own index "
                    f"({block.size} floats, expected {n_cells * K})")
            q = pkg._js_quantize(block.reshape(n_cells, K))
            require(not np.all(q[0] == 0.0),
                    f"{tid}: first root variation is the pass placeholder")
            visits = visits_by_tid[tid]
            require(visits.size == n_cells,
                    f"{tid}: shard SGF has {visits.size} root variations against "
                    f"{n_cells} rows in its per-cell block")
            base_above, base_below = pkg.alt_stat_masses(q, visits)
            n_problems += 1
            for name, ref in zip(names, (setwide[row_i], manifest.get(tid))):
                require(ref is not None,
                        f"{tid}: shard {zip_hash}'s manifest carries no vector for it")
                ref_q = pkg._js_quantize(ref)
                if np.all(ref_q == 0.0):
                    # The documented "absent from the axis fit" row, which a
                    # reader treats as missing rather than as average. Nothing to
                    # compare, so it leaves the denominator rather than counting
                    # as a move.
                    absent_ref[name] += 1
                    continue
                swapped = q.copy()
                swapped[0] = ref_q
                above, below = pkg.alt_stat_masses(swapped, visits)
                compared[name] += 1
                if not (np.array_equal(above, base_above)
                        and np.array_equal(below, base_below)):
                    moved[name] += 1

    require(n_problems > 0, "no sampled problem carries a per-cell block")
    for name in names:
        require(compared[name] > 0, f"nothing comparable against the {name}")
    return {
        "K": K, "n_problems": n_problems, "names": names,
        "rate": {n: moved[n] / compared[n] for n in names},
        "moved": moved, "compared": compared, "absent_ref": absent_ref,
    }


def figures(m=None, c=None):
    # The two measurements are passed in when `main` has already run them, so a
    # standalone run can print exactly the values the ledger would receive
    # without paying for either of them twice.
    m = measure() if m is None else m
    c = cross_projection() if c is None else c
    r = m["rank"]
    pop, shipped = r["arms"]["population mean"], r["arms"]["shipped null"]
    return [
        Figure("motif.null_axes_measured", m["axes"], "d",
               "shipped axes the availability-null measurement runs over"),
        Figure("motif.null_sign_reversals", m["reversals"], "d",
               "axes whose population error lean changes sign between the fixed even "
               "split and the shipped visits-linear availability null"),
        Figure("motif.menu_worst_toward_pct", m["menu_worst"], "%.0f",
               f"toward-side share of offered-alternative weight on the most "
               f"menu-skewed axis ({m['worst_axis']})"),
        Figure("motif.fixed_null_worst_flag_pct", m["fixed_flag_share"], "%.0f",
               "share of testable users the retired fixed-50% test flags on that axis"),
        Figure("motif.shipped_null_worst_flag_pct", m["shipped_flag_share"], "%.0f",
               "share of testable users the shipped null's test flags on that axis"),
        Figure("motif.depth_null_axes_better", m["depth_axes_better"], "d",
               "axes where the shipped visits-weighted null beats the equal-count "
               "one on per-error directional log-loss with the paired 95% bootstrap "
               f"interval clear of zero, of {m['axes']} measured"),
        Figure("motif.depth_null_pooled_gain", m["depth_gain"], ".4f",
               "nats of directional log-loss per scored error the visits-weighted "
               "null saves over the equal-count one, pooled over axes"),
        Figure("motif.fitted_null_gain_share_pct", m["fitted_share"], "%.0f",
               "what the fitted null adds over the shipped visits-weighted one, "
               "pooled, as a share of that null's own gain over equal counting"),
        Figure("motif.tendency_rank_users", pop["share_users"], "d",
               f"accounts with a fitted ability and at least {r['min_errors']} "
               f"scored errors on the axis the share below is measured on"),
        Figure("motif.tendency_rank_share_population_pct", pop["share_max"], "%.0f",
               "largest per-axis share of the between-user variance the user's rank "
               "alone explains when the lean is scored against a population average "
               f"instead of each problem's own null ({pop['share_axis']})"),
        Figure("motif.tendency_rank_share_shipped_pct", shipped["share_max"], "%.0f",
               "the same share for the reading that ships, scored against each "
               f"problem's own availability null ({shipped['share_axis']})"),
        Figure("motif.altstat_crossproj_setwide_pct",
               c["rate"]["set-wide sidecar"], "%.1f",
               f"share of a {c['n_problems']}-problem seeded sample whose published "
               "availability masses move when the reference is taken from the "
               "set-wide correct-move sidecar instead of the problem's own per-cell "
               "block"),
        Figure("motif.altstat_crossproj_manifest_pct",
               c["rate"]["shard manifest"], "%.1f",
               "the same rate against the shard manifest's copy of that vector"),
    ]


def main():
    m = measure()
    r = m["rank"]
    print("\n" + "=" * 78)
    print(f"axes measured                     : {m['axes']}")
    print(f"population-lean sign reversals    : {m['reversals']}   "
          f"(fixed 0.5 vs the shipped {SHIPPED_TAG} null)")
    print(f"most menu-skewed axis             : {m['worst_axis']} "
          f"(menu toward-share {100 * m['menu_worst']:.1f}%)")
    print(f"  flagged there by fixed 0.5      : {100 * m['fixed_flag_share']:.1f}% of testable users")
    print(f"  flagged there by the shipped null: {100 * m['shipped_flag_share']:.1f}%")

    print("\n--- adjudication (phase 2b, directional log-loss per scored error) ---")
    lo, hi = m["depth_gain_ci"]
    print(f"{'axes won by ' + SHIPPED_TAG + ', interval clear of 0':<44}: "
          f"{m['depth_axes_better']} of {m['axes']}")
    print(f"{'its pooled gain over equal counting':<44}: "
          f"{m['depth_gain']:.4f} nats [{lo:+.4f}, {hi:+.4f}] "
          f"over {int(m['scored_errors']):,} scored errors")
    print(f"{'what the fitted ' + FITTED_TAG + ' arm adds on top':<44}: "
          f"{m['fitted_gain']:.4f} nats = {100 * m['fitted_share']:.1f}% of it")

    print("\n--- rank's share of the between-user spread in the lean reading ---")
    print(f"accounts with a recorded error but no fitted ability: {r['n_no_ability']:,} "
          f"(dropped; the rest carry both)")
    print(f"{'axis':<18}{'users':>7}{'R2 shipped null':>18}{'R2 population':>16}")
    print("-" * 59)
    ship, pop = r["arms"]["shipped null"], r["arms"]["population mean"]
    for k, label in enumerate(m["labels"]):
        print(f"{label[:17]:<18}{ship['users'][k]:>7,}"
              f"{ship['r2'][k]:>18.3f}{pop['r2'][k]:>16.3f}")
    print(f"\nR2 = simple linear regression of one per-user reading on the user's "
          f"fitted\nability carried through the shipped rating->rank curve, over "
          f"the accounts with\n>= {r['min_errors']} scored errors on that axis. "
          f"'shipped null' is the excess toward-share\nagainst each problem's own "
          f"availability null, the quantity the client reports;\n'population' is "
          f"the same user's raw toward-share against the average over\nusers -- "
          f"the standing alternative, exactly rather than as a stand-in, since "
          f"the\nconstant it subtracts cannot move a regression.")

    c = cross_projection()
    print(f"\n--- availability-null reference: the cost of the wrong projection ---")
    print(f"seeded sample                     : {c['n_problems']:,} problems, "
          f"{c['K']} axes, seed {CROSSPROJ_SEED}")
    for name in c["names"]:
        extra = (f", {c['absent_ref'][name]:,} with no vector there"
                 if c["absent_ref"][name] else "")
        print(f"  reference from the {name:<17}: "
              f"{100 * c['rate'][name]:.1f}% of problems move a mass "
              f"({c['moved'][name]:,} of {c['compared'][name]:,}{extra})")
    print("A problem 'moves' when any of its 2K published masses differs from the "
          "one\npackaging ships, which references the problem's own per-cell "
          "block row 0.")

    print("\n--- the figures this run would put in the ledger ---")
    for fig in figures(m, c):
        print(f"  {fig.key:<40}{fig.text:>8}")


if __name__ == "__main__":
    main()
