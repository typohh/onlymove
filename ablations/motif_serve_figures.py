"""Does a motif flag find problems the flagged player actually does WORSE on?

    ./.venv/bin/python3 -m ablations.motif_serve_figures

The client can turn four per-motif readings into a serve direction: `weak` and
`strong` from the accuracy reading, `overplay` and `underplay` from the lean of
the errors. Only the first two are acted on (`pwa/js/weakness.js`). This is the
measurement behind that split, on the attempt log, OUT OF SAMPLE in time:

  flags    for every account and every position t in its history, the readings
           the client would have held BEFORE attempt t — the shipped readings,
           run unmodified under node over rows[0, t) (`motif_serve_flags.mjs`),
           turned into (axis, serve direction, kind) for all four kinds.
  score    how strongly attempt t's problem matches a flag: the corpus quantile
           of its correct-move loading in the flag's serve direction, among the
           published problems of that size. The motif slot serves the strongest
           matches in its pool, so the top of this scale is what a flag selects.
  outcome  y - p in percentage points, p from the RATING-ONLY curve at the
           shipped rating/slope/floor and the backend ability, on first-time
           attempts on the curve's slope. Negative = did worse than the rating
           predicts.

The figure per kind is the GAP: mean outcome on the strongest tenth of matches
minus mean outcome on the weaker half, with a user-clustered bootstrap interval.
A flag that finds a soft spot has a negative gap. The ability is in-sample (it
was fitted on these attempts), which shrinks every gap alike.
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations import motif_findings_figures as mf  # noqa: E402
from ablations._figures import Figure, require  # noqa: E402
from utils.constants import read_ignored_users  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import clear_logit_vec  # noqa: E402

TIER = "methodology"

HELPER = os.path.join(PROJECT_ROOT, "ablations", "motif_serve_flags.mjs")
RATINGS = os.path.join(PROJECT_ROOT, "ratings")
KINDS = ("weak", "strong", "overplay", "underplay")
SLOPE_BAND = (0.15, 0.85)
TOP_Q = 0.90          # the strongest tenth of matches
BOTTOM_Q = 0.50       # the weaker half
N_BOOT = 2000
SEED = 0


def history_ids(pub):
    """(problem id, timestamp) of every row `mf.load_histories` builds, in its
    order — the same query and filters, so the two line up row for row."""
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        cur = con.execute(
            "SELECT user_id, problem, created_at FROM tsumego_reports "
            " WHERE is_repeat = 0 ORDER BY user_id, created_at, id")
        by_user = {}
        for uid, tid, ts in cur:
            if str(tid) in pub["pos"]:
                by_user.setdefault(uid, []).append((str(tid), ts))
    finally:
        con.close()
    return {u: r for u, r in by_user.items() if len(r) >= mf.MIN_HISTORY}


def motif_flags(pub, histories, cache):
    """{user: [per position: [[axis, serve sign, kind], ...]]} from the node helper.
    `cache` keeps the helper's output so a re-score need not re-run it."""
    path = os.path.join(cache, "motif_flags.json")
    if os.path.exists(path):
        print(f"  motif flags from {path}", flush=True)
        with open(path) as f:
            return json.load(f)
    with open(os.path.join(mf.TSUMEGOS, "embedding_meta.json")) as f:
        meta = json.load(f)
    payload = os.path.join(cache, "motif_payload.json")
    with open(payload, "w") as f:
        json.dump({"pwa_root": mf.PWA_ROOT + os.sep, "K": pub["K"],
                   "labels": meta["labels"]["components"],
                   "users": [{"id": u, "rows": r} for u, r in histories.items()]}, f)
    print("  running the shipped motif readings at every position (node) ...", flush=True)
    subprocess.run(["node", HELPER, payload, path], cwd=PROJECT_ROOT, check=True)
    with open(path) as f:
        return json.load(f)


def load_inputs(cache):
    pub = mf.load_published()
    histories = mf.load_histories(pub)
    ids = history_ids(pub)
    require(set(ids) == set(histories)
            and all(len(ids[u]) == len(histories[u]) for u in ids),
            "history ids do not line up with motif_findings_figures.load_histories")
    flags = motif_flags(pub, histories, cache)
    require(set(flags) == set(histories), "the node helper did not answer every account")
    with open(os.path.join(RATINGS, "rating_constants.json")) as f:
        lam = json.load(f)["fat_finger_rate"]
    theta = pd.read_csv(os.path.join(RATINGS, "user_theta.csv"))
    theta = dict(zip(theta["profile"].astype(str), theta["theta"]))
    return pub, histories, ids, flags, lam, theta


def iter_users(pub, histories, ids, flags, lam, theta, slope_band=SLOPE_BAND, since=None):
    """Per scored account: (uid, tids, flags, size, y, p, use) — `use` marks the
    first-time attempts on the slope, the only ones an outcome is read from."""
    ignored = set(read_ignored_users())
    lo, hi = slope_band
    for uid, hist in histories.items():
        if uid in ignored or uid not in theta:
            continue
        tids = ids[uid]
        n = len(hist)
        size = np.array([h[0] for h in hist])
        y = np.array([h[1] for h in hist], float)
        r_, a_, c_ = (np.array([h[i] for h in hist], float) for i in (2, 3, 4))
        b_ = r_ - clear_logit_vec(c_, lam) / a_
        sg = 1 / (1 + np.exp(-a_ * (theta[uid] - b_)))
        p = c_ + (1 - c_ - lam) * sg
        seen, first = set(), np.zeros(n, bool)
        for t, tid in enumerate(tids):
            first[t] = tid[0] not in seen
            seen.add(tid[0])
        use = first & (sg >= lo) & (sg <= hi)
        if since:
            use &= np.array([ts >= since for _, ts in tids])
        yield uid, tids, flags[uid], size, y, p, use


def motif_quantiler(pub):
    """q(size, axis, sign, loading): the share of that size's published problems
    whose loading in the serve direction is BELOW this one."""
    size_of = np.array([s for _, _, s in pub["rows"]])
    has_vec = np.any(pub["emb"] != 0, axis=1)
    srt = {(s, k): np.sort(pub["emb"][(size_of == s) & has_vec, k])
           for s in (9, 13) for k in range(pub["K"])}

    def q(s, k, sign, x):
        arr = srt[(s, k)]
        if sign > 0:
            return np.searchsorted(arr, x, side="left") / len(arr)
        return 1.0 - np.searchsorted(arr, x, side="right") / len(arr)
    return q


def motif_rows(pub, scored_users):
    """One row per (scored attempt, flag held before it): user, kind, axis,
    outcome in points, match quantile. Plus one row per scored attempt."""
    q = motif_quantiler(pub)
    rows, base = [], []
    for uid, tids, fl, size, y, p, use in scored_users:
        for t in np.where(use)[0]:
            base.append((uid, 100 * (y[t] - p[t])))
            emb = pub["emb"][pub["pos"][tids[t][0]]]
            if not np.any(emb):
                continue
            for k, sign, kind in fl[t]:
                rows.append((uid, kind, k, 100 * (y[t] - p[t]), q(size[t], k, sign, emb[k])))
    return (pd.DataFrame(rows, columns=["user", "kind", "k", "d", "q"]),
            pd.DataFrame(base, columns=["user", "d"]))


def gap(df, rng, n=N_BOOT):
    """(top mean, bottom mean, gap, lo, hi): strongest tenth of matches minus
    the weaker half, resampling USERS for the interval."""
    top, bot = df[df["q"] >= TOP_Q], df[df["q"] < BOTTOM_Q]
    require(len(top) > 0 and len(bot) > 0, "a kind has no rows in one of the two match bands")
    users = np.unique(df["user"])
    idx = {u: i for i, u in enumerate(users)}

    def sums(part):
        i = part["user"].map(idx).to_numpy()
        return (np.bincount(i, part["d"], minlength=len(users)),
                np.bincount(i, minlength=len(users)).astype(float))
    ts, tc = sums(top)
    bs, bc = sums(bot)
    W = rng.multinomial(len(users), np.full(len(users), 1 / len(users)), size=n)
    b = (W @ ts) / np.maximum(W @ tc, 1) - (W @ bs) / np.maximum(W @ bc, 1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return top["d"].mean(), bot["d"].mean(), top["d"].mean() - bot["d"].mean(), lo, hi


def measure(cache=None):
    rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        pub, histories, ids, flags, lam, theta = load_inputs(cache or tmp)
    M, A = motif_rows(pub, iter_users(pub, histories, ids, flags, lam, theta))
    require(len(A) > 0, "no scored attempts")
    out = {"users": A["user"].nunique(), "attempts": len(A), "base": A["d"].mean(), "kinds": {}}
    for kind in KINDS:
        d = M[M["kind"] == kind]
        require(len(d) > 0, f"no account ever held a {kind} flag")
        top, bot, g, lo, hi = gap(d, rng)
        out["kinds"][kind] = {"users": d["user"].nunique(), "rows_top": int((d["q"] >= TOP_Q).sum()),
                              "top": top, "bottom": bot, "gap": g, "lo": lo, "hi": hi}
    return out


def figures():
    m = measure()
    figs = [
        Figure("serve.users", m["users"], ",d",
               "accounts scored: enough first-time attempts for the readings, and a backend ability"),
        Figure("serve.attempts", m["attempts"], ",d",
               "first-time attempts on the curve's slope that an outcome is read from"),
        Figure("serve.base_points", m["base"], "+.1f",
               "mean outcome over all of them, success minus the rating's prediction, in "
               "percentage points"),
        Figure("serve.top_share_pct", 1 - TOP_Q, "%.0f",
               "share of the corpus counted as a flag's strongest matches"),
    ]
    said = {"weak": "worse than the rating predicts on the motif's problems",
            "strong": "better than the rating predicts on the motif's problems",
            "overplay": "errors lean toward the motif (serves problems loading away from it)",
            "underplay": "errors lean away from the motif (would serve problems loading toward it)"}
    for kind in KINDS:
        k = m["kinds"][kind]
        figs += [
            Figure(f"serve.{kind}_users", k["users"], ",d",
                   f"accounts that ever held a `{kind}` flag — {said[kind]}"),
            Figure(f"serve.{kind}_top_points", k["top"], "+.1f",
                   f"mean outcome on the strongest tenth of matches to a `{kind}` flag held "
                   f"before the attempt, in percentage points"),
            Figure(f"serve.{kind}_bottom_points", k["bottom"], "+.1f",
                   "the same on the weaker half of matches"),
            Figure(f"serve.{kind}_gap_points", k["gap"], "+.1f",
                   "strongest tenth minus weaker half: negative = the flag finds problems the "
                   "player does worse on"),
            Figure(f"serve.{kind}_gap_lo", k["lo"], "+.1f",
                   "lower end of the gap's user-clustered bootstrap 95% interval"),
            Figure(f"serve.{kind}_gap_hi", k["hi"], "+.1f", "upper end of that interval"),
            Figure(f"serve.{kind}_finds_deficit", int(k["hi"] < 0), "d",
                   "1 when the whole interval is below zero"),
        ]
    return figs


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=None,
                    help="directory to keep the node helper's flags in (reused when present)")
    args = ap.parse_args()
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)
    m = measure(args.cache)
    print(f"\n{m['users']} accounts | {m['attempts']} first-time attempts on the slope | "
          f"mean outcome {m['base']:+.2f} points (success minus the rating's prediction)")
    print(f"\noutcome on problems matching a flag held BEFORE the attempt; top = strongest "
          f"{1 - TOP_Q:.0%} of matches, bottom = weaker {BOTTOM_Q:.0%}; gap < 0 = the flag "
          f"finds problems the player does worse on")
    print(f"{'kind':>10} {'accounts':>9} {'top rows':>9} {'top':>7} {'bottom':>7} {'gap':>7} {'95% interval':>17}")
    for kind in KINDS:
        k = m["kinds"][kind]
        print(f"{kind:>10} {k['users']:>9} {k['rows_top']:>9} {k['top']:>+7.2f} {k['bottom']:>+7.2f} "
              f"{k['gap']:>+7.2f} [{k['lo']:+6.2f},{k['hi']:+6.2f}]")


if __name__ == "__main__":
    main()
