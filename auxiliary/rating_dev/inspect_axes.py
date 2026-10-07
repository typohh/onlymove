#!/usr/bin/env python3
"""Inspect the extreme-loading problems on the shipped SKILL AXES.

Run with no arguments to render the current axes/per_problem.csv — the axis set
pipeline.train_mirt fitted and package_for_pwa will publish. That is the
labelling workflow: an axis ships and works unnamed, the client just stays
silent about it, so this page is what a name gets chosen from.

Each axis shows BOTH tails, because the axes are bipolar: they carry no skew
sign-canonicalization, so a negative loading is a real direction — problems a
user with positive axis position does WORSE on — rather than baseline. A tail
may legitimately end up named in labels.json while its opposite stays null.

THE BOARDS ARE THE WHOLE EVIDENCE, and that is a property of the model rather
than a gap in this tool. Naming an axis by what DRIVES it needs the axis to be
a linear map on nameable inputs, so that a problem's value decomposes exactly
into per-column contributions. The MIRT puts a trunk between the features and
the loadings, so no such decomposition exists — any per-input number here would
be a surrogate model's opinion, presented in the one place a human is deciding
what a motif IS.

Board rendering (parse / simulate / SVG) lives in utils/katago/board_render.py —
the single shared copy used by every inspection tool.

Usage:
  python -m auxiliary.rating_dev.inspect_residual_axes            # the shipped axes
  python -m auxiliary.rating_dev.inspect_residual_axes --open
  python -m auxiliary.rating_dev.inspect_residual_axes --size 13
"""

import argparse
import html
import json
import os
import sys
from functools import lru_cache

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.katago.board_render import load_lookup, render_problem  # noqa: E402
from utils.rating.rank_surveys import value_to_rank  # noqa: E402
from utils.rating.ratings_io import interp_extrap_vec as interp_extrap  # noqa: E402
from utils.rating.tail_agreement import TAIL_HINGE, hinge_z, hinged_r  # noqa: E402


# ---- artifact loading ---------------------------------------------------

def load_dump_npz(path, bank_path=None):
    """Components straight from a training dump's `loading` array.

    Identical content to a per_problem.csv — the shipped CSV is that same array
    rounded — so both paths feed the renderer the same thing and an arm's axes
    are read exactly as the shipped ones are. The axis INDEX is not meaningful
    across fits: nothing pins axis k of one run to axis k of another unless both
    were taught from the same bank column order, so two dumps are compared by
    looking at the boards, never by index.
    """
    z = np.load(path, allow_pickle=False)
    for need in ("tid", "loading"):
        if need not in z.files:
            raise SystemExit(f"{path} has no {need!r} array — is it a "
                             f"--dump-params output? (has: {', '.join(z.files)})")
    M = np.asarray(z["loading"], dtype=np.float64)
    tids = [str(t) for t in z["tid"]]
    labels = [None] * M.shape[1]
    # A caption carried INSIDE the dump is whatever its writer chose to say about
    # the axis — an alignment note, a candidate name — and is shown as is; only
    # a bank supplies the "taught from" claim, since only a bank taught anything.
    caption = " — {}"
    if "labels" in z.files:
        labels = [str(x) for x in z["labels"]]
    elif bank_path:
        caption = " — taught from {!r}"
        b = np.load(bank_path, allow_pickle=False)
        if "labels" not in b.files:
            raise SystemExit(f"{bank_path} carries no 'labels' array")
        bl = [str(x) for x in b["labels"]]
        if len(bl) < M.shape[1]:
            raise SystemExit(
                f"bank names {len(bl)} axes, the dump has {M.shape[1]} — these "
                f"are not the same run, and titling axis k from a different "
                f"bank's column k would attach every name to the wrong tail.")
        labels = bl[:M.shape[1]]
    components = []
    for k in range(M.shape[1]):
        name = labels[k] if k < len(labels) and labels[k] else None
        # The RAW label rides along as a fourth field. It is "A/B" for a bipolar
        # column, and each tail is named by its own half — folding it into the
        # title alone would leave the positive tail of "life and death/territory"
        # captioned with the territory name too.
        components.append((f"ax{k}",
                           f"axis {k}" + (caption.format(name) if name else ""),
                           M[:, k], name))
    # A dump carries `size` when the fit wrote one; absent, the caller falls back
    # to pwa/lookup.txt purely to learn which board each problem is on.
    sizes = ({str(t): int(v) for t, v in zip(z["tid"], z["size"])}
             if "size" in z.files else {})
    return tids, components, sizes


def load_per_problem_csv(path, labels_path=None):
    """Production-axes mode: components straight from pipeline.build_axes'
    axes/per_problem.csv (one fit, canonical orientation — the
    shipped loadings, exactly what labels.json names)."""
    import csv as _csv
    tids, rows, sizes = [], [], {}
    with open(path, newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        cols = [i for i, c in enumerate(header) if c.startswith("embed_")]
        # The CSV states each problem's board, so the board never has to be looked
        # up in packaging output. See the lookup note in main().
        size_col = header.index("size") if "size" in header else None
        for row in reader:
            tids.append(row[0])
            rows.append([float(row[i]) for i in cols])
            if size_col is not None:
                sizes[row[0]] = int(row[size_col])
    M = np.array(rows, dtype=np.float64)
    # Names come from axes/labels.json's `components`, which is the shipping
    # decision this CSV was produced under — so the standard no-argument run
    # captions each tail with the motif it was taught, exactly as a --bank run
    # does. Absent or short, the axes stay unnamed rather than mislabelled.
    # An ablation arm taught from a dev bank passes that bank's labels.json, so
    # its extra axes are captioned with what they were taught rather than blank.
    labels = []
    lp = labels_path or os.path.join(PROJECT_ROOT, "axes", "labels.json")
    if os.path.exists(lp):
        labels = [str(x) for x in json.load(open(lp)).get("components", [])]
    components = []
    for k in range(M.shape[1]):
        name = labels[k] if k < len(labels) and labels[k] else None
        components.append((f"ax{k}",
                           f"axis {k} (production)"
                           + (f" — taught from {name!r}" if name else ""),
                           M[:, k], name))
    return tids, components, sizes


# ---- HTML page generation ----------------------------------------------

@lru_cache(maxsize=None)
def _cached_render(tid, zip_id, size):
    """Memoize SVG rendering: a problem can land in several components' bands
    on the one-page layout, so render each board at most once."""
    return render_problem(tid, zip_id, size)


def _build_card(tid, score, lookup, rank=None):
    info = lookup.get(tid)
    if not info:
        return f'<div class="card missing">missing in lookup: {tid}</div>'
    zip_id, size = info
    try:
        rp = _cached_render(tid, zip_id, size)
    except (FileNotFoundError, KeyError) as e:
        return f'<div class="card missing">load failed: {tid} ({e})</div>'
    rank_html = "" if rank is None else f'<span class="rank">{value_to_rank(rank)}</span>'
    return (
        '<div class="card">'
        f'{rp["svg"]}'
        f'<div class="meta">'
        f'<span class="score">{score:+.3f}</span>'
        f'<span class="tid">{tid}</span>'
        f'{rank_html}'
        '</div>'
        '</div>'
    )


def load_named_bank_tails(tids, hinge_at=TAIL_HINGE):
    """Every NAMED tail in the motif store as a hinged weight vector over `tids`.

    Restricted to the problems this page is showing, so a match is quoted on the
    same population the boards come from — a rank-capped or size-filtered page
    would otherwise report agreement against a corpus the reader cannot see.

    Unnamed poles are skipped: they make no claim to match, being where
    everything not at the named end accumulates.
    """
    anchors = os.path.join(PROJECT_ROOT, "axes", "label_anchors.npz")
    labels_path = os.path.join(PROJECT_ROOT, "axes", "labels.json")
    if not (os.path.exists(anchors) and os.path.exists(labels_path)):
        return []
    z = np.load(anchors, allow_pickle=False)
    bank = json.load(open(labels_path)).get("bank", {})
    pos = {t: i for i, t in enumerate(tids)}
    idx = np.array([pos.get(str(t), -1) for t in z["tids"]])
    keep = idx >= 0
    out = []
    for j, aid in enumerate(z["ids"]):
        e = bank.get(str(aid), {})
        col = np.full(len(tids), np.nan)
        col[idx[keep]] = np.asarray(z["M"][j], np.float64)[keep]
        for sign, lab in ((+1, e.get("positive")), (-1, e.get("negative"))):
            if lab:
                out.append((f"{lab}{'+' if sign > 0 else '-'}",
                            hinge_z(sign * col, hinge_at)))
    return out


def load_taught_axis_tails(tids, hinge_at=TAIL_HINGE, per_problem=None):
    """Every NAMED tail of the shipped, rule-taught axes as a hinged weight
    vector over `tids` — the reference a free fit's tail is read against when
    the question is "which of the rule-based axes does this look like".

    The axes come from the production `axes/per_problem.csv` and their names
    from `axes/labels.json`'s `components`, exactly as the standard run captions
    them: a bare rule names the positive tail, `A/B` names both. An unnamed
    tail is where everything not at the named end accumulates and makes no
    claim, so it is skipped. Restricted to the problems this page shows, for the
    same reason the store tails are.
    """
    path = per_problem or os.path.join(PROJECT_ROOT, "axes", "per_problem.csv")
    labels_path = os.path.join(PROJECT_ROOT, "axes", "labels.json")
    if not (os.path.exists(path) and os.path.exists(labels_path)):
        return []
    t_tids, comps, _sizes = load_per_problem_csv(path)
    pos = {t: i for i, t in enumerate(tids)}
    idx = np.array([pos.get(str(t), -1) for t in t_tids])
    keep = idx >= 0
    out = []
    for _key, _title, vals, name in comps:
        if not name:
            continue
        halves = [h.strip() for h in name.split("/")]
        col = np.full(len(tids), np.nan)
        col[idx[keep]] = np.asarray(vals, np.float64)[keep]
        for sign, h in ((+1, 0), (-1, 1)):
            lab = halves[h] if h < len(halves) and halves[h] else None
            if lab:
                # Named by the RULE alone: the rule is the tail's identity, and
                # the shipped fit's axis index means nothing on another fit's page.
                out.append((f"{lab}{'+' if sign > 0 else '-'}",
                            hinge_z(sign * col, hinge_at)))
    return out


def _match_html(vals, sign, bank_tails, hinge_at, min_r, source="store"):
    """Which named stored motifs agree with this tail, by hinged correlation.

    Both sides are z-scored and floored at the mean, then correlated over the
    problems they share. Independence gives exactly 0 — hinging two independent
    variables leaves them independent — so there is no chance baseline to quote
    and no N to choose, and the score is driven by how far into each tail a
    problem sits rather than by whether it cleared a cutoff.

    Agreement is CO-SELECTION and has two causes it cannot separate — one
    direction under two names, or two motifs whose Go situations co-occur — so
    this is a list of leads, not identifications. It sits above the boards
    because the boards are what decide between the two.
    """
    if not bank_tails:
        return ""
    mine = hinge_z(sign * np.asarray(vals, np.float64), hinge_at)
    hits = []
    for lab, other in bank_tails:
        r = hinged_r(mine, other, pre_hinged=True)
        if np.isfinite(r) and r >= min_r:
            hits.append((r, lab))
    what = ("named tail of the rule-taught axes" if source == "taught"
            else "named motif in the bank")
    if not hits:
        return (f'<div class="matches none">no {what} agrees '
                f'with this tail above r = {min_r:g}</div>')
    body = " · ".join(
        f'<span class="match"><b>{html.escape(lab)}</b> '
        f'<span class="lift">{r:.2f}</span></span>'
        for r, lab in sorted(hits, reverse=True))
    return (f'<div class="matches">bank agreement (hinged r, 0 = independent): '
            f'{body}</div>')


def _pick_extremes(vals, rows, cols):
    """The `rows*cols` strongest problems on one tail, strongest first.

    Kept for --by-signal. The default is rank-bucketed (below), because a tail's
    strongest problems cluster at one end of the difficulty range and reading only
    those tells you what the motif looks like THERE, not what it is.
    """
    n = len(vals)
    take = min(rows * cols, n)
    if take <= 0:
        return []
    idx = np.argpartition(-vals, take - 1)[:take]
    idx = idx[np.argsort(vals[idx])[::-1]]
    return [(k + 1, idx[k:k + cols]) for k in range(0, take, cols)]


def _pick_rank_buckets(vals, rank_of, rows, cols):
    """The strongest problem from each of `rows*cols` EQUAL-COUNT rank buckets,
    in rank order.

    An axis's loading is not independent of difficulty even after the
    orthogonality penalty, so the top of a tail is drawn from a narrow slice of
    the ladder — read that alone and a motif looks like whatever it looks like at
    one strength. One problem per rank bucket shows the same motif across the
    whole range, which is what a name has to hold over.

    EQUAL COUNT rather than equal width, so every bucket is backed by the same
    number of problems: rank density is far from uniform, and equal-width buckets
    would put a handful of problems in the tails of the ladder beside thousands in
    the middle, then present the winner of each as equally representative.
    """
    take = rows * cols
    order = np.argsort(rank_of)
    picks = []
    for b in np.array_split(order, take):
        if len(b):
            picks.append(int(b[np.argmax(vals[b])]))
    return picks


def _render_tail_html(tids, vals, sign, rows, cols, lookup, ranks=None,
                     by_signal=False):
    """One tail's problems. sign=+1 renders the positive tail, -1 the negative
    (card scores are shown in ORIGINAL units either way)."""
    v = sign * vals
    rank_of = None
    if ranks is not None and not by_signal:
        rank_of = np.array([ranks.get(t, np.nan) for t in tids])
        if not np.isfinite(rank_of).all():
            rank_of = None      # a partial rank map cannot bucket; fall back
    if rank_of is None:
        groups = _pick_extremes(v, rows, cols)
        band = lambda lo, idx: f"#{lo}–{lo + len(idx) - 1}"
    else:
        picks = _pick_rank_buckets(v, rank_of, rows, cols)
        groups = [(k + 1, np.array(picks[k:k + cols]))
                  for k in range(0, len(picks), cols)]
        def band(lo, idx):
            r = rank_of[idx]
            return f"{value_to_rank(r.min())} – {value_to_rank(r.max())}"
    if not groups:
        return '<div class="band"><div class="card missing">no problems on this tail</div></div>'
    out = []
    for lo, idx in groups:
        cards = "".join(
            _build_card(tids[i], float(vals[i]), lookup,
                        None if ranks is None else ranks.get(tids[i]))
            for i in idx)
        out.append(
            f'<div class="band">'
            f'<div class="band-label">{band(lo, idx)}</div>'
            f'<div class="grid">{cards}</div>'
            f'</div>'
        )
    return "".join(out)


def _style_block(cols):
    return f"""<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background:#fafafa; color:#222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  h2 {{ margin-top: 1.5rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; }}
  h3 {{ margin-top: 1.25rem; color: #444; }}
  .grid {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 1rem; }}
  .band {{ margin: 0.75rem 0 1.25rem; }}
  .band-label {{
    font-size: 0.8rem; color: #555; margin-bottom: 0.35rem;
    font-family: ui-monospace, monospace;
  }}
  .card {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 0.5rem; text-align: center; }}
  .card svg {{ width: 100%; height: auto; display: block; }}
  .meta {{ font-size: 0.8rem; margin-top: 0.4rem; display: flex;
           justify-content: space-between; align-items: baseline; gap: 0.4rem; }}
  .score {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
  .tid {{ font-family: ui-monospace, monospace; color: #555; }}
  .rank {{ color: #2a6; font-weight: 600; }}
  .matches {{ font-size: 0.9rem; background:#eef4ff; border:1px solid #cfe0ff;
              border-radius:6px; padding:0.5rem 0.7rem; margin:0.5rem 0 1rem; }}
  .matches.none {{ background:#f4f4f4; border-color:#ddd; color:#777; }}
  .match {{ white-space: nowrap; }}
  .lift {{ color:#777; }}
  .missing {{ color: #c33; padding: 1rem; }}
  .legend {{ font-size: 0.85rem; color: #555; margin-bottom: 1rem; }}
  .legend .marker {{
    display: inline-block; width: 14px; height: 14px; border-radius: 50%;
    border: 2px solid #e63946; vertical-align: middle; margin: 0 0.2rem;
  }}
  .controls {{
    position: sticky; top: 0; background: #fafafa; padding: 0.6rem 0;
    margin-bottom: 0.5rem; border-bottom: 1px solid #ddd; z-index: 10;
  }}
  .controls select {{ font-size: 1rem; padding: 0.25rem 0.4rem; }}
  .comp-label {{ color: #1d6fa5; font-weight: 600; }}
  .comp-section {{ display: none; }}
  .comp-section.active {{ display: block; }}
</style>"""


def _build_page(source, options_html, sections_html, cols):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Residual axes extremes ({html.escape(source)})</title>
{_style_block(cols)}
</head>
<body>
<h1>Residual-MIRT axes — both tails</h1>
<div class="legend">
  Source: {html.escape(source)} &middot; Red ring <span class="marker"></span> = correct first move
  &middot; Each tail shows its strongest-loading problems, most extreme first.
  Both tails shown — residual axes have no skew sign-canonicalization, so the
  negative side is a real direction. Folds are separate fits with arbitrary
  axis rotation; never compare fold 0 axis k to fold 1 axis k by index.
</div>
<div class="controls">
  <label>Component: <select id="compSelect">{options_html}</select></label>
</div>
{sections_html}
<script>
  const sel = document.getElementById('compSelect');
  function showComp(key) {{
    document.querySelectorAll('.comp-section').forEach(s => {{
      s.classList.toggle('active', s.dataset.comp === key);
    }});
  }}
  sel.addEventListener('change', () => {{ showComp(sel.value); window.scrollTo(0, 0); }});
  showComp(sel.value);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-problem",
                        default="axes/per_problem.csv",
                        help="path to the shipped per_problem.csv (default: "
                             "axes/per_problem.csv)")
    parser.add_argument("--labels", default=None,
                        help="the labels.json the --per-problem fit was taught "
                             "under, for an ablation arm taught from a dev bank "
                             "(ablation_mirt --bank-labels); default axes/labels.json")
    parser.add_argument("--dump", default=None,
                        help="inspect a --dump-params npz from "
                             "auxiliary.rating_dev.train_joint_mirt instead of a "
                             "per_problem.csv. An experiment arm never writes "
                             "axes/, so without this its axes can only be looked "
                             "at by first overwriting the shipped fit's output — "
                             "which is the one thing a dev surface must not make "
                             "anyone do to see its own result.")
    parser.add_argument("--bank", default=None,
                        help="the axis bank the run was taught from, so each axis "
                             "is titled by the motif it was seeded with. A dump "
                             "carries no names — the bank holds them, and column "
                             "k of the bank is axis k of the fit — so without "
                             "this the page shows a taught axis and an untaught "
                             "one identically, which is the one distinction the "
                             "page exists to let you judge.")
    parser.add_argument("--n", type=int, default=10,
                        help="N top problems per tail in --no-html mode (default 10)")
    parser.add_argument("--host", default="http://localhost:8787",
                        help="PWA dev server origin (only used with --no-html)")
    parser.add_argument("--size", default="all",
                        help="9, 13, or 'all' (default; HTML mode defaults to 9 so grids "
                             "stay board-comparable)")
    parser.add_argument("--html", action=argparse.BooleanOptionalAction, default=True,
                        help="Render static HTML (default). --no-html prints PWA URLs instead.")
    parser.add_argument("--rows", type=int, default=8,
                        help="rows of extremes per tail; rows x cols problems shown, strongest first (default 8)")
    parser.add_argument("--cols", type=int, default=6, help="HTML grid columns (default 6)")
    parser.add_argument("--match-hinge", type=float, default=TAIL_HINGE, metavar="Z",
                        help="z threshold the tail weight is floored at for the "
                             "bank-agreement summary (default 0.0 = the mean). "
                             "Raise it to score agreement on a narrower tail.")
    parser.add_argument("--match-source", choices=("taught", "store", "none"),
                        default=None,
                        help="what each tail's agreement line is read against: "
                             "`taught` — the named tails of the shipped, "
                             "rule-taught axes (axes/per_problem.csv as "
                             "labels.json names them), the reference for "
                             "asking which rule-based axis a free direction "
                             "resembles; `store` — the named directions in the "
                             "motif store, the reference for asking whether a "
                             "fit retained a stored motif. Default: taught for "
                             "a --dump, store for the shipped axes themselves "
                             "(matching them against themselves says nothing)")
    parser.add_argument("--match-min-r", type=float, default=0.10,
                        help="minimum hinged correlation for a bank motif to be "
                             "listed (default 0.10; 0 = independent)")
    parser.add_argument("--by-signal", action="store_true",
                        help="show the strongest problems outright instead of the "
                             "strongest from each rank bucket. The bucketed "
                             "default exists because a tail's extremes cluster at "
                             "one end of the ladder, so reading them alone shows "
                             "the motif at one strength rather than across the "
                             "range a name has to hold over.")
    parser.add_argument("--rank-max", type=float, default=None, metavar="RANK",
                        help="show only problems at or below this internal rank "
                             "(0 = the 1k/1d boundary, so 0 caps at 1d). DISPLAY "
                             "ONLY — an axis is unbounded and its true extremes "
                             "run past 9d into problems no human can name a motif "
                             "from. Never apply it when building a bank: it would "
                             "delete the region the axis is most confident about. "
                             "Needs --dump (for the rating array).")
    parser.add_argument("--out", default=None,
                        help="Output HTML path (defaults to /tmp/axis_extremes.html)")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="Open the rendered HTML in the default browser.")
    args = parser.parse_args()

    if args.html and args.size == "all":
        args.size = "9"

    if args.dump:
        tids, components, sizes = load_dump_npz(args.dump, args.bank)
        source = os.path.basename(args.dump)
        if args.bank:
            source += f" taught from {os.path.basename(args.bank)}"
    else:
        src_path = args.per_problem
        if not os.path.isabs(src_path):
            src_path = os.path.join(PROJECT_ROOT, src_path)
        if not os.path.exists(src_path):
            raise SystemExit(f"missing {src_path}; run pipeline.train_mirt first")
        tids, components, sizes = load_per_problem_csv(src_path, args.labels)
        source = os.path.basename(src_path)

    # BOARDS COME FROM THE CORPUS, so this page does not depend on packaging at
    # all. That ordering is the point: the labelling workflow is refit → eyeball
    # the extremes here → name the axes in labels.json → package. Requiring
    # pwa/lookup.txt made the tool that GATES packaging refuse to run until
    # packaging had happened, and reading the shards on top of that raced a
    # rebuild — shard ids are content-derived, so a run in flight replaces the
    # files lookup.txt names and every board fails at once.
    #
    # per_problem.csv carries its own `size` column, written by the same fit whose
    # axes this page renders, so it is both sufficient and in step. lookup.txt is
    # a fallback for a dump that predates the column.
    lookup = {t: (None, s) for t, s in (sizes or {}).items()}
    if not lookup:
        lookup_path = os.path.join(PROJECT_ROOT, "pwa", "lookup.txt")
        if not os.path.exists(lookup_path):
            raise SystemExit(
                f"{source} carries no `size` column and {lookup_path} is absent — "
                f"no way to tell which board a problem is on.")
        lookup = load_lookup(lookup_path)

    # RANK CAP — for the eyes only, never for a bank. An axis is unbounded and
    # its true extremes run past 9d into problems no human is meant to read, so a
    # tail rendered honestly is mostly unjudgeable. Capping shows the strongest
    # problems a human can actually name a motif from. It must never be applied
    # to a stored direction: the cap would delete the region the axis is most
    # confident about and teach a truncated motif.
    # Ranks come from whichever artifact this page is reading. A dump carries its
    # OWN ratings, which is the only correct source for an experiment arm — its
    # ratings are not the shipped ones. A per_problem.csv holds embeddings only,
    # so the shipped ratings/ CSVs supply them; those were written by the same
    # run that wrote the CSV, so they belong to these axes.
    def _ranks_from(tid_arr, size_arr, rating_arr):
        curve = json.load(open(os.path.join(PROJECT_ROOT, "ratings",
                                            "rating_to_rank.json")))
        # Vectorised PER SIZE: interp_extrap takes an array, and each board
        # has its own curve.
        out = {}
        for sz in np.unique(size_arr):
            c = curve[str(int(sz))]
            m = size_arr == sz
            rv = interp_extrap(rating_arr[m], np.asarray(c["ratings"]),
                               np.asarray(c["ranks"]))
            out.update(zip(tid_arr[m], np.asarray(rv, dtype=float).tolist()))
        return out

    ranks = None
    if args.dump:
        z = np.load(args.dump, allow_pickle=False)
        if "rating" in z.files:
            ranks = _ranks_from(np.array([str(t) for t in z["tid"]]),
                                np.asarray(z["size"]).astype(int),
                                np.asarray(z["rating"], dtype=np.float64))
    else:
        from utils.constants import ALL_BOARD_SIZES
        from utils.rating.ratings_io import RATINGS_CSV, load_value_map
        rd = os.path.join(PROJECT_ROOT, "ratings")
        rt, rs = [], []
        want = set(tids)
        for sz in ALL_BOARD_SIZES:
            mp = load_value_map(os.path.join(rd, str(sz), RATINGS_CSV), "rating",
                                keep=want, missing_ok=True)
            rt += [(t, v) for t, v in mp.items()]
            rs += [sz] * len(mp)
        if rt:
            ranks = _ranks_from(np.array([t for t, _ in rt]),
                                np.asarray(rs, dtype=int),
                                np.asarray([v for _, v in rt], dtype=np.float64))
    if args.rank_max is not None:
        if ranks is None:
            raise SystemExit("--rank-max found no ratings: a --dump needs a "
                             "`rating` array, and a per_problem.csv needs the "
                             "ratings/<size>/tsumego_ratings.csv beside it.")
        # A problem with no rating is DROPPED by the cap rather than kept: the
        # cap exists to keep the page readable, and an unrateable problem cannot
        # be shown to be inside it. Reported, because silently thinning the pool
        # would read as the cap being stricter than it is.
        missing = sum(1 for t in tids if t not in ranks)
        keep = np.array([ranks.get(t, float("inf")) <= args.rank_max
                         for t in tids])
        if not keep.any():
            raise SystemExit(f"no problems at or below rank {args.rank_max}")
        print(f"  rank cap {args.rank_max}: {keep.sum()}/{len(tids)} problems "
              f"kept (display only)"
              + (f"; {missing} had no rating and were dropped" if missing else ""),
              flush=True)
        tids = [t for t, k in zip(tids, keep) if k]
        components = [(key, title, vals[keep], nm)
                      for key, title, vals, nm in components]

    # Size filter via lookup (per_problem.csv spans both boards, tid-aligned).
    if args.size != "all":
        want = int(args.size)
        keep = np.array([lookup.get(t, (None, None))[1] == want for t in tids])
        if not keep.any():
            raise SystemExit(f"no problems of size {args.size} in {src_path}")
        tids = [t for t, k in zip(tids, keep) if k]
        components = [(key, title, vals[keep], nm)
                      for key, title, vals, nm in components]

    if args.html:
        scope = f"{source}, size {args.size}"
        # ONE SECTION PER TAIL, not per axis. A name is a claim about one end, so
        # the tail is the unit a reader is actually judging; pairing both ends
        # under one selection makes them scroll past the other half to compare
        # two candidates for the same name.
        source = args.match_source or ("taught" if args.dump else "store")
        if source == "taught":
            bank_tails = load_taught_axis_tails(tids, args.match_hinge)
        elif source == "store":
            bank_tails = load_named_bank_tails(tids, args.match_hinge)
        else:
            bank_tails = []
        print(f"  {source}: {len(bank_tails)} named tails to match against",
              flush=True)
        options, sections = [], []
        n_tails = 2 * len(components)
        for i, (key, title, vals, name) in enumerate(components):
            print(f"  rendering {title} ({i + 1}/{len(components)})...", flush=True)
            # A bipolar label names each END separately: "A/B" puts A at the
            # positive tail and B at the negative. A one-sided "A" names only the
            # positive; the opposite end is where everything not-A accumulates and
            # makes no claim, so it is captioned as unnamed rather than borrowing
            # its partner's name.
            halves = [h.strip() for h in name.split("/")] if name else []
            for sign, suffix, word in ((+1, "p", "positive"), (-1, "n", "negative")):
                ti = 2 * i + (0 if sign > 0 else 1)
                tail_key = f"{key}{suffix}"
                h = 0 if sign > 0 else 1
                tail_name = halves[h] if h < len(halves) and halves[h] else None
                tail_title = (f"axis {i} {'+' if sign > 0 else '−'}  "
                              + (tail_name or "(unnamed)"))
                sel = " selected" if ti == 0 else ""
                options.append(f'<option value="{tail_key}"{sel}>'
                               f'{html.escape(tail_title)}</option>')
                body = _render_tail_html(tids, vals, sign, args.rows, args.cols,
                                         lookup, ranks, args.by_signal)
                matches = _match_html(vals, sign, bank_tails,
                                      args.match_hinge, args.match_min_r,
                                      source)
                sections.append(
                    f'<section class="comp-section{" active" if ti == 0 else ""}" '
                    f'data-comp="{tail_key}">'
                    f'<h2><span class="comp-label">{html.escape(tail_title)}</span></h2>'
                    f'{matches}{body}'
                    f'</section>')
        print(f"  {n_tails} tails", flush=True)
        page = _build_page(scope, "".join(options), "".join(sections), args.cols)
        out_path = args.out or os.path.join("/tmp", "axis_extremes.html")
        with open(out_path, "w") as f:
            f.write(page)
        print(f"wrote {out_path}")
        if args.open_browser:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(out_path)}")
        return

    base = f"{args.host}/?id="
    for key, title, vals, _nm in components:
        order = np.argsort(vals)
        print(f"\n{title} (source: {source}, scope: {args.size})")
        print(f"  top {args.n} positive tail:")
        for i in order[::-1][:args.n]:
            size = lookup.get(tids[i], (None, "?"))[1]
            print(f"    {vals[i]:>+7.3f} [{size}x{size}]  {base}{tids[i]}")
        print(f"  top {args.n} negative tail:")
        for i in order[:args.n]:
            size = lookup.get(tids[i], (None, "?"))[1]
            print(f"    {vals[i]:>+7.3f} [{size}x{size}]  {base}{tids[i]}")


if __name__ == "__main__":
    main()
