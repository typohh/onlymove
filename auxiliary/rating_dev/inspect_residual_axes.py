#!/usr/bin/env python3
"""Inspect the extreme-loading problems on the shipped SKILL AXES.

Run with no arguments to render the current axes/per_problem.csv — the axis set
pipeline.build_axes produced and package_for_pwa will publish. That is the
labelling workflow: package_for_pwa refuses to ship while any axis has neither
tail named, so this page is what a name gets chosen from.

Each axis shows both tails, and above them a table of WHAT DRIVES each tail.
The axis is a linear map on the standardized features, so a problem's value is
the sum over columns of weight x standardized input, and the mean of that
product across a tail's problems is exactly that column's contribution — an
exact decomposition, not a surrogate model. Inputs are grouped to be readable:

  - ACTIVATIONS collapse to one row carrying MAGNITUDE ONLY. The block is a PCA
    basis, so a component's sign is a property of the basis rather than of the
    world. This row is often most of the magnitude, and when it is, the boards
    rather than the table are the evidence — the named rows below it are then
    describing a minority of the signal.
  - BOTS keep their sign, and adjacent rungs of a ladder merge when they agree
    in sign and magnitude ("humanv0 rank ladder 12k-8k x5"). Merging is on the
    ladder only, never on contribution alone.
  - Everything else keeps its own signed row.

BOTH tails are rendered because the axes are bipolar: they carry no skew
sign-canonicalization, so a negative loading is a real direction — problems a
user with positive axis position does WORSE on — rather than baseline.

Per-fold development artifacts from auxiliary.rating_dev.train_residual_axes are
available via --npz, and differ in two ways. Folds are NEVER pooled, since the
objective is rotation-invariant within the K-dim subspace and each fold's axes
are an arbitrary basis of its own fit: fold 0 axis 2 and fold 1 axis 2 are
unrelated, and eyeballing whether any fold-0 axis matches any fold-1 axis is the
stability check that view is for. Those artifacts also include the population r
head (residual difficulty the rating model missed), whose extremes show what it
systematically over- and under-rates. They store loadings but no weight matrix,
so no input attribution is possible there.

Board rendering (parse / simulate / SVG) lives in utils/board_render.py — the
single shared copy used by every inspection tool.

Usage:
  python -m auxiliary.rating_dev.inspect_residual_axes            # the shipped axes
  python -m auxiliary.rating_dev.inspect_residual_axes --open
  python -m auxiliary.rating_dev.inspect_residual_axes --no-inputs   # boards only
  python -m auxiliary.rating_dev.inspect_residual_axes --npz --k 2   # a dev fold fit
"""

import argparse
import html
import os
import re
import sys
from functools import lru_cache

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.katago.board_render import load_lookup, render_problem  # noqa: E402
from utils.rating.features import CACHE_DIR  # noqa: E402


# ---- artifact loading ---------------------------------------------------

def load_per_problem_csv(path):
    """Production-axes mode: components straight from pipeline.build_axes'
    axes/per_problem.csv (one fit, canonical orientation — the
    shipped loadings, exactly what labels.json names)."""
    import csv as _csv
    tids, rows = [], []
    with open(path, newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        cols = [i for i, c in enumerate(header) if c.startswith("embed_")]
        for row in reader:
            tids.append(row[0])
            rows.append([float(row[i]) for i in cols])
    M = np.array(rows, dtype=np.float64)
    components = [(f"ax{k}", f"axis {k} (production)", M[:, k])
                  for k in range(M.shape[1])]
    return tids, components


def load_axes_npz(path):
    """Return (tids list, components list of (key, title, scores ndarray)).

    Components enumerate every fold's axes plus its r head. Folds are kept
    separate on purpose — see module docstring.
    """
    d = np.load(path, allow_pickle=False)
    tids = [str(t) for t in d["tids"]]
    components = []
    for fold in (0, 1):
        gk, rk = f"g_fold{fold}", f"r_fold{fold}"
        if gk in d:
            g = d[gk]
            for k in range(g.shape[1]):
                components.append(
                    (f"f{fold}a{k}", f"fold {fold} · axis {k}", g[:, k].astype(np.float64)))
        if rk in d:
            components.append(
                (f"f{fold}r", f"fold {fold} · r head (residual difficulty)",
                 d[rk].astype(np.float64)))
    if not components:
        raise SystemExit(f"no g_fold*/r_fold* arrays in {path}")
    return tids, components



# ---- input attribution --------------------------------------------------
# What drives each tail, by exact linear decomposition. The axis is a linear map
# on the standardized features, so a problem's axis value is the SUM over columns
# of W[k,j]*z[j] and the mean of that product over a tail's problems is exactly
# that column's contribution to the tail. No approximation and no surrogate
# model: the parts add up to the whole by construction.

BOT_PATTERNS = [
    # (regex, family label, how to read the ordinal from the match)
    (re.compile(r"humanv0_(\d+)k_t([\d.]+)$"),
     lambda m: (f"humanv0 {m.group(1)}k, temperature", float(m.group(2)))),
    (re.compile(r"humanv0_(\d+)([kd])$"),
     lambda m: ("humanv0 rank ladder",
                -int(m.group(1)) if m.group(2) == "k" else int(m.group(1)))),
    (re.compile(r"^kata1-.*_t([\d.]+)$"),
     lambda m: ("KataGo full, temperature", float(m.group(1)))),
    (re.compile(r"^kata9x9-.*_t([\d.]+)$"),
     lambda m: ("KataGo 9x9, temperature", float(m.group(1)))),
]


def _bot_slot(name):
    """(family, ordinal) for a bot that sits on a ladder, else None. The ladders
    are what make grouping meaningful: neighbouring ranks and neighbouring
    temperatures are near-duplicate opponents, so listing them separately says
    the same thing a dozen times."""
    for rx, read in BOT_PATTERNS:
        m = rx.search(name)
        if m:
            return read(m)
    return None


def _ord_label(family, o):
    if "rank ladder" in family:
        return f"{-o}k" if o < 0 else f"{o}d"
    return f"t{o:g}"


def _merge_runs(entries, rel_tol=2.5):
    """Collapse ADJACENT ladder entries whose contributions agree in sign and sit
    within rel_tol of each other in magnitude. Adjacency is on the ladder, never
    on contribution alone — merging two similar but distant rungs would invent a
    grouping the data does not have."""
    out = []
    for o, contrib in entries:
        if out:
            prev = out[-1]
            mean = prev["sum"] / prev["n"]
            same_sign = (mean >= 0) == (contrib >= 0)
            hi = max(abs(mean), abs(contrib))
            lo = min(abs(mean), abs(contrib))
            close = hi <= rel_tol * max(lo, 1e-4)
            if same_sign and close:
                prev["sum"] += contrib
                prev["n"] += 1
                prev["hi"] = o
                continue
        out.append({"lo": o, "hi": o, "sum": contrib, "n": 1})
    return out


def _group_contributions(contrib, names, blocks):
    """(activation_magnitude, groups) for one tail.

    Activations are reported as MAGNITUDE ONLY and never split: the block is a
    PCA basis, so an individual component's sign is arbitrary and "PC17 pushes
    the axis up" is not a statement about the world. Everything else keeps its
    sign, because "this bot's pass rate raises the axis" is one.
    """
    act = [i for i, b in enumerate(blocks) if b == "activation"]
    act_mag = float(np.abs(contrib[act]).sum())

    groups, ladders = [], {}
    for i, b in enumerate(blocks):
        if b == "activation":
            continue
        if b == "bot winrate":
            slot = _bot_slot(names[i])
            if slot:
                fam, o = slot
                ladders.setdefault(fam, []).append((o, float(contrib[i])))
                continue
        groups.append((names[i], b, float(contrib[i]), 1))

    for fam, entries in ladders.items():
        entries.sort(key=lambda e: e[0])
        for run in _merge_runs(entries):
            span = (_ord_label(fam, run["lo"]) if run["n"] == 1
                    else f'{_ord_label(fam, run["lo"])}\u2013{_ord_label(fam, run["hi"])}')
            groups.append((f"{fam} {span}", "bot winrate", run["sum"], run["n"]))
    groups.sort(key=lambda g: -abs(g[2]))
    return act_mag, groups


def compute_attribution(tids, vals, W_row, feats, feat_mu, feat_sd, scale,
                        names, blocks, tail_frac):
    """Per-tail grouped contributions for one axis. Returns (pos, neg), each
    (mean_axis_value, activation_magnitude, groups, n_problems)."""
    keep = [i for i, t in enumerate(tids) if t in feats]
    Z = (np.stack([feats[tids[i]] for i in keep]) - feat_mu) / feat_sd
    y = np.asarray([vals[i] for i in keep], dtype=float)
    w = W_row * scale
    m = max(1, int(round(len(y) * tail_frac)))
    order = np.argsort(y)
    out = []
    for sel in (order[-m:], order[:m]):
        contrib = (Z[sel] * w[None, :]).mean(0)
        act_mag, groups = _group_contributions(contrib, names, blocks)
        out.append((float(y[sel].mean()), act_mag, groups, m))
    return out[0], out[1]


def _attribution_html(pos, neg, top):
    def table(tag, packed):
        mean_val, act_mag, groups, m = packed
        named_mag = sum(abs(g[2]) for g in groups)
        total_mag = act_mag + named_mag or 1e-9
        rows = [
            f'<tr class="actrow"><td>activations (PCA block)</td>'
            f'<td class="blk">activation</td>'
            f'<td class="mag">magnitude {act_mag:.3f}</td>'
            f'<td class="shr">{act_mag / total_mag:.0%}</td></tr>']
        for name, blk, c, n in groups[:top]:
            cnt = f" &times;{n}" if n > 1 else ""
            rows.append(
                f'<tr><td>{html.escape(name)}{cnt}</td>'
                f'<td class="blk">{html.escape(blk)}</td>'
                f'<td class="{"up" if c >= 0 else "dn"}">{c:+.3f}</td>'
                f'<td class="shr">{abs(c) / total_mag:.0%}</td></tr>')
        return (
            f'<div class="attr"><h4>{tag} tail drivers '
            f'<span class="sub">({m} problems, mean axis value {mean_val:+.3f})</span></h4>'
            f'<table><thead><tr><th>input</th><th>block</th>'
            f'<th>contribution</th><th>share</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')
    return (f'<div class="attrwrap">{table("positive", pos)}'
            f'{table("negative", neg)}</div>'
            '<p class="attrnote">Contributions are the exact additive '
            'decomposition of the axis value (mean of weight&times;standardized '
            'input over the tail), so they sum to the tail mean. Ladder bots '
            '(adjacent ranks or temperatures) are merged where they agree in '
            'sign and magnitude. Activations carry magnitude only \u2014 the '
            'block is a PCA basis, so an individual component\u2019s sign is '
            'arbitrary.</p>')


# ---- HTML page generation ----------------------------------------------

@lru_cache(maxsize=None)
def _cached_render(tid, zip_id, size):
    """Memoize SVG rendering: a problem can land in several components' bands
    on the one-page layout, so render each board at most once."""
    return render_problem(tid, zip_id, size)


def _build_card(tid, score, lookup):
    info = lookup.get(tid)
    if not info:
        return f'<div class="card missing">missing in lookup: {tid}</div>'
    zip_id, size = info
    try:
        rp = _cached_render(tid, zip_id, size)
    except (FileNotFoundError, KeyError) as e:
        return f'<div class="card missing">load failed: {tid} ({e})</div>'
    return (
        '<div class="card">'
        f'{rp["svg"]}'
        f'<div class="meta">'
        f'<span class="score">{score:+.3f}</span>'
        f'<span class="tid">{tid}</span>'
        f'<span class="turn turn-{rp["to_move"][0]}">{rp["to_move"]} to play</span>'
        '</div>'
        '</div>'
    )


def _pick_extremes(vals, rows, cols):
    """Return the `rows*cols` most extreme problems for one tail, strongest
    first, chunked into `rows` groups of `cols`.

    Operates on the POSITIVE direction of `vals` (negate the input for the
    negative tail; scores are displayed from the original values by the
    caller). Each returned tuple is (rank_of_first_item, index_array) with
    rank 1 = the tail extreme, so the page reads as a straight leaderboard of
    the strongest signal rather than a walk from extreme toward baseline.
    """
    n = len(vals)
    take = min(rows * cols, n)
    if take <= 0:
        return []
    idx = np.argpartition(-vals, take - 1)[:take]
    idx = idx[np.argsort(vals[idx])[::-1]]
    return [(k + 1, idx[k:k + cols]) for k in range(0, take, cols)]


def _render_tail_html(tids, vals, sign, rows, cols, lookup):
    """One tail's most extreme problems. sign=+1 renders the positive tail, -1
    the negative (card scores are shown in ORIGINAL units either way)."""
    groups = _pick_extremes(sign * vals, rows, cols)
    if not groups:
        return '<div class="band"><div class="card missing">no problems on this tail</div></div>'
    out = []
    for lo, idx in groups:
        cards = "".join(
            _build_card(tids[i], float(vals[i]), lookup) for i in idx)
        out.append(
            f'<div class="band">'
            f'<div class="band-label">#{lo}–{lo + len(idx) - 1}</div>'
            f'<div class="grid">{cards}</div>'
            f'</div>'
        )
    return "".join(out)


ATTR_CSS = """
.attrwrap{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0 14px}
.attr{flex:1 1 420px;min-width:340px}
.attr h4{margin:6px 0 4px;font-size:14px}
.attr .sub{font-weight:400;opacity:.7}
.attr table{border-collapse:collapse;width:100%;font-size:12px}
.attr th,.attr td{padding:3px 6px;border-bottom:1px solid rgba(128,128,128,.25);text-align:left}
.attr th{font-weight:600;opacity:.75}
.attr td.blk{opacity:.65}
.attr td.up{color:#1a7f37;text-align:right;font-variant-numeric:tabular-nums}
.attr td.dn{color:#b3261e;text-align:right;font-variant-numeric:tabular-nums}
.attr td.mag,.attr td.shr{text-align:right;font-variant-numeric:tabular-nums;opacity:.8}
.attr tr.actrow td{font-style:italic}
.attrnote{font-size:11px;opacity:.7;max-width:900px;margin:0 0 18px}
"""


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
  .meta {{ font-size: 0.8rem; margin-top: 0.4rem; display: flex; flex-direction: column; gap: 0.1rem; }}
  .score {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
  .tid {{ font-family: ui-monospace, monospace; color: #555; }}
  .turn-B {{ color: #111; }}
  .turn-W {{ color: #888; }}
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
{_style_block(cols)}\n<style>{ATTR_CSS}</style>
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
    # No arguments renders the SHIPPED axes, which is the labelling workflow and
    # the only thing most runs want. The per-fold dev artifacts are opt-in.
    parser.add_argument("--per-problem", nargs="?",
                        default=os.path.join(PROJECT_ROOT, "axes",
                                             "per_problem.csv"),
                        const=os.path.join(PROJECT_ROOT, "axes",
                                           "per_problem.csv"),
                        help="path to the shipped per_problem.csv (default: "
                             "axes/per_problem.csv)")
    parser.add_argument("--npz", default=None,
                        help="render a per-fold dev artifact instead of the "
                             "shipped axes (auxiliary.rating_dev."
                             "train_residual_axes output). Folds are separate "
                             "fits, so no input attribution is available")
    parser.add_argument("--k", type=int, default=4,
                        help="with --npz and no path: K used at training time")
    parser.add_argument("--seed", type=int, default=0,
                        help="with --npz and no path: seed used at training time")
    parser.add_argument("--n", type=int, default=10,
                        help="N top problems per tail in --no-html mode (default 10)")
    parser.add_argument("--host", default="http://localhost:8787",
                        help="PWA dev server origin (only used with --no-html)")
    parser.add_argument("--size", default="all",
                        help="9, 13, or 'all' (default; HTML mode defaults to 9 so grids "
                             "stay board-comparable)")
    parser.add_argument("--html", action=argparse.BooleanOptionalAction, default=True,
                        help="Render static HTML (default). --no-html prints PWA URLs instead.")
    parser.add_argument("--rows", type=int, default=4,
                        help="rows of extremes per tail; rows × cols problems shown, strongest first (default 5)")
    parser.add_argument("--cols", type=int, default=6, help="HTML grid columns (default 6)")
    parser.add_argument("--out", default=None,
                        help="Output HTML path (defaults to /tmp/residual_axes_extremes.html)")
    parser.add_argument("--no-inputs", action="store_true",
                        help="skip the per-axis input attribution (it loads the "
                             "full feature matrix, about a minute)")
    parser.add_argument("--tail-frac", type=float, default=0.02,
                        help="fraction of problems per tail for the attribution")
    parser.add_argument("--top-inputs", type=int, default=14,
                        help="grouped inputs listed per tail")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="Open the rendered HTML in the default browser.")
    args = parser.parse_args()

    if args.html and args.size == "all":
        args.size = "9"

    use_npz = args.npz is not None or "--npz" in sys.argv
    if use_npz:
        npz_path = args.npz if isinstance(args.npz, str) else os.path.join(
            CACHE_DIR, f"residual_axes_k{args.k}_seed{args.seed}.npz")
        if not os.path.exists(npz_path):
            raise SystemExit(
                f"missing {npz_path}; run auxiliary.rating_dev.train_residual_axes first "
                f"(or point --npz at an existing artifact)")
        tids, components = load_axes_npz(npz_path)
        args.no_inputs = True      # per-fold artifacts carry no weight matrix
    else:
        npz_path = args.per_problem
        if not os.path.exists(npz_path):
            raise SystemExit(f"missing {npz_path}; run pipeline.build_axes first")
        tids, components = load_per_problem_csv(npz_path)
    source = os.path.basename(npz_path)

    lookup_path = os.path.join(PROJECT_ROOT, "pwa", "lookup.txt")
    if not os.path.exists(lookup_path):
        raise SystemExit(f"missing {lookup_path}; run pipeline.package_for_pwa first")
    lookup = load_lookup(lookup_path)

    # Size filter via lookup (the npz spans both boards, tid-aligned).
    if args.size != "all":
        want = int(args.size)
        keep = np.array([lookup.get(t, (None, None))[1] == want for t in tids])
        if not keep.any():
            raise SystemExit(f"no problems of size {args.size} in {npz_path}")
        tids = [t for t, k in zip(tids, keep) if k]
        components = [(key, title, vals[keep]) for key, title, vals in components]

    if args.html:
        scope = f"{source}, size {args.size}"
        attrib = None
        if not args.no_inputs:
            print("  loading features for input attribution...", flush=True)
            import utils.rating.residual_axes as _ra
            from auxiliary.inspect_axis_inputs import column_names
            feats, _rk, _pa, act_dim = _ra.build_axis_features()
            d = np.load(os.path.join(PROJECT_ROOT, "axes", "axes.npz"),
                        allow_pickle=False)
            W, feat_mu, feat_sd = d["W"], d["feat_mu"], d["feat_sd"]
            # the shipped value is the raw W.z rescaled by v_sd/(rms*axis_sd), a
            # positive per-axis constant, so applying it makes the decomposition
            # land in the units the page already shows
            scale = d["v_sd"] / (d["rms"] * d["axis_sd"])
            names, blocks = column_names(W.shape[1], int(act_dim))
            attrib = (feats, W, feat_mu, feat_sd, scale, names, blocks)

        options, sections = [], []
        for i, (key, title, vals) in enumerate(components):
            selected = " selected" if i == 0 else ""
            options.append(
                f'<option value="{key}"{selected}>{html.escape(title)}</option>')
            print(f"  rendering {title} ({i + 1}/{len(components)})...", flush=True)
            pos_html = _render_tail_html(tids, vals, +1, args.rows, args.cols, lookup)
            neg_html = _render_tail_html(tids, vals, -1, args.rows, args.cols, lookup)
            attr_html = ""
            if attrib is not None and i < attrib[1].shape[0]:
                feats, W, feat_mu, feat_sd, scale, names, blocks = attrib
                p_att, n_att = compute_attribution(
                    tids, vals, W[i], feats, feat_mu, feat_sd, scale[i],
                    names, blocks, args.tail_frac)
                attr_html = _attribution_html(p_att, n_att, args.top_inputs)
            active = " active" if i == 0 else ""
            sections.append(
                f'<section class="comp-section{active}" data-comp="{key}">'
                f'<h2><span class="comp-label">{html.escape(title)}</span></h2>'
                f'{attr_html}'
                f'<h3>positive tail</h3>{pos_html}'
                f'<h3>negative tail</h3>{neg_html}'
                f'</section>')
        page = _build_page(scope, "".join(options), "".join(sections), args.cols)
        out_path = args.out or os.path.join("/tmp", "residual_axes_extremes.html")
        with open(out_path, "w") as f:
            f.write(page)
        print(f"wrote {out_path}")
        if args.open_browser:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(out_path)}")
        return

    base = f"{args.host}/?id="
    for key, title, vals in components:
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
