#!/usr/bin/env python3
"""Inspect the extreme-loading problems on the residual-MIRT secondary axes.

Companion to auxiliary/inspect_extreme_problems.py, but for the supervised
axes trained by auxiliary.rating_dev.train_residual_axes (loaded from the
activations/_signal_cache/residual_axes_k<K>_seed<S>.npz it saves) instead of
the skew-ICA embedding. Three structural differences follow from that model:

  - BOTH tails are rendered. The residual axes carry no skew
    sign-canonicalization: a negative loading is a real direction (problems
    where a user with positive axis position does WORSE), not Gaussian
    baseline, so unlike the ICA inspector the LOW side is meaningful here.
  - Folds are NEVER pooled. The training objective is rotation-invariant
    within the K-dim subspace, so each fold's axes are an arbitrary basis of
    its own fit — fold 0 axis 2 and fold 1 axis 2 are unrelated components.
    Every (fold, axis) pair is its own dropdown entry; eyeballing whether any
    fold-0 axis visually matches any fold-1 axis is exactly the kind of
    stability check this page is for.
  - The population r head (residual difficulty the 4PL missed) is included
    as an extra component per fold — its extremes show what the rating model
    systematically over/under-rates.

Default mode is HTML: one self-contained page, a dropdown over every
(fold, component), `--rows` intensity bands per tail. Row 0 targets the tail
extreme; row k targets extreme · (rows-k)/rows. Use --open to launch the
page, or --no-html to print PWA dev-server URLs instead.

Board rendering (parse / simulate / SVG) lives in utils/board_render.py — the
single shared copy used by every inspection tool.

Usage:
  python -m auxiliary.rating_dev.inspect_residual_axes                # k=4 seed=0
  python -m auxiliary.rating_dev.inspect_residual_axes --k 2 --open
  python -m auxiliary.rating_dev.inspect_residual_axes --npz path/to/file.npz
  python -m auxiliary.rating_dev.inspect_residual_axes --no-html --n 10
"""

import argparse
import html
import os
import sys
from functools import lru_cache

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.board_render import load_lookup, render_problem  # noqa: E402
from utils.rating_features import CACHE_DIR  # noqa: E402


# ---- artifact loading ---------------------------------------------------

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


def _pick_intensity_bands(vals, rows, cols):
    """Return `rows` (target_score, index_array) tuples for one tail.

    Operates on the POSITIVE direction of `vals` (negate the input for the
    negative tail; targets/scores are displayed from the original values by
    the caller). Row k targets max(vals) · (rows-k)/rows and grabs the `cols`
    problems closest to that target; already-chosen problems are excluded so
    rows don't repeat each other on dense bands.
    """
    n = len(vals)
    extreme = float(vals.max()) if n else 0.0
    chosen = np.zeros(n, dtype=bool)
    bands = []
    for k in range(rows):
        target = extreme * (rows - k) / rows
        take = min(cols, n - int(chosen.sum()))
        if take <= 0:
            bands.append((target, np.array([], dtype=int)))
            continue
        dist = np.abs(vals - target)
        dist[chosen] = np.inf
        idx = np.argpartition(dist, take - 1)[:take]
        # Within a row, sort by score descending so the eye scans
        # peak→baseline left-to-right.
        idx = idx[np.argsort(vals[idx])[::-1]]
        chosen[idx] = True
        bands.append((target, idx))
    return bands


def _render_tail_html(tids, vals, sign, rows, cols, lookup):
    """One tail's bands. sign=+1 renders the positive tail, -1 the negative
    (band targets and card scores are shown in ORIGINAL units either way)."""
    bands = _pick_intensity_bands(sign * vals, rows, cols)
    out = []
    for target, idx in bands:
        if len(idx) == 0:
            cards = '<div class="card missing">no problems left in this band</div>'
        else:
            cards = "".join(
                _build_card(tids[i], float(vals[i]), lookup) for i in idx)
        out.append(
            f'<div class="band">'
            f'<div class="band-label">target {sign * target:+.3f}</div>'
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
{_style_block(cols)}
</head>
<body>
<h1>Residual-MIRT axes — both tails</h1>
<div class="legend">
  Source: {html.escape(source)} &middot; Red ring <span class="marker"></span> = correct first move
  &middot; Rows step from the tail extreme (top) toward baseline (bottom).
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
    parser.add_argument("--k", type=int, default=4,
                        help="K used at training time — picks the default npz (default 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed used at training time — picks the default npz (default 0)")
    parser.add_argument("--npz", default=None,
                        help="explicit path to a residual_axes_*.npz (overrides --k/--seed)")
    parser.add_argument("--n", type=int, default=10,
                        help="N top problems per tail in --no-html mode (default 10)")
    parser.add_argument("--host", default="http://localhost:8787",
                        help="PWA dev server origin (only used with --no-html)")
    parser.add_argument("--size", default="all",
                        help="9, 13, or 'all' (default; HTML mode defaults to 9 like "
                             "inspect_extreme_problems so grids stay board-comparable)")
    parser.add_argument("--html", action=argparse.BooleanOptionalAction, default=True,
                        help="Render static HTML (default). --no-html prints PWA URLs instead.")
    parser.add_argument("--rows", type=int, default=5,
                        help="intensity bands per tail; row k targets extreme · (rows-k)/rows (default 5)")
    parser.add_argument("--cols", type=int, default=5, help="HTML grid columns (default 5)")
    parser.add_argument("--out", default=None,
                        help="Output HTML path (defaults to /tmp/residual_axes_extremes.html)")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="Open the rendered HTML in the default browser.")
    args = parser.parse_args()

    if args.html and args.size == "all":
        args.size = "9"

    npz_path = args.npz or os.path.join(
        CACHE_DIR, f"residual_axes_k{args.k}_seed{args.seed}.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(
            f"missing {npz_path}; run auxiliary.rating_dev.train_residual_axes first "
            f"(or point --npz at an existing artifact)")
    tids, components = load_axes_npz(npz_path)
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
        options, sections = [], []
        for i, (key, title, vals) in enumerate(components):
            selected = " selected" if i == 0 else ""
            options.append(
                f'<option value="{key}"{selected}>{html.escape(title)}</option>')
            print(f"  rendering {title} ({i + 1}/{len(components)})...", flush=True)
            pos_html = _render_tail_html(tids, vals, +1, args.rows, args.cols, lookup)
            neg_html = _render_tail_html(tids, vals, -1, args.rows, args.cols, lookup)
            active = " active" if i == 0 else ""
            sections.append(
                f'<section class="comp-section{active}" data-comp="{key}">'
                f'<h2><span class="comp-label">{html.escape(title)}</span></h2>'
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
