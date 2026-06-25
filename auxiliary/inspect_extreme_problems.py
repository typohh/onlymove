"""Inspect the top-N highest-scoring problems on a given embedding component.

Only the positive side is shown: the default pipeline uses FastICA with the
skew contrast and sign-canonicalizes axes so the positive tail concentrates
the motif and the negative tail is Gaussian baseline (see
pipeline/build_embedding.py `_skew_contrast`). Negative-side problems have no
coherent theme by construction, so the LOW section that older versions of
this script printed was misleading.

Reads from embeddings/joint/per_problem.csv when available (the joint model
spans both board sizes); falls back to embeddings/<size>/per_problem.csv for
the legacy per-size pipeline. Human motif labels come from the sibling
labels.json.

Default mode is HTML: a self-contained static page with `--rows` intensity
bands. Row 0 targets max(col); row k targets max(col) · (rows-k)/rows. With
rows=5 the bottom row sits around 1/5 of the extreme — useful for watching
the visual theme decay from peak toward baseline. Use --open to launch the
page, or --no-html to print PWA dev-server URLs instead.

`--all-pcs` renders every component into one self-contained page with a
dropdown to switch between them; each option shows the component's label.

Board rendering (parse / simulate / SVG) lives in utils/board_render.py — the
single shared copy used by every inspection tool.

Usage:
  python -m auxiliary.inspect_extreme_problems [--pc N] [--rows 5] [--cols 5]
  python -m auxiliary.inspect_extreme_problems --pc 3 --open
  python -m auxiliary.inspect_extreme_problems --all-pcs --open
  python -m auxiliary.inspect_extreme_problems --pc 3 --no-html
"""

import argparse
import html
import json
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.board_render import load_lookup, render_problem  # noqa: E402


# ---- labels ------------------------------------------------------------

def _load_labels(csv_path):
    """Return {pc_index: positive_label or None} from labels.json next to the
    per_problem.csv (empty dict if the file is absent)."""
    path = os.path.join(os.path.dirname(csv_path), "labels.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for k, v in data.get("components", {}).items():
        try:
            out[int(k)] = (v or {}).get("positive")
        except ValueError:
            continue
    return out


# ---- HTML page generation ---------------------------------------------

@lru_cache(maxsize=None)
def _cached_render(tid, zip_id, size):
    """Memoize SVG rendering: a problem can land in several PCs' bands on the
    --all-pcs page, so render each board at most once."""
    return render_problem(tid, zip_id, size)


def _build_card(row, score_col, lookup):
    tid = row["tsumego_id"]
    score = float(row[score_col])
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


def _pick_intensity_bands(df, col, rows, cols):
    """Return `rows` (target_score, picks_df) tuples for HTML rendering.

    Row k targets `max(col) * (rows - k) / rows` and grabs the `cols`
    problems with score closest to that target. Already-chosen problems
    are excluded so rows don't repeat each other on dense bands.

    Works on the score column as a numpy array (positional masking +
    argpartition) rather than copying the whole frame per band — `--all-pcs`
    calls this 64 times, so the full-frame copy + `.isin` it used to do
    dominated runtime.

    Only the positive side; see module docstring for why.
    """
    vals = df[col].to_numpy(dtype=np.float64)
    n = len(vals)
    extreme = float(vals.max()) if n else 0.0
    chosen = np.zeros(n, dtype=bool)
    bands = []
    for k in range(rows):
        target = extreme * (rows - k) / rows
        take = min(cols, n - int(chosen.sum()))
        if take <= 0:
            bands.append((target, df.iloc[:0]))
            continue
        dist = np.abs(vals - target)
        dist[chosen] = np.inf
        idx = np.argpartition(dist, take - 1)[:take]
        # Within a row, sort by score descending so the eye scans
        # peak→baseline left-to-right.
        idx = idx[np.argsort(vals[idx])[::-1]]
        chosen[idx] = True
        bands.append((target, df.iloc[idx]))
    return bands


def _render_bands_html(high_bands, score_col, cols, lookup):
    out = []
    for target, row_df in high_bands:
        if row_df.empty:
            cards = '<div class="card missing">no problems left in this band</div>'
        else:
            cards = "".join(_build_card(r, score_col, lookup) for _, r in row_df.iterrows())
        out.append(
            f'<div class="band">'
            f'<div class="band-label">target {target:+.3f}</div>'
            f'<div class="grid">{cards}</div>'
            f'</div>'
        )
    return "".join(out)


def _style_block(cols):
    return f"""<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background:#fafafa; color:#222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  h2 {{ margin-top: 1.5rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; }}
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
  .pc-label {{ color: #1d6fa5; font-weight: 600; }}
  .pc-label.none {{ color: #999; font-weight: 400; font-style: italic; }}
  .pc-section {{ display: none; }}
  .pc-section.active {{ display: block; }}
</style>"""


def _build_html(pc, source, label, high_bands, score_col, cols, lookup):
    high_html = _render_bands_html(high_bands, score_col, cols, lookup)
    n_rows = max(len(high_bands), 1)
    label_html = (f'<span class="pc-label">{html.escape(label)}</span>'
                  if label else '<span class="pc-label none">(unlabeled)</span>')
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PC{pc} extremes ({source})</title>
{_style_block(cols)}
</head>
<body>
<h1>PC{pc} positive tail &middot; {label_html}</h1>
<div class="legend">
  Source: {source} &middot; Red ring <span class="marker"></span> = correct first move
  &middot; Rows step from peak (top) toward ~1/{n_rows} of the extreme (bottom).
  Negative tail not shown — baseline by construction under skew-ICA.
</div>
{high_html}
</body>
</html>
"""


def _build_all_pcs_html(source, options_html, sections_html, n_pcs, cols):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>All {n_pcs} PCs ({source})</title>
{_style_block(cols)}
</head>
<body>
<h1>Embedding components — positive tail</h1>
<div class="legend">
  Source: {source} &middot; Red ring <span class="marker"></span> = correct first move
  &middot; Rows step from peak (top) toward baseline (bottom).
  Negative tail not shown — baseline by construction under skew-ICA.
</div>
<div class="controls">
  <label>Component: <select id="pcSelect">{options_html}</select></label>
</div>
{sections_html}
<script>
  const sel = document.getElementById('pcSelect');
  function showPc(pc) {{
    document.querySelectorAll('.pc-section').forEach(s => {{
      s.classList.toggle('active', s.dataset.pc === pc);
    }});
  }}
  sel.addEventListener('change', () => {{ showPc(sel.value); window.scrollTo(0, 0); }});
  showPc(sel.value);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc", type=int, default=0, help="Component index (default 0)")
    parser.add_argument("--all-pcs", dest="all_pcs", action="store_true",
                        help="HTML with every component, switchable via a dropdown "
                             "(labels shown). Ignores --pc.")
    parser.add_argument("--n", type=int, default=10,
                        help="N top-scoring problems in --no-html mode (default 10)")
    parser.add_argument("--host", default="http://localhost:8787",
                        help="PWA dev server origin (only used with --no-html)")
    parser.add_argument("--size", default="all",
                        help="9, 13, or 'all' (default). 'all' picks across both sizes.")
    parser.add_argument("--html", action=argparse.BooleanOptionalAction, default=True,
                        help="Render static HTML (default). --no-html prints PWA URLs instead.")
    parser.add_argument("--rows", type=int, default=5,
                        help="HTML intensity bands; row k targets extreme · (rows-k)/rows (default 5)")
    parser.add_argument("--cols", type=int, default=5, help="HTML grid columns (default 5)")
    parser.add_argument("--out", default=None,
                        help="Output HTML path (defaults to /tmp/pc<N>_extremes.html, "
                             "or /tmp/all_pcs_extremes.html with --all-pcs)")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="Open the rendered HTML in the default browser.")
    args = parser.parse_args()

    if args.html and args.size == "all":
        args.size = "9"

    joint_csv = os.path.join(PROJECT_ROOT, "embeddings", "joint", "per_problem.csv")
    if os.path.exists(joint_csv):
        csv_path = joint_csv
        source = "joint"
    else:
        from utils.configurations import ALL_BOARD_SIZES
        size = args.size if args.size != "all" else str(ALL_BOARD_SIZES[0])
        csv_path = os.path.join(PROJECT_ROOT, "embeddings", size, "per_problem.csv")
        source = f"per-size ({size})"
    if not os.path.exists(csv_path):
        raise SystemExit("no per_problem.csv found (looked at joint and per-size locations)")

    df = pd.read_csv(csv_path)
    if args.size != "all" and "size" in df.columns:
        df = df[df["size"] == int(args.size)]
        if df.empty:
            raise SystemExit(f"no rows for size {args.size} in {csv_path}")

    embed_cols = sorted(
        (int(c.split("_")[1]) for c in df.columns if c.startswith("embed_")))
    if not embed_cols:
        raise SystemExit(f"no embed_* columns in {csv_path}")

    if not args.all_pcs:
        col = f"embed_{args.pc}"
        if col not in df.columns:
            raise SystemExit(
                f"No column {col} in {csv_path} (max component index = {embed_cols[-1]})")

    labels = _load_labels(csv_path)

    if args.html:
        lookup_path = os.path.join(PROJECT_ROOT, "pwa", "lookup.txt")
        if not os.path.exists(lookup_path):
            raise SystemExit(f"missing {lookup_path}; run pipeline.package_for_pwa first")
        lookup = load_lookup(lookup_path)
        scope = f"{source}, size {args.size}"

        if args.all_pcs:
            options, sections = [], []
            for i, pc in enumerate(embed_cols):
                label = labels.get(pc)
                label_txt = label if label else "(unlabeled)"
                selected = " selected" if i == 0 else ""
                options.append(
                    f'<option value="{pc}"{selected}>'
                    f'PC{pc} — {html.escape(label_txt)}</option>')
                print(f"  rendering PC{pc} ({i + 1}/{len(embed_cols)})...", flush=True)
                bands = _pick_intensity_bands(df, f"embed_{pc}", args.rows, args.cols)
                bands_html = _render_bands_html(bands, f"embed_{pc}", args.cols, lookup)
                label_html = (f'<span class="pc-label">{html.escape(label)}</span>'
                              if label else '<span class="pc-label none">(unlabeled)</span>')
                active = " active" if i == 0 else ""
                sections.append(
                    f'<section class="pc-section{active}" data-pc="{pc}">'
                    f'<h2>PC{pc} &middot; {label_html}</h2>'
                    f'{bands_html}</section>')
            page = _build_all_pcs_html(scope, "".join(options), "".join(sections),
                                       len(embed_cols), args.cols)
            out_path = args.out or os.path.join("/tmp", "all_pcs_extremes.html")
        else:
            high_bands = _pick_intensity_bands(df, col, args.rows, args.cols)
            page = _build_html(args.pc, scope, labels.get(args.pc),
                               high_bands, col, args.cols, lookup)
            out_path = args.out or os.path.join("/tmp", f"pc{args.pc}_extremes.html")

        with open(out_path, "w") as f:
            f.write(page)
        print(f"wrote {out_path}")
        if args.open_browser:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(out_path)}")
        return

    n = args.n
    col = f"embed_{args.pc}"
    df_sorted = df.sort_values(col, ascending=False)
    high = df_sorted.head(n)

    base = f"{args.host}/?id="
    has_size = "size" in df.columns

    def fmt(row):
        size_tag = f" [{int(row['size'])}x{int(row['size'])}]" if has_size else ""
        return f"  {row[col]:>+7.3f}{size_tag}  {base}{row['tsumego_id']}"

    label = labels.get(args.pc)
    label_tag = f" — {label}" if label else ""
    print(f"\nPC{args.pc}{label_tag} (source: {source}, scope: {args.size}): top {n} positive tail\n")
    for _, row in high.iterrows():
        print(fmt(row))


if __name__ == "__main__":
    main()
