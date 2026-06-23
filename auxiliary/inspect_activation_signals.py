"""Inspect problems by a single RAW KataGo trunk activation signal.

Sibling of inspect_extreme_problems.py, but it looks at the *pre*-transformation
features: the 6912 raw activations (18 residual blocks × 384 channels) that
extract_activations.py captures at the correct move, BEFORE the rating-
residualized PCA + skew-ICA of build_embedding.py. You pick one (layer, channel)
and the script ranks every problem by that scalar.

Two things differ from the post-PCA inspector:

  * Raw activations are NOT sign-canonicalized, so a channel is bidirectional —
    both the most-negative and most-positive problems can carry a motif. The
    HTML therefore steps across the FULL range (max at the top, min at the
    bottom), not just the positive tail.

  * Both board sizes are included by default ("all"); the signal index is
    board-independent, so 9x9 and 13x13 share one distribution.

Index layout (matches extract_activations.vecs_at_cells): block channels are
concatenated in rconv1..rconv18 order, so feature index = layer*384 + channel
with layer in 0..17 (17 = last block, the default) and channel in 0..383.

Board rendering (parse / simulate / SVG with last-move, ko and correct-move
markers) is the shared utils/board_render.py renderer.

The first run for a size scans every activations/<size>/*.npz and caches the
correct-move vectors to activations/_signal_cache/ (a memmap), so subsequent
layer/channel sweeps are instant. Use --rebuild-cache to force a rescan.

Usage:
  python -m auxiliary.inspect_activation_signals                 # layer 17, channel 0, both sizes
  python -m auxiliary.inspect_activation_signals --layer 5 --channel 120 --open
  python -m auxiliary.inspect_activation_signals --size 13 --channel 42 --no-html
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.board_render import load_lookup, render_problem  # noqa: E402
from pipeline.build_embedding import _load_correct_vec  # noqa: E402
# The activation layout (N_LAYERS/CHANNELS_PER_LAYER/N_FEATS) + signal-cache contract
# (CACHE_DIR + path helpers) live in utils.rating_features; _build_cache below writes
# the cache those path helpers point at.
from utils.rating_features import (  # noqa: E402
    CACHE_DIR, CHANNELS_PER_LAYER, N_FEATS, N_LAYERS,
    _cache_ids_path, _cache_vecs_path,
)


# ---- correct-vector cache ---------------------------------------------

def _load_chunk(task):
    """Worker: load the correct-move vectors for a chunk of problems.

    Returns (start_index, (chunk, N_FEATS) float16). A problem that fails to
    load gets a NaN row so the corpus stays index-aligned; the caller drops it.
    """
    start, act_dir, items = task
    out = np.full((len(items), N_FEATS), np.nan, dtype=np.float16)
    for i, (tid, npz_name, npy_name) in enumerate(items):
        by_ext = {
            "npz": {tid: npz_name} if npz_name else {},
            "npy": {tid: npy_name} if npy_name else {},
        }
        try:
            out[i] = _load_correct_vec(act_dir, tid, by_ext).astype(np.float16)
        except Exception:
            pass  # leave NaN row
    return start, out


def _build_cache(size, workers):
    act_dir = os.path.join(PROJECT_ROOT, "activations", str(size))
    if not os.path.isdir(act_dir):
        raise SystemExit(f"no activations dir for size {size}: {act_dir}")
    files = os.listdir(act_dir)
    npz = {f[:-4]: f for f in files if f.endswith(".npz")}
    npy = {f[:-4]: f for f in files if f.endswith(".npy")}
    ids = sorted(set(npz) | set(npy))
    if not ids:
        raise SystemExit(f"no .npz/.npy activation files in {act_dir}")
    n = len(ids)

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_path = _cache_vecs_path(size) + ".tmp"
    mm = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=(n, N_FEATS))

    chunk = 1024
    tasks = []
    for s in range(0, n, chunk):
        chunk_ids = ids[s:s + chunk]
        items = [(tid, npz.get(tid), npy.get(tid)) for tid in chunk_ids]
        tasks.append((s, act_dir, items))

    print(f"[size {size}] building correct-vector cache: {n} problems "
          f"-> {_cache_vecs_path(size)} ({n * N_FEATS * 2 / 1e9:.2f} GB)", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s, arr in ex.map(_load_chunk, tasks):
            mm[s:s + arr.shape[0]] = arr
            done += arr.shape[0]
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"\r  {done}/{n} ({rate:.0f}/s)", end="", flush=True)
    print(flush=True)
    mm.flush()
    del mm
    os.replace(tmp_path, _cache_vecs_path(size))
    with open(_cache_ids_path(size), "w") as f:
        json.dump(ids, f)


def _load_signal(size, layer, channel, workers, rebuild):
    """Return a DataFrame [tsumego_id, signal] for one (layer, channel)."""
    vecs_path = _cache_vecs_path(size)
    ids_path = _cache_ids_path(size)
    if rebuild or not (os.path.exists(vecs_path) and os.path.exists(ids_path)):
        _build_cache(size, workers)

    with open(ids_path) as f:
        ids = json.load(f)
    vecs = np.load(vecs_path, mmap_mode="r")
    if vecs.shape[1] != N_FEATS or vecs.shape[0] != len(ids):
        raise SystemExit(
            f"cache for size {size} is stale ({vecs.shape} vs {len(ids)} ids); "
            f"rerun with --rebuild-cache")
    col = layer * CHANNELS_PER_LAYER + channel
    sig = np.asarray(vecs[:, col], dtype=np.float32)
    df = pd.DataFrame({"tsumego_id": ids, "signal": sig})
    return df.dropna(subset=["signal"]).reset_index(drop=True)


# ---- full-range intensity bands ---------------------------------------

def _pick_full_range_bands(df, rows, cols):
    """Return `rows` (target, picks_df) tuples spanning max → min.

    Unlike the post-ICA inspector (positive tail only), raw activations are
    bidirectional, so row 0 targets max(signal) and the last row targets
    min(signal). Each row grabs the `cols` problems closest to its target,
    excluding already-chosen ones, sorted descending within the row.
    """
    hi = float(df["signal"].max())
    lo = float(df["signal"].min())
    chosen = set()
    bands = []
    for k in range(rows):
        frac = k / (rows - 1) if rows > 1 else 0.0
        target = hi - (hi - lo) * frac
        remaining = df[~df["tsumego_id"].isin(chosen)].copy()
        if remaining.empty:
            bands.append((target, remaining))
            continue
        remaining["_dist"] = (remaining["signal"] - target).abs()
        picks = remaining.nsmallest(cols, "_dist").copy()
        chosen.update(picks["tsumego_id"].tolist())
        picks = picks.sort_values("signal", ascending=False).drop(columns="_dist")
        bands.append((target, picks))
    return bands


# ---- HTML page generation ---------------------------------------------

def _build_card(row, lookup):
    tid = row["tsumego_id"]
    sig = float(row["signal"])
    info = lookup.get(tid)
    if not info:
        return f'<div class="card missing">missing in lookup: {tid}</div>'
    zip_id, size = info
    try:
        rp = render_problem(tid, zip_id, size)
    except (FileNotFoundError, KeyError) as e:
        return f'<div class="card missing">load failed: {tid} ({e})</div>'
    return (
        '<div class="card">'
        f'{rp["svg"]}'
        f'<div class="meta">'
        f'<span class="score">{sig:+.3f}</span>'
        f'<span class="tid">{tid} <span class="sz">{size}×{size}</span></span>'
        f'<span class="turn turn-{rp["to_move"][0]}">{rp["to_move"]} to play</span>'
        '</div>'
        '</div>'
    )


def _build_html(layer, channel, scope, bands, cols, lookup):
    def render_band(target, row_df):
        if row_df.empty:
            cards = '<div class="card missing">no problems left in this band</div>'
        else:
            cards = "".join(_build_card(r, lookup) for _, r in row_df.iterrows())
        return (
            f'<div class="band">'
            f'<div class="band-label">target {target:+.3f}</div>'
            f'<div class="grid">{cards}</div>'
            f'</div>'
        )

    bands_html = "".join(render_band(t, d) for t, d in bands)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>activation L{layer} ch{channel} ({scope})</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background:#fafafa; color:#222; }}
  h1 {{ margin-bottom: 0.25rem; }}
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
  .sz {{ color: #999; }}
  .turn-B {{ color: #111; }}
  .turn-W {{ color: #888; }}
  .missing {{ color: #c33; padding: 1rem; }}
  .legend {{ font-size: 0.85rem; color: #555; margin-bottom: 1rem; }}
  .legend .marker {{
    display: inline-block; width: 14px; height: 14px; border-radius: 50%;
    border: 2px solid #e63946; vertical-align: middle; margin: 0 0.2rem;
  }}
</style>
</head>
<body>
<h1>Raw activation — block {layer + 1} (rconv{layer + 1}), channel {channel}</h1>
<div class="legend">
  Feature index {layer * CHANNELS_PER_LAYER + channel} &middot; scope: {scope}
  &middot; Red ring <span class="marker"></span> = correct first move
  &middot; Rows step from the positive extreme (top) to the negative extreme (bottom).
  Raw activations are bidirectional — both tails can carry a motif.
</div>
{bands_html}
</body>
</html>
"""


# ---- main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, default=N_LAYERS - 1,
                        help=f"Residual block 0..{N_LAYERS - 1} (default {N_LAYERS - 1} = last)")
    parser.add_argument("--channel", type=int, default=0,
                        help=f"Channel 0..{CHANNELS_PER_LAYER - 1} (default 0)")
    parser.add_argument("--size", default="all",
                        help="9, 13, or 'all' (default; pools both sizes).")
    parser.add_argument("--html", action=argparse.BooleanOptionalAction, default=True,
                        help="Render static HTML (default). --no-html prints PWA URLs instead.")
    parser.add_argument("--rows", type=int, default=5,
                        help="HTML intensity bands spanning max→min (default 5)")
    parser.add_argument("--cols", type=int, default=5, help="HTML grid columns (default 5)")
    parser.add_argument("--n", type=int, default=10,
                        help="N problems per tail in --no-html mode (default 10)")
    parser.add_argument("--host", default="http://localhost:8787",
                        help="PWA dev server origin (only used with --no-html)")
    parser.add_argument("--out", default=None,
                        help="Output HTML path (default /tmp/act_L<layer>_ch<channel>.html)")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="Open the rendered HTML in the default browser.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force a rescan of activations/<size> into the cache.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Cache-build worker processes (default: all CPUs)")
    args = parser.parse_args()

    if not 0 <= args.layer < N_LAYERS:
        raise SystemExit(f"--layer must be in 0..{N_LAYERS - 1}")
    if not 0 <= args.channel < CHANNELS_PER_LAYER:
        raise SystemExit(f"--channel must be in 0..{CHANNELS_PER_LAYER - 1}")

    sizes = [9, 13] if args.size == "all" else [int(args.size)]

    frames = []
    for size in sizes:
        df = _load_signal(size, args.layer, args.channel, args.workers, args.rebuild_cache)
        df["size"] = size
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    lookup = load_lookup()
    df["zip_id"] = df["tsumego_id"].map(lambda t: lookup[t][0] if t in lookup else None)
    missing = int(df["zip_id"].isna().sum())
    if missing:
        print(f"[warn] {missing} problems not in lookup.txt (run package_for_pwa); dropping",
              flush=True)
        df = df.dropna(subset=["zip_id"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("no problems to show after joining lookup.txt")

    scope = "9x9 + 13x13" if args.size == "all" else f"{args.size}x{args.size}"

    if args.html:
        bands = _pick_full_range_bands(df, args.rows, args.cols)
        html = _build_html(args.layer, args.channel, scope, bands, args.cols, lookup)
        out_path = args.out or os.path.join("/tmp", f"act_L{args.layer}_ch{args.channel}.html")
        with open(out_path, "w") as f:
            f.write(html)
        print(f"wrote {out_path}")
        if args.open_browser:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(out_path)}")
        return

    base = f"{args.host}/?id="
    df_sorted = df.sort_values("signal", ascending=False)

    def fmt(row):
        return (f"  {row['signal']:>+8.3f} [{int(row['size'])}x{int(row['size'])}]  "
                f"{base}{row['tsumego_id']}")

    print(f"\nRaw activation L{args.layer} (rconv{args.layer + 1}) ch{args.channel} "
          f"[feat {args.layer * CHANNELS_PER_LAYER + args.channel}], scope {scope}\n")
    print(f"  top {args.n} (most positive):")
    for _, row in df_sorted.head(args.n).iterrows():
        print(fmt(row))
    print(f"\n  bottom {args.n} (most negative):")
    for _, row in df_sorted.tail(args.n).iloc[::-1].iterrows():
        print(fmt(row))


if __name__ == "__main__":
    main()
