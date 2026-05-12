"""Print URLs to a locally-running PWA dev server for the top-N high and
bottom-N low problems on a given PCA component, for visual inspection.

Reads from embeddings/joint/per_problem.csv when available (the new joint
model spans both board sizes); falls back to embeddings/<size>/per_problem.csv
for the legacy per-size pipeline.

Run the PWA via `cd pwa && npm run dev`, then open the URLs printed below.
The PWA's lookup table auto-resolves board size from the tsumego_id.

Alternative: --html generates a self-contained static HTML page with two
grids (top positive, top negative) of inline-SVG board diagrams marking
the correct first move. Use --open to launch it in the default browser.

Usage:
  python -m auxiliary.inspect_extreme_problems [--pc N] [--n 10] [--size 9|13|all]
  python -m auxiliary.inspect_extreme_problems --pc 3 --html --open
"""

import argparse
import os
import re
import sys
import zipfile

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ---- SGF parsing & Go board simulation ---------------------------------

def _parse_sgf(sgf_text):
    """Return (size, setup_moves, correct_move).

    setup_moves: list of (color, "ab") played in the linear chain before
    the first variation node. correct_move: (color, "ab") of the first
    move inside the first variation (the canonical correct answer).
    """
    size_m = re.search(r"SZ\[(\d+)\]", sgf_text)
    size = int(size_m.group(1)) if size_m else 9

    # Strip the leading outer "(" so we can find the first INNER "(".
    body = sgf_text.lstrip()
    if body.startswith("("):
        body = body[1:]
    first_inner = body.find("(")
    if first_inner == -1:
        setup_part, var_part = body, ""
    else:
        setup_part, var_part = body[:first_inner], body[first_inner + 1:]

    move_re = re.compile(r";\s*([BW])\[([a-z]{0,2})\]")
    setup = [(c, p) for c, p in move_re.findall(setup_part) if p and p != "tt"]
    first = move_re.search(var_part)
    correct = (first.group(1), first.group(2)) if first and first.group(2) and first.group(2) != "tt" else None
    return size, setup, correct


def _neighbors(x, y, size):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            yield nx, ny


def _group_and_libs(board, x, y, size):
    color = board[y][x]
    seen = set()
    group = []
    libs = set()
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in seen:
            continue
        seen.add((cx, cy))
        group.append((cx, cy))
        for nx, ny in _neighbors(cx, cy, size):
            v = board[ny][nx]
            if v == ".":
                libs.add((nx, ny))
            elif v == color and (nx, ny) not in seen:
                stack.append((nx, ny))
    return group, len(libs)


def _simulate(setup_moves, size):
    """Replay setup_moves with capture rules, return final 2D board."""
    board = [["." for _ in range(size)] for _ in range(size)]
    for color, coord in setup_moves:
        x = ord(coord[0]) - ord("a")
        y = ord(coord[1]) - ord("a")
        if not (0 <= x < size and 0 <= y < size):
            continue
        if board[y][x] != ".":
            # Shouldn't happen in well-formed SGFs, but skip rather than crash.
            continue
        board[y][x] = color
        opp = "W" if color == "B" else "B"
        for nx, ny in _neighbors(x, y, size):
            if board[ny][nx] == opp:
                grp, libs = _group_and_libs(board, nx, ny, size)
                if libs == 0:
                    for gx, gy in grp:
                        board[gy][gx] = "."
        # Self-capture (very rare in tsumego setups) — clear if no libs.
        grp, libs = _group_and_libs(board, x, y, size)
        if libs == 0:
            for gx, gy in grp:
                board[gy][gx] = "."
    return board


# ---- SVG rendering -----------------------------------------------------

_STAR_POINTS = {
    9: [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)],
    13: [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)],
    19: [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)],
}


def _render_svg(board, correct, size, cell=30, margin=18):
    side = 2 * margin + (size - 1) * cell
    out = [f'<svg viewBox="0 0 {side} {side}" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<rect width="{side}" height="{side}" fill="#dcb35c"/>')
    out.append('<g stroke="#222" stroke-width="1">')
    inner_lo = margin
    inner_hi = margin + (size - 1) * cell
    for i in range(size):
        v = margin + i * cell
        out.append(f'<line x1="{v}" y1="{inner_lo}" x2="{v}" y2="{inner_hi}"/>')
        out.append(f'<line x1="{inner_lo}" y1="{v}" x2="{inner_hi}" y2="{v}"/>')
    out.append('</g>')
    for sx, sy in _STAR_POINTS.get(size, []):
        cx, cy = margin + sx * cell, margin + sy * cell
        out.append(f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="#222"/>')
    r = cell * 0.46
    for y in range(size):
        for x in range(size):
            if board[y][x] != ".":
                cx, cy = margin + x * cell, margin + y * cell
                fill = "#111" if board[y][x] == "B" else "#fafafa"
                out.append(
                    f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="{fill}" '
                    f'stroke="#000" stroke-width="0.8"/>'
                )
    if correct:
        color, coord = correct
        cxc = ord(coord[0]) - ord("a")
        cyc = ord(coord[1]) - ord("a")
        cx, cy = margin + cxc * cell, margin + cyc * cell
        fill = "#111" if color == "B" else "#fafafa"
        out.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="{fill}" '
            f'fill-opacity="0.45" stroke="#e63946" stroke-width="2.5"/>'
        )
    out.append('</svg>')
    return "".join(out)


# ---- Lookup & SGF retrieval -------------------------------------------

def _load_lookup(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out[parts[0]] = (parts[1], int(parts[2]))
    return out


def _fetch_sgf(tsumego_id, zip_id, size):
    zip_path = os.path.join(
        PROJECT_ROOT, "pwa", "tsumegos", str(size), f"{zip_id}.zip"
    )
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"{tsumego_id}.sgf") as f:
            return f.read().decode("utf-8")


# ---- HTML page generation ---------------------------------------------

def _build_card(row, score_col, lookup):
    tid = row["tsumego_id"]
    score = float(row[score_col])
    info = lookup.get(tid)
    if not info:
        return f'<div class="card missing">missing in lookup: {tid}</div>'
    zip_id, size = info
    try:
        sgf = _fetch_sgf(tid, zip_id, size)
    except (FileNotFoundError, KeyError) as e:
        return f'<div class="card missing">load failed: {tid} ({e})</div>'
    sz, setup, correct = _parse_sgf(sgf)
    board = _simulate(setup, sz)
    svg = _render_svg(board, correct, sz)
    to_move = correct[0] if correct else "?"
    to_move_label = {"B": "Black", "W": "White"}.get(to_move, "?")
    return (
        '<div class="card">'
        f'{svg}'
        f'<div class="meta">'
        f'<span class="score">{score:+.3f}</span>'
        f'<span class="tid">{tid}</span>'
        f'<span class="turn turn-{to_move}">{to_move_label} to play</span>'
        '</div>'
        '</div>'
    )


def _build_html(pc, source, high_df, low_df, score_col, cols, lookup):
    high_cards = "".join(_build_card(r, score_col, lookup) for _, r in high_df.iterrows())
    low_cards = "".join(_build_card(r, score_col, lookup) for _, r in low_df.iterrows())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PC{pc} extremes ({source})</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background:#fafafa; color:#222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; }}
  .grid {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 1rem; }}
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
</style>
</head>
<body>
<h1>PC{pc} extremes</h1>
<div class="legend">
  Source: {source} &middot; Red ring <span class="marker"></span> = correct first move
</div>
<h2>Top {len(high_df)} HIGH (most positive on PC{pc})</h2>
<div class="grid">{high_cards}</div>
<h2>Top {len(low_df)} LOW (most negative on PC{pc})</h2>
<div class="grid">{low_cards}</div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc", type=int, default=0, help="Component index (default 0)")
    parser.add_argument("--n", type=int, default=10, help="N high + N low (default 10)")
    parser.add_argument("--host", default="http://localhost:8787",
                        help="PWA dev server origin")
    parser.add_argument("--size", default="all",
                        help="9, 13, or 'all' (default). 'all' picks across both sizes.")
    parser.add_argument("--html", action="store_true",
                        help="Render a static HTML grid instead of printing URLs.")
    parser.add_argument("--rows", type=int, default=3, help="HTML grid rows (default 3)")
    parser.add_argument("--cols", type=int, default=5, help="HTML grid columns (default 5)")
    parser.add_argument("--out", default=None,
                        help="Output HTML path (defaults to /tmp/pc<N>_extremes.html)")
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
        from utils.configurations import BOARD_SIZE
        csv_path = os.path.join(PROJECT_ROOT, "embeddings", str(BOARD_SIZE), "per_problem.csv")
        source = f"per-size ({BOARD_SIZE})"
    if not os.path.exists(csv_path):
        raise SystemExit("no per_problem.csv found (looked at joint and per-size locations)")

    df = pd.read_csv(csv_path)
    if args.size != "all" and "size" in df.columns:
        df = df[df["size"] == int(args.size)]
        if df.empty:
            raise SystemExit(f"no rows for size {args.size} in {csv_path}")

    col = f"embed_{args.pc}"
    if col not in df.columns:
        max_k = len([c for c in df.columns if c.startswith("embed_")]) - 1
        raise SystemExit(f"No column {col} in {csv_path} (max component index = {max_k})")

    n = args.rows * args.cols if args.html else args.n
    df_sorted = df.sort_values(col, ascending=False)
    high = df_sorted.head(n)
    low = df_sorted.tail(n).iloc[::-1]

    if args.html:
        lookup_path = os.path.join(PROJECT_ROOT, "pwa", "lookup.txt")
        if not os.path.exists(lookup_path):
            raise SystemExit(f"missing {lookup_path}; run pipeline.package_for_pwa first")
        lookup = _load_lookup(lookup_path)
        html = _build_html(args.pc, f"{source}, size {args.size}", high, low, col, args.cols, lookup)
        out_path = args.out or os.path.join("/tmp", f"pc{args.pc}_extremes.html")
        with open(out_path, "w") as f:
            f.write(html)
        print(f"wrote {out_path}")
        if args.open_browser:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(out_path)}")
        return

    base = f"{args.host}/?id="
    has_size = "size" in df.columns

    def fmt(row):
        size_tag = f" [{int(row['size'])}x{int(row['size'])}]" if has_size else ""
        return f"  {row[col]:>+7.3f}{size_tag}  {base}{row['tsumego_id']}"

    print(f"\nPC{args.pc} (source: {source}, scope: {args.size}): top {n} HIGH\n")
    for _, row in high.iterrows():
        print(fmt(row))
    print(f"\nPC{args.pc} (source: {source}, scope: {args.size}): top {n} LOW\n")
    for _, row in low.iterrows():
        print(fmt(row))


if __name__ == "__main__":
    main()
