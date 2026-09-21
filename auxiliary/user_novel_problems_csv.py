#!/usr/bin/env python3
"""Write a CSV of ONE user's novel problems, in the order they met them.

"Novel" is `tsumego_reports.is_repeat = 0` — the first time the user saw the
problem — so a problem answered again under the recall schedule contributes one
row, not several. Order is the row's `user_seq` (the user's own chronological
attempt counter, the same value the history list is built from), falling back to
the AUTOINCREMENT `id` when `user_seq` is NULL on a legacy row.

For each row the problem's rank is derived EXACTLY the way the client derives
`problemRank`: raw rating (`ratings/<size>/tsumego_ratings.csv`, shipped to the
client in `problem_params.bin`) put through the curve M
(`ratings/rating_to_rank.json`, `applyM` in `pwa/shared/rank-window.js`), whose
values are already on the project rank scale — 0 is the 1k/1d boundary, one unit
per rank, integers are band EDGES (see `pwa/shared/rank-math.js`). The kyu/dan
label is that value's band floor, so 2.5 is "3d" and -0.001 is "1k". No time dock
is applied: alpha*(log10 median solve min) is a per-USER adjustment fitted in
`map_rating_to_rank`, and a problem has no solve time of its own.

The problem link is the app's share URL — `<site>/<size>?id=<8hex>` (see
`pwa/js/share.js`) — which re-enters that exact problem.

Alongside the CSV a self-contained HTML table is written (same rows, rank as a
colour-coded kyu/dan badge) because Chrome DOWNLOADS a `.csv` instead of
rendering it — `google-chrome file.csv` shows nothing at all. Open the HTML to
read the result in a browser.

Usage:
    ./.venv/bin/python3 -m auxiliary.user_novel_problems_csv --user-id <user-id>
    ./.venv/bin/python3 -m auxiliary.user_novel_problems_csv --find-name "<display name>"
    ./.venv/bin/python3 -m auxiliary.user_novel_problems_csv --user-id <user-id> --out /tmp/user.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.rating.ratings_io import (  # noqa: E402
    internal_to_server,
    interp_extrap,
    load_rank_map,
    load_rating_rank_curve,
)

DEFAULT_DB = os.path.join(PROJECT_ROOT, "db", "d1.sqlite")
DEFAULT_RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")
DEFAULT_SITE = "https://onlymove.net"
DEFAULT_URL_SIZE = 9

# The band-label clamp and the AGA-style ladder, mirroring shared/rank-math.js.
RANK_BAND_LO, RANK_BAND_HI = -30, 8


def rank_band(value: float) -> int:
    """Which whole rank a continuous rank falls in (FLOOR: integers are edges)."""
    return math.floor(value)


def rank_to_label(value: float) -> str:
    """Rank scalar -> its kyu/dan label; 0 -> 1d, -0.001 -> 1k, 2.5 -> 3d."""
    return f"{-rank_band(value)}k" if rank_band(value) < 0 else f"{rank_band(value) + 1}d"


def clamp_rank_band(value: float) -> float:
    """Clamp before LABELLING only (never before computing) — 30k..9d."""
    return min(float(RANK_BAND_HI), max(float(RANK_BAND_LO), value))


def load_ratings(ratings_dir: str, sizes) -> dict:
    """{size: {tsumego_id: raw_rating}} from the per-size ratings CSVs."""
    out = {}
    for size in sizes:
        path = os.path.join(ratings_dir, str(size), "tsumego_ratings.csv")
        by_id = {}
        if os.path.exists(path):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        by_id[row["tsumego_id"]] = float(row["rating"])
                    except (KeyError, TypeError, ValueError):
                        continue
        out[size] = by_id
    return out


def resolve_user_id(db_path: str, user_id: str | None, name: str | None) -> str:
    """An explicit id is used as given; `--find-name` resolves a display name."""
    if user_id:
        return user_id
    if not name:
        raise SystemExit("give --user-id or --find-name")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT user_id, display_name FROM external_ids "
            "WHERE display_name LIKE ? COLLATE NOCASE ORDER BY user_id",
            (f"%{name}%",),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        raise SystemExit(f"no external_ids row matches name {name!r}")
    print(f"name {name!r} matched {len(rows)} account(s):", file=sys.stderr)
    for uid, display in rows:
        print(f"  {uid}  {display!r}", file=sys.stderr)
    if len(rows) > 1:
        raise SystemExit("ambiguous name — pass --user-id with one of the ids above")
    return rows[0][0]


# --- HTML rendering --------------------------------------------------------------
# Chrome cannot display a .csv (it downloads it), so the same rows are also
# written as one self-contained page with the rank as the headline column.

_HTML_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:16px 20px 40px; background:#faf9f7; color:#1c1c1e;
         font:13.5px/1.5 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif; }}
  h1 {{ font-size:18px; margin:0 0 2px; }}
  h1 .uid {{ font-weight:400; opacity:.6; font-size:13px; }}
  .sub {{ opacity:.75; margin-bottom:12px; }}
  .sub code {{ font-size:12px; background:rgba(128,128,128,.15); padding:1px 4px; border-radius:3px; }}
  .panel {{ display:flex; gap:24px; flex-wrap:wrap; align-items:flex-start;
            background:#fff; border:1px solid rgba(128,128,128,.25); border-radius:8px;
            padding:12px 14px; margin-bottom:14px; }}
  .panel h2 {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase;
               opacity:.6; margin:0 0 6px; font-weight:600; }}
  .stats {{ display:grid; grid-template-columns:auto auto; gap:2px 14px; align-content:start; }}
  .stats b {{ font-variant-numeric:tabular-nums; }}
  .stats span {{ opacity:.7; }}
  .bar {{ display:flex; align-items:center; gap:6px; font-size:11.5px; line-height:14px; }}
  .bl {{ width:26px; text-align:right; opacity:.75; font-variant-numeric:tabular-nums; }}
  .bf {{ height:9px; background:linear-gradient(90deg,#4c8dff,#8b5cf6); border-radius:2px; }}
  .bc {{ opacity:.6; font-variant-numeric:tabular-nums; }}
  .controls {{ margin-left:auto; display:flex; gap:8px; align-items:center; }}
  input[type=search] {{ font:inherit; padding:5px 9px; border-radius:6px;
      border:1px solid rgba(128,128,128,.4); background:transparent; color:inherit; width:220px; }}
  .chk {{ display:flex; gap:5px; align-items:center; user-select:none; }}
  table {{ border-collapse:collapse; width:100%; background:#fff;
           border:1px solid rgba(128,128,128,.25); border-radius:8px; overflow:hidden; }}
  th, td {{ padding:5px 10px; border-bottom:1px solid rgba(128,128,128,.16);
            text-align:left; white-space:nowrap; }}
  thead th {{ position:sticky; top:0; background:#f0eeea; z-index:2; font-weight:600;
              font-size:11.5px; letter-spacing:.03em; text-transform:uppercase; }}
  tbody tr:nth-child(even) {{ background:#f7f6f3; }}
  tbody tr:hover {{ background:#eef4ff; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.rank {{ width:1%; }}
  .badge {{ display:inline-block; min-width:44px; text-align:center; padding:2px 9px;
            border-radius:999px; font-weight:700; font-size:13px;
            font-variant-numeric:tabular-nums; }}
  .kyu  {{ background:#e8effc; color:#17458f; }}
  .dan  {{ background:#f3e8fd; color:#6b21a8; }}
  td.ok {{ color:#1a7f37; font-weight:600; }}
  td.no {{ color:#c0392b; font-weight:600; }}
  a {{ color:#0a58ca; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  tfoot td {{ padding:6px 10px; opacity:.7; border:0; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#16181c; color:#e8e8ea; }}
    .panel, table {{ background:#1e2127; }}
    thead th {{ background:#262a31; }}
    tbody tr:nth-child(even) {{ background:#1a1d23; }}
    tbody tr:hover {{ background:#2a2f38; }}
    a {{ color:#7fb2ff; }}
    .kyu {{ background:#1d2a44; color:#a9c7ff; }}
    .dan {{ background:#2c1d3d; color:#dcb6ff; }}
    td.ok {{ color:#7bd88f; }} td.no {{ color:#ff8a7a; }}
  }}
</style></head><body>
<h1>{who}</h1>
<div class="sub">{subtitle}</div>
<div class="panel">
  <div class="stats"><h2 style="grid-column:1/-1">Summary</h2>
    <span>Attempts</span><b>{n}</b>
    <span>Correct</span><b>{correct}</b>
    <span>Incorrect</span><b>{wrong}</b>
    <span>Rank range</span><b>{rank_lo} &rarr; {rank_hi}</b>
  </div>
  <div><h2>Problems per rank</h2>{bars}</div>
  <div class="controls">
    <input id="q" type="search" placeholder="Filter by rank, id or link…">
    <label class="chk"><input id="onlyWrong" type="checkbox"> wrong only</label>
  </div>
</div>
<table>
  <thead><tr><th>#</th><th>Rank</th><th>Problem</th><th>Link</th><th>Rating</th>
  <th>{server_head}</th><th>Answer</th><th>Move</th><th>Time (s)</th><th>Attempted at</th>
  </tr></thead>
  <tbody id="tb"></tbody>
  <tfoot><tr><td colspan="10" id="count"></td></tr></tfoot>
</table>
<script>
const DATA = {data};
const tb = document.getElementById('tb');
const count = document.getElementById('count');
const fmt = new Intl.NumberFormat();
function render() {{
  const q = document.getElementById('q').value.trim().toLowerCase();
  const onlyWrong = document.getElementById('onlyWrong').checked;
  const frag = document.createDocumentFragment();
  let shown = 0;
  for (const r of DATA) {{
    if (onlyWrong && r.ok) continue;
    if (q) {{
      const hay = (r.rank + ' ' + r.rv + ' ' + r.id + ' ' + r.link + ' ' + r.ogs).toLowerCase();
      if (!hay.includes(q)) continue;
    }}
    shown++;
    const tr = document.createElement('tr');
    const dan = parseFloat(r.rv) >= 0;
    tr.innerHTML =
      '<td class="num">' + r.o + '</td>' +
      '<td class="rank"><span class="badge ' + (dan ? 'dan' : 'kyu') + '">' + r.rank + '</span></td>' +
      '<td>' + r.id + '</td>' +
      '<td><a href="' + r.link + '" target="_blank" rel="noopener">' + r.link + '</a></td>' +
      '<td class="num">' + Number(r.rt).toFixed(3) + '</td>' +
      '<td class="num">' + r.ogs + '</td>' +
      '<td class="' + (r.ok ? 'ok' : 'no') + '">' + (r.ok ? 'correct' : 'incorrect') + '</td>' +
      '<td>' + r.mv + '</td>' +
      '<td class="num">' + (r.t ? fmt.format(r.t) : '') + '</td>' +
      '<td>' + r.at.replace('T', ' ').replace(/\\.\\d+Z$/, '') + '</td>';
    frag.appendChild(tr);
  }}
  tb.replaceChildren(frag);
  count.textContent = 'Showing ' + shown + ' of ' + DATA.length + ' problems';
}}
document.getElementById('q').addEventListener('input', render);
document.getElementById('onlyWrong').addEventListener('change', render);
render();
</script>
</body></html>"""


def html_cell(value) -> str:
    return html.escape(str(value), quote=True)


def write_html(rows, path, *, user_id, name, csv_path, server_head) -> None:
    """Render the same rows as a filterable table, rank as a kyu/dan badge."""
    bands = {}
    for r in rows:
        bands[rank_band(r["rank_value"])] = bands.get(rank_band(r["rank_value"]), 0) + 1
    top = max(bands.values())
    bars = "".join(
        f'<div class="bar"><span class="bl">{html_cell(rank_to_label(b))}</span>'
        f'<span class="bf" style="width:{max(2, round(120 * bands[b] / top))}px"></span>'
        f'<span class="bc">{bands[b]}</span></div>'
        for b in sorted(bands)
    )
    correct = sum(1 for r in rows if r["success"])
    data = [
        {
            "o": i,
            "id": r["problem_id"],
            "link": r["problem_link"],
            "rank": r["rank_label"],
            "rv": r["rank_value"],
            "rt": r["rating"],
            "ogs": r["server_rank"],
            "ok": bool(r["success"]),
            "mv": r["move"],
            "t": r["time_used"] if r["time_used"] is not None else "",
            "at": r["created_at"],
        }
        for i, r in enumerate(rows, 1)
    ]
    who = f'{html_cell(name)} <span class="uid">user_id {html_cell(user_id)}</span>' \
        if name else f'<span class="uid">user_id {html_cell(user_id)}</span>'
    subtitle = (
        f"{len(rows)} novel problems in the order they were met &middot; rank derived from the raw "
        f"rating through curve M, the same way the app does &middot; source "
        f"<code>{html_cell(csv_path)}</code>"
    )
    body = _HTML_HEAD.format(
        title=f"{name or user_id} — novel problems by rank",
        who=who,
        subtitle=subtitle,
        n=len(rows),
        correct=correct,
        wrong=len(rows) - correct,
        rank_lo=rank_to_label(min(bands)),
        rank_hi=rank_to_label(max(bands)),
        bars=bars,
        server_head=server_head or "Rank",
        data=_json_for_script(data),
    )
    with open(path, "w") as f:
        f.write(body)


def _json_for_script(obj) -> str:
    """JSON safe to inline in a <script> block (escape the sequence `</`)."""
    return json.dumps(obj).replace("</", "<\\/")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user-id", default=os.environ.get("ONLYMOVE_USER_ID"),
                    help="account id in tsumego_reports.user_id")
    ap.add_argument("--find-name", help="resolve the id from external_ids.display_name")
    ap.add_argument("--db", default=DEFAULT_DB, help="path to the local d1.sqlite snapshot")
    ap.add_argument("--ratings-dir", default=DEFAULT_RATINGS_DIR,
                    help="dir holding <size>/tsumego_ratings.csv + rating_to_rank.json")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: <project>/tmp/<user_id>_novel_problems.csv)")
    ap.add_argument("--site", default=DEFAULT_SITE, help="base URL for problem links")
    ap.add_argument("--url-size", type=int, default=DEFAULT_URL_SIZE,
                    help="board size used in the link path (problems are one size in practice)")
    ap.add_argument("--server", default="OGS",
                    help="server coupling for the reference rank column (blank to omit)")
    ap.add_argument("--all-attempts", action="store_true",
                    help="include repeat attempts too (default: novel only)")
    ap.add_argument("--no-html", action="store_true",
                    help="write only the CSV, not the browser-viewable table")
    args = ap.parse_args(argv)

    user_id = resolve_user_id(args.db, args.user_id, args.find_name)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        display_name = con.execute(
            "SELECT display_name FROM external_ids WHERE user_id = ?", (user_id,)
        ).fetchone()
        name = display_name[0] if display_name else None
        sql = """
            SELECT id, user_seq, problem, success, created_at, size, move,
                   time_used, review_time, is_repeat
              FROM tsumego_reports
             WHERE user_id = ?{repeat_filter}
             ORDER BY COALESCE(user_seq, id), id
        """.format(repeat_filter="" if args.all_attempts else " AND is_repeat = 0")
        attempts = [dict(r) for r in con.execute(sql, (user_id,))]
    finally:
        con.close()

    if not attempts:
        raise SystemExit(f"no {'rows' if args.all_attempts else 'novel rows'} for user {user_id}")

    sizes = sorted({int(a["size"]) for a in attempts})
    ratings_by_size = load_ratings(args.ratings_dir, sizes)

    # M is written identical for both board keys; size 9's key stands in for either.
    curve_cache = {}
    rank_map = load_rank_map(args.ratings_dir) if args.server else {}

    def curve_for(size: int):
        if size not in curve_cache:
            curve_cache[size] = load_rating_rank_curve(args.ratings_dir, size=size)
        return curve_cache[size]

    out_path = args.out or os.path.join(
        PROJECT_ROOT, "tmp", f"{user_id}_novel_problems.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    server_head = f"rank on {args.server}" if args.server else ""
    header = ["order", "user_seq", "problem_id", "problem_link", "board_size",
              "problem_rank", "problem_rank_value", "problem_rating"]
    if args.server:
        header.append(f"rank_on_{args.server}")
    header += ["answer", "correct", "move", "time_used_s", "attempted_at", "is_repeat"]

    missing_rating = 0
    records = []
    for order, a in enumerate(attempts, 1):
        size = int(a["size"])
        tid = a["problem"]
        rating = ratings_by_size.get(size, {}).get(tid)
        if rating is None:
            missing_rating += 1
            rank_value = None
        else:
            xs, ys = curve_for(size)
            rank_value = interp_extrap(rating, xs, ys)
        rank_label = rank_to_label(clamp_rank_band(rank_value)) if rank_value is not None else ""
        server_rank = ""
        if args.server and rank_value is not None:
            server_scalar = internal_to_server(args.server, rank_value, rank_map)
            if math.isfinite(server_scalar):
                server_rank = rank_to_label(clamp_rank_band(server_scalar))
        records.append({
            "order": order,
            "user_seq": a["user_seq"] if a["user_seq"] is not None else "",
            "problem_id": tid,
            "problem_link": f"{args.site.rstrip('/')}/{args.url_size}?id={tid}",
            "board_size": size,
            "rank_label": rank_label,
            "rank_value": round(rank_value, 4) if rank_value is not None else "",
            "rating": round(rating, 6) if rating is not None else "",
            "server_rank": server_rank,
            "success": bool(a["success"]),
            "move": a["move"] or "",
            "time_used": a["time_used"] if a["time_used"] is not None else "",
            "created_at": a["created_at"],
            "is_repeat": int(a["is_repeat"]),
        })

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            row = [r["order"], r["user_seq"], r["problem_id"], r["problem_link"],
                   r["board_size"], r["rank_label"], r["rank_value"], r["rating"]]
            if args.server:
                row.append(r["server_rank"])
            row += ["correct" if r["success"] else "incorrect", int(r["success"]),
                    r["move"], r["time_used"], r["created_at"], r["is_repeat"]]
            w.writerow(row)

    html_path = None
    if not args.no_html:
        html_path = os.path.splitext(out_path)[0] + ".html"
        write_html(records, html_path, user_id=user_id, name=name,
                   csv_path=out_path, server_head=server_head)

    print(f"user {user_id}" + (f" ({name})" if name else ""))
    print(f"  novel rows: {len(records)}  →  {out_path}")
    if html_path:
        print(f"  browser view (Chrome cannot display a .csv): {html_path}")
    print(f"  correct: {sum(1 for r in records if r['success'])} / {len(records)}")
    if missing_rating:
        print(f"  WARNING: {missing_rating} row(s) had no rating (left blank)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
