import argparse
import glob
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time

# Retry policy for the (network-bound) wrangler calls.
RETRY_BACKOFFS = [3, 10]  # seconds between attempts; len == number of retries

# The one big, append-only table we pull incrementally by its AUTOINCREMENT id.
# Everything else is small and re-fetched whole, so the table list never has to
# be hardcoded (it is discovered from the remote schema) and schema changes to
# the other tables are tolerated automatically.
INCREMENTAL_TABLE = "tsumego_reports"

# Make the repo importable when this file is run by path, not just as
# `python -m pipeline.fetch_database` (which puts the root on sys.path itself).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from utils.paths import DB_DIR, DB_SQL, DB_SQLITE, PROJECT_ROOT  # noqa: E402

# Page size for the remote SELECTs (keeps any single wrangler response bounded).
FETCH_PAGE = 50000


# Where this run writes. Normally the canonical db/ from utils.paths; --db-dir
# redirects the WHOLE fetch into a private directory instead, leaving db/ and the
# wrangler dev D1 exactly as they are. A private snapshot is a throwaway read
# baseline for one consumer (auxiliary/update_leaderboard_ranks.mjs, which needs
# an up-to-the-second mirror inside a production freeze without spending the
# canonical one), so it skips both shared side effects: the dev-D1 sync and the
# db/d1.sql regeneration.
_db_dir = DB_DIR


def _paths():
    pwa_dir = os.path.join(PROJECT_ROOT, "pwa")
    os.makedirs(_db_dir, exist_ok=True)
    return (PROJECT_ROOT, pwa_dir, _db_dir,
            os.path.join(_db_dir, os.path.basename(DB_SQL)),
            os.path.join(_db_dir, os.path.basename(DB_SQLITE)))


# ---------------------------------------------------------------------------
# Full export (the heavy path) — unchanged: a complete D1 dump via wrangler.
# ---------------------------------------------------------------------------

def fetch_database():
    """
    Download the *entire* database from Cloudflare D1 using `wrangler d1 export`.

    This reads every row (D1 bills rows read), so it is the explicit `--full`
    path. The default path is the incremental fetch below.

    Retries the export up to len(RETRY_BACKOFFS) times on failure.
    """
    _, pwa_dir, _, output_file, _ = _paths()

    print(f"Fetching FULL database to {output_file}...")

    command = [
        "npx", "wrangler", "d1", "export", "DB",
        "--remote",
        f"--output={os.path.abspath(output_file)}",
    ]

    total_attempts = len(RETRY_BACKOFFS) + 1
    for attempt in range(total_attempts):
        try:
            subprocess.run(command, cwd=pwa_dir, check=True)
            print("\nDatabase downloaded successfully.")
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"\nError downloading database (attempt {attempt + 1}/{total_attempts}): {e}")
        except FileNotFoundError:
            print("\nError: 'npx' command not found. Ensure Node.js and npm are installed.")
            sys.exit(1)

        if attempt < len(RETRY_BACKOFFS):
            backoff = RETRY_BACKOFFS[attempt]
            print(f"Retrying in {backoff}s...")
            time.sleep(backoff)

    print(f"\nFailed to download database after {total_attempts} attempts.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Incremental fetch (the default path).
# ---------------------------------------------------------------------------

def _run_wrangler_json(command, pwa_dir):
    """Run `wrangler d1 execute --remote --json` for one SQL statement and
    return the parsed top-level JSON (a list of per-statement result objects).

    wrangler prints warnings to stderr, so stdout is clean JSON. Retries
    transient (non-JSON / network) failures; aborts on a real D1 query error.
    """
    cmd = [
        "npx", "wrangler", "d1", "execute", "DB",
        "--remote", "--json", "--command", command,
    ]
    total_attempts = len(RETRY_BACKOFFS) + 1
    for attempt in range(total_attempts):
        try:
            proc = subprocess.run(cmd, cwd=pwa_dir, capture_output=True, text=True)
        except FileNotFoundError:
            print("\nError: 'npx' command not found. Ensure Node.js and npm are installed.")
            sys.exit(1)

        out = (proc.stdout or "").strip()
        if out:
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                data = None
            if data is not None:
                if isinstance(data, dict) and data.get("error"):
                    # A STRUCTURED error is retried like any other failure. The
                    # Cloudflare account authorization check (code 7403) fails
                    # spuriously on a first call often enough that treating a
                    # structured error as fatal aborts routine fetches; the
                    # same call succeeds seconds later. A genuinely malformed
                    # query costs the backoff ladder before failing the same
                    # way, which is cheap next to a false abort mid-pipeline.
                    print(f"\nD1 query failed (attempt {attempt + 1}/"
                          f"{total_attempts}): {data['error']}\n  query: {command}")
                    if attempt < len(RETRY_BACKOFFS):
                        time.sleep(RETRY_BACKOFFS[attempt])
                        continue
                    sys.exit(1)
                if proc.returncode == 0:
                    return data
                # Nonzero exit but still emitted JSON without a structured
                # error: don't trust a possibly-partial result — an empty list
                # here would blank a just-DROPped table on refill. Fall through
                # to the retry/give-up path instead of returning it.

        err = (proc.stderr or "").strip()
        print(f"\nwrangler query failed (attempt {attempt + 1}/{total_attempts}): "
              f"{err[:300] or 'no output'}")
        if attempt < len(RETRY_BACKOFFS):
            time.sleep(RETRY_BACKOFFS[attempt])

    print(f"\nGiving up on wrangler query after {total_attempts} attempts.")
    sys.exit(1)


def _d1_query(command, pwa_dir):
    """Return the rows (list of dicts) of a single remote SELECT/PRAGMA."""
    data = _run_wrangler_json(command, pwa_dir)
    return data[0].get("results", []) if data else []


def _remote_schema(pwa_dir):
    """Discover the remote user objects (tables + indexes) with their CREATE
    SQL, excluding SQLite/Cloudflare internals. No table name is hardcoded, so
    new tables are picked up and dropped tables simply stop being refreshed."""
    return _d1_query(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL "
        "AND name NOT LIKE 'sqlite_%' "
        "AND name NOT LIKE '\\_%' ESCAPE '\\' "
        "AND name <> 'd1_migrations'",
        pwa_dir,
    )


def _remote_columns(table, pwa_dir):
    rows = _d1_query(f"SELECT name FROM pragma_table_info('{table}') ORDER BY cid", pwa_dir)
    return [r["name"] for r in rows]


def _local_columns(con, table):
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _insert_rows(con, table, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    collist = ",".join(f'"{c}"' for c in cols)
    placeholders = ",".join(["?"] * len(cols))
    con.executemany(
        f'INSERT INTO "{table}" ({collist}) VALUES ({placeholders})',
        [tuple(r.get(c) for c in cols) for r in rows],
    )


def _paged_pull(con, table, cursor_col, since, pwa_dir):
    """Insert rows from the remote `table` in ascending `cursor_col` order,
    paginated so no single wrangler response is unbounded. Returns the rows added.

    Shared by both fetch loops, which differ only in the cursor column:
      - whole-table refill uses `rowid` (aliased to __rid so it survives the `*`
        expansion, then stripped before insert) starting at since=-1;
      - the tsumego_reports incremental append uses the real AUTOINCREMENT `id`
        column starting at the local MAX(id).
    The generated SQL is byte-identical to the two prior hand-rolled loops."""
    use_rowid = cursor_col == "rowid"
    select = "rowid AS __rid, *" if use_rowid else "*"
    cursor_key = "__rid" if use_rowid else cursor_col
    last = since
    total = 0
    while True:
        rows = _d1_query(
            f'SELECT {select} FROM "{table}" '
            f'WHERE {cursor_col} > {last} ORDER BY {cursor_col} LIMIT {FETCH_PAGE}',
            pwa_dir,
        )
        if not rows:
            break
        last = rows[-1][cursor_key]
        if use_rowid:
            for r in rows:
                r.pop("__rid", None)
        _insert_rows(con, table, rows)
        total += len(rows)
        if len(rows) < FETCH_PAGE:
            break
    return total


def _refill_whole_table(con, obj, schema, pwa_dir):
    """Drop, recreate from the remote CREATE SQL (+ its indexes), and refill a
    small table wholesale (paginated by rowid). Recreating from the remote schema
    is what makes the fetch tolerant of schema changes to these tables."""
    name = obj["name"]
    con.execute(f'DROP TABLE IF EXISTS "{name}"')
    con.execute(obj["sql"])
    for idx in schema:
        if idx["type"] == "index" and idx["tbl_name"] == name and idx["sql"]:
            con.execute(idx["sql"])
    return _paged_pull(con, name, "rowid", -1, pwa_dir)


def _fetch_incremental(con, table, since_id, pwa_dir):
    """Append rows with id > since_id, paginated by the AUTOINCREMENT id."""
    return _paged_pull(con, table, "id", since_id, pwa_dir)


def _row_insert_lines(con, table, where=""):
    """Yield D1-export-compatible INSERT lines for a table, generated by SQLite
    itself (quote() handles all escaping). Uses a quoted table name + positional
    VALUES — the form every downstream `.sql` parser in the repo accepts."""
    cols = _local_columns(con, table)
    if not cols:
        return
    quoted = "||','||".join(f'quote("{c}")' for c in cols)
    sql = f'''SELECT 'INSERT INTO "{table}" VALUES('||{quoted}||');' FROM "{table}" {where}'''
    for (line,) in con.execute(sql):
        yield line


def emit_dump(con, sql_file):
    """Regenerate db/d1.sql from the canonical local SQLite so the
    many text-parsing consumers stay in sync with database.sqlite."""
    objs = con.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL "
        "AND name NOT LIKE 'sqlite_%' "
        "AND name NOT LIKE '\\_%' ESCAPE '\\' "
        "AND name <> 'd1_migrations'"  # match _remote_schema — a --full baseline
                                       # can carry d1_migrations; don't re-emit it
    ).fetchall()
    tables = [o for o in objs if o[0] == "table"]
    with open(sql_file, "w", encoding="utf-8") as f:
        f.write("PRAGMA defer_foreign_keys=TRUE;\n")
        for _typ, name, _tbl, sql in tables:
            f.write(sql.rstrip().rstrip(";") + ";\n")
            for ityp, _iname, itbl, isql in objs:
                if ityp == "index" and itbl == name and isql:
                    f.write(isql.rstrip().rstrip(";") + ";\n")
            for line in _row_insert_lines(con, name):
                f.write(line + "\n")


def incremental_fetch(do_local_d1=True, emit_sql_dump=True):
    """Default path: append only new `tsumego_reports` rows (by AUTOINCREMENT
    id) and re-fetch every other table whole, merging into the canonical
    db/d1.sqlite, then regenerate db/d1.sql and sync the
    wrangler dev (local) D1.

    Falls back to a full fetch if there is no local SQLite baseline yet.

    `emit_sql_dump=False` skips the text dump: a --db-dir snapshot has no
    text consumers, and regenerating one writes every row.
    """
    _, pwa_dir, _, sql_file, sqlite_file = _paths()

    if not os.path.exists(sqlite_file):
        print("No local database.sqlite baseline found; doing a full fetch to seed it.")
        return _full_fetch(do_local_d1)

    print("Incremental fetch: discovering remote schema...")
    schema = _remote_schema(pwa_dir)
    remote_tables = [o for o in schema if o["type"] == "table"]
    remote_names = {o["name"] for o in remote_tables}
    if INCREMENTAL_TABLE not in remote_names:
        print(f"Remote has no '{INCREMENTAL_TABLE}'; doing a full fetch instead.")
        return _full_fetch(do_local_d1)

    whole_tables = sorted(n for n in remote_names if n != INCREMENTAL_TABLE)

    con = sqlite3.connect(sqlite_file)
    try:
        # 1. Every other table: refresh wholesale (cheap; <1000 rows total).
        refreshed = 0
        for obj in remote_tables:
            if obj["name"] == INCREMENTAL_TABLE:
                continue
            refreshed += _refill_whole_table(con, obj, schema, pwa_dir)

        # 2. tsumego_reports: append new rows by id. If the schema drifted (or
        #    the table is missing locally), re-pull it whole from the remote DDL.
        tsumego_obj = next(o for o in remote_tables if o["name"] == INCREMENTAL_TABLE)
        local_cols = _local_columns(con, INCREMENTAL_TABLE)
        remote_cols = _remote_columns(INCREMENTAL_TABLE, pwa_dir)
        if not local_cols or local_cols != remote_cols:
            print(f"  {INCREMENTAL_TABLE} schema changed/missing locally; re-pulling it whole.")
            con.execute(f'DROP TABLE IF EXISTS "{INCREMENTAL_TABLE}"')
            con.execute(tsumego_obj["sql"])
            for idx in schema:
                if idx["type"] == "index" and idx["tbl_name"] == INCREMENTAL_TABLE and idx["sql"]:
                    con.execute(idx["sql"])
            since_id = 0
        else:
            since_id = con.execute(
                f'SELECT COALESCE(MAX(id), 0) FROM "{INCREMENTAL_TABLE}"'
            ).fetchone()[0]

        new_count = _fetch_incremental(con, INCREMENTAL_TABLE, since_id, pwa_dir)
        con.commit()

        print(f"  +{new_count} new {INCREMENTAL_TABLE} rows; "
              f"{len(whole_tables)} other table(s) refreshed ({refreshed} rows).")

        # 3. Regenerate the text dump from the merged SQLite.
        if emit_sql_dump:
            print(f"Regenerating {sql_file} from {sqlite_file}...")
            emit_dump(con, sql_file)
    finally:
        con.close()

    # 4. Mirror the canonical SQLite into the wrangler dev (local) D1.
    if do_local_d1:
        sync_local_d1(sqlite_file, sql_file, pwa_dir)

    print("Incremental fetch complete.")
    return sql_file


def _full_fetch(do_local_d1):
    """Full export + rebuild the SQLite baseline + sync local D1."""
    _, pwa_dir, _, sql_file, sqlite_file = _paths()
    fetch_database()
    rebuild_sqlite(sql_file)
    if do_local_d1:
        sync_local_d1(sqlite_file, sql_file, pwa_dir)
    return sql_file


# ---------------------------------------------------------------------------
# Local artifacts: binary SQLite + wrangler dev (local) D1.
# ---------------------------------------------------------------------------

def rebuild_sqlite(sql_file):
    """
    Reconstruct the local binary SQLite DB (db/d1.sqlite) from the
    raw D1 dump (db/d1.sql).

    The dump is removed-and-rebuilt rather than re-imported: D1 emits plain
    `INSERT`s with no conflict handling, so importing into an existing DB would
    duplicate every row. The whole dump is wrapped in a single BEGIN/COMMIT so
    sqlite doesn't fsync per row (~290k single-row INSERTs otherwise take many
    minutes). Never feed the dump through head/tail — a SIGPIPE mid-import leaves
    a silently partial table.
    """
    sqlite_file = os.path.splitext(sql_file)[0] + ".sqlite"

    if not os.path.exists(sql_file):
        print(f"\nError: dump not found at {sql_file}; cannot rebuild SQLite DB.")
        sys.exit(1)

    # Import into a temp DB and os.replace onto the canonical path only after a
    # clean import. A failed import must NOT leave a partial database.sqlite —
    # the next incremental run would adopt it as its id-cursor baseline and
    # silently skip every row the truncated import missed.
    tmp_sqlite = sqlite_file + ".tmp"
    if os.path.exists(tmp_sqlite):
        os.remove(tmp_sqlite)

    print(f"Rebuilding local SQLite DB at {sqlite_file}...")

    def _cleanup_tmp():
        if os.path.exists(tmp_sqlite):
            os.remove(tmp_sqlite)

    try:
        with open(sql_file, "rb") as dump:
            proc = subprocess.Popen(
                ["sqlite3", tmp_sqlite],
                stdin=subprocess.PIPE,
            )
            proc.stdin.write(b"BEGIN;\n")
            for chunk in iter(lambda: dump.read(1 << 20), b""):
                proc.stdin.write(chunk)
            proc.stdin.write(b"\nCOMMIT;\n")
            proc.stdin.close()
            rc = proc.wait()
        if rc != 0:
            _cleanup_tmp()
            print(f"\nError: sqlite3 import exited with status {rc}.")
            sys.exit(1)
    except FileNotFoundError:
        _cleanup_tmp()
        print("\nError: 'sqlite3' command not found. Install the sqlite3 CLI.")
        sys.exit(1)

    os.replace(tmp_sqlite, sqlite_file)
    print("Local SQLite DB rebuilt successfully.")


def _table_names_from_dump(sql_file):
    """Return the table names defined in the D1 dump, in declaration order.

    Matches both `CREATE TABLE x (` and `CREATE TABLE IF NOT EXISTS "x" (`.
    """
    pat = re.compile(
        r'^CREATE TABLE(?:\s+IF NOT EXISTS)?\s+["\']?([A-Za-z_][A-Za-z0-9_]*)["\']?',
        re.IGNORECASE,
    )
    names = []
    seen = set()
    with open(sql_file, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                names.append(m.group(1))
    return names


def update_local_d1(sql_file):
    """
    Mirror the freshly-fetched dump into wrangler's *local* D1 — the miniflare
    database under pwa/.wrangler/state that `npm run dev` serves — so local dev
    runs against production data. This is the full path (drop + reimport).

    D1's export emits bare `CREATE TABLE` (no IF NOT EXISTS) for most tables, so
    re-importing the dump over an existing local DB would error on the first
    CREATE. We therefore: (1) DROP the dump's tables, (2) import the dump
    (recreates tables + data), (3) apply the idempotent schema.sql to (re)create
    any tables/indexes the dump omits — D1 export drops empty tables, and a dump
    can lag a newly-added index. All three run via `wrangler d1 execute DB
    --local` from the pwa dir, where the `DB` binding in wrangler.toml resolves.

    A wrangler failure here is non-fatal: the dump (the pipeline's actual
    product) is already on disk, so we warn and return rather than aborting.
    """
    _, pwa_dir, _, _, _ = _paths()
    schema_file = os.path.join(pwa_dir, "schema.sql")

    tables = _table_names_from_dump(sql_file)
    if not tables:
        print("\nWarning: no CREATE TABLE statements found in dump; skipping local D1 update.")
        return

    print("Updating local wrangler D1 (the `npm run dev` database)...")

    base = ["npx", "wrangler", "d1", "execute", "DB", "--local"]
    steps = [
        # 1. Drop the dump's tables so its bare CREATE TABLEs don't collide.
        base + ["--command", "".join(f'DROP TABLE IF EXISTS "{t}";' for t in tables)],
        # 2. Recreate the tables and load their rows from the fetched dump.
        base + [f"--file={os.path.abspath(sql_file)}"],
    ]
    # 3. Backfill tables/indexes the dump omits (empty tables, newer indexes).
    if os.path.exists(schema_file):
        steps.append(base + [f"--file={schema_file}"])

    try:
        for cmd in steps:
            subprocess.run(cmd, cwd=pwa_dir, check=True)
    except FileNotFoundError:
        print("\nError: 'npx' command not found. Ensure Node.js and npm are installed.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nWarning: failed to update local wrangler D1: {e}")
        print("(The dump was fetched fine; only the local dev DB was not refreshed.)")
        return

    print("Local wrangler D1 updated successfully.")


def _miniflare_d1_file(pwa_dir):
    """Locate the single miniflare D1 SQLite file backing the `DB` binding, or
    None. miniflare stores it as <hash>.sqlite alongside an internal
    `metadata.sqlite` under .wrangler/state/v3/d1/miniflare-D1DatabaseObject/."""
    d1_dir = os.path.join(pwa_dir, ".wrangler", "state", "v3", "d1",
                          "miniflare-D1DatabaseObject")
    if not os.path.isdir(d1_dir):
        return None
    candidates = [
        f for f in glob.glob(os.path.join(d1_dir, "*.sqlite"))
        if os.path.basename(f) != "metadata.sqlite"
    ]
    return candidates[0] if len(candidates) == 1 else None


def sync_local_d1(sqlite_file, sql_file, pwa_dir):
    """Mirror the canonical db/d1.sqlite into wrangler's dev (local)
    D1 by copying the file directly onto the miniflare D1 SQLite — far faster
    than a wrangler reimport (verified: D1 reads/writes work fine afterwards;
    its internal `_cf_*` tables aren't required). The stale -wal/-shm sidecars
    are removed so the fresh file isn't shadowed.

    If the local D1 file doesn't exist yet (dev never run), fall back to the
    wrangler import, which creates it. NOTE: stop `npm run dev` before fetching —
    overwriting the DB file underneath a live workerd can corrupt its view.
    """
    target = _miniflare_d1_file(pwa_dir)
    if target is None:
        print("Local dev D1 not found; seeding it via wrangler import "
              "(run once; subsequent fetches will file-copy).")
        update_local_d1(sql_file)
        return

    print(f"Copying canonical SQLite onto local dev D1 ({os.path.basename(target)})...")
    shutil.copyfile(sqlite_file, target)
    for ext in ("-wal", "-shm"):
        try:
            os.remove(target + ext)
        except FileNotFoundError:
            pass
    print("Local dev D1 updated (file copy). Restart `npm run dev` if it's running.")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch the D1 database into db/d1.{sql,sqlite} and "
                    "mirror it into wrangler's local dev D1. By default only NEW "
                    "tsumego_reports rows are pulled (by AUTOINCREMENT id) and the "
                    "other (small) tables are re-fetched whole — far fewer D1 rows "
                    "read than a full export, while staying up to date."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-download the entire database via `wrangler d1 export` (reads "
             "every row) instead of the incremental fetch. Also used automatically "
             "when there is no local database.sqlite baseline yet.",
    )
    parser.add_argument(
        "--no-local-d1",
        action="store_true",
        help="Skip refreshing wrangler's local D1 (the `npm run dev` database).",
    )
    parser.add_argument(
        "--db-dir",
        metavar="DIR",
        help="Fetch into DIR/d1.sqlite instead of the canonical db/, leaving db/ "
             "and the dev D1 untouched. DIR/d1.sqlite is the incremental fetch's "
             "id baseline, so copy the canonical one there first for a cheap pull "
             "(without it this seeds DIR with a --full export). A private snapshot "
             "has no text consumers and no dev-D1 role, so it also skips the "
             "db/d1.sql regeneration and the local-D1 sync.",
    )
    args = parser.parse_args()

    private = args.db_dir is not None
    if private:
        global _db_dir
        _db_dir = os.path.abspath(args.db_dir)
        print(f"Private snapshot: fetching into {_db_dir} "
              "(db/ and the dev D1 are left as they are).")

    if args.full:
        _full_fetch(do_local_d1=not args.no_local_d1 and not private)
    else:
        incremental_fetch(do_local_d1=not args.no_local_d1 and not private,
                          emit_sql_dump=not private)


if __name__ == "__main__":
    main()
