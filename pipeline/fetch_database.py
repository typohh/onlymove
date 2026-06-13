import argparse
import os
import re
import subprocess
import sys
import time

# Retry policy for the (network-bound) wrangler export.
RETRY_BACKOFFS = [3, 15]  # seconds between attempts; len == number of retries


def fetch_database():
    """
    Downloads the latest database from Cloudflare D1 using wrangler.
    The command run is: npx wrangler d1 export DB --remote --output=../attempts/database.sql
    It must be run from the 'pwa' directory context.

    Retries the export up to len(RETRY_BACKOFFS) times on failure, sleeping
    RETRY_BACKOFFS[i] seconds before each retry.
    """
    # Define paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pwa_dir = os.path.join(project_root, "pwa")
    attempts_dir = os.path.join(project_root, "attempts")
    output_file = os.path.join(attempts_dir, "database.sql")

    print(f"Fetching database to {output_file}...")

    # Ensure attempts directory exists
    os.makedirs(attempts_dir, exist_ok=True)

    # Construct the command
    # We use explicit relative path for output to match the original command intent,
    # but running it via subprocess with cwd set to pwa_dir is cleaner.
    command = [
        "npx", "wrangler", "d1", "export", "DB",
        "--remote",
        f"--output={os.path.abspath(output_file)}"
    ]

    total_attempts = len(RETRY_BACKOFFS) + 1
    for attempt in range(total_attempts):
        try:
            # Run the command from the pwa directory
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


def rebuild_sqlite(sql_file):
    """
    Reconstructs the local binary SQLite DB (attempts/database.sqlite) from the
    raw D1 dump (attempts/database.sql).

    The dump is removed-and-rebuilt rather than re-imported: D1 emits plain
    `INSERT`s with no conflict handling, so importing into an existing DB would
    duplicate every row. The whole dump is wrapped in a single BEGIN/COMMIT so
    sqlite doesn't fsync per row (~230k single-row INSERTs otherwise take many
    minutes). Never feed the dump through head/tail — a SIGPIPE mid-import leaves
    a silently partial table.
    """
    sqlite_file = os.path.splitext(sql_file)[0] + ".sqlite"

    if not os.path.exists(sql_file):
        print(f"\nError: dump not found at {sql_file}; cannot rebuild SQLite DB.")
        sys.exit(1)

    # Drop the old binary DB for a clean reconstruction (avoids duplicate rows).
    if os.path.exists(sqlite_file):
        os.remove(sqlite_file)

    print(f"Rebuilding local SQLite DB at {sqlite_file}...")

    # { echo "BEGIN;"; cat database.sql; echo "COMMIT;"; } | sqlite3 database.sqlite
    try:
        with open(sql_file, "rb") as dump:
            proc = subprocess.Popen(
                ["sqlite3", sqlite_file],
                stdin=subprocess.PIPE,
            )
            proc.stdin.write(b"BEGIN;\n")
            # Stream the dump straight through; never via head/tail.
            for chunk in iter(lambda: dump.read(1 << 20), b""):
                proc.stdin.write(chunk)
            proc.stdin.write(b"\nCOMMIT;\n")
            proc.stdin.close()
            rc = proc.wait()
        if rc != 0:
            print(f"\nError: sqlite3 import exited with status {rc}.")
            sys.exit(1)
    except FileNotFoundError:
        print("\nError: 'sqlite3' command not found. Install the sqlite3 CLI.")
        sys.exit(1)

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
    runs against production data.

    D1's export emits bare `CREATE TABLE` (no IF NOT EXISTS) for most tables, so
    re-importing the dump over an existing local DB would error on the first
    CREATE. We therefore: (1) DROP the dump's tables, (2) import the dump
    (recreates tables + data), (3) apply the idempotent schema.sql to (re)create
    any tables/indexes the dump omits — D1 export drops empty tables, and a dump
    can lag a newly-added index. All three run via `wrangler d1 execute DB
    --local` from the pwa dir, where the `DB` binding in wrangler.toml resolves.

    A wrangler failure here is non-fatal: the remote dump (the pipeline's actual
    product) is already on disk, so we warn and return rather than aborting.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pwa_dir = os.path.join(project_root, "pwa")
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
        print("(The remote dump was fetched fine; only the local dev DB was not refreshed.)")
        return

    print("Local wrangler D1 updated successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the latest D1 database dump, refresh wrangler's local "
                    "dev D1 with it, and optionally rebuild the local SQLite DB."
    )
    parser.add_argument(
        "--rebuild-sqlite",
        action="store_true",
        help="After fetching, reconstruct attempts/database.sqlite from the dump "
             "(drops and re-imports under a single transaction).",
    )
    parser.add_argument(
        "--no-local-d1",
        action="store_true",
        help="Skip refreshing wrangler's local D1 (the `npm run dev` database) "
             "with the fetched dump.",
    )
    args = parser.parse_args()

    sql_file = fetch_database()
    if args.rebuild_sqlite:
        rebuild_sqlite(sql_file)
    if not args.no_local_d1:
        update_local_d1(sql_file)


if __name__ == "__main__":
    main()
