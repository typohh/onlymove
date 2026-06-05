import argparse
import os
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


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the latest D1 database dump, optionally rebuilding the local SQLite DB."
    )
    parser.add_argument(
        "--rebuild-sqlite",
        action="store_true",
        help="After fetching, reconstruct attempts/database.sqlite from the dump "
             "(drops and re-imports under a single transaction).",
    )
    args = parser.parse_args()

    sql_file = fetch_database()
    if args.rebuild_sqlite:
        rebuild_sqlite(sql_file)


if __name__ == "__main__":
    main()
