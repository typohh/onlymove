#!/usr/bin/env python3
"""
Admin script: create an EGD-leaderboard invitation.

Generates a random 8-character token, validates the EGD PIN by fetching the
player's data from the European Go Database, and inserts an `invitation` row
into the leaderboard D1 (via wrangler, the same tool `pipeline.fetch_database`
uses). Prints the registration link to hand to the recipient.

Usage:
    python generate_invite.py <EGD_PIN> [--local]

    <EGD_PIN>   the player's European Go Database PIN.
    --local     target wrangler's local dev D1 instead of the remote D1.

Stdlib only.
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# Token alphabet: 8 random alphanumeric characters.
TOKEN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
TOKEN_LENGTH = 8

EGD_URL = "https://www.europeangodatabase.eu/EGD/GetPlayerDataByPIN.php?pin={pin}"

REGISTER_LINK = "https://onlymove.net/register?t={token}"


def generate_token():
    """A random 8-character alphanumeric token."""
    return "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(TOKEN_LENGTH))


def fetch_egd_player(pin):
    """Fetch + validate the player from EGD by PIN.

    Returns (egd_name, gor, country_code). Exits non-zero if the lookup fails
    or the EGD retcode is not 'Ok' (which is how a bad PIN is rejected).
    """
    url = EGD_URL.format(pin=urllib.parse.quote(str(pin)))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "onlymove-invite/1.0"})
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:  # network / HTTP error
        print(f"Error: failed to reach EGD for PIN {pin}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error: EGD returned non-JSON for PIN {pin}: {e}", file=sys.stderr)
        sys.exit(1)

    if data.get("retcode") != "Ok":
        print(
            f"Error: EGD lookup for PIN {pin} failed "
            f"(retcode={data.get('retcode')!r}); the PIN is likely invalid.",
            file=sys.stderr,
        )
        sys.exit(1)

    name = data.get("Name", "")
    last_name = data.get("Last_Name", "")
    egd_name = f"{name} {last_name}".strip()

    try:
        gor = int(data.get("Gor"))
    except (TypeError, ValueError):
        print(f"Error: EGD returned a non-integer Gor {data.get('Gor')!r} for PIN {pin}.",
              file=sys.stderr)
        sys.exit(1)

    country_code = data.get("Country_Code", "")
    return egd_name, gor, country_code


def _sql_str(value):
    """Quote a string for inline SQL, escaping single quotes by doubling them."""
    return "'" + str(value).replace("'", "''") + "'"


def insert_invitation(token, egd_pin, egd_name, country_code, gor, created_at, use_local):
    """Insert the invitation row via wrangler d1 execute.

    Mirrors pipeline.fetch_database's wrangler invocation: subprocess.run with
    cwd=pwa_dir (where wrangler.toml lives), and the same FileNotFoundError /
    CalledProcessError handling.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    pwa_dir = os.path.join(project_root, "pwa")

    sql = (
        "INSERT INTO invitation "
        "(invite_token, egd_pin, egd_name, country_code, gor, created_at, "
        "claimed_by_user_id, claimed_at) VALUES "
        f"({_sql_str(token)}, {_sql_str(egd_pin)}, {_sql_str(egd_name)}, "
        f"{_sql_str(country_code)}, {int(gor)}, {_sql_str(created_at)}, NULL, NULL)"
    )

    scope = "--local" if use_local else "--remote"
    command = [
        "npx", "wrangler", "d1", "execute", "tsumego-reports",
        scope, "--command", sql,
    ]

    try:
        subprocess.run(command, cwd=pwa_dir, check=True)
    except FileNotFoundError:
        print("\nError: 'npx' command not found. Ensure Node.js and npm are installed.",
              file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError: wrangler failed to insert the invitation: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Create an EGD-leaderboard invitation for a given EGD PIN."
    )
    parser.add_argument("egd_pin", help="the player's European Go Database PIN")
    parser.add_argument(
        "--local",
        action="store_true",
        help="target wrangler's local dev D1 instead of the remote D1",
    )
    args = parser.parse_args()

    token = generate_token()
    egd_name, gor, country_code = fetch_egd_player(args.egd_pin)
    created_at = datetime.now(timezone.utc).isoformat()

    insert_invitation(
        token=token,
        egd_pin=args.egd_pin,
        egd_name=egd_name,
        country_code=country_code,
        gor=gor,
        created_at=created_at,
        use_local=args.local,
    )

    print()
    print(f"Invitation created for {egd_name} (GoR {gor}, {country_code}).")
    print(f"Registration link: {REGISTER_LINK.format(token=token)}")


if __name__ == "__main__":
    main()
