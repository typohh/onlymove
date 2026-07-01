#!/usr/bin/env python3
"""
Admin script: create an EGD-leaderboard invitation.

Generates a random 8-character token, validates the EGD PIN by fetching the
player's data from the European Go Database, and inserts an `invitation` row
into the leaderboard D1 (via wrangler, the same tool `pipeline.fetch_database`
uses). Prints the registration link to hand to the recipient.

Usage:
    python generate_invite.py <EGD_PIN> [--local]     # numeric -> create invitation
    python generate_invite.py <name...>               # non-numeric -> search EGD

    <EGD_PIN>   a numeric European Go Database PIN -> validates it and inserts an
                invitation row, printing the registration link.
    <name...>   any non-numeric argument is a name search (no DB write):
                  one word  -> searched as both first name AND last name
                  two words -> searched as first+last name in BOTH orders
                results are combined + de-duplicated and printed as
                PIN / name / country / rank / GoR; re-run with a listed PIN to invite.
    --local     target wrangler's local dev D1 instead of the remote D1
                (invitation flow only; search never touches D1).

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

# Name search. Params: `name` = first name, `lastname` = last name. EGD requires
# a lastname (a bare `name=` search returns retcode "no search last name found").
EGD_SEARCH_URL = "https://www.europeangodatabase.eu/EGD/GetPlayerDataByData.php"

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


def build_search_queries(query):
    """Turn a name string into the EGD search param sets to try.

    One word  -> searched as both last name and first name (the first-name-only
                 attempt is a no-op since EGD requires a lastname, but the
                 lastname search does the work).
    Two words -> searched as name+lastname in BOTH orders, so word order in the
                 input doesn't matter.
    Three+    -> first token vs. the joined remainder, both orders.
    """
    tokens = query.split()
    if len(tokens) <= 1:
        t = query.strip()
        return [{"lastname": t}, {"name": t}]
    first, rest = tokens[0], " ".join(tokens[1:])
    return [
        {"name": first, "lastname": rest},
        {"name": rest, "lastname": first},
    ]


def search_egd_players(query):
    """Search EGD by name and return combined, de-duplicated player dicts.

    Issues every query from build_search_queries, unions the `players` arrays,
    and de-dupes by Pin_Player (first occurrence wins). A failed request or a
    non-'Ok' retcode (e.g. the first-name-only no-op) is skipped, not fatal.
    """
    seen = {}
    for params in build_search_queries(query):
        url = EGD_SEARCH_URL + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "onlymove-invite/1.0"})
            with urllib.request.urlopen(req) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as e:  # network / HTTP error — skip this query
            print(f"Warning: EGD search request failed ({params}): {e}", file=sys.stderr)
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if data.get("retcode") != "Ok":
            continue
        for player in data.get("players", []):
            pin = player.get("Pin_Player")
            if pin and pin not in seen:
                seen[pin] = player
    return list(seen.values())


def print_search_results(query, players):
    """Print matching players as PIN / name / country / rank / GoR, strongest first."""
    if not players:
        print(f"No EGD players found for {query!r}.")
        return

    def gor_of(player):
        try:
            return int(player.get("Gor"))
        except (TypeError, ValueError):
            return -1

    players = sorted(players, key=gor_of, reverse=True)
    print(f"Found {len(players)} player(s) for {query!r}:\n")
    print(f"{'PIN':<10}  {'Name':<28}  {'CC':<3}  {'Rank':<5}  {'GoR':>5}")
    print("-" * 60)
    for player in players:
        name = f"{player.get('Name', '')} {player.get('Last_Name', '')}".strip()
        print(
            f"{player.get('Pin_Player', ''):<10}  {name[:28]:<28}  "
            f"{player.get('Country_Code', ''):<3}  {player.get('Grade', ''):<5}  "
            f"{player.get('Gor', ''):>5}"
        )
    print()
    print("To create an invitation, re-run with a PIN from above, e.g.:")
    print(f"    python generate_invite.py {players[0].get('Pin_Player', '')}")


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
        description="Create an EGD-leaderboard invitation (numeric PIN) or search "
                    "EGD by name (non-numeric).",
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="a numeric EGD PIN (creates an invitation) OR a name to search "
             "(one word = first/last; two words = both orders)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="target wrangler's local dev D1 instead of the remote D1 "
             "(invitation flow only; search never touches D1)",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip()

    # Non-numeric -> name search (a lookup helper; no DB write). Numeric -> the
    # PIN it is: validate + insert the invitation.
    if not query.isdigit():
        print_search_results(query, search_egd_players(query))
        return

    token = generate_token()
    egd_name, gor, country_code = fetch_egd_player(query)
    created_at = datetime.now(timezone.utc).isoformat()

    insert_invitation(
        token=token,
        egd_pin=query,
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
