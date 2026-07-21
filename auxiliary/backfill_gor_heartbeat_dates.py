#!/usr/bin/env python3
"""One-off backfill: add rank_label + last_game_at to production's latest 'EGD'
heartbeat rows (the columns shipped 2026-07-21 empty; until a row carries a date
the /leaderboard activity gate fails open).

Runs LOCALLY on purpose — the Worker cannot fetch EGD (orange-to-orange 1106),
but this machine can, and the remote D1 is reachable via wrangler (hits
api.cloudflare.com, not the WAF-guarded zone).

Per registered EGD player, the LATEST 'EGD' heartbeat row is updated ONLY when
it is provably still current:
  * the row's rank_label AND last_game_at are both still NULL (never clobber a
    real heartbeat written by the weekly client piggyback), and
  * the freshly fetched EGD GoR equals the row's stored GoR (a mismatch means
    the player was re-rated since the heartbeat — a recent tournament; their
    next weekly heartbeat will carry the right pairing), and
  * the fetched Last_Appearance date is at least MIN_AGE_DAYS old (a tournament
    inside the current week must not be stamped onto a pre-tournament GoR row).
Everything else is skipped with a reason. Dry-run by default; --apply executes
the UPDATEs against the remote D1 in one batch.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.request
from datetime import date, timedelta

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PWA_DIR = PROJECT_ROOT / 'pwa'
EGD_PIN_URL = 'https://www.europeangodatabase.eu/EGD/GetPlayerDataByPIN.php?pin='
MIN_AGE_DAYS = 7   # a tournament newer than this is "recent" -> skip the row

LATEST_HEARTBEATS_SQL = (
    "SELECT s.id, s.user_id, s.rank, s.rank_label, s.last_game_at, x.egd_pin "
    "FROM rank_surveys s JOIN external_ids x ON x.user_id = s.user_id "
    "WHERE s.server = 'EGD' AND x.egd_pin IS NOT NULL "
    "AND s.id = (SELECT MAX(id) FROM rank_surveys "
    "            WHERE server = 'EGD' AND user_id = s.user_id)"
)


def d1_query(sql):
    """Run a read-only SQL against the remote D1, return the result rows."""
    out = subprocess.run(
        ['npx', 'wrangler', 'd1', 'execute', 'tsumego-reports', '--remote',
         '--json', '--command', sql],
        cwd=PWA_DIR, capture_output=True, text=True, check=True,
    ).stdout
    payload = json.loads(out[out.index('['):])
    return payload[0]['results']


def fetch_egd(pin):
    """EGD player JSON by PIN, or None on any failure."""
    try:
        with urllib.request.urlopen(EGD_PIN_URL + pin, timeout=15) as res:
            d = json.load(res)
        return d if d.get('retcode') == 'Ok' else None
    except Exception:
        return None


def short_rank_label(label):
    """EGD Grade -> normalized short label ('4d'/'17k'/'3p'), or None."""
    m = re.match(r'^(\d{1,2})\s*([dkp])$', str(label or '').strip().lower())
    return m.group(1) + m.group(2) if m else None


def last_appearance_to_iso(code):
    """EGD Last_Appearance ('T220122H') -> ISO date, or None (pivot at 70)."""
    m = re.match(r'^T(\d{2})(\d{2})(\d{2})[A-Z]?$', str(code or '').strip().upper())
    if not m:
        return None
    yy, mo, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= dd <= 31):
        return None
    return f'{1900 + yy if yy >= 70 else 2000 + yy}-{m.group(2)}-{m.group(3)}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--apply', action='store_true',
                    help='execute the UPDATEs against the remote D1 (default: dry-run)')
    args = ap.parse_args()

    rows = d1_query(LATEST_HEARTBEATS_SQL)
    cutoff = (date.today() - timedelta(days=MIN_AGE_DAYS)).isoformat()
    updates = []

    for r in rows:
        tag = f"{r['user_id']} pin {r['egd_pin']} (row {r['id']}, gor {r['rank']})"
        if r['rank_label'] is not None or r['last_game_at'] is not None:
            print(f'skip  {tag}: already has verified fields')
            continue
        d = fetch_egd(str(r['egd_pin']))
        if not d:
            print(f'skip  {tag}: EGD fetch failed')
            continue
        try:
            fetched_gor = int(d.get('Gor'))
            stored_gor = int(r['rank'])
        except (TypeError, ValueError):
            print(f'skip  {tag}: unparseable GoR (fetched {d.get("Gor")!r})')
            continue
        if fetched_gor != stored_gor:
            print(f'skip  {tag}: GoR changed on EGD ({stored_gor} -> {fetched_gor}; '
                  'recent tournament — next weekly heartbeat will pair them correctly)')
            continue
        last_game = last_appearance_to_iso(d.get('Last_Appearance'))
        if not last_game:
            print(f'skip  {tag}: no parseable Last_Appearance ({d.get("Last_Appearance")!r})')
            continue
        if last_game > cutoff:
            print(f'skip  {tag}: last tournament {last_game} is within {MIN_AGE_DAYS} days')
            continue
        grade = short_rank_label(d.get('Grade'))
        label_sql = f"'{grade}'" if grade else 'NULL'
        # Values are regex/int validated above, so inlining is safe (wrangler
        # d1 execute has no bind parameters). The NULL guards make a re-run or
        # a race against a fresh weekly heartbeat a no-op.
        updates.append(
            f"UPDATE rank_surveys SET rank_label = {label_sql}, last_game_at = '{last_game}' "
            f"WHERE id = {int(r['id'])} AND rank_label IS NULL AND last_game_at IS NULL;"
        )
        print(f'UPDATE {tag}: grade {grade or "-"}, last tournament {last_game}')

    print(f'\n{len(updates)} update(s) of {len(rows)} latest heartbeat row(s).')
    if not updates:
        return
    if not args.apply:
        print('Dry-run — re-run with --apply to execute:')
        for u in updates:
            print('  ' + u)
        return
    with tempfile.NamedTemporaryFile('w', suffix='.sql', delete=False) as f:
        f.write('\n'.join(updates) + '\n')
        sql_path = f.name
    subprocess.run(
        ['npx', 'wrangler', 'd1', 'execute', 'tsumego-reports', '--remote',
         '--file', sql_path],
        cwd=PWA_DIR, check=True,
    )
    print('Applied.')


if __name__ == '__main__':
    main()
