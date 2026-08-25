#!/usr/bin/env bash
# Ships analysis_server/server.py to the box that serves api.onlymove.net.
#
# There is no root on the far side: the service account owns its own working
# directory but is not in google-sudoers, so this script edits nothing under
# /etc. It restarts by killing the main process and letting the unit's
# Restart=always bring it back, and it installs the worker token as a 0600 file
# beside server.py rather than as a unit Environment= line.
#
# Usage:
#   analysis_server/deploy.sh                 # ship server.py + analysis.cfg, restart, verify
#   analysis_server/deploy.sh --token         # also install the worker token
#   analysis_server/deploy.sh --no-restart    # ship only, leave the running server alone
set -euo pipefail

HOST="${ONLYMOVE_API_HOST:-typo_hh@api.onlymove.net}"
DIR="${ONLYMOVE_API_DIR:-/home/typo_hh/server}"
URL="${ONLYMOVE_API_URL:-https://api.onlymove.net}"
TOKEN_ENV="${ONLYMOVE_WORKER_ENV_FILE:-$HOME/.config/onlymove/worker.env}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

install_token=0
restart=1
for arg in "$@"; do
  case "$arg" in
    --token) install_token=1 ;;
    --no-restart) restart=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# A syntax error reaches the box as a service that will not come back up, and
# the box has no second copy to fall back to. Check here, where it is free.
python3 -m py_compile "$HERE/server.py"
echo "==> server.py compiles"

stamp=$(date -u +%Y%m%d-%H%M%S)
ssh "$HOST" "cp -a '$DIR/server.py' '$DIR/server.py.bak-$stamp'"
echo "==> backed up to server.py.bak-$stamp"

rsync -c -e ssh "$HERE/server.py" "$HOST:$DIR/server.py"
rsync -c -e ssh "$HERE/analysis.cfg" "$HOST:$DIR/analysis.cfg"
echo "==> shipped server.py + analysis.cfg"

if [ "$install_token" = 1 ]; then
  [ -r "$TOKEN_ENV" ] || { echo "no token file at $TOKEN_ENV" >&2; exit 1; }
  # shellcheck disable=SC1090
  token=$(. "$TOKEN_ENV"; printf '%s' "${ONLYMOVE_WORKER_TOKEN:-}")
  [ -n "$token" ] || { echo "ONLYMOVE_WORKER_TOKEN empty in $TOKEN_ENV" >&2; exit 1; }
  # Written by the shell on the far side so the token never appears in an
  # argument vector, which is world-readable through /proc while it runs.
  printf '%s\n' "$token" | ssh "$HOST" "umask 077; cat > '$DIR/worker_token'"
  echo "==> installed worker_token (0600)"
fi

if [ "$restart" = 0 ]; then
  echo "==> --no-restart: the running server is still the previous version"
  exit 0
fi

# Restart without root: kill the main process, and Restart=always brings the
# unit back. The unit's KillMode=control-group means systemd reaps the KataGo
# child with it, so no engine is left orphaned holding the model in memory —
# which matters on a box with well under a gigabyte to spare.
before=$(ssh "$HOST" "systemctl show katago.service -p MainPID --value")
# `kill 0` signals the caller's whole process group, and MainPID reads 0 for a
# unit that is not running — so the value is checked, not trusted.
case "$before" in
  ''|*[!0-9]*|0|1) echo "!! refusing to kill MainPID='$before'" >&2; exit 1 ;;
esac
ssh "$HOST" "kill $before" || true
echo "==> killed pid $before, waiting for the unit to come back"

start=$(date +%s)
sgf='(;FF[4]GM[1]CA[UTF-8]SZ[9]RU[japanese]KM[6.5];B[gf];W[ef])'
for _ in $(seq 1 60); do
  sleep 5
  body=$(curl -s --max-time 20 -X POST --data-binary "$sgf" "$URL/" || true)
  case "$body" in *bestMove*) break ;; esac
done
elapsed=$(( $(date +%s) - start ))

case "${body:-}" in
  *bestMove*) echo "==> analysis answering after ${elapsed}s: $body" ;;
  *) echo "!! the server did not answer within ${elapsed}s" >&2
     echo "   roll back with:" >&2
     echo "   ssh $HOST \"cp -a '$DIR/server.py.bak-$stamp' '$DIR/server.py'; kill \\\$(systemctl show katago.service -p MainPID --value)\"" >&2
     exit 1 ;;
esac

after=$(ssh "$HOST" "systemctl show katago.service -p MainPID --value")
echo "==> pid $before -> $after"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 -X POST -H 'Authorization: Bearer probe' "$URL/worker/poll" || true)
case "$code" in
  404) echo "==> /worker/poll 404: new code live, worker feature off (no token installed)" ;;
  500) echo "!! /worker/poll 500: still the OLD code — the restart did not pick up the new file" >&2; exit 1 ;;
  *)   echo "==> /worker/poll $code" ;;
esac
