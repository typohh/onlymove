"""Serves onlymove's analysis requests from THIS machine instead of the cloud box.

Run it and the cloud server hands its analysis requests here; stop it and the
cloud server goes back to answering them itself, with nothing to switch over and
no user-visible change beyond the answers arriving slower. The PWA knows nothing
about any of this — it posts to the same URL either way.

The connection is OUTBOUND ONLY. This process dials the cloud box, parks a few
long polls there and takes jobs down the open responses, so this machine needs
no inbound port, no public address, no DNS record and no certificate. That is
also what makes stopping it safe at any instant: an unanswered job is one the
cloud server answers itself once its own timeout passes.

Usage:
    ./.venv/bin/python3 -m analysis_server.worker

The shared secret is read from ~/.config/onlymove/worker_token and must match
the `worker_token` file beside server.py on the box. It is the only thing
separating this worker from anyone else on the internet, and the answers it
returns are shown to users as ours.
"""
import os
import sys
import threading
import time

import requests

from utils.constants import ALL_BOARD_SIZES, KOMI, RULES
from utils.katago.controller import KatagoController
from utils.katago.models import DEFAULT_MODEL, get_model

# --- Configuration ---
SERVER_URL = (os.environ.get("ONLYMOVE_ANALYSIS_URL") or "https://api.onlymove.net").rstrip("/")
# Same rule as the server's end of this (analysis_server/server.py): the token
# comes from a file, with the environment winning when it is set. A file is what
# makes `python3 -m analysis_server.worker` the whole command — an env-only
# token forces every caller to source something first, and a secret passed on a
# command line is world-readable through /proc for as long as it runs. Outside
# the repo so it cannot be committed.
TOKEN_FILE = (os.environ.get("ONLYMOVE_WORKER_TOKEN_FILE")
              or os.path.expanduser("~/.config/onlymove/worker_token"))


def read_token():
    token = os.environ.get("ONLYMOVE_WORKER_TOKEN")
    if token:
        return token.strip()
    try:
        with open(TOKEN_FILE) as handle:
            return handle.read().strip()
    except OSError:
        return ""


TOKEN = read_token()
# Parked polls, and therefore the most jobs this machine can be running at once.
# The cloud box hands a job only to an already-parked poll, so this number IS
# the point at which it goes back to answering requests itself — it is a
# capacity statement, not a queue depth.
POLL_THREADS = int(os.environ.get("ONLYMOVE_WORKER_THREADS") or 4)
# THE POINT OF THIS MACHINE. The cloud box asks for one visit because that is
# all it can afford — one visit is the raw policy, an opinion with no reading
# behind it. A GPU can search properly in a time a person still reads as
# instant, so the visit count the query arrives with is REPLACED here rather
# than honoured. The answer's shape changes with it: a searched response
# carries moveInfos, which is what the server reads in preference to policy.
WORKER_VISITS = int(os.environ.get("ONLYMOVE_WORKER_VISITS") or 30)
# Search threads per query. Visits are sequential within one query unless
# several threads explore at once, so this is what decides whether 30 visits
# feels instant; it is a LATENCY setting and unrelated to how many queries run
# side by side, which is POLL_THREADS.
SEARCH_THREADS = int(os.environ.get("ONLYMOVE_WORKER_SEARCH_THREADS") or 8)
# Must exceed the server's own poll timeout, or every idle poll looks like a
# network failure here and reconnects for nothing.
POLL_READ_TIMEOUT = float(os.environ.get("ONLYMOVE_WORKER_READ_TIMEOUT") or 60.0)
CONNECT_TIMEOUT = 10.0
RECONNECT_BACKOFF_MAX = 30.0

_print_lock = threading.Lock()
_shutdown = threading.Event()
# The preflight has already proved the link, and a poll SERVES from the moment
# it parks — not from when it returns. So the link starts up and only its
# TRANSITIONS are worth a line; announcing "connected" on the first poll to
# come back would date the connection a whole idle poll cycle late.
_link_lock = threading.Lock()
_link_up = True


def log(message):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def note_link(up, detail=""):
    """Logs a link transition once, whichever poller notices it first."""
    global _link_up
    with _link_lock:
        if up == _link_up:
            return
        _link_up = up
    if up:
        log("Reconnected — this machine is serving analysis again.")
    else:
        log(f"Lost the server{detail}; the cloud box is answering requests "
            f"until this reconnects.")


def auth_headers():
    return {"Authorization": "Bearer " + TOKEN}


def preflight():
    """Checks the token before loading the network, which takes a while."""
    try:
        response = requests.get(f"{SERVER_URL}/worker/status", headers=auth_headers(),
                                timeout=(CONNECT_TIMEOUT, 20.0))
    except requests.RequestException as error:
        log(f"Cannot reach {SERVER_URL}: {error}")
        return False
    if response.status_code == 404:
        log("Rejected by the server. Either the token does not match its "
            "ONLYMOVE_WORKER_TOKEN, or the box is still running a server "
            "version that has no /worker routes.")
        return False
    if not response.ok:
        log(f"Unexpected status from {SERVER_URL}/worker/status: {response.status_code}")
        return False
    log(f"Server reachable, token accepted. {response.json()}")
    return True


def warm_up(controller):
    """One query per board size, so a user's request never pays the first-run cost.

    The OpenCL backend tunes itself per board shape on first use, which is far
    longer than any analysis: paid here it is invisible, paid on the first job
    it is a request the server times out on and answers itself anyway.
    """
    for board_size in ALL_BOARD_SIZES:
        started = time.time()
        controller.send_query({
            "moves": [], "initialStones": [], "rules": RULES, "komi": KOMI,
            "boardXSize": board_size, "boardYSize": board_size,
            "maxVisits": 1, "includePolicy": True,
        })
        log(f"Warmed up {board_size}x{board_size} in {time.time() - started:.1f}s")


def run_job(controller, job):
    """Runs one job and reports it back. The reply shape mirrors the server's."""
    job_id = job.get("jobId")
    query = dict(job.get("query") or {})
    # includePolicy goes with the one-visit query it arrived as; a searched
    # answer is read from moveInfos, and the policy array is ~n^2 floats of
    # response body nobody looks at.
    query["maxVisits"] = WORKER_VISITS
    query["includePolicy"] = WORKER_VISITS == 1
    started = time.time()
    try:
        response = controller.send_query(query)
        payload = {"jobId": job_id, "response": response}
        outcome = "ok"
    except Exception as error:
        # Reported rather than swallowed: the server needs to stop waiting and
        # answer the request itself, and it cannot see this engine.
        payload = {"jobId": job_id, "error": f"{type(error).__name__}: {error}"}
        outcome = f"failed ({type(error).__name__})"
    return payload, outcome, (time.time() - started) * 1000


def poll_loop(controller):
    session = requests.Session()
    backoff = 1.0
    while not _shutdown.is_set():
        try:
            response = session.post(f"{SERVER_URL}/worker/poll", headers=auth_headers(),
                                    timeout=(CONNECT_TIMEOUT, POLL_READ_TIMEOUT))
            if response.status_code == 404:
                log("Server rejected the token; stopping this poller.")
                return
            response.raise_for_status()
            note_link(True)
            backoff = 1.0
            if response.status_code == 204:
                continue
            job = response.json()
        except requests.RequestException as error:
            note_link(False, f" ({type(error).__name__})")
            _shutdown.wait(backoff)
            backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)
            continue

        payload, outcome, elapsed_ms = run_job(controller, job)
        try:
            reply = session.post(f"{SERVER_URL}/worker/result", json=payload,
                                 headers=auth_headers(), timeout=(CONNECT_TIMEOUT, 30.0))
            # A refused result is the ordinary outcome of a job the server gave
            # up on and answered itself; it is not an error here.
            accepted = reply.ok and reply.json().get("accepted")
        except (requests.RequestException, ValueError):
            accepted = False
        log(f"Job {payload['jobId']} {outcome} in {elapsed_ms:.0f} ms"
            f"{'' if accepted else ' (server had already answered it)'}")


def main():
    if not TOKEN:
        log(f"No token: set ONLYMOVE_WORKER_TOKEN or write it to {TOKEN_FILE}. "
            f"It must match the worker_token file on the box.")
        return 2
    if not preflight():
        return 1

    controller = KatagoController()
    log(f"Starting KataGo ({DEFAULT_MODEL})...")
    # ONE engine for every board size, sized to the largest. bots/analysis.cfg
    # sets requireMaxBoardSize, which makes that buffer an EXACT size rather
    # than a bound — and a query on any other size then kills KataGo's bot
    # thread with a message on stderr and NO response on stdout, so the caller
    # waits out its full query timeout on a request that will never be answered.
    # The pipeline never meets this because each of its runs is single-size; a
    # server takes both. Overridden per-caller, so the tuner's own file is
    # untouched. The NN cache is shrunk for the same reason: the config's is
    # sized for pipeline batch runs, while here a cached position only pays off
    # within one review session over a shared root — a problem is rarely
    # reviewed again before its entries would have been evicted anyway.
    controller.start_engine(
        get_model(DEFAULT_MODEL), max(ALL_BOARD_SIZES),
        extra_overrides=(f"requireMaxBoardSize=false,"
                         f"numAnalysisThreads={POLL_THREADS},"
                         f"numSearchThreadsPerAnalysisThread={SEARCH_THREADS},"
                         f"nnCacheSizePowerOfTwo=14"))
    try:
        warm_up(controller)
        threads = [threading.Thread(target=poll_loop, args=(controller,), daemon=True)
                   for _ in range(POLL_THREADS)]
        for thread in threads:
            thread.start()
        log(f"Polling {SERVER_URL} with {POLL_THREADS} threads at {WORKER_VISITS} "
            f"visits. Ctrl-C to hand analysis back to the cloud box.")
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("Stopping.")
    finally:
        _shutdown.set()
        controller.stop_engine()
    return 0


if __name__ == "__main__":
    sys.exit(main())
