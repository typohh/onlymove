"""The PWA's analysis fallback: KataGo behind a small HTTP server.

This runs on the cloud box and answers every request by itself, exactly as it
always has. On top of that it can hand a request to a FASTER MACHINE that has
dialled in — see WorkerPool. That machine is never reachable from outside: it
opens the connection, so there is no inbound port, no DNS record and no
certificate anywhere but here, and nothing to undo when it goes away. A request
with no worker parked, or one the worker does not answer in time, is served
here. The fallback is the normal path, not an error path.

Deliberately stdlib + sgfmill only: this file is deployed on its own, with no
repo checkout beside it. The home-side counterpart is analysis_server/worker.py,
which DOES live in the repo and reuses utils.katago.
"""
import hmac
import json
import os
import subprocess
import traceback
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlsplit
import sgfmill
import sgfmill.sgf

# --- Configuration ---
KATAGO_PATH = "./katago"
# MODEL_PATH = "kata1-b6c96-s175395328-d26788732.txt.gz"
MODEL_PATH = "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
# MODEL_PATH = "kata1-b15c192-s1672170752-d466197061.txt.gz"
# MODEL_PATH = "kata9x9-b18c384nbt-20231025.bin.gz"
CONFIG_PATH = "analysis.cfg"
SERVER_PORT = 8080
PLAYOUTS = 1

# --- Remote worker configuration ---
# EMPTY TOKEN MEANS THE FEATURE IS OFF WITH NO SECOND SWITCH: the /worker routes
# 404 and every request is served locally, so this file can be deployed before
# any worker exists. The token is the only thing separating a worker from anyone
# else on the internet, and a worker's answers are shown to users as ours.
WORKER_TOKEN = os.environ.get("ONLYMOVE_WORKER_TOKEN") or ""
# How long a parked poll waits for work before answering 204. Must stay under
# the reverse proxy's read timeout, or the proxy kills the poll first.
WORKER_POLL_TIMEOUT = float(os.environ.get("ONLYMOVE_WORKER_POLL_TIMEOUT") or 25.0)
# How long an analysis waits on the worker before answering it here instead.
WORKER_JOB_TIMEOUT = float(os.environ.get("ONLYMOVE_WORKER_JOB_TIMEOUT") or 8.0)
# After a job goes unanswered, stop offering work for this long. Without it a
# worker that parks polls but never answers costs EVERY request the full job
# timeout; with it, it costs one.
WORKER_COOLDOWN = float(os.environ.get("ONLYMOVE_WORKER_COOLDOWN") or 30.0)


# --- Coordinate Helper Functions ---
GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def get_move_from_idx(idx, board_size=9):
    """
    Converts a KataGo policy index to a Go move string (e.g., "A1", "pass").
    """
    if idx == board_size * board_size:
        return "pass"

    row = idx // board_size
    col = idx % board_size

    return f"{GTP_COLUMNS[col]}{board_size - row}"



def sgf_to_gtp_coords(sgf_coord, board_size):
    """Converts SGF coordinates (e.g., 'ff') to GTP format (e.g., 'F4')."""
    if sgf_coord is None or sgf_coord.lower() == 'pass' or sgf_coord == '':
        return "pass"
    col_s, row_s = sgf_coord[0], sgf_coord[1]
    col, row = ord(col_s) - ord('a'), ord(row_s) - ord('a')
    gtp_col, gtp_row = GTP_COLUMNS[col], board_size - row
    return f"{gtp_col}{gtp_row}"


def gtp_to_sgf_coords(gtp_coord, board_size):
    """Converts GTP coordinates (e.g., 'F4') to SGF format (e.g., 'ff')."""
    if gtp_coord is None or gtp_coord.lower() == 'pass':
        return "pass"
    col_str, row_str = gtp_coord[0].upper(), gtp_coord[1:]
    col, row = GTP_COLUMNS.find(col_str), board_size - int(row_str)
    sgf_col, sgf_row = chr(ord('a') + col), chr(ord('a') + row)
    return f"{sgf_col}{sgf_row}"


# --- In-flight requests ---
class PendingRequest:
    """One analysis in flight, waiting on whichever backend took it.

    Both backends deliver through this object, which is what lets the choice of
    backend be made per request and changed mid-request: the waiter blocks on
    the event and never learns who set it.
    """

    __slots__ = ("event", "response", "cancelled", "undeliverable")

    def __init__(self):
        self.event = threading.Event()
        self.response = None
        self.cancelled = False
        self.undeliverable = False

    def deliver(self, response):
        self.response = response
        self.event.set()

    def cancel(self):
        self.cancelled = True
        self.event.set()


class ClientQueue:
    """Keeps only the newest request per client.

    The PWA fires one analysis per click, so a user walking a variation outruns
    the engine. The older request is answered 499 the moment a newer one lands
    and the client re-asks for the position it is actually looking at.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = {}

    def begin(self, client_ip):
        request = PendingRequest()
        with self._lock:
            previous = self._latest.get(client_ip)
            self._latest[client_ip] = request
        # Outside the lock: cancel() wakes another thread, which then wants it.
        if previous is not None:
            previous.cancel()
        return request

    def end(self, client_ip, request):
        with self._lock:
            if self._latest.get(client_ip) is request:
                self._latest.pop(client_ip, None)


# --- KataGo Controller (local engine) ---
class KataGoController:
    def __init__(self):
        self.katago_process = None
        self._lock = threading.Lock()
        # Serializes writes to KataGo's stdin so two threads can't interleave
        # halves of a JSON line. SEPARATE from self._lock on purpose: a write
        # blocks whenever KataGo stops draining stdin, and the reader thread
        # needs self._lock to drain stdout.
        self._write_lock = threading.Lock()
        self._query_counter = 0
        self._requests = {}

    def start(self):
        command = [KATAGO_PATH, "analysis", "-config", CONFIG_PATH, "-model", MODEL_PATH]
        print("Starting KataGo engine...")
        self.katago_process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self._reader_thread = threading.Thread(target=self._read_katago_output, daemon=True)
        self._reader_thread.start()
        print("KataGo engine started successfully.")

    def stop(self):
        if self.katago_process:
            print("\nShutting down KataGo engine...")
            self.katago_process.terminate()
            self.katago_process.wait()
            print("Server stopped.")

    def _read_katago_output(self):
        try:
            while True:
                line = self.katago_process.stdout.readline()
                if not line: break
                try:
                    response = json.loads(line)
                except ValueError:
                    # A non-JSON diagnostic line is not worth killing the reader
                    # for: every waiter hangs if this thread dies.
                    continue
                query_id = response.get("id")
                if not query_id:
                    continue
                with self._lock:
                    request = self._requests.get(query_id)
                if request is not None:
                    request.deliver(response)
        finally:
            # On any exit path release every waiter, or they block forever.
            with self._lock:
                pending = list(self._requests.values())
            for request in pending:
                request.deliver({"error": "KataGo engine stopped responding."})

    def analyze(self, query_data, request):
        """Runs the query on the local engine and delivers into `request`."""
        with self._lock:
            self._query_counter += 1
            query_id = f"q_{self._query_counter}"
            self._requests[query_id] = request
        try:
            query = dict(query_data, id=query_id)
            with self._write_lock:
                self.katago_process.stdin.write(json.dumps(query) + '\n')
                self.katago_process.stdin.flush()
            request.event.wait()
        finally:
            with self._lock:
                self._requests.pop(query_id, None)


# --- Remote worker pool ---
class Poller:
    """One parked long-poll from the worker, and the job handed down it."""

    __slots__ = ("event", "job")

    def __init__(self):
        self.event = threading.Event()
        self.job = None


class WorkerPool:
    """The faster machine, when it is connected.

    It holds a few long polls open here and takes jobs down the open response,
    so THE POOL'S CAPACITY IS THE NUMBER OF PARKED POLLS: a job with no poller
    waiting is one the local engine serves, which is the whole failover rule and
    needs no health check to state it. A worker that is merely busy therefore
    degrades to local service rather than queueing behind itself.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._idle = []
        self._jobs = {}
        self._counter = 0
        self._offline_until = 0.0
        self._last_park = 0.0
        self._connected = False

    def park(self, poller):
        with self._lock:
            reconnected = not self._connected
            self._connected = True
            self._last_park = time.time()
            self._idle.append(poller)
        if reconnected:
            log("Worker connected — analysis is being served remotely.")

    def unpark(self, poller):
        """True when this poll timed out before a job reached it."""
        with self._lock:
            try:
                self._idle.remove(poller)
                return True
            except ValueError:
                return False

    def dispatch(self, query_data, request):
        """Hands the query to an idle poller. Returns a job id, or None."""
        now = time.time()
        with self._lock:
            if now < self._offline_until or not self._idle:
                # A worker that stops polling never says so — it just stops. The
                # absence only matters at the moment a request has to go local,
                # which is here.
                if (self._connected and not self._idle
                        and now - self._last_park > WORKER_POLL_TIMEOUT * 2):
                    self._connected = False
                    log("Worker gone — serving analysis locally.")
                return None
            poller = self._idle.pop()
            self._counter += 1
            job_id = f"j_{self._counter}"
            self._jobs[job_id] = request
            # Assigned and signalled UNDER THE LOCK so that a poll whose unpark()
            # loses the race is guaranteed to find its job already set.
            poller.job = {"jobId": job_id, "query": query_data}
            poller.event.set()
        return job_id

    def deliver(self, job_id, response):
        with self._lock:
            request = self._jobs.pop(job_id, None)
            if request is not None:
                # An answer is proof of life, so it ends a cooldown early.
                self._offline_until = 0.0
        if request is None:
            return False
        request.deliver(response)
        return True

    def release(self, job_id):
        """Takes a job back when it could not be handed over at all.

        A parked poll whose socket has died is the ordinary shape of Ctrl-C at
        the worker, and it is worth distinguishing from a worker that took the
        job and went quiet: this one is known undeliverable NOW, so the request
        waiting on it can go local immediately instead of sitting out the job
        timeout. Only a write that fails proves it — a dropped network gives no
        such error, and that case still falls back on the timeout.
        """
        with self._lock:
            request = self._jobs.pop(job_id, None)
        if request is None:
            return
        request.undeliverable = True
        request.event.set()

    def abandon(self, job_id, cooldown=True):
        """Drops a job the worker will no longer be answering for.

        `cooldown` is what separates the two callers: a job that RAN OUT of
        time is evidence against the worker, while one whose client walked away
        is not, and taking the worker out of rotation for a user's fast clicking
        would be a self-inflicted outage.
        """
        with self._lock:
            self._jobs.pop(job_id, None)
            if cooldown:
                self._offline_until = time.time() + WORKER_COOLDOWN

    def status(self):
        now = time.time()
        with self._lock:
            # Reported from the last poll rather than from the lazily-updated
            # flag: the flag only moves when a request has to go local, so it
            # can read "connected" for a worker that stopped polling minutes ago
            # — and this endpoint is where someone goes to ask exactly that.
            fresh = bool(self._last_park) and now - self._last_park < WORKER_POLL_TIMEOUT * 2
            return {
                "connected": bool(self._idle) or fresh,
                "idlePolls": len(self._idle),
                "jobsInFlight": len(self._jobs),
                "cooldownSeconds": max(0.0, self._offline_until - time.time()),
                "secondsSinceLastPoll": (time.time() - self._last_park) if self._last_park else None,
            }


def run_analysis(query_data, request):
    """Prefers the connected worker, falls back to the local engine.

    Returns the name of the backend that answered. Falling back is silent by
    design: a late answer from the worker is worth less to the user than an
    answer now from here.
    """
    reason = None
    job_id = worker_pool.dispatch(query_data, request)
    if job_id is not None:
        request.event.wait(timeout=WORKER_JOB_TIMEOUT)
        # An ANSWER is what decides this, not the event: the event also fires
        # for a supersede and for an undeliverable job, and both leave the
        # response empty.
        if request.cancelled:
            worker_pool.abandon(job_id, cooldown=False)
            return "superseded"
        if request.response is not None:
            return "remote"
        worker_pool.abandon(job_id)
        reason = "worker unreachable" if request.undeliverable else "worker timed out"
        # The signal that sent us here is spent; the local wait needs a fresh
        # one. A supersede landing in the gap re-sets it, so re-read the flag
        # rather than the event — cancel() writes the flag first.
        request.event.clear()
        if request.cancelled:
            request.event.set()
            return "superseded"
    katago_controller.analyze(query_data, request)
    return "local" if reason is None else f"local ({reason})"


def interpret_response(response, board_size):
    """The KataGo answer as the PWA reads it. Shared by both backends."""
    if PLAYOUTS == 1:
        # When using 1 playout, use policy instead of moveInfos
        policy = response.get("policy", [])
        if not policy:
            return {"bestMove": "pass", "winrate": response.get('rootInfo', {}).get("winrate"), "score": response.get('rootInfo', {}).get('scoreLead')}
        # Find the move with the highest policy probability
        best_move_idx = policy.index(max(policy))
        gtp_move = get_move_from_idx(best_move_idx, board_size)
        sgf_move = gtp_to_sgf_coords(gtp_move, board_size)
        return {"bestMove": sgf_move, "winrate": response.get('rootInfo', {}).get("winrate"), "score": response.get('rootInfo', {}).get('scoreLead')}
    if not response.get("moveInfos"):
        return {"bestMove": "pass", "winrate": response.get('rootInfo', {}).get("winrate"), "score": response.get('rootInfo', {}).get('scoreLead')}
    best_move_info = response["moveInfos"][0]
    gtp_move = best_move_info.get("move")
    sgf_move = gtp_to_sgf_coords(gtp_move, board_size)
    return {"bestMove": sgf_move, "winrate": best_move_info.get("winrate"), "score": best_move_info.get("scoreLead")}


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


# --- HTTP Server and Handler ---
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True


class KataGoHandler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        origin = self.headers.get('Origin')
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
        else:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def read_body(self):
        length = int(self.headers.get('Content-Length') or 0)
        return self.rfile.read(length) if length else b""

    def send_json(self, status, payload, cors=False, reason=None):
        body = b"" if payload is None else json.dumps(payload).encode('utf-8')
        if reason:
            self.send_response(status, reason)
        else:
            self.send_response(status)
        if body:
            self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        if cors:
            self.send_cors_headers()
        self.end_headers()
        if body:
            self.wfile.write(body)

    def worker_authorized(self):
        """No token configured reads as the feature being off, not as open."""
        if not WORKER_TOKEN:
            return False
        return hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + WORKER_TOKEN)

    def do_GET(self):
        if urlsplit(self.path).path == "/worker/status" and self.worker_authorized():
            self.send_json(200, worker_pool.status())
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/worker/poll":
            self.handle_worker_poll()
        elif path == "/worker/result":
            self.handle_worker_result()
        else:
            self.handle_analysis()

    # --- Worker endpoints ---
    def handle_worker_poll(self):
        # 404 rather than 401 on a bad token: an unauthenticated caller learns
        # nothing about whether the route exists.
        if not self.worker_authorized():
            self.send_json(404, {"error": "not found"})
            return
        self.read_body()
        poller = Poller()
        worker_pool.park(poller)
        if not poller.event.wait(timeout=WORKER_POLL_TIMEOUT) and worker_pool.unpark(poller):
            self.send_json(204, None)
            return
        # Either a job arrived, or the poll timed out at the instant one was
        # dispatched — in which case unpark() lost the race and the job is set.
        try:
            self.send_json(200, poller.job)
        except OSError as error:
            self.close_connection = True
            worker_pool.release(poller.job["jobId"])
            log(f"Worker poll died holding {poller.job['jobId']} ({type(error).__name__}); "
                f"serving it locally.")

    def handle_worker_result(self):
        if not self.worker_authorized():
            self.send_json(404, {"error": "not found"})
            return
        try:
            payload = json.loads(self.read_body().decode('utf-8'))
        except ValueError:
            self.send_json(400, {"error": "malformed result"})
            return
        response = payload.get("response")
        if response is None:
            response = {"error": payload.get("error") or "worker returned no result"}
        accepted = worker_pool.deliver(payload.get("jobId"), response)
        # A rejected result is the normal outcome of a job that already timed
        # out and was answered locally; the worker only logs it.
        self.send_json(200, {"accepted": accepted})

    # --- Analysis endpoint ---
    def handle_analysis(self):
        start_time = time.time()
        status_message = "Unknown"
        backend = "none"
        request = None
        client_ip = self.client_address[0]
        try:
            sgf_data = self.read_body().decode('utf-8')
            moves, komi, rules, board_size = self.parse_sgf(sgf_data)
            query_data = {
                "moves": moves, "initialStones": [], "rules": rules,
                "komi": komi, "boardXSize": board_size, "boardYSize": board_size, 
                "maxVisits": PLAYOUTS, "includePolicy": PLAYOUTS == 1
            }
            request = client_queue.begin(client_ip)
            backend = run_analysis(query_data, request)

            if request.cancelled:
                status_message = "Cancelled"
                backend = "superseded"
                self.send_json(499, {"error": "Request superseded by a newer one."},
                               cors=True, reason="Client Closed Request")
                return

            response = request.response or {}
            if "error" in response: raise Exception(f"KataGo engine error: {response['error']}")

            analysis_result = interpret_response(response, board_size)

            status_message = "Success"
            self.send_json(200, analysis_result, cors=True)
        except Exception as e:
            status_message = f"Error: {type(e).__name__}"
            print("--- AN EXCEPTION OCCURRED ---")
            traceback.print_exc()
            print("-----------------------------")
            self.send_json(500, {"error": f"{type(e).__name__}: {e}"}, cors=True)
        finally:
            if request is not None:
                client_queue.end(client_ip, request)
            duration_ms = (time.time() - start_time) * 1000
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Request handled. Status: {status_message}. Backend: {backend}. Time: {duration_ms:.2f} ms")

    def parse_sgf(self, sgf_string):
        game = sgfmill.sgf.Sgf_game.from_string(sgf_string)
        board_size = game.get_size()
        root_node = game.get_root()
        komi = float(root_node.get('KM'))
        rules = root_node.get('RU').lower()
        moves = []
        node = root_node
        while True:
            if node.has_property("B") or node.has_property("W"):
                player, move_bytes = node.get_raw_move()
                sgf_move = move_bytes.decode('utf-8') if move_bytes else "pass"
                gtp_move = sgf_to_gtp_coords(sgf_move, game.get_size())
                moves.append([player.upper(), gtp_move])
            if len(node) > 1: raise ValueError("SGF is not linear (contains variations)")
            if len(node) == 0: break
            node = node[0]
        return moves, komi, rules, board_size

    def log_message(self, format, *args):
        """This method is overridden to suppress the default server logs."""
        return


katago_controller = KataGoController()
client_queue = ClientQueue()
worker_pool = WorkerPool()


# --- Main Execution ---
if __name__ == '__main__':
    try:
        katago_controller.start()
        server_address = ('127.0.0.1', SERVER_PORT)
        httpd = ThreadingHTTPServer(server_address, KataGoHandler)
        print(f"Starting multi-threaded KataGo SGF server on http://127.0.0.1:{SERVER_PORT}")
        if WORKER_TOKEN:
            print(f"Remote worker enabled: POST /worker/poll, /worker/result "
                  f"(poll {WORKER_POLL_TIMEOUT:.0f}s, job timeout {WORKER_JOB_TIMEOUT:.0f}s).")
        else:
            print("Remote worker disabled (ONLYMOVE_WORKER_TOKEN unset).")
        httpd.serve_forever()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
    finally:
        katago_controller.stop()
