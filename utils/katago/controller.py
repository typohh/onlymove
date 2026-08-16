import json
import os
import re
import subprocess
import threading

from utils.katago.models import MODEL_HUMAN, get_model

from utils.paths import PROJECT_ROOT
CONFIG_FILE = os.path.join(PROJECT_ROOT, "bots", "analysis.cfg")
# KATAGO_PATH defaults to the OpenCL build at bots/katago. Override via the
# KATAGO_PATH environment variable on hosts that need a different binary
# (e.g. bots/katago_cuda on a vast.ai NVIDIA instance).
KATAGO_PATH = os.environ.get("KATAGO_PATH") or os.path.join(PROJECT_ROOT, "bots", "katago")

# How long a single send_query blocks before giving up on a stuck/crashed engine.
QUERY_TIMEOUT_SECONDS = 60 * 60


class KatagoController:
    """
    Manages the KataGo analysis engine with a robust, encapsulated, and thread-safe query system.
    The external interface is a simple, blocking send_query call.
    """

    def __init__(self):
        self.process = None
        self._master_reader_thread = None
        self.running = False
        self.batch_size = None
        self.lock = threading.Lock()
        # Serializes writes to KataGo's stdin so two threads can't interleave
        # halves of a JSON line. It is SEPARATE from self.lock on purpose: a
        # write blocks whenever KataGo stops draining stdin, and the master
        # reader needs self.lock to drain stdout. See send_query.
        self.write_lock = threading.Lock()
        self.query_counter = 0
        self.response_events = {}  # Maps query_id -> threading.Event
        self.responses = {}  # Maps query_id -> response_dict

    def start_engine(self, model, board_size, human=False, verbose=False,
                     extra_overrides=None):
        # extra_overrides is a comma-separated `key=value` string appended to the
        # -override-config flag. It is a COMMAND-LINE override, so it does not
        # touch bots/analysis.cfg and pipeline.tune_engine's hardware tuning of
        # that file survives — use it for per-caller settings (a bulk NN sweep
        # wants a far larger nnMaxBatchSize than interactive search does).
        overrides = f"maxBoardXSizeForNNBuffer={board_size},maxBoardYSizeForNNBuffer={board_size}"
        if extra_overrides:
            overrides += "," + extra_overrides
        command = [
            KATAGO_PATH, "analysis",
            "-config", CONFIG_FILE,
            "-override-config", overrides,
        ]
        command.extend(["-model", model])
        if human:
            command.extend(["-human-model", get_model(MODEL_HUMAN)])

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if verbose else subprocess.DEVNULL, text=True, bufsize=1
        )
        self.running = True
        self._master_reader_thread = threading.Thread(target=self._master_reader)
        self._master_reader_thread.daemon = True
        self._master_reader_thread.start()

        if verbose:
            def read_stderr(process):
                for line in process.stderr:
                    print(f"KataGo STDERR: {line.strip()}")

            self.stderr_thread = threading.Thread(target=read_stderr, args=(self.process,))
            self.stderr_thread.daemon = True  # Allow main program to exit even if thread is still running
            self.stderr_thread.start()

    def stop_engine(self):
        self.running = False
        if self.process:
            try:
                self.process.stdin.close()
            except (IOError, BrokenPipeError):
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            finally:
                self.process = None
        if self._master_reader_thread and self._master_reader_thread.is_alive():
            self._master_reader_thread.join(timeout=2)

    def get_batch_size(self) -> int:
        if self.batch_size is None:
            """Reads nnMaxBatchSize from a KataGo config file."""
            with open(CONFIG_FILE, 'r') as f:
                for line in f:
                    # Find the line defining nnMaxBatchSize and return its value
                    match = re.match(r"^\s*nnMaxBatchSize\s*=\s*(\d+)", line)
                    if match:
                        self.batch_size = int(match.group(1))
                        break
        return self.batch_size

    def _master_reader(self):
        """Master thread to read all stdout responses and signal events.

        Hardened against the three ways this thread used to die silently and
        hang every waiter for the full QUERY_TIMEOUT_SECONDS:
          - a non-JSON stdout line is skipped, not fatal (json.loads used to
            propagate straight out of the loop);
          - the membership check + .set() run under self.lock so they can't
            race send_query's timeout-path pop (which raised KeyError here);
          - the capture of the stdout handle up front, plus the finally,
            mean a concurrent stop_engine() setting self.process = None can't
            turn the loop condition into an AttributeError, and any exit path
            still unblocks every remaining waiter.
        """
        stdout = self.process.stdout if self.process else None
        try:
            while self.running and stdout is not None and not stdout.closed:
                response_str = stdout.readline()
                if not response_str:
                    break

                try:
                    response = json.loads(response_str)
                except ValueError:
                    # KataGo can emit a non-JSON diagnostic line to stdout;
                    # drop it rather than killing the reader for the run.
                    continue
                query_id = response.get("id")

                with self.lock:
                    if query_id in self.response_events:
                        self.responses[query_id] = response
                        self.response_events[query_id].set()
        finally:
            # On any exit path, signal all remaining waiters to prevent deadlocks.
            with self.lock:
                for event in self.response_events.values():
                    event.set()

    def send_query(self, query):
        """
        Sends a query, waits for the response, and returns it.
        This is a blocking, thread-safe call that hides the internal async logic.
        """
        event = threading.Event()
        query_id = None

        try:
            with self.lock:
                self.query_counter += 1
                query_id = str(self.query_counter)
                query["id"] = query_id
                self.response_events[query_id] = event
                query_str = json.dumps(query)

            # The write is OUTSIDE self.lock, and that placement is what keeps
            # both pipes moving. Holding self.lock across it deadlocks the pair:
            # once enough answers are in flight to fill KataGo's stdout pipe it
            # blocks in write() and stops reading stdin, so this write blocks
            # too — and _master_reader, which is the only thing that can drain
            # stdout and unblock it, needs self.lock to deliver the line it has
            # already read. Both sides then sit alive at zero throughput until
            # QUERY_TIMEOUT_SECONDS. It is rate-dependent, so it appears only
            # when a caller's queries outrun the reader (many worker threads,
            # cheap per-query Python work), not in a light run.
            with self.write_lock:
                self.process.stdin.write(query_str + "\n")
                self.process.stdin.flush()

            # Release lock and wait for the response event from the master reader
            if not event.wait(timeout=QUERY_TIMEOUT_SECONDS):
                raise TimeoutError(f"KataGo query {query_id} timed out after {QUERY_TIMEOUT_SECONDS} seconds. KataGo may be stuck or crashed.")

            response = self.responses.get(query_id)
            if not response:
                raise Exception(f"Engine may have exited while awaiting response for query {query_id}.")
            if "error" in response:
                raise Exception(f"KataGo error: {response['error']}")

            return response
        finally:
            # Clean up the dictionaries for this query_id to prevent memory leaks
            if query_id:
                with self.lock:
                    self.response_events.pop(query_id, None)
                    self.responses.pop(query_id, None)
