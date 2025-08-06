import json
import os
import re
import subprocess
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "analysis.cfg")
# CONFIG_FILE = os.path.join(BASE_DIR, "KataGo", "cpp", "configs", "analysis_example.cfg")
KATAGO_PATH = os.path.join(BASE_DIR, "bots", "katago")
SL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "b18c384nbt-humanv0.bin.gz")
# RL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")
RL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "kata9x9-b18c384nbt-20231025.bin.gz")


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
        self.query_counter = 0
        self.response_events = {}  # Maps query_id -> threading.Event
        self.responses = {}  # Maps query_id -> response_dict

    def start_engine(self, verbose=False, weight=RL_MODEL_PATH, human=True):
        command = [
            KATAGO_PATH, "analysis",
            "-config", CONFIG_FILE,
        ]
        command.extend(["-model", weight])
        if human:
            command.extend(["-human-model", SL_MODEL_PATH])

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
        """Master thread to read all stdout responses and signal events."""
        while self.running and self.process and not self.process.stdout.closed:
            response_str = self.process.stdout.readline()
            if not response_str: break

            response = json.loads(response_str)
            query_id = response.get("id")

            if query_id in self.response_events:
                self.responses[query_id] = response
                self.response_events[query_id].set()

        # On exit, signal all remaining waiters to prevent deadlocks
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
                self.process.stdin.write(query_str + "\n")
                self.process.stdin.flush()

            # Release lock and wait for the response event from the master reader
            if not event.wait(timeout=60 * 60):
                raise TimeoutError(f"KataGo query {query_id} timed out after {timeout} seconds. KataGo may be stuck or crashed.")

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
