# FILE: katago_controller.py
import subprocess
import json
import threading
import time


class KatagoController:
    """
    Manages the KataGo analysis engine with a robust, encapsulated, and thread-safe query system.
    The external interface is a simple, blocking send_query call.
    """

    def __init__(self, katago_path, model_path, human_model_path=None):
        self.katago_path = katago_path
        self.model_path = model_path
        self.human_model_path = human_model_path
        self.process = None
        self._master_reader_thread = None
        self.running = False
        self.lock = threading.Lock()
        self.query_counter = 0
        self.response_events = {}  # Maps query_id -> threading.Event
        self.responses = {}  # Maps query_id -> response_dict

    def start_engine(self, verbose=False):
        command = [
            self.katago_path, "analysis",
            "-config", "/home/ootakamoku/Repositories/YoseTsumego/analysis.cfg",
            "-model", self.model_path
        ]
        if self.human_model_path:
            command.extend(["-human-model", self.human_model_path])

        print( command )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self.running = True
        self._master_reader_thread = threading.Thread(target=self._master_reader)
        self._master_reader_thread.daemon = True
        self._master_reader_thread.start()

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

    def send_query(self, query, timeout=300):
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
            if not event.wait(timeout=timeout):
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