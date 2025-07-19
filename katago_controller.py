import subprocess
import json

LOW_PLAYOUTS = 10
HIGH_PLAYOUTS = 100


class KatagoController:
    """
    A class to manage the KataGo analysis engine process.
    """

    def __init__(self, katago_path, model_path, human_model_path=None):
        """
        Initializes the KatagoController.

        Args:
            katago_path (str): The path to the KataGo executable.
            model_path (str): The path to the KataGo model file (for the RL network).
            human_model_path (str, optional): The path to the KataGo human SL model file. Defaults to None.
        """
        self.katago_path = katago_path
        self.model_path = model_path
        self.human_model_path = human_model_path
        self.counter = 0
        self.process = None
        self.stderr_thread = None

    def start_engine(self, verbose=False):
        """
        Starts the KataGo analysis engine as a subprocess.

        Args:
            verbose (bool): If True, shows KataGo stderr output
        """
        try:
            command = [
                self.katago_path,
                "analysis",
                "-config",
                "/home/ootakamoku/Repositories/YoseTsumego/analysis.cfg",
                "-model",
                self.model_path
            ]
            stderr_setting = subprocess.PIPE if verbose else subprocess.DEVNULL
            if self.human_model_path:
                command.extend(["-human-model", self.human_model_path])
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_setting,
                text=True,
                bufsize=1  # Line-buffered
            )
            # Only start a thread to read stderr if verbose
            if verbose:
                import threading
                def read_stderr(process):
                    for line in process.stderr:
                        print(f"KataGo STDERR: {line.strip()}")

                self.stderr_thread = threading.Thread(target=read_stderr, args=(self.process,))
                self.stderr_thread.daemon = True  # Allow main program to exit even if thread is still running
                self.stderr_thread.start()
        except FileNotFoundError:
            raise Exception(f"KataGo executable not found at: {self.katago_path}")
        except Exception as e:
            raise Exception(f"Error starting KataGo engine: {e}")

    def stop_engine(self):
        """
        Stops the KataGo analysis engine.
        """
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait()

    def send_query(self, query):
        """
        Sends a JSON query to the KataGo analysis engine.

        Args:
            query (dict): The query to send to the engine.

        Returns:
            dict: The response from the engine.
        """
        if not self.process:
            raise Exception("KataGo engine is not running.")

        query["id"] = str(self.counter)
        self.counter += 1

        query_str = json.dumps(query)
        self.process.stdin.write(query_str + "\n")
        self.process.stdin.flush()

        return self._read_response()

    def _read_response(self):
        """
        Reads and parses a JSON response from the KataGo engine.

        Returns:
            dict: The parsed JSON response.

        Raises:
            Exception: If response is empty or contains error/warning.
        """
        response_str = self.process.stdout.readline()
        if not response_str:
            raise Exception("KataGo engine returned empty response")

        response = json.loads(response_str)
        if "error" in response:
            raise Exception(f"KataGo error: {response['error']}")
        if "warning" in response:
            raise Exception(f"KataGo warning: {response['warning']}")

        return response



