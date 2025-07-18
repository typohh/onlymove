import subprocess
import json

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
                bufsize=1 # Line-buffered
            )
            # Only start a thread to read stderr if verbose
            if verbose:
                import threading
                def read_stderr(process):
                    for line in process.stderr:
                        print(f"KataGo STDERR: {line.strip()}")
                self.stderr_thread = threading.Thread(target=read_stderr, args=(self.process,))
                self.stderr_thread.daemon = True # Allow main program to exit even if thread is still running
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

        query_str = json.dumps(query)
        self.process.stdin.write(query_str + "\n")
        self.process.stdin.flush()

        return self._read_response()

    def build_explore_children_query(self, moves, **kwargs):
        """
        Builds a standardized query object that explores all children.
        Note: You cannot use root score or winrate from this query.
        """
        query = {
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 25,  # Halved from original
            "rootPolicyTemperature": 49
        }
        query.update(kwargs)
        return query

    def _read_response(self):
        """
        Reads and parses a JSON response from the KataGo engine.

        Returns:
            dict: The parsed JSON response.
        """
        response_str = self.process.stdout.readline()
        if not response_str:
            return None
        return json.loads(response_str)

    def get_move_from_idx(self, idx):
        """
        Converts a KataGo policy index to a Go move string (e.g., "A1", "pass").
        Assumes a 9x9 board for now.
        """
        board_x_size = 9
        board_y_size = 9

        if idx == board_x_size * board_y_size:
            return "pass"

        row = idx // board_x_size
        col = idx % board_x_size

        # Convert to Go board coordinates (A1 is bottom-left, I9 is top-right)
        # KataGo uses 0-indexed from top-left, so A1 is (0,0) in KataGo's internal representation
        # and I9 is (8,8). We need to convert to standard Go notation.
        # Column: A=0, B=1, ..., I=8
        # Row: 1=8, 2=7, ..., 9=0

        col_char = chr(ord('A') + col)
        if col >= 8:  # Skip 'I'
            col_char = chr(ord('A') + col + 1)
        row_char = str(board_y_size - row)

        return f"{col_char}{row_char}"
