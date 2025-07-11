import subprocess
import json

class KatagoController:
    """
    A class to manage the KataGo analysis engine process.
    """
    def __init__(self, katago_path, model_path):
        """
        Initializes the KatagoController.

        Args:
            katago_path (str): The path to the KataGo executable.
            model_path (str): The path to the KataGo model file.
        """
        self.katago_path = katago_path
        self.model_path = model_path
        self.process = None

    def start_engine(self):
        """
        Starts the KataGo analysis engine as a subprocess.
        """
        try:
            command = [
                self.katago_path,
                "analysis",
                "-model",
                self.model_path,
                "-config",
                "--",
                "-override-config",
                "analysisThreads=1,numSearchThreads=1"
            ]
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
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