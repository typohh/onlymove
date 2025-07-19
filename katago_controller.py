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

    def query_root(self, moves, humanRank=None):
        """
        Sends a standardized query for root position analysis (1 visit).
        Returns rootInfo but no moveInfos.
        """
        query = {
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": LOW_PLAYOUTS,
            "includePolicy": True,  # Always include policy
            "id": id
        }
        if humanRank:
            query["overrideSettings"] = {
                "humanSLProfile": humanRank
            }
        return self.__send_query(query)

    def query_children(self, moves):
        """
        Sends a standardized query for children analysis (visits adjusted by the number of legal moves).
        Returns moveInfos but not necessarily accurate rootInfo.
        """
        # Get root policy to count legal moves
        root_response = self.query_root(moves)
        num_legal_moves = sum(1 for p in root_response["policy"] if p >= 0)

        query = {
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": num_legal_moves * LOW_PLAYOUTS,
            "rootPolicyTemperature": 100,
            "id": id
        }
        return self.__send_query(query)

    def query_children_100_visits(self, moves):
        """
        Sends a standardized query for children analysis with 100 visits per child.
        Returns moveInfos but not necessarily accurate rootInfo.
        """
        query = {
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": HIGH_PLAYOUTS,
            "id": id
        }
        result = self.__send_query(query)
        if not result or "moveInfos" not in result:
            raise Exception("Failed to get 100-visit response for solution move")
        return result

    def query_children_exclude_move(self, moves, exclude_move, player):
        """
        Sends a query for children analysis while excluding a specific move.
        Returns moveInfos but not necessarily accurate rootInfo.
        """
        query = {
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": HIGH_PLAYOUTS,
            "avoidMoves": [
                {
                    "player": player,
                    "moves": [exclude_move],
                    "untilDepth": 1
                }
            ],
            "id": id
        }
        return self.__send_query(query)

    def __send_query(self, query):
        """
        Sends a JSON query to the KataGo analysis engine.

        Args:
            query (dict): The query to send to the engine.

        Returns:
            dict: The response from the engine.
        """
        if not self.process:
            raise Exception("KataGo engine is not running.")

        query["id"]="random"

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

    def get_idx_from_move(self, move):
        """
        Converts a Go move string (e.g., "A1", "pass") to a KataGo policy index.
        Assumes a 9x9 board.
        """
        board_x_size = 9
        board_y_size = 9

        if move == "pass":
            return board_x_size * board_y_size

        col_char = move[0].upper()
        row_str = move[1:]

        # Convert column character to index
        col = ord(col_char) - ord('A')
        if col_char > 'H':  # Skip 'I'
            col -= 1

        # Convert row string to index
        row_num = int(row_str)
        row = board_y_size - row_num

        return row * board_x_size + col

    def get_position_info(self, moves):
        """
        Helper to get the current position score and winrate from Black's perspective.
        Returns (score, winrate) or (None, None) if failed.
        Uses maxVisits=10 for a quick but reliable read.
        """
        response = self.query_root(moves)
        if not response or "rootInfo" not in response:
            raise Exception("Failed to get position info from KataGo")

        score_lead = response["rootInfo"]["scoreLead"]
        winrate = response["rootInfo"]["winrate"]

        # Determine current player to move
        if not moves:
            current_player = "B"
        else:
            last_player = moves[-1][0]
            current_player = "W" if last_player == "B" else "B"

        return score_lead, winrate



def other_player(player):
    if player == "B":
        return "W"
    else:
        return "B"
