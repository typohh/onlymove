import json
import subprocess
import traceback
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
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


# --- KataGo Controller ---
class KataGoController:
    def __init__(self):
        self.katago_process = None
        self._lock = threading.Lock()
        self._query_counter = 0
        self._latest_query_for_client = {}
        self._response_events = {}
        self._responses = {}

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
        while True:
            line = self.katago_process.stdout.readline()
            if not line: break
            response = json.loads(line)
            query_id = response.get("id")
            if query_id:
                with self._lock:
                    self._responses[query_id] = response
                    event = self._response_events.get(query_id)
                if event: event.set()

    def submit_and_wait(self, query_data, client_ip):
        query_id = None
        event = threading.Event()
        with self._lock:
            self._query_counter += 1
            query_id = f"q_{self._query_counter}"
            old_query_id = self._latest_query_for_client.get(client_ip)
            self._latest_query_for_client[client_ip] = query_id
            self._response_events[query_id] = event
            query_data["id"] = query_id
            if old_query_id:
                old_event = self._response_events.get(old_query_id)
                if old_event:
                    self._responses[old_query_id] = {"error": "Request superseded by a newer one.", "is_cancelled": True}
                    old_event.set()
        query_json = json.dumps(query_data)
        self.katago_process.stdin.write(query_json + '\n')
        self.katago_process.stdin.flush()
        event.wait()
        with self._lock:
            response = self._responses.pop(query_id)
            self._response_events.pop(query_id)
            if self._latest_query_for_client.get(client_ip) == query_id:
                self._latest_query_for_client.pop(client_ip)
        return response


# --- HTTP Server and Handler ---
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


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

    def do_POST(self):
        start_time = time.time()
        status_message = "Unknown"
        try:
            client_ip = self.client_address[0]
            content_length = int(self.headers['Content-Length'])
            sgf_data = self.rfile.read(content_length).decode('utf-8')
            moves, komi, rules, board_size = self.parse_sgf(sgf_data)
            query_data = {
                "moves": moves, "initialStones": [], "rules": rules,
                "komi": komi, "boardXSize": board_size, "boardYSize": board_size, 
                "maxVisits": PLAYOUTS, "includePolicy": PLAYOUTS == 1
            }
            response = katago_controller.submit_and_wait(query_data, client_ip)
            if response.get("is_cancelled"):
                status_message = "Cancelled"
                self.send_response(499, "Client Closed Request")
                self.send_header('Content-type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                error_payload = json.dumps({"error": response["error"]})
                self.wfile.write(error_payload.encode('utf-8'))
                return
            if "error" in response: raise Exception(f"KataGo engine error: {response['error']}")

            if PLAYOUTS == 1:
                # When using 1 playout, use policy instead of moveInfos
                policy = response.get("policy", [])
                if not policy:
                    analysis_result = {"bestMove": "pass", "winrate": response.get('rootInfo', {}).get("winrate"), "score": response.get('rootInfo', {}).get('scoreLead')}
                else:
                    # Find the move with the highest policy probability
                    best_move_idx = policy.index(max(policy))
                    gtp_move = get_move_from_idx(best_move_idx, board_size)
                    sgf_move = gtp_to_sgf_coords(gtp_move, board_size)
                    analysis_result = {"bestMove": sgf_move, "winrate": response.get('rootInfo', {}).get("winrate"), "score": response.get('rootInfo', {}).get('scoreLead')}
            elif not response.get("moveInfos"):
                analysis_result = {"bestMove": "pass", "winrate": response.get('rootInfo', {}).get("winrate"), "score": response.get('rootInfo', {}).get('scoreLead')}
            else:
                best_move_info = response["moveInfos"][0]
                gtp_move = best_move_info.get("move")
                sgf_move = gtp_to_sgf_coords(gtp_move, board_size)
                analysis_result = {"bestMove": sgf_move, "winrate": best_move_info.get("winrate"), "score": best_move_info.get("scoreLead")}

            status_message = "Success"
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(analysis_result).encode('utf-8'))
        except Exception as e:
            status_message = f"Error: {type(e).__name__}"
            print("--- AN EXCEPTION OCCURRED ---")
            traceback.print_exc()
            print("-----------------------------")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            error_payload = json.dumps({"error": f"{type(e).__name__}: {e}"})
            self.wfile.write(error_payload.encode('utf-8'))
        finally:
            duration_ms = (time.time() - start_time) * 1000
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Request handled. Status: {status_message}. Time: {duration_ms:.2f} ms")

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


# --- Main Execution ---
if __name__ == '__main__':
    katago_controller = KataGoController()
    try:
        katago_controller.start()
        server_address = ('127.0.0.1', SERVER_PORT)
        httpd = ThreadingHTTPServer(server_address, KataGoHandler)
        print(f"Starting multi-threaded KataGo SGF server on http://127.0.0.1:{SERVER_PORT}")
        httpd.serve_forever()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
    finally:
        katago_controller.stop()