import json
import os
import subprocess
import threading

CONFIG_FILE = "KataGo/cpp/configs/analysis_example.cfg"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KATAGO_PATH = os.path.join(BASE_DIR, "KataGo", "cpp", "katago")
SL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "b18c384nbt-humanv0.bin.gz")
RL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")

if __name__ == "__main__":
    command = [
        KATAGO_PATH, "analysis",
        "-config", CONFIG_FILE,
    ]
    command.extend(["-model", RL_MODEL_PATH])
    command.extend(["-human-model", SL_MODEL_PATH])

    print(command)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1
    )

    q = {
        "id": "123",
        "moves": [],
        "initialStones": [],
        "rules": "tromp-taylor",
        "komi": 7.5,
        "boardXSize": 9,
        "boardYSize": 9,
        "maxVisits": 1,
        "includePolicy": True,
        "overrideSettings": {
            "humanSLProfile": "rank_2d",
        }
    }


    def read_stderr(process):
        for line in process.stderr:
            print(f"KataGo STDERR: {line.strip()}")


    stderr_thread = threading.Thread(target=read_stderr, args=(process,))
    stderr_thread.daemon = True
    stderr_thread.start()

    query_str = json.dumps(q)
    print("sending query:", query_str)
    process.stdin.write(query_str + "\n")
    process.stdin.flush()

    response_str = process.stdout.readline()
    print("receiving response:", response_str)

    process.terminate()
