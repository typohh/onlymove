"""Orchestrate Vast.ai instances for running onlymove pipeline steps remotely.

Two pipeline steps benefit from a beefier GPU than what's available locally:
  * pipeline.generate_tsumegos  — long-running tsumego search
  * pipeline.benchmark_bots     — bot performance evaluation
The rest (rank_tsumegos_and_bots, package_for_pwa, fetch_database) stays local.
This wrapper:
  * provisions a Vast.ai NVIDIA-GPU instance,
  * uploads pipeline/utils code, the KataGo binary, and the analysis cfg,
  * fetches KataGo network weights directly on the remote,
  * runs the chosen step in a tmux session,
  * lets you tail the log, sync the produced data back, kill the run, destroy.

Run without args for the command list.
"""

import json
import os
import subprocess
import sys
import time

VASTAI_BIN = ".venv/bin/vastai"
REMOTE_DIR = "/workspace/onlymove"
STATE_FILE = "orchestration_state.json"

# Pipeline steps allowed on the remote. Maps short name → python -m module path.
PIPELINE_STEPS = {
    "tune": "pipeline.tune_engine",
    "generate": "pipeline.generate_tsumegos",
    "benchmark": "pipeline.benchmark_bots",
}

# Files uploaded from the local repo. We ship the CUDA katago build to the
# remote (vast.ai NVIDIA GPUs); the local repo's bots/katago is OpenCL and is
# not uploaded. utils/katago_controller.py reads KATAGO_PATH from the
# environment, so we set that to bots/katago_cuda when launching the step.
LOCAL_UPLOAD_FILES = [
    "bots/analysis.cfg",
    "bots/katago_cuda",
]
REMOTE_KATAGO_PATH = "bots/katago_cuda"
LOCAL_UPLOAD_DIRS = ["pipeline", "utils"]

# Network weights — fetched on the remote (faster than uploading hundreds of MB).
# URLs mirror those in utils/models.py.
MODEL_DOWNLOADS = {
    "models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz":
        "https://media.katagotraining.org/uploaded/networks/models/kata1/"
        "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz",
    "models/kata1-b6c96-s175395328-d26788732.txt.gz":
        "https://media.katagotraining.org/uploaded/networks/models/kata1/"
        "kata1-b6c96-s175395328-d26788732.txt.gz",
    "models/kata9x9-b18c384nbt-20231025.bin.gz":
        "https://github.com/lightvector/KataGo/releases/download/"
        "v1.13.2-kata9x9/kata9x9-b18c384nbt-20231025.bin.gz",
    "models/b18c384nbt-humanv0.bin.gz":
        "https://github.com/lightvector/KataGo/releases/download/"
        "v1.15.0/b18c384nbt-humanv0.bin.gz",
}

# Output dirs to rsync back. Each contains per-size subdirs (e.g. tsumegos/13/).
REMOTE_OUTPUT_DIRS = ["tsumegos", "attempts"]

# Minimal pip deps for the two remote steps. The full requirements.txt brings
# torch/scipy/sklearn for rank+package, which we never run remotely.
REMOTE_PIP_PACKAGES = ["sgfmill>=1.1.1", "numpy>=1.21.0"]

# Single tmux session name for the running pipeline step.
TMUX_SESSION = "pipeline"
LOG_FILENAME = "pipeline.log"


# ---------------------------------------------------------------------------
# Vast.ai helpers
# ---------------------------------------------------------------------------

def run_vast_cmd(args):
    cmd = [VASTAI_BIN] + args + ["--raw"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = res.stdout.strip()
        if not output:
            return None
        return json.loads(output)
    except subprocess.CalledProcessError as e:
        print(f"VastAI Error: {e}\nstderr: {e.stderr.strip()}")
        sys.exit(1)
    except Exception as e:
        print(f"VastAI Error: {e}")
        sys.exit(1)


def load_env():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    if key.strip() == "VAST_API_KEY":
                        os.environ[key.strip()] = val.strip().strip("'").strip('"')


def get_all_instances():
    instances = run_vast_cmd(["show", "instances"])
    return instances if instances else []


def get_ssh_details(inst):
    proxy_host = inst.get("ssh_proxy_host")
    proxy_port = inst.get("ssh_proxy_port")
    if proxy_host and proxy_port:
        return proxy_host, int(proxy_port) + 1
    host = inst.get("ssh_host")
    port = inst.get("ssh_port")
    if host and port:
        return host, int(port) + 1
    return host, port


def get_ssh_cmds(host, port):
    base_ssh = [
        "ssh", "-q", "-o", "LogLevel=ERROR",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-p", str(port), f"root@{host}",
    ]
    base_rsync = [
        "rsync", "-avz", "-e",
        f"ssh -q -o LogLevel=ERROR -o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null -p {port}",
    ]
    return base_ssh, base_rsync


def wait_for_ssh(host, port):
    base_ssh, _ = get_ssh_cmds(host, port)
    while True:
        try:
            res = subprocess.run(
                base_ssh + [f"mkdir -p {REMOTE_DIR} {REMOTE_DIR}/models "
                            f"{REMOTE_DIR}/bots && touch ~/.hushlogin"],
                capture_output=True, text=True, timeout=15,
            )
            if res.returncode == 0:
                return
            print(f"  SSH not ready ({host}:{port}), retrying...")
        except subprocess.TimeoutExpired:
            print(f"  SSH timeout ({host}:{port}), retrying...")
        time.sleep(5)


def ensure_models(base_ssh, attempts=3):
    """Fetch KataGo network weights directly on the remote.

    Idempotent — files that already pass `gzip -t` are left alone, so re-running
    on a healthy host is a cheap no-op. curl --retry handles transient blips,
    -C - resumes a partial file, and the outer Python loop retries from scratch
    (rm + redownload) if the file ends up corrupt after curl gives up.
    """
    for remote_rel, url in MODEL_DOWNLOADS.items():
        remote_path = f"{REMOTE_DIR}/{remote_rel}"
        cmd = (
            f"mkdir -p \"$(dirname {remote_path})\" && "
            f"if [ -s {remote_path} ] && gzip -t {remote_path} 2>/dev/null; then "
            f"  echo \"have {remote_rel}\"; "
            f"else "
            f"  curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 -C - "
            f"-o {remote_path} {url} && gzip -t {remote_path} || "
            f"{{ rm -f {remote_path}; exit 1; }}; "
            f"fi"
        )
        for attempt in range(attempts):
            print(f"  fetching {remote_rel} (attempt {attempt + 1}/{attempts})...")
            res = subprocess.run(base_ssh + [cmd])
            if res.returncode == 0:
                break
            time.sleep(5)
        else:
            print(f"ERROR: failed to fetch {remote_rel} after {attempts} attempts.")
            return False
    return True


# ---------------------------------------------------------------------------
# Local state (instance ID <-> running step)
# ---------------------------------------------------------------------------

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def prune_state(state, live_instances):
    """Drop state entries whose instance ID no longer exists on vast.ai."""
    live_ids = {str(inst["id"]) for inst in live_instances}
    stale = [(k, v.get("step")) for k, v in state.items() if k not in live_ids]
    for k, _ in stale:
        del state[k]
    return stale


_SENTINEL = object()


def register_instance(inst_id, step=_SENTINEL, started=_SENTINEL):
    """Add or update an instance in local state.

    Pass None explicitly to clear a field. Omit to keep existing value.
    """
    state = load_state()
    key = str(inst_id)
    prev = state.get(key, {})
    state[key] = {
        "step": prev.get("step") if step is _SENTINEL else step,
        "started": prev.get("started") if started is _SENTINEL else started,
    }
    save_state(state)


def unregister_instance(inst_id):
    state = load_state()
    state.pop(str(inst_id), None)
    save_state(state)


def get_instance_step(inst_id):
    info = load_state().get(str(inst_id), {})
    return info.get("step")


def find_instance(target):
    """Find an instance by numeric ID."""
    instances = get_all_instances()
    try:
        target_id = int(target)
        inst = next((i for i in instances if i["id"] == target_id), None)
        if inst:
            return inst
    except ValueError:
        pass
    print(f"Instance '{target}' not found.")
    return None


# ---------------------------------------------------------------------------
# Code/data upload helpers
# ---------------------------------------------------------------------------

def upload_code(host, base_rsync):
    """Push pipeline/, utils/, requirements.txt, bots/analysis.cfg, bots/katago.

    Excludes __pycache__ and .pyc files.
    """
    excludes = ["--exclude", "__pycache__/", "--exclude", "*.pyc"]
    for d in LOCAL_UPLOAD_DIRS:
        if not os.path.isdir(d):
            print(f"  Warning: {d}/ not found, skipping")
            continue
        res = subprocess.run(
            base_rsync + excludes + [f"{d}/", f"root@{host}:{REMOTE_DIR}/{d}/"],
        )
        if res.returncode != 0:
            print(f"  {d}/ upload failed.")
            return False
    for f in LOCAL_UPLOAD_FILES:
        if not os.path.exists(f):
            print(f"  Warning: {f} not found, skipping")
            continue
        # Preserve subdir layout (e.g. bots/katago → /workspace/onlymove/bots/katago)
        remote_subdir = os.path.dirname(f)
        if remote_subdir:
            res = subprocess.run(
                base_rsync + [f, f"root@{host}:{REMOTE_DIR}/{remote_subdir}/"],
            )
        else:
            res = subprocess.run(base_rsync + [f, f"root@{host}:{REMOTE_DIR}/"])
        if res.returncode != 0:
            print(f"  {f} upload failed.")
            return False
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def setup_instance(inst):
    """Install prerequisites, upload code + katago, fetch model weights."""
    inst_id = inst["id"]
    host, port = get_ssh_details(inst)
    if not host or not port:
        print(f"Instance {inst_id} not reachable yet.")
        return False

    print(f"Setting up instance {inst_id} ({host}:{port})...")
    wait_for_ssh(host, port)

    base_ssh, base_rsync = get_ssh_cmds(host, port)

    # System packages: python, rsync, tmux, curl for model downloads. The
    # CUDA katago build needs libcublas + cuDNN, which already ship in the
    # nvidia/cuda:*-cudnn-runtime image we provision below.
    apt_cmd = ("bash -l -c 'apt-get update && apt-get install -y "
               "python3-pip rsync tmux curl'")
    for attempt in range(3):
        res = subprocess.run(base_ssh + [apt_cmd])
        if res.returncode == 0:
            break
        print(f"  apt install failed (attempt {attempt + 1}/3), retrying...")
        time.sleep(5)
    else:
        print("ERROR: apt install failed after 3 attempts.")
        return False

    # Minimal python deps for the two remote steps.
    pip_cmd = (f"bash -l -c 'python3 -m pip install --break-system-packages "
               f"{' '.join(REMOTE_PIP_PACKAGES)}'")
    for attempt in range(3):
        res = subprocess.run(base_ssh + [pip_cmd])
        if res.returncode == 0:
            break
        print(f"  pip install failed (attempt {attempt + 1}/3), retrying...")
        time.sleep(5)
    else:
        print("ERROR: pip install failed after 3 attempts.")
        return False

    if not upload_code(host, base_rsync):
        return False

    # Make katago executable on the remote.
    subprocess.run(base_ssh + [f"chmod +x {REMOTE_DIR}/{REMOTE_KATAGO_PATH}"])

    if not ensure_models(base_ssh):
        return False

    register_instance(inst_id)
    print(f"Instance {inst_id} ready.")
    return True


def setup_instance_with_retry(inst, attempts=3):
    for attempt in range(attempts):
        try:
            if setup_instance(inst):
                return True
        except subprocess.CalledProcessError as e:
            print(f"  setup raised: {e}")
        if attempt < attempts - 1:
            print(f"Setup did not complete (attempt {attempt + 1}/{attempts}), retrying in 10s...")
            time.sleep(10)
    print(f"ERROR: setup failed after {attempts} attempts.")
    return False


def cmd_add():
    """Create a new Vast.ai NVIDIA-GPU instance and run setup."""
    print("Searching for RTX 5090 offers...")
    offers = run_vast_cmd(["search", "offers",
                           "gpu_name=RTX_5090 rented=False rentable=True"])
    if not offers:
        print("No RTX 5090 offers; falling back to RTX 4090.")
        offers = run_vast_cmd(["search", "offers",
                               "gpu_name=RTX_4090 rented=False rentable=True"])
    if not offers:
        print("No offers found.")
        return
    offers = [o for o in offers if o.get("dph_total", 0) >= 0.1]
    if not offers:
        print("No valid offers (dph_total filter).")
        return
    offers.sort(key=lambda x: x.get("dph_total", float("inf")))

    best = offers[0]
    print(f"Creating instance ({best.get('gpu_name', '?')}, "
          f"${best['dph_total']:.2f}/hr)...")
    res = run_vast_cmd([
        "create", "instance", str(best["id"]),
        # cudnn-runtime brings libcublas and libcudnn that the CUDA katago
        # build links against. The plain base image does not.
        "--image", "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        "--disk", "32", "--ssh",
    ])
    instance_id = res["new_contract"]

    print(f"Waiting for instance {instance_id}...")
    while True:
        instances = run_vast_cmd(["show", "instances"])
        inst = next((i for i in instances if i["id"] == instance_id), None)
        if inst:
            host, port = get_ssh_details(inst)
            if host and port:
                break
        time.sleep(5)

    setup_instance_with_retry(inst)


def cmd_setup(target):
    """Re-run setup on an existing instance."""
    inst = find_instance(target)
    if not inst:
        return
    setup_instance_with_retry(inst)


def _is_step_running(base_ssh):
    res = subprocess.run(
        base_ssh + ["pgrep -f '[p]ython3.*pipeline\\.' >/dev/null 2>&1 "
                    "&& echo RUNNING || echo IDLE"],
        capture_output=True, text=True, timeout=15,
    )
    return "RUNNING" in (res.stdout or "")


def cmd_run(args):
    """Start a pipeline step in tmux on an instance.

    Usage: run <id> <step> [-- <step args...>]
    """
    if len(args) < 2:
        print(f"Usage: run <id> <step> [args...]   step ∈ {{{', '.join(PIPELINE_STEPS)}}}")
        return
    target, step, *step_args = args
    if step not in PIPELINE_STEPS:
        print(f"Unknown step '{step}'. Available: {', '.join(PIPELINE_STEPS)}")
        return

    inst = find_instance(target)
    if not inst:
        return
    inst_id = inst["id"]
    host, port = get_ssh_details(inst)
    if not host or not port:
        print(f"Instance {inst_id} not reachable.")
        return
    base_ssh, _ = get_ssh_cmds(host, port)

    if _is_step_running(base_ssh):
        print(f"Instance {inst_id} already has a pipeline step running. "
              f"Run 'stop' first.")
        return

    module = PIPELINE_STEPS[step]
    extra = " ".join(step_args)
    log_path = f"{REMOTE_DIR}/{LOG_FILENAME}"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    print(f"Starting '{step}' on instance {inst_id}...")
    cmd = (
        f"echo '{step}' > {REMOTE_DIR}/running_step && "
        f"date -Iseconds > {REMOTE_DIR}/running_started && "
        f": > {log_path} && "
        f"cd {REMOTE_DIR} && "
        f"tmux new-session -d -s {TMUX_SESSION} "
        f"'KATAGO_PATH={REMOTE_KATAGO_PATH} PYTHONUNBUFFERED=1 "
        f"python3 -u -m {module} {extra} "
        f">> {log_path} 2>&1'"
    )
    res = subprocess.run(base_ssh + [cmd])
    if res.returncode != 0:
        print(f"  failed to start tmux session.")
        return
    register_instance(inst_id, step=step, started=started)
    print(f"Instance {inst_id} running '{step}' (tmux '{TMUX_SESSION}').")
    print(f"  log: {log_path}")
    print(f"  follow with: stream {inst_id}")


def cmd_stop(target):
    """Kill the running pipeline step (instance stays alive)."""
    inst = find_instance(target)
    if not inst:
        return False
    inst_id = inst["id"]
    host, port = get_ssh_details(inst)
    if not host or not port:
        print(f"Instance {inst_id} not reachable.")
        return False
    base_ssh, _ = get_ssh_cmds(host, port)

    print(f"Stopping pipeline step on instance {inst_id}...")
    # SIGTERM first so generate_tsumegos's KeyboardInterrupt branch can clean up
    # the KataGo subprocess. Then wait, then force-kill stragglers.
    stop_cmd = (
        f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null; "
        f"pkill -f '[p]ython3.*pipeline\\.' 2>/dev/null; "
        f"for i in $(seq 1 30); do "
        f"  pgrep -f '[p]ython3.*pipeline\\.' >/dev/null 2>&1 || break; "
        f"  sleep 1; "
        f"done; "
        f"pkill -9 -f '[p]ython3.*pipeline\\.' 2>/dev/null; "
        f"pkill -9 -f '[k]atago analysis' 2>/dev/null; "
        f"sleep 1; "
        f"if pgrep -f '[p]ython3.*pipeline\\.' >/dev/null 2>&1; then "
        f"  echo STILL_RUNNING; exit 1; "
        f"fi; "
        f"rm -f {REMOTE_DIR}/running_step {REMOTE_DIR}/running_started; "
        f"echo CLEAN"
    )
    try:
        res = subprocess.run(base_ssh + [stop_cmd],
                             capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        print("  stop timed out; local state NOT cleared.")
        return False

    if res.returncode != 0 or "CLEAN" not in (res.stdout or ""):
        print("  stop did not complete cleanly; local state NOT cleared.")
        out = (res.stdout or "").strip().splitlines()[-5:]
        err = (res.stderr or "").strip().splitlines()[-5:]
        if out:
            print("  stdout: " + " | ".join(out))
        if err:
            print("  stderr: " + " | ".join(err))
        return False

    register_instance(inst_id, step=None, started=None)
    print(f"Instance {inst_id} step stopped.")
    return True


def cmd_shutdown(target):
    """Destroy an instance and remove it from local state."""
    inst = find_instance(target)
    if not inst:
        return
    inst_id = inst["id"]
    print(f"Destroying instance {inst_id}...")
    run_vast_cmd(["destroy", "instance", str(inst_id)])
    unregister_instance(inst_id)
    print(f"Instance {inst_id} destroyed.")


def cmd_sync(target):
    """rsync tsumegos/ and attempts/ from the remote into the local repo."""
    inst = find_instance(target)
    if not inst:
        return
    inst_id = inst["id"]
    host, port = get_ssh_details(inst)
    if not host or not port:
        print(f"Instance {inst_id} not reachable.")
        return
    _, base_rsync = get_ssh_cmds(host, port)

    print(f"Syncing outputs from instance {inst_id}...")
    for d in REMOTE_OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)
        # Trailing slash on src + dst means "copy contents".
        # Don't pull database.sql back — it's a local-only D1 dump.
        excludes = ["--exclude", "database.sql"]
        res = subprocess.run(
            base_rsync + excludes + [
                f"root@{host}:{REMOTE_DIR}/{d}/",
                f"{d}/",
            ],
            timeout=600,
        )
        if res.returncode == 0:
            print(f"  {d}/ synced.")
        else:
            print(f"  {d}/ sync had errors (rsync rc={res.returncode}).")


def cmd_stream(target):
    """Continuously print new lines from the pipeline log."""
    inst = find_instance(target)
    if not inst:
        return
    inst_id = inst["id"]
    host, port = get_ssh_details(inst)
    if not host or not port:
        print(f"Instance {inst_id} not reachable.")
        return
    base_ssh, _ = get_ssh_cmds(host, port)
    log_path = f"{REMOTE_DIR}/{LOG_FILENAME}"

    step = get_instance_step(inst_id) or "?"
    print(f"Streaming {log_path} for step '{step}' (Ctrl-C to stop)...")

    offset = 0
    interval = 2.0
    min_interval = 0.5
    max_interval = 30.0

    try:
        while True:
            try:
                res = subprocess.run(
                    base_ssh + [f"tail -c +{offset + 1} {log_path} 2>/dev/null"],
                    capture_output=True, text=True, timeout=15,
                )
                new_data = res.stdout
            except subprocess.TimeoutExpired:
                new_data = ""

            if new_data:
                print(new_data, end="", flush=True)
                offset += len(new_data.encode())
                interval = max(min_interval, interval / 1.05)
            else:
                interval = min(max_interval, interval * 1.2155)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def cmd_update(target):
    """Re-upload code (no restart). To pick up changes on a running step,
    'stop' it first then 'run' again.
    """
    inst = find_instance(target)
    if not inst:
        return
    inst_id = inst["id"]
    host, port = get_ssh_details(inst)
    if not host or not port:
        print(f"Instance {inst_id} not reachable.")
        return
    _, base_rsync = get_ssh_cmds(host, port)
    print(f"Re-uploading code to instance {inst_id}...")
    if upload_code(host, base_rsync):
        print(f"Instance {inst_id} code updated. (No restart — use stop+run.)")


def _probe_instance(inst, state):
    """Probe a single instance via SSH. Returns a one-line summary."""
    inst_id = inst["id"]
    info = state.get(str(inst_id), {})
    step = info.get("step") or "-"
    started = info.get("started") or "-"

    def fmt_line(status="-", uptime="-", remote_step="-"):
        return f"{inst_id:>10}  {step:<10}  {remote_step:<10}  {status:<10}  {uptime:>8}"

    host, port = get_ssh_details(inst)
    if not host or not port:
        return inst_id, fmt_line(status="starting")

    base_ssh, _ = get_ssh_cmds(host, port)

    gather_cmd = (
        f"pgrep -f '[p]ython3.*pipeline\\.' >/dev/null 2>&1 && echo running || echo idle; "
        f"echo '|||'; "
        f"cat {REMOTE_DIR}/running_step 2>/dev/null || echo '-'"
    )

    try:
        res = subprocess.run(base_ssh + [gather_cmd],
                             capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            return inst_id, fmt_line(status="error")

        parts = [p.strip() for p in res.stdout.split("|||")]
        parts += [""] * max(0, 2 - len(parts))
        proc_state, remote_step = parts[:2]
        if not remote_step:
            remote_step = "-"

        uptime = "-"
        if started and started != "-":
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(started)
                now = datetime.now(start_dt.tzinfo) if start_dt.tzinfo else datetime.now()
                delta = now - start_dt
                secs = int(delta.total_seconds())
                h = secs // 3600
                m = (secs % 3600) // 60
                uptime = f"{h}h{m:02d}m"
            except Exception:
                pass

        # Mismatch between local-tracked step and remote-running step is worth
        # surfacing — usually means a stop didn't update local state, or
        # something restarted out-of-band.
        if step != "-" and remote_step != "-" and step != remote_step:
            status = "mismatch"
        elif step == "-":
            status = proc_state  # idle / running (orphaned)
        elif proc_state == "running":
            status = "ok"
        else:
            status = "stopped"

        return inst_id, fmt_line(status=status, uptime=uptime,
                                 remote_step=remote_step)
    except Exception:
        return inst_id, fmt_line(status="timeout")


def cmd_list(_args):
    """Show status of all instances."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    instances = get_all_instances()
    state = load_state()

    stale = prune_state(state, instances)
    if stale:
        save_state(state)
        for _id, _step in stale:
            print(f"  pruned stale state: {_id} ({_step or 'no step'})")

    if not instances:
        print("No instances.")
        return

    header = (f"{'ID':>10}  {'LocalStep':<10}  {'RemoteStep':<10}  "
              f"{'Status':<10}  {'Uptime':>8}")
    print(header)
    print("-" * len(header))

    with ThreadPoolExecutor(max_workers=max(1, len(instances))) as pool:
        futures = {pool.submit(_probe_instance, inst, state): inst["id"]
                   for inst in instances}
        results = {}
        for fut in as_completed(futures):
            inst_id, line = fut.result()
            results[inst_id] = line

    for inst in instances:
        print(results[inst["id"]])


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def _require_arg(usage, fn):
    def handler(args):
        if not args:
            print(f"Usage: {usage}")
            return
        fn(args[0])
    return handler


COMMANDS = {
    "add": {
        "usage": "add",
        "help": ["Create a new GPU instance and run setup"],
        "handler": lambda args: cmd_add(),
    },
    "setup": {
        "usage": "setup <id>",
        "help": ["Re-run setup on an existing instance"],
        "handler": _require_arg("setup <id>", cmd_setup),
    },
    "run": {
        "usage": "run <id> <step> [args...]",
        "help": [
            f"Start a pipeline step in tmux. step ∈ {{{', '.join(PIPELINE_STEPS)}}}",
            "Extra args are forwarded to python -m pipeline.<step>",
        ],
        "handler": cmd_run,
    },
    "stop": {
        "usage": "stop <id>",
        "help": ["Kill the running pipeline step (instance stays alive)"],
        "handler": _require_arg("stop <id>", cmd_stop),
    },
    "shutdown": {
        "usage": "shutdown <id>",
        "help": ["Destroy an instance"],
        "handler": _require_arg("shutdown <id>", cmd_shutdown),
    },
    "sync": {
        "usage": "sync <id>",
        "help": ["Pull tsumegos/ and attempts/ from remote into the local repo"],
        "handler": _require_arg("sync <id>", cmd_sync),
    },
    "stream": {
        "usage": "stream <id>",
        "help": ["Continuously tail the pipeline log on an instance"],
        "handler": _require_arg("stream <id>", cmd_stream),
    },
    "update": {
        "usage": "update <id>",
        "help": ["Re-upload pipeline/utils/cfg/katago code (does NOT restart)"],
        "handler": _require_arg("update <id>", cmd_update),
    },
    "list": {
        "usage": "list",
        "help": ["Show status of all instances"],
        "handler": cmd_list,
    },
}


def help_text():
    lines = [__doc__.strip(), "", "Commands:"]
    width = max(len(c["usage"]) for c in COMMANDS.values())
    for name in COMMANDS:
        c = COMMANDS[name]
        lines.append(f"  {c['usage']:<{width}}   {c['help'][0]}")
        for extra in c["help"][1:]:
            lines.append(f"  {' ' * width}   {extra}")
    return "\n".join(lines)


def main():
    load_env()
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(help_text())
        sys.exit(1)
    COMMANDS[sys.argv[1]]["handler"](sys.argv[2:])


if __name__ == "__main__":
    main()
