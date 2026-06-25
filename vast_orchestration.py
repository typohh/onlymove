"""Orchestrate Vast.ai instances for running onlymove pipeline steps remotely.

Two pipeline steps benefit from a beefier GPU than what's available locally:
  * pipeline.generate_tsumegos  — long-running tsumego search
  * pipeline.benchmark_bots     — bot performance evaluation
The rest (train_rating, package_for_pwa, fetch_database) stays local.
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

from utils.configurations import ALL_BOARD_SIZES

VASTAI_BIN = ".venv/bin/vastai"
REMOTE_DIR = "/workspace/onlymove"
STATE_FILE = "orchestration_state.json"

# Pipeline steps allowed on the remote. Maps short name → python -m module path.
# `calibrate19` and `calibrate_handicap` are auxiliary diagnostics, not
# main-pipeline steps, but they fit the same provision-upload-run-sync flow.
PIPELINE_STEPS = {
    "tune": "pipeline.tune_engine",
    "generate": "pipeline.generate_tsumegos",
    "benchmark": "pipeline.benchmark_bots",
    "calibrate19": "auxiliary.calibrate_ranks_19x19",
    "calibrate_handicap": "auxiliary.calibrate_handicap_ranks",
}

# Steps that take a mandatory positional board size as their first arg — it's
# forwarded verbatim to the remote `python -m pipeline.<step> <size>`, which
# selects tsumegos/<size>/, attempts/<size>/, and the engine board dimensions.
# The calibrate* steps are fixed-board auxiliary diagnostics and take no size.
SIZE_REQUIRED_STEPS = {"tune", "generate", "benchmark"}

# Files uploaded from the local repo. We ship the CUDA katago build to the
# remote (vast.ai NVIDIA GPUs); the local repo's bots/katago is OpenCL and is
# not uploaded. utils/katago_controller.py reads KATAGO_PATH from the
# environment, so we set that to bots/katago_cuda when launching the step.
LOCAL_UPLOAD_FILES = [
    "bots/katago_cuda",
]
# bots/analysis.cfg is special: it's hardware-tuned (numAnalysisThreads,
# nnMaxBatchSize) by the remote tune step. Uploading from local would clobber
# those values. So it's only pushed during initial `setup`; `update` skips it.
SETUP_ONLY_UPLOAD_FILES = [
    "bots/analysis.cfg",
]
REMOTE_KATAGO_PATH = "bots/katago_cuda"
LOCAL_UPLOAD_DIRS = ["pipeline", "utils", "auxiliary"]

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
# calibration_19x19_sgf and calibration_handicap_sgf are the SGF dumps from
# the calibrate19 / calibrate_handicap steps.
REMOTE_OUTPUT_DIRS = ["tsumegos", "attempts",
                      "calibration_19x19_sgf", "calibration_handicap_sgf"]

# Per-step input dirs pushed TO the remote before launching a step. These are
# state dirs the step reads to resume incrementally. For benchmark, attempts/
# holds prior bot results — uploading them lets test_bot skip already-processed
# tsumegos and append only new entries (it already merges existing+new on write).
# For calibrate_handicap, the rank fit and pair-selection priorities are
# seeded from the existing SGF corpus; without these, the remote starts cold
# and wastes games re-learning the kyu / dan gaps.
STEP_INPUT_DIRS = {
    "benchmark": ["attempts"],
    "calibrate_handicap": ["calibration_handicap_sgf"],
}

# Minimal pip deps for the remote steps. The full requirements.txt brings
# torch/sklearn for rank+package, which we never run remotely. scipy is
# needed only by calibrate_handicap (logistic MLE refit) but is cheap to
# install, so we add it unconditionally rather than branching by step.
REMOTE_PIP_PACKAGES = ["sgfmill>=1.1.1", "numpy>=1.21.0", "scipy>=1.10.0"]

# tmux session and log files use the prefix; multi-instance generate runs use
# {prefix}_{i} so they don't collide.
TMUX_SESSION_PREFIX = "pipeline"
LOG_FILENAME_PREFIX = "pipeline"

# Auto-tuner for parallel `generate` instances. We can't predict the optimal N
# from static probing — CUDA-context contention on a single GPU is hard to
# model without measuring. Instead, ramp up incrementally and watch GPU util:
#  * keep adding while util increases by ≥ TUNE_THRESHOLD
#  * stop when it plateaus or drops (the N=7 collapse case)
# Result is cached per (GPU model, board size) in state["_gpu_cache"], so
# subsequent runs on a same-GPU host (or even a brand-new vast.ai instance with
# the same GPU) at that board size skip the ramp.
TUNE_WARMUP_SECS = 90    # KataGo b18c384nbt cold start on a fast GPU
TUNE_SAMPLE_SECS = 30    # window over which GPU util is averaged
TUNE_SAMPLE_INTERVAL = 2 # seconds between nvidia-smi samples
TUNE_THRESHOLD = 0.05    # 5% gain to count as meaningful improvement
TUNE_HARD_CAP = 8        # safety cap on the doubling-style ramp
# CPU floor: max N a host with a small CPU could actually run regardless of
# what the GPU cache says. ~10 cores per instance was measured at batch=64
# (~8.5 katago + ~1 python).
CPU_PER_INSTANCE = 10
INSTANCE_HEADROOM = 0.8


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
    """Drop state entries whose instance ID no longer exists on vast.ai.

    Keys starting with `_` are non-instance metadata (e.g. `_gpu_cache`) and
    must be skipped — otherwise the cache gets pruned every time `list` runs
    against an instance set that doesn't contain a key called `_gpu_cache`.
    """
    live_ids = {str(inst["id"]) for inst in live_instances}
    stale = [(k, v.get("step")) for k, v in state.items()
             if not k.startswith("_") and k not in live_ids]
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


def find_instance(target=None):
    """Find an instance by numeric ID, or auto-select if only one exists."""
    instances = get_all_instances()
    if target is None:
        if len(instances) == 1:
            return instances[0]
        if not instances:
            print("No instances. Run 'add' first.")
            return None
        ids = ", ".join(str(i["id"]) for i in instances)
        print(f"Multiple instances; specify ID. Available: {ids}")
        return None
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

def upload_code(host, base_rsync, include_setup_only=False):
    """Push pipeline/, utils/, the katago binary, and (when include_setup_only)
    bots/analysis.cfg. Excludes __pycache__ and .pyc files.

    `include_setup_only` is True from `setup` and False from `update`, so the
    remote's hardware-tuned analysis.cfg survives code refreshes.
    """
    excludes = ["--exclude", "__pycache__/", "--exclude", "*.pyc"]
    files = list(LOCAL_UPLOAD_FILES)
    if include_setup_only:
        files = list(SETUP_ONLY_UPLOAD_FILES) + files
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
    for f in files:
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


def upload_input_dirs(host, base_rsync, dirs):
    """Push state dirs (e.g. attempts/) to the remote so a step can resume.

    Excludes database.sql (local-only D1 dump) — same as cmd_sync.
    """
    excludes = ["--exclude", "database.sql", "--exclude", "__pycache__/",
                "--exclude", "*.pyc"]
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  {d}/ not present locally, skipping upload")
            continue
        res = subprocess.run(
            base_rsync + excludes + [f"{d}/", f"root@{host}:{REMOTE_DIR}/{d}/"],
            timeout=1800,
        )
        if res.returncode != 0:
            print(f"  {d}/ upload failed (rsync rc={res.returncode}).")
            return False
        print(f"  {d}/ uploaded.")
    return True


def _probe_remote_capacity(base_ssh):
    """Return (gpu_model, cpu_cores) from the remote, or (None, None) on error."""
    try:
        res = subprocess.run(
            base_ssh + [
                "nproc; "
                "nvidia-smi --query-gpu=name --format=csv,noheader | head -1"
            ],
            capture_output=True, text=True, timeout=15,
        )
        if res.returncode != 0:
            return None, None
        lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
        cores = int(lines[0])
        gpu_model = lines[1] if len(lines) > 1 else None
        return gpu_model, cores
    except (ValueError, IndexError, subprocess.TimeoutExpired):
        return None, None


def _spawn_pipeline_session(base_ssh, idx, module, extra=""):
    """Start one pipeline_<idx> tmux session running `python -m module extra`."""
    log_path = f"{REMOTE_DIR}/{LOG_FILENAME_PREFIX}_{idx}.log"
    session = f"{TMUX_SESSION_PREFIX}_{idx}"
    cmd = (
        f": > {log_path} && "
        f"cd {REMOTE_DIR} && "
        f"tmux new-session -d -s {session} "
        f"'KATAGO_PATH={REMOTE_KATAGO_PATH} PYTHONUNBUFFERED=1 "
        f"python3 -u -m {module} {extra} "
        f">> {log_path} 2>&1'"
    )
    res = subprocess.run(base_ssh + [cmd])
    return res.returncode == 0


def _kill_pipeline_session(base_ssh, idx):
    """Stop a single pipeline_<idx> session.

    tmux kill-session sends SIGHUP; generate_tsumegos.py installs a SIGHUP
    handler that raises KeyboardInterrupt, so its finally calls stop_engine
    on KataGo and katago exits cleanly.
    """
    cmd = (
        f"tmux kill-session -t {TMUX_SESSION_PREFIX}_{idx} 2>/dev/null; "
        f"sleep 5"
    )
    subprocess.run(base_ssh + [cmd], capture_output=True)


def _wait_and_measure_util(base_ssh, warmup_secs, sample_secs,
                           interval=TUNE_SAMPLE_INTERVAL):
    """Sleep `warmup_secs`, then sample GPU util over `sample_secs`. Returns avg %."""
    samples = max(1, sample_secs // interval)
    cmd = (
        f"sleep {warmup_secs}; "
        f"sum=0; n=0; "
        f"for i in $(seq 1 {samples}); do "
        f"  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1); "
        f"  sum=$((sum + u)); n=$((n + 1)); "
        f"  sleep {interval}; "
        f"done; "
        f"echo \"$sum $n\""
    )
    timeout = warmup_secs + sample_secs + 30
    try:
        res = subprocess.run(base_ssh + [cmd], capture_output=True,
                             text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    util measurement ssh timed out after {timeout}s")
        return 0.0
    if res.returncode != 0:
        print(f"    util measurement failed (rc={res.returncode})")
        return 0.0
    parts = (res.stdout or "").strip().split()
    if len(parts) < 2:
        return 0.0
    try:
        return int(parts[0]) / max(1, int(parts[1]))
    except ValueError:
        return 0.0


def tune_parallelism(base_ssh, module, extra=""):
    """Ramp parallel instances incrementally until GPU util plateaus.

    Spawns instances one at a time. After each spawn, waits TUNE_WARMUP_SECS
    for KataGo to warm up, then samples nvidia-smi GPU util over
    TUNE_SAMPLE_SECS. Stops when a step adds < TUNE_THRESHOLD relative
    improvement, kills the most recent instance, and returns the prior N.

    On exit `final_n` instances are running on the remote.
    """
    print("Auto-tuning parallel instance count (each step ~2 min)...")
    util_history = []
    final_n = 0

    for n in range(1, TUNE_HARD_CAP + 1):
        if not _spawn_pipeline_session(base_ssh, n - 1, module, extra):
            print(f"  failed to spawn instance {n - 1}")
            break
        print(f"  spawned instance #{n - 1}; warmup {TUNE_WARMUP_SECS}s + "
              f"sample {TUNE_SAMPLE_SECS}s at N={n}...")
        util = _wait_and_measure_util(base_ssh, TUNE_WARMUP_SECS, TUNE_SAMPLE_SECS)
        print(f"    GPU util at N={n}: {util:.1f}%")

        if util_history and util < util_history[-1] * (1 + TUNE_THRESHOLD):
            change_pct = (util / util_history[-1] - 1) * 100 if util_history[-1] else 0
            print(f"  no meaningful improvement ({change_pct:+.1f}%); "
                  f"reverting to N={n - 1}")
            _kill_pipeline_session(base_ssh, n - 1)
            break
        util_history.append(util)
        final_n = n

    if final_n == 0:
        print("  WARNING: tune could not establish even N=1; defaulting to N=1")
        final_n = 1
    print(f"  settled at N={final_n}")
    return final_n


def determine_parallelism(base_ssh, step, module, extra=""):
    """Resolve N for `step`. For non-generate, always 1. For generate, look in
    the per-(GPU model, board size) cache and use it (with a CPU floor); else
    run the ramp, cache the result.

    On return, if `step == 'generate'` and we ran the tune, the chosen N
    instances are already running on the remote. If we used the cache, they
    are NOT running — caller must spawn them.

    Returns (n, was_just_tuned).
    """
    if step != "generate":
        return 1, False

    gpu_model, cores = _probe_remote_capacity(base_ssh)
    cpu_floor = (
        max(1, int((cores or 96) * INSTANCE_HEADROOM // CPU_PER_INSTANCE))
        if cores else None
    )

    # Key the cache by (GPU model, board size): per-query GPU cost differs by
    # board size, so a 9x9 ramp shouldn't dictate 13x13's N. The size is the
    # first forwarded step arg (cmd_run validated it for generate).
    size = extra.split()[0] if extra else None
    cache_key = f"{gpu_model}|{size}" if gpu_model else None
    desc = f"'{gpu_model}' size {size}"

    state = load_state()
    cache = state.get("_gpu_cache", {})
    cached = cache.get(cache_key) if cache_key else None
    if cached:
        n = cached
        if cpu_floor is not None and cpu_floor < n:
            print(f"  cached N={cached} for {desc} but CPU caps at "
                  f"{cpu_floor}; using {cpu_floor}")
            n = cpu_floor
        else:
            print(f"  using cached N={n} for {desc}")
        return n, False

    print(f"  no cached N for {desc}; running ramp tune...")
    n = tune_parallelism(base_ssh, module, extra)
    if cpu_floor is not None and cpu_floor < n:
        print(f"  capping {n} → {cpu_floor} due to CPU floor")
        n = cpu_floor

    if cache_key:
        cache[cache_key] = n
        state = load_state()  # re-read in case other writes happened
        state["_gpu_cache"] = cache
        save_state(state)
        print(f"  cached N={n} for {desc}")
    return n, True


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

    # System packages: python, rsync, tmux, curl for model downloads.
    # The CUDA katago build links against libcublas.so.13 (CUDA 13) and
    # libzip.so.4 — neither is present in the vast.ai CUDA-12.8 base image,
    # so install them explicitly. libcudnn.so.9 already ships in the image.
    # --allow-change-held-packages: the image holds the CUDA 12.8 metapackage
    # to prevent accidental upgrades; libcublas-13-0 doesn't conflict with the
    # 12.8 install (they coexist under /usr/local/cuda-{12.8,13.0}/) but apt
    # still wants to nudge the held packages, so we whitelist that.
    apt_cmd = ("bash -l -c 'apt-get update && apt-get install -y "
               "--allow-change-held-packages "
               "python3-pip rsync tmux curl libcublas-13-0 libzip4t64'")
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

    if not upload_code(host, base_rsync, include_setup_only=True):
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
    # bots/katago_cuda is built against CUDA 13, so the host's driver must
    # support it (cuda_max_good>=13.0). Forward compat is unavailable on
    # consumer GPUs. reliability2>0.99 avoids hosts that drop offline mid-run
    # (full pipeline takes hours and a reschedule means starting over).
    base_filter = ("rented=False rentable=True "
                   "cuda_max_good>=13.0 reliability>0.99")
    print("Searching for RTX 5090 offers...")
    offers = run_vast_cmd(["search", "offers",
                           f"gpu_name=RTX_5090 {base_filter}"])
    if not offers:
        print("No RTX 5090 offers; falling back to RTX 4090.")
        offers = run_vast_cmd(["search", "offers",
                               f"gpu_name=RTX_4090 {base_filter}"])
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
        "--template_hash", "f779495b2d83b2c5b56e3c03f233f0ad",
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
        base_ssh + ["pgrep -f '[p]ython3.* -u -m' >/dev/null 2>&1 "
                    "&& echo RUNNING || echo IDLE"],
        capture_output=True, text=True, timeout=15,
    )
    return "RUNNING" in (res.stdout or "")


def cmd_run(args):
    """Start a pipeline step in tmux on an instance.

    Usage: run [<id>] <step> [<step args...>]
    If <id> is omitted and only one instance exists, it is used.
    tune/generate/benchmark require a board size as their first step arg
    (e.g. `run generate 13`), forwarded to the remote module.
    """
    if not args:
        print(f"Usage: run [<id>] <step> [args...]   step ∈ {{{', '.join(PIPELINE_STEPS)}}}")
        return
    # First arg is a step name → auto-select instance. Otherwise it's an ID.
    if args[0] in PIPELINE_STEPS:
        target = None
        step, *step_args = args
    else:
        if len(args) < 2:
            print(f"Usage: run [<id>] <step> [args...]   step ∈ {{{', '.join(PIPELINE_STEPS)}}}")
            return
        target, step, *step_args = args
    if step not in PIPELINE_STEPS:
        print(f"Unknown step '{step}'. Available: {', '.join(PIPELINE_STEPS)}")
        return

    # tune/generate/benchmark take a mandatory board size as their first arg.
    # Validate locally so a missing/invalid size fails here with a clear message
    # rather than crashing the remote module's argparse after provisioning.
    if step in SIZE_REQUIRED_STEPS:
        sizes = ", ".join(str(s) for s in ALL_BOARD_SIZES)
        if not (step_args and step_args[0].isdigit()
                and int(step_args[0]) in ALL_BOARD_SIZES):
            print(f"Step '{step}' requires a board size as its first argument "
                  f"(one of {sizes}). e.g. run {step} {ALL_BOARD_SIZES[0]}")
            return

    inst = find_instance(target)
    if not inst:
        return
    inst_id = inst["id"]
    host, port = get_ssh_details(inst)
    if not host or not port:
        print(f"Instance {inst_id} not reachable.")
        return
    base_ssh, base_rsync = get_ssh_cmds(host, port)

    if _is_step_running(base_ssh):
        print(f"Instance {inst_id} already has a pipeline step running. "
              f"Run 'stop' first.")
        return

    input_dirs = STEP_INPUT_DIRS.get(step, [])
    if input_dirs:
        print(f"Uploading input dirs for '{step}': {', '.join(input_dirs)}")
        if not upload_input_dirs(host, base_rsync, input_dirs):
            return

    module = PIPELINE_STEPS[step]
    extra = " ".join(step_args)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    parallel, was_tuned = determine_parallelism(base_ssh, step, module, extra)

    if was_tuned:
        # tune_parallelism left `parallel` instances already running.
        print(f"Tune completed; {parallel} instance(s) running.")
    else:
        # Cached or single-instance path: spawn fresh.
        print(f"Starting '{step}' (×{parallel}) on instance {inst_id}...")
        for i in range(parallel):
            if not _spawn_pipeline_session(base_ssh, i, module, extra):
                print(f"  failed to start tmux session pipeline_{i}.")
                return

    # Mark remote state regardless of which path produced the running set.
    subprocess.run(
        base_ssh + [f"echo '{step}' > {REMOTE_DIR}/running_step && "
                    f"date -Iseconds > {REMOTE_DIR}/running_started"],
        capture_output=True,
    )
    register_instance(inst_id, step=step, started=started)
    suffix = f" × {parallel}" if parallel > 1 else ""
    sessions = (f"{TMUX_SESSION_PREFIX}_0..{parallel - 1}"
                if parallel > 1 else f"{TMUX_SESSION_PREFIX}_0")
    print(f"Instance {inst_id} running '{step}'{suffix} (tmux '{sessions}').")
    print(f"  follow with: stream {inst_id}  (tails {LOG_FILENAME_PREFIX}_0.log)")


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
    # the KataGo subprocess. Then wait, then force-kill stragglers. Iterates
    # over all tmux sessions matching the pipeline prefix so multi-instance
    # generate runs are all torn down.
    stop_cmd = (
        f"for s in $(tmux ls 2>/dev/null | grep '^{TMUX_SESSION_PREFIX}_' "
        f"| cut -d: -f1); do tmux kill-session -t \"$s\" 2>/dev/null; done; "
        f"pkill -f '[p]ython3.* -u -m' 2>/dev/null; "
        f"for i in $(seq 1 30); do "
        f"  pgrep -f '[p]ython3.* -u -m' >/dev/null 2>&1 || break; "
        f"  sleep 1; "
        f"done; "
        f"pkill -9 -f '[p]ython3.* -u -m' 2>/dev/null; "
        f"pkill -9 -f '[k]atago analysis' 2>/dev/null; "
        f"sleep 1; "
        f"if pgrep -f '[p]ython3.* -u -m' >/dev/null 2>&1; then "
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


def cmd_sync(target, sgf_only=False):
    """rsync tsumegos/ and attempts/ from the remote into the local repo.

    With sgf_only=True, only tsumegos/ is pulled (attempts/ can be large and
    isn't needed when you just want fresh SGFs locally).
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

    dirs = ["tsumegos"] if sgf_only else REMOTE_OUTPUT_DIRS
    suffix = " (sgf only)" if sgf_only else ""
    print(f"Syncing outputs from instance {inst_id}{suffix}...")
    for d in dirs:
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


def cmd_sync_cli(args):
    """Parse `[<id>] [--sgf]` and call cmd_sync."""
    args = list(args)
    sgf_only = False
    if "--sgf" in args:
        sgf_only = True
        args.remove("--sgf")
    target = args[0] if args else None
    cmd_sync(target, sgf_only=sgf_only)


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
    log_path = f"{REMOTE_DIR}/{LOG_FILENAME_PREFIX}_0.log"

    step = get_instance_step(inst_id) or "?"
    print(f"Streaming {log_path} for step '{step}' (Ctrl-C to stop)...")
    print(f"  (tails the first instance only — for multi-instance generate, "
          f"others log to {LOG_FILENAME_PREFIX}_1.log etc.)")

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
        f"pgrep -f '[p]ython3.* -u -m' >/dev/null 2>&1 && echo running || echo idle; "
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


def cmd_monitor(args):
    """Every `interval` seconds, sync outputs from each instance with a step
    registered in local state. Default 1h, override with `monitor <secs>`.

    Generate runs for hours; if the host disappears mid-run we lose
    everything not yet synced. Periodic sync keeps the local copy current
    so a destroy or preemption costs at most one cycle.

    Pass `--sgf` to skip attempts/ on each cycle.
    """
    args = list(args)
    sgf_only = False
    if "--sgf" in args:
        sgf_only = True
        args.remove("--sgf")
    interval = int(args[0]) if args else 3600
    suffix = " (sgf only)" if sgf_only else ""
    print(f"Monitor started (interval {interval}s{suffix}, Ctrl-C to stop).")
    try:
        while True:
            cycle_start = time.time()
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] sync cycle")

            instances = get_all_instances()
            state = load_state()
            live_ids = {str(i["id"]) for i in instances}
            active_ids = [k for k, v in state.items()
                          if not k.startswith("_")
                          and v.get("step")
                          and k in live_ids]

            if not active_ids:
                print("  no instances with an active step.")
            for inst_id in active_ids:
                try:
                    cmd_sync(inst_id, sgf_only=sgf_only)
                except Exception as e:
                    print(f"  sync error for {inst_id}: {e}")

            elapsed = time.time() - cycle_start
            sleep_for = max(0, interval - elapsed)
            print(f"Cycle done in {int(elapsed)}s. Sleeping {int(sleep_for)}s.")
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def _optional_id(fn):
    """Dispatch with args[0] as the ID, or None if no ID given. find_instance
    auto-selects when there's only one instance."""
    def handler(args):
        fn(args[0] if args else None)
    return handler


COMMANDS = {
    "add": {
        "usage": "add",
        "help": ["Create a new GPU instance and run setup"],
        "handler": lambda args: cmd_add(),
    },
    "setup": {
        "usage": "setup [<id>]",
        "help": ["Re-run setup on an existing instance"],
        "handler": _optional_id(cmd_setup),
    },
    "run": {
        "usage": "run [<id>] <step> [args...]",
        "help": [
            f"Start a pipeline step in tmux. step ∈ {{{', '.join(PIPELINE_STEPS)}}}",
            "Extra args are forwarded to python -m pipeline.<step>",
            "tune/generate/benchmark need a board size first, e.g. run generate 13",
        ],
        "handler": cmd_run,
    },
    "stop": {
        "usage": "stop [<id>]",
        "help": ["Kill the running pipeline step (instance stays alive)"],
        "handler": _optional_id(cmd_stop),
    },
    "shutdown": {
        "usage": "shutdown [<id>]",
        "help": ["Destroy an instance"],
        "handler": _optional_id(cmd_shutdown),
    },
    "sync": {
        "usage": "sync [<id>] [--sgf]",
        "help": [
            "Pull tsumegos/ and attempts/ from remote into the local repo",
            "--sgf: only pull tsumegos/ (skip attempts/)",
        ],
        "handler": lambda args: cmd_sync_cli(args),
    },
    "stream": {
        "usage": "stream [<id>]",
        "help": ["Continuously tail the pipeline log on an instance"],
        "handler": _optional_id(cmd_stream),
    },
    "update": {
        "usage": "update [<id>]",
        "help": ["Re-upload pipeline/utils/cfg/katago code (does NOT restart)"],
        "handler": _optional_id(cmd_update),
    },
    "list": {
        "usage": "list",
        "help": ["Show status of all instances"],
        "handler": cmd_list,
    },
    "monitor": {
        "usage": "monitor [interval_secs] [--sgf]",
        "help": [
            "Every interval (default 3600s): sync outputs from each instance",
            "with an active step. Runs until Ctrl-C.",
            "--sgf: only pull tsumegos/ (skip attempts/) each cycle",
        ],
        "handler": cmd_monitor,
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
