"""Orchestrate Vast.ai instances for running onlymove pipeline steps remotely.

Two pipeline steps benefit from a beefier GPU than what's available locally:
  * pipeline.generate_tsumegos  — long-running tsumego search
  * pipeline.benchmark_bots     — bot performance evaluation
The rest (train_mirt, package_for_pwa, fetch_database) stays local.
This wrapper:
  * provisions a Vast.ai NVIDIA-GPU instance,
  * uploads pipeline/utils code, the KataGo binary, and the analysis cfg,
  * fetches KataGo network weights directly on the remote,
  * runs the chosen step in a tmux session,
  * lets you tail the log, sync the produced data back, kill the run, destroy.

`ablate` is the one command that runs no KataGo at all: it puts a refit-tier
MIRT ablation panel (ablations.run_panel) on a rented card and pulls the arm
records back as they land.

Run without args for the command list.
"""

import json
import glob
import os
import shlex
import subprocess
import sys
import tempfile
import time

from utils.constants import ALL_BOARD_SIZES

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
    "losing_lines": "auxiliary.add_losing_lines",
    "calibrate19": "auxiliary.calibrate_ranks_19x19",
    "calibrate_handicap": "auxiliary.calibrate_handicap_ranks",
}

# Steps that take a mandatory positional board size as their first arg — it's
# forwarded verbatim to the remote `python -m pipeline.<step> <size>`, which
# selects tsumegos/<size>/, attempts/<size>/, and the engine board dimensions.
# The calibrate* steps are fixed-board auxiliary diagnostics and take no size.
SIZE_REQUIRED_STEPS = {"tune", "generate", "benchmark", "losing_lines"}

# Files uploaded from the local repo. We ship the CUDA katago build to the
# remote (vast.ai NVIDIA GPUs); the local repo's bots/katago is OpenCL and is
# not uploaded. utils/katago_controller.py reads KATAGO_PATH from the
# environment, so we set that to bots/katago_cuda12 when launching the step.
# THE CUDA VERSION IS IN THE NAME because the binary is not portable across
# them: it links libcublas.so.<major>, so a host whose runtime is a different
# major cannot load it at all, and a host with a PARTIAL install of the right
# major fails later and less obviously (cuBLASLt dlopens libnvJitLink of the
# same major at init, so a missing one surfaces as "Cannot load symbol
# cublasLtCreate" with nothing missing in ldd).
LOCAL_UPLOAD_FILES = [
    "bots/katago_cuda12",
]
# bots/analysis.cfg is special: it's hardware-tuned (numAnalysisThreads,
# nnMaxBatchSize) by the remote tune step. Uploading from local would clobber
# those values. So it's only pushed during initial `setup`; `update` skips it.
SETUP_ONLY_UPLOAD_FILES = [
    "bots/analysis.cfg",
]
REMOTE_KATAGO_PATH = "bots/katago_cuda12"
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

# Freshly generated SGFs land HERE, not in tsumegos/<size>/, so that a generator
# running on a schedule cannot change the corpus underneath work in progress.
# Every derived artifact — activations, bot results, ownership, the rated set —
# is keyed to the corpus, so a directory that grows every 30 minutes means each
# of them is stale the moment it is built, and a fit halts on rows that appeared
# after it started. Staging makes adopting new problems an explicit `mv`.
#
# The corpus scanners deliberately do NOT look here (utils.rating.features
# corpus_tids_by_size and residual_axes.corpus_tid_size read tsumegos/<size>/
# only), which is the whole point. The one scanner that MUST look here is
# generate_tsumegos.tsumego_id_taken: a staged id is taken, and reissuing it
# would hand a new problem another problem's history.
TSUMEGO_STAGING = os.path.join("tsumegos", "new")

# Per-step input dirs pushed TO the remote before launching a step. These are
# state dirs the step reads to resume incrementally. For benchmark, attempts/
# holds prior bot results — uploading them lets test_bot skip already-processed
# tsumegos and append only new entries (it already merges existing+new on write).
# For calibrate_handicap, the rank fit and pair-selection priorities are
# seeded from the existing SGF corpus; without these, the remote starts cold
# and wastes games re-learning the kyu / dan gaps.
# losing_lines is the one step whose input dir IS the corpus it works on. Run it
# with `--out-dir tsumegos/fixed/<size>` so it writes beside its input instead of
# over it: the uploaded tsumegos/<size>/ then never changes, which is what makes
# it safe both to re-upload on a restart and to pull the finished files down
# repeatedly while the run is still going.
STEP_INPUT_DIRS = {
    "benchmark": ["attempts"],
    "losing_lines": ["tsumegos"],
    "calibrate_handicap": ["calibration_handicap_sgf"],
}

# Minimal pip deps for the remote steps. The full requirements.txt brings
# torch/sklearn for rank+package, which we never run remotely. scipy is
# needed only by calibrate_handicap (logistic MLE refit) but is cheap to
# install, so we add it unconditionally rather than branching by step.
REMOTE_PIP_PACKAGES = ["sgfmill>=1.1.1", "numpy>=1.21.0", "scipy>=1.10.0"]

# --- the refit-tier ablation panel (`ablate`) ------------------------------
#
# One arm of a panel IS pipeline.train_mirt.run() with a knob turned
# (ablations.ablation_mirt), so the remote needs the fit's stack on top of the
# minimal set above: torch for the trunk, pandas for the attempt frame, sklearn
# for the joint PCA. Pinned to requirements.txt so the arms are fitted on the
# stack the shipped fit is. scipy is already there and is reached at import time
# — utils.rating.model_constants imports utils.rating.fit_rank_curves.
#
# The image carries a CUDA runtime but no torch; the PyPI wheel brings its own
# CUDA libraries, so the host only has to satisfy the DRIVER floor cmd_add's
# offer query already imposes.
ABLATE_PIP_PACKAGES = ["torch~=2.7.1", "pandas~=2.3.1", "scikit-learn~=1.7.1"]

# THE EXACT INPUT SET, traced through what that call reads rather than guessed:
#   db/d1.sqlite                    the attempt log (features.load_first_time_attempts)
#                                   and the arm record's db_max_id pin
#   attempts/<size>/*.txt           the bot pass-rate block — the one wide block
#                                   the PRODUCTION_BLOCKS layout carries
#   activations/_signal_cache/      the correct-cell vectors the joint PCA is
#     correct_{ids,vecs}_<size>     fitted over, and their id list
#   activations/_signal_cache/      move evals, game phase and ownership swing:
#     move_evals_<size>, game_phase   NOT trunk inputs in the shipped layout, but
#     _<size>, ownership_swing        the comparison arm of a block-layout panel
#                                   is exactly the fit that reads them
#   axes/labels.json                the shipping decision K_AXES must match
#   axes/motif_proxies.npz          the directions the axes are taught onto
#   tsumegos/retired_ids.txt        the id-uniqueness check features.py runs
#                                   before it will build a problem set at all
#
# THE PER-PROBLEM STORES ARE NOT SHIPPED. activations/<size>/ is already rolled
# up into the signal cache. Of tsumegos/<size>/ this layout reads only the
# LISTING — corpus_tids_by_size names the problem set off the .sgf filenames and
# nothing here opens one — so what travels is a manifest of those names,
# materialised remotely as empty stubs by _ablate_stub_corpus.
ABLATE_INPUT_FILES = [
    "db/d1.sqlite",
    "axes/labels.json",
    "axes/motif_proxies.npz",
    "tsumegos/retired_ids.txt",
] + [f"activations/_signal_cache/correct_{kind}_{size}.{ext}"
     for size in ALL_BOARD_SIZES
     for kind, ext in (("ids", "json"), ("vecs", "npy"))] + [
    # The ANSWER-DERIVED blocks, which the shipped layout does not carry and a
    # panel sweeping the block layout does. They travel because the failure
    # without them is remote and late — the arm loads its features, aborts, and
    # the panel comes home with a hole in exactly the comparison it was run for.
    "activations/_signal_cache/move_evals_9.npz",
    "activations/_signal_cache/move_evals_13.npz",
    "activations/_signal_cache/game_phase_9.json",
    "activations/_signal_cache/game_phase_13.json",
    "activations/_signal_cache/ownership_swing.csv",
]
ABLATE_INPUT_DIRS = [f"attempts/{size}" for size in ALL_BOARD_SIZES]

# ablations/ is pushed by this step alone rather than joining LOCAL_UPLOAD_DIRS:
# no other step imports it, and `update` must not start pushing it either — see
# the dev_runs/ exclusion in _ablate_upload.
ABLATE_CODE_DIRS = ["ablations"]
ABLATE_RESULTS = os.path.join("ablations", "dev_runs")
ABLATE_PANELS = os.path.join("ablations", "panels")
ABLATE_LOG_PREFIX = "ablate"
ABLATE_QUEUE = os.path.join("ablations", "queue.txt")
ABLATE_CONCURRENCY = "auto"   # run_panel ramps in place; an integer pins the level
ABLATE_POLL_SECS = 300

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
# Offer filter: CPU has to be sized to the CARD, not to an absolute floor, since
# KataGo's search is CPU-fed and a bigger card is ramped to more instances. The
# value is derived from the two constants above against the tuned N in
# state["_gpu_cache"] — cores_needed = N * CPU_PER_INSTANCE / INSTANCE_HEADROOM,
# read back as TFLOPS/core. That derivation is an UPPER bound on this ratio: a
# cached N that was itself CPU-capped reconstructs the host's cores rather than
# the GPU's appetite, so erring to a SMALLER value (more CPU) is the safe way to
# be wrong. This value sits at the loose end of what the cache implies.
MAX_TFLOPS_PER_CORE = 3.0
# Link floor, Mbps, required in both directions (see cmd_add).
MIN_INET_MBPS = 400
# PCIe floor, also relative to the card: KataGo moves input planes up and policy
# /value back every batch, so the host-to-device link is walked at a rate set by
# the card's throughput, not by anything absolute. Bracketed by two points, both
# for a 5090 (~108 TFLOPS): PCIe 3.0 x16 measures ~13.6 GB/s = 7.9 and is known
# INSUFFICIENT, while 4.0 x16 measures ~25 GB/s = 4.3 and is the nearest
# known-good. The value sits inside that bracket, nearer the known-good end.
#
# Test `pcie_bw` (MEASURED GB/s), never `pci_gen`: the pool carries gen-5 hosts
# measuring 13.4 and 18.3 GB/s, so a host can advertise a generation whose
# nominal width it does not deliver, and the 13.4 one is worse than the gen-3
# case this is here to exclude. `gpu_lanes` needs no separate test either — a
# narrow link shows up in the measurement already.
MAX_TFLOPS_PER_PCIE_GBPS = 4.5

# The rental pair, stated here rather than inside cmd_add so `ablate --dry-run`
# can PRINT what renting would issue without a second copy of it drifting from
# what `add` actually sends. The reasoning behind each filter is at cmd_add.
#
# --limit is mandatory, not a tuning knob: the API returns 64 offers by default,
# ranked by ITS score and not by ours, so without this cmd_add's sort picks the
# best of an arbitrary truncation of the pool.
OFFER_QUERY = ("rented=False rentable=True num_gpus=1 "
               "cuda_max_good>=12.8 reliability>0.99 "
               "cpu_cores_effective>=8 "
               f"inet_down>={MIN_INET_MBPS} inet_up>={MIN_INET_MBPS}")
OFFER_LIMIT = "1000"
# A fully explicit registry tag, NOT a Vast template. Vast's own CUDA template
# pins vastai/base-image at @vastai-automatic-tag, which the server resolves per
# host from (compute capability, driver CUDA version) — a Vast-side lookup table
# this repo does not control, and when it stopped resolving ("No compatible image
# tag found" for every architecture and driver we could offer it) it refused every
# create. An explicit tag consults no such table. Three properties make this the
# right one, and any replacement has to keep all three:
#  * a COMPLETE CUDA 12.8 runtime INCLUDING cuDNN — bots/katago_cuda12 links both,
#    and setup_instance deliberately installs neither (see the cuBLAS/nvJitLink
#    pairing failure documented there).
#  * ubuntu24.04 — setup_instance's apt line names libzip4t64, which is the 24.04
#    package; 22.04 calls it libzip4 and the install fails.
#  * runtime, not devel — nothing is compiled on the box.
#
# --direct asks for a directly reachable SSH port on the host's public IP instead
# of a hop through Vast's SSH proxy, which is what get_ssh_details prefers.
# Without it the API reports no 22/tcp binding at all and the direct candidate
# does not exist, so the preference is unreachable. It costs nothing in offer
# selection — the filters above have never yet left an offer without direct ports
# in the pool — and the proxy candidates stay in get_ssh_details as the fallback
# if a host declines to map one.
# Disk sized to the biggest footprint any box type carries — the image, the
# torch stack or the KataGo models, the input caches, and output that is
# rsynced off rather than accumulated — and Vast bills storage per GB-hour, so
# the request stays close to what is actually used.
CREATE_ARGS = ["--image", "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04",
               "--disk", "32", "--ssh", "--direct"]


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


# get_ssh_details probes, and remembers what answered.
_SSH_PORT_CACHE = {}


def _ssh_port_answers(host, port):
    try:
        return subprocess.run(
            ["ssh", "-q", "-o", "LogLevel=ERROR", "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=8",
             "-p", str(port), f"root@{host}", "true"],
            capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def get_ssh_details(inst):
    """(host, port) for an instance, choosing the route that actually answers.

    The DIRECT route is tried first: the instance's own public_ipaddr on
    whatever it maps to 22/tcp — NOT direct_port_start, which on this image is
    container port 1111. It skips Vast's proxy entirely, so it is both faster
    and the only route some instances accept.

    The two PROXY candidates follow, because a direct route is not always
    offered (the ports map can be missing or unpopulated on a booting box) and
    the proxy port is NOT a fixed offset from the API's ssh_port: some
    instances answer on ssh_port + 1 and others on ssh_port itself, and
    assuming either one leaves the other permanently "SSH not ready". +1 comes
    first because it is the older and more common case.

    Whatever answers is cached, so this costs one probe per instance.
    """
    host = inst.get("ssh_proxy_host") or inst.get("ssh_host")
    port = inst.get("ssh_proxy_port") or inst.get("ssh_port")
    if not host or not port:
        return host, port
    key = (host, int(port))
    if key in _SSH_PORT_CACHE:
        return _SSH_PORT_CACHE[key]
    candidates = []
    direct_host = inst.get("public_ipaddr")
    for binding in (inst.get("ports") or {}).get("22/tcp") or []:
        if binding.get("HostPort") and direct_host:
            candidates.append((direct_host, int(binding["HostPort"])))
    candidates += [(host, int(port) + 1), (host, int(port))]
    for cand_host, cand_port in candidates:
        if _ssh_port_answers(cand_host, cand_port):
            _SSH_PORT_CACHE[key] = (cand_host, cand_port)
            return cand_host, cand_port
    # Nothing answered yet — a box that is still booting. Hand back the first
    # candidate so wait_for_ssh retries on the route we would prefer anyway.
    return candidates[0]


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


# A box that cannot be reached this long after it was reserved is not booting,
# it is broken — wrong key injection, a dead route, a defective host — and a
# retry loop that keeps knocking anyway both wastes billed hours and can trip
# the host's connection rate-limiter, manufacturing its own lockout. So the
# wait carries a hard deadline and a growing backoff, and expiry raises with
# the diagnosis steps instead of spinning.
SSH_WAIT_DEADLINE_SECS = 600


def wait_for_ssh(host, port):
    base_ssh, _ = get_ssh_cmds(host, port)
    deadline = time.time() + SSH_WAIT_DEADLINE_SECS
    delay = 5
    while True:
        try:
            res = subprocess.run(
                base_ssh + [f"mkdir -p {REMOTE_DIR} {REMOTE_DIR}/models "
                            f"{REMOTE_DIR}/bots && touch ~/.hushlogin"],
                capture_output=True, text=True, timeout=15,
            )
            if res.returncode == 0:
                return
            print(f"  SSH not ready ({host}:{port}), retrying in {delay}s...")
        except subprocess.TimeoutExpired:
            print(f"  SSH timeout ({host}:{port}), retrying in {delay}s...")
        if time.time() > deadline:
            raise RuntimeError(
                f"SSH to {host}:{port} still failing after "
                f"{SSH_WAIT_DEADLINE_SECS}s — diagnose or drop the box, do not "
                f"wait: check `show instances` actual_status; probe auth with a "
                f"verbose ssh (key injection lags or fails on some hosts — "
                f"`vastai attach ssh` + one reboot can fix it); a handshake "
                f"drop (kex_exchange_identification) means the host is "
                f"rate-limiting this machine after too many attempts, so STOP "
                f"knocking and wait quietly before one retry. Otherwise "
                f"`vastai destroy instance` and rent another.")
        time.sleep(delay)
        delay = min(delay * 2, 60)


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

def _run_or_show(cmd, dry=False, **kw):
    """Execute `cmd`, or under `dry` print it and report nothing ran.

    Returns the CompletedProcess, or None when the command was only printed —
    callers read None as "assume it worked", which is what lets `ablate
    --dry-run` walk the SAME path the real run takes instead of narrating a
    second version of it.
    """
    if dry:
        print("  $ " + " ".join(shlex.quote(str(c)) for c in cmd))
        return None
    return subprocess.run(cmd, **kw)


def upload_code(host, base_rsync, include_setup_only=False, dry=False):
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
        res = _run_or_show(
            base_rsync + excludes + [f"{d}/", f"root@{host}:{REMOTE_DIR}/{d}/"],
            dry,
        )
        if res is not None and res.returncode != 0:
            print(f"  {d}/ upload failed.")
            return False
    for f in files:
        if not os.path.exists(f):
            print(f"  Warning: {f} not found, skipping")
            continue
        # Preserve subdir layout (e.g. bots/katago → /workspace/onlymove/bots/katago)
        remote_subdir = os.path.dirname(f)
        if remote_subdir:
            res = _run_or_show(
                base_rsync + [f, f"root@{host}:{REMOTE_DIR}/{remote_subdir}/"],
                dry,
            )
        else:
            res = _run_or_show(base_rsync + [f, f"root@{host}:{REMOTE_DIR}/"], dry)
        if res is not None and res.returncode != 0:
            print(f"  {f} upload failed.")
            return False
    return True


def upload_input_dirs(host, base_ssh, base_rsync, dirs, size=None):
    """Push state dirs (e.g. attempts/) to the remote so a step can resume.

    For a size-dependent step (benchmark), pass `size` to upload ONLY that
    board's subdir (attempts/<size>/). That alone skips the other board size
    AND the top-level D1 dumps — database.sql / database*.sqlite live in
    attempts/ directly, not attempts/<size>/ — so no per-file excludes are
    needed. rsync won't create missing parent dirs, so mkdir -p the target
    first.
    """
    # `new/` is local staging for SGFs pulled off this very remote — it already
    # has them in tsumegos/<size>/, so mirroring it back is pure transfer cost.
    excludes = ["--exclude", "__pycache__/", "--exclude", "*.pyc",
                "--exclude", "new/"]
    for d in dirs:
        local = os.path.join(d, str(size)) if size is not None else d
        if not os.path.isdir(local):
            print(f"  {local}/ not present locally, skipping upload")
            continue
        remote = f"{REMOTE_DIR}/{local}"
        subprocess.run(base_ssh + [f"mkdir -p {remote}"], capture_output=True)
        try:
            res = subprocess.run(
                base_rsync + excludes + [f"{local}/", f"root@{host}:{remote}/"],
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            print(f"  {local}/ upload timed out (rsync >1800s).")
            return False
        if res.returncode != 0:
            print(f"  {local}/ upload failed (rsync rc={res.returncode}).")
            return False
        print(f"  {local}/ uploaded.")
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

    # System packages: python, rsync, tmux, curl for model downloads, plus
    # libzip.so.4, which the katago build needs and the image lacks. cuBLAS and
    # cuDNN are NOT installed: bots/katago_cuda12 links the CUDA 12.8 runtime
    # that this image already ships complete.
    #
    # IT USED TO INSTALL libcublas-13-0 for a CUDA 13 build, and that broke when
    # NVIDIA's repo moved libcublas-13-0 to 13.1.x while its newest
    # libnvjitlink-13-0 stayed at 13.0.88. cuBLASLt dlopens nvJitLink of its own
    # minor at init, so the pair no longer satisfied each other and every query
    # died with "Cannot load symbol cublasLtCreate" — with nothing missing in
    # ldd, since the load is at runtime. Pinning cuBLAS back to 13.0.2.14 did not
    # help either. Building against the image's own CUDA removes the whole
    # dependency rather than chasing a version pairing that NVIDIA controls.
    apt_cmd = ("bash -l -c 'apt-get update && apt-get install -y "
               "--allow-change-held-packages "
               "python3-pip rsync tmux curl libzip4t64'")
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
    # Each spec is double-quoted inside the single-quoted `bash -l -c` string:
    # the inner shell parses that string itself, and there a bare `>=1.21.0`
    # is a REDIRECTION, not a version bound. Unquoted, pip was handed plain
    # `numpy` (so the bound was silently dropped) with its whole output
    # redirected into files named `=1.21.0` — a setup that looks hung because
    # nothing prints, and installs whatever version it likes.
    specs = " ".join(f'"{pkg}"' for pkg in REMOTE_PIP_PACKAGES)
    pip_cmd = (f"bash -l -c 'python3 -m pip install --break-system-packages "
               f"{specs}'")
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
    # No GPU model is named: the steps we remote are throughput-bound on a
    # single card, so the offer is chosen by compute per dollar and whatever
    # model wins that is the right one. Hard requirements:
    #  * num_gpus=1 — every remoted step drives one card (parallelism comes
    #    from multiple KataGo processes on it, see the auto-tuner above), so a
    #    multi-GPU offer is paid for and left idle.
    #  * cuda_max_good>=12.8 — bots/katago_cuda12 links against CUDA 12.8, and
    #    forward compat is unavailable on consumer GPUs. This is the DRIVER's
    #    maximum, which is necessary but NOT sufficient: it says nothing about
    #    which toolkit the image ships, and the binary needs a matching RUNTIME.
    #    Pairing this filter with the explicit CUDA-12.8 image below is what
    #    makes it sufficient. It stays a floor and not an equality: drivers are
    #    backward-compatible with older runtimes, so a 13.x host runs the 12.8
    #    image fine.
    #  * reliability>0.99 — a full run takes hours with no checkpointing, so a
    #    host that drops offline mid-run costs the whole run.
    #  * cpu_cores_effective>=8 — an absolute floor, enough to feed ONE generate
    #    instance (CPU_PER_INSTANCE). It is not the real CPU criterion, only the
    #    part expressible server-side; MAX_TFLOPS_PER_CORE below is the rest.
    #  * inet_down/inet_up>=400 — setup pushes the repo, the katago binary and
    #    the model weights before any work starts, and generate's output is
    #    rsynced back off the box, so a slow link is paid twice on every run and
    #    is not visible in TFLOPS/$ either. Required in BOTH directions: the two
    #    are independently reported and the up figure is the weaker one in the
    #    pool, which is the direction the results come back on.
    #
    # The query itself, and the create arguments below, are OFFER_QUERY /
    # CREATE_ARGS at the top of the file.
    print("Searching offers (single GPU, CUDA >= 12.8, reliability > 99%, "
          f">= 8 effective cores, >= {MIN_INET_MBPS} Mbps up/down)...")
    offers = run_vast_cmd(["search", "offers", OFFER_QUERY,
                           "--limit", OFFER_LIMIT])
    if not offers:
        print("No offers found.")
        return

    # Rank by TFLOPS per $/hr, computed here rather than read off the offer's
    # flops_per_dphtotal so the ranking doesn't depend on an undocumented
    # field. Offers without a positive price or a positive flops figure have
    # no defined ratio, so they're dropped rather than ranked.
    priced = [o for o in offers
              if o.get("dph_total", 0) > 0 and o.get("total_flops", 0) > 0]
    if not priced:
        print(f"No offers with a usable price/TFLOPS pair ({len(offers)} seen).")
        return

    # CPU sized to the card. Expressed here rather than in the query because the
    # search DSL compares a field to a CONSTANT and this is a ratio between two
    # fields. An offer with no core count is dropped rather than assumed fine:
    # the whole point is that TFLOPS/$ cannot see a starved CPU.
    fed = [o for o in priced
           if o.get("cpu_cores_effective", 0) > 0
           and o["total_flops"] / o["cpu_cores_effective"] <= MAX_TFLOPS_PER_CORE]
    if not fed:
        print(f"No offers with >= 1 core per {MAX_TFLOPS_PER_CORE} TFLOPS "
              f"({len(priced)} priced offers seen).")
        return
    print(f"  {len(fed)}/{len(priced)} offers carry enough CPU "
          f"(1 core per {MAX_TFLOPS_PER_CORE} TFLOPS or better)")

    # Same shape as the CPU rule, and local for the same reason: a ratio between
    # two fields is not expressible in the search DSL. A missing measurement is
    # dropped rather than trusted, since an unmeasured link is exactly the case
    # the TFLOPS/$ ranking cannot see.
    linked = [o for o in fed
              if o.get("pcie_bw", 0) > 0
              and o["total_flops"] / o["pcie_bw"] <= MAX_TFLOPS_PER_PCIE_GBPS]
    if not linked:
        print(f"No offers with >= 1 GB/s PCIe per {MAX_TFLOPS_PER_PCIE_GBPS} "
              f"TFLOPS ({len(fed)} CPU-adequate offers seen).")
        return
    print(f"  {len(linked)}/{len(fed)} of those carry enough PCIe "
          f"(1 GB/s per {MAX_TFLOPS_PER_PCIE_GBPS} TFLOPS or better)")
    priced = linked
    priced.sort(key=lambda o: o["total_flops"] / o["dph_total"], reverse=True)

    print(f"{len(priced)} candidate offers; top by TFLOPS/$:")
    for o in priced[:5]:
        print(f"  {o.get('gpu_name', '?'):<16} "
              f"{o['total_flops']:6.1f} TFLOPS  ${o['dph_total']:.3f}/hr  "
              f"{o['total_flops'] / o['dph_total']:6.1f} TFLOPS/$  "
              f"{o.get('gpu_ram', 0) / 1024:.0f} GB  "
              f"{o.get('cpu_cores_effective', 0):.0f} cores  "
              f"{o['total_flops'] / o['cpu_cores_effective']:.2f} TF/core  "
              f"gen{o.get('pci_gen', 0):.0f} {o['pcie_bw']:.1f} GB/s")

    best = priced[0]
    print(f"Creating instance ({best.get('gpu_name', '?')}, "
          f"${best['dph_total']:.3f}/hr, "
          f"{best['total_flops'] / best['dph_total']:.1f} TFLOPS/$)...")
    res = run_vast_cmd(["create", "instance", str(best["id"])] + CREATE_ARGS)
    if not isinstance(res, dict) or "new_contract" not in res:
        # run_vast_cmd returns None on empty CLI output; guard so we abort
        # cleanly instead of TypeError'ing on res["new_contract"].
        print("VastAI create returned no contract id; aborting.")
        sys.exit(1)
    instance_id = res["new_contract"]

    print(f"Waiting for instance {instance_id}...")
    # RESOLVE THE ROUTE ONLY ONCE THE CONTAINER IS RUNNING. While the host is
    # still pulling the image the API already reports an ssh host and port but
    # no port map, so get_ssh_details can only hand back a guess (the proxy at
    # ssh_port + 1) — and wait_for_ssh then knocks on that one guess until its
    # deadline, while the box comes up on a different route entirely. Gating on
    # actual_status costs nothing: a container that is not running cannot
    # answer on any route, and once it is, the port map is populated and the
    # direct route is probed first.
    while True:
        instances = run_vast_cmd(["show", "instances"])
        inst = next((i for i in instances if i["id"] == instance_id), None)
        if inst and inst.get("actual_status") == "running":
            host, port = get_ssh_details(inst)
            if host and port:
                break
        time.sleep(5)

    if not setup_instance_with_retry(inst):
        return None
    # The id, so a caller that rented the box can go on to use it. The CLI
    # handler discards it.
    return instance_id


def cmd_setup(target):
    """Re-run setup on an existing instance."""
    inst = find_instance(target)
    if not inst:
        return
    setup_instance_with_retry(inst)


def _is_step_running(base_ssh):
    try:
        res = subprocess.run(
            base_ssh + ["pgrep -f '[p]ython3.* -u -m' >/dev/null 2>&1 "
                        "&& echo RUNNING || echo IDLE"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        # ssh hung — can't confirm; treat as not-running rather than crashing
        # the caller with a raw traceback.
        return False
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
        # benchmark is size-dependent (reads tsumegos/<size>/ + attempts/<size>/),
        # so upload only that board's subdir — which also skips the local-only
        # DB dumps (database.sql/.sqlite) that sit in attempts/ directly.
        in_size = (step_args[0] if step in SIZE_REQUIRED_STEPS and step_args
                   else None)
        label = ", ".join(f"{d}/{in_size}" if in_size else d for d in input_dirs)
        print(f"Uploading input dirs for '{step}': {label}")
        if not upload_input_dirs(host, base_ssh, base_rsync, input_dirs,
                                 size=in_size):
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


def _report_staged():
    """Say what is waiting in staging, per size. Silence would make a scheduled
    sync indistinguishable from one that pulled nothing."""
    if not os.path.isdir(TSUMEGO_STAGING):
        return
    for size in sorted(os.listdir(TSUMEGO_STAGING)):
        d = os.path.join(TSUMEGO_STAGING, size)
        if not os.path.isdir(d):
            continue
        n = sum(1 for f in os.listdir(d) if f.endswith(".sgf"))
        if n:
            print(f"  STAGED: {n} SGFs in {d}/ — adopt with "
                  f"`mv {d}/*.sgf tsumegos/{size}/` when you want them in the corpus")


def cmd_sync(target, sgf_only=False, size=None):
    """rsync tsumegos/ and attempts/ from the remote into the local repo.

    With sgf_only=True, only tsumegos/ is pulled (attempts/ can be large and
    isn't needed when you just want fresh SGFs locally).

    `size` SCOPES THE TSUMEGO PULL TO ONE BOARD, and a periodic sync should
    normally set it. The remote still holds the pre-merge, pre-purge originals of
    any board it is not currently generating, and every one of those is a file
    the local corpus is missing on purpose — so an unscoped pull drags thousands
    of deliberately purged problems into staging. They land there rather than in
    the corpus, so nothing is silently restored, but there is no reason to
    transfer and then sift them.
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
        # Trailing slash on src + dst means "copy contents".
        # Don't pull database.sql back — it's a local-only D1 dump.
        excludes = ["--exclude", "database.sql"]
        src_sub = d
        if d == "tsumegos":
            # PULL FROM THE REMOTE'S STAGING DIR. generate_tsumegos writes into
            # tsumegos/new/<size>/, so the remote separates generated problems
            # from the uploaded corpus exactly as the local tree does and this
            # sync is a staging-to-staging mirror.
            #
            # --compare-dest still points at the local CORPUS so a problem that
            # has already been adopted is skipped rather than re-staged. That
            # matters after an adoption: the remote keeps its copy in staging,
            # and without the compare every later sync would hand it back.
            #
            # SGFs ONLY. The whole-staging form below mirrors a DIRECTORY, so
            # anything else the remote leaves there is copied verbatim -- which
            # is how a stale tsumegos/new/retired_ids.txt appeared locally, a
            # second copy of the id list at a path nothing reads, shadowing the
            # canonical one for any human who found it first.
            if size is not None:
                src_sub = f"{d}/new/{size}"
                dest = os.path.join(TSUMEGO_STAGING, str(size)) + os.sep
                compare = os.path.abspath(os.path.join(d, str(size)))
            else:
                src_sub = f"{d}/new"
                dest = TSUMEGO_STAGING + os.sep
                compare = os.path.abspath(d)
            excludes += ["--include", "*/", "--include", "*.sgf",
                         "--exclude", "*", "--compare-dest", compare]
        else:
            dest = f"{d}/"
        os.makedirs(dest, exist_ok=True)
        res = subprocess.run(
            base_rsync + excludes + [
                f"root@{host}:{REMOTE_DIR}/{src_sub}/",
                dest,
            ],
            timeout=600,
        )
        if res.returncode == 0:
            print(f"  {d}/ -> {dest} synced.")
        else:
            print(f"  {d}/ sync had errors (rsync rc={res.returncode}).")
    _report_staged()


def cmd_sync_cli(args):
    """Parse `[<id>] [--sgf] [--size N]` and call cmd_sync."""
    args = list(args)
    sgf_only = False
    if "--sgf" in args:
        sgf_only = True
        args.remove("--sgf")
    size = None
    if "--size" in args:
        i = args.index("--size")
        size = int(args[i + 1])
        del args[i:i + 2]
    target = args[0] if args else None
    cmd_sync(target, sgf_only=sgf_only, size=size)


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
# Refit-tier ablation panels
# ---------------------------------------------------------------------------

# Puts the corpus NAMES in place on the remote and nothing else. It refuses a
# directory holding a file with bytes in it: this deletes what the manifest does
# not list, and on a box that has generated or been sent a real corpus that would
# destroy SGFs. It prunes rather than only adding because the problem set IS the
# listing — a stub left behind after a local purge would enter the fit as a
# problem with no activation row and abort it.
_STUB_CORPUS_PY = """
import os, sys
d, m = sys.argv[1], sys.argv[2]
os.makedirs(d, exist_ok=True)
want = set(l.strip() for l in open(m) if l.strip())
have = os.listdir(d)
real = [f for f in have if os.path.getsize(os.path.join(d, f)) > 0]
if real:
    sys.exit("REFUSING: " + d + " holds " + str(len(real)) + " non-empty file(s)"
             " - that is a real corpus, not the name stubs this step writes")
for f in set(have) - want:
    os.remove(os.path.join(d, f))
for f in want - set(have):
    open(os.path.join(d, f), "w").close()
print("corpus names in place: " + d + " (" + str(len(want)) + ")")
"""


def _ablate_log(panel):
    return f"{REMOTE_DIR}/{ABLATE_LOG_PREFIX}_{panel}.log"


def _ablate_panel_arms(panel):
    """Arm names from the local manifest, or None if there is no such panel.

    Read locally and FIRST, so a mistyped panel name costs nothing — the check
    happens before any instance is found, let alone rented.
    """
    path = os.path.join(ABLATE_PANELS, f"{panel}.json")
    if not os.path.exists(path):
        print(f"No panel manifest at {path}.")
        avail = sorted(f[:-5] for f in os.listdir(ABLATE_PANELS)
                       if f.endswith(".json")) if os.path.isdir(ABLATE_PANELS) else []
        if avail:
            print(f"  Available: {', '.join(avail)}")
        return None
    try:
        return list(json.load(open(path))["arms"])
    except (ValueError, KeyError) as e:
        print(f"  {path} is not a usable manifest: {e}")
        return None


def _ablate_provision(base_ssh, dry):
    """git plus the fit's python stack, on top of what setup already installed.

    git is not decoration: ablations.ablation_mirt records the revision an arm
    was fitted at by shelling out to it, and the arm aborts if it cannot.
    """
    apt_cmd = "bash -l -c 'apt-get update && apt-get install -y git'"
    # Each spec double-quoted inside the single-quoted `bash -l -c` string, for
    # the reason spelled out in setup_instance: unquoted, `~=2.7.1` is parsed by
    # the inner shell and the bound silently disappears.
    specs = " ".join(f'"{pkg}"' for pkg in ABLATE_PIP_PACKAGES)
    pip_cmd = (f"bash -l -c 'python3 -m pip install --break-system-packages "
               f"{specs}'")
    for label, cmd in (("apt", apt_cmd), ("pip", pip_cmd)):
        for attempt in range(3):
            res = _run_or_show(base_ssh + [cmd], dry)
            if res is None or res.returncode == 0:
                break
            print(f"  {label} install failed (attempt {attempt + 1}/3), retrying...")
            time.sleep(5)
        else:
            print(f"ERROR: {label} install failed after 3 attempts.")
            return False
    return True


def _ablate_check_torch(base_ssh, dry):
    """Refuse to launch a panel onto a CPU. A fit that falls back to CPU does not
    fail, it just never finishes, and the box bills the whole time."""
    cmd = ("python3 -c 'import torch; "
           "print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'")
    res = _run_or_show(base_ssh + [cmd], dry, capture_output=True, text=True,
                       timeout=180)
    if res is None:
        return True
    out = (res.stdout or "").strip()
    print(f"  remote: {out or (res.stderr or '').strip()}")
    if res.returncode != 0 or "cuda True" not in out:
        print("ERROR: the remote has no CUDA-visible torch; not launching.")
        return False
    return True


def _ablate_git_stamp(base_ssh, dry):
    """Let `git rev-parse --short HEAD` answer inside the uploaded tree.

    The upload is a snapshot of the local tree at HEAD with no .git beside it, and
    an arm record pins the revision it was fitted at. A detached HEAD holding that
    sha is what tells git which revision the snapshot IS; core.abbrev is set to the
    local abbreviation length so the pin reads identically whichever machine ran
    the arm.
    """
    root = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        return subprocess.run(["git"] + list(args), cwd=root, capture_output=True,
                              text=True).stdout.strip()

    head, short = git("rev-parse", "HEAD"), git("rev-parse", "--short", "HEAD")
    if not head:
        print("  no local git revision; arms would not be able to pin one.")
        return False
    if git("status", "--porcelain"):
        print(f"  WARNING: the working tree is dirty. Arms will pin {short}, "
              f"which is not what is being uploaded.")
    cmd = (f"mkdir -p {REMOTE_DIR}/.git/objects {REMOTE_DIR}/.git/refs && "
           f"printf '%s\\n' {head} > {REMOTE_DIR}/.git/HEAD && "
           f"printf '[core]\\n\\tabbrev = %s\\n' {len(short)} "
           f"> {REMOTE_DIR}/.git/config")
    res = _run_or_show(base_ssh + [cmd], dry, capture_output=True)
    if res is not None and res.returncode != 0:
        print("  could not stamp the revision on the remote.")
        return False
    print(f"  arms will pin revision {short}")
    return True


def _ablate_stub_corpus(host, base_ssh, base_rsync, dry):
    """Ship the corpus LISTING and materialise it as empty files.

    features.corpus_tids_by_size is the fit's problem set and it is the .sgf
    filenames in tsumegos/<size>/; the PRODUCTION_BLOCKS layout opens none of
    them. So the names travel and the files are made on arrival, which keeps the
    two per-problem stores off the wire entirely.
    """
    with tempfile.TemporaryDirectory() as tmp:
        for size in ALL_BOARD_SIZES:
            local_corpus = os.path.join("tsumegos", str(size))
            if not os.path.isdir(local_corpus):
                print(f"  {local_corpus}/ not present locally; cannot list the corpus.")
                return False
            names = sorted(f for f in os.listdir(local_corpus) if f.endswith(".sgf"))
            if not names:
                print(f"  {local_corpus}/ holds no SGFs; refusing to ship an "
                      f"empty problem set.")
                return False
            manifest = os.path.join(tmp, f"corpus_files_{size}.txt")
            with open(manifest, "w") as fh:
                fh.write("\n".join(names) + "\n")
            remote_manifest = f"{REMOTE_DIR}/tsumegos/corpus_files_{size}.txt"
            print(f"  tsumegos/{size}/: {len(names)} names")
            res = _run_or_show(base_rsync + [manifest,
                                             f"root@{host}:{remote_manifest}"],
                               dry, timeout=600)
            if res is not None and res.returncode != 0:
                print(f"  corpus manifest upload failed for size {size}.")
                return False
            cmd = (f"python3 -c {shlex.quote(_STUB_CORPUS_PY)} "
                   f"{REMOTE_DIR}/tsumegos/{size} {remote_manifest}")
            res = _run_or_show(base_ssh + [cmd], dry, capture_output=True,
                               text=True, timeout=900)
            if res is not None:
                print("  " + ((res.stdout or "") + (res.stderr or "")).strip())
                if res.returncode != 0:
                    return False
    return True


def _ablate_upload(host, base_ssh, base_rsync, dry):
    """Code, then the exact input set, then the corpus listing. All idempotent, so
    a re-invocation after a killed run just re-runs it.

    dev_runs/ is excluded in this direction on purpose: the panel's arm records
    are the REMOTE's resume state, and pushing a local copy up would let an arm
    that ran somewhere else mask one this box still owes.
    """
    if not upload_code(host, base_rsync, dry=dry):
        return False
    excludes = ["--exclude", "__pycache__/", "--exclude", "*.pyc"]
    for d in ABLATE_CODE_DIRS:
        res = _run_or_show(
            base_rsync + excludes + ["--exclude", "dev_runs/",
                                     f"{d}/", f"root@{host}:{REMOTE_DIR}/{d}/"],
            dry)
        if res is not None and res.returncode != 0:
            print(f"  {d}/ upload failed.")
            return False

    missing = [p for p in ABLATE_INPUT_FILES + ABLATE_INPUT_DIRS
               if not os.path.exists(p)]
    if missing:
        print("  missing local inputs, nothing to fit against: "
              + ", ".join(missing))
        return False
    # ONE rsync with --relative, so every input keeps its path under the remote
    # root and rsync creates the parents itself. Additive, with no --delete: an
    # input the local tree has SHED (a retired bot's attempt file, say) survives
    # on a box that is being reused, and would put a column in this fit that the
    # local one does not have. Panel comparability is unaffected — every arm on
    # the box sees the same layout — but a cross-machine comparison is not one
    # this step promises anyway.
    print("  inputs: " + ", ".join(ABLATE_INPUT_FILES + ABLATE_INPUT_DIRS))
    res = _run_or_show(
        base_rsync + ["-R"] + ABLATE_INPUT_FILES + ABLATE_INPUT_DIRS
        + [f"root@{host}:{REMOTE_DIR}/"],
        dry, timeout=3600)
    if res is not None and res.returncode != 0:
        print(f"  input upload failed (rsync rc={res.returncode}).")
        return False
    return _ablate_stub_corpus(host, base_ssh, base_rsync, dry)


def _ablate_running(base_ssh):
    """Is a panel runner alive on the box? The bracket in the pattern keeps pgrep
    from matching its own command line."""
    try:
        res = subprocess.run(
            base_ssh + ["pgrep -f '[a]blations.run_panel' >/dev/null 2>&1 "
                        "&& echo RUNNING || echo IDLE"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False
    return "RUNNING" in (res.stdout or "")


def _ablate_launch(base_ssh, panel, concurrency, smoke_epochs, dry):
    """Start the panel detached, so the ssh session closing cannot take it down.

    The log is truncated rather than appended: the poll loop decides the run is
    over by finding run_panel's terminal line in it, and a previous run's line
    left at the top would end this one before it started.
    """
    log = _ablate_log(panel)
    extra = f" --smoke-epochs {int(smoke_epochs)}" if smoke_epochs else ""
    # Backgrounded inside a SUBSHELL, not by putting `&` on the whole chain: a
    # chain that failed at the truncate or the cd would otherwise fail in the
    # background where nothing reads its status.
    #
    # THE CHECK IS A SECOND SSH, and it has to be. _ablate_running's pattern is
    # bracketed so pgrep cannot match its own argv, but that says nothing about
    # the shell it runs UNDER — and this command's argv carries the launch line,
    # `ablations.run_panel` unbracketed and all. Asking here would match the
    # shell doing the asking, and report LAUNCHED whatever happened.
    cmd = (f": > {log} && cd {REMOTE_DIR} && "
           f"(setsid nohup env PYTHONUNBUFFERED=1 "
           f"python3 -u -m ablations.run_panel {panel} "
           f"--concurrency {concurrency}{extra} "
           f">> {log} 2>&1 < /dev/null &)")
    res = _run_or_show(base_ssh + [cmd], dry, capture_output=True, text=True,
                       timeout=60)
    if res is None:
        _run_or_show(base_ssh + ["pgrep -f '[a]blations.run_panel' "
                                 ">/dev/null 2>&1 && echo RUNNING || echo IDLE"],
                     dry)
        return True
    if res.returncode != 0:
        print(f"  launch command failed: "
              f"{((res.stdout or '') + (res.stderr or '')).strip()[-400:]}")
        return False
    time.sleep(5)
    if not _ablate_running(base_ssh):
        # Not a failure by itself — a panel whose arms all have records finishes
        # in under a second. Show the log and let the poll loop classify it: it
        # pulls first, then reads run_panel's own terminal line.
        tail = subprocess.run(base_ssh + [f"tail -n 40 {log} 2>/dev/null"],
                              capture_output=True, text=True, timeout=30)
        print(f"  runner already gone. Log so far:\n"
              f"{(tail.stdout or '').strip()}")
    return True


def _ablate_poll(host, base_ssh, base_rsync, panel, arms, dry):
    """Pull results back every cycle and stop on run_panel's terminal line.

    The pull runs BEFORE the verdict is read, and every cycle rather than at the
    end, so a host that disappears mid-panel costs at most the arm that was in
    flight — an arm record is the resume unit on the remote too, so the next
    invocation picks up exactly there.
    """
    local_dir = os.path.join(ABLATE_RESULTS, panel)
    remote_dir = f"{REMOTE_DIR}/{ABLATE_RESULTS}/{panel}"
    pull = base_rsync + [f"root@{host}:{remote_dir}/", local_dir + os.sep]
    # run_panel ends with one of PANEL COMPLETE / PANEL INCOMPLETE, or with
    # "panel complete - nothing to run" when every arm already had a record.
    check = (f"tail -n 200 {_ablate_log(panel)} 2>/dev/null | "
             f"grep -E 'PANEL COMPLETE|PANEL INCOMPLETE|^panel complete' "
             f"| tail -1; echo '|||'; "
             f"pgrep -f '[a]blations.run_panel' >/dev/null 2>&1 "
             f"&& echo running || echo idle")
    if dry:
        print(f"Poll cycle, every {ABLATE_POLL_SECS}s until the log carries a "
              f"verdict:")
        _run_or_show(pull, True)
        _run_or_show(base_ssh + [check], True)
        return None

    os.makedirs(local_dir, exist_ok=True)
    while True:
        try:
            subprocess.run(pull, timeout=1800)
        except subprocess.TimeoutExpired:
            print("  results pull timed out; will retry next cycle.")
        have = [a for a in arms
                if os.path.exists(os.path.join(local_dir, f"{a}.json"))]
        pending = [a for a in arms if a not in have]
        print(f"[{time.strftime('%H:%M:%S')}] {panel}: "
              f"records {', '.join(have) or 'none'} | "
              f"pending {', '.join(pending) or 'none'}")

        try:
            res = subprocess.run(base_ssh + [check], capture_output=True,
                                 text=True, timeout=30)
            parts = [p.strip() for p in (res.stdout or "").split("|||")]
        except subprocess.TimeoutExpired:
            parts = []
        parts += [""] * max(0, 2 - len(parts))
        verdict, proc = parts[:2]
        if verdict:
            return verdict
        if proc == "idle":
            # No verdict and no runner: the launcher died. Say so rather than
            # spinning — whatever it managed to write has just been pulled.
            return "RUNNER GONE (no terminal line in the remote log)"
        time.sleep(ABLATE_POLL_SECS)


def _take_opt(args, name, default=None):
    """Pop `--name value` out of `args`, returning the value."""
    if name not in args:
        return default
    i = args.index(name)
    if i + 1 >= len(args):
        raise ValueError(f"{name} needs a value")
    val = args[i + 1]
    del args[i:i + 2]
    return val


def cmd_queue(args):
    """Keep a pool of arms training from a queue that spans panels.

    Usage: queue [--instance ID] [--concurrency N] [--sync-only] [--dry-run]

    `ablate` runs ONE panel and returns, so a sequence of panels restarts the
    ramp at every boundary and drains the card to a single arm at each panel's
    tail. This launches `ablations.run_queue` instead, which holds the pool full
    across panels and re-reads `ablations/queue.txt` whenever a slot opens — so
    the way to reprioritise is to edit that file and push it, with nothing
    killed and nothing relaunched.

    `--sync-only` pushes the queue file and pulls results without touching the
    runner, which is how a queue is reordered while it is running — and how it
    is POLLED: every invocation ends by expanding the queue through the
    manifests and reporting which named arms still lack a record. Exit 0 means
    QUEUE COMPLETE (every arm has a record — the drain signal a caller can act
    on, e.g. by running `python -m ablations --tier refit`); exit 4 means arms
    are still missing.
    """
    args = list(args)
    dry = "--dry-run" in args
    sync_only = "--sync-only" in args
    args = [a for a in args if a not in ("--dry-run", "--sync-only")]
    concurrency = str(_take_opt(args, "--concurrency", ABLATE_CONCURRENCY))
    if concurrency.lower() != "auto":
        concurrency = str(int(concurrency))
    target = _take_opt(args, "--instance", args[0] if args else None)

    local_queue = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ABLATE_QUEUE)
    if not os.path.exists(local_queue):
        print(f"no {ABLATE_QUEUE} — write one first. Lines are "
              f"`<panel> [<arm> ...]`; a bare panel name means all of its arms.")
        return 1

    inst = find_instance(target)
    if inst is None:
        return 1
    inst_id = inst["id"]
    host, port = get_ssh_details(inst)
    if host is None:
        print(f"instance {inst_id} has no ssh endpoint yet.")
        return 1
    base_ssh, base_rsync = get_ssh_cmds(host, port)

    if not sync_only:
        if not _ablate_provision(base_ssh, dry):
            return 1
        if not _ablate_upload(host, base_ssh, base_rsync, dry):
            return 1
        if not _ablate_git_stamp(base_ssh, dry):
            return 1
        if not _ablate_check_torch(base_ssh, dry):
            return 1

    # The queue file AND the manifests travel every time, sync-only included.
    # Both, because they are one control surface: a queue line names arms that
    # only the manifest defines, so pushing the queue alone makes the pool warn
    # that the arms do not exist — which is what it did the first time a budget
    # was appended this way.
    root = os.path.dirname(os.path.abspath(__file__))
    _run_or_show(base_rsync + [local_queue,
                               f"root@{host}:{REMOTE_DIR}/{ABLATE_QUEUE}"],
                 dry, capture_output=True)
    _run_or_show(base_rsync + [os.path.join(root, "ablations", "panels") + os.sep,
                               f"root@{host}:{REMOTE_DIR}/ablations/panels/"],
                 dry, capture_output=True)
    print(f"  pushed {ABLATE_QUEUE} and the panel manifests")

    if not sync_only and not dry:
        # ONE POOL PER BOX. A pool idles on an empty queue rather than exiting,
        # so the previous one is usually still alive — and it re-reads the
        # queue file just pushed, so it is already doing what a second launch
        # would do. Two pools each fill the card to their own level, launch
        # into each other's VRAM, and hand out the same arms twice.
        # The bracket keeps the probing shell's own command line, which
        # carries this pattern, from matching it.
        live = _run_or_show(base_ssh + ["pgrep -fc '[a]blations.run_queue' "
                                        "|| true"],
                            False, capture_output=True, text=True, timeout=30)
        n_live = int((getattr(live, "stdout", "") or "0").strip().splitlines()[0]
                     or 0) if live is not None else 0
        if n_live:
            print(f"  a pool is already live on {inst_id} ({n_live} process); "
                  f"it re-reads the pushed queue at its next slot, so nothing "
                  f"was launched. Kill it first to restart with new code.")
            sync_only = True

    if not sync_only:
        log = f"{REMOTE_DIR}/{ABLATE_LOG_PREFIX}_queue.log"
        cmd = (f": > {log} && cd {REMOTE_DIR} && "
               f"(setsid nohup env PYTHONUNBUFFERED=1 "
               f"python3 -u -m ablations.run_queue "
               f"--concurrency {concurrency} "
               f">> {log} 2>&1 < /dev/null &)")
        _run_or_show(base_ssh + [cmd], dry, capture_output=True, timeout=60)
        print(f"  pool launched on {inst_id}, log {log}")

    # Pull whatever exists now. The pool does not end on its own — it idles on an
    # empty queue waiting for more — so this returns rather than watching, and
    # `queue --sync-only` is how results are collected again later.
    _run_or_show(base_rsync
                 + [f"root@{host}:{REMOTE_DIR}/{ABLATE_RESULTS}/",
                    ABLATE_RESULTS + os.sep],
                 dry, capture_output=True)
    have = len(glob.glob(os.path.join(ABLATE_RESULTS, "*", "*.json")))
    print(f"  {have} arm record(s) under {ABLATE_RESULTS}/")

    # THE DRAIN SIGNAL. Expand the queue through the manifests — the same
    # reading the pool does — and say which named arms still lack a record,
    # so a caller polling with --sync-only gets completeness as an exit code
    # rather than inferring it from a record count that also counts archived
    # regimes and panels the queue never named.
    wanted = []
    for raw in open(local_queue):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        panel, *arms = line.split()
        manifest_path = os.path.join(root, "ablations", "panels",
                                     f"{panel}.json")
        if not os.path.exists(manifest_path):
            print(f"  queue names panel {panel!r} with no manifest — not "
                  f"counted toward completion")
            continue
        manifest_arms = json.load(open(manifest_path))["arms"]
        for arm in (arms or list(manifest_arms)):
            if arm not in manifest_arms:
                print(f"  queue names {panel}/{arm} which the manifest does "
                      f"not define — not counted toward completion")
                continue
            if (panel, arm) not in wanted:
                wanted.append((panel, arm))
    missing = [f"{p}/{a}" for p, a in wanted
               if not os.path.exists(os.path.join(ABLATE_RESULTS, p,
                                                  f"{a}.json"))]
    if not wanted:
        print("  the queue file names no arms — nothing to complete")
        return 0
    if missing:
        shown = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
        print(f"  queue: {len(wanted) - len(missing)}/{len(wanted)} arms have "
              f"records; missing {len(missing)}: {shown}")
        return 4
    print(f"  QUEUE COMPLETE — all {len(wanted)} arms named by {ABLATE_QUEUE} "
          f"have records. The pool idles until the queue file changes; the "
          f"ledger is brought up to date separately "
          f"(./.venv/bin/python3 -m ablations --tier refit).")
    return 0


def cmd_ablate(args):
    """Run a refit-tier MIRT ablation panel on a rented card.

    Usage: ablate <panel> [--instance ID] [--concurrency N] [--smoke-epochs N]
                          [--keep] [--dry-run]

    The panel is ablations/panels/<panel>.json; each arm is
    pipeline.train_mirt.run() with one knob turned, so the box needs the fit's
    inputs and none of the per-problem stores. Arms are independent, hence
    --concurrency on one GPU. A re-invocation on the same box does NOT relaunch
    while a panel is live — it attaches to the poll loop, and run_panel's own
    skip-if-exists makes the ARM the resume unit if it is not.

    `--instance` expects a box this tool set up (`add`, or `setup` on an
    existing one): the base packages and the repo layout come from there, and
    this step adds only what the fit needs on top.
    """
    args = list(args)
    dry = "--dry-run" in args
    if dry:
        args.remove("--dry-run")
    keep = "--keep" in args
    if keep:
        args.remove("--keep")
    try:
        target = _take_opt(args, "--instance")
        concurrency = str(_take_opt(args, "--concurrency", ABLATE_CONCURRENCY))
        if concurrency != "auto":
            concurrency = str(int(concurrency))
        smoke_epochs = _take_opt(args, "--smoke-epochs")
        smoke_epochs = None if smoke_epochs is None else int(smoke_epochs)
    except ValueError as e:
        print(f"Bad arguments: {e}")
        return
    if len(args) != 1 or args[0].startswith("-"):
        print("Usage: ablate <panel> [--instance ID] [--concurrency N] "
              "[--smoke-epochs N] [--keep] [--dry-run]")
        return
    panel = args[0]

    arms = _ablate_panel_arms(panel)
    if arms is None:
        return
    print(f"Panel '{panel}': {len(arms)} arm(s) — {', '.join(arms)}")
    if smoke_epochs:
        print(f"  SMOKE: every arm capped at {smoke_epochs} epochs. The budget is "
              f"recorded in each arm JSON and producers refuse to quote it.")

    # --- find or rent -------------------------------------------------------
    if dry:
        inst_id = target or "<instance-id>"
        host, port = "<host>", "<port>"
        if target:
            print(f"Would resolve the ssh route for instance {target} "
                  f"(get_ssh_details).")
        else:
            print("Would take the one existing instance, or with none rent one "
                  "exactly as `add` does:")
            _run_or_show([VASTAI_BIN, "search", "offers", OFFER_QUERY,
                          "--limit", OFFER_LIMIT, "--raw"], True)
            _run_or_show([VASTAI_BIN, "create", "instance", "<best-TFLOPS/$>"]
                         + CREATE_ARGS + ["--raw"], True)
            print("  then setup_instance_with_retry (apt, pip, code, weights).")
    else:
        if target is None and not get_all_instances():
            print("No instances; renting one. IT BILLS UNTIL YOU DESTROY IT.")
            target = cmd_add()
            if not target:
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

    step = f"ablate:{panel}"
    live = False
    if not dry:
        wait_for_ssh(host, port)
        live = _ablate_running(base_ssh)
        if live:
            remote_step = get_instance_step(inst_id) or "?"
            # ATTACHING IS ONLY RIGHT FOR THE PANEL THAT IS ACTUALLY RUNNING.
            # Attaching to a different one polls ITS log, which still holds the
            # terminal line of whenever it last ran — so the run returns
            # instantly, reports that stale verdict as this invocation's, and
            # the arms it was asked for are never trained. A resumed panel is
            # exactly the case where that log is populated and wrong.
            if remote_step != step:
                print(f"Instance {inst_id} is running '{remote_step}', not "
                      f"'{step}'. Refusing to attach: this panel's log still "
                      f"holds its LAST run's verdict, and polling it would "
                      f"report that as this run's answer. Wait for the running "
                      f"panel, or `stop {inst_id}` to end it.")
                return 1
            print(f"Instance {inst_id} is already running this panel. Not "
                  f"relaunching; attaching to the results pull.")
        elif _is_step_running(base_ssh):
            print(f"Instance {inst_id} is busy with another pipeline step. "
                  f"Run 'stop' first.")
            return

    if not live:
        print(f"Provisioning instance {inst_id} for '{step}'...")
        if not _ablate_provision(base_ssh, dry):
            return
        print(f"Uploading code and inputs to instance {inst_id}...")
        if not _ablate_upload(host, base_ssh, base_rsync, dry):
            return
        if not _ablate_git_stamp(base_ssh, dry):
            return
        if not _ablate_check_torch(base_ssh, dry):
            return
        print(f"Launching '{step}' (concurrency {concurrency}) on {inst_id}...")
        if not _ablate_launch(base_ssh, panel, concurrency, smoke_epochs, dry):
            return
        _run_or_show(
            base_ssh + [f"echo '{step}' > {REMOTE_DIR}/running_step && "
                        f"date -Iseconds > {REMOTE_DIR}/running_started"],
            dry, capture_output=True)
        if not dry:
            register_instance(inst_id, step=step,
                              started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        print(f"  log: {_ablate_log(panel)}  (stream {inst_id} tails "
              f"{LOG_FILENAME_PREFIX}_0.log, not this one)")

    try:
        verdict = _ablate_poll(host, base_ssh, base_rsync, panel, arms, dry)
    except KeyboardInterrupt:
        # The panel is detached, so this stops WATCHING and nothing else. Say so:
        # the box is still fitting and still billing either way.
        print(f"\nStopped watching. The panel is still running on {inst_id} — "
              f"re-run `ablate {panel}` to resume the pull, or `stop {inst_id}` "
              f"to end it.")
        return
    if dry:
        print(f"On a verdict: clear running_step, report the arm records under "
              f"{os.path.join(ABLATE_RESULTS, panel)}/, and "
              + ("keep the instance (--keep)." if keep else
                 f"print `python vast_orchestration.py shutdown {inst_id}`."))
        return

    local_dir = os.path.join(ABLATE_RESULTS, panel)
    have = [a for a in arms
            if os.path.exists(os.path.join(local_dir, f"{a}.json"))]
    missing = [a for a in arms if a not in have]
    print(f"\n{panel}: {verdict}")
    print(f"  records in {local_dir}/: {', '.join(have) or 'none'}")
    if missing:
        print(f"  STILL MISSING: {', '.join(missing)} — re-run this command to "
              f"resume; run_panel skips the arms that already have a record.")
    subprocess.run(base_ssh + [f"rm -f {REMOTE_DIR}/running_step "
                               f"{REMOTE_DIR}/running_started"],
                   capture_output=True)
    register_instance(inst_id, step=None, started=None)
    if keep:
        print(f"  instance {inst_id} kept (--keep) and still billing.")
    else:
        print(f"  instance {inst_id} is STILL RUNNING and still billing. "
              f"Destroy it with:")
        print(f"    python vast_orchestration.py shutdown {inst_id}")
    # A PANEL WITH A HOLE IN IT EXITS NON-ZERO. Everything above is printed
    # either way, but a caller chaining panels — or reading $? at all — cannot
    # otherwise tell a finished panel from one that lost arms, and an arm that
    # aborted remotely leaves no record to notice later. Resumable, so the
    # answer to a non-zero exit is usually to run the same command again.
    if missing:
        return 1
    return 0


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
            "tune/generate/benchmark/losing_lines need a board size first, e.g. run generate 13",
        ],
        "handler": cmd_run,
    },
    "queue": {
        "usage": "queue [<id>] [opts]",
        "help": ["Keep a POOL of ablation arms training from ablations/queue.txt,",
                 "spanning panels — no ramp restart or drained tail at a panel",
                 "boundary. The runner re-reads the queue whenever a slot opens,",
                 "so reprioritising is editing that file and pushing it:",
                 "[--instance ID] [--concurrency N] [--sync-only] [--dry-run]"],
        "handler": cmd_queue,
    },
    "ablate": {
        "usage": "ablate <panel> [opts]",
        "help": [
            "Run a refit-tier MIRT ablation panel (ablations/panels/<panel>.json)",
            "on a rented card and pull the arm records back as they land.",
            "[--instance ID] [--concurrency N] [--smoke-epochs N] [--keep] [--dry-run]",
        ],
        "handler": cmd_ablate,
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
    # A handler's return value IS the exit status when it gives one: `ablate`
    # reports an incomplete panel that way, and a chain of panels has no other
    # way to notice a hole. Handlers that return nothing keep exiting 0.
    sys.exit(COMMANDS[sys.argv[1]]["handler"](sys.argv[2:]) or 0)


if __name__ == "__main__":
    main()
