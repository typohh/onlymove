"""Which KataGo binary this machine runs, and the environment it needs.

`bots/` may hold several builds — CUDA 13, CUDA 12, OpenCL — and which one works
is a property of the host: the driver, and whether the CUDA runtime libraries the
build links are anywhere the loader looks. The CUDA builds link cuBLAS/cuDNN by
soname, and those ship inside the venv's `nvidia-*` wheels (torch's
dependencies) rather than on the system path, so a CUDA build that fails bare
with "cannot open shared object file" usually runs once those directories are on
LD_LIBRARY_PATH. A per-user `~/.local/lib/katago-deps` is searched too, for
libraries a host lacks outright (libzip for the OpenCL build, for one).

A CUDA build is accepted only on a REAL QUERY, not on linking: a build can load
its libraries and still fail at device initialisation against a mismatched
driver. The probe starts the analysis engine, asks for one 1-visit evaluation and
requires the answer. CUDA 13 is tried before CUDA 12, and the OpenCL build is the
fallback — it is never probed, because its first start runs a kernel tuning pass
that takes minutes.

The verdict is cached per host in `bots/.katago_binary.json` (bots/ is
gitignored) against the chosen binary's size and mtime, so a probe runs once per
machine and again only when that binary is replaced.

KATAGO_PATH in the environment overrides the choice outright, as before.
"""

import glob
import json
import os
import socket
import subprocess
import sys

from utils.paths import PROJECT_ROOT

BOTS_DIR = os.path.join(PROJECT_ROOT, "bots")
CONFIG_FILE = os.path.join(BOTS_DIR, "analysis.cfg")
CACHE_PATH = os.path.join(BOTS_DIR, ".katago_binary.json")
CANDIDATES = ("katago_cuda13", "katago_cuda12")   # probed, in order of preference
FALLBACK = "katago"                               # the OpenCL build
PROBE_TIMEOUT_SECONDS = 180

_resolved = None


def library_dirs():
    """Directories holding the shared libraries the CUDA builds link: the
    `nvidia-*` wheels of the running interpreter's environment and of the repo
    venv, plus the per-user deps directory."""
    prefixes = {sys.prefix, os.path.join(PROJECT_ROOT, ".venv")}
    dirs = []
    for p in sorted(prefixes):
        dirs += sorted(glob.glob(os.path.join(p, "lib", "python3*", "site-packages",
                                              "nvidia", "*", "lib")))
    deps = os.path.expanduser("~/.local/lib/katago-deps")
    if os.path.isdir(deps):
        dirs.append(deps)
    return list(dict.fromkeys(dirs))


def katago_env():
    """os.environ with library_dirs() prepended to LD_LIBRARY_PATH."""
    env = dict(os.environ)
    parts = library_dirs() + [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(parts))
    return env


def _stamp(path):
    st = os.stat(path)
    return [st.st_size, int(st.st_mtime)]


def _probe(path, env):
    """True when `path` answers one real evaluation."""
    from utils.katago.models import DEFAULT_MODEL, get_model
    query = json.dumps({"id": "probe", "moves": [], "rules": "japanese", "komi": 6.5,
                        "boardXSize": 9, "boardYSize": 9, "maxVisits": 1})
    try:
        r = subprocess.run(
            [path, "analysis", "-config", CONFIG_FILE, "-model", get_model(DEFAULT_MODEL),
             "-override-config", "maxBoardXSizeForNNBuffer=9,maxBoardYSizeForNNBuffer=9,"
                                 "numAnalysisThreads=1,nnMaxBatchSize=1"],
            input=query + "\n", capture_output=True, text=True, env=env,
            timeout=PROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return False
    for line in r.stdout.splitlines():
        try:
            if json.loads(line).get("id") == "probe" and "rootInfo" in json.loads(line):
                return True
        except ValueError:
            continue
    return False


def resolve_katago():
    """(binary path, environment) for launching KataGo on this machine."""
    global _resolved
    if _resolved is not None:
        return _resolved
    env = katago_env()
    override = os.environ.get("KATAGO_PATH")
    if override:
        _resolved = (override, env)
        return _resolved

    host = socket.gethostname()
    try:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(host)
    if hit and os.path.exists(hit["path"]) and _stamp(hit["path"]) == hit["stamp"]:
        _resolved = (hit["path"], env)
        return _resolved

    chosen = None
    for name in CANDIDATES:
        path = os.path.join(BOTS_DIR, name)
        if os.path.exists(path) and _probe(path, env):
            chosen = path
            break
    if chosen is None:
        chosen = os.path.join(BOTS_DIR, FALLBACK)
        if not os.path.exists(chosen):
            raise FileNotFoundError(
                f"no KataGo binary in {BOTS_DIR}: none of {CANDIDATES} answered a probe "
                f"query and the fallback {FALLBACK} is absent")
    cache[host] = {"path": chosen, "stamp": _stamp(chosen)}
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=1)
    except OSError:
        pass
    print(f"KataGo binary for {host}: {os.path.relpath(chosen, PROJECT_ROOT)}", flush=True)
    _resolved = (chosen, env)
    return _resolved
