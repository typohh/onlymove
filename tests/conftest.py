"""Shared fixtures: the node bridge the JS-twin tests call through."""

import json
import math
import os
import shutil
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

BRIDGE = os.path.join(PROJECT_ROOT, "tests", "js_bridge.mjs")

# The modules that import package_for_pwa's output as JSON at load time. That
# output is gitignored, so on a tree that has never been packaged they cannot
# load at all — a skip, not a failure.
PACKAGED_JSON_DIR = os.path.join(PROJECT_ROOT, "pwa", "tsumegos")


def _encode(v):
    """JSON-safe form of `v`; the numbers JSON cannot carry cross as markers the
    bridge decodes (see js_bridge.mjs)."""
    if isinstance(v, float):
        if math.isnan(v):
            return {"__num__": "NaN"}
        if math.isinf(v):
            return {"__num__": "Infinity" if v > 0 else "-Infinity"}
        if v == 0.0 and math.copysign(1.0, v) < 0:
            return {"__num__": "-0"}
        return v
    if hasattr(v, "tolist"):          # numpy scalars and arrays
        return _encode(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, dict):
        return {k: _encode(x) for k, x in v.items()}
    return v


def _decode(v):
    if isinstance(v, list):
        return [_decode(x) for x in v]
    if isinstance(v, dict):
        if set(v) == {"__num__"}:
            return {"NaN": math.nan, "Infinity": math.inf,
                    "-Infinity": -math.inf, "-0": -0.0}[v["__num__"]]
        return {k: _decode(x) for k, x in v.items()}
    return v


class JsBridge:
    """Batches calls into ONE node process: `run([(module, fn, args), ...])`
    returns the values in order. `args=None` reads the export as a constant."""

    def __init__(self, node):
        self.node = node

    def run(self, calls):
        payload = [{"module": m, "fn": f, **({} if a is None else {"args": _encode(list(a))})}
                   for m, f, a in calls]
        proc = subprocess.run([self.node, "--no-warnings", BRIDGE],
                              input=json.dumps(payload), capture_output=True,
                              text=True, cwd=PROJECT_ROOT, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"js_bridge exited {proc.returncode}:\n{proc.stderr}")
        results = json.loads(proc.stdout)
        values = []
        for (m, f, _), r in zip(calls, results):
            if "error" in r:
                if "ENOENT" in r["error"] and "pwa/tsumegos" in r["error"]:
                    pytest.skip(f"{m} imports package_for_pwa output that is not "
                                f"present under pwa/tsumegos/ — run package_for_pwa")
                raise AssertionError(f"{m}:{f} threw in node:\n{r['error']}")
            values.append(_decode(r["ok"]))
        return values

    def call(self, module, fn, *args):
        return self.run([(module, fn, args)])[0]

    def const(self, module, name):
        return self.run([(module, name, None)])[0]


@pytest.fixture(scope="session")
def js():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    return JsBridge(node)
