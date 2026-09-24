"""The utils/ import boundary, checked at import time.

  * importing any utils/rating/ module loads nothing from utils/katago/
    (its vendored PyTorch KataGo included) — the engine helpers the rating side
    needs are imported INSIDE the functions that use them;
  * importing any utils/katago/ module loads nothing from utils/rating/;
  * utils/constants.py loads neither side;
  * utils/rating/ratings_io.py is stdlib + numpy only, so the light readers
    (package_for_pwa, se_floor) can read a curve without scipy.

Each module is imported in a FRESH interpreter, since one import's leftovers in
sys.modules would hide another's. The check is on the FILES loaded, not module
names, so the vendor's top-level `katago` package counts as utils/katago/.
"""

import json
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KATAGO_DIR = os.path.join(PROJECT_ROOT, "utils", "katago")
RATING_DIR = os.path.join(PROJECT_ROOT, "utils", "rating")


def _modules(package_dir):
    """Dotted names of the package's own modules (the vendored tree excluded)."""
    rel = os.path.relpath(package_dir, PROJECT_ROOT).replace(os.sep, ".")
    names = [rel]
    for f in sorted(os.listdir(package_dir)):
        if f.endswith(".py") and f != "__init__.py":
            names.append(f"{rel}.{f[:-3]}")
    return names


# Reports only what the import ADDS: interpreter startup (site, sitecustomize)
# is not the module's doing.
_PROBE = """
import importlib, json, sys
sys.path.insert(0, {root!r})
before = set(sys.modules)
importlib.import_module({module!r})
new = set(sys.modules) - before
files = sorted({{getattr(sys.modules.get(n), '__file__', None) or '' for n in new}})
print(json.dumps({{"files": files, "top": sorted({{n.split('.')[0] for n in new}})}}))
"""


def _loaded(module):
    proc = subprocess.run([sys.executable, "-c", _PROBE.format(root=PROJECT_ROOT, module=module)],
                          capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300)
    assert proc.returncode == 0, f"importing {module} failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _under(files, directory):
    prefix = directory + os.sep
    return [f for f in files if f.startswith(prefix)]


@pytest.mark.parametrize("module", _modules(RATING_DIR))
def test_rating_side_loads_no_katago_side(module):
    assert _under(_loaded(module)["files"], KATAGO_DIR) == []


@pytest.mark.parametrize("module", _modules(KATAGO_DIR))
def test_katago_side_loads_no_rating_side(module):
    assert _under(_loaded(module)["files"], RATING_DIR) == []


def test_constants_loads_neither_side():
    files = _loaded("utils.constants")["files"]
    assert _under(files, KATAGO_DIR) == [] and _under(files, RATING_DIR) == []


def test_ratings_io_is_stdlib_and_numpy():
    top = _loaded("utils.rating.ratings_io")["top"]
    allowed = set(sys.stdlib_module_names) | {"numpy", "utils", "__main__"}
    extra = [n for n in top if n not in allowed and not n.startswith("_")]
    assert extra == []
