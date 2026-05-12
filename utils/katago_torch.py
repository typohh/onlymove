"""Bootstrap module: importing this adds the vendored KataGo subpackage to
sys.path so callers can do `from katago.train.load_model import load_model`
etc. without polluting the rest of the project's import surface.
"""
import os
import sys

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "katago_vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
