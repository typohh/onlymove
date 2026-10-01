"""Bootstrap module: importing this adds the vendored KataGo subpackage to
sys.path so callers can do `from katago.train.load_model import load_model`
etc. without polluting the rest of the project's import surface.
"""
import os
import sys

# The transformer blocks carry training-speed paths — fused Triton kernels and
# a compiled flex-attention — that build against the Python headers at first
# use. Extraction is inference on small batches, the plain PyTorch paths compute
# the same forward pass, and a host without the headers cannot build the others.
for _flag in ("KATAGO_FUSED_ROPE_BACKWARD", "KATAGO_FUSED_SWIGLU_KERNEL",
              "KATAGO_FLEX_ATTENTION"):
    os.environ.setdefault(_flag, "0")

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
