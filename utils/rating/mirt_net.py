"""The MIRT feature trunk, defined ONCE.

Three callers build it: pipeline.train_mirt fits it, pipeline.write_mirt_per_cell
reloads it to project every legal cell, and auxiliary.rating_dev.train_joint_mirt is
the experiment surface. Each used to restate the layer stack, and they agreed only by
convention. That is the same failure utils.rating.features.load_joint_tail_blocks
exists to prevent, one layer up: the fanout has to rebuild the FITTED architecture
exactly, and a mismatch that happens to keep the shapes — a different activation at
the same widths — loads cleanly and projects every cell through the wrong function.

`dropout` is a constructor argument rather than a constant because it is NOT part of
the state_dict: a reloader passes 0.0 and still loads weights fitted at DROPOUT.
"""

import torch
import torch.nn as nn

from utils.rating.model_constants import DIFF_HIDDEN, DROPOUT


class SwiGLU(nn.Module):
    """silu(xW) * xV, replacing [Linear, ReLU]. The gate is its own projection, so a
    same-width swap is also a capacity increase."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.w = nn.Linear(d_in, d_out)
        self.v = nn.Linear(d_in, d_out)

    def forward(self, x):
        return torch.nn.functional.silu(self.w(x)) * self.v(x)


def build_trunk(fdim, dropout=DROPOUT, hidden=DIFF_HIDDEN, act="swiglu"):
    """(trunk, out_dim) for a feature width of `fdim`.

    `hidden` and `act` are open only so the dev trainer can sweep them; production
    takes the defaults, DIFF_HIDDEN being an interior optimum in both directions.
    """
    layers, d = [], int(fdim)
    for h in hidden:
        layers += ([SwiGLU(d, h)] if act == "swiglu"
                   else [nn.Linear(d, h), nn.ReLU()])
        layers += [nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers), d
