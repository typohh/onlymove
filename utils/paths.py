"""The repository root, resolved in exactly one place.

Every module under `utils/` used to derive this itself, as a fixed number of
`dirname()` calls on its own `__file__`. That count is a function of how deep the
file sits, so splitting `utils/` into `utils/katago/` and `utils/rating/` moved
nine files one level down and silently pointed all nine at `utils/` as the
project root — the failure surfaces far from the cause, as a missing artifact
under a path with `utils/` doubled into it.

Resolving it here makes a future move a non-event. This module deliberately
imports nothing from the package (`utils/constants.py` needs it too, and
`constants` already reaches into `utils.katago.models` for the default network,
so any dependency in the other direction would close a cycle).
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
