"""The repository root, resolved in exactly one place.

Every module under `utils/` used to derive this itself, as a fixed number of
`dirname()` calls on its own `__file__`. That count is a function of how deep the
file sits, so splitting `utils/` into `utils/katago/` and `utils/rating/` moved
nine files one level down and silently pointed all nine at `utils/` as the
project root — the failure surfaces far from the cause, as a missing artifact
under a path with `utils/` doubled into it.

Resolving it here makes a future move a non-event. This module deliberately
imports nothing from the package (`utils/constants.py` and `utils.katago.models`
both need it, so any dependency in the other direction would close a cycle).
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The local snapshot of the REMOTE production D1, written by
# pipeline.fetch_database: db/d1.sqlite is the canonical binary (query this), and
# db/d1.sql the regenerated text dump beside it. Not to be confused with the
# wrangler DEV database under .wrangler/, which fetch_database resets from this one.
#
# It lived at attempts/database.{sqlite,sql} until 2026-08-03, where it shared a
# directory with the tracked per-size bot benchmark results — a pipeline input,
# whereas this is a ~50 MB gitignored snapshot of real user data. Fifteen modules
# each rebuilt the literal; they read it from here now.
DB_DIR = os.path.join(PROJECT_ROOT, "db")
DB_SQLITE = os.path.join(DB_DIR, "d1.sqlite")
DB_SQL = os.path.join(DB_DIR, "d1.sql")
