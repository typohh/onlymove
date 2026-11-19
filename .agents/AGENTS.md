# Project rules — onlymove.net

**Read [`CLAUDE.md`](../CLAUDE.md) in the repository root. It is the single source of truth** for the subsystem layout, the pipeline commands, the coding conventions, and the cross-file invariants — and it applies to any agent working here, not only to Claude Code.

This file deliberately holds no copy of that material. It used to, and the copy drifted: it went on describing the weakness recommender as a "FastICA-rotated K-dim embedding (K=64)" for months after the supervised residual skill axes replaced ICA (2026-07-18), and it listed a `pipeline.build_embedding` command that no longer existed. A second, lossier description of the architecture is worse than none, because nothing signals when it goes stale.

Further reading, all linked from `CLAUDE.md`: `README.md` (project overview and the ordered pipeline), `docs/methodology.md` (the rating/rank/motif methodology, including the alternatives that were measured and rejected), `pwa/README.md` (front-end and Worker), `auxiliary/README.md` (the diagnostic and research tooling), `data/README.md` (what may and may not be copied out of `data/`).

## The one rule that is not in `CLAUDE.md`

- **When the user asks a question, answer it.** Give the explanation or the suggestion; do not turn a question into code changes.
