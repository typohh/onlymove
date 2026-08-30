# Ablations

Repeatable measurements against the SHIPPED fit. Everything here answers a standing question about the production model — the kind of figure `docs/methodology.md` or `docs/mirt_benchmark.md` quotes — and is meant to be re-run after a refit rather than archaeologically recovered.

Contract for a script in this directory:

- **Repeatable**: seeded, deterministic given its inputs, runnable as `./.venv/bin/python3 -m auxiliary.ablations.<name>` with defaults that reproduce the quoted figure.
- **Reads production artifacts only** — the shipped model (`ratings/mirt_model.pt`), the shared feature assembly (`utils.rating.features`), the corpus. No private restatement of the trunk, the feature order, or the projection; import the single-source helpers so the measurement cannot drift from what ships.
- **Self-describing output**: the printed table carries its own column meanings; a reader should not need the script to interpret the numbers.
- **Kept current**: when a refit or format change invalidates a figure a doc quotes from here, re-run the script and update the doc in the same pass.

Scripts:

- `tail_conditioning_sensitivity.py` — how much of the per-cell axis readings' WITHIN-PROBLEM structure (the tendency sign test and the availability null) depends on the per-problem tail conditioning, measured by projecting variation cells with the real tail against a fully-dropped tail.
