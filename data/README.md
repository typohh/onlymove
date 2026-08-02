# `data/`

Three kinds of thing that used to sit side by side at this level. They differ in where they come from, whether they are tracked, and what you may do with them.

## `reference/` — external inputs, tracked

Rank data the project did not generate and cannot regenerate.

- **`even_game_winrates.txt`** — KataGo's tabulated weak-side win rate for even games at each rank gap. Read by `utils/rating/rank_curve_shape.py` (`WINRATE_TABLE_PATH`), which fits the shape `pipeline.map_rating_to_rank` uses as the prior mean for the rating→rank curve. The one file here with a live code consumer.
- **`rank_survey_2018.csv`**, **`rank_survey_2024.csv`** — community surveys of how players' ranks correspond across servers (OGS, KGS, Fox, Tygem, IGS, EGF, AGA, …). Archival: no code reads them, and the live equivalent is now the in-app `rank_surveys` D1 table that `map_rating_to_rank` fits its per-server couplings from. Kept because they are the historical record of where the cross-server correspondence came from.

## `tsumego_audit/` — audit outputs, tracked

Findings from `auxiliary/check_tsumego_sequences.py`: problems whose winning line does not hold up. `bad_terminal_results.txt` is the plain id list — the form `auxiliary/purge_tsumegos.mjs --ids` consumes — and the `_detail.csv` beside it carries the per-problem reasons. Tracked because a quarantine decision is a record, not a cache.

## `user_dumps/` — real user data, NEVER tracked

localStorage and IndexedDB exports pulled from real devices to debug sync and history problems. **Every file contains an account credential (the 8-char `userId`, which IS the password — see the account-sync note in `CLAUDE.md`) and that user's full answer history.** Gitignored, and it must stay that way: do not commit these, quote ids from them, or copy them anywhere outside this directory.

It was called `test/` until 2026-08-03, which read as disposable fixtures.
