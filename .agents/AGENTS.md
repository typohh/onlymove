# Project Rules & Customizations

This file provides style guidelines, command references, and architectural constraints for Antigravity when working on the YoseTsumego / onlymove.net repository.

## Subsystem Overview

The project is a Go (Baduk/Weiqi) tsumego pipeline where KataGo generates and rates problems, packaged into zip archives for a Cloudflare-hosted PWA.

1. **`pipeline/`**: Python entrypoints run via `python -m pipeline.<step>`. `package_for_pwa` is the single writer of `pwa/tsumegos/`.
2. **`utils/`**: Shared Go/KataGo integration logic, e.g., `KatagoController` (query dispatcher) and `ProblemFinder` (heuristics).
3. **`pwa/`**: Vanilla-JS ESM PWA built with esbuild, running via a Cloudflare Worker (`pwa/src/index.js`).
4. **`analysis_server/`**: A tiny HTTP server wrapping KataGo for client-side analysis fallback.
5. **`auxiliary/`**: One-off diagnostic scripts.

---

## Commands & Build Setup

### Python Pipeline Commands
Always execute Python pipeline scripts using the virtual environment:
```bash
./.venv/bin/python -m pipeline.fetch_database          # Pull new reports from D1 database
./.venv/bin/python -m pipeline.tune_engine <size>      # Hardware tune analysis config (size: 9 or 13)
./.venv/bin/python -m pipeline.generate_tsumegos <size># Generate tsumego problems (size: 9 or 13)
./.venv/bin/python -m pipeline.benchmark_bots <size>   # Bots solve tsumegos (size: 9 or 13)
./.venv/bin/python -m pipeline.extract_activations     # GPU trunk activations extraction
./.venv/bin/python -m pipeline.cache_move_features    # Build move features cache
./.venv/bin/python -m pipeline.train_rating            # Train rating model (produces ratings/)
./.venv/bin/python -m pipeline.build_axes              # Fit the supervised skill axes
./.venv/bin/python -m pipeline.package_for_pwa         # Package data into pwa/tsumegos/
./.venv/bin/python -m pipeline.map_rating_to_rank      # Map raw rating to consensus ranks
./.venv/bin/python -m pipeline.fit_population_constants# Fit retention/time-bias population parameters
```

### PWA Commands
```bash
cd pwa
npm install
npm run build      # esbuild -> pwa/dist/
npm run dev        # local worker & assets -> http://localhost:8787
npm run deploy     # publish to Cloudflare Workers
```
*Note: Local testing must go through `npm run dev` since URL routing and `/report` are handled by the worker.*

---

## Development Conventions

### Python Code Style & Imports
- Use **absolute imports** rooted at the repository root (e.g., `from utils.constants import ...`).
- Appending `PROJECT_ROOT` to `sys.path` is standard at the start of pipeline entry points.
- **Tests**: The `tests/` folder is empty and there is no test runner. Do not claim to run automated tests unless you explicitly write and run them.

### Coordinates
- **KataGo coordinates** skip the letter `I` (columns: `ABCDEFGHJKLMNOPQRST`).
- **SGF coordinates** do not skip `I`.
- Use the helpers `katago_to_sgf_coords` in [utils/katago/queries.py](file:///home/ootakamoku/Repositories/onlymove/utils/katago/queries.py) and `sgf_to_gtp` in [pipeline/benchmark_bots.py](file:///home/ootakamoku/Repositories/onlymove/pipeline/benchmark_bots.py) instead of rewriting coordinate math.

### Database Operations
- `attempts/database.sqlite` is the canonical merge target (gitignored).
- If you need to rebuild the sqlite DB from the sql dump manually:
  ```bash
  { echo "BEGIN;"; cat attempts/database.sql; echo "COMMIT;"; } | sqlite3 attempts/database.sqlite
  ```
- **Avoid piping sql files through `head`/`tail`** as this can trigger `SIGPIPE` and silently corrupt the import.

### Glob Safety (Large Directories)
- Many directories have 100k+ files (e.g., `tsumegos/`, `activations/`, `embeddings/`).
- **Do not use simple shell globs** like `ls dir/*.npz` as they will exceed `ARG_MAX` limits and return 0.
- Instead, use `find` or `xargs`:
  ```bash
  find <dir> -maxdepth 1 -name '*.npz' -printf . | wc -c
  ```

---

## Architecture & Invariants

### Tsumego Identity
- Symmetry-aware: `tsumego_id` is an 8-hex character truncation of the SHA-256 hash computed over all 8 D4 transforms of the board (`min()` canonical orientation).
- Every problem has exactly **one correct first move** (the `solution`).

### Board Sizes
- Supported sizes: **`9`** and **`13`**.
- PWA uses size hints from the URL path at boot (`/9` or `/13`), then normalizes the URL.
- Linked mode (`?id=...`) is an exception and retains its URL structure through reloads.

### PWA State & Sync Invariants
- `pwa/js/state.js` centralizes mutable state.
- **Linked-mode solving**: Attempts in linked mode do not count toward stats, D1, or history.
- **Account Sync**: Uses D1 AUTOINCREMENT `id` as the cursor (`sinceId`), **not `created_at`**, to ensure offline-flushed historical logs are never skipped during sync.
- **`answerHistory` Split Storage**: Synchronous localStorage stores only the most recent 3000 attempts (`ANSWER_HISTORY_WORKING_SET`). Older records are evicted to IndexedDB tail storage.

### Ratings & Embedding Customizations
- Ranks are derived client-side from raw ratings using map `M`. Shards in `metadata.json` store raw `rating` and `slope` ($a_p$), not rank.
- **Weakness Recommender**: Augments logistic regression using a FastICA-rotated $K$-dim embedding ($K=64$).
- FastICA uses a skew contrast (positive tail has the motif, negative tail is Gaussian baseline).
- **Cache Busting**: Asset fetches use `?v=${DATA_VERSION}` where `DATA_VERSION` is a content hash of `metadata.json` files and is inlined in `embedding_meta.json`.

---

## Interaction Rules
- **Responding to Questions**: If the user asks a question, provide a textual explanation or suggestions rather than making code changes.

