# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

YoseTsumego / onlymove.net — a Go (Baduk/Weiqi) tsumego pipeline. KataGo generates problems and rates them; the rated set is packaged into zip archives served by a Cloudflare-hosted PWA. There are four cooperating subsystems:

1. **`pipeline/`** — six executable scripts run in order (see "Pipeline commands" below). Each is a standalone `python -m pipeline.<step>` entrypoint.
2. **`utils/`** — shared logic: `KatagoController` (subprocess + thread-safe query dispatcher), `KatagoQueries` (analysis-engine wrapper), `ProblemFinder` (move-difficulty heuristics), `add_variations` (SGF tree construction), `models` (KataGo network downloads), `configurations` (global knobs).
3. **`pwa/`** — vanilla-JS ESM PWA built with esbuild, deployed as a Cloudflare Worker. The Worker entrypoint is `pwa/src/index.js` (handles `/report` POSTs into D1 and an SPA fallback for navigations); static assets ship via the `[assets]` binding in `wrangler.toml`. D1 binding `DB` → database `tsumego-reports`.
4. **`analysis_server/`** — a tiny HTTP server wrapping KataGo for the PWA's analysis fallback when SGFs lack embedded win-rate data.

Plus `auxiliary/` (one-off diagnostics, legacy code; not part of the main flow) and `vast_orchestration.py` (rents Vast.ai GPU instances to run the heavy pipeline steps remotely).

## Pipeline commands

Always invoke pipeline modules through the venv's Python so PYTHONPATH resolves correctly:

```bash
./.venv/bin/python3 -m pipeline.fetch_database        # pull D1 attempts via wrangler
./.venv/bin/python3 -m pipeline.tune_engine           # tune nnMaxBatchSize / numAnalysisThreads in bots/analysis.cfg
./.venv/bin/python3 -m pipeline.generate_tsumegos     # long-running generator → tsumegos/<size>/
./.venv/bin/python3 -m pipeline.benchmark_bots        # bots solve generated tsumegos → attempts/<size>/
./.venv/bin/python3 -m pipeline.rank_tsumegos_and_bots  # PyTorch mixture-of-sigmoids rating model → ratings/<size>/
./.venv/bin/python3 -m pipeline.extract_activations   # KataGo trunk activations per problem → activations/<size>/<id>.npy (GPU; resumable)
./.venv/bin/python3 -m pipeline.build_embedding       # rating-residualized PCA over activations → embeddings/<size>/
./.venv/bin/python3 -m pipeline.package_for_pwa       # zip + lookup.txt + embedding_meta.json → pwa/tsumegos/<size>/
```

Run on a remote Vast.ai GPU instead: `python vast_orchestration.py` (no args prints subcommands; only `tune`, `generate`, `benchmark` are remoted — the rest stay local).

## PWA commands

```bash
cd pwa
npm install
npm run build      # esbuild → pwa/dist/
npm run dev        # build + wrangler dev → http://localhost:8787 (use this for local testing)
npm run deploy     # build + wrangler deploy
```

Local testing must go through `npm run dev`: the Worker runs `/report` and the SPA fallback (URL-based size hints), so opening the static bundle alone will misbehave. Deployment is **manual** — there is no git-triggered auto-deploy.

## Architecture notes that span multiple files

**Tsumego identity is symmetry-aware and single-solution.** Filenames come from `get_orientation_invariant_hash` (`pipeline/generate_tsumegos.py:154`) — SHA-256 taken over all 8 D4 transforms of the board, with `min()` picking a canonical orientation. The dataset therefore cannot contain two problems that are rotations or reflections of each other; `tsumego_id` is the symmetry-canonical hash truncated to 8 hex chars. By generator design every problem has exactly one correct first move (the `solution` returned by `ProblemFinder.create_problem`); downstream code can rely on the "one canonical answer per problem" invariant.

**Board size — two supported modes: 9x9 and 13x13.** Both ship in production; the PWA picks one from the URL path (`/9` or `/13`) via the SPA fallback in `pwa/src/index.js`, and the D1 `report` endpoint records `size` per attempt. On the **pipeline side**, the size is a config-file switch: edit `BOARD_SIZE` in `utils/configurations.py` and re-run the pipeline — there is no in-process multi-size support, no CLI flag, and no automatic dual-size run. Artifact directories are keyed by size: `tsumegos/<size>/`, `attempts/<size>/`, `ratings/<size>/`, `pwa/tsumegos/<size>/`. To refresh both sizes, run the full pipeline once with `BOARD_SIZE = 9`, then again with `BOARD_SIZE = 13`.

**KataGo binary selection.** `utils/katago_controller.py` reads `KATAGO_PATH` from env, defaulting to `bots/katago` (OpenCL build). `vast_orchestration.py` uploads `bots/katago_cuda` and sets `KATAGO_PATH=bots/katago_cuda` on the remote. Don't hardcode the binary path; rely on the env var.

**`bots/analysis.cfg` is hardware-tuned in place.** `pipeline.tune_engine` rewrites `numAnalysisThreads` and `nnMaxBatchSize` for the current GPU. `vast_orchestration.py`'s `setup` ships the local cfg once but `update` deliberately skips it so remote tuning isn't clobbered. If you edit the cfg, expect the tuner to overwrite your changes.

**`KatagoController` is thread-safe by design.** A single master reader thread fans responses out via per-query `threading.Event`s keyed on query IDs. `send_query` is blocking from the caller's perspective. Multiple worker threads in `generate_tsumegos.py` and `benchmark_bots.py` rely on this.

**Models live outside git.** `models/` is gitignored. `utils/models.get_model(filename)` lazily downloads from `media.katagotraining.org` (or GitHub for 9x9 / human models) on first use. The default model is `MODEL_FULL` (`kata1-b18c384nbt-...`).

**PWA SGF format carries embedded analysis.** Each branch comment in a tsumego SGF is `"<winrate>%/<score>"` (e.g., `"86.4%/3.1"`). Client-side analysis (`pwa/js/sgf.js`) reads this directly; only when missing does it call the server (`pwa/js/api.js` → `analysis_server/server.py`). Don't strip these comments when manipulating SGFs.

**PWA state is centralized.** `pwa/js/state.js` holds the single mutable object; all changes go through exported setters. `state.appMode` switches between `'solving'` (game-logic.js), `'analyzing'` (analysis.js), and `'linked'` (rank-frozen URL-shared problems). Click handling in `board.js` routes by `appMode`.

**Feature-embedding recommender (per-component weakness mining).** `pipeline/extract_activations.py` runs a vendored PyTorch copy of KataGo b18 (`utils/katago_vendor/`, bootstrapped via `utils/katago_torch.py`) and stores the post-skip-add output of all 18 residual blocks at the correct-move cell — 18 × 384 = 6912 floats per problem in `activations/<size>/<id>.npy`. `pipeline/build_embedding.py` fits a **joint** PCA across all configured sizes (read activations from each, residualize per-feature on `[rating, size_is_13]` jointly, z-score, randomized SVD with K=64, then standardize component scores to unit variance). Output to `embeddings/joint/{pca.npz, per_problem.csv, explained_variance.csv, extreme_problems.json, labels.json}`. `per_problem.csv` carries `size` so packaging filters by it. `labels.json` is a manually-curated map of PC index → `{positive, negative}` human labels. **Labels survive PCA refits**: on each `build_embedding` run, prior `pca.npz` loadings are snapshotted before being overwritten; after the new fit, each previously labeled PC is matched to the new PC with maximum |cosine similarity| in the original 6912-dim feature space (above 0.85 threshold). Negative cosine flips the axis, so `positive`/`negative` label sides swap. Survivors carry `auto_mapped_from` and `auto_mapped_cos` for traceability; below-threshold matches reset to null. `pipeline/sync_labels.py` is a fast sync that pushes a hand-edited `labels.json` into the joint `pwa/tsumegos/embedding_meta.json` without rezipping. `pipeline/package_for_pwa.py` joins per-problem PCA scores into the per-zip `metadata.json` (`embedding` field, length K=64) and writes a single joint `pwa/tsumegos/embedding_meta.json` with K, embedding_version, and labels (the PWA only consumes those — variance curve, extreme problems, kurtosis, etc. live in `embeddings/joint/` locally as inspection artifacts and aren't shipped). PWA-side: `pwa/js/weakness.js` trains a logistic regression `P(correct) = σ(intercept + γ·rating + Σβ_k·embedding_k)` on the user's localStorage attempt log (with embeddings copied into each entry); `problem-loader.js`'s `selectProblem` weights candidates by `quality · exp(T · Σ(-β_k)·embedding_k)` so a calibrated user gets pushed toward weak directions and away from strong ones. Cold start (≤20 attempts with embeddings): `weaknessScore` returns 0, weighting collapses to pure quality — matches prior behavior. `pwa/js/embedding-backfill.js` runs on app load: fetches the joint `embedding_meta.json` once (version + labels are joint), then iterates per size for the wipe/refill logic — filling `embedding` on legacy answerHistory entries so returning users get the recommender calibrated immediately. Refits auto-invalidate: `build_embedding.py` writes a content-hash `embedding_version` into `pca.npz` (computed over the projection-relevant arrays — feature mask/means/stds, OLS coefs, PCA components/mean/stds, ICA unmixing/stds), `package_for_pwa.py` surfaces it in `pwa/tsumegos/embedding_meta.json`, and the PWA wipes stale embeddings per size whenever the localStorage-cached version differs. No manual JS edit needed on refit. Extraction needs ~8 GPU minutes per 1k problems on a Quadro M2200; remote it via Vast.ai for full-corpus runs.

**KataGo PyTorch loader requires `.ckpt`, not `.bin.gz`.** `models/kata1-b18c384nbt-...bin.gz` is the binary export the C++ analysis engine reads — not loadable via `torch.load`. The feature-embedding pipeline expects `models/kata1-b18c384nbt-s9996604416-d4316597426/model.ckpt` (extracted from the corresponding `.zip` checkpoint distribution). Don't point `extract_activations.py --ckpt` at a `.bin.gz`; KataGo's training code only goes PyTorch → binary, not the other way.

## Conventions to be aware of

- `tests/` exists but is empty — there is no test runner configured. Don't add a "run the tests" step to claims of completion unless tests are actually added.
- Python code uses absolute imports rooted at the repo (e.g., `from utils.configurations import ...`). Pipeline scripts append `PROJECT_ROOT` to `sys.path` defensively.
- KataGo coordinates skip the letter `I` (so cols are `ABCDEFGHJKLMNOPQRST`). SGF coordinates do not. Helper conversions live in `utils/katago_queries.py` (`katago_to_sgf_coords`) and `pipeline/benchmark_bots.py` (`sgf_to_gtp`). Prefer reusing them over rewriting coordinate math.
- `pwa/README.md` section 7 has additional LLM-targeted instructions specific to PWA work (build after changes, conservative scope, investigate before fixing). Read it before substantive PWA edits.
