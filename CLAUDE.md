# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

YoseTsumego / onlymove.net — a Go (Baduk/Weiqi) tsumego pipeline. KataGo generates problems and rates them; the rated set is packaged into zip archives served by a Cloudflare-hosted PWA. Four cooperating subsystems:

1. **`pipeline/`** — thirteen standalone `python -m pipeline.<step>` entrypoints run in order. `package_for_pwa` is the single writer of `pwa/tsumegos/`; there are no side-channel patch scripts.
2. **`utils/`** — shared logic, **split into two subpackages that import nothing from each other** (worth keeping true). `utils/katago/` is the Go/KataGo side — `controller` and `queries` (the engine subprocess and its thread-safe dispatcher), `problem_finder`, `add_variations` (SGF tree construction), `coords`, `models` (network downloads), `board_render`, `sgf_parsing`, and `torch_model`, which bootstraps `utils/katago/vendor/`, the vendored PyTorch KataGo (that top-level `katago` package is the VENDOR's, distinct from this `utils.katago`; they do not shadow each other). `utils/rating/` is the statistical side — features, the trunk's layer stack (`mirt_net`, stated ONCE: `train_mirt` fits it and `write_mirt_per_cell` rebuilds it to project every cell, and a stack matching by shape while differing in its activation loads cleanly and projects through the wrong function), `ratings_io` (the `ratings/` artifacts + curve reader), `fit_rank_curves`, `rank_surveys`, `rank_curve_shape`, `motif_rules`, `se_floor`, `tail_agreement`, `model_constants`. `residual_axes` and `slope_calibration` are retired two-stage machinery with no production importer — only `auxiliary/rating_dev/` reads them. `utils/constants.py` holds the few values both sides need — `ALL_BOARD_SIZES`, `KOMI`, `RULES`, `DEFAULT_MODEL`, the debug flag.
3. **`pwa/`** — vanilla-JS ESM PWA built with esbuild, deployed as a Cloudflare Worker (`pwa/src/index.js`), with a D1 binding `DB` → database `tsumego-reports`. Schema is `pwa/schema.sql` (idempotent `CREATE … IF NOT EXISTS`); apply new tables/indexes to the remote D1 with `npx wrangler d1 execute tsumego-reports --remote --file=schema.sql`. Routes, tables and the client architecture are in `docs/` — see below.
4. **`analysis_server/`** — a tiny HTTP server wrapping KataGo for the PWA's analysis fallback when SGFs lack embedded win-rate data. `server.py` runs on the cloud box (stdlib + sgfmill only — deployed with no repo beside it); `worker.py` runs on a fast machine and takes that box's work over while it is up.

Plus `auxiliary/` (diagnostics, surgery scripts, the rating-model research surface; not part of the main flow) and `vast_orchestration.py` (rents Vast.ai GPU instances for the heavy generation steps).

### Where the detail lives

This file is the always-loaded operating layer: what to run, what breaks, and what not to re-try. Derivations, measurement trails and protocol detail live in `docs/`. **Keep the rule here and the evidence there** — if a passage only tells you what the system *used* to do, it belongs in neither.

| Document | Holds |
| --- | --- |
| **`docs/contract.md`** | **The editing contract for this file and everything under `docs/`. Read it before writing documentation here.** |
| `docs/overview.md` | What each subsystem is for, who owns what state, what a change in one place implies elsewhere. No algorithms. Start here. |
| `docs/methodology.md` | Ratings, ranks and motifs end to end (§1 construction, §2 client consumption, §3 modules/artifacts/constants). |
| `docs/mirt_benchmark.md` | **Development**: the fixed reference point for the joint-MIRT work — DB hash, held-out set hashes, the two-fold table, survey rho, reproduction commands. |
| `docs/pipeline_operations.md` | Per-step incrementality, abort semantics, `--force` flags, long-run discipline, artifact policy. |
| `docs/corpus_and_engine.md` | Corpus invariants, SGF contracts, the three coordinate spaces, KataGo integration, the analysis service. |
| `docs/motif_axes_workflow.md` | The axis shipping contract, tail agreement, the naming workflow, the published embedding layout. |
| `docs/pwa_client.md` | Client state, storage, re-derivation generations, the three ranks, the serve draw, recall, motif readings. |
| `docs/client_sync_and_cache.md` | Account sync, Google sign-in, precache ownership, served-data cache freshness. |
| `docs/leaderboard_and_community_stats.md` | The `/leaderboard` resolver and the Community stats aggregation. |
| `docs/worker_operations.md` | Write-endpoint guards, the maintenance freeze, `pipeline.fetch_database`. |
| `docs/db_surgery_runbook.md` | The freeze-and-apply procedure for snapshot-relative DB operations. |

**Before editing any of them, read `docs/contract.md`** — it holds the three tests every passage has to pass, the no-changelog rule, and where a new fact belongs. Two per-document obligations reach beyond it: **`methodology.md` is stricter still** (self-contained, and its method chapters name no code, because they are published on the site), so read its own "Scope" section before touching it and update it in the same pass as any rating/rank/motif change; and **`mirt_benchmark.md` needs its provenance pinned** before any new run is compared against it, since a `fetch_database` changes the holdout draw and voids the comparison.

## Pipeline commands

Always invoke pipeline modules through the venv's Python so PYTHONPATH resolves correctly:

```bash
./.venv/bin/python3 -m pipeline.fetch_database            # incremental pull of new tsumego_reports rows + small tables whole → db/d1.{sqlite,sql} + local D1. --full re-downloads
./.venv/bin/python3 -m pipeline.tune_engine <size>        # tune bots/analysis.cfg for this GPU (size ∈ {9,13}, mandatory)
./.venv/bin/python3 -m pipeline.generate_tsumegos <size>  # long-running generator → tsumegos/<size>/
./.venv/bin/python3 -m pipeline.benchmark_bots <size>     # bots solve generated tsumegos → attempts/<size>/
./.venv/bin/python3 -m pipeline.extract_activations       # KataGo trunk activations at every legal cell → activations/<size>/<id>.npz (GPU; resumable)
./.venv/bin/python3 -m pipeline.extract_ownership_swing   # ownership swing per problem → activations/_signal_cache/ownership_swing.csv (GPU; resumable). MANDATORY, not diagnostic
./.venv/bin/python3 -m pipeline.cache_rating_features     # both roll-up caches the trainer reads (move evals + correct-cell vectors). Incremental, reconciles both ways
./.venv/bin/python3 -m pipeline.build_motif_proxies       # rule verdicts per problem → axes/motif_proxies.npz. A rule/threshold change needs --force
./.venv/bin/python3 -m pipeline.train_mirt                # single-stage rating + K skill axes in ONE fit → ratings/ + axes/{per_problem.csv,axes.npz}
./.venv/bin/python3 -m pipeline.write_mirt_per_cell       # per-legal-cell axis vectors → axes/per_cell/<size>/<tid>.npz. Resumable
./.venv/bin/python3 -m pipeline.map_rating_to_rank        # → ratings/{rating_to_rank.json, rank_map.json}. Backend; package_for_pwa publishes it
./.venv/bin/python3 -m pipeline.fit_population_constants  # → ratings/population_constants.json (retention, time bias, rank prior). Backend
./.venv/bin/python3 -m pipeline.package_for_pwa           # SINGLE writer of pwa/tsumegos/. Change-detects per size; --force to override
```

Remote: `python vast_orchestration.py` (no args prints subcommands; only `tune`, `generate`, `benchmark` are remoted — the rest stay local).

**Run order is a hard data dependency, not a convention** — `methodology.md` §3.1 spells out each edge. Two steps mislead: `map_rating_to_rank` and `fit_population_constants` are board-independent and need no GPU, but **board-independent is not detachable** and both sit INSIDE the sequence. `fit_population_constants` reads nothing from `axes/` — it fits no per-axis quantity, so an axis refit does not require re-running it.

**Every step aborts on a missing or unreadable per-problem input rather than skipping the row**, and every id-keyed cache reconciles in BOTH directions ("up to date" means the sets are EQUAL, not that there is nothing new to add). The reasoning and the per-step detail are in `docs/pipeline_operations.md`.

**Several steps run for HOURS, none checkpoint, and a naively launched run WILL be killed part-way:**

- **A shell that backgrounds with `&` and then exits takes the child down with it** (SIGTERM, exit 143) — including a wrapper that backgrounds the real work and returns. Launch detached: `setsid nohup ./.venv/bin/python3 -u -m pipeline.<step> > /tmp/<step>.log 2>&1 < /dev/null &`.
- **`timeout N` kills at N with exit 124 and no partial result.** A too-short guess is indistinguishable from a crash after the fact.
- **Piping through `tail`/`head` buffers until exit**, so a healthy run looks hung and a dead one looks alive. Use `-u`, redirect to a file, poll the file.

`train_mirt` is up to a couple of hours on a FIXED 3000-epoch budget — the printed stop split is a record, not a stopping rule, and it improves in bursts separated by flat stretches, so **do not read a long run as a hung one**.

**The steps that write `ratings/`, `activations/`, `axes/` and `pwa/tsumegos/` overwrite IN PLACE — do NOT copy them aside before a refit.** They are derived; the inputs (corpus, attempt log, committed code) are what's kept, and a bad fit is reverted by re-running the step. The exception is two TRACKED files under `axes/` — `labels.json` and `label_anchors.npz` — which nothing can regenerate. `motif_proxies.npz` and `axes.npz` are NOT tracked despite being a fit's input and output; see `docs/pipeline_operations.md`.

## PWA commands

```bash
cd pwa
npm install
npm run build      # esbuild → pwa/dist/
npm run dev        # build + wrangler dev → http://localhost:8787 (use this for local testing)
npm run deploy     # build + wrangler deploy, then verifies (exit 2 = deployed but checks failed)
npm run status     # probe the LIVE Worker on demand (auxiliary/check_live_status.mjs)
```

Local testing must go through `npm run dev`: the Worker runs `/report` and the SPA fallback (URL-based size hints), so opening the static bundle alone will misbehave. Deployment is **manual** — there is no git-triggered auto-deploy.

**Two live routes return 404 ON PURPOSE, and probing them by hand reads as an outage — use `npm run status` rather than a bare `curl`.** (a) `/`, `/9` and `/13` have no file behind them: the SPA fallback rewrites them **only for a navigation-shaped request** (`sec-fetch-dest: document`, or an `Accept` naming text/html), because an asset-style miss must keep 404ing so a stale service worker re-fetches instead of receiving HTML where JS is expected — and `curl` defaults to `Accept: */*`. (b) The `INLINED_DATA_FILES` (`build.mjs`) are compiled into the bundle, so `/tsumegos/{rating_to_rank,rank_map,population_constants,se_floor,embedding_meta}.json` 404 — **a 200 there is the real regression**, since it would let a client fetch a curve under a fixed name with no `?v=` and pin itself to a stale one. `check_live_status.mjs` asserts both directions plus the freeze state (`--expect-frozen` / `--expect-unfrozen`), and **it retries**: a deploy propagates over seconds, so an unfreeze still reporting `maintenance` on the first probe is usually propagation, not a failed unfreeze.

## The rating + axes fit (`pipeline.train_mirt`)

**One model produces the rating and the K skill axes together**, a compensatory M4PL where the axes are factors 2…K+1 and the rating is factor 1:

`P = c_p + (1 − c_p − λ)·σ( a_p·(θ_u − b_p) + Σ_k g_k(x_p)·v_u[k] )`

**ONE TRUNK feeds every head, axes included** — a `Linear`'s output rows are independent, so this is exactly a shared head emitting `(b, a, c, g_1..g_K)`. The axis path is COUPLED, so the trunk learns what the AXES need and not only what the rating needs; detaching measures the same but is feature reuse rather than a joint fit.

Method: `methodology.md` §1.5–§1.14. Evidence for every constant, ablation and sweep: `mirt_benchmark.md`. Five things bind from outside those:

- **`train_mirt` aborts without the motif proxies and `axes/labels.json`** — it derives no directions of its own. It **retires any existing `axes/per_cell/`** by renaming it, because this fit's `per_problem.csv` beside the previous fit's `per_cell/` ships a shard whose two embedding sources disagree, and both files are individually well-formed so nothing downstream can catch it.
- **It does not select K.** K equals the number of `components` in `labels.json` — the SHIPPING decision, one axis per taught rule, not a loss curve. `K_AXES` (`utils/rating/model_constants.py`) must match it, and the failure is one-sided: abort when `components` is shorter, silent TRUNCATION when longer. See `docs/motif_axes_workflow.md`.
- **`ORTHO_RATING` keeps the axes off the shipped triple — rating, slope and floor — and it is scored PER TAIL** (`tail_weight(+g)` and `tail_weight(−g)` penalised separately), **not per axis.** A whole-axis correlation is a linear statistic and cannot see a symmetric dependence, so a fit recorded as rating-orthogonal on that basis can carry laden tails at both ends. Penalising difficulty alone only displaces the leak into slope or floor. Expect the resulting axes to be **unfamiliar** — forbidding the correlation forbids the entangled motifs and returns different ones.
- **Read the combined column, PAIRED.** `axis adds` rises with K whether or not the model predicts better, because more axes mechanically absorb more residual — and **axis value is a property of the holdout draw**: over four independent draws it ran +3.59 to −1.44 mnat, two of four negative. **Never quote a single-draw axis-adds figure as the effect.** Comparing ARMS wants the draw PINNED (paired); estimating a MAGNITUDE wants it MOVING, and a figure produced under the first must not be quoted as though it came from the second.
- **`slope_scale` is 1.0 and slopes ship as fitted** — no post-hoc multiplier.

## Invariants that span multiple files

### Corpus and generation

- **Tsumego identity is symmetry-aware and single-solution.** SHA-256 over all 8 D4 transforms, `min()` canonical, truncated to 8 hex — so the set cannot hold two problems that are rotations or reflections of each other. **The answer is `variation_nodes[0]` and generation ENFORCES it** (`promote_winning_first_move`): the client picks by evaluation and is order-blind, but every backend stage reads it POSITIONALLY, so a misordered file feeds a LOSING move's activations into the ratings and the shipped axis vector while still playing correctly in the app. `auxiliary/check_tsumego_sequences.py` re-verifies.
- **`tsumegos/retired_ids.txt` ids are treated as taken.** Removing a problem keeps its `tsumego_reports` rows, so a reused id silently inherits a stranger's history — the one failure mode that raises nothing anywhere and cannot be untangled afterwards.
- **Branches read to a conclusion are marked `TE[1]`; the client must never deduce them again.** The retired deduction (a single pass anywhere in the subtree) is the whole cause of two corpus purges — **don't reintroduce any weakened form of it.** The mark only narrows the pool; **correctness still reads the unrestricted `findBestMoveFromChildren`**, and at the decision node only the ANSWER may be marked.
- **SGF branch comments are `"<winrate>%/<score>"`** and client-side analysis reads them directly. Don't strip them when manipulating SGFs.
- **A problem exists in three orderings**: the activation npz (board-scan, a SUPERSET of the SGF's cells), the SGF root variations (pruned, plus a `pass` the npz never embeds), and the published per-cell bin (SGF order, with an all-zeros row for the pass). **Pick the correct row by `correct_idx`, never positionally; map cells by `(x, y)`, never by index, when crossing npz↔SGF.**
- Full detail, plus `auxiliary/audit_solution_lines.py` and the pruning approach it measured-rejects: `docs/corpus_and_engine.md`.

### KataGo

- **Don't hardcode the binary path** — `utils/katago/controller.py` reads `KATAGO_PATH`, defaulting to `bots/katago` (OpenCL).
- **`bots/analysis.cfg` is hardware-tuned in place** by `pipeline.tune_engine`; expect hand edits to be overwritten.
- **`requireMaxBoardSize = true` in `bots/analysis.cfg` belongs to THIS repo's config only.** It turns the NN-buffer maximum into an exact requirement, and an off-size query kills KataGo's async bot thread with a message on stderr and NO response on stdout — so the caller waits out its full query timeout on a request that can never be answered. The pipeline never notices (single-size runs); `worker.py` passes `false` and runs one engine sized to the largest board.
- **`KatagoController` is thread-safe by design** — one master reader thread fans responses out via per-query events, so `send_query` blocks from the caller's view.
- **Models live outside git** and download lazily. **The PyTorch loader needs a `.ckpt`, not the `.bin.gz`** — KataGo converts PyTorch → binary and not back, so one is not derivable from the other.

### Backend fits and publishing

- **`package_for_pwa` is the single writer of `pwa/tsumegos/`.** A labels-only or curve-only change publishes by re-running it (change detection skips the shard rebuild). **A change to the packaging FORMAT is not captured by the input hash — run `--force` once after such a code change.**
- **Packaging is rank-agnostic**: it ships the raw rating, the raw slope and the floor, no rank, and shard bounds are raw too. The client does all conversion. **`rating` means raw everywhere** — don't reintroduce a baked rank into `metadata.json`.
- **`M` maps to a latent CONSENSUS rank; every server INCLUDING EGF is a coupling off it.** Ratings are time-adjusted throughout the mapping, so **any accuracy statistic against a DECLARED server rank must be computed on the TIME-ADJUSTED rating** — a raw comparison has the two sides in different units and fails SILENTLY, the wrong number still being a plausible one. `auxiliary/check_mapped_vs_declared_rank.py` emits both and they are NOT interchangeable: only **`err_server`** (time-adjusted, forward-mapped into that server's own units) is quotable, while **`err_internal`** (raw, pooled internal units) exists in the `--csv` output alone; and the per-server user-weighted rho it does not print is defined in `auxiliary/rating_dev/validate_ratings.py`. **These figures are PUBLISHED in `pwa/about.html` in FOUR languages (en/zh/ja/ko)** — re-measure and update all four in the same pass as a rank-map change.
- **Axis labels are optional to ship and an unnamed axis is silent, not broken** — it still steers selection and carries its share of the reading; the client declines to print it, and there is deliberately no placeholder label. Naming workflow, tail-agreement tooling, and the calibration that says **agreement below ~0.75 does not preserve a name**: `docs/motif_axes_workflow.md`.

### Client

- **PWA state is centralized** in `pwa/js/state.js`; all changes go through exported setters. **Linked-mode attempts are recorded NOWHERE** — `sendReport` and `updateUserStats` both early-return on the canonical `linked` flag, and the two guards must stay in lockstep on that one flag.
- **`/9` and `/13` are entry-only size HINTS, not persistent addresses** — read once at boot, then `cleanUrl()` collapses the path. Linked problems are the deliberate exception, which is what makes any reload re-enter the same shared problem instead of dropping to the standing challenge.
- **Rank, stats and the learned models merge across sizes; histories, curves and problem pools stay per-size.**
- **`reanchorRank` is the SINGLE writer of the three rank slots.** `userRank` (the posterior mean) is what everything CALIBRATED reads — never feed a shifted value here or the calibration loop closes on it; `displayRank` is what the user is told; `serveRank` is what selection targets. **Serving must not follow the display down** (it starves the estimator), and **the serve-gap cap must not go back to a constant** — a constant doesn't limit the offset, it INVERTS it.
- **The rank is never clamped** — `applyM`/`interpExtrap` extrapolate past both end knots, so treat it as an unbounded continuous coordinate. Clamping lives only where LABELS are built. **`pwa/shared/rank-math.js` owns the one internal rank scale** (0 = the 1k/1d boundary, one unit per rank, integers are band EDGES) and is imported by both the Worker and the client — **don't make a local copy**. Any change to that scale must be label-preserving by construction and shipped behind a state-epoch bump.
- **The SE floor is a SHIPPED CORPUS CURVE (`se_floor.json`), not a runtime scan.** Its geometry is single-sourced in `utils/rating/se_floor.py` across the step that picks the grid and the step that emits the curve. **Do not go back to measuring it per-reader** — the dominant cost is the reader that holds nothing, which concedes the floor to zero and lands the shown rank a whole band low.
- **A stored per-attempt field is governed by its generation's DEVICE-LEVEL version — there is no per-row stamp.** Two conditions hold it up: nothing writes a derived field while its own build's generation and the device's disagree, and the re-derive reaches EVERY holder. That second one is an **enumeration** — any new place that stores a problem object must be added to it, and it has been wrong three times.
- **Every long re-derive pass must checkpoint in batches, not once at the end.** A version stamp written last marks the work DONE but says nothing about work already paid for, and these passes are network walks that get killed routinely. The rest of the chain (`/sync`, `evictColdTail`, the cold-tail passes, `pruneRemovedProblems`, the account switch) is already interruption-safe — do not "fix" it.
- **There is NO shipped weakness constant of any kind** — every motif reading stands against a neutral point of the user's own: the two error-composition readings against the per-problem AVAILABILITY NULL (uniform over the alternatives the board offered, computed client-side from the per-cell bin — parameter-free by design), the strength reading against zero residual effect. **Don't reintroduce a population baseline to "correct" any of them** (the two things it would correct for are exactly what make the reading about the user's RANK rather than their play), and **don't put the tendency test back on a fixed 50% null** — that reports the problem MENU as a property of the player (measured: it flagged tenuki over-reach for ~94% of users; the menu explains essentially all of it).
- **`MOTIF_Z` is ONE constant gating three things** (problem-motif display, the serve flag, the stats admission bar), locked so they cannot drift apart — retuning it anywhere retunes all three.
- **Candidates come from the unbroken run of loaded shards containing the serve rank's own band — never `state.allProblems` at large.** The serve draw is the flagged categories PLUS one NEUTRAL no-criterion slot, ALWAYS. Don't gate the motif slots on rank convergence, don't sample proportional to any weight, and don't argmax the information slot (it is itself a ko selector in the SDK/low-dan range).
- **`stabilityDaysForRepeat` has no upper clamp — don't reintroduce one.** Graduation IS the interval outgrowing the working window; a ceiling turns it into a perpetual recycle and serving drifts toward all recalls.
- All of the above in full, with the measurements: `docs/pwa_client.md`.

### Worker, sync and D1

- **The userId IS the account credential; history syncs by append-only cursor.** The cursor is the AUTOINCREMENT `id` (`sinceId`), **never `created_at`**, because backfill inserts historical event times at the id frontier. A device's acked history is a 1:1 **mirror** of the server's rows keyed by `serverId`, so a complete from-0 pull converges in both directions and there is no timestamp dedup anywhere. **`idx_tsumego_reports_user_id` must stay in place.** The client-side daily full-pull cap is display only — the real one is server-side in `src/handlers/sync.js`, so the two must change together. Protocol: `docs/client_sync_and_cache.md`.
- **Google sign-in is a LOOKUP KEY ONTO AN ACCOUNT CODE, never a second kind of account.** A Google login ends in the SAME `switchAccount` a typed code goes through. `pwa/shared/google-auth.js` holds the client id ONCE, imported by the Worker (which pins every ID token's `aud` to it) and the browser — that pin is the only thing stopping a token minted for any OTHER site from logging its bearer in here. **EMPTY means the feature is off everywhere with no second switch.** `http://localhost:8787` belongs on a SEPARATE dev client, never the production one. `POST /google/signin` RESOLVES and writes nothing; only the opaque subject id is stored.
- **Never resolve a precached asset through `caches.match()`** — it searches caches in CREATION order, which puts the runtime cache ahead of the precache from the first deploy onward, and a badly-stored copy is then served for the life of the install.
- **Served data files are cache-busted by a content `data_version`, not `APP_VERSION`.** `pwa/tsumegos/` is gitignored, so a git-HEAD version is blind to a `package_for_pwa` re-ship from the same commit: any fixed-name data fetch must carry `?v=${DATA_VERSION}`, and content-hash-named files need nothing. **`data_version` (cache freshness) is deliberately separate from `ratings_version`/`embedding_version` (the re-derive gate).**
- **Write endpoints are guarded by three cheap layers upstream of D1, plus a maintenance freeze.** Validation is **deliberately format-only**, so any deploy smoke test must POST a real 8-hex problem id. **`MAINTENANCE_MODE` is a wrangler SECRET, not a `[vars]` entry** — a var of that name would collide with it and win on the next deploy, silently thawing an open window. A rollout lands UNEVENLY, so a freeze counts as confirmed only when several consecutive probes agree; the authoritative check that no write slipped through is the before/after `MAX(id)` comparison, never the probe alone. Never put the flag behind a per-request D1 or KV lookup. `docs/worker_operations.md`, `docs/db_surgery_runbook.md`.
- **D1 bills rows read**, which shapes two things. `pipeline.fetch_database` is incremental by `id` — **stop `npm run dev` before fetching** (it overwrites the miniflare D1 file), and **re-run rather than re-authenticate** on a spurious `code: 7403`, which is not a credentials problem. And the Community stats are ONE shared 7-day window query bucketed in JS, so **`idx_tsumego_reports_created` must stay indexed**; don't split it back into per-metric queries, and don't take rank off that window scan — what it measures is the difficulty a user was SERVED.
- **A player LINKS THEMSELVES on the leaderboard and NOTHING VERIFIES THE CLAIM.** Exclusivity and reversibility stand in for verification, both server-enforced. **Every path that removes or replaces a binding must delete that account's heartbeat rows with it** — they are keyed by `user_id`, not by player id, so one left behind reports the PREVIOUS player's verified rating under the next name linked. The row stores the SHOWN rank, so est-vs-verified on that table is **not** a calibration measurement. `auxiliary/update_leaderboard_ranks.mjs` runs under the freeze and `--remote` opens and closes its own window; a post-re-rate refresh is `package_for_pwa` → `npm run deploy` → that script, in that order. `docs/leaderboard_and_community_stats.md`.

## Conventions to be aware of

- **There are no tests and no test runner** — no `tests/` directory, no CI. Don't add a "run the tests" step to claims of completion unless tests are actually added, and don't assume a change is covered by anything automated. Verification is manual: for pipeline changes, run the step and check its printed diagnostics; for PWA changes, `npm run dev` and exercise the path in the browser.
- Python code uses absolute imports rooted at the repo (e.g. `from utils.constants import ...`). Pipeline scripts append `PROJECT_ROOT` to `sys.path` defensively.
- **KataGo coordinates skip the letter `I`; SGF coordinates do not. Every conversion lives in `utils/katago/coords.py`** — reuse those helpers rather than rewriting coordinate math, and add a new one there rather than beside its caller.
- `pwa/README.md` has additional context on the PWA's overall structure (overview, tech stack, project layout, architecture, dev workflow). Read it before substantive PWA edits.
- **For local DB analysis, prefer the binary `db/d1.sqlite` over the raw `db/d1.sql` dump.** Both come from `pipeline.fetch_database`, both are gitignored, and it keeps them in sync on every run, so you normally don't import by hand. If you ever rebuild the sqlite manually: `{ echo "BEGIN;"; cat db/d1.sql; echo "COMMIT;"; } | sqlite3 db/d1.sqlite` — the wrapper matters, because a full D1 export emits one un-batched INSERT per row and without a transaction sqlite fsyncs per row. **Never pipe the import through `head`/`tail`**: it SIGPIPE-kills `sqlite3` mid-import and silently leaves a partial table. Note a `--full` dump carries only tables that had rows at export time, whereas the incremental path's regenerated dump carries every table it discovered remotely.
- **Many artifact directories hold 100k+ files**, and **shell globs like `ls dir/*.npz` silently expand past `ARG_MAX` and report 0** — a verified false negative on `activations/9/`, not a theoretical one. Count with `find <dir> -maxdepth 1 -name '*.ext' -printf . | wc -c` and iterate with `find … -exec` or `xargs -0`. Treat anything under `tsumegos/`, `activations/`, `axes/`, `attempts/`, `ratings/` or `pwa/tsumegos/<size>/` as "may exceed argv" by default.
