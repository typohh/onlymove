# Pipeline operations: incrementality, failure modes, and running long steps

<!-- Editing rules: docs/contract.md. Present tense, no changelog, rules here and evidence with them. -->

What each pipeline step reads and writes, what it does when it is re-run, and what it does when an input is missing or unreadable. The **run order and the data-dependency edges** are `methodology.md` §3.1 and are not repeated here; the **how-to-run walkthrough** is the root `README.md`. CLAUDE.md carries the one-line form of each rule below and points here.

## The governing rule: abort, never skip

Every step that reads a per-problem input treats a **missing or unreadable row as a hard failure**, not as a row to log and move past. The reason is uniform across the pipeline and worth stating once: a silently dropped problem is invisible downstream. It keeps its place in whichever id list the step derives, it acquires or retains numbers nothing recomputed, and no later stage can tell it apart from a problem that was processed correctly. A crash points at the step that has the problem; a skip points at nothing at all, one step later and in the wrong place.

The same reasoning inverts for **id-set reconciliation**: a step that maintains a cache keyed by tsumego id reconciles in **both** directions on every run. New ids are added, departed ids are pruned, and "up to date" means the cached id set *equals* the target set — never merely "nothing new to add". A stale set that only ever grows silently drops a corpus top-up out of the rated set, or keeps orphan rows for problems that no longer exist.

## Per-step operating notes

### `fetch_database`

Incremental by default and read-cost sensitive; full detail in `worker_operations.md`. Two things to remember at the call site: **stop `npm run dev` first** (the fetch overwrites the miniflare D1 file, and doing that under a live workerd corrupts its view), and the Cloudflare authorization check **fails spuriously on a first call** (`code: 7403`, "the given account is not valid or is not authorized") with the identical call succeeding seconds later. That is not a credentials problem and re-authenticating is the wrong response — `_run_wrangler_json` retries structured D1 errors on the `RETRY_BACKOFFS` ladder, and if a fetch still aborts this way, run it again before investigating anything else.

**`--db-dir DIR` fetches a PRIVATE snapshot instead, leaving `db/` and the dev D1 exactly as they are.** `DIR/d1.sqlite` is both the destination and the incremental fetch's id baseline, so copy the canonical one in first and the pull stays the usual handful of rows (without it the run seeds `DIR` with a `--full` export, reading every row). A private snapshot has no text consumers and no dev-D1 role, so it also skips the `db/d1.sql` regeneration — a ~60 MB write of every row — and the local-D1 sync, which is why the `npm run dev` rule above does not apply to it: it writes neither shared file. It exists because the canonical mirror is a SHARED input — other analysis is mid-flight against it, and a re-rate is reproducible only against a fixed snapshot — so a caller that merely needs rows fresher than the last fetch must not advance it. `auxiliary/update_leaderboard_ranks.mjs` takes its in-freeze mirror this way.

### `tune_engine <size>`

Rewrites `numAnalysisThreads` and `nnMaxBatchSize` in `bots/analysis.cfg` **in place** for the current GPU. If you hand-edit that file, expect the tuner to overwrite your changes. `vast_orchestration.py`'s `setup` ships the local cfg once but `update` deliberately skips it, so remote tuning is not clobbered by a local one.

### `generate_tsumegos <size>` / `benchmark_bots <size>`

Hours-long, no checkpointing, mandatory positional size. See *Running the long steps* below. Generation's corpus invariants — identity, the single-solution gate, retired ids — are in `corpus_and_engine.md`.

### `extract_activations`

GPU, resumable by file existence, iterates both sizes. Feeds both the rating features and the motif axis fit. **It aborts without `activations/encoder.npz`** — writing raw vectors into a store every reader expects to be one encoder's output would be silent, so the missing artifact fails loud; `--raw` (for an encoder refresh's fitting subset) requires its own `--out-dir` for the same reason. On resume it fingerprint-verifies every existing file in a background thread and hard-aborts on a file written under a different encoder. The pass pipelines reader threads (SGF parse + net inputs) ahead of a batched forward, slices and encoder-projects on device, and compresses through a writer pool; the encoded store is small enough (~15 GB corpus-wide) that neither disk nor writing is the budget any more. A full corpus is around two hours locally and does not need remoting. The npz row order, the `correct_idx` contract, the encoder artifact and the refresh procedure are in `corpus_and_engine.md`.

### `extract_ownership_swing`

**Mandatory, not diagnostic.** Per problem, it measures how many stones' worth of ownership the correct first move swings against the best resolved refutation, into `activations/_signal_cache/ownership_swing.csv`. Two of the shipped motif rules (`life and death`, `semeai`) are written over it, so it reaches the fit as taught directions rather than as trunk columns — the trunk's own blocks are `PRODUCTION_BLOCKS` (`methodology.md` §1.3). **That makes it no less required**: `build_motif_proxies` **aborts** on a live tsumego with no row rather than labelling a short corpus, so a top-up that skips this step stops the labelling one step later, pointing at the wrong place. An unreadable SGF aborts for the same reason.

Resumable by the `tid` column, appended and fsynced periodically. Iterates all board sizes, takes no arguments, and has no alternate evaluator — it is production-lean by design.

### `cache_rating_features`

Builds **both** aggregate caches the fit depends on, in one command so they cannot drift apart:

1. `activations/_signal_cache/move_evals_<size>.npz` — the candidate-move-eval cache, holding **both** the root `feats` array that the move block reads **and** the `cont` array that the continuation block reads. One SGF parse fills both, and a row counts as cached only when both are present, so a cache predating `cont` re-extracts whole. Neither array is in the shipped trunk layout (`methodology.md` §1.3): they serve the `"all"` comparison arm, the tightness axis analyses, and `build_alt_vec_cache`'s runner-up coordinate. The tightness motif rules read the same evaluations straight from the SGF comments, so `build_motif_proxies` does not depend on this file.
2. `activations/_signal_cache/correct_vecs_<size>.npy` + `correct_ids_<size>.json` — the correct-cell activation vectors, built first, by `utils.rating.features.build_correct_vec_cache`. `correct_ids` is **both the rated set and this step's own id set**, which is what keeps them in lockstep.

Both are incremental and reconcile in both directions (see the governing rule above). `--force` re-extracts the move features only; `--rebuild-activation-cache` additionally re-reads every activation npz, which takes hours and is for a corrupt cache rather than routine use.

**The npz set must be a subset of `tsumegos/<size>/`, asserted before any work** (`_assert_activations_subset_of_corpus`). The assertion is deliberately **one-directional**: corpus ⊋ activations is normal, since extraction is resumable and a fresh top-up has SGFs with no npz yet. An npz with **no SGF is a hard failure**, because `correct_ids` is derived from that listing and an orphan silently promotes a problem that no longer exists into the rated set — it gets features, a rating, a slope and a floor, ships nowhere (packaging works off `tsumegos/`), and nothing downstream can tell it from a real one.

A corrupt npz raises out of the worker, and a non-finite row inherited from an older cache is refused at **write** time — the temporary file is removed and the existing cache left intact. The alternative, writing a NaN row, is invisible: the id stays in `correct_ids` while `_fit_joint_pca`'s finite mask removes it from the fit, so the problem leaves the rated set with nothing anywhere recording it.

The per-problem npz artifacts are inherently incremental via file existence; only these two roll-up caches needed explicit incremental support.

### `build_motif_proxies`

Labels every problem against the motif proxy rules into `axes/motif_proxies.npz` — one boolean column per rule, plus a `rules` array so a reader never has to assume column order. These are the directions `train_mirt` teaches its axes onto.

Incremental by id and reconciling in both directions. **Every rule is a threshold on the problem's own numbers**, which is what makes that sound: a stored row can never be made wrong by a problem added later. (`opening` is an absolute prefix length per board for exactly this reason — as a corpus *percentile* it moved as the corpus grew, silently re-labelling problems that had not changed.)

**A rule or threshold change is invisible to the id key, so it needs `--force`.** The rules live in `utils/rating/motif_rules.py`, which is also the single source for what each motif is called.

### `train_mirt`

Production-only: no argparse, configuration baked as module constants, always seed 0. Writes `ratings/` and `axes/{per_problem.csv,axes.npz}`. The experiment surface is `auxiliary/rating_dev/train_joint_mirt.py`.

Prerequisites it **aborts** without: the activation extraction, the feature caches, and the motif proxies together with `axes/labels.json`. It derives no directions of its own.

It **renames any existing `axes/per_cell/` aside** rather than leaving it in place. `per_problem.csv` from this fit beside `per_cell/` from the previous one ships a shard whose two embedding sources disagree, and because both files are individually well-formed, no client check can catch it.

It does **not** write `labels.json` — that is hand-maintained — and does not write `axes/per_cell/`, which is the next step.

It prints a per-axis identity report at the end: hinged tail agreement for both ends, over attempted problems, saying how well the fit expresses each rule. **Read the tail count, not the correlation** (`motif_axes_workflow.md`). Note what a low figure means *here*: against a rule it says the model cannot represent the criterion, not that a motif drifted — the criterion is still sitting in `motif_rules.py` to be re-read.

**Runtime is up to a couple of hours, and the length is known before it starts.** `EPOCHS` is a fixed budget of 3000; nothing is selected along the way and the final epoch is what ships. The stop split is measured and printed every `CHECK_EVERY` = 10 epochs as the **record** of where the fit got to, and is deliberately not a stopping rule: it goes flat well before the end and then wanders inside a band a thousandth wide, so an epoch chosen on it is chosen out of that noise, and the choice moves by thousands of epochs between runs that are otherwise identical.

**Do not read a long run as a hung one.** The split improves in bursts separated by flat stretches of a couple of hundred epochs, and the train/stop gap widens throughout.

Per-step cost is dominated by the shallow-slope barrier, which is why it reads `SHALLOW_SAMPLES` = 20000 corpus rows rather than all of them: an unbiased sample of the same mean, and substantially faster per step than reading the whole corpus, which costs two thirds of a step's trunk work for no extra coverage.

### `write_mirt_per_cell`

Writes `axes/per_cell/<size>/<tid>.npz` — a K-vector for **every legal cell**, which the client needs because it scores the move the user *picked*, not only the answer. Resumable by file existence, so an interruption costs only the batch in flight; the whole pass is on the order of fifteen minutes.

A per-cell value needs the full projection plus a trunk pass, which is far more arithmetic than a linear map on features would need and is nonetheless irrelevant: the pass is bound by **reading** the activation npz files, so read parallelism is the whole optimization.

**It assembles the non-activation tail from `PRODUCTION_BLOCKS` through the same loader the trainer uses**, never a local block list. That tail is constant per problem — it is the correct move's — so the layout is exactly what keeps the fanout honest: a block describing the problem through its answer would hand every alternative cell the answer's description (`methodology.md` §1.3). A layout named here instead would be a second switch, and feeding the trunk columns it was not fitted with produces a shard that is well-formed and wrong.

### `map_rating_to_rank`

Reads the rank surveys and `db/d1.sqlite` (timed attempts for the career-block anchors and the median solve time); writes `ratings/rating_to_rank.json` (`{ratings, ranks}` plus a top-level `time_adjust {alpha_per_log10_min, center_log10_min}`) and `ratings/rank_map.json` (`{server: {our, server}}` knots, EGF included as a real coupling rather than an identity line). Both board keys are identical. Backend only — `package_for_pwa` publishes both.

Production-clean: no experiment flags. The experiment surface is `auxiliary/map_rating_to_rank_dev.py`, and the cross-validation harness that validated the career-block anchors is `ablations/eval_rank_map_cv.py`; the end-to-end calibration check to run after a refit is `ablations/check_mapped_vs_declared_rank.py`. The model itself is `methodology.md` §1.12.

**Ratings are time-adjusted throughout the mapping**, so the client headline maps the *raw* rating through the curve while every server translation maps the *time-adjusted* one. Dropping the dock for the headline is a client display choice, and the consequence for measurement is in CLAUDE.md's rule about which error column is quotable.

**Measured-rejected, don't re-add:** errors-in-variables weighting over each rating's Fisher confidence interval. Declaration noise dominates, and it is consistently worse held-out.

### `fit_population_constants`

Reads the fitted response-curve parameters and the attempt log; writes `ratings/population_constants.json` — the retention and time-bias constants plus the rank prior. The time adjustment is **read through** from `rating_to_rank.json`, not fitted here.

It fits **no** per-axis quantity and reads nothing from `axes/` at all, so an axis refit that changes K does not require re-running it. It fits **no motif constant at all** — the motif readings are computed client-side against the user's own neutral point, so there is no population centre to fit, ship or validate.

**`first_success_step` is fit by moment matching, never by retention likelihood.** It is an *indicator* on "the previous attempt succeeded", not a second growth rate — the second success adds nothing measurable — and it is moment-matched from the within-user odds ratio between the first rung and the rest. The likelihood is the wrong instrument here: it scores predictions on intervals the scheduler itself chose, and sizes the step about threefold too small. Two cautions when quoting the surrounding measurements: the retention test is **one-sided** (a ceiling, not a tolerance edge — a serve realizes less retention than the window's open), and the raw delivery figure is mostly confound, so don't quote it. Detail: `methodology.md` §2.8.

Backend only; `package_for_pwa` publishes it, and the PWA build inlines it into the JS bundle.

### `package_for_pwa`

**The single writer of `pwa/tsumegos/`.** There are no side-channel patch scripts: to publish a labels-only or curve-only change without rebuilding archives, re-run this step and let its per-size change detection skip the heavy shard rebuild. `--force` overrides the change detection.

A change to the **packaging format itself** is not captured by the input hash — run `--force` once after such a code change.

It has **no axis-related ship gate**: nothing downstream of a refit needs a matching population vector, and every consumer reads the shipped axis count from the published metadata rather than assuming it. It prints which components are unlabeled and ships them anyway.

The output layout, and the ordering contract the per-cell sidecar carries, are in `motif_axes_workflow.md`; the artifact inventory is `methodology.md` §3.4.

## Running the long steps

**Several steps run for hours, none of them checkpoint, and a naively launched run will be killed part-way — losing all of it.** `generate_tsumegos`, `benchmark_bots`, `extract_activations` and `train_mirt` are all in this class. Three ways such a run dies quietly:

- **A shell that backgrounds with `&` and then exits takes the child down with it** (SIGTERM, exit 143) — including a wrapper that backgrounds the real work and then returns. Launch detached instead:
  ```bash
  setsid nohup ./.venv/bin/python3 -u -m pipeline.<step> > /tmp/<step>.log 2>&1 < /dev/null &
  ```
- **`timeout N` kills at N with exit 124 and no partial result.** Don't wrap a step in a timeout unless you are certain it exceeds the real runtime; a too-short guess is indistinguishable from a crash after the fact.
- **Piping through `tail`/`head` buffers everything until exit**, so a healthy long run looks hung and a dead one looks alive. Run Python with `-u`, redirect to a file, and poll the file.

## Artifacts: overwrite in place, and the two tracked exceptions

**The steps that write `ratings/`, `activations/`, `axes/` and `pwa/tsumegos/` overwrite in place — do not copy them aside before a refit.** They are derived artifacts. What is kept is the *inputs*: the corpus, the attempt log, and the committed code. A fit that comes out wrong is reverted by re-running the step, not by restoring a snapshot. Copying them aside costs many gigabytes and creates a second thing to keep straight, so back one up only when asked.

**Two files under `axes/` are tracked, and neither can be regenerated by re-running anything** — which is exactly what makes them the exception:

- `axes/labels.json` — `components`, the ordered rules this fit ships, plus the exploratory motif store's names and provenance per id.
- `axes/label_anchors.npz` — that store's directions, keyed by a content hash of the direction itself.

Three things that look like they should be tracked and are not:

- `axes/motif_proxies.npz` is a `train_mirt` **input** but is a pure function of the rules and the corpus. The committed rule in `utils/rating/motif_rules.py` is the durable artifact and the npz is a cache of it. It is also corpus-shaped, so a column in it cannot be content-hashed the way a stored direction is.
- `axes/axes.npz` is written by every run and carries the `embedding_version` stamp packaging reads, so a committed copy is a stale version waiting to be shipped onto a different fit's embeddings.
- `pwa/tsumegos/` is gitignored in its entirety, which is why the published data carries its own content version rather than riding the git revision (`client_sync_and_cache.md`).

## Remote execution

`python vast_orchestration.py` with no arguments prints its subcommands. Only `tune`, `generate` and `benchmark` are remoted; everything else stays local. `run <step> <size>` forwards the size verbatim, validated locally against `ALL_BOARD_SIZES`. The remote gets a CUDA KataGo build uploaded as `bots/katago_cuda12` with `KATAGO_PATH` pointed at it — see `corpus_and_engine.md` for why the binary path is never hardcoded.

## Working with large artifact directories

**Several artifact directories hold six figures of files** — `tsumegos/<size>/` and `activations/<size>/` are around a hundred thousand each, plus `axes/per_cell/<size>/` and the shard archives. **A shell glob like `ls dir/*.npz` silently expands past `ARG_MAX` and reports 0**, which is a verified false negative on these directories rather than a theoretical one. Count with `find <dir> -maxdepth 1 -name '*.ext' -printf . | wc -c` (or `find ... | wc -l` where newlines are safe) and iterate with `find ... -exec` or `xargs -0`. The same trap applies to anything piping a glob into another command. Treat any directory under `tsumegos/`, `activations/`, `axes/`, `attempts/`, `ratings/` or `pwa/tsumegos/<size>/` as "may exceed argv" by default.
