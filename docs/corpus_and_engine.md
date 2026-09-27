# The corpus and the engine: file contracts and KataGo integration

<!-- Editing rules: docs/contract.md. Present tense, no changelog, rules here and evidence with them. -->

What a problem file guarantees, how those guarantees are enforced and re-verified, how the derived per-cell arrays line up against the SGF, and how KataGo is driven from both the pipeline and the analysis service. The *method* behind the two problem-set invariants is `methodology.md` §1.1 and §3.1; this document is the engineering that surrounds them. CLAUDE.md carries the short form and points here.

## Identity

**Tsumego identity is symmetry-aware.** Filenames come from `get_orientation_invariant_hash` (`pipeline/generate_tsumegos.py`) — SHA-256 taken over all eight D4 transforms of the board, with `min()` picking a canonical orientation. The dataset therefore cannot contain two problems that are rotations or reflections of each other, and `tsumego_id` is that canonical hash truncated to 8 hex characters.

`auxiliary/check_corpus_ids.py` re-verifies the corpus against the rule, and `--emit-map` regenerates an id remap (`surgery/id_map.json`) if a defect ever needs one.

### Retired ids are never reused

**`tsumegos/retired_ids.txt` lists ids removed from the corpus but kept in the attempt log, and `generate_tsumegos` treats them as taken.** Removing a problem no longer deletes its `tsumego_reports` rows, so an id handed to a new problem would silently inherit a stranger's history — the one failure mode that raises nothing anywhere and cannot be untangled afterwards. `tsumego_id_taken` checks the file alongside the corpus.

Format is one 8-hex id per line with `#` comments, appended in blocks that say when and why. Regenerate it from a fresh `pipeline.fetch_database` as *distinct `problem` in `tsumego_reports` not present in `tsumegos/`* — an id whose rows were deleted by an older purge is legitimately absent and reusable.

## The single-solution invariant, and why order is load-bearing

By generator design every problem has exactly one correct first move — the `solution` returned by `ProblemFinder.create_problem` — so downstream code can rely on "one canonical answer per problem".

**That answer is `variation_nodes[0]` by construction, and there is nothing to enforce.** `construct_initial_sgf` writes the verified `solution` as the decision node's first child; both deep passes and the shallow pass only ever *append* (`get_child_move` reuses an existing child or calls `new_child()`), so nothing can displace it for the life of the tree.

**The invariant is load-bearing because every consumer reads the solution positionally** — `extract_activations`' `correct_idx`, `benchmark_bots`' grading, row 0 of the published per-cell block, and the client's `isCorrectMove`. So the answer is never re-derived from the evaluation comments. Those come from the annotation passes at their own playout budgets and are shallower than the re-read `ProblemFinder.is_valid` performed; picking by them is how the client used to end up on a different move from every backend stage.

**A guard used to sit here and has been removed.** `promote_winning_first_move` reordered the decision node's children so the comments' winner led. Two corpus-wide scans showed it had never once changed a shipped file: all 220,463 live problems carry the `DEEP_ENTRY_PLAYOUTS = 1000` comment on `variation_nodes[0]` — the signature only the answer node's own deep read can write — and no root variation anywhere carries a 1000-visit comment at index ≥ 1 (alternatives top out at 900–999). It was removed rather than turned into a check, because a check states that the ordering could be wrong when the construction says it cannot be.

`auxiliary/check_tsumego_sequences.py` re-verifies the corpus; `winner is NOT variation_nodes[0]` must be 0.

## What a shipped SGF carries

**Each branch comment is `"<winrate>%/<score>/<visits>"`** (e.g. `"86.4%/3.1/1000"`). `pwa/js/sgf.js` reads this directly, and only when it is missing does the client call the analysis service. Don't strip these comments when manipulating SGFs.

The visit count is the third field because the first two cannot say how much search is behind them, and that is how a corpus-wide 30-visit annotation went unnoticed: generation ran its deep passes at a flat 30 and then let the shallow sweep overwrite the decision node unconditionally, so every number in every file looked exactly like a deep one. 14.5% of 13×13 answers and 12.2% of 9×9 answers recorded a winrate below the 80% floor generation refuses to admit — impossible as a property of the position, and invisible without the field. It is appended rather than restructured so that every parser, which matches the first two fields with an unanchored search, keeps reading pre-field comments unchanged; a comment lacking the field is a 30-visit value by construction. `set_eval_comment` is the single writer on the pipeline side and never replaces an estimate with a shallower one; `setEvalComment` (`pwa/js/sgf.js`) is its client twin, single-sourcing the same format and the same never-downgrade rule for the answers the client gets back from the analysis service.

### A resolved branch is read off the tree's own shape, and the client must never deduce it

**Admission's 20-step escalation is ONE engine session, so its steps are not independent draws.** `ProblemFinder.is_valid` escalates from 10 to `PLAYOUTS_VERIFY = 1000` playouts over 20 steps and requires the answer to clear the winrate floor at every one, which reads like 20 independent confirmations. It is not: the steps share the engine's neural-net cache, so a position that draws high on its first search tends to draw high on all of them. Measured on `004978fa`, seven searches at the same 1000 playouts with the same model spread from 32% to 92% winrate — bimodal, and always on the same move — while five repeats *inside one session* returned 76.9% every time. Two of the seven clear the 80% floor, so both its admission and its later rejection are draws from the same distribution rather than a disagreement about the position.

The practical consequence is that a re-verification at the admission budget re-litigates admission rather than auditing it, and on this class of problem it will disagree stochastically. Such positions are recognisable: the score stays firmly positive (+6.8 to +21.6 across those seven) while the winrate flickers, which is what an unresolved life-and-death or ko looks like when the score assumes one resolution. They are candidates for retirement on the grounds of instability, not error — neither reading is the wrong one.

**A branch was driven to a conclusion when its subtree contains TWO CONSECUTIVE pass nodes.** `generate_variations_directly_to_sgf` (`utils/katago/add_variations.py`) follows one primary line to a double pass and abandons every other branch at the `FREE_BRANCH_VISITS` floor, and it creates the second pass node only where it actually reached that terminal — so the tree's own shape carries the fact and nothing has to be annotated. `subtree_reaches_conclusion` and the client's `subtreeReachesConclusion` are the two readers, and they must stay in step. `analysis.js` narrows its display pool to branches that pass the test.

**Annotating each resolved branch with an SGF marker property instead is redundant and strictly worse.** The writer's recursion is what creates the second pass, so the tree already carries the fact; a marker adds a second source of truth that can disagree with the structure, and every reader then needs a rule for which one wins. `auxiliary/mark_recommended_variations.py` wrote such a marker and is retired to its `--unmark` mode, which strips one from any file still holding it.

Three cheaper-looking substitutes for the predicate are wrong, and were measured to be:

- **"A pass appears anywhere in the subtree" — measured-rejected, don't re-attempt.** It accepts 2.2% of refuted first moves against the double-pass test's 1.1%: twice the pool, the extra half being lines nobody played out. The generator plays some low-visit branches out to a pass without the move quality to match, so the client offers continuations along which the solver's own recorded moves degrade into a loss. This is the whole cause of two corpus purges; **no weakened form of the test is safe.**
- **"A pass PAIR appears anywhere in the subtree" — measured-rejected.** The shallow sweep writes a move and its best response, so a pass answered by a pass manufactures the shape with nothing played out, admitting 32,445 extra children.
- **Walking the highest-visit child — measured-rejected.** KataGo ranks `moveInfos` by LCB rather than raw visits, so the move the writer follows is not the highest-visit move at 20.8% of multi-child nodes, and that walk recognises only 42.5% of answer lines against the pass test's 100%.

Two boundaries around the predicate:

- **Selection is unchanged and the predicate only narrows the pool.** Black shows the first child — the same move `isCorrectMove` accepts; white takes lowest winrate then lowest score, so two ghost moves still appear when those differ. **Correctness never enters here at all**: it is `children[0]`, so how far a line was searched cannot redefine what counts as the right answer.
- **Resolution says the line was played out, not that it resolves the way the problem claims.** A principal line can end with the solver losing, so whether a resolved line is a win is read from the evaluation the terminal node carries, never from the predicate.

## Corpus soundness

**`auxiliary/audit_solution_lines.py` is the soundness check**: exact minimax over the branches that reach a double pass. The answer must hold the win, and no refuted first move may be winnable.

**Only resolved branches count.** An abandoned stub is not evidence in either direction. The retired `depth >= 10` ply floor (whose findings are still listed in `auxiliary/broken_review_paths.txt`) was standing in for that and got it wrong in both directions — it scored abandoned lines, and its floor hid resolved short ones.

Coverage differs by board and the difference matters when reading a result. **Answer coverage is complete on both boards. Refuted-move coverage is complete on 9×9 and a floor on 13×13**: `auxiliary/add_losing_lines.py` gave every 9×9 problem lacking one a losing line — the best refuted first move by the client's own rule, played out to a double pass — and the problems that could not get a sound one were purged. The generator still pursues refutations far less often on 13×13, so a refuted-move count there counts only what it happened to resolve.

Note what the 9×9 pass cost to learn: **coverage is what makes the audit able to fail.** Thousands of problems whose refutation the corpus contradicted were invisible until they had a resolved line to judge.

**Measured-rejected, don't re-attempt:** any rule that *prunes* a side's options and then judges it. Pruning to double-pass-reaching branches forces a side into a move it would never play — most visibly a forced pass whose score-it-now terminal reverses the whole line — and it manufactures both false positives and false negatives.

## The three coordinate spaces, and how they map

A problem exists in three different orderings, and crossing between them by index rather than by coordinate is the standing bug.

**1. Activation npz — board-scan order, a superset.** `extract_activations.prepare_problem` enumerates every legal cell for the player to move (the colour of `variation_nodes[0]`) sorted by `(y, x)`, and records `correct_idx` pointing at the correct move's scan position. `pipeline/extract_activations.py` runs the vendored PyTorch KataGo b18 (`utils/katago/vendor/`, bootstrapped via `utils/katago/torch_model.py`, so `import katago.…` resolves there), captures the post-skip-add output of all 18 residual blocks (18 × 384 = 6912 floats) at every legal cell, and projects each cell's vector through the **frozen activation encoder** on the GPU before anything touches disk. Output per problem: `candidates (n_legal, ACT_ENCODER_DIM) fp16`, `xys (n_legal, 2) int32`, `correct_idx () int32`, `encoder_fp () uint32`.

**The encoder is one immutable linear map** — per-channel standardization folded with a PCA projection (`utils/rating/activation_encoder.py`), fitted by `pipeline.fit_activation_encoder` on the cells that ship or get compared against: correct cells drawn uniformly over the corpus, correct cells drawn by attempt count, and distinct human-picked wrong cells, in equal thirds. Storing the encoding instead of the raw trunk cuts the store ~13× and lets every downstream fit (the joint-PCA feature basis, the per-cell axis projection) refit *inside* the encoded space without the raw vectors existing anywhere. `ACT_ENCODER_DIM` = 512 is measured, not chosen: under a simulated fit-set redesign (encoder fitted on correct cells only, scored on recovering a correct+picked top-128) refit parity rises monotonically with width and saturates by 512, while iid corpus growth is already at parity at every width — the sweep is `ablations/encoder_m_sweep.py`. The end-to-end checks that licensed the compression: the availability-null comparisons flip only at near-ties and below the cross-projection reference rate (`ablations/encoder_altstats_gate.py`), and a paired dev-trainer A/B lands within same-fold seed noise on the combined held-out criterion.

**`activations/encoder.npz` is tracked in git and is the one irreplaceable artifact of the store**: with only encoded files on disk, nothing can regenerate it short of a full GPU re-extraction. Every encoded npz carries the encoder's 4-byte fingerprint, and every writer treats a mixed-fingerprint store as a hard error — extraction verifies existing files' fingerprints in a background thread on resume, and `pipeline.encode_activations` (the converter for a raw store, when one exists) refuses to skip past a file written under a different encoder.

**An encoder refresh is rare and total.** Extract a raw subset with `extract_activations --raw --out-dir <tmp>` (the flag refuses to target the encoded store), refit with `fit_activation_encoder --act-dir <tmp>`, then re-extract the whole corpus with `--force` — raw vectors stream through the new encoder and the full-width matrix never needs disk. It must end with `cache_rating_features --rebuild-activation-cache`: the roll-up caches reconcile by id set, and a re-encoded store has the same ids at the same width, so without the forced rebuild they would keep serving the previous encoder's rows while looking current.

The alternative — walking the SGF's pruned variations in document order with `correct_idx=0` — ships a non-superset cell set and does not match the on-disk corpus. `correct_idx` is non-zero in general (13×13 `00000f4c` has `correct_idx=38`; a re-extraction was verified byte-for-byte against the on-disk corpus).

**2. SGF root variations — a pruned subset, plus a pass.** So npz ⊇ {SGF non-pass variations}: every move a user can actually click is embedded, but around a tenth of problems have extra embedded cells with no SGF variation, and **every** problem has a `pass` variation the npz never embeds.

**3. The published per-cell sidecar — SGF-variation order.** Because the npz is scan-order and a superset, `package_for_pwa.project_cells_for_problem` re-parses each SGF and rebuilds the rows in SGF-variation order through an `(x, y) → npz row` map, dropping the extra scan-order cells and writing an **all-zeros placeholder row for the pass**, so that bin row *i* equals SGF child index *i* exactly. See `motif_axes_workflow.md` for the format revision that carries this.

Two rules follow for every consumer:

- **Pick the correct row by `correct_idx`, never positionally** — every consumer selects `candidates[int(correct_idx)]`; do not assume row 0.
- **Map cells by `(x, y)` coordinate, not by index, when crossing the npz↔SGF boundary.**

The sgfmill parse dominates packaging time, which is why shard building is parallelized across processes (`--workers`, default `os.cpu_count()`).

## Board size on the pipeline side

There is **no global `BOARD_SIZE`** and no in-process dual-size run. The three single-size generation steps — `tune_engine`, `generate_tsumegos`, `benchmark_bots` — take a **mandatory positional size** (`9` or `13`), threaded explicitly through the utils: `KatagoQueries(controller, board_size)` owns it as a public `board_size` attribute, `KatagoController.start_engine(model, board_size, …)` takes it, `ProblemFinder` reads `queries.board_size`, and the leaf helpers `construct_initial_sgf(…, board_size)` and `get_child_move(…, board_size)` take it explicitly.

Every downstream step iterates `ALL_BOARD_SIZES` and takes no size argument. Artifact directories are keyed by size throughout. To refresh both boards, run each generation step once per size (`… generate_tsumegos 9` then `… generate_tsumegos 13`), then run the downstream steps once.

The client-side view of board size — the URL hints, and how the two boards merge into one rank — is in `pwa_client.md`.

## KataGo integration

**Binary selection is per machine, by probe.** `utils/katago/binary.py` (`resolve_katago`) picks which `bots/` build every launcher uses: the CUDA builds (`katago_cuda13`, then `katago_cuda12`) are each accepted only on answering one real 1-visit query, since a build can link its libraries and still fail at device initialisation, and the OpenCL `bots/katago` is the unprobed fallback (its first start is a minutes-long kernel tuning pass). The launch environment prepends the `nvidia-*` wheel library directories of the venv — the CUDA builds link cuBLAS/cuDNN by soname and those ship there, not on the system path — plus `~/.local/lib/katago-deps` for libraries a host lacks outright. The verdict is cached per host in `bots/.katago_binary.json` against the chosen binary's size and mtime. `KATAGO_PATH` still overrides the choice; `auxiliary/vast_orchestration.py` uploads a CUDA build as `bots/katago_cuda12` and sets it on the remote. **The CUDA major version is in the filename on purpose** — the binary links `libcublas.so.<major>`, so one build does not serve hosts of different majors. **Don't hardcode the binary path**; launch through the resolver.

**`KatagoController` is thread-safe by design.** A single master reader thread fans responses out via per-query `threading.Event`s keyed on query ids, so `send_query` is blocking from the caller's point of view. The worker threads in `generate_tsumegos` and `benchmark_bots` rely on this.

**Models live outside git.** `models/` is gitignored; `utils.katago.models.get_model(filename)` lazily downloads from `media.katagotraining.org`, or from GitHub for the 9×9 and human models, on first use. The default is `MODEL_FULL` (`kata1-b18c384nbt-…`).

**The PyTorch loader requires a `.ckpt`, not a `.bin.gz`.** The `.bin.gz` in `models/` is the binary export the C++ analysis engine reads and is not loadable via `torch.load`. Activation extraction expects `models/kata1-b18c384nbt-s9996604416-d4316597426/model.ckpt`, extracted from the corresponding checkpoint `.zip`, or a `--ckpt` pointing at one. Don't point `--ckpt` at a `.bin.gz` — KataGo's training code converts PyTorch → binary and not the other way, so one is not derivable from the other.

**`bots/analysis.cfg` is hardware-tuned in place** by `pipeline.tune_engine`; see `pipeline_operations.md`.

### The `requireMaxBoardSize` trap

**`requireMaxBoardSize = true` in `bots/analysis.cfg` belongs to this repository's config only.** The deployed analysis service uses its own `analysis.cfg` on the box and passes no board-size override, so it has always served both sizes and is untouched by this.

`maxBoardXSizeForNNBuffer` is a **maximum**; `requireMaxBoardSize` is what turns it into an exact requirement (KataGo names it `requireExactNNLen` at runtime, and the cfg's own comment says "EXCLUSIVELY for exactly the board size specified"). A query on any other size then kills KataGo's async bot thread with a message on stderr and **no response on stdout**, so the caller waits out its full query timeout on a request that can never be answered.

Anything reaching KataGo through `utils.katago.controller` inherits the setting, since `start_engine` always emits the override. The pipeline never notices because each of its runs is single-size; a server takes both, so `worker.py` passes `requireMaxBoardSize=false` and runs one engine sized to the largest board. Verified: 9×9 answers on the padded buffer match a native 9×9 engine to float rounding.

## The analysis service and its worker

`analysis_server/server.py` runs on the cloud box — stdlib plus sgfmill only, since it is deployed with no repository beside it. It answers the PWA's analysis fallback when an SGF lacks embedded evaluation. Its own `analysis.cfg` is tracked at `analysis_server/analysis.cfg` and is **not** `bots/analysis.cfg`, which is the pipeline's and tuned for the opposite machine.

**The board-size ceiling is stated twice and the two must move together**: `MAX_BOARD_SIZE` in `server.py` refuses anything larger with a 400 before any evaluation, and `maxBoardXSizeForNNBuffer` in that cfg sizes the NN buffer to the same bound, so an oversized board fails inside KataGo regardless.

### What the worker buys is search, not speed

`analysis_server/worker.py` (`./.venv/bin/python3 -m analysis_server.worker`) opens a few long polls against `api.onlymove.net` and runs whatever jobs come back down them; the server prefers a parked poll and otherwise answers with its own KataGo exactly as it always has. The PWA is not involved at all — `pwa/js/analysis-api.js` posts to the same URL either way — so there is no client version to keep in step, and no inbound port, DNS record or certificate on the worker's side.

**The point is search depth.** Measured, the cloud box answers in about 200 ms of compute and a GPU in about 45, but reaching that GPU costs a round trip that spends nearly the whole saving — so at equal visits the two are a wash and the feature would be pointless. What the box *cannot* do is search: one visit is the raw policy, and 30 visits there would take seconds. The worker therefore **replaces the visit count** on every query it takes (`WORKER_VISITS`, 30), costing roughly a second end to end against the box's third of one, and returns a genuinely read answer instead of an opinion.

Three consequences, all load-bearing:

- **`interpret_response` branches on what came back** — `moveInfos` when present, policy otherwise — never on the server's own `PLAYOUTS` constant. The two backends deliberately search to different depths, so reading the constant sends a searched reply down the policy path, discarding the search with no symptom.
- **The reply carries `visits`**, and the client shows its blinking-red unreliable marker only for a single-visit answer (`_searched` in `analysis.js`, tested in `ui.js`). `_origin` stays `'api'` either way, since the move-selection waterfall keys on it. The reply's own `winrate`, `score` and `visits` are written onto the node through `setEvalComment` in the corpus comment format, so a server answer reads back through the same parsers as a baked one — **both engines report from BLACK** (`analysis_server/analysis.cfg`, `bots/analysis.cfg`), so there is no frame to convert. A reply missing any of the three carries no comment at all rather than an invented one, which simply leaves the position to be re-queried later.
- **`SEARCH_THREADS` is a latency setting, unrelated to `POLL_THREADS`.** Visits are sequential within one query unless several threads explore at once.

**The pool's capacity is the number of parked polls.** A job is only ever handed to a poll that is already waiting, so a busy or absent worker degrades to local service rather than queueing behind itself, and stopping the worker at any instant is safe. Failure is bounded rather than absent: a worker that dies holding a poll is usually caught on the write and the job comes straight back, but a dropped network gives no such error, so one request pays `WORKER_JOB_TIMEOUT` before `WORKER_COOLDOWN` takes the worker out of rotation.

**Both ends read the shared secret from a file** — `~/.config/onlymove/worker_token` locally, `worker_token` beside `server.py` on the box — with the environment winning when set, so starting either needs no wrapper and no secret on a command line where `/proc` would expose it. **An empty `ONLYMOVE_WORKER_TOKEN` on the box means the feature is off with no second switch**: the three `/worker` routes 404 and every request is served locally, so the server can be deployed before any worker exists. Turning it on is setting that variable and restarting. Token auth is the only thing separating a worker from anyone on the internet, and its answers are shown to users as ours.

**Deploy with `analysis_server/deploy.sh`** (`--token` also installs the shared secret; `--no-restart` ships without one). It reaches `typo_hh@api.onlymove.net` over plain ssh and edits nothing under `/etc`: the account owns its working directory but is not in `google-sudoers`, so the script restarts by killing the main process and letting the unit's `Restart=always` bring it back, and installs the token as a 0600 file beside `server.py` rather than a unit `Environment=` line — which would need root **and** be world-readable through `systemctl show`. It ships by `scp` (the box has no rsync) and verifies by md5 what actually landed.
