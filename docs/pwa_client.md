# PWA client: state, storage, re-derivation and the serve path

<!-- Editing rules: docs/contract.md. Present tense, no changelog, rules here and evidence with them. -->

The client's operating rules — where state lives, what may write it, what has to move together, and which plausible-looking changes are known to break something. The **method** behind the live rank, problem selection, recall scheduling and the motif readings is `methodology.md` chapter 2, and is not repeated here. **Sync and cache freshness** are `client_sync_and_cache.md`; the **file layout** is `pwa/README.md`. CLAUDE.md carries the short form and points here.

## State and application modes

**State is centralized.** `pwa/js/state.js` holds the single mutable object and every change goes through an exported setter.

`state.appMode` switches between `'solving'` (`game-logic.js`), `'analyzing'` (`analysis.js`) and `'linked'` (rank-frozen, URL-shared problems). Click handling in `board.js` routes by mode — linked clicks go through game logic, the same as solving.

**Linked-mode attempts are deliberately recorded nowhere.** `sendReport` and `updateUserStats` (`attempt-recorder.js`) both early-return on `state.problemHistory[0]?.linked` — the canonical flag, set in `problem-loader.js` — so a solved shared-link problem reaches neither D1 nor local history. **Keep the two guards in lockstep on that one flag** — a rank-frozen share must not count toward the corpus, anyone's statistics, or the server-stats and rank histograms.

### The URL is an entry hint, not an address

`/9` and `/13` are read **once at boot** (`getBoardSizeFromPath` → `resolveInitialBoardSize` in `main.js`); the chosen size persists in state and local storage thereafter, not in the URL. `cleanUrl()` (in `problem-loader.js`, called from `displayProblem`) collapses the path to `/` and strips `?id`, `?o` and `?recall` whenever a **non-linked** problem is painted. **Do not assume a reload leaves the bar at `/9` or `/13`; it does not.**

**Linked problems are the deliberate exception**: their `?id` and `?o` are kept for the whole linked session, because `displayProblem` skips `cleanUrl` while `appMode === LINKED`. That is what makes every kind of reload re-enter the *same* shared problem at boot (`parseUrlIntent` reads `?id`) instead of dropping to the standing challenge — a service-worker update takeover (`index.html`'s `controllerchange` → `location.reload()`, the most common one, which in-app browsers hit when a stale cached worker is replaced), the `?id`-404 and repeated-load-failure reloads, the version-mismatch reload, or a manual refresh. The URL is cleaned the instant the user moves on to a non-linked problem (`returnToChallenge` / `advanceToNextProblem` → `displayProblem`).

## Cross-size merge: what is shared and what is per-size

From the user's point of view 9×9 and 13×13 share **one rank** (front-page label, problem selection, recall eligibility) and **one statistics page**. Histories, rank curves and problem pools stay per-size.

`state.js` keeps `answerHistory` as **one unified cross-size array**, date-ordered with each entry tagged by `size`, plus single shared values: `userRank`, `displayRank`, `serveRank` and the recall model's `modelParams`. The motif readings are not among them — they carry no persisted state at all, being recomputed from the merged history on demand. The only per-size persisted bucket is `challengeProblem`.

`getMergedAnswerHistory()` returns the unified working set, which the recall model trains from (it slices a recent window, so the working set suffices). The statistics panel instead awaits `getMergedFullAnswerHistory()` so its charts span the whole career.

**Pooling is sound because every feature these consumers read is board-independent**: the internal rank is the consensus rank, so a given value means the same strength on either board, and the motif axes are jointly fit across sizes.

**Play-time size-awareness is preserved.** A recall problem can only be re-shown on its own board, so `getRecallCandidates` takes an explicit history — defaulting to the active size, with the merged union passed only for statistics counts — and `buildRecallSchedule(size)` reads the shared model and shared rank but that size's history.

Two consumers do real work to pool:

- The recall logistic carries a **`SIZE_IS_13` fixed effect** (9×9 is the baseline) whose coefficient divided by the logit-per-rank is the statistics panel's "stronger/weaker on 13×13" readout.
- The career decomposition (`rank-decomposition.js`) pools in **internal-rank space** with a per-attempt slope `LOGIT_PER_RATING / |M'(rating)|` that maps each attempt through its own size's curve, so both boards share one skill curve without a rank-space slope bias.

A `mergeSharedAcrossSizes_v2` migration flattens any legacy per-size rank and model buckets: the rank becomes the **maximum** of the two, so a strong 9×9 player immediately gets hard 13×13 problems, and the model buckets are dropped so the shared models retrain.

## Stored history: a working set in local storage, a cold tail in IndexedDB

`answerHistory` is around 99% of local storage and on heavy users overflows the quota, so `state.js` keeps the most recent `ANSWER_HISTORY_WORKING_SET` attempts — unified across both sizes — as a **synchronous local-storage array**, and evicts everything older to IndexedDB via `pwa/js/history-store.js` (`appendTail` / `getTail` / `updateTail` / `replaceTail` / `clearTail`, durable via `navigator.storage.persist()`). **Every method is a safe no-op or empty result when IndexedDB is unavailable**, so play degrades to working-set-only and never breaks.

The working-set size is chosen to be the largest hot window the app uses on the merged history, which makes it a **superset of every synchronous read**: existing consumers read `state.answerHistory` (a cached active-size view) or `getMergedAnswerHistory()`, and already slice within that bound. Only the full-career charts await `getMergedFullAnswerHistory()` (tail ⧺ working set, de-duped by `(problemId, date)`).

**The write path stays synchronous.** `setAnswerHistoryForSize` and `updateAnswerHistory` route through one `reconcileSize` that replaces a size's entries in the unified array and persists. A debounced `evictColdTail()` then appends overflow to IndexedDB **and confirms before trimming** the in-memory and local-storage copy — a crash-safe seam, since a kill mid-evict leaves a de-duped duplicate in both, never a gap, and an unavailable append simply keeps the rows in the working set.

`initHistoryStorage()` runs once at boot, after the state-epoch gate has settled the storage shape; `main.js` awaits it before the re-derivation passes write, so a quota-stressed device has local storage freed first.

### The cold tail carries no embeddings, and that is enforced at the write boundary

`history-store.js` **drops the embedding block from every row it stores** (`appendTail`, `replaceTail`). This is not an assumption to be maintained by callers: the only pass that clears stale vectors walks the working set alone, so a vector evicted to IndexedDB is beyond any refill, frozen at the version of the day it was evicted, and unreachable by every later gate.

On top of that a refit **sweeps the tail** (`sweepTailEmbeddings`), which is redundant against the write boundary on purpose — it gives the invariant a repair rather than only an assertion. It is deliberately **not** gated into the refit's `allOk`, since nothing reads those vectors and re-running a whole refit to reclaim dead bytes would be the expensive half paying for the cheap one.

### What a refit does to the tail

- A **curve-only** refit recomputes the tail's internal rank from its stored raw rating (`backfill.js` via `updateTail`), with no shard walk.
- A **ratings change** re-derives the tail's full rating fields from the published parameters — the same walk the working set uses, extended to the tail's problem ids. The tail's ratings are frozen at eviction time, so after a re-rate they go stale **and device-dependent**, each device having evicted at a different rating era; the chart plots them, and two synced devices' rank graphs diverged at the start until this was added.
- A problem that has **left the corpus** has its rating fields **stripped rather than left stale**. The attempt stays, rating-less, and `flattenAttempts` then drops it from every estimate — front-page rank, chart, weakness and recall all already skip rating-less and embedding-less rows.
- An **account switch** clears the tail and re-splits the full pull.

**Two accepted tradeoffs, both kept synchronous on purpose:** a re-draw of a problem last seen beyond the working set is treated as new, since repeat detection scans the working set only; and `timeAdjustDeltaRating` uses the working-set median solve time.

## Derived per-attempt fields and their generations

**A stored per-attempt field is governed by its generation's device-level version — there is no per-row stamp — and there are three generations under one rule.** All three keys live in `constants.js` and all three are read through `storedVersionMatches` (`data-version.js`):

| Generation | Version key | Fields it governs |
| --- | --- | --- |
| the motif fit | `EMBEDDING_VERSION_KEY` | the correct-move and picked-move vectors |
| the rating fit | `RATINGS_VERSION_KEY` | rating, slope, floor, quality |
| the rank curve | `RANK_CURVE_VERSION_KEY` | the internal rank |

The scheme is sound only under two conditions, and both have been broken in production.

### Condition 1 — nothing writes a derived field while its own build's generation and the device's disagree

Asked **at write time** in `attempt-recorder` — the vectors on `motifDataIsCurrent()`, the rating block on both ratings and curve — and again inside the deferred picked-vector patch, which lands after an await.

**The case it exists for:** the app upgrades between problems, but the signal rides `X-App-Version` on the `/report` **response**, while the row is written *before* that response arrives. So an instance one release behind always records exactly one attempt out of the generation the device has moved past. **A boot-time check cannot see it** — the instance booted when the versions agreed — and no later check can either, every version by then agreeing.

Refusing rather than wiping-and-restamping is a cost choice: a behind instance that wipes buys the device two full refills for one attempt.

### Condition 2 — the re-derivation reaches every holder

A value that survives it can never be found again. The holders are:

- history rows,
- the standing challenge (**discarded** on a motif change rather than refreshed — the sidecar restores its vector but not the rating fields from the same superseded shard),
- `previousProblem`,
- the on-screen problem in memory (`state.currentProblem`), because the re-derivation runs **detached** so the first problem can already be painted (`pwa/js/main.js`),
- the **sessionStorage resume slot**, cleared for *any* generation change since it is serialized JSON that cannot be refreshed in place.

Those last two were the reported bug, twice. **The standing risk is that this is an enumeration: any new place that stores a problem object must be added to the re-derivation.** It has been wrong three times.

The version gate also covers the **persisted problem objects** (`refreshPersistedProblems`), which is load-bearing rather than tidy. `challengeProblem` and `previousProblem` are whole problem objects in local storage carrying the same fit-derived fields the history rewrite refreshes, so while they sat outside the gate a standing challenge kept its pre-refit vector across the update and `attempt-recorder` stamped it onto the *next* attempt of that problem — one row from the previous fit inside a window from this one, on essentially every user, since everyone holds a standing challenge when a refit lands. Nothing downstream can screen that out: a refit can keep K and move every axis under it, so the stale vector is the right length and simply means something else. The refresh **mutates in place**, which is what reaches a challenge already painted this session, and re-reads the vector from the same lookup sidecar the history fill uses so the two can never disagree.

### The repair passes

**`repopulateMissingFields` is permanent, ungated and per generation, and runs while that generation's versions agree.** It gives any in-window row missing a vector or a rating block its current value, from the set-wide sidecars — so it is one Range read per problem, and the published parameter file is fetched only if some row needs it, which in the steady state is never. That is also what makes an interrupted re-derivation cheap: the next load finishes it instead of waiting for the next version change.

**`noEmb` is set from the corpus list and nothing else.** Absence from `lookup.txt`, already loaded and binary-searched, is the only unambiguous "can never be filled". **An empty sidecar read is not** — an offline load and a problem the fit published no vector for are indistinguishable at the reader — so those rows are simply retried, and the mark clears with the vectors so a new fit gives every marked row one fresh attempt.

**`verifyStoredVectorsOnce`** is the one-time audit of what was stored before any writer respected the version: each in-window vector is checked against the sidecar, and a mismatch is replaced with its `pickedEmbedding` dropped (`pickedCellIndex` kept), since the tendency reading compares the two *inside* one row.

**The reader's K test is a shape guard only** (`SHIPPED_K`, so a per-axis loop cannot run off the end), never a provenance test — K is the shipping decision, and a refit that keeps it moves every axis underneath. **The rejected reader is `K = rows[0].embedding.length` plus voiding the window on any disagreement**: measured, one row in 795 blanked the whole pane ("not enough data yet"), emptied the serve categories so no motif steered selection, and left the bubble naming no training focus.

**The rejected alternative to the whole scheme is a per-row version stamp** — measured at 28 bytes per row carrying a vector, about 28 KB per device. It buys *detection* after the fact, which the two conditions make unnecessary, and post-hoc diagnosis does not need it either: a poisoned row is identifiable offline by comparing its stored rating against the published parameters, which is how the reported one was found.

### Every long re-derivation pass checkpoints in batches

**A version stamp written last is what marks the work done** — a kill leaves nothing stamped and the whole pass re-runs idempotently — **but it says nothing about the work already paid for**, and these passes are network walks that are killed routinely by a backgrounded tab, a closed app or a dropped connection.

The failure this fixes: a pass that persisted only after *both* its cheap sidecar fill and its expensive shard walk discarded the sidecar refills too when the walk was killed — while the stale-vector wipe (`wipeStaleEmbeddings`) *had* persisted, so a device that never survives one uninterrupted walk sat with its vectors wiped, the pane empty, and no forward progress ever.

Both the boot pass (`backfillForSize`) and the account-adoption pass (`fillSyncedHistoryFields`, the heaviest walk in the app) now checkpoint: after the ratings rewrite, after the correct-vector fill, every `FILL_CHECKPOINT_EVERY` fills inside it, and after every shard. That is safe because the fills are fill-if-missing and therefore idempotent, and `persistAnswerHistory` writes the **live** array, so a mid-walk write cannot clobber attempts recorded during the awaits.

**The rest of the chain was audited at the same time and is already interruption-safe — do not "fix" it.** `/sync` persists a page's rows *before* advancing the id cursor, so a kill re-pulls the page and the merge by server id cannot gap. `evictColdTail` confirms the IndexedDB append before trimming, so a kill duplicates rather than gaps. The cold-tail passes are each one read-write transaction and therefore atomic. `pruneRemovedProblems` writes its signature last. The account switch clears `SWITCH_FLAG_KEY` last, so an interrupted adoption re-runs from scratch.

### Fetching a wrong attempt's vectors and its availability null

`picked-embedding.js` Range-fetches the per-cell sidecar **one whole problem block at a time** (`n_cells × bytes_per_cell`, under a kilobyte at K=5), because a wrong attempt needs two things from the same block: the picked cell's vector, and the **availability null** — `computeAltStats`, the per-axis mean/spread of the loadings over the problem's alternatives (every variation minus the correct move and the all-zeros pass row) plus the counts of alternatives loading strictly above/below the correct move. Two entry points:

- `fetchWrongAttemptStats({tsumegoId, pickedIndex, correctIndex, zipFilename, boardSize})` — the live path, called from `game-logic._handleIncorrectMoveSequence` via `state.embeddingsIndexCache`, loaded alongside shard metadata in `problem-loader`. Both indices are `children.indexOf(...)` against the SGF root variations — the same order as extraction and packaging, with no coordinate recomputation — and the correct move's index is captured before the board advances.
- `fetchProblemCellsWithIndexEntry(…)` — the re-derivation path, used when iterating shards; `backfill.fillWrongAttemptFields` memoizes blocks per problem within a shard so several attempts of one problem pay one fetch.

**`computeAltStats` works entirely in `quantizeEmbedding`'s resolution**, so the tie set of the stored pick against the stored correct vector is exactly the tie set the null's above/below counts used — a pick that reads as directionless never meets a null that says otherwise. The correct row is excluded by index when known and by quantized-vector equality against the stored `embedding` otherwise (the legacy-row path, which is why the correct-move fill must run before the wrong-attempt walk — both callers already order it that way). A problem with fewer than two usable alternatives gets `altMean: []`, the computed-but-degenerate mark that stops the fill re-asking; like `noEmb` it clears with the vectors on a refit.

An answer-history entry carries `embedding` (the correct move, from the shard metadata), `pickedCellIndex` and `correctCellIndex` (SGF variation indices), `pickedEmbedding` (the decoded sidecar value), and on wrong attempts the null quartet `altMean`/`altSd`/`altAbove`/`altBelow`. **Entries older than the model input window get their embedding fields stripped**, to keep local storage bounded.

Because the motif readings are a single shared instance across sizes, **an embedding wipe on either size needs no model invalidation and no retrain** — the readings are a pure function of whatever entries still carry a vector, so a refreshed or pruned history is picked up on the next read.

Refits auto-invalidate with no manual edit: `train_mirt` writes a content-hash version into `axes.npz`, `package_for_pwa` surfaces it in `embedding_meta.json`, and the client wipes stale vectors per size whenever the cached version differs.

The account-adoption fill uses the same sources, and its archive walk covers only the `syncedMove → pickedCellIndex` derivation, which needs the SGF, plus the picked fill — measured on a worst-case two-thousand-attempt adoption, 38 archives against 342 with the lookup-aligned sidecar absent.

The **correct**-move fill goes through the lookup-aligned sidecar instead (`fillCorrectEmbeddingsViaSidecar`): binary-search the id in the already-fetched `lookup.txt` lines, then fetch that row. Repeats share a memoized fetch, and hosts that ignore Range — wrangler dev **and** production Workers assets both do — cost one whole-file download, cached by the service worker and served row-by-row from memory thereafter. **The picked fill is the only embedding fill that still downloads shard archives** (`walkPickedShards`, and only shards carrying such a need), because its per-problem byte offsets live inside the archive.

## The three ranks

**The rank is a grid posterior over a measured population prior, and three numbers come out of it.** The model, the prior's provenance and the survivorship caveat are `methodology.md` §2.3. The rules that bind from elsewhere:

- **`userRank`** = the posterior mean — the honest number everything **calibrated** reads: the server-coupling anchors, the Worker's stored rank, recall, weakness. **Never feed a shifted value here**, or the calibration loop closes on it.
- **`displayRank`** = `mean + K_DISPLAY·excess` (`K_DISPLAY` is negative) — what the user is told: the badge, the rank-change overlay, the rank-translation modal, and the Rank chart's front-page-rank line. The chart's slow *skill* line deliberately stays on the mean.
- **`serveRank`** = `min(mean + excess, displayRank + serveGapCap(attempts))` — what selection targets.

`excess` = `max(0, se − seFloor)`, so all three collapse onto the mean at convergence and neither offset leaves permanent bias.

**`reanchorRank` (`rank-decomposition.js`) is the single writer of all three slots** — attempt-recorder, account-sync and boot all route through it — so they can never be written under different rules. It is a pure function of the history, hence idempotent and safe to re-run. `main.js` re-anchors once **before** the first problem load, because a stored rank is read (shard pick, badge) before it is rewritten.

**The population prior is mandatory with no fallback.** An unanchored estimator still returns a number; it just tracks wherever it last drifted, so the failure is invisible. The prior is **never** the user's own previous estimate — that is the overshoot root cause, since it damps steps but anchors level to nothing, and early windows are frequently all-correct or all-wrong where the unpenalized likelihood has no interior maximum.

**A fresh device's entire cold start is the same rule**: `coldStartEstimate()` (`shared/rank-window.js`) is the ordinary window estimate over an empty window, which is the population prior exactly. There is no starting-rank constant — `state.js` computes it once at import and each getter falls back to its own slot of it. `estimateWindowRankFromFlat` returns **null** on an empty window so callers leave a stored rank alone, which is why the cold start is a separate export.

**The `displayRank` getter must fall back to the stored `userRank`, never to the `userRank` getter.** That getter's own fallback is the posterior mean, so chaining through it puts the population average on a brand-new user's badge — the one thing the display offset exists to prevent.

### The serve gap

**Serving must not simply follow the display down.** Measured, it starves the estimator: successes on far-below problems carry no information, the standard error never falls, the display never recovers, and a beginner sits several ranks under their true rank for hundreds of attempts.

**The gap resolves to the minimum of a multiple of the excess and a cap, so the cap — not the confidence term — is what binds through onboarding**, and therefore what sets how fast a strong user is found. `serveGapCap` is one plus the current run of correct answers, plus a bonus while the window holds no miss at all; it is reset by any miss and is never zero.

It counts **novel attempts only**, since `flattenAttempts` filters `isRepeat` and `linked` before the array reaches it (the Worker side uses `is_repeat = 0` in the leaderboard buffer). That is deliberate: a missed *review* is evidence about forgetting, not about serve difficulty.

**The point of the streak form is the asymmetry, obtained with no branch and no threshold.** A weak player misses often so their run stays short and the cap stays near its floor, while a strong player's run grows monotonically and the cap stops binding within about ten attempts. Measured on a fixed-ability simulation over the shipped corpus, the share of the first thirty problems landing more than three ranks above the player: streak 13.3%, constant-5 12.0%, **attempt-count 64.9%** — so `cap = attempts` reaches the same speed by serving beginners far over their head, and is the specific thing this form exists to avoid.

**Do not put the cap back to a constant.** A constant does not merely limit the bonus, it **inverts** it — display plus a constant is *below* the mean whenever the excess exceeds half the constant, which is all of onboarding at any plausible value, so the app serves under its own best guess and starves as above. Measured at a constant 5: an unlimited-strength player needs about 38 first-time attempts to reach a dan badge against about 25 under the streak rule, and a simulated fixed-ability 20k clears far more of what it is served than the target. The accepted cost of the streak rule is that the badge-versus-served gap is bounded only in distribution rather than by a constant.

### The SE floor is a shipped curve, not a runtime scan

`pwa/tsumegos/se_floor.json`, written by `package_for_pwa` and read through `seFloorAt` in `shared/rank-window.js`, which `estimateWindowRankFromFlat` applies internally — **no caller passes a floor function**. **Its knots are the rank prior's own grid (`PRIOR_X`), and that is load-bearing**: `windowMLE` returns a weighted mean over that grid, so no reachable rank falls outside the knots and the curve is never extrapolated. The derivation, the grid, the kernel and the two sparsity mechanisms are `methodology.md` §2.3 and §3.5; the geometry is single-sourced in `utils/rating/se_floor.py`, imported by the step that *chooses* the grid and the step that *emits* the curve so the two cannot drift apart.

**Do not go back to measuring it per reader.** Three costs, in order of size. **The dominant one is the reader that holds nothing**: with no pool the floor cannot be priced at all and is conceded to zero, which lands the shown rank `2·floor` low — measured on real histories, most users a whole band or more. The next attempt recomputes it with a pool loaded, and the overlay reports the recovery as a *promotion* to a band the user already held. Second, the readers disagreed with each other: a client holds only the shards it loaded that session, which cluster around the *serve* rank rather than the mean the floor is priced at, while the Worker and the offline rebuilds each built a different index. Third and smallest, a sample mean drifts with the estimate — and that one must not be oversold: of the band crossings a converged user sees, measured 25 of 231 came from the display offset moving and 206 from the posterior mean itself. Re-crossing a band edge and being told each time is **correct** behaviour; the target is the crossing that no answer caused.

**The lookup clamps to the curve's range**, so a future grid change degrades to the end knot instead of silently returning infinity — a linear extension of interpolated information goes negative within a few stones, and non-positive information maps to an infinite floor, which drops the excess term and shows the raw mean with no symptom at the call site.

### Ranks are derived client-side, and the rank is never clamped

`package_for_pwa` ships per problem the **raw** rating, the raw slope and the floor — **no rank at all**; shard bounds are raw too. The client does all rating→rank conversion: `problem-loader.js` derives `p.rank = applyM(rating)` on ingest and leaves `p.rating` raw. **Don't reintroduce a baked rank into shard metadata — `rating` means raw everywhere.**

**The kyu/dan ladder is what we label, not a bound.** `applyM` / `interpExtrap` extrapolate linearly past both end knots, so problems easier than the weakest knot fall below 30k and the unattempted engine-hard tail runs past 9d. Treat the rank as an unbounded continuous coordinate: don't clamp it, don't read it as a percentile. Clamping lives only where **labels** are built.

**The curve is required** — `main.js` awaits `loadRatingRankMap()` at init and aborts if it cannot load, with no fallback.

Answer-history entries store `problemRating` (raw), `problemRank` (internal) and `problemSlope` (the raw fitted slope); `backfill.js` refreshes all three, triggered by a single `ratings_version` — the joint cross-board fit re-rates both boards together, so there is no per-board rating version — **and** by a client-computed `ratingRankCurveVersion`, so a curve re-fit with unchanged ratings still refreshes stored ranks. The gate and the stamp live **once** in the orchestrator, wrapped around the per-size work loop, so stamps are written only after every size's walk has persisted — stamp-last is crash-safe, and an interrupted run re-runs idempotently.

### One internal rank scale, in one module

**`pwa/shared/rank-math.js` owns it, and there is no local copy.** The convention: **0 is the 1k/1d boundary, every rank is one unit wide, and integers are band edges** — 1d = [0, 1), 2d = [1, 2), 1k = [−1, 0), so "2 dan" averages 1.5.

**It is AGA-style, not AGA.** Real AGA ratings carry a (−1, 1) gap and put 2d at [2, 3), so crossing into that frame stays a real conversion: `agaRatingToRank` (`src/leaderboard.js`) and its pipeline mirror `aga_rating_to_value` (`utils/rating/rank_surveys.py`), one label-preserving shift, used by every AGA reader.

The module holds `interpExtrap`, `rankToLabel`, `rankLabelToRank` (label → band **centre**, so the "minus a half for the lower edge" idiom stays literal), `rankBand` (**floor**, since integers are edges), `rankProgress`, and `clampRankBand` (the labelled ladder, for **labels only** — never clamp a rank you compute with). It is imported by **both** the Worker and the client deliberately, so the server-computed community rank histogram cannot drift against the client's display.

The curve file's key is `ranks`, so a stale reader looking for `rank_indices` fails loudly rather than silently double-shifting. **Any change to this scale must be label-preserving by construction**, verified value-by-value across ratings, (rank, server) pairs, the federation tier gates and synthetic users through the real `computeRank` — and shipped behind a state-epoch bump, because every **stored** rank changes units and a stale value is not detectably wrong.

## The candidate pool and the serve draw

The method — why the pool is a contiguous run, why the draw is categorical, and why one criterion at a time — is `methodology.md` §2.4.

### The pool

**Candidates come from the unbroken run of loaded shards containing the serve rank's own band — never from `state.allProblems` at large.** Contiguity plus reaching the covering band is what makes the run's nearest fifty trustworthy, and a **partial** run stays safe on the same argument, with the error against the true fifty bounded by one band's width.

**The pool at large is not safe, and a deep link is how it goes wrong.** Opening a shared problem loads that problem's shard and nothing else, since the linked path never assigns a challenge. With the whole pool as the candidate set, any reassignment during that session — answering the standing challenge is enough — drew the next challenge from one arbitrary off-rank shard, and `setChallengeProblem` **persists**, so the pick outlived the session. Measured on the live corpus: a 3d who opened a 2k share was handed a 2k challenge.

`selectChallengeCandidate` returns null when the run is empty, and `assignNewChallenge` then awaits `expandPoolForCandidate`, which probes the service-worker cache **before** spending a network attempt so an offline device keeps playing from the bands it holds. **A run that is merely short is not a reason to block** — it is already a correct neighbourhood, so selection serves it and widens in the background; blocking instead stalls a serve that had a good answer in hand.

**`warmBeyondCandidateSpan` loads the shards just outside the bands the current fifty occupy** — not the serve rank's neighbours, because the *window* is what must be fillable, so the window is what we widen around. It is **self-limiting with no cap**: bands are contiguous in rating and the curve is monotone, so once the band beyond an edge is loaded everything in it is farther than the edge band's own problems, the fifty cannot reach past it, and the prefetch runs dry. Without it the run only grows after a serve has already found it too small, putting a download inside the answer-result overlay — simulated over 300 serves, 3–20 blocking loads against 1.

**Only an empty run after expansion serves off-rank**, via `anchorToLoaded` under `enterOffRankServing()`, which re-anchors to the nearest loaded band while keeping the run property, and tells the user. The recently-answered exclusion applies on **every** path including that one.

**Measured-rejected, don't add:** predicting the post-answer rank for a correct and an incorrect outcome and preloading those bands. An answer moves the rank far less than a band is wide, so both outcomes already sit inside the widened span; same blocking count at small drift, worse at large, and more shards downloaded.

### The draw

`problem-loader.js` makes **one categorical draw over a slot list**. `rankConverged()` (fail-closed) gates only whether the **information** slot is in that list — **not** whether motifs serve at all.

`weaknessServeCategories()` (`weakness.js`) flags every (axis, **named** tail) whose motif reading clears the shared `MOTIF_Z`. The draw hash-picks one slot uniformly — deterministic, seeded on the last attempt date so a resumed tab re-derives the same pick — and argmaxes that axis's tail loading `max(sign·emb[k], 0)` alone.

**The slots are the flagged categories plus one neutral no-criterion slot, always** — `slots = categories.length + (withInfo ? 1 : 0) + 1`, with **no lone-motif special case**. So *n* flagged motifs take one in *n*+1 each once converged, no motif exceeds half, and every user gets some share of serves drawn under no criterion.

**Don't go back to letting two or more motifs cap each other with no neutral slot.** That reaches the same bound only if distinct motifs select distinct problems, and the rules are **not** chosen for orthogonality — their tails overlap in the corpus, so two correlated flags can both argmax into the same corner and leave the mixture with no un-criterioned component at all. The neutral slot is also the only part of the draw blind to a motif's **corpus abundance** at the user's rank, which is strongly non-uniform (`motif_axes_workflow.md`), so it is what bounds how far that abundance carries into a served history.

**The information slot draws uniformly from the top `INFO_TOP_FRACTION` of the pool by `quality`, rather than argmaxing it.** A plain argmax is an extreme-value pick and therefore **a motif choice nobody asked for**: the single best-scoring problem of fifty is atypical on every axis at once, and measured it served the moderate `ko`/`tenuki` tail at about two and a half times the rate the pool contains it. The pool's quality curve is flat at the top, so the slice costs far less information than its width suggests. A residual tilt survives on a motif whose loading genuinely correlates with discrimination; residualizing quality against the pool's own loadings would remove that too and is deliberately not done, at a measurable information cost for a second-order gain.

**The information criterion is itself a ko selector in the single-digit-kyu and low-dan range**, which is why it *retires* at convergence rather than persisting as a criterion, and why the neutral slot is nearest-to-rank rather than quality. Simulated over difficulty-matched pools, the argmax is a ko problem 97–100% of the time across the mid kyus (pool rate 23–34%) and essentially 0% at the weakest ranks. Both factors do it independently, and the within-band correlations are modest — **an argmax lives in the extreme tail, where a modest correlation is enough to make the pick near-deterministic**, so don't read a small correlation as a small selection effect. Measured on the live log with the motif tilt disabled (`WEAKNESS_TILT_ENABLED = false`), a converged 1d user fell back onto this criterion and took 53% ko against a 19% rank-matched expectation.

**Don't put the motif slots back behind `rankConverged()`.** A flag is a property of the user's own picked-versus-correct contrast, identified independently of how tight the rank posterior is, so gating on convergence withholds personalization from exactly the users whose rank is still moving — while `motif-bubble.js`, reading the *same* categories, already names those motifs as the training focus.

**Don't go back to sampling proportional to any weight.** Measured on the shipped corpus, over fifty difficulty-matched candidates a criterion's weights span so narrow a range that a softmax at the old temperature left the effective pool at 49.9 of 50 — a hundredth of a rank of tilt costing the entire candidate set. Variety comes from pool turnover instead, every answered problem being excluded for the length of the recall-candidate window.

**`tenuki` is barred from selection by name** (`SELECTION_EXCLUDED`, `weakness.js`) while remaining fully **reported**. The hazard the bar guards is a category carried by most users at once — near-constant, hence collinear with the intercept, so its deficit launders into the rank estimate instead of steering anything. Under a fixed even-split tendency null that hazard was fully realised (nearly every user flagged); the availability null attributes most of that universality to the problem menu, so the bar stays only until flag-rate homogeneity is re-measured under the corrected null. `flaggedMotifCategories()` is the reporting list the bubble reads; `weaknessServeCategories()` is that list minus the excluded.

**Only a named tail can describe a category, and each carries five possible kinds** (`kind` on the category). *Weak* and *underplay* both argmax the named tail's loadings — drill the deficit, drill recognising when it applies. *Overplay* argmaxes the **opposite** direction, problems whose answer is *not* the motif, because that is where reaching for it actually costs the user: a **redirect**, not a de-prioritisation, since skipping those problems would skip the very ones that expose the habit. *Condition* — the judgment reading's positive slope — is two-sided by construction and emits **two** categories at once, one in each direction; only the toward-the-motif one carries the `SELECTION_EXCLUDED` bar, matching how *overplay* escapes it. *Strong* argmaxes the opposite direction as a plain suppression, and a negative judgment slope emits nothing. `motif-bubble.js` drops *strong* from the training-focus list — "you are good at this, serve less" is a serve direction, not something to work on — and its per-motif best-line map collapses the two *condition* entries to one line.

Cold start (few attempts with vectors): nothing flags, neutral serve.

## Recall

The model, the equilibrium and the measurement against the attempt log are `methodology.md` §2.8; the evaluation harness is `auxiliary/eval_recall_model.py`.

**A due recall wins a serve decision only with probability `recallServeShare()`** (`recall-model.js`), the constant-volume equilibrium share solved at a fraction of the user's trailing-week daily average. **Note the direction is the opposite of what the fraction suggests**: the working window is counted in *attempts*, so a lower assumed volume makes the served share **rise**.

While the share is not estimable — no model, no week volume, too few recent novels, or no rank — due recalls always win, and the gate **fails open** on any estimator error (`recallGateAllows` in `problem-loader.js`; `selectNextProblem` never throws).

The roll happens **before** the recall is built, so a gated-out recall costs no shard fetch, and it is **deterministic** — hashed from (most-due candidate id, last recorded attempt date), never `Math.random()`. The tab-visible refresh must re-derive the same decision rather than re-roll, since independent resume rolls would compound into many chances to swap a due recall onto an idle novel. Every new attempt re-seeds, so genuine serve decisions stay independent draws.

**The gate paces only unprompted serves.** An explicit user request bypasses it: notification clicks (`?recall=1`) and the Recall button (`showDueRecall`), which serves the most-due candidate exactly as the automatic path would and leaves the standing challenge for afterwards. That button takes the last button slot while solving — Share keeps it in analysis and linked mode — `updateSolvingModeUI` / `updateAnalysisModeUI` in `ui.js` show exactly one, so positions don't jump — and carries a badge of `waitingRecallCount()`, the due candidates **minus the one on screen**, since a served recall is not waiting. It is disabled with nothing due, while a recall is already shown, and while `acceptClicks` is false; the count is recomputed per repaint, so it is only as fresh as the last `updateAllUI`.

**`stabilityDaysForRepeat` has no upper clamp — don't reintroduce one.** Graduation *is* the streak's exponential interval outgrowing the recall-candidate working window; a ceiling turns graduation into a perpetual recycle, so the pool can only grow and serving drifts toward all recalls. Overflow needs no clamp, and a non-finite stability maps to "never due", which the schedule builder skips explicitly.

### The fitter is damped Newton, cold-started, and carries no optimizer state

It is the only client fitter left. `optimizer.js` holds the shared step — linear solve, Armijo backtrack, and the **Newton decrement** as the convergence measure. `recall-model.js` fits its two heads as **separate blocks**: the correctness head on the **expected** (Fisher) information — a four-parameter log-likelihood with a non-zero guessing floor is not concave everywhere, so the observed Hessian need not be positive definite where the expected one is — then the retention constant alone on its exact second derivative, which is non-negative for both outcomes.

**One pass over the two blocks is exact and no alternation is needed**, because `predictAndGrad`'s stop-gradient means the correctness block's objective does not involve `LN_C` at all.

**The decrement — not a loss delta — is the stopping rule, and that is the whole point**: the objective is anisotropic in the direction the model is *read* for, so a loss-delta rule stops while that direction is still moving, and the decrement is curvature-weighted so it does not. Measured against an exhaustive reference, a first-order budget reported convergence while sitting far away in parameter space — worth a third of `SIZE_IS_13` (the 13×13 readout) and half of `REVIEW_TIME` at a realistic window — where Newton reaches that reference to within a rounding error in a handful of steps, about ten times faster in wall-clock.

Consequently there is **no yield-slicing, no work-time budget and no coalescing queue** on the retrain (it is synchronous and takes milliseconds), and **no refine loop on the result overlay**: taking extra optimizer steps until the displayed delay agreed with the answer only ever worked because the fit was stopping short, and from a converged fit the remaining disagreement is the two heads genuinely disagreeing.

**Cold start is deliberate.** The fit is a pure function of the window, so two devices with the same history agree, and an account adoption or a state-epoch wipe is the **ordinary** path rather than a rare one. A separable window (all correct or all wrong) sends the unpenalized intercept to the `RIDGE_GUARD`-bounded saturation point, which is harmless — the intercept is a nuisance parameter.

The standard-error step accumulates the same Fisher matrix directly rather than forming a dense weighted product through a matrix library: bit-identical output, and it — not the fit — was the dominant cost of a retrain.

**`adamState`, `weaknessAdam`, `weaknessParams`, `weaknessSe` and `weaknessAxisCov` remain in the wipe lists as legacy blobs. Nothing writes them.**

## The motif readings

`motif-stats.js` computes **three independent readings per axis**, all pure functions of the user's own window and the stored per-problem availability nulls — no trained model, no prior, no population constant — so two devices with the same history agree and an account adoption is the ordinary path:

- **Tendency** — of the user's **wrong** attempts, the share whose picked cell loaded *higher* on the motif than the correct cell did (`d = correct_emb[k] − picked_emb[k] < 0`), tested against the **availability null**: each error contributes its problem's own null toward-probability `altAbove/(altAbove+altBelow)`, so the test is Poisson-binomial (`z = (nOver − Σp)/√Σp(1−p)`) and the display prints `share − nullShare` with the track's mid-mark at the null share. **Availability must enter**: within one problem the offered directions are wildly asymmetric (a tsumego's answer is by construction not a tenuki, so nearly every alternative loads higher there), and a fixed 50% null reports that menu as a lean of the player's — measured on the attempt log it flagged tenuki over-reach for ~94% of testable users, of which the availability null explains essentially all. A row without the null quartet contributes nothing; there is deliberately no 50% fallback, which would mix two neutral points into one statistic.
- **Condition** (the Judgment pane) — standardise each error's pick against its problem's null, `t = (altMean − picked)/altSd` (negative = leaned toward the motif more than a typical error), and regress `t` on the correct answer's loading with an intercept. The **slope** is the reading: positive = leans toward the motif where it is *not* the answer and away where it *is* — the both-at-once deficit a single pooled share cancels to nothing — negative = errors track the condition better than a typical error. Under the null both coefficients are zero per problem, so the OLS z is honest for any problem mix. The slope is **mirror-invariant** (negating the axis negates both variables), so no tail flip applies anywhere downstream. The intercept is fitted, not reported — net lean is the tendency's job; fitting it keeps the slope orthogonal to it.
- **Strength** — the shipped response model's residual (`correct − predicted`, the prediction from the problem's own rating/slope/floor and the user's rank) regressed on the motif loading, converted to ranks by the curve's local `dP/drank`. This half genuinely **needs** the difficulty control, because motif loading correlates with problem rank, so a bare correlation between loading and outcome would score a user weak at whatever motif happens to live in hard problems.

Why each has that shape is `methodology.md` §2.9. What binds from elsewhere:

**There is no shipped weakness constant of any kind** — no ridge prior, no population centre, no empirical-Bayes pass. `fit_population_constants` fits none and `population-constants.js` asserts none. Every reading stands against a neutral point of the user's own: zero ranks of residual effect, the lean a direction-blind error would have shown on their own served problems, a slope of zero against the answer's loading. The availability null is per problem and computed client-side from the published per-cell bin (uniform over the offered alternatives — deliberately parameter-free), so there is nothing to length-check against K and nothing that can go stale across an axis refit.

**Don't reintroduce a population baseline to "correct" either reading.** The two things it would correct for — how prevalent a motif is at the user's rank, and what the typical player scores — are exactly what make the reading about the user's *rank* rather than their play. Measured over 487 users, rank explained 15.5% of the between-user contrast variance on `ko` and 11.7% on `territory`, so part of what a population-scored pane called a tendency was the user's rank. Worse, a ridge centred at zero while the read-time statistic subtracted a non-zero population centre flagged a spurious deficit for essentially every user. Both defects are structural to scoring a personal quantity against a population point; neither is fixable by a better constant.

**The displayed band is the gate's own width (`MOTIF_Z`), not a conventional 95%.** At any other multiple a flagged row can show a band straddling the neutral point — a reported finding contradicting the number printed under it. It is locked so that "band clear of the neutral line" **is** "flagged", by construction.

**`MOTIF_Z` is one constant deliberately gating three things** — problem-motif display in `motif-bubble.js`, the serve flag, and the statistics pane's admission bar `normCdf(MOTIF_Z)` — locked so the three cannot drift apart. They are z-scores of *different* statistics, corpus salience versus posterior confidence, so retuning it anywhere retunes all three.

**The gate is confidence only, with no minimum effect size**: a bias the user is reliably making is worth reporting however small.

## Versioning and reload

**`build.mjs` writes `pwa/src/version.js`** so the Worker imports the same hash the client gets via the esbuild define. Every `/report` response stamps `X-App-Version`; if it differs from the client's, the client sets `state.reloadPending`. `maybeReload` fires from `setAcceptClicks(true)` only when `appMode !== ANALYZING`, so the refresh always lands **between problems, never mid-review**. New client features that depend on the Worker version ride this signal.

**`computeVersion()` folds a hash of every inlined data file into the app version** so the reload signal moves on a data-only re-ship: `pwa/tsumegos/` is gitignored, so a version derived from the git revision alone would be blind to a repackage from the same commit. The separate content `data_version` that cache-busts served files is `client_sync_and_cache.md`.

## The statistics modal

`showDetailedStats` → `showStatsModal(tabs)` renders Summary / Weaknesses / Rank / Time / Community pages; a pane with no content is dropped, and the tab bar is omitted when only one remains. `stats-ui.js` holds the rendering, `recall-model.js` holds only the numeric model and its statistics — the dependency is one-directional, UI → model.

Both charts derive their `viewBox` height from the visible modal area (`chartViewBoxHeight`) so each fits whole when scrolled. The Time chart is a Gaussian-kernel-smoothed daily solving and review-time trend — a no-data gap **breaks the line** rather than reading as zero — with a multiplier bootstrap confidence band (empirical, robust to idle-tab outliers) and daily problem-count bars on a secondary right axis.

The hamburger menu (`index.html` plus `updateMenuItems` in `ui.js`) is grouped into Practice / Review / More; the Review group shows only in analysis mode.
