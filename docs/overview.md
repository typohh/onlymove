# System overview — what each part is for

<!-- Editing rules: docs/contract.md. Present tense, no changelog, rules here and evidence with them. -->

This document describes the shape of the project from a goal and responsibility point of view. It contains no algorithms, no constants and no measurements — for how the numbers are actually constructed read `methodology.md`, and for how to run things read the root `README.md`. Read this one first if you are new, or if you are about to change something and want to know what else it touches.

## What the product is

onlymove.net is a problem trainer for the game of Go. It serves one position at a time on a 9×9 or 13×13 board — a whole-board position from play with exactly one winning move, of whatever kind the position calls for — and the player answers by clicking an intersection. In return the app tells them their kyu/dan rank, chooses what to show them next, reminds them to revisit problems they got wrong, and reports what kind of Go skill they are weak at.

There is no curriculum, no lesson text and no human-authored content of any kind. Every problem is machine-generated, every difficulty is measured rather than assigned, and every personalized number the player sees is computed on their own device. That last point is the organising constraint of the whole system: the backend's job is to publish a static, impersonal description of the problem set, and the client's job is to turn that plus the player's own history into everything personal.

## The four subsystems

**`pipeline/` — manufacture and measurement.** Its goal is to end each run holding a set of Go problems that are known-good, and a number for each one saying how hard it is, how sharply it separates strong players from weak ones, how often it can be guessed, and what kind of skill it exercises. It gets there by generating candidate positions with a Go engine, discarding the ill-posed ones, having bots attempt them, and fitting a model over the log of what real players did. It is a batch system: it runs on a workstation or a rented GPU, takes hours, and its output is files. Nothing in it is online and nothing in it serves a request.

**`utils/` — the shared vocabulary.** Two halves with a deliberate one-way boundary: the KataGo side never imports the rating side, and importing the rating side loads nothing from the KataGo side — the few engine helpers it needs (reading an SGF tree to its conclusion, converting a coordinate, simulating a board) are reached only inside the functions that use them. `utils/katago/` knows about Go: how to drive the engine, what a legal move is, how coordinates convert, how an SGF game tree is built and read. `utils/rating/` knows about the statistics: what a problem's features are, what the fitted model looks like, how the artifacts on disk are laid out, how a rank curve is fitted. Keeping them apart is what lets somebody work on either side without loading the other's assumptions; `utils/constants.py` holds the handful of values genuinely common to both, and imports from neither.

**`pwa/` — the product.** A browser app and the Cloudflare Worker that hosts it. Its goal is to be usable offline, on a phone, with no account signup, and still deliver a personalized experience. That drives most of its design: the problem set arrives as downloadable chunks the device caches; the player's identity is a short code rather than a login; the models that decide what to show next run in the browser; and the server's role is reduced to logging attempts, letting a player's devices agree with each other, and a handful of small aggregate features that genuinely need a central view (the leaderboard, community statistics, push reminders). The Worker is the only part of the system that holds live user data, and it holds it in one Cloudflare D1 database.

**`analysis_server/` — the fallback brain.** Most positions ship with the engine's own evaluation baked into the file, so the app can explain a variation without asking anybody. Where that is missing, the app asks this service, which wraps a Go engine behind a small HTTP interface. It exists so that review mode never has a dead end. A companion `worker.py` lets a faster machine dial in and take the service's work over while it is running, which buys depth of search rather than speed; the app is unaware of which machine answered.

Alongside those: `auxiliary/` holds diagnostics, surgery scripts, the rating-model research surface and `vast_orchestration.py`, which rents a cloud GPU for the three heavy generation steps; `ablations/` holds the repeatable measurements that stand behind the shipped choices. Nothing in the main flow reads either.

## How the parts hand off

The system is a one-way chain with a single narrow interface in the middle.

**Engine → problems.** The generator asks the engine for positions and keeps only those with exactly one winning first move. Identity is content-derived and symmetry-aware, so the set cannot contain the same problem twice under rotation or reflection, and a problem's name never changes.

**Problems + human attempt log → numbers.** The rating fit is the one place the two sources meet. It reads what the engine thinks about each position and what real people actually did on it, and produces the per-problem quantities. Everything downstream reads those numbers and never re-derives them.

**Numbers → published set.** One step packages the corpus into the archives and sidecar files the app downloads, keyed so that a client can tell whether what it holds is current. This is the narrow interface: the published set is impersonal, versioned by content, and the only thing the client ever fetches.

**Published set + the player's own history → everything personal.** The client computes the displayed rank, the next problem, the review schedule and the weakness readings locally. The server never computes any of these for a player mid-session.

**Attempts → back to the log.** Each answer is posted to the Worker, which stores it and hands back enough for the device to know its place. That log is the input to the next rating fit, closing the loop.

## Who owns what

**The backend owns the description of the problems, and nothing about people.** Ratings, slopes, floors, motif vectors and the rank curve are properties of positions and of the population as a whole. They ship to every device identically.

**The client owns everything about one player.** Their rank, what they are weak at, what is due for review, what to serve next. These are recomputed from the player's history rather than accumulated, which is why adopting an account on a new device and syncing an existing one are the same operation, and why there is no per-user state on the server that could drift.

**D1 owns the attempt log and the small central features.** The per-attempt rows, the account bindings, push subscriptions, the review schedule the reminder job walks, self-reported ranks and the community aggregates. It is deliberately a log rather than a state store: the rows are append-only, and anything derived from them is derived on read.

**The device owns its own copy of history.** Recent attempts sit in local storage where they can be read synchronously; older ones move to a browser database. Sync makes a device's copy a mirror of the server's rows rather than merging two opinions.

## What is deliberately absent

Knowing what the system does *not* do prevents most wrong turns.

- **No accounts in the usual sense.** No email, no password, no session. A short code identifies a profile and anyone holding it holds the profile. Signing in with Google is a way to look that code up again, not a second kind of account.
- **No server-side personalization.** The server does not decide what a player sees. If you find yourself wanting it to, the reason is usually that some client state was not made durable enough.
- **No per-user model state anywhere.** Every personal quantity is a function of the player's history. Nothing is carried forward, warmed up or annealed, so nothing can be corrupted into a state it cannot leave.
- **No test suite and no CI.** Verification is manual: run the pipeline step and read its diagnostics, or run the app locally and exercise the path.
- **No automatic deploys.** Publishing the data and deploying the app are two separate manual acts, and they can be out of step.
- **No curriculum, no difficulty tiers, no hand-labelled tags.** Every label in the app is either measured or generated.

## What a change in one place implies elsewhere

- **Changing how problems are generated** changes the corpus, which invalidates the fitted numbers for the new problems only; the fit is incremental over the set but the model is refit whole.
- **Refitting the ratings or the motifs** changes numbers already stored on every device against individual attempts. Those devices have to be told to throw away and recompute what they hold, which is what the version stamps in the published set are for. This is the single most failure-prone seam in the system.
- **Changing the rank curve alone** is cheap: it re-labels existing ratings without touching the packaged problem archives.
- **Changing the published file layout** requires the client to notice, which again means a version stamp — a silently reshaped file is read as valid data of the wrong shape.
- **Changing anything the Worker computes about ranks** must move in step with the client's own copy of that logic; the two share modules precisely so they cannot disagree.
- **Removing a problem from the corpus** does not remove the attempts people made on it. Its identity is retired rather than freed, so it can never be handed to a different position.

## Where to read next

| Question | Document |
| --- | --- |
| How are ratings, ranks and motifs built and consumed? | `methodology.md` |
| What was measured while building the rating model? | `mirt_benchmark.md` |
| How do I run the pipeline, and what does each step need? | root `README.md`, then `pipeline_operations.md` |
| What guarantees does the problem set carry, and how is the engine driven? | `corpus_and_engine.md` |
| How does a motif axis get taught, named and shipped? | `motif_axes_workflow.md` |
| What are the client's rules about state, storage and re-derivation? | `pwa_client.md`, `client_sync_and_cache.md` |
| How is the write path protected, and how is the local DB pulled? | `worker_operations.md` |
| How do the leaderboard and community stats work? | `leaderboard_and_community_stats.md` |
| How do I safely operate on the live database? | `db_surgery_runbook.md` |
| What is the front-end's file layout? | `pwa/README.md` |
