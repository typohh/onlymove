# The documentation contract

**Read this before editing anything under `docs/`, or CLAUDE.md.** It states what these documents are for, what may go in them, how a claim must be phrased, and which document a new fact belongs in. Individual documents point here rather than restating it; where one adds a stricter rule of its own, that rule wins.

## What the documents are, and how they divide

| Layer | File | Job |
| --- | --- | --- |
| Operating | `CLAUDE.md` | Always loaded. What to run, what breaks, what not to re-try. Rules only — no derivations, no measurement trails. |
| Orientation | `docs/overview.md` | What each subsystem is for and who owns what state. No algorithms. |
| Method | `docs/methodology.md` | How ratings, ranks and motifs are constructed and consumed, and why each choice went the way it did. |
| Evidence | `docs/mirt_benchmark.md` | The development record for the rating fit: provenance, sweeps, ablations, reproduction commands. |
| Engineering | `docs/pipeline_operations.md`, `docs/corpus_and_engine.md`, `docs/motif_axes_workflow.md`, `docs/pwa_client.md`, `docs/client_sync_and_cache.md`, `docs/leaderboard_and_community_stats.md`, `docs/worker_operations.md` | File contracts, invariants, failure modes and the reasoning behind them, per subsystem. |
| Procedure | `docs/db_surgery_runbook.md` | Step-by-step operations against live data. |

**Keep the rule in CLAUDE.md and the evidence in `docs/`.** CLAUDE.md is loaded into every session, so its cost is paid on every task whether or not the task touches the subject. A note earns a place there only if acting without it would produce a wrong result — and then in its shortest correct form, with a pointer to the document holding the argument. The test for a CLAUDE.md line is not "is this true and interesting" but "would someone plausibly do the wrong thing without it".

**Nothing goes in two places.** A fact restated in two documents will be updated in one of them. When a rule needs to appear in CLAUDE.md *and* in a document, CLAUDE.md carries the imperative and the document carries the reason, and the imperative must be short enough that nobody is tempted to keep the two in sync by copying.

## The three tests every passage must pass

**1. The delete test — past findings belong here; history does not.** The distinction is what a fact is *doing*, not how old it is. A measurement earns its place when it is the *support for a present claim*: that a constant has the value it has, that an alternative was tried and lost, that a design choice is forced rather than arbitrary. It does not earn its place as narrative. Delete the historical sentence and ask whether some present statement is now unsupported. "This weight was chosen because the alternatives measured worse in these specific ways" survives. "First we did A, then B, and later C" does not — only C is the system.

**2. The phrasing rule — a rejected alternative is a standing option and its cost, never a predecessor.** The delete test is too permissive alone: almost any anecdote survives it once framed as justification, and the drift that invites is gradual. So the second test is grammatical. "A constant floor mis-scales because …" is evidence; "the constant it replaced sat near the sharp end" is the same evidence wearing a date, and the date does no work — the alternative is just as wrong for a reader considering it fresh. **The tells are temporal words: *previously*, *used to*, *the old*, *it replaced*, *once*, *no longer*.** A sentence needing one has usually stopped explaining the alternative and started recounting the switch; the repair is to describe what the alternative *does* rather than when it was abandoned. **Past tense is legitimate in one place only:** a measurement was taken at a point in time and on a particular snapshot, and saying so is part of stating it.

**3. The would-anyone-pick-it test.** A well-phrased alternative can still be dead weight. Both tests above are satisfied by a present-tense, evidence-backed description of an option no reader would ever have considered — and that passage still costs space and attention while teaching nothing. **Keep an alternative only when it is what a competent reader reaches for FIRST**: the simpler, cheaper, more standard or more obvious-looking choice, whose drawback they could not have predicted. An option that is worse on every axis a reader can already see, and no cheaper to build, is not a standing option; it is something that merely happened, and writing it up only tells them what they would have concluded unaided. The question is never whether the alternative is wrong — it is whether saying so saves anyone the trouble.

## No changelogs, no status

**These are not release notes and not status reports.** No dated entries, no migration stories, no "recently changed", no "as of the last refit", no record of who decided what and when. Dates and version references appear only where a claim is genuinely conditional on a snapshot.

**Never record transient state.** "Not yet deployed", "pending review", "currently running" — all of these are wrong within days and there is no mechanism that will correct them. A completed past fact is fine; an in-flight one is not.

## The code holds the values; the documents hold the reasoning

Where the two disagree, **the code is right and the document is stale**. So prefer stating what a number *means* and which way it points over transcribing a figure that moves every run. Name the module and the constant so the check is cheap.

**How to weight a quoted measurement.** Nearly every choice here is justified by something measured against the data, corpus and configuration as they stood at the time. They are not re-run when something downstream changes, and a sweep whose conclusion still holds may carry figures that no longer reproduce. So the *argument* — which alternative loses, in which direction, by what mechanism — is what the design rests on and what is meant to survive, while a quoted number is an order of magnitude and a sign, valid against its snapshot. **Where a figure would change a decision, re-measure it rather than trusting it.** Where a comparison is unresolved, say so.

## Where a new fact goes

Ask what kind of claim it is.

- **"Run this / don't do this, or you get a wrong result"** → CLAUDE.md, one or two sentences, plus a pointer.
- **"This is what the subsystem is for"** → `overview.md`.
- **"This quantity is built this way, for this reason"** → `methodology.md`, subject to its extra rules below.
- **"This arm beat that arm by this much, on this data"** → `mirt_benchmark.md`.
- **"This file/format/step behaves this way, and here is the failure it prevents"** → the matching engineering document.
- **"Do these steps in this order against production"** → `db_surgery_runbook.md`.

If a fact seems to belong in two engineering documents, it usually belongs in the one that owns the *writer* of the thing, with a pointer from the reader's side.

## Per-document rules that go beyond this contract

**`methodology.md` is self-contained and names no code in its method chapters.** A reader with no other document, no commit log and no prior conversation must follow it start to finish; nothing may lean on knowing what the system used to do or what was discussed when a decision was taken. Chapters 1 and 2 mention no module, file, artifact path, constant identifier or command, and never refer forward to chapter 3 — because those chapters are **published on the site**, generated from the file at build time with the scope section and chapter 3 removed, so a code name leaking into them reaches a reader with no repository to check it against, and a forward reference points at a section that is not on the page. Everything mapping the method onto this repository lives in chapter 3. Read that file's own "Scope" section before editing it.

**`mirt_benchmark.md` is a development record, and its provenance is load-bearing.** Pin the database hash and held-out set hashes before comparing any new run against it — a `pipeline.fetch_database` changes the holdout draw and voids the comparison.

**The engineering documents may name code freely.** That is their job: they are the bridge between a rule and the file it lives in.

## Formatting

- **Write paragraphs as single lines.** No hard wrapping to a column width — it makes every later edit produce a noisy diff.
- **Identifiers, paths, constants and file names in backticks**, so they can be grepped out of prose.
- **Bold the imperative**, not the whole sentence. A paragraph in which everything is emphasised has emphasised nothing.
- **State the direction of an effect, not just its presence.** "Widening the kernel imports the ridge and lands five times off its neighbours" is usable; "the bandwidth matters" is not.
- **Mark a refuted approach explicitly** — "measured-rejected, don't re-attempt" — so a reader scanning for what to try next can see it without reading the argument.
