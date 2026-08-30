# Motif axes: the shipping contract, the naming workflow, and the published layout

<!-- Editing rules: docs/contract.md. Present tense, no changelog, rules here and evidence with them. -->

The **method** — what the axes are, why they are taught rather than derived, and how the client reads them — is `methodology.md` §1.14 and §2.9. The **evidence** — every sweep, ablation and identity measurement behind the choices — is `mirt_benchmark.md`. This document is the operating layer between them: what has to agree with what before a fit ships, how a direction gets found and named, and what the published files look like. CLAUDE.md carries the short form and points here.

## The shipping contract

**`K_AXES` in `utils/rating/model_constants.py` must equal the number of `components` in `axes/labels.json`.** The count is the *shipping decision* — one axis per taught rule — and not a fitted hyperparameter: the combined loss cannot adjudicate it, while split-half reliability separates the rules several-fold and can (`mirt_benchmark.md`).

**The two are kept in step by hand and the failure is one-sided.** `train_mirt` **aborts** when `components` is shorter than `K_AXES`, and silently **truncates** when it is longer — so a rule dropped from the tail of the list ships an axis that is still fitted and still consumes a serve slot on the client. Check the list length whenever either moves.

Nothing downstream needs a matching count: every consumer reads the shipped K from `pwa/tsumegos/embedding_meta.json` rather than assuming one, so changing `K_AXES` needs no client edit and no `fit_population_constants` re-run.

### Selection by rule name is safe here, and is not safe for the store

A rule name is unique by construction (asserted on `LABEL_SET`) and identifies exactly one column, resolved through the proxy npz's stored `rules` array rather than by position. `utils/rating/motif_rules.py` is also the single source for what each motif is **called** — `RULE_LABELS`, which `package_for_pwa` resolves shipped tails through.

The exploratory store cannot be read that way and never should be. It holds three directions called `counting`, three called `opening`, and a `ko` that is one entry's *positive* tail and another's *negative* — so a name lookup there fixes neither the direction nor the sign, which is why the store is keyed by content hash. The converse also holds: a rule column has one entry per problem and is recomputed as the corpus grows, so it cannot be content-hashed stably, and a hand-assigned id would just be the rule name with a prefix.

### The shipped set, and how a rule earns an axis

**The shipped set is five one-sided rules — `ko`, `yose`, `tenuki`, `life and death`, `territory` — chosen on RELIABILITY, not on likelihood.** A rule earns an axis by how fast a user's *position* on it becomes identifiable: split-half reliability against history length, summarised as the attempts at which reliability reaches 0.5 (`auxiliary/rating_dev/split_half.py`). That follows from what the axes are **for** — reporting weaknesses and tilting selection, both of which need a user's coordinate pinned from a history somebody actually has. Measured over four paired draws the candidate rules split into two tiers that hold at every K tried and that no K rescues, with the working tier several-fold ahead. The combined loss cannot decide this: paired, the K levels move it by less than the draw-to-draw spread.

**`LABEL_SET` is the CANDIDATE POOL, and membership is not vetting.** A rule can be defined, labelled over the corpus and carried in `axes/motif_proxies.npz` without ever having been through the reliability measurement — `semeai` is one, added to sit in the region `life and death` rejects by construction (`mirt_benchmark.md`). Nothing ships until `components` names it, so the cost of an untested rule is one boolean column; the mistake to avoid is reading a rule's presence in the pool as evidence that it earned anything.

**`tenuki` ships against its own evidence, deliberately.** Its reliability figure qualifies, but its reliability *curve* is flat across three banks and twelve seeds, which says between-user variance is about zero and makes the figure a curve-fit artifact. There is no pipeline gate testing that any more — the empirical-Bayes pass that did was removed with the population constants — so `tenuki`'s between-user variance is now checked, if at all, by measuring the client's own readings across users. It is also barred from *selection* by name while remaining fully reported (`pwa_client.md`).

### `components` syntax

`components` pairs rules onto axes in axis order. `"A/B"` puts rule A at the positive tail and rule B at the negative; a bare `"A"` is one-sided. Columns are glued as `f(A) − f(B)` with the shared tail weight, so a problem at neither pole contributes about zero rather than cancelling two plateaus, and each column is z-scored because scale is arbitrary — a rescale is absorbed by the user's own axis position. Each fit stamps the taught specification into `axes.npz`.

### Labels are optional to ship, and an unnamed axis is silent rather than broken

`package_for_pwa` prints which components are unlabeled and ships them anyway. An unnamed axis steers selection and carries its share of the weakness reading exactly like a named one; the **client** declines to print it — `stats-weaknesses.js` requires the flagged tail to be named (**both** tails, tested at candidate time so an undescribable axis doesn't consume the false-flag budget) and `motif-bubble.js` skips an unnamed tail.

**There is deliberately no placeholder label.** "You are weak at style #2" is not a finding a user can act on, so the pane says nothing rather than something meaningless.

**A one-sided component's negative tail is untaught residual**, ships unnamed, and the client stays silent on it — but a negative score is still a real position on the axis, read as such by the model at both ends. Don't read symmetry into it, and don't read an unnamed negative tail as a missing label.

## Measuring identity: tail-weighted agreement

**Single-sourced in `utils/rating/tail_agreement.py`.** Z-score each direction, weight it `z` above `TAIL_HINGE` = 1.0 and `at·exp((z−at)/at)` below, and correlate. `auxiliary/rating_dev/bank_identity.py` scores a fit against a bank (`--cross` for the axes × poles matrix a bipolar bank needs); `auxiliary/rating_dev/tail_overlap.py` scores stored motifs against each other. `train_mirt` prints the same statistic at the end of every production run, which is why the objective and the report are the same shape.

Why this form and not the obvious ones:

- **A name is a claim about one END of an axis**, and an axis is a trunk output rather than a linear map on nameable inputs, so the extremes and the bulk are free to diverge. **Bulk correlation gets answers backwards** — it reversed which of two arms held identity better, and reported a bipolar column collapsing onto one pole where the tails show every pole retained.
- **Top-N overlap is also wrong.** N is arbitrary, the cutoff is hard so problem N and N+1 are total agreement and total disagreement, and membership discards magnitude, so it manufactures matches out of problems that merely landed inside the boundary. Against a *binary* rule column it is worse still: the tied zeros make `argsort` pick an arbitrary tail.
- **The hinged form has no N** and is **exactly zero under independence**, so there is no chance baseline to quote. The two pieces meet in value and slope, and that smoothness is what makes the threshold safe: a hard floor collapses everything below the cut to one plateau, and the cut then *decides* the answer — two arms swapped places between z = 0.5 and 1.0. Sweeping the threshold from 1.0 to 3.0 moves agreement by about 0.01 and never reverses an ordering. **Do not tune the threshold to move a result.**
- **A tail that is constant after hinging returns NaN, never 0.** An empty tail makes no claim and must not read as "measured, independent".

Everything is scored on **attempted** problems: about half the corpus is never attempted, loadings there are unconstrained extrapolation, and the enrichment is one-sided — the untaught tail of every axis runs far more never-attempted than the taught, named tail does.

### Three rules for reading an agreement figure

**Agreement is a one-way test.** High agreement establishes that two directions name the same thing. **Low agreement establishes nothing** and must never be read as a motif having failed to transfer: a motif can be present and expressed along a different direction, and a K-axis fit structurally cannot span a bank of more motifs than it has axes. The way to test whether a stored motif survives is to *teach* it and measure retention.

**Below about 0.75 the name does not survive.** The measure is calibrated against human inspection at exactly one point, and it is not where the numbers suggest: six legacy poles retained at 0.43–0.73, which reads as success, while a human reading the boards found four of six mutated or unrecognisable. For contrast `ko` reproduces at 0.91–0.97 across fits and keeps its name. **Treat agreement as ordinal only, and anything under about 0.75 as a direction that must be re-inspected and probably re-named before it ships.** That applies to a rule-taught axis too — a rule is a starting point, not a guarantee that the axis means it. Detail in `mirt_benchmark.md`.

**Agreement measures co-selection, which has two causes it cannot separate**: one direction under two names, or two real motifs whose situations co-occur in Go. `escape atari` and `opening` share a third of both tails because on a small board escaping atari is frequently the opening answer, and they remain distinct — hold opening-ness fixed and escape atari varies *more* than corpus-wide. Treat a high-overlap pair as a pair to **examine**, never as a redundancy; the conditional check is what decides, and without it the measure will condemn perfectly good motifs.

### There is no input-attribution tool, and that is a property of the model

Naming an axis by what *drives* it needs the axis to be a linear map on nameable inputs, so that a problem's value decomposes exactly into per-column contributions. The model puts a trunk between the features and the loadings, so no such decomposition exists — any per-input number would be a surrogate model's opinion presented in the one place a human is deciding what a motif *is*. **The boards are the evidence.**

## Motif occupancy over rank

**Motif occupancy is strongly non-uniform over rank, and a near-zero correlation does not say otherwise.** Measured on the shipped corpus for `ko`, the corpus ko share runs about 0.6× its own base rate at the weak-kyu end up to about 1.8× around 4k, with a peak-to-trough ratio of roughly three; both boards show it independently and the taught proxy column reproduces it, so it is a property of the corpus rather than of the fit.

The pooled correlation between ko loading and rank is nonetheless about +0.08, because the shape is **non-monotone** — down through the weak kyus, up to the low kyus, flat after — and a linear statistic cancels it. So the rating-orthogonality penalty, which penalises *tail* correlation with rating, slope and floor, is doing its job and constrains none of this. **Never read a low axis-vs-rating correlation as an evenly-spread motif.**

The practical consequence is that a flagged motif's serve pool is genuinely thin at the weak end and rich around single-digit kyu — a difference in what the criterion can find, not just in a headline number. The selection consequences are in `pwa_client.md`.

## The naming workflow (development only)

Production always teaches from the rules. This is exploration, and doing its two stages at once is what makes a fit hard to read.

**Stage one — get a direction to exist.** Two routes, neither needing to produce a shippable axis set:

- A **handcrafted rule** labels the corpus from a stated criterion and seeds a motif nobody has seen yet.
- An **uninitialized fit** (no taught directions) is let go and names whatever it lands on. This is how `escape atari`, `defend territory` and `capture stone to gain liberty` were found.

Rules bootstrap a motif that does not exist; a free fit finds one nobody thought to write a rule for. Either way a human reads the extremes — `auxiliary.rating_dev.inspect_axes --dump …` renders both tails straight from `axes/per_problem.csv`; `--rank-max 0` caps the output for legibility only — and names what is there. Each accepted direction goes into the store with its provenance (`auxiliary.rating_dev.update_motif_store`), the durable record that a direction was found at all.

**Stage two — glue directions onto K axes.** `auxiliary.rating_dev.build_bank_from_dumps` composes a bank from store entries: one motif per axis, or two per axis as a bipolar column. Nothing is re-derived; this stage only decides what goes together and in what order. `components` in `labels.json` is the production analogue and works the same way over rule names.

**Do not try to discover and glue in one step** — a fit asked to both find directions and satisfy a prescribed set does neither cleanly.

**A free fit does not reproduce itself, and that is expected.** The objective is exactly invariant to rotating the axis space, so nothing prefers one basis and two free seeds have no reason to agree — measured, two seeds of one configuration agree at 0.56 / 0.25 / 0.20. So **a direction named off one free fit is a candidate, not a settled motif**: name it and store it, but expect the next fit to need it taught rather than to find it again.

**The store stays tracked because a direction found once is otherwise never rediscovered.** Nothing in production reads it.

### Expect rating-orthogonal axes to be unfamiliar

They are a different slice of the residual, not a cleaned-up version of a familiar one. The motifs and the rating parameters are genuinely entangled — ko-ness is partly guessability, life-and-death is partly discrimination — so forbidding the correlation forbids those motifs and returns different ones. They still predict; they simply have to be named afresh.

### Measured-rejected, don't re-attempt

Each of these is the natural next move and each costs a study to disprove. Full numbers in `mirt_benchmark.md`.

- **Teaching from previously-fitted directions instead of from the rules.** It fails in the way that is worse than failing: it half-works. Legacy poles return at 0.43–0.73, which reads as retention, while inspection finds most of them mutated — so the number that would certify identity is satisfied while the names it certified have come loose. And the round trip buys nothing to pay for that risk: a fitted direction is itself only an approximation of the rule it was taught from, so chaining through it adds a reference that moves. **Teach from the fixed reference every time.**
- **Residualizing the rule columns against rating, slope and floor.** The motivation is sound — a rule can be rating-dependent without being collinear — but it splits along the rule-tier line in the wrong direction: four of the five working rules degrade and only the two worst improve, with the combined loss worse. A rule's rating dependence is part of what the rule *means*.
- **Pinning the user axis positions at zero for an opening span** (a burn-in). The obvious companion to teaching, and measured to buy nothing on either identity or loss; it costs a couple of millinats over the opening few hundred epochs, repaid well before the end. There is no phase structure in the training loop at all, and the orthogonality penalty is on from step one.
- **An eigen warm start.** Loss-equivalent to random initialization at adequate patience, buying about four times fewer epochs and one particular arbitrary subspace — which is exactly what teaching exists to stop being arbitrary.
- **Sampling problems separately for the teaching term.** It rides the batch each attempt sample already computes; sampling separately makes the trunk adapt to a mixture of two input distributions rather than the one the rest of the objective sees. Riding the attempt batch also matches the population the axes are *defined* on, since loadings are identified by between-user residuals, which exist only where somebody attempted the problem.
- **Releasing the teaching pull before the end.** Released completely, a rule-taught fit does not keep the direction it was taught.

## The published layout

`package_for_pwa` ships **two things per shard and one corpus-wide sidecar**. The full artifact inventory is `methodology.md` §3.4; what follows is the part with an ordering contract attached.

**1. The correct-move vector, joined into the shard's `metadata.json`** under `embedding` — length K, what a drawn serve category's criterion reads on any candidate problem.

**2. A sibling `<hash>.embeddings.bin` per shard** — fp16, K floats per cell, all problems concatenated **in SGF root-variation order**, plus an `embeddings_index.json` inside the archive mapping `{tid: {offset, n_cells}}`. This is the picked-move sidecar: the client Range-fetches `bytes=offset+i·bytes_per_cell..offset+(i+1)·bytes_per_cell-1`, where `i = children.indexOf(moveNode)` over the SGF root variations.

**The bin's order is set at packaging time, not upstream, and the reorder is load-bearing rather than defensive** — the client indexes the bin by SGF-variation position, so shipping raw scan order would silently return the wrong cell's vector for every picked move. See `corpus_and_engine.md` for the three coordinate spaces and how they map. The `pass` variation, which the activation npz never embeds, gets an **all-zeros placeholder row** so that bin row *i* equals SGF child index *i* exactly. The revision that introduced the placeholder is `BIN_FORMAT_REV` 2; revision 1 skipped the pass, which shifted every row after a non-last pass by one — and the pass is not last in about half of 9×9 and three quarters of 13×13 problems.

**The revision is folded into the shipped `embedding_version`** (`shipped_embedding_version` = the content hash plus `-b<rev>`), so a layout change wipes and re-derives clients' stored picked vectors through the standing version gate, and rebuilds shards without `--force`.

**3. `pwa/tsumegos/lookup.embeddings.bin`** — the lookup-aligned correct-move sidecar. Row *i* is the correct-cell vector of the problem on `lookup.txt` line *i*. Lookup is id-sorted and spans both sizes, so the line index — which the client already has from its own binary search — doubles as the row address: no keys, no offsets. Emitted by `emit_correct_embeddings` on every non-sentinel run, straight from `per_problem.csv` plus the final `lookup.txt`, with no shard rebuild — the same way `problem_params.bin` is.

**Row alignment is the contract.** It must be written after the run's final `lookup.txt`, and it is folded into `data_version` so it can never cache-desync from it. The client additionally verifies the file size against its line count via `Content-Range` and poisons the sidecar for the session on mismatch. Ids missing from `per_problem.csv` get an all-zeros row the client treats as absent.

This sidecar is what lets the client's re-derivation passes fetch a correct-move vector as one small Range request instead of downloading the multi-megabyte archive whose `metadata.json` carries the same value.

**`embedding_meta.json`** carries `K` (the shipped axis count — read it, never assume it), `dtype`, `bytes_per_cell` (so the client can decode without guessing), `embedding_version`, and the two-tailed `labels`.
