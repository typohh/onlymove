# MIRT benchmark: provenance and results

<!-- Editing rules: docs/contract.md. Present tense, no changelog, rules here and evidence with them. -->

The fixed reference point for judging whether a change to `auxiliary/rating_dev/train_joint_mirt.py` is progress. Every number below is reproducible from the provenance in the first section; a future run that changes any of it is measuring something else and must say so.

The MIRT is now the production trainer (`pipeline.train_mirt`); this document is its development record, and is allowed history and alternatives where `docs/methodology.md` is not.

## Provenance — pin these before comparing anything

```
attempts DB      db/d1.sqlite
  sha256[:16]    267d103823b924a8
  rows           387,232 tsumego_reports   max id 405,591
  window         2025-08-20T23:04:55.761Z .. 2026-08-10T13:51:57.760Z
  rank_surveys   624 rows
corpus           153,720 problems with features; 135,434 with >=1 first-time attempt

held-out PROBLEM sets — auxiliary/rating_dev/holdout_set.py, frac 0.15
  seed 0         hash 6619fdc487345fc0   20,315 problems (9: 16,189  13: 4,126)
  seed 1         hash fc33d7e4c2dffc55   20,315 problems (9: 16,206  13: 4,109)
```

**A pinned copy of that DB is kept at `db/d1.baseline-267d103823b924a8.sqlite`** (gitignored via `/db/d1.baseline-*.sqlite`, verified byte-identical at copy time). `pipeline.fetch_database` treats `db/d1.sqlite` as its merge target and overwrites it in place, so without this copy the first routine fetch would silently retire the baseline — the held-out draw would move, the hashes would change, and nothing already recorded here could be reproduced or extended. To restore: `cp db/d1.baseline-267d103823b924a8.sqlite db/d1.sqlite`, then confirm `holdout_set --seed 0 --frac 0.15` still prints `6619fdc487345fc0`.

**A recorded hash is also reconstructible from a LATER DB, by attempt-id cutoff — which is what covers every result that has no pinned copy.** `pipeline.fetch_database` appends rows strictly above the local `MAX(id)` and rewrites nothing below it, so an id PREFIX of `tsumego_reports` is an exact earlier state of the table rather than an arbitrary subset, and an earlier snapshot is one integer. `auxiliary.rating_dev.reconstruct_snapshot` searches for that integer: given a hash and the kind it was drawn at (`problems`, `users` or `rows`, with seed and fraction) it descends the cutoffs from `MAX(id)`, carrying per-key attempt counts incrementally and recomputing the hash only where the key set moves. `--max-attempt-id` on `holdout_set`, `eval_user_holdout` (which carries it into the split-half histories) and `train_joint_mirt` then replays it. A pinned copy is one file per recorded result and covers only results someone thought to pin in advance; this reaches back to any of them, and the two coexist — the copy above resolves to cutoff 405,591.

**Two limits, both measured.** The hash identifies the UNIVERSE and not the cutoff — every cutoff between two change points draws the same set — so the tool reports the plateau and its largest member, and pinning the exact snapshot needs a second recorded quantity (the attempt counts an eval prints beside its hash do it). And **the cutoff belongs INSIDE the query, ahead of the dedup**: `load_first_time_attempts` keeps the earliest attempt per `(user, problem)`, and backfill inserts historical event times at the id frontier, so a row above the cutoff can carry an earlier `created_at` than one below it — cutting first keeps whichever row the snapshot really held, while filtering a deduped frame afterwards drops the pair outright. Which row survives never decides which users, problems or pairs EXIST, so all three hashes commute with the cutoff; `--verify` re-derives the universe through the real loader and checks that on the DB in hand rather than assuming it.

End to end, a fold-0 user-holdout hash recorded before a fetch comes back as a plateau of a few thousand ids that the eval's own printed attempt counts narrow to one, and `eval_user_holdout --max-attempt-id` there reproduces the recorded log line for line — forward walk, seen/unseen partition and split-half tables alike. Survey rho is the one figure outside the cutoff's reach: the declarations and the attempts behind them come in through `rank_surveys`, on its own reader.

**The DB hash is part of the identity of every result here.** `holdout_set` draws from the problems that have attempts, so a `pipeline.fetch_database` run changes the universe, changes the draw, and changes the hash — which is the intended tripwire, not a nuisance. Both models print the hash they trained under; if the two differ, the comparison is void.

Reproduce:

```bash
./.venv/bin/python3 -m auxiliary.rating_dev.holdout_set --seed 0 --frac 0.15
# or, from a DB that has moved on: find the cutoff, then pass it to every step
./.venv/bin/python3 -m auxiliary.rating_dev.reconstruct_snapshot \
    --target-hash 6619fdc487345fc0 --hash-kind problems:0:0.15 --verify
# the joint fit, at the shipped configuration
./.venv/bin/python3 -m auxiliary.rating_dev.train_joint_mirt \
    --k 3 --seeds 0 --holdout-seed 0 --lr-net 3e-4 --lr-field 1e-2 \
    --holdout-problem-frac 0 --holdout-user-frac 0.15 --dump-params /tmp/k3.npz
# score either
./.venv/bin/python3 -m auxiliary.rating_dev.eval_user_holdout \
    --dumps /tmp/k3.npz --seeds 0 --label MIRT
```

## CURRENT BASELINE — held-out USERS, 2 folds, Newton scorer — MEASURED 2026-08-13

Supersedes the 2026-08-12 table below, whose axis and combined columns were produced by the mis-scaled online axis estimator (see the scorer section). Both models at their own best configuration: the MIRT at current defaults (burn-in warm start at 300, lr 3e-4 / 1e-2, one barrier, 1-of-each filter, axis map reading the rating head's outputs), production as `train_rating_staged --production-flags` with its per-fold axis fit. Fold s = holdout seed s, model seed s.

```
                      rating           rating+axis        axis adds (mnat)
  MIRT            0.4930 / 0.4998    0.4911 / 0.4977      +1.89 / +2.09
  production      0.4955 / 0.5031    0.4919 / 0.5007      +3.66 / +2.41
  MIRT - prod       -2.5 / -3.3        -0.8 / -3.0        +1.77 / +0.32

  partitions of the same walk, axis adds:
    seen (81%)      prod 1.37 / 0.66     MIRT 0.93 / 0.44
    never trained   prod 13.57 / 9.70    MIRT 6.03 / 8.95

  worst |corr(v, theta)|: MIRT 0.081 / 0.053   (no axis is doubling as ability)
```

**THE MIRT LEADS ON THE COMBINED NUMBER, on both folds.** Its rating head leads by 2.5 and 3.3 mnat, as it always has; what changed is that its axes are worth ~2 mnat rather than ~0, so production's axis layer no longer cancels the rating advantage. The joint fit is ahead of the two-stage pipeline on the criterion that matters — a new user, predicted one attempt ahead from their first.

Two qualifications, both about the axis column. **The MIRT's axes are the more CONSISTENT** — +1.99 +/- 0.14 against production's +3.04 +/- 0.88 — so "production's axes are better" holds decisively on fold 0 (+1.77) and barely on fold 1 (+0.32). And **production's advantage is concentrated in the never-trained partition** (13.57 vs 6.03 on fold 0), which is 19% of attempts; on the 81% majority the two are close and both positive. The earlier finding that the MIRT's axes HURT on problems training saw was entirely the scorer.

Survey rho remains unusable on this holdout: MIRT 0.9123 / 0.7752 against production 0.9108 / 0.8061, on 92 and 67 declarations.

## SWEEP RESULTS — 2026-08-13, shared-trunk architecture, Newton scorer

All arms fold 0, single seed, against the fold-0 baseline. **Read the COMBINED
column, never "axis adds" alone**: a degraded rating leaves more residual for the
axes to explain, so axis adds RISES as the model gets worse — it read +10.25 on
the arm that destroyed the rating and +4.33 on an over-burned one.

**Same-fold seed noise, measured (fold 0, model seeds 0 vs 1, current build).**
Every earlier judgement used ~0.0004 rating / ~0.2 axis borrowed from the old
architecture and the broken scorer; the real thresholds are 2-2.5x wider, and
several calls had to be revised against them.

```
                seed 0     seed 1       sd
  rating        0.4930     0.4942     0.00085
  combined      0.4910     0.4916     0.00042
  axis adds     +1.93      +2.63      0.49
```

n = 2, so these are crude; a third seed would tighten them.

```
arm                  rating   combined   axis    verdict
  baseline           0.4930    0.4910    +1.93   --
  shallow_w 0        0.5154    0.5052   +10.25   CATASTROPHIC
  hidden 256 128     0.4988    0.4984    +0.35   worse, 6.8 sd
  hidden 64 32       0.4941    0.4927    +1.38   worse on combined (4 sd)
  burn-in 150        0.4964    0.4949    +1.46   worse, 4 sd
  burn-in 450        0.4931    0.4905    +2.55   1.2 sd — NOT established
  burn-in 600        0.4949    0.4906    +4.33   rating worse, 2.2 sd
  burn-in 750        0.4950    0.4914    +3.64   worse
  V_RIDGE 0.15       0.4934    0.4912    +2.15   indistinguishable
  warm start none    0.4951    0.4940    +1.10   worse, and 1360 epochs
  global slope       0.4948    0.4929    +1.88   worse, 1.8 mnat
```

**Settled, and why each is closed:**

- **Trunk 128x64 is an interior optimum**, bracketed both sides. Wider costs 4x what narrower does — the overfitting gradient the LR sweep found, appearing in capacity too. The value was inherited unswept; it turns out right.
- **`V_RIDGE` is not binding.** 0.05, 0.15 and 0.5 all score alike. It is not what limits the axes.
- **`SHALLOW_W = 0` reads as refuted in this snapshot, and it is what ships.** The refit panel on the shipping configuration reverses the verdict: without the barrier the held-out-problem score is marginally BETTER at every seed, and the blow-out it prevents is a small share of flat-slope problems whose ratings are left where they land (`docs/methodology.md` §1.7). This snapshot's own record: without it slopes collapse (median 0.94, 0.1th percentile 0.097) and derived ratings blow out to a range of [-225, +146], since rating = b + kappa(c)/a diverges as a falls. That is exactly the failure the barrier is written to stop. **RE-MEASURED on the shipping configuration** by the `barrier` refit panel (`ablations/panels/barrier.json`, replicated across seeds): the conclusion holds and the magnitudes do not — the blow-out is one-sided and an order of magnitude smaller, the held-out cost is a quarter of what is recorded here, and the size of the blow-out is itself not reproducible between seeds. Ten times the weight is NOT the failure direction this document assumes: it fits better held out. The panel's figures are the current ones and are ledger-backed; the numbers above stand as this snapshot's record.
- **The calibrated held-row loss does not separate barrier from no barrier, and without it the fit is less stable.** Production configuration, split rates, the same 179,600 held rows (`split_seed` 0), paired by initialisation seed, `held loss … shipped`, barrier 0.3 → none: seed 0 0.49267 → 0.48981, seed 1 0.49200 → 0.49496, seed 2 0.49181 → 0.49360. Flat-slope share (< 0.25) without the barrier: 1.28%, 1.40%, 7.67%; the seed-2 fit puts its 95th-percentile rating at +36 with a calibrated floor of 0.479 and a median slope × player sd of 4.24, against 2.04 and 2.19 at the other two. One shared rate without the barrier: 0.49668 at seed 0 and 0.49328 at seed 1 (split: 0.48981, 0.49496); with the barrier, seed 0, 0.49364 (split 0.49267). Seed 0 without the barrier is what ships. These are single fits per cell, read from `train_mirt`'s log, not ledger figures.
- **The per-problem slope head earns its place** — one global slope costs 1.8 mnat. This REFUTES, in this model, the on-record claim that per-problem slope is largely assignment noise; that finding is from the two-stage production model.
- **`BURN_IN_EPOCHS` stays at 300.** 450 looked like an architecture-shifted optimum at 0.5 mnat, but that is 1.2 sd against the measured seed noise. The bracket shape holds (600/750 degrade the rating for real); the claim that 450 beats 300 does not.
- **The warm start is what makes coupling SAFE.** Under raw features, dropping it left the rating untouched and only cost the axes. Under the shared trunk it costs the rating 2.1 mnat here, because a badly-initialized axis map sends its gradients into the trunk the rating depends on — and takes 1360 epochs instead of 380. The two changes are complements, not independent knobs. **See the warm-start section below**: re-measured under the production protocol at patience 30, where the cold arm needs 2670-3200 epochs to stop improving, the rating gap does not survive — 1360 epochs was very likely still descending.

## THE SCHEDULE AND THE BUDGET — COSINE AT 7000 SHIPS — MEASURED 2026-09-01

`EPOCHS = 7000`, `LR_SCHEDULE = "cosine"` (`pipeline/train_mirt.py`), replacing a constant rate at 3000. Both rates decay to zero over the budget with their ratio held. The panel is `ablations/panels/schedule.json`, 40 arms on one RTX 3080 Ti at attempt-log frontier 457,114, every arm at ONE pinned holdout draw (`split_seed 0`) so a between-run difference is the fit moving and not the scored rows; the producer is `ablations/refit_schedule_figures.py`, and this is the one panel that is NOT re-measured under the baseline it produced.

**The level is the FINAL level** — the mean held-out score over the last 2% of checkpoints — because the final epoch is what ships. Reading the constant fit at its closing-third mean instead overstates the cosine advantage by several millinats: the constant fit's problem-holdout score is still descending through that window, so its window mean sits above where the run ends. That misreading was made and withdrawn during this measurement.

Held-out PROBLEMS (whole problems held aside; the unit a claim about generalization is scored on), final level and run-to-run sd in millinats:

| arm | runs | work (constant-rate epochs) | final level | sd | vs constant@3000 |
|---|---|---|---|---|---|
| constant@3000 | 6 | 3000 | 0.50467 | 4.28 | — |
| constant@7000 | 3 | 7000 | 0.51139 | 6.29 | +6.7 |
| cosine@3000 | 3 | 1500 | 0.51179 | 6.41 | +7.1 |
| cosine@5000 | 3 | 2500 | 0.50775 | 6.82 | +3.1 |
| cosine@6000 | 3 | 3000 | 0.50679 | 3.11 | +2.1 |
| **cosine@7000** | 3 | 3500 | 0.50397 | 1.36 | −0.7 |
| cosine@8000 | 3 | 4000 | 0.50205 | 3.01 | −2.6 |
| cosine@10000 | 3 | 5000 | 0.50389 | 1.77 | −0.8 |
| cosine@12000 | 3 | 6000 | 0.50970 | 8.71 | +5.0 |

**Level: parity.** Pooled over the plateau (7000–10000, n = 9) the decayed fit is −1.4 ± 1.9 mnat against constant@3000. Nothing here is a level improvement.

**Spread: about halved.** Pooled within-budget sd over the plateau is 2.16 mnat (df 6) against the constant fit's 4.28 (df 5): ratio 3.9 in variance, one-sided F-test p ≈ 0.06. The constant fit's spread is not seed noise in the usual sense but the STOPPING EPOCH: within its six runs, the score at 80% of the budget averages 4.6 mnat above the final, and at 90% matches it — the shipped value is one draw from a ±4 mnat wander that the decay averages out. Bucketing adjacent budgets (5000+6000 vs 7000+8000) was tried first and reads the spread badly, because it discards the budget coordinate inside a bucket; the pooled within-budget residual is the right statistic, and a curve fit over budget (power law and exponential both, resid sd 4.0 ≈ the within-budget sd, so no lack of fit) cannot pin an asymptote from six budgets — the model-free isotonic read is what the window bounds rest on.

**The window is bounded above, and the mechanism is overfitting the decay locks in.** At 12000 the held-out minimum comes early (epochs 2980 / 3850 / 4830 across the three runs) and the final sits above it by 16.2 / 1.1 / 9.9 mnat while train loss keeps falling; the high-rate half overfits and the decay freezes the fit there. One 10000 run already shows it at 4.0. The learning-rate integral says why: 12000 cosine = 6000 constant-equivalent epochs, past the point where the constant fit itself turns (constant@7000 is +6.7). So a decayed budget is chosen inside the constant fit's own useful work range, and "longer" is not safer. **7000 is the cheapest budget that has arrived on both level and spread and the farthest from the cliff**, which also makes 8000 a worse hedge than it looks: its −2.6 is one seed set, 10000 is back at −0.8, and the isotonic fit pools 8000–12000.

**Cost.** Median elapsed 1278 s for cosine@7000 against 680 s for constant@3000 in the same shared pool — a budget ratio of 2.33, an elapsed ratio nearer 1.9. What the spread buys is that a two-seed paired contrast has half the standard error it had, which is the difference between resolving the 3–4 mnat effects the remaining panels are asked about and not.

**Held-out ROWS** (the trainer's own convergence split) are not the level's unit and were measured only for the within-run band: the cosine tail is flat by construction and the constant tail wanders by several millinats peak to peak, which settles nothing on its own and is why the between-run spread above is the figure that was decided on.

Reproduction: `auxiliary/vast_orchestration.py ablate schedule` (or the queue); `./.venv/bin/python3 -m ablations.refit_schedule_figures` prints the table above from the records.

## THE AXIS SPACE — FREE FITS, THE OPENING RAMP AND THE MUTUAL PENALTY — MEASURED 2026-09-03

Panel `ablations/panels/axis_space.json`: 20 arms on one RTX 4080 at attempt-log frontier 457,114, cosine@7000, EVERY arm at `split_seed 0` and `pca_seed 0` so the seed moves the network's initialization and minibatch order alone; four seeds per configuration, six pairs; producer `ablations/refit_axis_space_figures.py` (figures `axis_space.*`). Five configurations: the FREE fit (`teach_weight 0`, every other term as shipped), free + a 500-epoch linear ramp of both rates (`lr_warmup_epochs 500`, 249 constant-rate epochs less work), free + a mutual-orthogonality penalty (`ortho_axes 0.5`, the squared off-diagonal of the loadings' batch correlation matrix at the weight `utils/rating/residual_axes.py` uses for the same term), the SHIPPED taught fit, and taught + the ramp. Agreement is over the 107,088 attempted problems every arm carries; tails are the 1000 most extreme problems per end.

| configuration | matched axes mean / min | canonical lead / 3rd / min | tails shared, best / mean | effective dim | worst own pair | level sd (mnat) |
| --- | --- | --- | --- | --- | --- | --- |
| free | 0.24 / 0.05 | 0.63 / 0.37 / 0.01 | 22% / 14% | 3.2 | 0.96 | 2.1 |
| free + ramp | 0.23 / 0.02 | 0.60 / 0.18 / 0.01 | 17% / 12% | 2.3 | 0.99 | 1.2 |
| free + mutual penalty | 0.22 / 0.08 | 0.65 / — / 0.02 | 7% / 5% | 6.5 | 0.20 | 2.8 |
| **taught (ships)** | 0.93 / 0.89 | 0.98 / 0.96 / 0.80 | 29% / 35% | 5.3 | 0.41 | 2.4 |
| taught + ramp | 0.94 / 0.89 | 0.98 / 0.96 / 0.82 | 30% / 38% | 5.3 | 0.43 | 2.7 |

**The free space is underdetermined under the decayed schedule too, and it is not the constant rate's noise.** With the draw and the feature basis pinned, four free fits landing within 2.1 mnat sd of one another on held-out problems share one leading direction (canonical 0.63) and little else (third 0.37, weakest 0.01); matched axes agree at 0.24 and the best-matched axis shares 22% of its extremes. The same initial weights under the two schedules land at 0.21, no closer than two seeds do, so the endpoint is decided by the path rather than by where it starts.

**A free fit duplicates its axes, and the reason is in the objective.** Effective dimension 3.2 of 7 (2.4–4.0 across seeds), the most correlated pair of a fit's own axes at 0.89–1.00 — one axis another up to sign, so its opposite tails read as two motifs and are one. Neither the likelihood nor the ridge asks the axes to differ from each other: the ridge is on the centred positions and is isotropic, every axis is unit-RMS, so splitting a direction over two axes at half the position each halves the ridge cost at identical predictions. Every arm here ran without the trunk's mutual term (`ORTHO_AXES 0`; `utils/rating/residual_axes.py` carries the same term as `ORTHO_WEIGHT`); the `penalty` panel's free control, which also drops the rating-orthogonality penalty, collapses all seven axes onto one direction (every pair at |r| 1.00). The taught fit holds 5.3 distinct directions with its worst pair at 0.41: without a mutual term the pull is all that keeps the axes apart, and it does not hold seven.

**The mutual penalty restores the count and nothing else.** At `ortho_axes 0.5`: effective dimension 6.5, worst pair 0.20 — and matched agreement 0.22, tails 7%/5%, the least reproducible extremes of any family. It costs the RATING 13.1 mnat on held-out problems, paired at every seed (+16.2, +14.8, +11.1, +10.1; with the axis term zeroed +16.4, +15.0, +11.1, +10.3). Why is not separated here: the term's gradient reaches the shared trunk, which is one candidate route; the detach panel (below) runs the form applied through the axis readout alone, on the taught fit at half this weight, and it costs nothing there. A lower weight undetached is not run.

**The ramp does not fix the basis.** Free + ramp: agreement 0.23, effective dimension 2.3. Its effect on the LEVEL is the `ramp` panel's question (next section): on the four seeds here it read −0.71 / −2.73 / −1.34 / −0.26 mnat on the taught fit, which eight seeds resolve as the upper half of a spread centred near zero.

**Teaching costs the rating 4–5 mnat on unseen problems, on every seed.** Taught minus free, paired at pinned draw and basis: +9.14 / +1.00 / +6.39 / +1.50 (mean +4.5); with the axis term zeroed +9.21 / +0.77 / +6.01 / +1.33 — the cost sits in the rating's own outputs, the axis channel itself predicting held-out problems within ±0.2 mnat either way. The route is not separated (the pull's gradient into the shared trunk is the candidate). The `seeds` panel read this contrast as sign-inconsistent (+4.9 / −0.1) with the PCA basis re-rolling alongside the seed; pinned, it is consistent. What the pull buys is the identity in the table.

**On the users with enough history for a position to be detectable, free axes are worth as much as taught ones, and the mutual-penalty axes more.** `auxiliary.rating_dev.panel_axis_by_history` — the `eval_user_holdout` forward walk (ability and axis position both estimated causally from the prefix) over two account draws (seeds 0 and 1, 15% of accounts; 32,035 and 12,177 attempts in the 500+ bin), rating-only minus rating+axis, mnat, mean of the four fits per family:

| family | 500+ prior attempts (draw 0 / 1) | 100–500 | 30–100 |
| --- | --- | --- | --- |
| taught (ships) | +1.55 / +1.38 | −2.58 / −1.51 | −2.61 / −2.65 |
| free | +1.46 / +1.32 | −0.04 / −0.77 | −0.09 / −0.53 |
| free + mutual penalty | +2.87 / +5.63 | +0.19 / +0.35 | +0.11 / +1.11 |
| shipped production fit | +1.89 / +2.40 | −2.78 / −1.06 | −2.70 / −2.46 |

The arms held out problems, not users, so the drawn accounts' attempts reached every arm's problem parameters — mildly optimistic, identically for every fit on one draw, so the contrasts are paired and the absolute figures are not the methodology's clean user-holdout number. Two confounds the table cannot separate. The mutual-penalty family's larger axis gain co-occurs with a rating 13 mnat worse: the walk's rating-only baseline is each fit's own, so a worse rating leaves more residual for the axes to explain, and "better axes" and "worse rating" are not distinguished here. And the taught fits' mid-history LOSS is a property of this walk's estimator meeting taught loadings — the applied axis term grows several times faster for a taught fit (mean |g·v| 0.15–0.19 logits from 30 prior attempts on, against 0.02–0.06 free) while pointing the right way on 51–53% of attempts until past 500 — and the client's own estimator is not this walk.

Human inspection: `auxiliary.rating_dev.align_arm_axes` writes each arm as an `inspect_axes` dump with its axes matched one-to-one to a reference arm's (the same slot is the same direction on every page) and `inspect_axes --dump … --match-source taught` captions each free tail with the rule-taught axes it agrees with.

Reproduction: `auxiliary/vast_orchestration.py ablate axis_space`; `./.venv/bin/python3 -m ablations.refit_axis_space_figures` prints the table; `./.venv/bin/python3 -m auxiliary.rating_dev.panel_axis_by_history --arms axis_space/<arm> … shipped --seeds 0 1` the walk.

## THE OPENING RAMP — MEASURED-REJECTED, DON'T RE-ATTEMPT — MEASURED 2026-09-03

Panel `ablations/panels/ramp.json`: the shipped fit with both rates scaled linearly from near zero over the first R epochs (`lr_warmup_epochs`, multiplied into the cosine so the decay is untouched), R in {0, 250, 500, 1000}, EIGHT seeds per length at `split_seed 0` / `pca_seed 0`; 32 arms on one RTX 4080 at attempt-log frontier 457,114; producer `ablations/refit_ramp_figures.py` (figures `ramp.*`). Run because the axis_space panel's four-seed reading of R=500 came out better on every seed by 1.3 mnat — a same-sign-on-four-seeds result, which is a sign test at p = 0.125 and an interval spanning zero.

| ramp | work (const-rate epochs) | level | sd | gain vs 0, paired | ±95% | p | better | rating share | cross-seed basis | taught agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3500 | 0.5051 | 2.0 | — | | | | | 0.93 | 0.66 |
| 250 | 3376 | 0.5046 | 2.9 | +0.48 | 2.35 | 0.64 | 6/8 | +0.45 | 0.94 | 0.67 |
| 500 | 3251 | 0.5047 | 2.1 | +0.42 | 1.20 | 0.44 | 6/8 | +0.35 | 0.94 | 0.66 |
| 1000 | 3005 | 0.5047 | 3.2 | +0.35 | 2.23 | 0.72 | 6/8 | +0.29 | 0.94 | 0.66 |

Per-seed gains (mnat): 250: +3.29 +1.79 +3.02 +0.13 +0.26 +1.56 −0.62 −5.58; 500: +0.71 +2.73 +1.34 +0.26 −0.35 −2.32 +0.53 +0.44; 1000: +2.64 +0.24 +2.33 −1.74 +1.25 +1.99 −5.30 +1.38.

**Nothing at any length.** Every gain sits inside its interval at p 0.44–0.72; the best length (250) is interior, so the sweep brackets and no longer ramp is owed. The first four seeds of R=500 reproduce the axis_space reading to the hundredth (determinism across two boxes of one GPU model), and the next four land on the other side. Identity does not move either way. The no-ramp schedule stays; the knob stays for the panel. Reproduction: `auxiliary/vast_orchestration.py ablate ramp`; `./.venv/bin/python3 -m ablations.refit_ramp_figures`.

## WHICH TERMS ON THE LOADINGS NEED THE TRUNK — MEASURED 2026-09-03

Panel `ablations/panels/detach.json`: the shipped fit against the same fit with one term's path into the shared trunk closed — the term acts on loadings computed from a detached copy of the trunk output (`teach_detach`, `ortho_detach`, `ortho_axes_detach` in `pipeline.train_mirt`; same values, no second trunk pass, no gauge update) — plus the taught fit carrying a DETACHED mutual-orthogonality penalty at `ortho_axes 0.25`, and the free fit under an anisotropic ridge (`v_ridge_ladder 1.3`, geometric over the axis index). EIGHT seeds per paired configuration at `split_seed 0` / `pca_seed 0`, four for the anisotropic free fit; 44 arms on one RTX 4080 at attempt-log frontier 457,114; producer `ablations/refit_detach_figures.py` (figures `detach.*`). Gains are shipped level minus the arm's on held-out problems, paired per seed; identity is agreement with the taught rule ends, cross-seed matched-axis agreement over all 28 pairs, and the effective number of distinct directions.

| configuration | gain vs shipped (mnat) | ±95% | p | better | rating share | taught agreement | cross-seed basis | effective dim | worst own pair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shipped | — | | | | | 0.66 | 0.93 | 5.4 | 0.40 |
| teaching pull detached | +6.00 | 2.77 | 0.001 | 8/8 | +5.85 | 0.10 | 0.25 | 2.4 | 0.91 |
| rating-orthogonality penalty detached | −14.85 | 1.75 | <0.001 | 0/8 | −15.20 | 0.66 | 0.95 | 4.9 | 0.55 |
| both detached | +2.80 | 2.47 | 0.031 | 6/8 | +2.74 | 0.12 | 0.64 | 1.6 | 0.99 |
| taught + detached mutual penalty, 0.25 | +1.90 | 1.33 | 0.012 | 7/8 | +1.62 | 0.66 | 0.95 | 6.7 | 0.20 |
| free + anisotropic ridge, 1.3 (4 seeds) | level 0.4999, sd 1.3 | | | | | — | 0.20 (min 0.03) | 2.5 | 0.96 |

Per-seed gains: pull detached +10.40 +2.75 +9.28 +4.17 +7.86 +1.15 +7.88 +4.48; penalty detached −15.07 −14.91 −13.42 −15.32 −12.67 −19.38 −14.97 −13.08; both +6.45 +0.70 +6.91 −0.12 +3.86 +0.79 +4.22 −0.44; mutual +3.65 −0.35 +0.83 +0.99 +4.24 +1.00 +3.00 +1.82. Pooled run-to-run sd 1.7 mnat.

**The pull has to reach the trunk.** Detached, the rating recovers what teaching costs it (+6.0, every seed — the axis_space panel's +4.5 read on the same seeds), and the axes stop being taught: agreement with the rule ends 0.10, cross-seed 0.25, an effective 2.4 directions — the free fit's numbers. A rule direction is not linearly readable from a representation the pull did not shape. Untried: scaling the pull's trunk gradient down rather than to zero.

**The rating-orthogonality penalty's path into the trunk is load-bearing for the RATING.** Detached, the fit is 14.9 mnat worse on held-out problems at every seed, with the axes' identity untouched. The route is not separated; what is measured is that the term, acting through the trunk, does the rating a service its own heads do not. The section-1.5 argument for the penalty (the axes must not absorb the shipped triple) stands on identity grounds; this adds a level reason to leave its gradient where it is.

**A detached mutual penalty on the taught fit is better on every count the panel reads.** +1.9 mnat (interval clear of zero, 7 of 8 seeds, +1.6 of it in the rating's own outputs), seven distinct directions instead of five (worst own pair 0.20 against 0.40), identity and cross-seed agreement unchanged. Its axis VALUE is higher too, by the walk (`auxiliary.rating_dev.panel_axis_by_history`, the eight mutual arms against the eight shipped ones, paired by seed, two account draws): past 500 prior attempts +0.94 ± 0.42 (8/8, p = 0.001) and +0.66 ± 0.83 (7/8, p = 0.10) mnat over the shipped axes' +1.80 / +2.03; in the 30–100 band the shipped axes' loss shrinks by +0.61 and +0.79 (8/8 at both draws, p ≤ 0.004); over all attempts +0.66 and +0.58 (8/8, p ≤ 0.001). The weight is one point (0.25); the free fits carried 0.5 undetached and paid 13 mnat of rating for it (axis_space panel), so detachment is what makes the term affordable, and a weight sweep on the taught fit is owed before the value is treated as tuned. **It ships at that point**: `ORTHO_AXES 0.25` on the detached readout (`ORTHO_AXES_DETACH`) is the production default, the regime pin carries the term's weight and detachment, and every panel in this document was measured under the baseline without it — their internal contrasts stand, and only a comparison of their arms against a current-baseline arm is void.

**The anisotropic ridge does nothing.** The free fit under a 1.3 ladder reproduces no better (0.20) and still duplicates its axes (effective dimension 2.5, worst pair 0.96): a distinct ridge per axis prefers one basis among distinct directions, but two axes sharing a direction still pay less than one carrying it, so the duplicate solution stands and there is no ordering to stabilize. Measured-rejected.

Reproduction: `auxiliary/vast_orchestration.py ablate detach`; `./.venv/bin/python3 -m ablations.refit_detach_figures`.

## THE K PANEL AND THE RULE WIDTHS — MEASURED 2026-08-21

Nested rule banks at K = 4, 6, 8 — K=4 a subset of K=6 a subset of K=8 — four
seeds each, training seed and holdout seed moving together. Nested on purpose: an
arm that changes K *and* swaps rules measures both at once, so the banks differ
only in how many of the SAME ordered rules are taught. Ordered by measured N50:
counting, tenuki, ko, life and death | territory, opening | sente, shape. DB
f04c0848a6e1e6e1 (see the section below for the provenance rule).

N50 = attempts at which split-half reliability reaches 0.5
(`auxiliary/rating_dev/split_half.py`), mean over the four draws.

```
  motif              K=4     K=6     K=8       share
  counting           239     380     348        6.8%
  tenuki             542     411     493       16.4%
  ko                 812     745     518       15.4%
  life and death   1,020   1,241     751        3.9%
  territory            —     717     787        9.4%
  opening              —   2,628   3,607       19.2%
  sente                —       —   3,510       20.2%
  shape                —       —   8,220        5.0%
  CTRL (ability)      12      16      16

  rating         0.4987  0.4975  0.4975
  COMBINED       0.4989  0.4970  0.4965   <-- the column to read
  axis adds       -0.23   +0.58   +1.00   mnat (a trap, see below)
```

READ THE COMBINED COLUMN, and read it PAIRED. Every arm here runs the same four
holdout draws, so arm-vs-arm at the same draw cancels the draw noise that swamps
any single level — the standing rule that comparing arms wants the draw pinned
while estimating a magnitude wants it moving. Paired combined deltas, lower
being better:

```
  K=4 -> K=6   +1.90 +/- 1.43 mnat    per seed  1.6  4.0  1.1  0.9   ALL FOUR
  K=6 -> K=8   +0.45 +/- 0.44         per seed  0.3  0.9  0.7 -0.1
  K=4 -> K=8   +2.35 +/- 1.77         per seed  1.9  4.9  1.8  0.8   ALL FOUR
```

**K=6 beats K=4 on every draw. K=8 does not clearly beat K=6** — three of four
draws, and +0.45 against its own +/-0.44. So six taught rules is the supported
choice here and the eighth axis is unpaid for.

**Axis adds says K=8 is best and that is the trap this document warns about
elsewhere**: it rises monotonically with K (-0.23, +0.58, +1.00) because more axes
mechanically absorb more residual whether or not the model predicts better. On the
combined column the gain flattens after K=6.

Three of the four shared motifs are nonetheless best identified at K=8.
This REFUTES the over-teaching reading taken from an earlier K=5 arm, where extra
taught axes appeared to crowd each other: on the nested design, up to K=6, they do
not. What looked like crowding was two miscalibrated columns (`opening`, `shape`),
not the axis count. `counting` is the one motif that prefers a sparse bank, posting its
best figure anywhere at K=4.

**The rules split into two tiers that hold at every K** — working: counting,
tenuki, ko, life and death, territory (239-1,241); broken: opening, sente, shape
(2,628-8,220). No K rescues a rule from the second tier.

### WIDTH IS PER RULE. A SHARE TARGET IS NOT A DESIGN PRINCIPLE.

The module's stated aim of putting every rule in a comparable share of the corpus
does not survive measurement. Aiming all of them at ~20% helped one rule, broke
another, and left the two narrowest — never touched — as among the best.

```
  rule       width change          N50 effect
  opening    5.7% -> 19.2%         3,708 -> 2,628 (K=6)   mild help
  shape      5.0% -> 20.1%         1,982 -> 6,692         BROKE IT
  territory  21.5% -> 9.4%         1,364 ->   717         narrowing FIXED it
  counting   6.8%, never moved       239-404              best motif measured
```

**The test before moving a threshold is whether moving it changes what the rule
MEANS.** `opening` at prefix <=12 versus <=18 is the same concept, an early
position, with more of it admitted — safe. `shape` at winrate gap <=67 versus <=82
is NOT: at 82 the two best moves are no longer close, so the criterion stops
denoting its own name, and the fitted column came back anti-correlated between
halves at short histories (r@125 of -0.72 on every seed).

**The mechanism is how much the fit reshapes a rule, and it depends on the rule's
width.** Measuring the participation ratio of the fitted loading's positive tail,
`PR = (sum g+)^2 / (n * sum (g+)^2)`, which equals the share exactly for a uniform
indicator, so PR/share says whether the fit spread or concentrated:

```
  rule            share    PR/share   mass on rule   lift
  life and death   3.9%    1.56-1.75      59-65%    15.3-16.7x
  counting         6.8%    1.50-1.85      55-64%     8.1-9.5x
  ko              15.4%    0.89-1.32      74-92%     4.8-6.0x
  tenuki          16.4%    1.17-1.41      53-68%     3.2-4.2x
  opening         19.2%    1.03-1.13      82-89%     4.3-4.6x
  territory       21.5%    1.14-1.20      69-72%     3.2-3.4x
```

The fit SPREADS a narrow rule (1.5-1.85x its share) and leaves a ~20% rule
essentially as stated (1.03-1.20x). So a narrow rule is a CORE the model
elaborates outward and only needs its centre right; a wide rule is taken as
written and must already BE the concept. That is why `counting` works at 6.8% and
why `territory` had to be narrowed rather than widened. Compare lift only within a
similar share — `on_rule` is bounded below by the share, so lift mechanically
favours narrow rules, and the fair pairing is ko against tenuki at 15.4% vs 16.4%,
which ko wins decisively.

**`ko` is the rule the model adopts most faithfully** — the only one it
concentrates rather than spreads, and the highest mass-on-rule at 74-92%. It is
also the motif that reproduces across every arm, seed and ridge tested.

**`tenuki` should not ship as a skill axis on this evidence, whatever its N50.**
Its reliability curve is FLAT — 0.37/0.40/0.51/0.49 at K=8, and flat across three
banks and twelve seeds. Reliability is tau^2/(tau^2 + sigma^2/n), so a curve that
does not climb over an 8x increase in data says tau^2 is ~0 and the N50 is a
curve-fit artifact rather than a trait sharpening with evidence. Its mass lift is
also the lowest of the eight, so the fitted axis is not much "about" the rule.

### RESIDUALIZING THE RULE COLUMNS AGAINST THE RATING IS REFUTED

The idea is reasonable and the observation behind it is real: a rule can be
rating-dependent without being collinear with it, and `tenuki` is the clear case —
it fires on 3.6% of the corpus at the weak end of the rating range and 33% at the
strong end (point-biserial +0.236), because weaker players answer the last move.
`ORTHO_RATING` pushes the fitted AXES off rating, slope and floor, but nothing
constrains the TARGET, so a rating-laden target and the penalty are pulling against
each other. Residualizing every rule column against rating, slope and floor before
teaching removes that conflict at the source.

Arms `iso_ar0..3` (K=8, all eight columns residualized) against `iso_p8s0..3` (the
same bank raw), the SAME four draws, so every ratio below is paired.
Residualization did what it says — every target's rating/slope/floor correlation
goes to 0.000 — and the axes got WORSE where it matters:

```
  motif              raw   resid   ratio      p   per-seed ratio
  counting           348     399    1.18  0.148   1.17 1.28 0.93 1.38
  tenuki             493     578    1.17  0.312   0.80 1.48 1.22 1.31
  ko                 518     787    1.55  0.074   2.51 1.23 1.32 1.43
  life and death     751     824    1.12  0.595   1.50 1.06 1.49 0.66
  territory          787     700    0.85  0.363   1.16 0.76 0.60 1.01
  opening          3,607   5,941    1.59  0.336   0.87 5.13 1.04 1.37
  sente            3,510   1,460    0.49  0.079   0.27 0.35 0.72 0.83
  shape            8,220   2,482    0.32  0.133   0.13 0.63 0.39  (3/4 draws)

  combined logloss: resid WORSE by 0.68 mnat, p 0.230, per seed 0.1 -1.8 -1.0 0.0
```

**It splits exactly along the working/broken tier line, in the wrong direction**:
four of the five working rules degrade and only the two worst improve. `ko` — the
rule the model otherwise adopts most faithfully — degrades most, and it carries the
second-largest floor correlation, so what residualization removed there was signal
rather than nuisance. `tenuki`'s curve does not merely fail to climb, it inverts:
0.43/0.44/0.45/0.38 residualized against 0.37/0.40/0.51/0.49 raw.

Nothing here clears p<0.05 and with eight motifs nothing survives correction, so
the DIRECTION is the finding and the margins are not. But the direction is
consistent across the tier split and the combined column agrees, which is enough to
stop pursuing it: a rule's own rating dependence is not the thing to remove.

### THE PRODUCTION RULE SET IS THE SEVEN OF `comp7_core` — MEASURED 2026-08-30

`axes/labels.json` ships `life and death`, `yose`, `ko`, `opening`, `tenuki`,
`sente`, `semeai`, one rule per axis, unidirectional, so `K_AXES` = 7. **The
criterion is reliability, not likelihood**, and that is a decision about what the
axes are FOR: the axes exist to report a user's weaknesses and tilt selection, which
needs a user's position on an axis to be identifiable from a plausible history.
Combined logloss cannot adjudicate at this resolution anyway — paired, K=4 -> 6 -> 8
moves it by less than the draw-to-draw spread — while N50 separates the tiers
several-fold at every K tried, and no K rescues a second-tier rule.

**Provenance, and it is NOT the pinned one above.** These arms ran on the production
data as it stood on the measurement date — 298,501 attempt rows over 2,507 users —
with `--production` and no problem holdout, so the figures below are ARM-AGAINST-ARM
at a fixed seed and cannot be put beside the held-out tables in this document. The
comparison that carries them is the pairing across the two seeds, nothing else.

The arms are two seeds each, dumped to `dev_runs/<arm>_s{0,1}.{log,npz}`.
`comp7_core` carries the sweep's best TIME-ADJUSTED SURVEY RHO — highest on seed 0
at 0.8590 and second on seed 1 at 0.8581, and the only arm whose WORSE seed clears
0.855, against the nine-rule `all9`'s 0.8416 / 0.8454. Stop-split does not separate
the arms at all: comp7_core's 0.4822 / 0.4804 sits inside a field running about
0.480 to 0.488, the same resolution problem the K panel above reports. Read the
pairing rather than the margin — a single-seed rho gap this size sits inside the
seed spread elsewhere in the same sweep (`comp8_noopen` runs 0.8472 / 0.8634).

**Every axis in this set passes the CURVE test as well as the N50** (the test itself,
and why the N50 alone cannot make it, are in `motif_axes_workflow.md`). Scored on
`dev_runs/comp7_core_s{0,1}.npz`, N50 spans 501–1,441 on seed 0 and 521–2,527 on
seed 1 against the ability control's 102 / 109, and the ordering inside that span
does not survive the seed change — the set is what replicates, not the ranking.
`tenuki` climbs on both — r at a thousand attempts 0.72 / 0.53, trend +0.80 on each,
user-bootstrap interval there clear of zero ([+0.12, +0.89], [+0.27, +0.70]) — and it
climbs in the wide `all9` arm as well (0.08/0.42/0.42/0.53, N50 680).

## THE TRUNK'S INPUT IS ACTIVATIONS + BOT WINRATE — MEASURED 2026-08-30

`PRODUCTION_BLOCKS` = `"activations+bot"` (`utils/rating/model_constants.py`). The trunk
reads the joint-PCA activation block, the bot pass-rate block and the board-size flag;
the answer-derived blocks — move evals, ownership swing, game phase — are dropped from
it and keep their pipeline role as the material the motif rules are written over. **The
rating case is PARITY, so the decision rests on the per-cell side**: the same trunk is
fanned out over every legal cell, and there the tail is the CORRECT move's, so an
answer-derived block describes every alternative cell by the answer.

**Provenance, and it is NOT the pinned one above.** Both arms ran on the production data
as it stood on the measurement date — 299,320 attempt rows over 2,508 users, 220,463
problems with features, joint-PCA cache `joint_pca_dc739b061d817dd0.npz` — under
`--production`, no problem holdout, `--holdout-user-frac 0.15`, one model seed per fold.
Nine INDEPENDENT held-out-user draws, seeds 0–8, PAIRED: each fold's two arms share a
holdout hash and row count, so the arms may be differenced fold by fold and the spread
across folds may not be read as an error bar on the arm difference. Arms are
`dev_runs/featprod_s0-s8` (`--features production`, which selects the ALL-blocks layout,
feat 213 / dropout region 212) and `dev_runs/actbot_s0-s8` (`--features
activations+bot`, feat 182 / 181), scored by `eval_user_holdout` into
`dev_runs/eval_<arm>_s<n>.log`.

```
  combined (rating+axis) held-out-user logloss, forward walk (all)
  fold      all    act+bot     delta (mnat, + = act+bot better)
    s0   0.5076   0.5057        +1.90
    s1   0.5060   0.5034        +2.60
    s2   0.4885   0.4960        -7.50
    s3   0.4799   0.4822        -2.30
    s4   0.4907   0.4926        -1.90
    s5   0.4960   0.4963        -0.30
    s6   0.5129   0.5095        +3.40
    s7   0.4928   0.4923        +0.50
    s8   0.5142   0.5138        +0.40
  mean -0.36 mnat (sd 3.29 over folds, sem 1.10); attempt-weighted -0.09; act+bot ahead on 5 of 9
  rating column alone: mean -0.18 mnat
```

**Neither arm is separable from the other on likelihood.** The mean sits at a fifth of
its own fold-to-fold standard error, the sign changes five times across the nine folds,
and the largest single-fold gap (7.5 mnat on s2) is the same size as the axis-value
spread the next section reports for one arm against itself. A single fold quoted from
this table is a statement about that draw.

**Time-adjusted survey rho is flat too** — cross-fitted, per-server n-weighted, printed
by each training run: mean 0.8539 (all) against 0.8526 (act+bot) over the nine folds,
per-fold difference −0.0014 on average with four of nine positive, against a per-fold
spread of about 0.005 within each arm. Base rho is not quotable here (`CLAUDE.md`).

**What separates them is the per-cell reading, and it is net-neutral with the two
ownership axes gaining.** `ablations/tendency_split_half.py` scores the client's own
tendency reading — 347 users with at least 20 recorded errors, 38,356 attempts over
30,519 problems, halves by parity — with arm A the shipped all-blocks fit (fdim 213)
and arm B a dev checkpoint on its own full input:

```
  drho (arm B - arm A) per axis, arm B = dev_runs/actbot_s{3,4}.pt
  axis                s3                        s4
  life and death   +0.231 [+0.066, +0.392]   +0.087 [-0.067, +0.234]
  yose             -0.109                    +0.048
  ko               -0.123                    -0.150 [-0.294, -0.002]
  opening          -0.010                    -0.036
  tenuki           -0.152 [-0.276, -0.021]   -0.119
  sente            +0.003                    +0.075
  semeai           +0.127 [-0.029, +0.283]   +0.045
  mean drho across axes: -0.005 / -0.007
```

The two axes taught from ownership rules — `life and death` and `semeai` — improve on
both dev seeds, `ko` and `tenuki` lose on both, and the mean is zero to within a
thousandth. **The control says the tail is what moves them**: masking the wide tail on
the SHIPPED checkpoint (arm B = mask, activation scale 1/(1−`INPUT_DROPOUT`)) gives the
same mean drho of −0.003 while `opening` alone gains +0.104 rho / +0.235 r — so these
shifts are the conditioning changing, not a differently-converged fit.

**The mechanism, measured directly.** `ablations/tail_conditioning_sensitivity.py`
projects each problem's SGF root-variation cells twice through the all-blocks fit, once
on its real tail and once with the wide tail at its pooled mean, and compares the
within-problem sign against the correct cell at the client's own resolution (fp16 +
0.01 rounding). Of the alternative cells strictly signed in both arms, **10–36% change
side, depending on the axis** — the tendency test's observed direction and the
availability null's masses both live on exactly that sign. A tail acting as a monotone
contraction toward the correct cell's reading would cancel out of both; the trunk is
nonlinear and it does not.

### Reproduction

```bash
# one fold, both arms (repeat --holdout-seed/--seeds 0..8)
./.venv/bin/python3 -m auxiliary.rating_dev.train_joint_mirt --production \
    --features production --seeds 0 --holdout-seed 0 --holdout-user-frac 0.15 \
    --dump-params dev_runs/featprod_s0.npz
./.venv/bin/python3 -m auxiliary.rating_dev.train_joint_mirt --production \
    --features activations+bot --seeds 0 --holdout-seed 0 --holdout-user-frac 0.15 \
    --dump-params dev_runs/actbot_s0.npz
./.venv/bin/python3 -m auxiliary.rating_dev.eval_user_holdout \
    --dumps dev_runs/actbot_s0.npz --seeds 0 --label actbot_s0
# the per-cell reading, shipped fit against a dev checkpoint
./.venv/bin/python3 -m ablations.tendency_split_half \
    --arm-b-model dev_runs/actbot_s3.pt --arm-b-features activations+bot
./.venv/bin/python3 -m ablations.tail_conditioning_sensitivity
```

**`--features production` in the dev trainer names the ALL-blocks arm**, not
`PRODUCTION_BLOCKS`. The flag's choices are the names its recorded runs were made under
and are left alone so those runs stay reproducible; the shipped layout is the constant,
not the flag.

## THE AVAILABILITY NULL IS VISITS-LINEAR — MEASURED 2026-08-30

The tendency reading's null weighs each alternative by the visits its own SGF evaluation rests on, rather than counting the alternatives equally. Three measurements decide it, and the third is about which criterion is admissible at all.

**The criterion is the DIRECTIONAL log-loss, because that is the only thing the client consumes.** A null enters the reading as one number per problem per axis — the toward-probability `p` — scored against a binary outcome. So a null is scored on `p` against toward/away, per error, paired over the same errors across arms. Scoring instead on *which* of a decision node's ~58 wrong variations gets clicked (phase 1 of `eval_error_null_models`) measures a question the client never asks: concentrating the pick distribution on the right move says nothing about splitting the alternatives correctly ABOVE and BELOW the answer, and a null can win there and lose here.

**Every weighted null beats equal counting, and visits-linear is the best parameter-free one.** Equal counting scores 0.3257 directional log-loss. It is beaten on **7 of 7 axes with every confidence interval clear of zero**, and visits-linear is the largest parameter-free improvement at **−0.0292**. The *fitted* forms — a visits power with a free exponent, winrate×visits, and the rest of the panel — add only about **0.003** beyond that, a tenth of the margin the weighting itself buys. That does not pay for shipping a parameter into a reading whose whole claim is that it has none, and it would have to be refit and republished with every corpus, so **visits-linear ships and the fitted panel stays a measurement**.

**Split-half reliability is NOT the criterion, and the control is what proves it.** `tendency_null_variants` carries the retired fixed-0.5 null as a control row, and it scores as by far the most reliable arm — **+0.35 mean Δρ against uniform, on 14 of 14 (seven axes × two repeats) intervals**. It is also the arm that is known to be wrong, because it retains the per-problem menu confound entirely: the answer to a tsumego is by construction not a tenuki, so nearly every alternative loads higher on that axis, and a menu is a property of the rank band a user is served — stable within a user, so it replicates across halves exactly like a real lean. **Reliability therefore rises with retained confound rather than with measurement quality**, and the argument is not special to the fixed null: it applies to any pair of nulls differing in how much problem structure they divide out. Selecting a null on reliability selects for the confound. Validity is what is left, which is what the directional log-loss measures.

Reproduce (neither writes to the repo; both read the production corpus, not the pinned baseline DB):

```bash
# the adjudicating measurement — phase 2b, directional log-loss per axis
./.venv/bin/python3 -m ablations.eval_error_null_models --embeddings shipped
# the reliability control that disqualifies reliability as the criterion
./.venv/bin/python3 -m ablations.tendency_null_variants
```

## AXIS VALUE IS A PROPERTY OF THE HOLDOUT DRAW, NOT OF THE MODEL — MEASURED 2026-08-21

Every axis-adds figure in this document, the one below included, comes from one or
two holdout draws. Across FOUR draws the number is not stable, and it is not even
reliably positive.

Provenance — a LATER DB than the baseline at the top of this file, so these numbers
sit beside the rest rather than replacing them:

```
attempts DB      sha256[:16] f04c0848a6e1e6e1
  rows           409,043 tsumego_reports   max id 427,402
  window         2025-08-20T23:04:55.761Z .. 2026-08-20T10:02:25.774Z
  rank_surveys   693 rows
arms             iso_k6s0..s3 — K=6, store bank (bank_k6_unipolar), centred v ridge
                 --seeds s --holdout-seed s, s = 0..3, holdout-user-frac 0.15
```

The training seed and the holdout seed MOVE TOGETHER, so each arm is an independent
replicate: different initialisation, different batch order, and a different 85% of
users to train on.

```
                      seed0    seed1    seed2    seed3        mean ± sd
  axis adds            3.59     2.81    -1.40    -1.44     +0.89 ± 2.69  mnat
  never-trained       13.50    10.12    -0.30    -2.60     +5.18 ± 7.83  mnat
  rating logloss     0.5105   0.5110   0.4841   0.4919    0.4994 ± 0.0135
  survey rho         0.8118   0.8913   0.7361   0.6715    0.7777 ± 0.0950
```

**Two of four draws give a NEGATIVE axis contribution.** Averaged, the axes are
worth +0.89 ± 2.69 mnat — indistinguishable from zero, where a single draw reads
+3.59 and looks like a solid effect. The never-trained partition is worse still:
+5.18 ± 7.83, a sd half again the mean.

**Pinning the holdout seed is what hid this.** With one draw shared across arms,
every replicate scores the SAME held-out users, so the spread between them measures
initialisation and batch order only — the smaller of the two sources, and not the
one a claim about magnitude depends on. The seed count then reads as replication it
is not delivering.

**The right protocol depends on the question, and the two are opposite.** Comparing
ARMS wants the draw PINNED: same rows, paired, so the difference is the arm and not
the sample. Estimating the MAGNITUDE of an effect wants the draw MOVING, or the
interval is conditional on one sample of users and the quoted number carries no
statement about the next one. A figure produced under the first protocol must not
be quoted as though it came from the second.

**Identifiability and predictive value are separate properties, and only the first
is stable here.** The same four arms give split-half N50 (attempts to reliability
0.5, `auxiliary/rating_dev/split_half.py`) that reproduces tightly across the same
draws that swing the prediction:

```
  motif             N50 mean    N50 range over the 4 draws
  counting             398        [309, 493]
  ko                   580        [363, 831]
  life and death       871        [786, 982]
  territory            910        [523, 1,732]
  opening            1,958        [1,368, 2,574]
  shape              2,062        [1,185, 3,578]
  CTRL (ability)        13        [10, 15]
```

So a user's position on an axis is measurable, and measurably so at a consistent
cost in attempts, while whether knowing that position improves a held-out
prediction depends on which users are held out. Do not treat a stable reliability
as evidence for a predictive claim, or the reverse.

The scale worth carrying: ability reaches reliability 0.5 in ~13 attempts, the best
motif in ~400. Axis positions are roughly 30x more expensive to identify than
ability, which bounds what per-user axis personalisation can do for a user who has
not played hundreds of problems.

## AXIS VALUE BY USER HISTORY — the axes need ~500 attempts before they pay

Held-out-user walk binned by how many attempts PRECEDE each scored one, both
folds pooled (`auxiliary/rating_dev/axis_by_history.py`). **Two folds only, so
every figure here inherits the draw-dependence measured in the section above —
the +4.95 is one sample's value, not an estimate with an interval.** What the
binning shows about SHAPE (the payoff concentrated in long histories) is the part
to carry; the levels are not repeatable to the precision printed. This is the USER-side
partition; seen/never-trained describes the PROBLEM and is blind to it.

```
                    all      pfx 0-30   pfx 30-100  pfx 100-500   pfx 500+
      n =        104,096      14,470      15,529       28,371      45,726
  rating         0.4963      0.4440      0.5001       0.4925      0.5140
  rating+axis    0.4942      0.4440      0.5003       0.4926      0.5090
  axis adds      +2.12       +0.03       -0.13        -0.13       +4.95

  mean |predicted P(correct) - outcome|, percentage points
    rating        32.01       24.31       31.65        32.18       34.46
    +axis         31.61       24.38       31.70        32.05       33.59
    effect       -0.405      +0.070      +0.046       -0.138      -0.874

  accuracy at the 0.5 threshold, %
    rating        78.32       82.73       78.11        78.49       76.89
    +axis         78.48       82.76       78.11        78.53       77.21
```

**The axes contribute nothing below ~500 attempts and everything above it.** The
headline +2.12 is an average over a population where most attempts get nothing.
In probability terms, a user past 500 attempts gets their predicted chance of
success moved 0.87 percentage points closer to the outcome and 0.32 points more
predictions on the correct side of 50%; below 500, nothing.

Read the percentage-point figures against the floor, not against 34: binary
outcomes bound this metric below by mean 2p(1-p), about 34 points at a 78% base
rate, so the rating is already near the irreducible level and the axes take their
0.87 out of the thin remainder.

Two things this cannot separate. The 500+ bin conflates LONG HISTORY with HEAVY
USER — the same accounts — so "the axes need history" and "the axes work on people
who play a lot" are not distinguished. And the bin is unbounded, so where inside
it the payoff begins is unknown; a finer split (500-1000 / 1000-3000 / 3000+)
would say.

**It also reframes the model comparison.** Production's axis advantage is
entirely a 500+ effect (+8.58 vs +4.09 on fold 0); in the 30-500 band production
is WORSE than the MIRT on both folds. And it bears on the client, whose weakness
pane gates at 20 attempts — a different estimator, so not a direct read-across,
but the direction is worth checking against `weaknessModelReady`.

## K IS 3, BRACKETED BOTH WAYS — MEASURED 2026-08-13

Fold 0, single seed, shared-trunk architecture. K is baked into the shared head,
so each value is a full training run; there is no incremental path of the kind
the two-stage axis stage had, where a frozen 4PL let K grow cheaply.

```
  K      rating    combined    axis adds
  2      0.4936     0.4929       +0.64
  3      0.4930     0.4910       +1.93     <-- shipped
  4      0.4933     0.4910       +2.38
  5      not run
```

**K=2 costs 1.9 mnat (4.5 sd), so the third axis earns its place. K=4 gains
exactly nothing.** K=5 was dropped rather than run: with 4 flat there is no path
by which 5 helps.

**READ THE COMBINED COLUMN. The axis column here is a trap, and this sweep is the
cleanest demonstration of it in the whole document.** Axis adds rises
monotonically with K — +0.64, +1.93, +2.38 — because more axes mechanically
absorb more residual whether or not the model predicts better. Read alone it
recommends K=4 or higher. On the criterion that matters the fourth axis buys
nothing at all, and it arrives with a slightly WORSE rating.

**The decision rule was fixed before the numbers, and that mattered.** Taking the
argmin over 4 candidates inflates the winner by about 1.03 sd under a pure null —
with the combined column's measured sd of 0.42 mnat, roughly 0.43 mnat of
apparent gain out of nothing. Single seed per K makes each estimate carry the
full sd rather than an averaged-down one, and the sd is itself an n=2 estimate.
So the bar was: **beat K=3 by >= 2 sd (0.84 mnat) on fold 0 AND replicate on fold
1**, with replication doing the real work, since an out-of-sample confirmation
tests the choice rather than adjusting a p-value for it. Nothing came close
enough for the replication step to run.

## AXIS IDENTITY DOES NOT SURVIVE A REFIT — MEASURED 2026-08-14

The question is whether a refit's axes can be matched to the previously NAMED motifs in `axes/labels.json`, so a name carries forward. They cannot, and the reason is structural rather than a tuning failure.

**The objective is exactly invariant to rotating the axis space.** The likelihood reads the loadings and the user positions only through their product, so rotating both by the same orthogonal matrix leaves it unchanged. `V_RIDGE` is isotropic, `V_MEAN_PENALTY` acts on the mean vector, and `ORTHO_WEIGHT` targets the loading correlation being the identity — a condition every rotation preserves. The fit therefore has a continuum of exactly-equal solutions and no preference among them; which one comes out is decided entirely by where the warm start drops it, and the warm start is read off the burn-in residuals, hence off the data.

Similarity is measured three ways on the problem side, over the problems two fits share, after per-axis z-scoring: **pairwise** correlation under an optimal 1-1 assignment (`scipy.optimize.linear_sum_assignment` on `-|corr|`, so a permutation costs nothing and a sign flip means swapping a motif's two tail labels, not losing it); **canonical correlations** between the two K-dim spans, which bound what any linear alignment can achieve; and **tail-weighted agreement**, each direction z-scored then weighted linearly above `TAIL_HINGE` and exponentially decayed below, then correlated, restricted to the shared problem set so corpus growth cannot deflate it. Tail agreement is the one that speaks to naming, since a name is a claim about the extremes and the client only ever reads a flagged tail.

| comparison | canonical correlations | best axis's tails, of 1000 |
|---|---|---|
| same data, seed 0 vs 1 | 0.409 / 0.792 / 0.959 | 334 |
| same data, seed 0 vs 2 | 0.596 / 0.699 / 0.967 | 428 |
| same data, seed 1 vs 2 | 0.659 / 0.755 / 0.965 | 507 |
| production `train_mirt` vs dev trainer, both seed 0 | 0.599 / 0.869 / 0.982 | — |
| seed 0, minus the most recent 2% of attempts (~9 days) | 0.391 / 0.655 / 0.944 | 200 |
| seed 0, minus the most recent 5% (~2.5 weeks) | 0.572 / 0.700 / 0.964 | 314 |
| disjoint user halves | 0.025 / 0.490 / 0.847 | 252 |
| both seeds trained to convergence (`--flat-patience 30`) | 0.335 / 0.804 / 0.964 | 219 |
| ONE seed, patience 5 (@360) vs patience 30 (@850) | 0.863 / 0.938 / 0.990 | 612 |

Read in order, those rows rule out every post-hoc remedy:

- **A 2% data change disturbs the basis as much as a different seed does.** Same seed, 98% shared data, and the axes land as far apart as two independent fits. There is no refit cadence short enough to keep them close.
- **The disturbance does not scale with the data delta.** The 5% cut agrees BETTER with the full fit than the 2% cut does. All the cut rows sit in the same band as the seed rows, which is the signature of a re-draw rather than a drift.
- **Training to convergence does not help**, and slightly hurts. Two seeds run to their own optima agree no better than two stopped early. Nothing pulls them together, because the objective scores every rotation identically.
- **Along ONE trajectory the axes barely move** (0.863 across 490 extra epochs, against 0.409 for a seed change). They are set early by the warm start and refined locally thereafter.
- **Rotation is part of the story but not all of it.** Orthogonal Procrustes lifts the worst pairwise correlation from 0.492 to 0.754 on one seed pair but only 0.603 to 0.612 on the 2% cut, because the SUBSPACE moves as well as the basis inside it.

### Per shipped axis — which one could be named

Canonical correlations describe the SPAN. A name attaches to a COLUMN of `axes/per_problem.csv`, so the figure that governs naming is per-axis, and it is lower. Max |corr| of each of seed 0's axes against any axis of the other fit:

| | seed1 | seed2 | pat30 s0 | pat30 s1 | half0 | half1 |
|---|---|---|---|---|---|---|
| seed0 axis 0 | 0.751 | 0.829 | 0.917 | 0.753 | 0.642 | 0.654 |
| seed0 axis 1 | 0.631 | 0.681 | 0.932 | 0.631 | 0.565 | 0.433 |
| seed0 axis 2 | 0.768 | 0.883 | 0.976 | 0.766 | 0.806 | 0.646 |

allowing the best linear COMBINATION of the other fit's axes — i.e. what a gauge fix would recover:

| | seed1 | seed2 | pat30 s0 | pat30 s1 | half0 | half1 |
|---|---|---|---|---|---|---|
| seed0 axis 0 | 0.828 | 0.841 | 0.940 | 0.820 | 0.851 | 0.766 |
| seed0 axis 1 | 0.756 | 0.711 | 0.933 | 0.764 | 0.690 | 0.579 |
| seed0 axis 2 | 0.890 | 0.929 | 0.977 | 0.890 | 0.848 | 0.806 |

**Axis 2 is the one direction worth naming.** It leads on all six comparisons including the halves, and axis 1 trails on all six. The single-axis to combination gap is the free rotation; what remains after it is genuine subspace movement, which no alignment recovers.

### The legacy motifs transfer at the TAILS, not in the bulk

Measured against the 16 frozen legacy directions in `label_anchors.npz`, over shared attempted
problems. By correlation they look lost: best single-anchor correlation **0.41**, R² on the full
16-anchor span 0.16 / 0.21 / 0.28, canonical correlations against the three motifs the two-stage
fit shipped 0.009 / 0.411 / 0.582.

By tail-weighted agreement two of them plainly survive, and the control is what makes that readable.
Across the 93 CROSS-ERA pairs of differently-named tails the agreement distribution is median -0.002,
p90 +0.123, p99 +0.248, max +0.255. Against that background:

| same-name cross-era pair | tail agreement | vs different-name pairs |
|---|---|---|
| `opening+` / `opening+` | **+0.492** | above all 93 |
| `counting+` / `counting+` | **+0.448** | above all 93 |
| `ko+` / `ko+` | +0.079 | above 83% |

Opening and counting sit at roughly twice the largest different-name agreement, so they reproduce
across two entirely different model generations — support no within-generation measurement can
establish.

**THE TEST RUNS ONE WAY ONLY.** High agreement establishes that two directions name the same thing;
LOW agreement establishes nothing at all, and must not be read as a motif failing to transfer. Two
reasons, and the second is structural. A motif can be present in a fit and expressed along a
different direction, since nothing forces two model generations to place it in the same coordinates.
And the fit carries far fewer axes than the 16-motif legacy bank has motifs, so it CANNOT span them however
real they are — most legacy directions scoring near zero is a statement about there being three
slots, not about the motifs. The way to find out whether a legacy motif survives is to teach it and
measure retention, not to look for it in a fit that was never asked to hold it.

Note also that `ko`, `ko threat` and `ko threats` are not one motif found three times. Ko has many
sub-variations and a fit settles on a specific one rather than the general concept, so their mutual
agreement of ~0.43 is what distinct sub-motifs of one family look like, not degeneracy.

The control is not optional here. WITHIN the legacy fit, tails agree at rates that look identical to
the same-name matches — `shape-`/`ko threats-` +0.63, `reading+`/`killing+` +0.63,
`complexity+`/`attack+` +0.39 — because that fit carried 16 motifs with no orthogonality pressure
between them. Compared against THAT background a +0.49 match is unremarkable; compared against the
cross-era background it is off the scale. An agreement figure means nothing without the population it
is being judged against.

The disagreement between the two instruments is the point, and it generalises — see "Identity is a
tail measure" below. A name is a claim about one end of an axis, so a direction can be
substantially reproduced where it is named while sharing little of the corpus ordering elsewhere.

### Identity is a TAIL-WEIGHTED measure, and the weight's shape is what makes it trustworthy

**A name is a claim about one END of an axis**, and an axis is a trunk output rather than a linear map
on nameable inputs, so its extremes and its ordering of the middle are free to diverge. Three
instruments were tried; only the third survives.

**Bulk correlation** gets answers backwards. A fit whose bulk correlation with its bank FELL held MORE
tail than the arm it was compared against, reversing which looked better. A bipolar arm read by
correlation appears to collapse onto one pole per column (0.27 against an expected 0.74) when on tails
every pole is retained. And it overstates transfer exactly where confidence is highest: 0.945 implies
~610 shared tail problems under normality and delivers 375.

**Top-N overlap** fails three ways at once: N is arbitrary, the cutoff is hard so problem N and N+1
are total agreement and total disagreement, and membership discards magnitude, so a problem 5 sd into
the tail is the same evidence as one 0.4 sd in. It manufactures matches from problems that merely
landed inside the boundary — an axis whose only real match is ko also showed two motifs at "4%" that
vanish entirely under a weighted measure. It also cannot score a BINARY rule bank at all, because
~100k tied zeros make `argsort` pick an arbitrary tail.

**Tail-weighted correlation** is the measure: z-score each direction, weight it `z` above
`TAIL_HINGE` and `at * exp((z - at) / at)` below, correlate over the problems both cover. No N, no
chance baseline (any deterministic function of independent variables leaves them independent, so it
is exactly zero), and a tail that is constant after weighting returns NaN rather than 0 — an empty
tail makes no claim and must not read as "measured, independent".

**The two branches meet in value AND slope**, so the weight is C1. Above the threshold it is the
IDENTITY, which is what keeps one confident problem from dominating a squared error — the reason a
plain `exp()` is unusable as a teaching target or a measure. Below it the exponential preserves the
ORDER of near-tail problems while decaying it toward zero, rather than declaring them all equal.

**That smoothness is what makes the threshold safe, and a hard floor is what made it dangerous.**
With `max(z, at)` everything below the cut takes one value and contributes no variance, so the cut
decides what is measured — swept on the ortho arms, two of them SWAP PLACES between z = 0.5 and 1.0:

| hinge z | kept | no ortho | ortho on b | ortho on rating+slope+floor |
|---|---|---|---|---|
| 0.0 | 20.7% | 0.879 0.455 0.944 | 0.790 0.508 0.942 | 0.690 0.449 0.824 |
| 0.5 | 10.9% | 0.867 0.390 0.962 | 0.846 0.480 0.959 | 0.674 0.432 0.854 |
| 1.0 | 7.6% | 0.853 0.334 0.970 | 0.875 0.463 0.966 | 0.651 0.412 0.854 |
| 2.0 | 4.3% | 0.825 0.252 0.968 | 0.876 0.432 0.964 | 0.608 0.363 0.837 |

Under the smooth weight the same sweep is inert — mean agreement moves ~0.01 from z = 1.0 to 3.0 and
never reverses an ordering (baseline 0.759 / 0.757 / 0.755 / 0.753 / 0.753, ortho-b 0.762 / 0.765 /
0.765 / 0.763 / 0.761 at 1.0 / 1.5 / 2.0 / 2.5 / 3.0). The threshold stops being a thing to argue
about, which also means it must not be tuned to move a result.

A consequence worth recording: the hard floor OVERSTATED the ortho-b identity margin. It reported
0.768 against 0.719, the smooth weight 0.762 against 0.759. The direction is the same under both and
holds at every threshold, but the size of the gain was an artifact of discarding sub-threshold
information at one particular cut.

**Agreement measures CO-SELECTION and cannot say which of its two causes produced it** — one direction
under two names, or two distinct motifs whose situations co-occur. `escape atari` and `opening` agree
at +0.52 and are nevertheless distinct: conditioning on opening-ness, escape atari's spread RISES from
0.887 corpus-wide to 1.93 within the top 2% of opening, so it discriminates hard among opening
positions rather than restating them. The conditional check separates the cases; agreement alone only
nominates a pair for one.

Single-sourced in `auxiliary/rating_dev/tail_agreement.py`, used by `bank_identity.py` (paired, and
`--cross` for the axes x poles matrix a bipolar bank needs) and `tail_overlap.py`.

### The rule-seeded arms, now scoreable

Top-N could not score these at all: their banks are BINARY, and ~100k tied zeros make `argsort` pick
an arbitrary tail, so their identity was unmeasured. Hinged agreement handles them — hinging a binary
column leaves the positives at one value, so the correlation asks whether the fitted tail's weight
concentrates on the rule's positives. The negative side is constant after hinging and correctly
returns NaN rather than 0.

| arm | axis 0 | axis 1 | axis 2 |
|---|---|---|---|
| drop_opening (ko, L&D, counting) | 0.355 | 0.369 | 0.406 |
| drop_counting (ko, L&D, opening) | 0.562 | 0.635 | 0.656 |
| drop_lad (ko, counting, opening) | 0.345 | 0.440 | 0.520 |
| drop_ko (L&D, counting, opening) | 0.324 | 0.414 | 0.444 |

Every rule-seeded arm lands well below what a CONVERGED bank returns (0.875 / 0.463 / 0.966 for the
same three motifs under ortho-b).

**That gap does NOT say a converged direction is the better thing to teach from, and production teaches
from the rules.** Two findings landed after this table and both cut against reading it as identity.
The measure is calibrated against human inspection at ONE point and the numbers overstate what they
certify: six legacy poles taught back at 0.43-0.73, read as retention, and a human reading the boards
found four of six mutated or unnameable. And the round trip earns nothing for that risk — a stored
direction is itself only an approximation of whatever it was taught from, so chaining through it holds
an axis no better than teaching from the fixed reference does directly.

What the table DOES measure is how far a fit's axis lands from the rule it was seeded with, and that is
the quantity production reports every run. Read low agreement here as the model expressing the criterion
poorly, not as a motif having drifted: the criterion is still written down in `utils/rating/motif_rules.py`
and can be re-read, which is the whole asymmetry between a rule and a fitted column. Note also that these
arms were seeded from ONE-SIDED binary columns; the shipped form pairs two rules per axis and glues them
with the hinged tail weight, so it is not the same target.

### What a single axis is worth cannot be measured by ablating the others

Zeroing K-1 loadings and re-walking gives a NEGATIVE contribution for an axis that is worth
+4.41 mnat alongside its neighbours (ko alone: -0.91). One axis cannot pay for its own user-position
estimation noise on the forward walk, so solo-axis contribution is systematically negative and
useless for ranking instances of a motif. Rank them by the arm-level held-out contribution and by
leakage against the rating parameters instead.

### ORTHO_RATING holds each TAIL off rating, slope and floor

`ORTHO_RATING` 0.5 is the only term in the objective that decorrelates an axis from the rating side.
`ORTHO_WEIGHT` is K x K over the loadings alone; `V_MEAN_PENALTY` pins only the population mean of
`v`, which blocks a global difficulty shift but not per-problem tracking. Without it a free fit
spends an axis on a rating-side property: the free control's third axis correlates -0.638 with SLOPE
and only +0.182 with difficulty — the leak picks a parameter rather than difficulty specifically —
and that arm is worst on rating (0.5117) and worst on never-trained problems (0.5612) of every arm
measured.

Two properties of the term were each arrived at by getting them wrong first.

**IT TARGETS THE SHIPPED TRIPLE, NOT DIFFICULTY ALONE.** Penalising `b` by itself only DISPLACES the
leak, because each axis finds a different rating parameter to absorb: closing `b` sent one axis from
-0.04 to -0.43 on the floor head and left another at +0.62 with slope. Closing rating, slope and floor
together removes it.

An earlier reading of this document called the three-target form REFUTED, on one arm where it left
the axes worth 0.16 mnat against 4.41. That arm was TAUGHT from a bank whose own columns were 0.36
rating-correlated, so it was being pulled toward those directions and pushed off every rating channel
at once. Given a FREE fit, with nothing prescribing where to go, the same penalty is 6.4 mnat BETTER
on rating than b-only with the axes still worth 3.7 mnat; and a BIPOLAR bank at 0.274 correlation
reaches 4.21 mnat. So the failure was the CONFLICT, not the constraint, and how much conflict is
tolerable is still unresolved — see the bipolar section.

**IT IS SCORED PER TAIL, AND THE WHOLE-AXIS FORM WAS HIDING MOST OF THE LEAK.** A whole-axis
correlation is a LINEAR statistic and cannot see a symmetric dependence: an axis whose two tails are
both rating-laden in OPPOSITE directions averages to zero and satisfies the penalty completely, while
both NAMED tails — the only part the client reads — are laden. Measured, that is what was happening.

| free fit, seed 1 | worst tail correlation per axis | mean |
|---|---|---|
| whole-axis penalty | 0.273 / 0.292 / 0.307 | 0.291 |
| PER-TAIL penalty | 0.117 / 0.094 / 0.103 | **0.105** |

The whole-axis arm reported correlations of 0.02-0.09 and was recorded as rating-orthogonal on that
basis. Its tails carried 0.27-0.31 — a 3-9x understatement. The b-only arm shows the mechanism
outright: one axis carries floor -0.245 on its positive tail and +0.567 on its negative, so the
whole-axis figure is an average of two opposing tail effects, and a symmetric version averages to
exactly zero. Per-tail cuts the worst tail 2.8x. The whole-axis figure rises slightly under it
(0.016 -> 0.084 on one axis), which is expected: it is no longer the statistic being optimised and
was never the one that mattered.

**The walk-forward cost of per-tail is NOT established.** Same seed, free, only the penalty form
differing: rating 0.5073 against 0.5093, axis adds 1.53 against 3.73, never-trained 0.5511 against
0.5576, rho 0.8067 against 0.8154. Every one of those is inside the free-fit seed spread measured
elsewhere in this document (5.3 mnat on rating, 4.4x on adds), so one pair cannot separate them. It
is adopted on the leakage result, which is a direct property of the fit rather than a held-out
estimate. On the stop split it ran 1.6 mnat behind its control and took the longest of any arm to
converge (4280 epochs against 3950), consistent with a tighter constraint being slower to fit.

Implementation note that cost a crash: the torch tail weight must CLAMP its exponent. `torch.where`
evaluates both branches, so a large positive z overflows the exponential branch to inf even though
the result is discarded, and the backward pass turns it into NaN through the zero gradient — which
surfaces as `binary_cross_entropy`'s `[0,1]` device-side assert a step later, pointing at a line
nowhere near the fault.

### Superseded: the b-only comparison

Kept because the arms below were all run under it and their other columns are still valid.

| arm | rating | rating+axis | axis adds | never-trained | survey rho | mean tail agreement |
|---|---|---|---|---|---|---|
| taught, no ortho | 0.5123 | 0.5064 | 5.84 | 0.5725 | 0.8131 | 0.759 |
| taught, ortho on b | 0.5130 | 0.5086 | 4.41 | 0.5631 | **0.8232** | **0.762** |
| taught, ortho on rating+slope+floor | 0.5148 | 0.5146 | 0.16 | 0.5639 | 0.8123 | 0.656 |

The ortho-b identity margin over untreated is NOT ESTABLISHED: a paired bootstrap over problems puts
it at +0.003, 95% CI [-0.001, +0.007]. It survives every threshold from z = 1 to 3, but that is
stability of the METRIC and not of the estimate — the sweep rescores the same single pair of fits.
Nor is the bootstrap the right error bar: it resamples PROBLEMS, while the dominant variability for
comparing two fits is ACROSS FITS, and each arm is n = 1. What stands is that ortho-b does not COST
identity.

Note also that constraining rating, slope and floor does NOT constrain `b`: measured -0.312, because
zero correlation with each of three quantities separately implies nothing about a nonlinear
combination of them.

### A FREE fit does not reproduce itself — the seed floor, measured

Two runs of the identical production configuration (free, no bank, full rating-orthogonality),
differing only in seed:

| | rating | rating+axis | axis adds | never-trained | survey rho |
|---|---|---|---|---|---|
| seed 1 | 0.5093 | 0.5056 | **3.73** | 0.5576 | 0.8154 |
| seed 2 | 0.5146 | 0.5137 | **0.84** | 0.5578 | 0.8185 |

Direction agreement between them, best match per seed-1 axis: **0.56 / 0.25 / 0.20, mean 0.34.**

**This is the EXPECTED baseline, not a defect.** The objective is exactly invariant to rotating the
axis space, so nothing in a free fit prefers one basis over another and two seeds have no reason to
agree; 0.34 is that known arbitrariness put on a scale, and it is the arbitrariness the axis bank
exists to remove. The full penalty widens the space further — forbidding correlation with rating,
slope and floor leaves a large residual subspace to be arbitrary within.

Two things follow, and the second is a caution about this whole document.

**Directions named off a single free fit are seed-specific.** The store's `ax_4e23f75b` (defend
territory), `ax_2091d5bf` (capture stone to gain liberty) and `ax_4cb94a2c` (escape atari) come from
the seed-1 fit and are matched by seed 2 at 0.56 / 0.25 / 0.20. The boards a human read to name them
are real; the DIRECTION is not stable, so those entries are candidate seeds rather than settled
motifs. It also dissolves a puzzle: the free `defend territory` disagreeing with the b-only one at
-0.018 was not two different concepts, it was a free fit landing arbitrarily.

**Two free fits are not the same solution, so their spread bounds only FREE-arm comparisons.**
Rating moves 5.3 mnat and axis contribution 4.4x between two runs of one configuration — but the two
fits settled on different axis bases, so this is variation between distinct solutions rather than
measurement noise around one. It applies where both arms are free: the full penalty beating b-only on
rating by 6.4 mnat is a free-against-free comparison and is inside this spread, so it is unresolved.

IT DOES NOT APPLY TO TAUGHT ARMS, which are pinned by a bank and are not free to land elsewhere —
the decay sweep, the ortho comparison and the bipolar arms are all taught, and none of them is
called into question by this. The seed floor for a TAUGHT fit is a different quantity, is not
measured, and is the one that would calibrate those comparisons.

### The shipped rule set

Six rules, `utils/rating/motif_rules.py`, labelled over the whole corpus by
`pipeline.build_motif_proxies` (160489 problems, 9x9 and 13x13 together):

| rule | problems | share | criterion in one line |
|---|---|---|---|
| life and death | 5962 | 3.71% | one directional part of the ownership swing dominates, no trade beside it |
| counting | 11456 | 7.14% | top two first moves within a few points on score while winrate separates hard |
| ko | 25074 | 15.62% | a ko shape stands at the root, or within KO_DEPTH plies of a resolved line |
| opening | 9014 | 5.62% | setup prefix at or under an absolute per-board length |
| territory | 51288 | 31.96% | score swing beats a base plus a per-flipped-stone allowance |
| shape | 11194 | 6.97% | best two RESOLVED first moves within a small winrate gap — an uncertain outcome |

Labels per problem: 67977 carry none, 72087 one, 19375 two, 1049 three, 1 four.

Two properties are load-bearing rather than incidental. **Every threshold is on the problem's OWN
numbers**, which is what makes the incremental labelling sound — a stored row can never be made wrong
by a problem added later. `opening` is the worked example of getting that wrong: as a corpus
PERCENTILE it moved as the corpus grew, silently re-labelling problems that had not changed, and it
is an absolute prefix length per board for exactly that reason. **And every rule is evaluated both
ways on every problem**, which is what hand labelling cannot supply cheaply and what lets a column be
fitted at all — without negatives there is nothing to separate from.

Shipped pairing, `components` in `axes/labels.json`, chosen on tail directionality alone (worst
cancellation -0.081): `life and death/territory`, `counting/shape`, `ko/opening`. Note axis 0 is
LOPSIDED in the taught column — life and death reaches +4.33 against territory's -0.87, roughly 3x
the variance, because territory is 32% of the corpus against 5-7% for the others. Balance was
deliberately not a selection criterion, having proved unreliable, but it is the thing to look at if
the territory pole comes back weak.

### The bank-building workflow: discover, then glue (DEV — production teaches from rules)

Two stages, deliberately separate, because a fit asked to both find directions and satisfy a
prescribed set does neither cleanly.

**Stage one, discover a direction.** Either HANDCRAFTED — a stated rule labels the corpus
(`auxiliary.labeler.rules`, `export_bank --from-rules`), which is how ko, counting, opening and life
and death were seeded — or UNINITIALIZED, a fit with no `--teach-directions` let go to land where it
will, which is how escape atari, defend territory and capture stone to gain liberty were found. A
rule bootstraps a motif nobody has seen; a free fit finds one nobody thought to write a rule for.
Neither stage-one fit needs to be shippable. A human reads the extremes and names what is there;
`update_motif_store` records the direction with its provenance, and the store is the durable
artifact — a direction found once is never rediscovered.

**Stage two, glue stored directions onto K axes.** `build_bank_from_dumps` composes a bank from store
entries, one motif per axis or two as a bipolar column. Nothing is re-derived; this stage decides only
what goes together and in what order. Production's analogue is `components` in `axes/labels.json`,
which does the same over RULE names and is what `pipeline.train_mirt` reads; `build_bank_from_dumps`
is for exploring a composition before it is written as rules.

The measured payoff of pairing carries over to the shipped form, since it is a property of the glue
rather than of what is glued: both use the same hinged tail weight, `f(A) - f(B)`.

What the second stage costs and buys, measured: pairing six motifs onto three axes retained every
pole, improved the two WEAKEST most (counting 0.26 -> 0.67, escape atari 0.17 -> 0.74 against the
one-sided arm), and gave the best never-trained pricing of any fit run. The cost is that a pole's
retention tracks its own reproducibility, so a weak motif stays weak — pairing does not rescue a
direction that was never well determined.

### A bipolar bank under the full penalty — the best never-trained pricing measured

Six motifs on three axes, columns built as `f(z(A)) - f(z(B))` with the shared tail weight, taught
under the full rating-orthogonality penalty. Its bank is NOT rating-orthogonal — the columns carry
0.246 / 0.264 / 0.274 against the shipped triple, mostly on slope — so this is the penalty working
against its own teaching target.

| arm | rating | rating+axis | axis adds | never-trained | survey rho |
|---|---|---|---|---|---|
| bipolar, ortho on b, raw subtraction | 0.5074 | 0.5041 | 3.29 | 0.5552 | **0.8294** |
| bipolar, full ortho, weighted columns | 0.5072 | 0.5030 | **4.21** | **0.5533** | 0.8223 |
| free, full ortho, no bank | 0.5093 | 0.5056 | 3.73 | 0.5576 | 0.8154 |

Per pole, tail-weighted agreement:

| pole | ortho on b | full ortho |
|---|---|---|
| counting | 0.256 | **0.670** |
| life and death | 0.440 | 0.448 |
| defend territory | 0.553 | 0.577 |
| opening | **0.858** | 0.603 |
| escape atari | 0.172 | **0.738** |
| ko | **0.973** | 0.931 |
| mean | 0.542 | **0.661** |

Leakage falls from a worst cell of +0.79 (slope) and +0.49 (floor) to +0.065 and -0.112.

Two readings to keep apart. The retention gain is concentrated in the two WEAKEST poles — counting
0.26 -> 0.67 and escape atari 0.17 -> 0.74 — while the two strongest give up a little; that is the
better shape, since a bank is worth what its weakest named tail is worth. But the arm changes TWO
things against its predecessor, the penalty and the column construction, so the gain is not
attributable. The pattern points at the construction: the weighted form specifically removes
"not-A pushes a problem toward B", and it is exactly the weak poles that would suffer most from that.

**It also unsettles the conflict story.** A one-sided bank at 0.357 correlation under this penalty
left the axes worth 0.16 mnat; a bipolar bank at 0.274 reaches 4.21. Either bipolar pairing gives the
fit enough freedom to satisfy both objectives, or the construction is doing the work, or 0.357 is past
a cliff 0.274 sits under. Until that is separated, "the bank must be rating-orthogonal" is not
supported as a prerequisite.

### CALIBRATION — what tail agreement means, checked against a human once

Every agreement figure in this document is ordinal. The one place the scale has been checked against
a person reading boards is the legacy refit, and it does not sit where the numbers suggest.

| pole | taught-from agreement | what inspection found |
|---|---|---|
| `killing` | +0.43 | became SEMEAI — related, distinct, renamed |
| `judgement` | +0.64 | drifted too far to carry the name — now UNNAMED |
| `attack` | +0.66 | unidentifiable — now UNNAMED |
| `ko threat` | +0.66 | narrowed to FIND KO THREAT AND ANSWER FORCING MOVE |
| `defence` | +0.73 | became ESCAPE ATARI AND KO — renamed |
| `endgame` | +0.73 | holds |

For contrast `ko` reproduces at 0.91-0.97 between fits and keeps its name unchanged.

**Four of six poles at 0.43-0.73 mutated, and those are among the highest retention figures measured
anywhere in this document.** Retention is not identity. Anything under about 0.75 has to be
re-inspected and probably re-named before it ships; a figure in that range means a real direction was
held, not that the name still describes it.

This is the single calibration point the measure has. It bounds what every comparison here can claim:
arms can be ranked by agreement, but no threshold in this document except ~0.9+ has been shown to
correspond to a motif a person would still call by the same name.

### Choosing a bipolar pairing: cancellation and dominance

Which two motifs share an axis is decided on the GLUED column `c = f(z(A)) - f(z(B))`, not on the raw
directions — the reshape decides both failure modes, so a criterion computed before `f` cannot see
either. `auxiliary/rating_dev/pair_motifs.py`.

CANCELLATION. Where both tails are active the difference subtracts and those problems leave the
column. `Var(f_A - f_B) = V_A + V_B - 2Cov`, so the mean squared difference between the reshaped
poles IS the surviving signal; reported as `2Cov/(V_A+V_B)`, zero for independent poles and 1 for
identical ones.

DOMINANCE. A column can have ample variance and still carry one motif, if one pole's reshaped spread
swamps the other's. Reported as `min(V)/max(V)`.

**Cancellation is almost never what binds.** Over the six b-only motifs it is ~0 or negative for 13
of 15 pairs — these motifs do not co-occupy tails. The single exception is `opening`/`escape atari` at
+0.502, which destroys half the summed variance and is the co-occurrence pair; it must never share an
axis. Balance instead ranges 0.47 to 0.98.

**Balance predicts retention measured before the criterion existed.** In the first bipolar arm the
pair at balance 0.595 returned poles at 0.172 and 0.973 (ratio 0.18), while pairs at 0.816 and 0.850
returned 0.58 and 0.64. The pairing shipped in that arm was chosen on most-negative BULK correlation
and ranks 3rd of 15 under this criterion.

Rank candidate pairings by the WORST pair, not the mean: an axis is only as good as its weakest named
tail.

### Averaging several instances of one motif — when it works

The store holds multiple fits' takes on the same motif. Averaging them denoises toward the shared
direction, but only when they ARE one direction, and the diagnostic is whether the consensus agrees
with every member better than the members agree with each other.

| concept | n | pairwise agreement | consensus vs each member | verdict |
|---|---|---|---|---|
| counting | 3 | 0.45 / 0.50 / 0.54 | 0.74 – 0.84 | average |
| opening | 3 | 0.38 / 0.44 / 0.49 | 0.70 – 0.86 | average |
| life and death | 2 | 0.45 | 0.82 – 0.86 | average |
| ko (modern only) | 2 | 0.93 | 0.98 | same direction; average or take either |
| ko + legacy ko | 3 | 0.06 – 0.93 | 0.12 – 0.95 | do NOT — legacy ko is unrelated to the two modern ones |
| defend territory | 3 | -0.10 – 0.58 | 0.48 – 0.83 | do NOT — the instances disagree |

Read by pairwise agreement: **>=0.8** same direction, average or pick either; **0.3-0.6 with the
consensus clearly higher** noisy views of one direction, average — this is the real win; **<0.15**
not the same direction, pick one, averaging is meaningless.

Averaging the TAIL WEIGHTS (a union, since they are non-negative and nothing cancels) and averaging
the raw z-scores (a consensus) came out nearly identical everywhere — 0.74-0.84 against 0.76-0.84 on
counting — so the distinction does not matter empirically. Average the raw z-scores.

Two cautions. `ko`, `ko threat` and `ko threats` are distinct SUB-MOTIFS of one family, not one motif
found three times, so averaging across them blurs rather than denoises. And a pairwise figure of
+0.49 against one instance and +0.15 against another does NOT mean a direction sits between them:
regressing the new `escape atari` on both gives R2 0.254 of which the second contributes 0.018, the
two are independent of each other, and 75% of it is explained by nothing in the store.

### A fifth rule: territory

`(score_correct - score_alt) - 2 * stone_swing_abs > 20` points. A stone flipping ownership is worth
TWO points — the prisoner and the point it stood on — so `2 * stones` is the swing that capture
explains, and the remainder came from territory. The swing is ABSOLUTE so a trade does not cancel:
stones dying on both sides is a trade whatever the net comes to, and a trade is not a territory
problem. Empty-point swings are NOT charged, being the territory itself.

10652 positives, 6.6% of the corpus, splitting 7593 / 3059 across boards — almost exactly the corpus
proportion. A ratio form (share of the swing not explained by stones) was rejected: at a matched
count it skews 85% to 9x9.

Overlap with the existing rules, as a share of territory's own positives: life and death 8.2%,
counting 0.3%, ko 25.0%, opening 0.4%. Essentially disjoint from counting — the other score-based
rule, and the one it could most easily have duplicated — and the 25% with ko is barely above ko's
19.3% base rate.

### A ninth rule: semeai — the balanced pair `life and death` cannot express

`rule_life_and_death` judges every other ownership part against a fraction of the LARGEST one, so the second FOCUS part — the kill, when the save is bigger, or the save when the kill is — is scored by the same bar as a sacrifice. A capturing race therefore fails it: both groups' fates are the one question, and the rule reads the second group as the trade that disqualifies the first. Measured over the 215,341-row ownership sweep, that case is **11.04% of the corpus against the 4.01% the rule itself takes** — the balanced pair is rejected nearly three times as often as the rule fires. A genuine trade (a sacrifice or a let-out above the bar) accounts for a further 6.94%, which is the exclusion the rule is actually for.

`rule_semeai` takes exactly that rejected region: `save + kill >= SEMEAI_MIN_STONES`, the split `|save/(save+kill) - 0.5| <= SEMEAI_BALANCE`, and the two non-focus parts under `LD_TRADE_RATIO` of the PAIR rather than of one part.

**The bar is on the split, not on either part, because a race does not need two groups of equal size.** What a semeai is about is both groups being at stake; their stone counts are whatever the position gives. At 0.25 a 1:3 mismatch still fires, and a tighter bar would turn the rule into "two groups the same size", which names nothing in Go.

**The two rules are disjoint by construction, and `LD_TRADE_RATIO` is what puts the boundary there.** Judging every other part against `ratio` of the largest holds the smaller focus part at or under `ratio/(1+ratio)` = 1/6 of the pair, i.e. a split at least 1/3 from even. Any balance bar below 1/3 is therefore mutually exclusive with `life and death`; 0.25 keeps a margin so the region between 1/6 and 1/4 labels as NEITHER, rather than two rules meeting at a shared edge where a hair of drift decides which motif a problem seeds. Measured on the corpus: 0 problems carry both labels. **Moving `LD_TRADE_RATIO` moves that boundary**, so the two constants are one decision.

**`SEMEAI_MIN_STONES` is the pair analogue of `LD_MIN_STONES`.** One decisive group is worth labelling at 5 stones; a race is decided by two, so the bar goes on their sum at 10. It also floors the smaller side at `(0.5 - balance) * min_stones` = 2.5 stones, which is what stops a throw-in beside a large group reading as a race — a floor a bar on the sum alone does not provide. The sweep, with the trade guard on:

```
  min_stones   share    9x9     13x13   smaller side p5 / median (stones)
       6      14.93%   11.54%   18.69%      2.2 /  4.3
       8      11.31%    8.51%   14.42%      2.8 /  5.2
      10       8.67%    6.42%   11.17%      3.4 /  6.1
      12       6.67%    4.83%    8.71%      4.0 /  7.0
```

10 lands the rule between `counting` (6.3%) and `territory` (8.6%), and narrow is the safe side by the width evidence above: the fit spreads a narrow rule outward from its core, while a ~20% rule is taken as written and must already BE the concept. Dropping the trade guard raises the share to 10.81% by admitting races won at the cost of a third group.

Overlap at the shipped setting, as a share of semeai's own 18,676 positives: life and death 0.0%, counting 0.1%, territory 0.0%, shape 0.9%, opening 5.3%, sente 11.9%, tenuki 24.8%, ko 28.8%. The only substantial one is `ko`, at 1.7x its 16.6% base rate and covering 15.0% of ko — a race that comes down to a ko is a real overlap of concepts rather than a duplicated column.

**It is a CANDIDATE, not a shipped axis.** What decides that is split-half reliability (`auxiliary/rating_dev/split_half.py`), which needs a fit with the rule taught; `components` in `axes/labels.json` and `K_AXES` are untouched until it clears the working tier. The exploratory store holds a `semeai` direction found by a free fit (`ax_cc62e376`, the negative pole taught from legacy `killing` at 0.43 and renamed on inspection), which says the model does land on this concept unprompted — a reason to test the rule, not evidence that it ships.

### TEACH_DECAY_EPOCHS is an interior optimum at 1000

Swept on converged-axis banks under ortho-b, one seed each, scored on tail-weighted agreement.

| decay | escape atari | life and death | ko | mean | rating | axis adds |
|---|---|---|---|---|---|---|
| 250 | 0.744 | 0.160 | 0.889 | 0.598 | 0.5112 | 3.19 |
| 500 | 0.774 | 0.275 | 0.941 | 0.663 | 0.5079 | 4.09 |
| **1000** | **0.818** | 0.515 | **0.952** | **0.762** | 0.5130 | **4.41** |
| 2000 | 0.677 | **0.558** | 0.947 | 0.727 | 0.5163 | -1.58 |

1000 leads on mean agreement and on two of three axes, and carries the highest held-out axis
contribution, which falls away on both sides. 2000 edges it on life and death alone. At 2000 the axes are worth NEGATIVE mnat: past some point the teaching
stops stabilising a direction and starts pinning it to the source fit's idiosyncrasy.

A sweep point must stop well AFTER its own decay ends or it measures nothing. At decay 3000 the
patience rule fires at epoch 1740 with the teaching weight still at 0.52; the axes are never
released, their identity is trivially near-perfect (0.933 / 0.938 / 0.984, mean 0.952), and the
survey rho is the worst on the board (0.7728). That arm is not a measurement of decay 3000.

Read on bulk correlation instead, 500 and 1000 look near-equivalent on two of three axes and the
choice appears to rest on life and death alone. On tail agreement 1000 leads 500 on all three, and the
trade is explicit: 500 buys 5.1 mnat of rating, 1000 buys materially more tail identity everywhere.

### THE DECAY MUST NOT REACH ZERO — TEACH_FLOOR_EPOCH 750, MEASURED 2026-08-19

Released completely, a rule-taught fit does not keep the direction it was taught. On a full
production run the axes ended at 0.091 / 0.111 / 0.056 against the columns they were taught from,
which is indistinguishable from independent — the teaching had bought nothing that survived to the
artifact. `TEACH_FLOOR_EPOCH` stops the decay CLOCK rather than clamping the weight, so past it the
pull is simply constant at `TEACH_WEIGHT*(1 - TEACH_FLOOR_EPOCH/TEACH_DECAY_EPOCHS)`.

Four arms, seed 0, 15% held-out users (hash `55885a30ea72c090`), everything else at production.
Retention is the hinged tail agreement per pole against the taught column, on attempted+banked
problems; held-out is the full held-out-user row set with the axes on.

| floor | residual | life&death/territory | counting/shape | ko/opening | mean | held-out | axis mnat | stop | rating range |
|---|---|---|---|---|---|---|---|---|---|
| none (1000) | 0.00 | 0.390 / 0.235 | 0.610 / 0.410 | 0.405 / 0.511 | 0.427 | 0.7930 | 0.044 | 2770 | [-43.2, 11.5] |
| 950 | 0.05 | 0.779 / 0.650 | 0.763 / 0.530 | 0.834 / 0.837 | 0.732 | 0.9175 | 0.306 | 4750 | [-103.0, 14.8] |
| **750** | **0.25** | **0.828 / 0.694** | **0.783 / 0.544** | **0.860 / 0.856** | **0.761** | **0.7147** | **0.171** | 3240 | **[-6.5, 9.4]** |

**750 is better than no floor on identity AND on both cost measures at once, which is the part worth
stating plainly.** Retention 0.761 against 0.427; held-out logloss 0.7147 against 0.7930; axis
contribution 0.171 mnat against 0.044. A residual pull is not a trade of accuracy for identity here —
the released arm is worse at everything, because an axis that has wandered off its taught direction
is not thereby a better-fitting axis, it is one that spent the run's freedom on nothing in
particular. The rating range says the same: [-6.5, 9.4] against [-43.2, 11.5], and across all four
arms the range orders EXACTLY as held-out logloss does.

**A human read the boards and the motifs were still there.** That is the check that matters and the
only one that can settle it — the calibration note below records six poles at 0.43-0.73 reading as
retention while four of six had mutated, so a number in this range does not by itself preserve a
name. Two of 750's six poles sit under the 0.75 bar (0.694 and 0.544) and the inspection passed
anyway; read that as one data point extending the calibration under a floor, not as lowering the bar.
The 950 page was inspected too and drifted visibly, consistent with its lower figures.

**Held-out is NOT monotone in the residual, so the axis needs filling in rather than extrapolating.**
0.05 is worse than both 0.00 and 0.25 (0.9175 against 0.7930 and 0.7147), and its rating range is the
widest measured. Do not read the sweep as "more pull is better" in either direction.

#### The multiplier is not the lever — MEASURED 2026-08-19

Retention does not exceed ~0.77 against the rule columns even while the pull is at full weight, which
invites tripling `TEACH_WEIGHT`. Measured, it does nothing. An arm at weight 3.0 with the floor at 917
— so the RESIDUAL is 0.249, matching the 750 arm's 0.250, with 3x the pull during teaching — lands on
the same place:

| arm | weight | residual | mean retention | held-out | axis mnat | stop |
|---|---|---|---|---|---|---|
| floor 750 | 1.0 | 0.250 | 0.761 | 0.7147 | 0.171 | 3240 |
| floor 917 | 3.0 | 0.249 | 0.760 | 0.7384 | 0.109 | 3240 |

Per-pole they are within 0.033 everywhere, and both stopped at epoch 3240 from a best at 2940. The
residual sets where the fit ends up; the peak weight does not reach it. The stronger pull is also
slightly WORSE on both cost measures, so it is not a free knob either.

The retention TRAJECTORY (probed every `PROBE_EVERY` epochs on the live parameters) is what shows why:

```
ep   50  w 2.85  0.497     ep  600  w 1.20  0.766  <- peak
ep  300  w 2.10  0.738     ep  700  w 0.90  0.764
ep  500  w 1.50  0.760     ep  950  w 0.25  0.725  <- residual holds
                           ep 3050  w 0.25  0.760  <- re-tightens
```

Retention is FLAT at 0.74-0.77 while the pull falls from 2.10 to 0.90 — a 2.3x range in weight with
no response. That is saturation, not a weak pull. It also shows the floor's drift is 0.006 (peak
0.766, final 0.760), against 0.427 for the released arm: with a floor the post-release drift problem
is simply gone, and what is left is a ceiling the pull never touches.

#### The ceiling is the bipolar pairing, not the model

Three candidates, two eliminated:

- **Not feature capacity.** The best linear map on the trunk's own input features reaches 0.402/0.557,
  0.590/0.464, 0.785/0.734 out-of-sample (`dev_runs/rule_predictability.py`). The fitted axes already
  beat that everywhere, so the MLP has headroom left.
- **Not the ortho penalty.** Every per-tail correlation between a rule column and rating, slope or
  floor is within ±0.16 (`dev_runs/rule_vs_rating.py`). There is no conflict for the penalty to
  create — unlike the rating-correlated banks recorded above.
- **The pairing.** The six rule poles are mutually near-independent, every pairwise hinged agreement
  between -0.104 and +0.084. An axis asked to be rule A at its positive tail and rule B at its
  negative can only do both to the extent A and B are opposite ends of one dimension, and the
  strongest pairing available is -0.10. They are not opposites, they are unrelated — and they can
  co-occur (12,982 attempted problems carry two rule labels, 716 carry three), so this is measured
  rather than a definitional artifact of the labelling. On top of it **41% of attempted problems carry
  no rule label at all**, so the target is silent about them and the likelihood is free to put them in
  the fitted axis's tails.

Retention tracks pairing quality across the three axes, in order:

| axis | pair | pair agreement | retention mean |
|---|---|---|---|
| ko / opening | most anti-correlated | -0.101 | 0.858 |
| life and death / territory | | -0.089 | 0.761 |
| counting / shape | least | -0.075 | 0.664 |

Three points, but monotone, and the worst pole in the whole set (`shape`, 0.544) sits on the
worst-paired axis. This is the reason a CONVERGED direction retains better than a rule column: a
stored direction is continuous, covers every problem, and its two ends are opposite by construction.
That is a statement about the TARGET's shape, and it does not license shipping one — see the
calibration note and the demotion argument in CLAUDE.md.

### Reproduction

```bash
# each arm is one training run; --dump-params writes tid/size/loading/theta/v
./.venv/bin/python3 -m auxiliary.rating_dev.train_joint_mirt \
    --seeds 0 --holdout-problem-frac 0 --dump-params seed0.npz
#   --drop-recent-frac 0.02    an earlier refit of itself (date cut)
#   --axis-holdout-fold 0|1    disjoint user halves
#   --flat-patience 30         trained to convergence
#   --init-from M.pt           continue a previous fit instead of re-deriving a basis
```

`--holdout-problem-frac 0` matters: the dev default is the benchmark's 20% problem holdout, and a problem no fit saw is priced from the feature prior alone, which inflates the apparent instability with variation production does not have.

## CARRYING THE PREVIOUS FIT OVER IS THE REMEDY — MEASURED 2026-08-14

Given that no post-hoc matcher can work (previous section), the remaining option is to stop re-deriving a basis: initialize each refit from the previous fit's model (`--init-from`), so consecutive runs CONTINUE the same axes. The problem-side modules — trunk, rating head, axis map — transfer whole; `theta` and the axis positions scatter by profile/user NAME, so accounts that did not exist yet start at zero, which is what they would have been initialized to anyway. Carrying implies no burn-in: the carried axes ARE the warm start.

Design. A geometric schedule of date cuts, mirroring real growth — early refits add a lot of relative data, later ones little. CHAIN A carries: 50% cut (cold, the origin) -> 25% -> 12.5% -> 6.25% -> full, each continuing the last. CHAIN B is the control, the same cuts fitted INDEPENDENTLY, which is what separates "the data agrees" from "it never left where it was put". Both at the experiment surface's `--flat-patience 5`.

| step (new accounts) | CARRIED, min canonical | COLD, min canonical |
|---|---|---|
| 50% -> 25% (26% new) | 0.843 | 0.088 |
| 25% -> 12.5% (7.6% new) | 0.906 | 0.070 |
| 12.5% -> 6.25% (3.9% new) | 0.983 | 0.732 |
| 6.25% -> full (~3% new) | 0.958 | 0.552 |
| **end to end, origin vs final** | **0.676** | **0.134** |

Fitted independently, the two endpoints share **14 tail problems of 1000** on the best-matched axis against 3.3 by chance — unrelated fits. Carried through four refits, the same span holds **568**, and the axes stay recognizably the same directions.

**Drift does NOT compound with refit count.** Cumulative agreement against the origin runs 0.843 / 0.609 / 0.666 / 0.676 after one, two, three and four steps: it drops once, at the single large perturbation, then stops moving. What the cumulative figure reflects is the SIZE of the largest delta, not how many refits have happened. Steps 3 and 4 are the realistic production case (3.9% and ~3% new accounts) and come in at 0.983 and 0.958, leaving the cumulative flat — at a normal cadence, carry-over effectively freezes the axes.

### It costs nothing — it gains

Best stop-split at the SAME cut, so the two are directly comparable:

| cut | carried | cold | delta |
|---|---|---|---|
| 0.25 | 0.4668 @110 | 0.4731 @330 | **-6.3 mnat** |
| 0.125 | 0.4714 @60 | 0.4760 @450 | **-4.6** |
| 0.0625 | 0.4835 @20 | 0.4930 @350 | **-9.5** |
| 0 (full) | 0.4772 @50 | 0.4834 @360 | **-6.2** |

The entrenchment risk — that a carried fit starts near its optimum, stops immediately, and buys stability by never moving — did NOT materialize. It moves to a BETTER place, faster. In hindsight that follows: a refit inheriting a converged model is not handicapped, it is starting from the answer to a strictly easier version of the same problem, which is exactly what a production refit is.

**The margin is not a patience artifact — CONFIRMED by a 2x2 at cut 0.25.** Both arms were re-run at `--flat-patience 30` and both find the IDENTICAL optimum they found at 5 (cold 0.4731@330 either way, carried 0.4668@110 either way), so neither was being truncated; the stability figures are likewise unchanged, each patience-30 fit having restored the same epoch as its twin.

Worth carrying forward from that: patience 5 stopped the FULL-corpus fits short (optima at 850 and 580) but converges genuinely by epoch 330 at 75% of the data. Premature stopping is not universal — it appears where there is more data and a longer improvement tail — so the ~0.6 mnat that raising `FLAT_PATIENCE` buys should not be expected at every data size.

### What this leaves for the tail bank

Carry-over removes the need to RECOGNIZE a motif, but not the need to notice when one has drifted out from under its name — consecutive pairs sit at 0.96-0.98 while the origin has moved to 0.68, so a consecutive check would never fire. The job the tail bank is actually suited to is a DRIFT MONITOR against the frozen origin: per refit, report how many of a motif's original top/bottom 500 problems it still holds, and flag the name for re-inspection below a bar. That is a monitor on a known quantity, not a matcher against strangers, and it is the one form of the idea the measurements support.

## THE WARM START BUYS SPEED AND A SUBSPACE, NOT ACCURACY — MEASURED 2026-08-14

Two seeds per arm, production protocol (full corpus, no problem holdout), every arm at `--flat-patience 30` so the slow arm is not scored while still descending. Baseline is `pat30_seed{0,1}`.

| | loss | epochs to best |
|---|---|---|
| `burnin` (shipped) | 0.4829 / 0.4838 | 850 / 580 |
| `none` (random init) | 0.4829 / 0.4847 | 3200 / 2670 |

**Loss-equivalent within seed noise.** The arms average 0.4834 and 0.4838 while the BASELINE'S OWN two seeds differ by 0.9 mnat, so the gap is smaller than the noise it would have to clear. What the warm start buys on the rating side is a factor of ~4 in epochs.

This is not the +2.1 mnat in the sweep table above, and the protocols differ — that arm was fold 0 at a single seed against a fold-0 baseline — so this does not overturn it directly. But that arm ran **1360 epochs** where these need 2670-3200 to stop improving, which is the more likely explanation than a real quality gap: a cold arm scored at 1360 epochs is being read mid-descent.

### Same loss, different subspaces

| comparison | canonical correlations |
|---|---|
| cold vs warm, seed 0 | 0.029 / 0.302 / 0.434 |
| cold vs warm, seed 1 | 0.089 / 0.532 / 0.570 |
| the two COLD seeds against each other | 0.142 / 0.328 / 0.592 |
| the two WARM seeds against each other | 0.335 / 0.804 / 0.964 |

Seed 0's two fits score IDENTICALLY (0.4829 both) and share almost nothing (min canonical 0.029, 26 tail problems of 1000 against 3.3 by chance). **This is stronger than the rotation degeneracy.** Rotation invariance says the BASIS within a subspace is free; this says the SUBSPACE is underdetermined — several different 3-dimensional slices of the residual space predict equally well. It matches the eigenvalue spectrum (0.00098 / 0.00048 / 0.00034, trailing off): past the leading direction the residual is broad and weakly differentiated, so any three directions drawn from it carry about the same signal.

The warm start does improve reproducibility — warm-started seed pairs reach 0.335 against cold-started 0.142 — but it is buying consistency in a range where the axes are still not reliably identifiable.

**What this means for naming.** The motifs are not discovered facts about Go. They are an arbitrary basis of a weakly-structured space, selected by initialization rather than by the data, and a fit that scores identically finds different ones. Naming them stays defensible as a product device — an unnamed axis ships, steers selection and carries its share of the weakness posterior — but a name should not be read as having identified a real skill dimension. It is a label on a coordinate, and the coordinate is a choice.

## TEACHING THE AXIS MAP A BANK HOLDS IT — MEASURED 2026-08-14

Carry-over (previous section) fixes identity by continuing the whole model. This asks the narrower question: can a refit be held on a FROZEN set of directions, so names attach to an absolute reference that never drifts rather than to a chain that does? `--teach-directions` adds an auxiliary MSE pulling the axis loadings toward a banked set, with the trunk learning normally, and `--teach-decay-epochs` releases it linearly.

Two design points, both learned by getting them wrong first.

**TEACH ON THE ATTEMPT STREAM, reusing the `gh` each batch already produces.** The first version sampled problems UNIFORMLY from the corpus every step — a second stream, disjoint from the attempt stream — and cut it at the burn-in handover. That fit ended **0.584 min canonical from its OWN bank**. The cause is not the loss term vanishing but the DISTRIBUTION changing: the trunk spent 300 epochs adapting to a mixture, half of it disappeared, and ~1850 epochs of attempt-only training overwrote whatever served the vanished half — taking the taught directions with it. Riding `gh` means no second distribution exists, so nothing shifts when the weight decays; only a scalar moves. It also costs nothing (the loadings are already computed) and matches the population the axes are DEFINED on, since loadings are identified by between-user residuals and those exist only where somebody attempted the problem. A third of the corpus never is: its banked values are the previous fit's extrapolation into a region that never carried signal, so teaching them would pin down something arbitrary and spend trunk capacity doing it.

**A SMOOTH release beats an abrupt one, and beats never releasing.** Seed 1 taught toward seed 0's bank, patience 30 throughout:

| arm | identity (min canonical) | tails, of 1000 | loss |
|---|---|---|---|
| untaught (this baseline pair) | 0.335 | 116 / 275 / 110 | 0.4838 @580 |
| decay 600 | 0.859 | 471 / 455 / 375 | 0.4834 @810 |
| decay 1000 | 0.876 | 477 / 518 / 435 | 0.4839 @1150 |
| decay 1500 | **0.917** | 523 / 632 / 500 | 0.4845 @1150 |
| constant, never released | 0.915 | 515 / 623 / 502 | 0.4856 @510 |

**It is a FRONTIER, not an optimum.** Identity rises monotonically to 1500 and then flattens; cost rises monotonically throughout. Nothing crosses, so there is no interior best — you pick a point. The frontier is cheap: 600 -> 1500 buys +0.058 identity for +1.1 mnat against 0.9 mnat of baseline seed noise, and past 1500 you pay without gaining (constant costs another 1.1 for nothing). Hold long, then let go.

Note this is the HARDER task than the failed first attempt faced: a different seed taught toward a FOREIGN bank, versus that one's own bank, and it scores far higher.


### The distribution, isolated

Same seed, same bank, same abrupt cutoff at epoch 300 — only the sampling differs:

| | identity | tails, of 1000 |
|---|---|---|
| UNIFORM sample (the failure) | 0.584 | 179 / 158 / 264 |
| ATTEMPT stream | **0.784** | 345 / 271 / 421 |

A clean single-variable comparison, and the distribution alone is worth +0.200. The schedule term is NOT cleanly measured against it — the decay arms ran a different seed against a foreign bank, an easier-to-harder change that confounds it — so do not subtract these to attribute the remainder.

### It replicates, and it sets the LEVEL rather than adding an increment

| | untaught baseline | taught, decay 1500 | tails | loss |
|---|---|---|---|---|
| seed 1 -> seed 0's bank | 0.335 | 0.917 | 523 / 632 / 500 | 0.4845 |
| seed 2 -> seed 0's bank | 0.549 | 0.900 | 498 / 560 / 454 | 0.4833 |

The two pairs START from very different agreement and land within 0.017 of each other. Teaching sets the level; it does not add a fixed increment to wherever a seed happened to begin. That is the stronger property, since it means the outcome does not depend on a lucky pairing. It also shows the UNTAUGHT baseline is itself highly variable (0.335 against 0.549 on the same protocol), so the size of the jump is a poor summary and the taught LEVEL is the number to quote.

### THE PRODUCTION CONDITION — a refit, not a reseed

Every arm above holds the data fixed and varies the seed. A refit does the opposite, and the time cut established that a 2% data change re-rolls the basis as hard as a seed change — so seed-stability does not establish data-stability, and only the second is what a refit faces. Bank from the 10%-time-cut fit (data as of ~4 weeks earlier); new fit on the full corpus, same seed, taught toward it:

| | identity (min canonical) | tails, of 1000 | loss |
|---|---|---|---|
| UNTAUGHT (the ordinary refit) | 0.204 | 31 / 214 / 26 | 0.4829 @850 |
| TAUGHT, decay 1500 | **0.899** | 462 / 842 / 783 | **0.4820** @1450 |

Two of the three axes preserve ~80% of their extremes across a real refit, and the extremes are exactly what a motif name claims. Three things make this stronger than the reseed arms rather than equivalent: it starts from a WORSE position than either untaught seed pair (0.204 against 0.335 and 0.549) and still ends highest on tails; the loss is the best measured in this document (0.4820 against the ordinary full-data fit's 0.4829), so a taught refit is not merely no worse but slightly ahead; and it is the operation that actually runs.

ONE FIT. Repeat before shipping. But 0.204 -> 0.899 is far outside anything the noise here produces.

### Tail overlap is scored on ATTEMPTED problems only

A third of the corpus is never attempted and its loadings are unconstrained extrapolation, so their extremes churn freely between fits and were being counted as failures to reproduce. They are ENRICHED in the tails rather than merely present — 57% of axis 2's extremes against a 32% base rate, and axis 2 is the one axis that reproduces well enough to be worth naming. Restricting to attempted problems raises overlap in 8 of 9 measured cases by 10-20% relative. No qualitative conclusion moves, but every tail figure taken before this correction understates what the model actually reproduces.

**Tail overlap cannot be inferred from correlation, and the gap widens where it matters.** Under bivariate normality a correlation of 0.945 predicts 611 shared tail problems of 1000; the observed figure is 375. At 0.964 the prediction is 682 and the observation 219. The bulk of an axis transfers well while the specific problems at its ends keep changing — so the EXTREMES are the least reproducible part, which is unlucky given they are exactly what a human reads to name a motif and what `stats-weaknesses.js` reads to flag one. A drift monitor must therefore measure tails DIRECTLY; scoring correlation would overstate how well names transfer, by a factor that grows precisely in the range where confidence would be highest.

### SHIPPED — teaching is the production trainer's only axis initialization

`pipeline/train_mirt.py` no longer derives directions at all. `burn_in_directions` and the `trunk_out` helper that fed it are deleted; the axis map is pulled onto the motif store by the MSE above, riding `gh` on the attempt stream, `TEACH_WEIGHT` 1.0 decaying linearly to zero over `TEACH_DECAY_EPOCHS` 1000. The store is `axes/label_anchors.npz` — every direction ever kept, keyed by an `ax_<hex>` content hash — and `axes/labels.json`'s `components` lists which ids this fit ships, in axis order; the first `K_AXES` are taught, so that list is the shipping decision. Both are TRACKED in git, being the only record of what a name is attached to. Directions enter the store from a `--dump-params` npz via `auxiliary.rating_dev.update_motif_store`. A missing store, a `components` list shorter than K, an id with no direction behind it, or a store sharing no problems with the corpus each abort the run.

**THE BURN-IN IS GONE ENTIRELY — the loop is uniform.** Pinning `v` at zero for an opening span is the obvious companion to teaching, and it buys nothing; the measurement is the next subsection. What remains is one decaying weight, so there is no `v_pinned` state, no release event and no gating of the axis penalties. The ortho penalty is now on from the first step rather than switching on at full weight partway through, which retires the open concern in section D about abrupt objective changes at the handover.

Two things the shipped step prints, both diagnostics with no flag behind them: how many corpus problems the bank covers, and a per-axis `BANK IDENTITY` report at the end — hinged tail agreement for both ends against the bank, over attempted problems only, with bulk correlation printed alongside as context. Correlation alone gets the ranking backwards, so the tail figures are what governs a name. `axes/axes.npz` also records a hash of the SELECTED motif ids, so a fit can say which directions it was held to without moving when an unrelated motif is added to the store.

### THE v-PIN IS REFUTED — MEASURED 2026-08-15, production trainer

Both arms in the shipped code, same bank (a dev dump on the same corpus), same seed, `--flat-patience` 30, differing only in whether `v` is held at zero for an opening 300 epochs.

| | axis 0 | axis 1 | axis 2 | loss |
|---|---|---|---|---|
| pin 300 | 0.938 / 512 | 0.955 / 640 | 0.953 / 492 | 0.4978 @1090 |
| NO pin | 0.938 / 511 | 0.955 / 630 | 0.952 / 492 | **0.4974** @1110 |

(correlation / shared tail problems of 1000, against the bank, attempted problems only.)

**Identical to three decimals on all three axes, tails within ten of a thousand, and the loss a hair better WITHOUT the pin.** This is not a noisy tie — three independent axes agreeing to that precision is the pin having no effect on what it was kept for. The mechanism argued for it (a free `v` at 33x the net learning rate fits positions against a random projection of a random trunk, then unlearns them) is REAL and visible in the trajectory — the pinned arm leads by ~2 mnat through epoch 300 and the gap closes monotonically — but it is a transient, gone by epoch 1000 and slightly reversed after. A cost paid early and repaid before convergence is not a reason to carry machinery.

Caveats: one seed per arm, and the 0.4 mnat loss difference is well inside the 0.85 mnat seed noise, so read this as "the pin does nothing" rather than "removing it helps". The bank here shares a corpus with the fit, so this is a reproduction test rather than a refit test — but the pin's claimed job is purely about the opening epochs, which this exercises fully.

**THE DECAY MUST END BEFORE THE FIT DOES, and that is a second constraint on its length beyond the frontier above.** `FLAT_PATIENCE` counts staleness against the objective, and while the decay is still running that objective is still changing — so a schedule outlasting convergence stops the run with the bank still holding the map, and the axes are scored before they were ever released, which flatters every identity figure. The margin is not generous: arms measured at a decay of 1500 reached their best epoch at 1090 and 1110 and stopped at 1390 and 1410, with the bank still at roughly a quarter strength throughout. A length picked on a corpus that converges late does not transfer to one that converges early, so this is a value to re-check whenever convergence moves, not a constant to inherit. The frontier is flat enough around here that shortening costs little identity — which is what makes the schedule constraint the binding one.

### Sweeping burn-in against the EIGEN warm start measures nothing

Ranking burn-in lengths by how well two seeds agree measures HOW REPRODUCIBLE AN IRREPRODUCIBLE PROCEDURE IS. The warm start reads the top-K eigenvectors of a residual moment whose trailing eigenvalues sit within 1.4x of each other (0.00048 vs 0.00034 in the trained `cov(v)`), so those directions are unstable by construction and every burn-in length lands somewhere arbitrary. The comparison ranks arbitrary against arbitrary, and even a favourable result yields a MORE REPRODUCIBLY ARBITRARY subspace, which is not what the identity problem needs.

Measured before this was noticed, and worth keeping: burn-in 100 gives WORSE cross-seed agreement than 300 (min canonical 0.121 against 0.335, tails 53/53/244 against 116/275/110). That is consistent with less-converged residuals giving less stable eigenvectors — but it is not a reason to act, for the reason above.

The question that survives supplies a PREVIOUSLY LEARNED direction. Then burn-in length is how long `v` stays pinned while the map is taught: a long one shapes `g` with no pressure from the axis term in the likelihood, a short one releases `v` mid-teaching. That interaction has an actionable answer; the eigen version does not.

## WHAT STILL NEEDS ABLATING OR SWEEPING

Nothing here is a wish list — each entry is a knob or structural choice currently carrying a value that no measurement in this model supports. Grouped by why it is open.

### A. VOID — measured, but through the mis-scaled scorer, so the numbers mean nothing

Each of these was run on 2026-08-13 and read off the broken axis estimator. The rating side of each is still valid; only the axis and combined columns need re-running. Re-run cost is one training run per arm.

| knob | current | what was measured, and is void |
|---|---|---|
| `AXIS_INPUT_DROPOUT` | 0.3 | 0.15 / 0.5 swept, read as flat |
| axis map reading `(b, a, c)` | on | read as no effect; `--axis-blind` is the ablation |
| `V_RIDGE` | 0.5 | 0.05 swept, read as no effect |
| warm start | `burnin` | CLOSED for `none` — see the warm-start section below. `eigen` is retired rather than open: it reads an external fit, the leak this trainer was changed to remove. |
| the `v` pin | REMOVED | CLOSED — refuted 2026-08-15 in the production trainer, see the v-pin section. Pin 300 against none: bank agreement identical to three decimals on all three axes, and the loss marginally better without. The early ~2 mnat the pin buys is a transient repaid by epoch 1000. Removing it took the `v_pinned` state machine, the release event and the axis-penalty gate with it. |

### B. NEVER TESTED — structural

| choice | current | why it is open |
|---|---|---|
| axis path coupled vs detached | COUPLED (shipped) | measured indistinguishable; coupled chosen because detaching makes the trunk learn for the rating alone and hands the axes the result, which is feature reuse rather than a joint fit. |
| `--act relu` | swiglu | SwiGLU doubles trunk parameters at a given width, so width and activation are confounded until one is held fixed. |
| ability granularity | per-user (`--window-months 0`) | measured best on the rating side, but the axes are per-user while ability could be per-window; the mixed-granularity question is dodged rather than answered. |
| an ANISOTROPIC prior on the axis positions | isotropic (`V_RIDGE` scalar) | THE ONLY IN-OBJECTIVE FIX FOR AXIS IDENTITY, and measured-rejected by the detach panel (a 1.3 geometric ladder: cross-seed agreement 0.20, effective dimension 2.5 — two axes sharing a direction still pay less ridge than one carrying it under any per-axis weights, so the duplicate solution stands). The argument for it, kept because it is what a reader reaches for first: The rotation degeneracy survives precisely because the v prior is isotropic: rotate the loadings and the positions by the same orthogonal matrix and the likelihood, `V_RIDGE`, `V_MEAN_PENALTY` and `ORTHO_WEIGHT` are all unchanged. DISTINCT per-axis ridge weights break that symmetry inside the objective, so the basis becomes identified rather than gauge-fixed from outside — no carry-over, no frozen bank, no external anchor. Two cautions. It cannot fix the SUBSPACE being underdetermined (see the warm-start section: two fits at identical loss sharing min canonical 0.029), only which basis of a given subspace comes out. And a prior that distinguishes axes by scale imposes an ordering the data may not support, so the arm has to be scored on loss as well as stability. |

### C. NEVER SWEPT — constants

| constant | current | note |
|---|---|---|
| `V_MEAN_PENALTY` | 10.0 | strong. It is the only thing stopping an axis carrying population difficulty. |
| `ORTHO_AXES` | 0.25, detached | adopted at the one point measured (the detach panel: +1.9 mnat, 7/8 seeds, seven distinct directions, identity unchanged). Unswept on the taught fit in either direction; undetached it is measured only at 0.5 on a free fit, where it costs 13 mnat. `residual_axes` carries the same term as `ORTHO_WEIGHT` 0.5. |
| `SHALLOW_TAU` | 0.1 | the barrier's threshold, inert at weight 0. Only its WEIGHT was swept (30 epochs, 0/0.3/1/3). |
| `SHALLOW_W` | 0 | OFF. This document's own sweep section records 0 as a failure (slopes collapse, median 0.94 / 0.1th percentile 0.097; ratings blow out to [-225, +146]; 22 mnat), measured on an earlier configuration. On the shipping configuration the `barrier` refit panel finds the fit without it marginally better held-out at every seed, the blow-out confined to a small share of flat-slope problems above every player, and ten times the 0.3 weight at no resolved held-out cost — a tail whose width follows the weight, not the data. The flat problems keep the ratings they land on. |
| `GAUGE_MOM` | 0.01 | EMA rate for the difficulty gauge. |
| `BATCH` | 65536 | |
| `STOP_ROW_FRAC` | 1/3 | the rows held out of training, on which the shipped floor, lapse and slope scale are calibrated (`methodology.md` §1.5). A trade between the levels' precision and the rows the positions are fitted on; the share itself is unswept in the ledger. |
| `CHECK_EVERY` / `FLAT_PATIENCE` | 10 / 30 | MEASURED 2026-08-14 and raised from 5. The stop split improves in bursts separated by long flat stretches; the largest gap before the final optimum was 25 checks on one seed and 10 on the other, so anything under 25 halts mid-stretch. At 5 both seeds stopped near epoch 350 against optima at 850 and 580, costing 0.5 and 0.7 mnat of RATING (paired, same seed either side). Buys no axis stability. Runtime ~9 -> ~25 min; the experiment surface keeps 5. Two pairs only — a third seed would firm it up. |
| `PRIOR_AX_SD` (scorer) | 0.5 | CLOSED — MEASURED 2026-08-14 and it is AT the optimum, not below it as this row previously claimed. Swept post-hoc against a fixed pair of user-holdout dumps (seeds 0/1), axis adds in mnat: 0.25 -> 1.73, 0.35 -> 2.00, **0.5 -> 2.14**, 0.7 -> 2.09, 1.0 -> 1.52, 1.5 -> 0.22. 0.7 ties within seed sd. The rating column is identical to four decimals at every value, which is the check that the constant touches only the axis path. STRONGLY ASYMMETRIC — too loose costs far more than too tight (1.9 mnat at 1.5 against 0.4 at 0.25), because a loose prior lets the online axis estimate overshoot on short prefixes; err low if it is ever revisited. |
| `DROPOUT`, `INPUT_DROPOUT` | 0.3 | inherited unswept in this model; measured only on the two-stage one. |

### D. Open questions that are not knobs

- **The axis block overfits faster than the rating heads** — measured on the stopping diagnostic, the axes swing +0.8 to −4.1 mnat between epochs 350 and 650 while the rating moves 1.9. A separate stopping signal for the axis path has never been tried.
- **An LR warmup schedule is REFUTED at two ramp lengths — MEASURED 2026-08-14, under the two-stage trainer; and again under the shipped joint fit by the `ramp` panel (2026-09-03, above): nothing at 250, 500 or 1000 epochs at eight seeds.** `--lr-warmup-epochs`, linear ramp from zero, two seeds per arm. The case for it was endpoint consistency: the axis objective is rotation-invariant, so nothing pulls two runs together and the endpoint is decided entirely by the path — a less perturbation-sensitive path should land two runs closer. Measured min canonical correlation between the two seeds: **0.248 at ramp 100, 0.409 with no ramp, 0.591 at ramp 300** (the ramp must cover the burn-in to matter at all, since the warm start is read at epoch 300). That looks like a trend and is not one: the NO-RAMP baseline's own spread across three seed pairs is 0.409 / 0.596 / 0.659, and all three arms sit inside it. One seed pair per arm cannot separate the effect from the draw. What IS consistent is the cost — ramp 300 is worse on both seeds (0.4848 vs 0.4834 and 0.4864 vs 0.4845, ~1.5 mnat) and converges later (540/730 against 360/330). Resolving it properly needs ~4 seeds per arm, 12 fits, for a hypothesis that has now cost loss twice. The mechanism argues against it too: warmup damps OPTIMIZER-noise sensitivity, but the 2% time cut re-rolled the basis at the SAME seed and init, so the sensitivity is to the DATA, through the warm start's eigendecomposition — whose trailing eigenvalues sit within 1.4x of each other (0.00048 vs 0.00034 in the trained `cov(v)`), and eigenvectors of a near-degenerate matrix swing under any perturbation of the matrix. That is closed-form arithmetic, not a gradient path.
- **The ORTHO PENALTY appears abruptly at the handover, and abrupt objective changes are a known failure mode.** Of the three terms gated behind `if not in_burn`, two are smooth by construction: `V_RIDGE` and `V_MEAN_PENALTY` both act on `v`, which is exactly zero at release, so they start at zero and grow. The ortho penalty is not — it acts on `corr(g) - I` at weight 0.5, and `g` has just been OVERWRITTEN wholesale with the eigen directions. Those are orthogonal in residual-moment space, but `g` is `g_lin(trunk(x))` over the batch's problems and its empirical correlation matrix has no reason to be near identity, so the term arrives at full weight with a potentially large value in a single step. Ramping it in over some epochs is untested. NOT a problem, and ruled out: `g_lin`'s Adam moments are clean at the overwrite, since at `v = 0` the loading gradient is identically zero and the map receives no gradient during the burn-in at all. The same failure in the other direction is MEASURED — a teaching term that VANISHES at the handover does not stick (see `--teach-decay-epochs`), which is what motivated a smooth release there.
- **Trunk initialization and input scaling.** Both currently unexamined rather than chosen: no explicit init on the trunk or rating head (PyTorch defaults), and every input block pooled column z-scored with no normalization layer inside the trunk. At 216 -> 128 -> 64 neither is expected to matter, but that is an argument from the architecture and not a measurement.
- **Fold sensitivity.** Every axis figure here rests on two folds, and the MIRT's own fold-to-fold spread is wide enough that two cannot resolve a small difference. A third and fourth would. (The entry this replaces compared against the two-stage model, which is deleted and cannot be re-run.)

## SUPERSEDED — held-out USERS, 2 folds, paired per fold — MEASURED 2026-08-12

The user holdout is the criterion from here; the problem- and row-holdout tables below are kept as what they are, measurements of different questions.

These numbers were measured with the leaky `eigen` warm start, since they predate its fix. The correction is now quantified rather than open: **the rating figures stand unchanged, and the MIRT's axis figures are inflated by about 0.5 mnat.** Applying it widens production's axis lead from ~2.6 to ~3.1 mnat and leaves the combined comparison a tie either way.

Production is `train_rating_staged --production-flags`, which has no phase 2. All three logloss rows are PARTITIONS OF ONE one-step-ahead prequential walk from each held-out user's first attempt — ability from a running grid posterior, axis position `v` learned online from zero — split by whether training saw the problem (81% seen / 19% not). They are not three protocols.

`± sd` across two folds is fold difficulty, not the model difference; both models are scored on identical folds with identical denominators, so the PAIRED per-fold gap is the statistic.

```
rating only                f0 prod -> MIRT      f1 prod -> MIRT     MIRT-prod (mnat)
  forward walk (all)       0.4955 -> 0.4929     0.5031 -> 0.5000      -2.6 / -3.1
    seen      (81%)        0.4890 -> 0.4863     0.4928 -> 0.4889      -2.7 / -3.9
    never trained (19%)    0.5237 -> 0.5217     0.5463 -> 0.5462      -2.0 / -0.1

rating + axis
  forward walk (all)       0.4917 -> 0.4920     0.5001 -> 0.4993      +0.3 / -0.8
    seen                   0.4872 -> 0.4865     0.4913 -> 0.4896      -0.7 / -1.7
    never trained          0.5114 -> 0.5158     0.5364 -> 0.5398      +4.4 / +3.4

axis adds (mnat)              production            MIRT
  forward walk (all)         3.82 / 3.09         0.93 / 0.73
    seen                     1.85 / 1.44        -0.22 / -0.63
    never trained           12.33 / 9.96         5.95 / 6.37
```

**THE AXIS AND COMBINED ROWS ABOVE ARE VOID** — they were taken through the mis-scaled online axis estimator described in the next section, which ran the two models at ~6x different effective rates. The RATING rows stand: the scorer's axis path never enters them. Rescored on fold 0 under the Newton estimator, production's axes give +3.66 and the MIRT's +1.48, for combined 0.4919 against 0.4915 — the MIRT ahead on the joint number rather than behind, because its 2.5 mnat rating lead exceeds production's 2.2 mnat axis lead. Read the rating conclusion below; ignore the axis one.

**Joining the fits improves the rating and degrades the axes.** MIRT's rating head leads by ~2.9 mnat (6/6 paired) and production's axis layer leads, though by how much is now a question for the re-measurement rather than these numbers.

**Survey rho is not usable on this holdout.** It is restricted to held-out accounts' declarations — 92 on seed 0, 67 on seed 1, against the 374 that gave the ±0.012 resolution quoted below. Production 0.9108 / 0.8061, MIRT 0.9167 / 0.7766: the fold-to-fold swing of 0.10 dwarfs both model differences, which flip sign between folds. Do not quote it.

## THE SCORER WAS MIS-SCALED — found and fixed 2026-08-13. EVERY AXIS NUMBER BELOW IS VOID.

`eval_user_holdout.forward_walk` estimates a held-out user's latent quantities causally, from their own prefix, and it estimated the two of them to wildly different standards.

**Ability** rode a running grid log-posterior: the grid log-likelihood is additive, so the sufficient statistic accumulates and `theta` is the exact posterior mean given every prior attempt. No rate, no approximation.

**The axis position** rode ONE SGD step per attempt, `v += v_lr*(y-p)*g`, at a hard-coded `v_lr = 0.05`. Three defects, compounding:

1. **The rate multiplies the loading scale, which is a parameterisation choice.** A step moves the prediction by `v_lr * sum_k g_k^2`. The MIRT RMS-normalises `g` to 1 and keeps the magnitude in `v` so `V_RIDGE` can bound it; production's loadings carry their own scale, RMS 0.41. So one shared rate ran the MIRT's axis positions ~6x faster than production's. Swept on fold 0:

```
                 0.005    0.01    0.02    0.05    0.1     0.2     0.4
  MIRT           +1.51   +1.74   +1.50   -0.47   -5.38  -13.06  -26.16
  production     +1.06   +1.77   +2.67   +3.82   +4.28   +3.93   +1.86
                                          ^ the hard-coded default
```

The default sat on production's optimum and ~5x past the MIRT's. **The MIRT's axes read -0.47 mnat when they are worth +1.74.**

2. **The gradient was the plain-logistic `(y - p)*g`**, not the 4PL one. With a floor and lapse, `dP/dz = span*s(1-s)`, so a miss on a high-floor problem carries far less information about `v` than the logistic form claims, and the update over-credited it.

3. **One step is not what causality requires.** Causality only asks that attempt *t* be predicted from attempts 1..*t*-1, which a full refit satisfies too — as the ability estimator demonstrates in the same loop.

Fixed by replacing the SGD step with a **recursive Laplace (online Newton) posterior** over the same prefix: accumulate the Fisher information `sum w*g g^T` with `w = (dP/dz)^2 / (P(1-P))`, take a Newton step. **This has no rate and is invariant to the loading scale by construction**, since the information term carries `|g|^2` and divides it back out — so the question of whose rate to use, and whether to tune it on the scored rows, stops existing. The single remaining constant `PRIOR_AX_SD` is stated in PREDICTION space (logits) rather than on `v`, because rescaling `g` by `c` and `v` by `1/c` leaves predictions identical, so only `g^T.Sigma.g` is well-posed. K is read from the loadings; nothing assumes an axis count, since the two models select K independently.

**Consequence: every axis figure recorded in this document, and every axis-side conclusion drawn on 2026-08-13 — axis dropout, the rating-output columns, `V_RIDGE`, the warm-start arms — was measured through the broken scorer and is void.** Rating figures are untouched: the scorer's axis path never enters them.

## THE WARM-START LEAK — found and fixed 2026-08-13

`train_joint_mirt` initializes its axis map from `eigen_directions`, which needs an axis-free 4PL to take residuals against. A single-stage model has none at initialization, so it calls `build_axis_features()` with no `params_dir` — reading the **shipped `ratings/`**, a two-stage fit trained on every user and every problem. Two independent leaks follow, both favouring the MIRT:

- **User holdout.** `--holdout-user-frac` masks rows from the SGD training set only. It never reaches `rbu`, `ax_users` or `tu_ax`, which are filtered by the *axis* fold alone, so held-out accounts warm-start the model directly: **66 of 383 warm-start users (17.2%, 31,931 attempts) on seed 0, and 52 of 383 (13.6%, 20,589) on seed 1.**
- **Problem holdout.** The shipped per-problem parameters were fitted on all 153,720 problems, the 20,315 held-out ones included, and `rbu` carries attempts on them.

Production's side is clean in both: `fit_axes_userholdout` excludes held-out users and reads the matching per-fold rating fit, and `prod_axis_heldout` excludes held-out problems. So the leak was one-sided, inflating the MIRT. Measured after the fix it is worth **0.51 mnat on the axis column and ~0 on rating** — real, and smaller than the sizes feared while it was open.

The rank curve is a second, smaller leak shared by BOTH models: `build_axis_features` reads the shipped `rating_to_rank.json` for its rank column regardless of `params_dir`.

The fix is `auxiliary/rating_dev/burn_in_warm_start.py`, wired in as `--warm-start burnin`: the warm start derived inside the pass. At `v = 0` the loading gradient is identically zero, so a burn-in with `v` pinned at zero trains exactly the axis-free baseline the residual scores are defined against, and taking them over training rows only excludes held-out users and rows by construction rather than by a filter. It reads no shipped artifact, so the rank-curve leak leaves the MIRT with it.

A second defect of the borrowed form, independent of the leak: `eigen_directions` works in the 216-column axis feature space and the result is truncated by `gi_full[:, :fdim]` to the MIRT's 212 **without renormalising**, so the initialization is a non-unit projection of directions chosen partly along four coordinates this model does not have — including the two-stage model's own rating and slope.

### RESOLVED 2026-08-13. The leak was worth ~0.5 mnat; the warm start now reads no external fit.

Two fixes landed. `--holdout-user-frac` now reaches the warm start's own user set — it previously reached only the SGD row mask, which is why held-out accounts were in it — and `--axis-params-dir` lets the `eigen` construction read a per-fold rating fit instead of the shipped one. With both, a clean `eigen` is measurable, and the default moved to `burnin`, which reads no external fit at all.

The axis column here is VOID (mis-scaled scorer, see its section); the rating column and the seed spreads stand, and the warm start's ~2.8 mnat margin over random init is far too large to be an artefact of the estimator's rate.

```
                       rating              axis adds (mnat, VOID)
  eigen (leaky)     0.4930 +/- 0.0003      +0.68 +/- 0.19
  eigen (clean)     0.4931 +/- 0.0001      +0.17 +/- 0.30
  burnin (300)      0.4933 +/- 0.0004      -0.13 +/- 0.23   <-- default
  none              0.4932 +/- 0.0001      -2.64 +/- 0.86
```

**The leak was worth 0.51 mnat on the axis column and ~0 on rating** — real, but far smaller than the caveat this section previously carried. Every rating figure in this document stands; the axis figures were inflated by half a mnat.

**`burnin` and a clean `eigen` are indistinguishable.** Paired by seed the difference is +0.67 and −0.07, i.e. 0.30 +/- 0.52 — nothing two seeds can resolve. So `burnin` is chosen on STRUCTURE: no shipped `ratings/`, no shipped rank curve, and the leak class gone by construction rather than by a filter that has to be remembered.

**The warm start itself is firmly vindicated** — ~2.8 mnat over random init even after cleaning. And random init's failure here is NOT the one `train_axes` recorded (sd 0.126, one draw in three collapsing): the rating is seed-stable to 0.0001 and the axes fail *consistently*, at −2.64. Quote that finding with the model it was measured on.

Two design points that came out against expectation, both worth not re-deriving. **A fixed burn-in beats running to convergence** (spread 0.23 vs 1.46): the axis-free fit's convergence epoch is itself seed-dependent — 380 and 550 on the two seeds — and since the outcome is epoch-sensitive the criterion inherits that variance, so a fixed handover removes a variance source rather than papering over a missing criterion. **Do not rewind to the axis-free best at handover** (costs 1.4 mnat, quadruples the spread): the two-stage analogy fails because there the fit ENDS at its best epoch, while here training continues, so restoring parameters while Adam's moments describe a later trajectory resumes the fit with one epoch's weights under another's momentum.

### The ablation that established it — MEASURED 2026-08-13

At the swept optimum `3e-4 / 1e-2`, held-out users, holdout seed 0:

> **The pair was RE-MEASURED on the shipping configuration** by the `lr` refit
> panel (`ablations/panels/lr.json`, replicated across seeds). What it finds:
> the field rate is not resolved anywhere between this 33x ratio and 1x, and one
> shared rate does not diverge — so the divergence the split was adopted against
> belongs to an earlier arrangement. The network's own rate is the sensitive one.
> The optimum recorded here is not contradicted; what is contradicted is the
> reason it was believed to be load-bearing on both sides.

```
                    rating            axis adds (mnat)
  eigen  (2 seeds)  0.4930 +/- 0.0003   +0.68 +/- 0.19
  burnin (2 seeds)  0.4944 +/- 0.0021   +0.22 +/- 1.20
  none   (3 seeds)  0.4932 +/- 0.0001   -2.64 +/- 0.86
```

**The warm start earns its place** — `eigen` beats random init by 3.3 mnat on the axis column, with a tight seed spread, and all three `none` seeds put the axes firmly negative. So the leak cannot be resolved by deleting `eigen_directions`. Note the value is entirely axis-side: the rating is the same to within 0.4 mnat whether or not `g` is warm-started, which is what the mechanism predicts, since at `v = 0` the loading gradient vanishes and the warm start is a statement about `g` alone.

**The burn-in at `BURN_IN_EPOCHS = 100` is not a usable replacement.** It is worse than both alternatives on rating and its seed spread is ~7x wider on rating and ~6x wider on axes. The suspected mechanism is that the residuals are taken against a baseline nowhere near converged — the stopping diagnostic puts held-out rating at 0.4980 at epoch 100 against 0.4928 at 300 — so they carry rating error rather than skill structure. Centring removes the direction in which the baseline mispredicts for EVERYONE; an undertrained baseline also mispredicts in user-specific ways correlated with ability, which centring cannot touch, and those seed `g` with directions that duplicate the rating. That fits the symptom: the damage lands on the rating, not only on the axes.

`--burn-in` makes the length a knob so this is testable. If a converged baseline closes the gap the mechanism is sound and 100 was simply wrong; if it does not, the warm start's value depends on having a converged EXTERNAL fit, which is a genuine problem for the single-stage thesis rather than a tuning matter.

This section is superseded by the resolution above: the leak is fixed, `burnin` is the default, and the rating figures stand.

## Results — held-out samples, 2 folds, mean ± sd

81,439 rows (seed 0) and 81,030 rows (seed 1), identical for both models: same problems, same profile structure, same denominators.

```
                    rating only        rating + axis      axis adds (mnat)
  production      0.4952 ± 0.0000    0.4971 ± 0.0006    -1.876 ± 0.516
  MIRT            0.4986 ± 0.0009    0.4984 ± 0.0008    +0.142 ± 0.030
  MIRT shortfall    3.39 ± 0.83        1.32 ± 1.44

  per fold:  rating only   production 0.4952 / 0.4952   MIRT 0.4992 / 0.4980
             rating+axis   production 0.4967 / 0.4975   MIRT 0.4990 / 0.4978
  constant-rate baseline   0.5253 / 0.5266  (base rate 0.781 / 0.780)
```

**Rating alone: production is ahead by 3.39 ± 0.83 mnat**, both folds agreeing in direction, and production is fold-invariant to four decimals. Read it against the ~30 mnat of total signal available over a best-constant predictor: the MIRT delivers about 87% of production's discrimination.

**Rating + axis: unresolved.** The interval spans zero. Production's constructed axis layer costs it 1.88 mnat while the MIRT's gains 0.14, and that swing closes most of the gap. Per fold the combined shortfall is 2.34 and 0.31 — the two draws disagree by more than the effect.

`n = 2` makes "± sd" a range with a symbol on it. The rating column survives that (gap is 4x its spread); the combined column does not, and needs a third and fourth fold before anyone quotes 1.32.

**Production has no native rating+axis prediction** — its axes feed the client's weakness pane and never enter the rating. That column is constructed (`prod_axis_heldout.py`): its frozen 4PL with `sum g*v` added to the logit, `v` fitted per user on training rows, `v = 0` recovering rating-only exactly. Every link excludes the held-out problems — the 4PL, the axis features (which supply four feature columns AND the axis baseline), the axis fit, and `v`. One residual leak, stated not hidden: the rank feature column still comes from the shipped `rating_to_rank.json`, a global monotone map fitted on survey declarations rather than held-out outcomes.

## Survey rho — the external check

Never used in training by either model. `survey_weighted_rho`, per-server Spearman averaged with user-count weights, alpha cross-fitted per run.

```
                    base rho             time-adjusted
  production      0.8672 / 0.8690      0.8903 / 0.8931   (alpha -0.60 / -0.43)
  MIRT            0.8675 / 0.8681      0.8880 / 0.8918   (alpha -0.30 / -0.43)
  MIRT shortfall  -0.0003 / +0.0009    0.0023 / 0.0013
```

**Indistinguishable, and it cannot say more.** The paired user bootstrap (`compare_survey_rho.py`) puts this statistic's resolution at about **±0.012** with 189 users and 374 declarations, an order of magnitude above the 0.0018 difference. Quote the TIME-ADJUSTED figure: every server coupling was fitted against time-adjusted ratings, so the untimed number compares two sides in different units and fails silently.

Note what this does and does not corroborate. Survey rho reads `rating` and `slope` only — no axes — so it is a RATING-side criterion. It agrees with the rating+axis column in *verdict* (both unresolved) but for unrelated reasons: the combined column is unresolved through fold variance, survey rho through low power. It does NOT corroborate the rating-only result, which held-out logloss resolves and survey rho cannot see at all. When two criteria disagree, the one with power decides.

## Three holdout forms, and what each answers

One generator (`auxiliary/rating_dev/holdout_set.py`), one hash discipline, wired into both trainers. They are not a hierarchy — they are three different requirements, and the two models are NOT uniformly ordered across them.

```
kind      seed 0 hash          size                     asks
problem   6619fdc487345fc0     20,315 problems          can it price content nobody has played?
user      2ac648dc60813366     576 accounts             can it serve someone new, from attempt 1?
row       db89466ad768dd9b     50,997 attempts          can it predict a known user's next attempt?
```

The ROW holdout is the client's steady-state condition: the problem keeps its other attempts and the user keeps their other rows, so parameters are informed exactly as in deployment and only the scored attempt is unseen. The PROBLEM holdout is the opposite extreme (parameters from the feature prior alone) and the USER holdout is a blend — measured at **79% of its scored attempts on problems that ARE in training, 21% on problems that are not**, near-identical across two folds (79.0/21.0 and 78.8/21.2).

All evaluations here use `load_first_time_attempts`, which filters `is_repeat = 0` and then dedups to the earliest attempt per (user, problem). Training and scoring are NOVEL attempts throughout: one row per user per problem, the first time they saw it.

## K is hardcoded in the MIRT

K equals the number of `components` in `axes/labels.json` — one axis per taught rule, so the count follows the shipping decision rather than a loss curve (see the production-rule-set section above). There is no incremental search either way: K is baked into the shared head, so each value is a full training run.

## The framework

```
holdout_set.py            the shared problem holdout; hash-confirmed on both sides
train_joint_mirt.py       --holdout-seed --window-months --dump-params --drop-recent-frac
eval_user_holdout.py      the Newton scorer: rating / rating+axis / axis increment
axis_by_history.py        the same walk, split by attempts PRECEDING the scored one
compare_heldout_logloss.py  paired logloss, user-CLUSTERED bootstrap
compare_survey_rho.py     paired survey rho, user bootstrap
```

**Queue fits with `--fit-only`.** The survey-rho and axis-criterion evaluations that follow a fit run on CPU and can outlast the fit itself — measured 2026-08-14 at ~40 minutes of evaluation against a ~9 minute fit, with four queued runners blocked behind it and the GPU idle. `--dump-params` is written BEFORE any of it and every scorer here reads dumps independently, so a sweep loses nothing by skipping it and runs roughly five times faster.

**Give every arm the same patience, and enough of it.** Twice now a comparison has been set up where one arm converges and another is truncated, which is the defect that voided the A-list. The `none` warm-start arm needs 2670-3200 epochs against the baseline's 580-850: at `--flat-patience 5` it reads as 25 mnat worse, at 30 it is loss-equivalent. Before quoting any margin, check where each arm's best epoch fell relative to where it stopped.

Both bootstraps are **paired** and cluster by user. Attempts within a user share an ability estimate, so a row-level resample understates the interval — in the direction that makes a difference look real.
