# Joint MIRT: one model for the rating and the skill axes

**Status: development, nothing shipped.** The production path is unchanged — `pipeline.train_rating` then `pipeline.build_axes`, as described in `docs/methodology.md` §1.3–1.11 and §1.14. This document records why a single-stage replacement is being tried, what its architecture inherits from measurement rather than from taste, and which questions are open. It is a development log and is allowed to talk about history and alternatives; `docs/methodology.md` is not, and must not absorb this until something ships.

## Why

The shipped arrangement fits two models with two regimens. The rating comes from a trunk under one dropout scheme. The axes come afterwards from a linear map under a different one, defined against a frozen 4PL, with a quarantined residual head to absorb whatever the first stage left behind.

Under the standard multidimensional-IRT reading that split is artificial. The axes are not a different kind of object from the rating — they are factors 2…K+1 of one model, and the rating is factor 1. Each is a per-problem quantity multiplying a per-user quantity inside the same logit:

```
rating:   a_p · (theta_u − b_p)
axes:     sum_k  g_k(x_p) · v_u[k]
```

Collect the terms and the shipped system already *is* a compensatory M4PL, merely fitted in two passes:

```
P = c_p + (1 − c_p − lambda) · sigmoid( A_p · Theta_u + d_p )

    A_p     = [a_p, g_1(x_p), …, g_K(x_p)]     per-problem discriminations
    Theta_u = [theta_u, v_u[1], …, v_u[K]]     per-user abilities
    d_p                                        per-problem intercept
    c_p                                        per-problem guessing floor
    lambda                                     one global slip rate
```

The one unavoidable departure from textbook MIRT is that `A_p`, `d_p` and `c_p` come from a **feature net** rather than being free per item. About a third of the corpus has no attempts at all, so free per-item parameters cannot price it. This is the same choice production already makes — free per-problem *residual* parameters are separately measured-rejected, and the distinction matters: predicted-from-features generalises to the unattempted tail, free does not.

## What the architecture inherits from measurement

Every asymmetry below was measured on the two-stage trainer before being carried across. None of it is a preference.

**The axis block is a linear map with column dropout on its input.** Giving the axis path its own trunk — added alongside the warm-started linear map, so capacity is separated from initialisation, and not shared with the residual head — costs −0.430 pooled and −0.413 per-user across both folds and both seeds. Input dropout on the linear map instead *gains*, positive in 9 of 9 rate × seed comparisons and peaking at 0.3 with +0.164 per-user.

**The rating heads keep a trunk**, with the rating trainer's dropout regimen. Their parameters are informed by every attempt on a problem, so they tolerate capacity the axes cannot.

Those two are the same fact seen twice. **The signal available to a factor falls off with its index**: ability is estimated against the dominant source of variance in a user's history, while the k-th axis position is estimated from the same history against whatever earlier factors left. In a bilinear form a poorly-determined user factor contaminates its problem factor, so the later loadings are effectively fitted on far less usable signal even though both see every attempt. Capacity should therefore decrease with k, and regularisation increase. That predicts the measurements rather than accommodating them, and it also predicts the input-dropout asymmetry: the same column dropout that gains +0.164 on the axis map costs −0.086 on the residual trunk, which sees every attempt in the fold and needs no help.

**The residual head `r` is gone**, on the measurement and not on the argument first recorded here. That argument — `r` buys back freedom lost when the intercept is forced to factor as `−a_p·b_p`, and a free `d_p` supplies it directly — is void. For any `(a, d)` with `a > 0` the factorisation reproduces it exactly at `b = −d/a`, so the two forms carry identical parameter counts and neither is freer than the other; they differ only in which quantity the gauge constrains. What survives is `r`'s measured size: removing its capacity costs 0.05 pooled and 0.18 per-user, and 7,489 parameters saturate it against the shipped 16,001. A correction that small and that bounded is what a per-problem intercept fitted jointly with the axes should absorb, whichever way the logit is written.

**The axis warm start is kept.** At `v = 0` the loading gradient is identically zero, so a randomly-started axis escapes along its draw rather than along the data.

**Features exclude the rating model's own outputs.** The two-stage axis fit reads rating, slope, floor and rank as four of its columns, which is sound only because they come from an earlier frozen fit. Here they would be circular.

## Three criteria, deliberately not merged

Each half faces more than one kind of novelty in deployment, and each criterion is scored against one of them.

- The **rating** must price problems nobody has attempted — a third of the shipped corpus. Held-out *problem* logloss answers that.
- The **axes** must describe users nobody fitted on, since the directions are chosen from the fitted accounts' residuals and could be overfit to them. Held-out *user* gain answers that.
- **The axes must also survive on problems nobody has attempted**, and this is the criterion the two-stage harness structurally cannot produce. `axes_ll_gain` walks a user's own history, so every problem it scores is by construction one somebody played. An axis that is accurate where people have played and noise everywhere else passes every test the two-stage pipeline has. That matters because the loadings ship for the **whole** corpus while the axis loss only ever sees attempts, and the unattempted tail is not a random sample of feature space — it is the KataGo-hard region, systematically elsewhere. `methodology.md` already carries this caveat for ratings ("extrapolated rather than interpolated, and per-problem calibration there is not established"); it applies to the axes with equal force and has not been stated. Measured here as held-out-problem logloss with the axis term zeroed against the same rows with it active.

These are separate questions, not several answers to one, and merging them would need an exchange rate nobody can justify. All three are reported per arm. The axis gain is scored through the same harness and against the same shipped-4PL baseline as every other axis measurement, so it sits on one scale with them rather than being only internally consistent.

**Axis gain is strongly sensitive to the quality of the 4PL it is scored against, so every axis number is partly a statement about its baseline.** Measured directly across the defect fixes: on the same arm, gain against a FIXED external baseline (the shipped 4PL) barely moved, 0.987/1.386 → 0.953/1.291, so the loadings themselves were unchanged. Gain against THIS MODEL's own 4PL roughly halved, 1.579/1.502 → 0.637/0.859, because that baseline improved by 31 mnat. So about half of what the axes appeared to contribute was explaining an undertrained rating model. **0.64 pooled / 0.86 per-user, against a converged baseline, is the honest figure** — and an axis number that held constant while its baseline improved that much would have been the suspicious outcome, not the reassuring one. It also means this model's axis figures were never comparable with the two-stage ones until the rating side converged, whatever they appeared to say.

**The own-4PL criterion is also FRAGILE and must be checked before it is quoted.** A negative pooled gain is impossible for a converged fit — the calibration is fitted in-sample on the rows it is scored on and its design carries an intercept, so `w = 0` is always attainable — therefore **any negative value is a failed solve, never a result about the axes.** It has now failed twice, and across six clean runs the failure is the arm with by far the widest rating range (135 units against 68–79 elsewhere, its positive extreme +58 against a maximum of +18 anywhere else). That is consistent with leverage in the per-user IRLS, whose design column is the difficulty: `_walk_forward` already standardises the axis score for exactly this reason and leaves the difficulty column raw, and standardising would not be enough anyway, since it fixes scale and not leverage. Note the percentiles show nothing — the failing arm's 99.9th is an unremarkable 6.76 — so the min/max print is what makes this visible at all. Harden the harness before relying on this criterion.

**Consequently the own-4PL criterion CANNOT be used to select between arms.** It rewards leaving residual on the table, since the axes are scored on what the rating did not explain.

An earlier version of this note claimed the two criteria were *exactly* inverted, from three window arms at one seed that ranked perfectly in reverse. **That does not reproduce and should not be quoted.** On the shared held-out split the production-window arm has both the worse rating (0.4992 against whole history's 0.4969) and the lower axis gain (0.564/0.793 against 0.649/0.940) — same direction, not opposite. Three points on one split was never enough to assert a law, and the anti-correlation argument only says the criterion is *contaminated* by baseline quality, never that it is monotonically determined by it. Arm selection reads the **rating** criterion, or the shipped-baseline reference, which holds the baseline fixed and is therefore the arm-invariant measure of loading quality; on that reference the same two arms are indistinguishable (0.953/1.291 against 0.954/1.291), so the ability windowing does not affect the axes at all. The own-4PL number answers "what do the axes add to *this* model", never "which model".

**Measured seed stability of the three criteria, same arm across two seeds** — worth knowing before choosing which to read: the RATING criterion moves 0.0001–0.0005, the shipped-baseline axis reference moves 0.002, and the own-4PL axis moves 0.034–0.052, i.e. more than the arm differences it would be used to judge. The fixed-baseline reference is stabler than the own-4PL one because it varies only with the loadings, while the own-4PL varies with the loadings *and* the baseline beneath them. So the rating criterion is the sharp instrument by three orders of magnitude, and of the two axis measures the fixed-baseline one is both the arm-comparable and the more precise.

**The float-rounding floor on that scale is ~0.03.** Two algebraically identical formulations of the same penalty, differing only in summation order, produce final gains 0.014 pooled and 0.028 per-user apart — a 1e-8 perturbation amplified over 400 epochs of a bilinear fit. Nothing below that is a result. A related caution: across five separate experiments the **per-user** column carried a consistent signal while pooled did not, because pooled is dominated by a handful of heavy accounts and has far fewer effective observations than its row count suggests.

## Simplifications being tested

Ordered by how much they remove.

**Global slope** (`--global-slope`) — one discrimination for the whole corpus, difficulty still per-problem: the Rasch reduction of factor one. Motivated by §1.11's finding that most of the phase-2 multiplier is rating-axis gauge rather than a statement about the slope, and by the earlier record that per-problem slope is largely assignment noise. If held-out problem logloss is unchanged it removes a head and takes the per-problem slope out of the shipped artifact.

**SwiGLU for ReLU** — currently the default. It **doubles the trunk's parameters** at a given width, because the gate is its own projection, so a same-width swap is also a capacity increase and must be read against both a same-width and a parameter-matched ReLU control. On the two-stage trainer, same-width SwiGLU lost (−0.070 / −0.145) while a narrower one gained (+0.070 / +0.058) — non-monotone in parameters, so neither arm alone attributes the effect.

**Halving the residual trunk** — moot here since `r` is gone, but measured free on the two-stage model at `(32,16)` against the shipped `(64,32)`, and worth applying there if the two-stage path survives.

## Constants do not transfer from `train_rating`, even where the code is identical

This has now cost four separate debugging rounds, each starting from the same reasoning — the structure is the same, so the constant should be too — and each wrong. The structure being identical is exactly what makes the mistake easy: nothing looks off at the call site.

- **Batch size.** 65536 is sized for a corpus of over a million first-time attempts. On the axis fit's ~136k rows per fold it yields three minibatches per epoch, so a 200-epoch run is 600 optimizer steps against the incumbent's ~6800. Matching the *step count* is what makes two fits comparable, not matching the constant.
- **Learning rate.** The rating trainer uses a single 0.01 because its only per-user parameter is one ability embedding. The axis fit needs the network an order of magnitude slower than the per-user field, and collapsing them to one rate trains the trunk ten times too fast and diverges late. **Confirmed in the joint model, 2026-08-12 — defect 2 above.** This bullet was written here from the two-stage trainer and then not applied to a model with strictly more per-user structure, which cost every arm of the first window sweep. The lesson is not the constant; it is that writing a transfer failure down does not prevent repeating it in the next model that has the same shape.
- **Barrier form and weight.** `mean(1/a²)` at 1e-4 balances in the rating trainer. Here it drove the median slope to 929 and the 99th percentile to 1.2e7 while held-out logloss worsened 75% — because that form is unbounded below in the slope, so its gradient runs as −2/a³ and it does not say "do not collapse" but "larger is always better". Replaced with the high-floor barrier's own form applied on the other side and **in the same space**: `softplus(o_tau − o_a)²` on the *pre-softplus* slope logit, mirroring the floor barrier acting on its raw pre-sigmoid logit. Comparing slopes directly does not work — at a median slope of 0.4 with `tau` = 0.1, `softplus(tau − a)` is still 0.55, so the penalty pushes healthy slopes as hard as the form it replaces.
- **Epoch count.** 200 is provenance rather than validation in the rating trainer's own comment; here the held-out signal peaks and turns well before any fixed budget, which is why the stopping rule watches a separate held-out split.

The general rule: a constant is only meaningful relative to the loss landscape it was tuned against. Carry the *reasoning* across, re-measure the *value*.

## Four defects that made every earlier arm unreadable — found and fixed 2026-08-12

Everything measured on this model before 2026-08-12 should be discarded. Four independent defects were live at once, and none of them was the thing under investigation; three would have silently corrupted any redo as well. Fixing the two model-side ones is worth **31 mnat/attempt** on held-out problems — same arm, same split, same seed, 0.5296 → 0.4985 — against 0.17 mnat for the axes' entire marginal contribution and roughly 7 mnat for the widest window-length gap the sweep was built to resolve.

The sections below record each defect, because each was invisible in a different way and the failure modes are the transferable part.

### 1. The gauge was on the intercept, so the rating scale floated

The dev code wrote `z = a_p·theta_u + d_p` and whitened **`d`**, the intercept. `train_rating` writes `a_p·(theta_u − b_p)` and whitens **`b`**, the difficulty. Those look interchangeable and are not.

Two freedoms have to be removed — **shifting** every ability by a constant, and **rescaling** every ability — and whitening the intercept removes only the first.

- **Shift.** Add a constant to every ability, and keeping predictions unchanged forces the intercept to absorb that constant times the slope. That drags the intercept's average off zero, which whitening forbids. Blocked either way.
- **Scale.** Double every ability and halve every slope, and the intercept **does not have to move at all** — it is simply added on, not multiplied by anything that changed. Its average is still zero and its spread still one, so whitening has no opinion. Not blocked.

Whitening difficulty blocks both, because difficulty is *multiplied* by the slope: rescale the slope and difficulty must rescale with it, which the spread-one constraint forbids.

So the earlier claim that "the gauge is unpinned" was too loose — location was pinned, scale was not — but scale is the one that matters, because the rating **is** an ability, so under the unblocked rescaling every rating scales with it and the whole axis floats. Nothing else in the loss has an opinion: ability carries no ridge, the axis-position ridge and orthogonality terms act on the axis block, the floor barrier acts on the floor. **The only term breaking the invariance was the shallow-slope barrier**, which pushes the slope up one-sidedly — so a guard against a numerical failure was setting the units of the rating.

Two consequences, both already visible in runs recorded here:

- The `SHALLOW_W` sweep reads as a regularisation sweep and is partly a gauge sweep. At `w` = 3 "the whole corpus spans −2.2 to +1.1" and at `w` = 0 ratings reach ±10³; since `rating = (kappa − d)/a` with `d` pinned at sd 1, the rating spread is essentially `1/a`, and the barrier sets `a`. The weight was then selected on held-out logloss, which is **exactly invariant** to the gauge and so cannot have chosen the scale on any evidence at all.
- Every arm reports a 0.1-percentile rating near −12 against a 1st percentile near −1. **This is not evidence of anything and must not be quoted as a symptom.** The corpus contains genuinely trivial problems just as it contains a KataGo-hard tail, and the rank coordinate is deliberately unbounded, so an extreme rating at either end is exactly what a correct model of an extreme problem produces. The magnitude cannot distinguish a real intercept from a small `a`. What *would* distinguish them is the mechanism — whether the extreme-rating problems carry a small `a` or an ordinary `a` with a large intercept — and that has not been measured; the model is not checkpointed, so asking costs a re-run.

The deeper reason is that the rating is a **ratio** in this parameterisation and a **head output plus a bounded correction** in the shipped one:

```
here:     rating = (kappa − d) / a      slope divides the whole quantity
shipped:  rating = b + kappa/a          slope enters as an additive term,
                                        bounded by the floor clamp and SLOPE_FLOOR
```

So the slope/rating trade-off is unbounded here and bounded there. **Whitening `theta` does not fix it** — it pins the global unit while the trade-off lives entirely on the problem side, leaving `d` free to sit obediently at sd 1 while `a` goes small.

**Fixed** by adopting the shipped parameterisation verbatim: emit `(b, a, c)`, whiten `b`, write `z = a·(theta − b) + sum_k g_k·v_u[k]`. It costs no degrees of freedom (see the note on `r` above), restores an identified slope, and makes the compression scalar measurable and comparable against the shipped value — which is still to be done.

Nothing shipped was ever affected — `pipeline/train_rating.py` whitens `b`.

### 2. One learning rate for a trunk beside two free per-user fields — every run diverged

Every arm diverged, at epochs between 40 and 110, and reported its pre-detonation snapshot as if it were a converged fit. The signature is unambiguous: **eval-mode** training loss rising — 0.4603 to 0.5209 between epochs 70 and 90 in one case, a jump of 0.13 in ten epochs in another — so the model was getting worse on its own training data, which overfitting cannot do.

The cause was `Adam(params, lr=0.01)` for everything. This model has a trunk *and* a free ability field *and* a free K-dimensional axis position per user; the rating trainer gets away with one rate because its only per-user parameter is a single ability embedding. **This document already contained that warning**, in "constants do not transfer" — written from the two-stage axis trainer and then not applied here, which is the more useful lesson than the constant itself.

The gauge parameterisation is not the cause and does not mitigate it, measured as a clean one-variable test: the difficulty-whitened fit diverged at epoch 80 where the intercept-whitened one lasted to 110.

**Fixed** with split rates — network at 1e-3, the free fields at 1e-2, both exposed as flags. The run then descended monotonically to a genuine plateau at epoch 190, rolled over into ordinary overfitting, and stopped at 240. A second signature confirms the diagnosis: at equal epoch 40 the split-rate fit matched the old one's held-out loss exactly (0.5051) with a training loss 0.024 *higher* — the old configuration was buying its progress with memorisation, which is also the classic precursor to the instability.

### 3. The dev window helper doubled every width, so production was never in the sweep

Production keys profiles by month and writes each attempt to month M and M−1: a **2-month window stepping 1 month**. The dev copy keyed by a `W`-wide bucket and wrote to that bucket and its predecessor: a **2W-month window stepping W**. So every arm was mislabelled by a factor of two and the incumbent configuration was absent from a sweep built to evaluate it.

**Fixed** by separating width from step — an attempt in month M is written to the last `W` monthly buckets, so width is exactly `W`, step is exactly 1, and `W = 2` reproduces production. Verified against `build_profiles` directly: same row count, same unit count, identical partition. A consequence worth knowing is that rows now scale with width, so wide arms cost proportionally more compute; under the old code every arm cost the same.

### 4. The criterion averaged over users no window choice can reach

Two-thirds of fitted users have all their attempts inside a single calendar month. For them every arrangement — sliding window, disjoint block, whole history — produces ability units holding exactly that user's attempts, so the arms differ in how often those rows are counted and never in content. Their rows carry no signal about window length and full variance, and they are 41% of attempt mass.

**Fixed** by reporting the criterion additionally over windowing-sensitive users only, defined from raw attempt times so the row set is identical across arms. On the current data that keeps 465 of 1,299 users and 67% of rows.

## Head-to-head against `pipeline.train_rating` — MEASURED 2026-08-12: NOT competitive yet

Both models trained excluding the SAME 20,315 problems (shared `holdout_set`, seed 0, frac 0.15, hash `6619fdc487345fc0`, confirmed by hash on both sides), and the joint model run at production's ability structure so the row sets match exactly — 81,439 held-out rows either way.

```
                              production    joint MIRT    verdict
held-out logloss (81,439)       0.4952        0.4992      production
discrimination vs constant     +0.0301       +0.0261      joint delivers 87%
survey rho (time-adjusted)      0.8903        0.8880      indistinguishable
```

**Held-out logloss: production wins, resolvably.** Paired user-clustered bootstrap over 1,249 users, 2,000 replicates: delta +3.98 mnat, 95% CI [+2.20, +5.63], P(joint better) = 0.000. Read it relatively — the whole signal available is production's 30 mnat edge over a best-constant predictor (and serving is difficulty-matched, so that range is already compressed), so a 4 mnat deficit is **13% of the model's entire edge**. The axes recover 0.121 mnat of it, about 3%.

**Survey rho: indistinguishable, and that is a statement about the criterion.** Paired user bootstrap: delta −0.0017, 95% CI [−0.0150, +0.0093]. With 189 users and 374 declarations this statistic resolves to roughly **±0.012**, so it cannot support comparisons at the third decimal at all. Every survey-rho difference weighed during this work — joint versus production (0.002), phase-2 slope scaling (0.009), subset versus full training (0.002) — sits inside that interval. When the two criteria disagree, the one with power decides; the other is absence of evidence, not evidence of parity.

Two structural traps this comparison had to clear, both of which would have produced a wrong table silently:

- **Row multiplicity.** `validate_ratings.heldout_metrics` builds its rows with `build_profiles()`, so every attempt is scored twice and the logloss averages over (attempt, profile) pairs. The whole-history joint arm averages over unique attempts — same problems, half the rows. Hence the production-window arm exists purely to make the comparison legitimate.
- **Which production.** The shipped `ratings/` trains on every problem, so it is not the like-for-like partner for a model trained on 85%; and it carries the phase-2 slope scaling, which the staged trainer does not produce. The comparison uses `train_rating_staged --production-flags` on the same subset.

Context, offered as context and not as mitigation: production is the endpoint of a long measured development history, while this model's first non-diverging run was the same day as this measurement. The 13% gap is now a target with a harness behind it (`compare_heldout_logloss.py`, `compare_survey_rho.py`, and `--dump-params`), so the next iteration is scored against this bar without re-deriving anything.

## Components to ablate or sweep

Ordered by expected value. Anything already measured on the two-stage trainer is marked, because those results transfer as a prior but not as a fact — the joint model differs enough that a reversal is possible, and the ortho penalty is the standing example of a knob whose sign flips between the two.

### Ability window length — MEASURED 2026-08-12: windowing costs, whole history wins

Result first. Three arms × two seeds, on the fixed model:

```
RATING criterion (held-out problems, axes off)     windowing-sensitive users only
                    seed 0   seed 1    mean          seed 0   seed 1    mean
production (2mo)    0.4996   0.4997   0.4997         0.5060   0.5062   0.5061
2-month disjoint    0.4990   0.4987   0.4989         0.5055   0.5052   0.5054
whole history       0.4985   0.4980   0.4983         0.5044   0.5047   0.5046
```

**Whole history wins at both seeds on both criteria — four of four — and the ordering is monotone in how much windowing there is.** The whole-history-versus-production gap averages 1.4 mnat/attempt while within-arm seed variation is 0.1–0.5 mnat, so the effect clears its noise by roughly 3–10×. The pipeline is fully deterministic (verified: a repeated arm reproduced every digit including the stopping epoch), so at a fixed seed there is no run-to-run component at all and the only uncertainty is initialisation.

So the simplest available arrangement — one ability per user, no windowing machinery, ability and axis-position granularity coinciding — is also the most accurate. It removes a constant, a helper, the overlap-versus-disjoint question and the mixed-granularity open question together.

Two limits on how far to read it. **This is the joint model, not the shipped trainer**: `pipeline.train_rating` uses the same two-month sliding windows and may be paying the same cost, but it differs in activation, learning rates, the axis block and phase 2, so transfer is a separate experiment — a worthwhile one, since it would be a free accuracy gain on a shipped model. And **the effect is small in context**: 1.4 mnat against the 31 mnat the four defect fixes were worth, so the honest ranking puts "the model was not converging" an order of magnitude above "the window is slightly too short".

The axes are unaffected either way — the shipped-baseline reference spans 0.950–0.954 pooled across all six runs, flat well inside its ~0.03 floor.

#### How this knob was originally justified

`build_profiles` currently assigns each attempt to two **overlapping two-month windows**. That length was chosen for the rating model to track ability drift, and it has never been weighed against what it costs: a shorter block means fewer attempts per ability unit, and ability is the parameter every other estimate is measured against. A longer block, or the full user history, trades drift-tracking for a much better determined ability.

Sweep **production (2-month sliding) / 2-month disjoint / whole history**, two seeds each. Three arms rather than five, for reasons that only became visible once the window helper was fixed:

- **Rows scale with window width** under correct sliding-window semantics, since each attempt is written into `W` monthly units. A 4-month arm costs more compute than both of the arms answering the actual question combined, and it existed only to reveal an interior optimum. Dropped.
- **The disjoint arm is not a width arm**, it is the control that separates width from the duplication overlapping introduces. Without it those two effects are confounded in every comparison, which is what the exact-2× row-count difference between the old arms was really showing.
- **Two seeds is the minimum**, because a single seed cannot separate an arm effect from initialisation — a lesson from the first attempt, where several single-seed readings had to be walked back.

Three things make this worth doing first:

- The whole-history arm is the *simplest possible* choice — one ability per user, no windowing machinery, and the profile-to-user mapping problem disappears entirely.
- Drift is an empirical claim, not an axiom. If a four-month block matches two months, drift is slower than the current window assumes and half the ability units are being estimated on noise for nothing.
- It interacts directly with the axis positions. Those are fitted per user over the whole history; if ability is *also* per user, the two granularities coincide and the mixed-granularity question below evaporates.

Read it on held-out **problem** logloss primarily — ability quality is what the rating criterion is sensitive to — and watch the axis gain for a second-order effect. **Read the windowing-sensitive line, not the pooled one** (defect 4): two-thirds of fitted users sit in a single calendar month and are identical under every arm by construction, so they contribute variance and no signal.

A calibration on what "worth doing first" is now worth. The two model-side defects were together worth 31 mnat/attempt, while the widest gap this sweep ever showed between window lengths was about 7 mnat — and that gap came from arms compared at their divergence points, so it is an upper bound on noise as much as on signal. Window length may well turn out not to matter at all; it was promoted to "highest-value untested knob" on an argument, never on a measurement, and it kept that billing while four defects sat underneath it.

### Structural simplifications, in decreasing order of what they remove

- **Global slope** (running) — one discrimination corpus-wide. Removes a head.
- **Global floor** — `c_p` per problem versus one corpus-wide floor. The floor's per-problem calibration is already known to be fraught: a fitted floor *scale* is an estimator artifact rather than signal, and the shipped floor is deliberately unscaled. If a single floor holds, that removes a second head and with it the floor-scale question entirely.
- **Fixed lapse** — `lambda` is currently learned; pinning it at the client's constant removes a global parameter and a coupling between the floor cap and the ceiling.
- **Trunk depth** — one hidden layer versus two. Capacity has lost every test so far; depth has not been tested at all.
- **Axis count K** — selected on the two-stage model; must be re-selected here rather than inherited.

### Regularisation and optimisation

- **Axis input dropout rate** — 0.3 is an interior optimum on the two-stage model (9/9 per-user); confirm the optimum has not moved.
- **Trunk dropout rate**, input and hidden — inherited from the rating trainer, never swept in this context.
- **Ortho weight**, and whether it must now act against `d_p` rather than a residual head.
- **`v` ridge and zero-mean penalty** — both inherited unchanged from the two-stage fit, where the ridge was scaled by attempt count; with overlapping profiles the counts double, so the effective strength has silently changed.
- **Epochs, learning rate, batch** — the two-stage fit needed a network rate an order of magnitude below the per-user field rate; the joint fit currently uses a single rate for everything, which is the configuration that diverged late on the other model.
- **Warm start** — measured decisive on the two-stage model; cheap to confirm here.

### Data inclusion

- **`MIN_ATTEMPTS`** (5 correct and 5 incorrect) and **`MIN_PROFILE`** (8) — both inherited. The second interacts with block length: a longer block admits users a short one excludes.
- **Repeat attempts** — excluded throughout. They carry retention information rather than difficulty information, which is why the recall model owns them, but the exclusion has not been tested against this objective.

### Already measured on the two-stage trainer

Transfer as priors, not as facts: axis map linear rather than trunk (−0.430 for a trunk), axis input dropout 0.3 (+0.164 per-user, interior optimum), residual-trunk input dropout harmful (−0.086), residual trunk halvable for free, mutual ortho penalty load-bearing and not replaceable by dropout, same-width SwiGLU harmful on that model while a narrower one gained.

## Open questions

**Dropout compression, and whether "single stage" survives it.** `train_rating` needs its phase-2 slope-scale fit because dropout compresses per-problem parameters toward the feature-conditional mean; the correction is 1.5–1.7×, measured, and it is *caused by* dropout rather than by errors-in-variables — with dropout removed the multiplier collapses to one and held-out logloss explodes. The joint model has no such correction and would ship compressed slopes.

The gauge fix (defect 1 above) is what makes the question askable at all — under the intercept gauge the quantity phase 2 corrects was unidentified, so the compression could not even be measured. That prerequisite is now met and **measuring the multiplier here against the shipped 1.5–1.7× is the next thing to do on the rating side.** Whether a clean-pass scalar counts as "a second stage" or as calibration is a judgement — it fits no structure, only one number, on a frozen model — but it cannot simply be dropped. One in-pass alternative exists and is untried: the model already runs an eval-mode, dropout-off forward every batch for the barriers, so a free global `log s` scoring `a·s` in an auxiliary no-dropout term could ride along in the same pass. The risk is that `s` and the free `theta` trade off against each other, which is exactly the confusion the gauge defect caused, so it should not be attempted before the gauge is pinned.

**User granularity.** Ability is fitted per user × 2-month profile because ability drifts; axis positions are fitted per user because a skill profile should not, and a 2-month block carries far too few attempts to estimate one. Mixing granularities inside one model is defensible but untested, and getting the profile-to-user mapping wrong silently fits axis positions per block — the profile id is `{user_id[:8]}_{YYYY-MM}`, so the user is the first eight characters and not a split on a separator.

**Overlapping profiles double-count attempts.** `build_profiles` assigns each attempt to two overlapping 2-month windows, which is by design for the ability field but means each attempt contributes twice to the axis-position gradient and to the ridge normalisation. Consistent across users, so unlikely to bias direction, but it is not neutral.

**The quarantine becomes a modelling problem.** Under staging, "difficulty cannot become a skill axis" is enforced by construction: ratings are fitted and frozen, so the axes only ever see leftovers. Fitted jointly, nothing prevents an axis absorbing difficulty except a penalty that has to be designed and defended — and `fit_population_constants` already refuses axes on exactly that ground. The orthogonality penalty against the residual head served this role in the two-stage model; with `r` gone it has to act against the per-problem difficulty term instead — `d_p` as currently written, `b_p` once the gauge is moved.

**Rotation is still unresolved and is what actually blocks shipping.** The axis subspace has no preferred basis, which is why two of the three shipped axes remain unnamed. Factor rotation — varimax or oblimin — is the standard remedy, is likelihood-invariant, and has not been tried. It applies equally to the two-stage and joint fits.

## Where the code is

- `auxiliary/rating_dev/train_joint_mirt.py` — the joint model and its evaluation. Knobs that matter when reproducing anything here: `--window-months` (2 = production exactly, 0 = one ability per user) with `--disjoint-blocks`, and `--lr-net` / `--lr-field`, which must stay an order of magnitude apart (defect 2).
- `auxiliary/rating_dev/train_axes_staged.py` — the two-stage axis trainer's experiment surface, whose defaults are verified bit-identical to `utils.rating.residual_axes.train_axes`. Production carries no knobs; this is where alternatives are tested.
- `auxiliary/rating_dev/compare_heldout_logloss.py` — paired held-out logloss between a joint dump and a production fit, user-CLUSTERED bootstrap. Clustering is by user rather than row because attempts within a user share an ability estimate; a row-level bootstrap understates the interval, in the direction that makes a difference look real.
- `auxiliary/rating_dev/compare_survey_rho.py` — paired survey-rho with a user bootstrap. Use it before quoting any survey-rho difference: the statistic resolves to about ±0.012 at the current declaration count, which is wider than most differences worth arguing about.
- `auxiliary/rating_dev/holdout_set.py` — the SHARED held-out generator both models must draw from, hash-confirmed. A locally drawn split is a different universe and its numbers compare to nothing.
- `auxiliary/rating_dev/eval_r_head.py`, `eval_axis_capacity.py`, `eval_axis_activation.py` — the arms behind the measurements quoted above.
