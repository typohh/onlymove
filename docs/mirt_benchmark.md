# MIRT benchmark: provenance and results

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

**The DB hash is part of the identity of every result here.** `holdout_set` draws from the problems that have attempts, so a `pipeline.fetch_database` run changes the universe, changes the draw, and changes the hash — which is the intended tripwire, not a nuisance. Both models print the hash they trained under; if the two differ, the comparison is void.

Reproduce:

```bash
./.venv/bin/python3 -m auxiliary.rating_dev.holdout_set --seed 0 --frac 0.15
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
- **`SHALLOW_W = 0` is refuted and the barrier is load-bearing.** Without it slopes collapse (median 0.94, 0.1th percentile 0.097) and derived ratings blow out to a range of [-225, +146], since rating = b + kappa(c)/a diverges as a falls. That is exactly the failure the barrier is written to stop.
- **The per-problem slope head earns its place** — one global slope costs 1.8 mnat. This REFUTES, in this model, the on-record claim that per-problem slope is largely assignment noise; that finding is from the two-stage production model.
- **`BURN_IN_EPOCHS` stays at 300.** 450 looked like an architecture-shifted optimum at 0.5 mnat, but that is 1.2 sd against the measured seed noise. The bracket shape holds (600/750 degrade the rating for real); the claim that 450 beats 300 does not.
- **The warm start is what makes coupling SAFE.** Under raw features, dropping it left the rating untouched and only cost the axes. Under the shared trunk it costs the rating 2.1 mnat here, because a badly-initialized axis map sends its gradients into the trunk the rating depends on — and takes 1360 epochs instead of 380. The two changes are complements, not independent knobs. **See the warm-start section below**: re-measured under the production protocol at patience 30, where the cold arm needs 2670-3200 epochs to stop improving, the rating gap does not survive — 1360 epochs was very likely still descending.

## AXIS VALUE BY USER HISTORY — the axes need ~500 attempts before they pay

Held-out-user walk binned by how many attempts PRECEDE each scored one, both
folds pooled (`auxiliary/rating_dev/axis_by_history.py`). This is the USER-side
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

Similarity is measured three ways on the problem side, over the problems two fits share, after per-axis z-scoring: **pairwise** correlation under an optimal 1-1 assignment (`scipy.optimize.linear_sum_assignment` on `-|corr|`, so a permutation costs nothing and a sign flip means swapping a motif's two tail labels, not losing it); **canonical correlations** between the two K-dim spans, which bound what any linear alignment can achieve; and **tail overlap**, the shared problems among each axis's top and bottom 500, restricted to the shared problem set so corpus growth cannot deflate it. Tail overlap is the one that speaks to naming, since a name is a claim about the extremes and the client only ever reads a flagged tail.

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

### The 16 motifs in the bank are gone

The names in `axes/labels.json` were fitted against the two-stage axes and do not transfer. Current MIRT axes against all 16 frozen anchors in `label_anchors.npz`, over 152,346 shared problems: best single-anchor correlation **0.41**, R² on the full 16-anchor span 0.16 / 0.21 / 0.28, canonical correlations against the three motifs the previous fit shipped **0.009 / 0.411 / 0.582**. `ax_0860ca04` "killing / defending territory", which sits in `components[0]`, shares 10, 8 and 3 tail problems of 1000 with the three new axes. `train_mirt` writes no `motif_ids` and `package_for_pwa` reads `components` positionally, so shipping today would attach that name to an axis correlating 0.10 with it.

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
| `BURN_IN_EPOCHS` | 300 | 100 / 300 / 500 / run-to-convergence |

### B. NEVER TESTED — structural

| choice | current | why it is open |
|---|---|---|
| axis path coupled vs detached | COUPLED (shipped) | measured indistinguishable; coupled chosen because detaching makes the trunk learn for the rating alone and hands the axes the result, which is feature reuse rather than a joint fit. |
| `--act relu` | swiglu | SwiGLU doubles trunk parameters at a given width, so width and activation are confounded until one is held fixed. |
| ability granularity | per-user (`--window-months 0`) | measured best on the rating side, but the axes are per-user while ability could be per-window; the mixed-granularity question is dodged rather than answered. |
| an ANISOTROPIC prior on the axis positions | isotropic (`V_RIDGE` scalar) | THE ONLY IN-OBJECTIVE FIX FOR AXIS IDENTITY, and untried. The rotation degeneracy survives precisely because the v prior is isotropic: rotate the loadings and the positions by the same orthogonal matrix and the likelihood, `V_RIDGE`, `V_MEAN_PENALTY` and `ORTHO_WEIGHT` are all unchanged. DISTINCT per-axis ridge weights break that symmetry inside the objective, so the basis becomes identified rather than gauge-fixed from outside — no carry-over, no frozen bank, no external anchor. Two cautions. It cannot fix the SUBSPACE being underdetermined (see the warm-start section: two fits at identical loss sharing min canonical 0.029), only which basis of a given subspace comes out. And a prior that distinguishes axes by scale imposes an ordering the data may not support, so the arm has to be scored on loss as well as stability. |

### C. NEVER SWEPT — constants

| constant | current | note |
|---|---|---|
| `V_MEAN_PENALTY` | 10.0 | strong. It is the only thing stopping an axis carrying population difficulty. |
| `ORTHO_WEIGHT` | 0.5 | inherited from `residual_axes`. Its mutual half is load-bearing THERE; never checked here. |
| `SHALLOW_TAU` | 0.1 | the barrier's threshold. Only its WEIGHT was swept (30 epochs, 0/0.3/1/3). |
| `SHALLOW_W` | 0.3 | CLOSED — `0` IS refuted, in this document's own sweep section: slopes collapse (median 0.94, 0.1th percentile 0.097) and ratings blow out to [-225, +146], a 22 mnat loss, because rating = b + kappa(c)/a diverges as a falls. What remains open is the VALUE, swept at 30 epochs on a configuration since changed. |
| `GAUGE_MOM` | 0.01 | EMA rate for the difficulty gauge. |
| `BATCH` | 65536 | |
| `STOP_ROW_FRAC` | 0.03 | size of the stop carveout. |
| `CHECK_EVERY` / `FLAT_PATIENCE` | 10 / 30 | MEASURED 2026-08-14 and raised from 5. The stop split improves in bursts separated by long flat stretches; the largest gap before the final optimum was 25 checks on one seed and 10 on the other, so anything under 25 halts mid-stretch. At 5 both seeds stopped near epoch 350 against optima at 850 and 580, costing 0.5 and 0.7 mnat of RATING (paired, same seed either side). Buys no axis stability. Runtime ~9 -> ~25 min; the experiment surface keeps 5. Two pairs only — a third seed would firm it up. |
| `PRIOR_AX_SD` (scorer) | 0.5 | CLOSED — MEASURED 2026-08-14 and it is AT the optimum, not below it as this row previously claimed. Swept post-hoc against a fixed pair of user-holdout dumps (seeds 0/1), axis adds in mnat: 0.25 -> 1.73, 0.35 -> 2.00, **0.5 -> 2.14**, 0.7 -> 2.09, 1.0 -> 1.52, 1.5 -> 0.22. 0.7 ties within seed sd. The rating column is identical to four decimals at every value, which is the check that the constant touches only the axis path. STRONGLY ASYMMETRIC — too loose costs far more than too tight (1.9 mnat at 1.5 against 0.4 at 0.25), because a loose prior lets the online axis estimate overshoot on short prefixes; err low if it is ever revisited. |
| `DROPOUT`, `INPUT_DROPOUT` | 0.3 | inherited unswept in this model; measured only on the two-stage one. |

### D. Open questions that are not knobs

- **The axis block overfits faster than the rating heads** — measured on the stopping diagnostic, the axes swing +0.8 to −4.1 mnat between epochs 350 and 650 while the rating moves 1.9. A separate stopping signal for the axis path has never been tried.
- **An LR warmup schedule is REFUTED at two ramp lengths — MEASURED 2026-08-14.** `--lr-warmup-epochs`, linear ramp from zero, two seeds per arm. The case for it was endpoint consistency: the axis objective is rotation-invariant, so nothing pulls two runs together and the endpoint is decided entirely by the path — a less perturbation-sensitive path should land two runs closer. Measured min canonical correlation between the two seeds: **0.248 at ramp 100, 0.409 with no ramp, 0.591 at ramp 300** (the ramp must cover the burn-in to matter at all, since the warm start is read at epoch 300). That looks like a trend and is not one: the NO-RAMP baseline's own spread across three seed pairs is 0.409 / 0.596 / 0.659, and all three arms sit inside it. One seed pair per arm cannot separate the effect from the draw. What IS consistent is the cost — ramp 300 is worse on both seeds (0.4848 vs 0.4834 and 0.4864 vs 0.4845, ~1.5 mnat) and converges later (540/730 against 360/330). Resolving it properly needs ~4 seeds per arm, 12 fits, for a hypothesis that has now cost loss twice. The mechanism argues against it too: warmup damps OPTIMIZER-noise sensitivity, but the 2% time cut re-rolled the basis at the SAME seed and init, so the sensitivity is to the DATA, through the warm start's eigendecomposition — whose trailing eigenvalues sit within 1.4x of each other (0.00048 vs 0.00034 in the trained `cov(v)`), and eigenvectors of a near-degenerate matrix swing under any perturbation of the matrix. That is closed-form arithmetic, not a gradient path.
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

K is fixed at 3, and measured as an interior optimum rather than inherited — see the K section above. There is no incremental search: K is baked into the shared head, so each value is a full training run.

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
