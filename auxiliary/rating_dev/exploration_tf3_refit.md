# Exploration log: the transformer refit and the low end of the rating fit

A working record of the ideas tried while moving the rating fit onto the transformer network, kept so nothing is re-tried by accident and open threads are not lost. It is a development note, not documentation: figures are single fits on windtunnel unless a seed count is given, and nothing here is quotable. Candidate fits live on windtunnel under `ablations/dev_runs/prod_candidates/<name>/`.

**What we are judging on:** the low end — inflated floors, problems with excessively low ratings, and whether the rank map passes. The high end (very high ratings) is not a concern.

**The problem that started it:** the first transformer fit had median floor 0.379, max floor 0.908, 116 floors above 0.8, 143 problems rated below −5, and the rank map aborted (an anchor below the lowest published knot). The problems with the highest floors had sane values in the deployed fit, so the suspicion is convergence, not the network.

## Status at a glance

| # | Idea | Status |
|---|---|---|
| 1 | Final trunk state instead of all layers | Adopted |
| 2 | Transformer network (tf3-b10c512), 512 channels unrotated, no PCA/encoder | Adopted |
| 3 | Global-vector inputs (trunk mean/max, value head hidden) | Extracted, never tested |
| 4 | Fewer bot temperatures as inputs | Panel built, not run |
| 5 | All bot temperatures from one benchmark run | Done |
| 6 | Production fit holds no rows aside | Adopted |
| 7 | Drop the remainder batch after the shuffle | Adopted (its hypothesis was not confirmed) |
| 8 | Half learning rate vs full | Open — no consistent difference in single fits |
| 9 | Drop the cosine schedule | Rejected at half rate |
| 10 | Snapshot / weight averaging in place of cosine | Discussed, not built |
| 11 | Synthetic random player | Parked — does not fit a model with a flat floor |
| 12 | Synthetic strong player | Parked with the random one |
| 13 | Idiocy: a second slope below the floor | Open — "own steepness" form is the live one |
| 14 | Soft corners so gradient reaches the idiocy rating | Built (idiocy side only) |
| 15 | Floor cap at 0.5 | Rejected |
| 16 | Rating read off the full curve | Computed offline once; not in the trainer |
| 17 | Minimum rating where the second slope meets the floor | Running (`min_full_s0`) |
| 18 | Hinge floor; ignoring players below chance | Raised, not tried |

## 1–3. What the trunk reads

- **Final trunk state.** A block's reported output is its increment, not the trunk state, so "last layer" has to mean `trunkfinal`. Last-layer full channels matched the all-layers encoding closely enough that the all-layers store was dropped. Increment-only arms were pointless and dropped.
- **Transformer network.** `kata1-tf3-b10c512-s3203M-d5937M` had the lowest loss of the candidates and is now `DEFAULT_MODEL` for everything; the b18 residual net is a bot only. Activations are its 512 trunk channels, standardized and unrotated. Existing SGFs are not re-annotated.
- **Global vector.** Trunk mean, trunk max and the value head's hidden layer are extracted (`auxiliary/rating_dev/extract_last_block.py`, `_signal_cache_global*`) but not wired into any arm. A separate test, still owed.

## 4–5. Bots

- **Temperature inputs.** Layouts `activations+bot:one|two|residual` exist and `ablations/panels/bot_temperatures.json` is built (all / one / two / residual × 4 seeds). Only `all_s0` has run. Deprioritized as a nice-to-have.
- **One run for all temperatures.** `benchmark_bots` now treats a temperature family as one bot. Done.

## 6–7. Rows and batches

- **No rows held aside in production** (`STOP_ROW_FRAC = 0`): with a fixed budget a split decides nothing. Fits on all rows lost the low rating tail (`baseline_all_rows`: no floors above 0.8, lowest rating −4.4).
- **Drop the remainder batch.** Chosen over equal batch sizes. The hypothesis that a small last batch was inflating floors was **not** confirmed: `droplast_w0` still had 112 floors above 0.8. Kept anyway.

## 8–10. Learning rate and schedule

| Fit | Median floor | Floors above 0.8 | Lowest rating | Notes |
|---|---|---|---|---|
| `baseline_all_rows` (full rate, cosine, 7,000) | 0.277 | 0 | −4.4 | clean |
| `lr_half` (half rate, cosine) | 0.219 | 0 | −3.6 | clean, rank map passes |
| `lr_half_holdout` (same, 3% held aside) | 0.160 | 0 | −3.6 | clean, calibration ≈ 2.3 ranks |
| `lr_half_const_holdout` (half rate, constant) | 0.000 | 3 | −9.7 | floors collapse, one rating +101 |

- **Half vs full.** Without a second slope both are clean. With idiocy the direction flips between pairs (held-at-−100: half worse; own-steepness: half much better). Three-seed capped fits differed between seeds as much as between rates. No detectable rate effect from single fits; full rate is used for now because it is faster.
- **Half-rate budget.** 10,000 epochs under cosine. The held-out score was best (≈ 0.4922) at about 5,000 constant-rate epochs of work, which is what a 10,000-epoch cosine delivers.
- **Constant rate.** At half rate the held-out score wanders (0.493–0.507) and the end state is bad. Cosine stays.
- **Weight averaging.** Averaging snapshots as a replacement for the decay was discussed; no switch was built.
- An opening ramp is already measured-rejected (see `CLAUDE.md`) and was not re-tried.

## 11–12. Synthetic players (parked)

Both are ordinary users in the fit: no pinned ability, same dropout, axes included, abilities written to `user_theta.csv` as `synthetic:random` / `synthetic:strong`. Weights default to 0.

- **Random player** answers at one over the number of legal moves. Even at weight 0.01 it collapses the median floor to about 0.02 (`droplast_w001`: 0.025, with 325 floors above 0.8; `droplast_w003`: 0.016). Its ability fitted to −10 / −33. Conclusion: a pure-chance player is a plausible first-timer but cannot be described by a curve with a flat floor. This is what led to idiocy.
- **Strong player** answers at `(1 − lapse) · q`, where `q` is the softmax (temperature 1) of logit win rate over the evaluated first moves. It is finite-ability by design: it does poorly on genuinely uncertain problems, so it presses down on runaway hard ratings. Ability fitted to about +29. Earlier results with pinned ability are void.

## 13–15. Idiocy: a second slope below the floor

The floor is not flat: below it a second slope runs, in logit space, down to chance (one over legal moves) at one learned global **idiocy rating**. It replaces the random player.

**Form A — "floor onset" (dropped).** The second slope starts where the main slope meets the floor. The two slopes can stand in for each other, and the fit used that:

| Fit | Idiocy rating | Median floor | Floors above 0.8 | Problems below −5 | Rank map |
|---|---|---|---|---|---|
| `idiocy_full_s0` (uncapped) | −2.45 | 0.712 | 58,257 | 41,534 | error 3.97 |
| `capped_full_s0` (floor cap 0.5) | −0.06 | 0.500 | 0 | 0 | aborts |
| `late_full_s0` (start −100, held 1,000 epochs) | −56.9 | 0.180 | 29 | 208 | passes, 2.18 |
| `late_half_s0` | −56.0 | 0.220 | 185 | 520 | not run |

- The 0.5 floor cap put 91% of floors at the cap and pushed the median slope to 5.09. Rejected.
- Held at −100 the idiocy rating never came back into the player range (4 of 2,998 users below it), so the second slope did nothing.

**Form B — "own steepness" (live).** The second slope has one fitted global steepness and starts wherever the line drawn up from chance at the idiocy rating reaches the problem's floor. The main slope does not enter. Started at −10, not held (`idiocy_form="own_slope"`, `idiocy_init=-10`, `idiocy_free_epoch=0`).

| Fit | Idiocy rating | Steepness | Median floor | Floors above 0.8 | Problems below −5 | Lowest rating | Rank map |
|---|---|---|---|---|---|---|---|
| `own_full_s0` | −1.90 | 4.08 | 0.335 | 146 | 226 | −24.6 | aborts (rank −32.3) |
| `own_half_s0` | not yet read | not yet read | 0.335 | 0 | 7 | −6.5 | not yet run |

What `own_full_s0` does, which looks healthy:

- 39 of 2,998 users sit below the idiocy rating. Abilities are on a sane scale (1st / 50th / 99th percentile −7.6 / 0.63 / 14.1).
- The second slope starts between −1.47 and −0.60 across problems. It reaches all problems for a 1st-percentile player, 133,949 for a 5th-percentile player, 14,960 for a 10th-percentile player, none for the median player.
- 4,286 of 407,612 attempt rows (1.05%) are on the second slope, covering 3,790 problems and 270 users; 46 rows are at pure chance.

What is questionable: 5,674 problems are rated below the idiocy rating. They are the high-floor, shallow-slope problems (median floor 0.55, slope 0.86), and include all 146 floors above 0.8.

**Soft corners.** The ramp's two corners are rounded (softplus, `IDIOCY_SHARPNESS = 10`) so a player below the idiocy rating still passes gradient to it. The lapse side needed nothing: the main curve is a sigmoid that approaches `1 − lapse` without reaching it, so it has no plateau or corner and already passes a smooth, fading gradient. A constant small slope on either side was considered and not built — it never stops pushing, so abilities at the extremes drift for as long as training runs, and it needs a slope constant.

## 16–17. Fixing the ratings below the idiocy rating

- **Full-curve rating (offline).** The published rating is the 80% point of the main curve alone. Recomputed on `own_full_s0` as the 80% point of the full curve, for the 220,286 problems with floor under 0.79: none below the idiocy rating (was 5,502), lowest −1.71 (was −12.18), none below −5 (was 92); the median problem does not move. Cost: the client computes probability from rating, slope and floor only, so it would need the second slope too.
- **Minimum rating (running).** Inside the fit, a problem's difficulty is held up so its rating cannot fall below where the second slope meets its floor (`rating_min=True`, own-steepness form only). A floor at or above 0.8 has no 80% point, so such a problem's curve goes flat at its floor above the second slope. `min_full_s0` is the first fit.

## 18. Raised, not tried

- A hinge in place of an absolute floor.
- Ignoring players below a chance threshold instead of modelling them.

## Still owed after a configuration is settled

Production refit on windtunnel, about-tier ablations, `pwa/about.html` in four languages, docs for the new trainer behaviour, then (after deploy) full ablations and the methodology rewrite. Also deferred: re-tuning `bots/analysis.cfg` for the transformer and moving the analysis server off the b18 net.

## Addendum (2026-10-02, later the same day): the two-sigmoid form, constraints, barriers

This supersedes the status of items 13–18 above. The "floor onset" form and the opening hold are removed from the code. The rank map no longer aborts on a low anchor; its published knots extend down to cover it.

**Board verdicts from the user (EGF 4-dan), which change what counts as a fault in a second-slope fit:**

- Floors above 0.8 are legitimate on problems with one obvious move and no reasonable alternative (checked on `3a3434b2`, `04aba1a0`, `0f98b195`, `0191c252`, `3f72771e`). With a second slope the floor no longer has to cover the weakest players, so it can be high.
- Floors near chance are never justified. Even on bland positions with several good candidates, many moves are obviously wrong; expect about 0.1–0.2, not under 0.05.
- Still faults: a floor near 1 (the main slope has no height), and a rating read as a sentinel because the main slope never crosses 80%.

**Forms now in the code** (`idiocy_form`):

- `own_slope`: a straight line in logit space up from chance at the idiocy rating, corners rounded.
- `sigmoid`: `P = chance + (c − chance)·σ(s(θ − I)) + (1 − c − λ)·σ(a(θ − b) + axes)`. Random + idiocy sigmoid + rating sigmoid + lapse sum to 1; the floor is the split. Three per problem (b, a, c), three global (λ, I, s).

**Switches added:**

- `ability_min`: abilities approach, and never reach, the second slope's bottom (the idiocy rating on the line; `I − 2/s` in the sigmoid form).
- `rating_min`: on the line, the rating stays above where the second slope meets the floor; in the sigmoid form, `b − 2/a ≥ I + 2/s` (the slopes do not overlap). `2/steepness` is where the tangent at a sigmoid's centre meets its levels.
- `floor_barrier_w`: the floor is placed between chance and `1 − λ`, with the penalty `−log(1 − chance/floor) − log(1 − floor)`; neither term reads the lapse. An earlier symmetric version (log barrier on the position in the range) squashed all floors to the middle.

**All-rows fits (full rate, seed 0):**

| Fit | Form and switches | Idiocy rating | Steepness | Median floor | Floors > 0.8 | Lowest rating | Training loss | Rank map error |
|---|---|---|---|---|---|---|---|---|
| `own_full_s0` | line | −1.90 | 4.08 | 0.335 | 146 | −24.6 | 0.4450 | 2.25 |
| `min_full_s0` | line, rating min | 0.04 | 9.13 | 0.076 | 0 | 0.04 | 0.4455 | 2.24 |
| `amin_full_s0` | line, hard ability cut | −2.61 | 3.06 | 0.352 | 0 | −6.03 | 0.4463 | 2.23 |
| `asoft_full_s0` | line, smooth ability min | −2.10 | 2.71 | 0.262 | 0 | −8.77 | 0.4455 | 2.30 |
| `sig_full_s0` | sigmoid | 0.44 | 11.75 | 0.223 | 0 | −13.3 | 0.4458 | not rerun |
| `amin_sig_full_s0` | sigmoid, hard cut at centre | −1.10 | 1.55 | 0.196 | 3,541 | −119.9 | 0.4450 | dead |
| `two_sig_s0` | sigmoid, both conditions | −4.81 | 0.51 | 0.123 | 556 | −5.31 | 0.4481 | 2.30 |

About 40 users are pinned at the lowest allowed ability in every fit with an ability minimum. Whether they are the same users, and what their attempts look like, is unchecked.

**Fits with 3% of rows held aside** (sigmoid form with both conditions unless stated):

| Fit | Variant | Training | Held-out | Idiocy rating | Steepness | Floors, 1st–99th pct |
|---|---|---|---|---|---|---|
| `ho_none` | no second slope, seed 0 | 0.4429 | 0.4926 | – | – | 0.13–0.30 |
| `ho_sig_free` | sigmoid, no conditions, seed 0 | 0.4434 | 0.4928 | not read | not read | not read |
| `ho_line_amin` | line, ability min, seed 0 | 0.4419 | 0.4945 | not read | not read | not read |
| `ho_two_sig` | seed 0 | 0.4476 | 0.4866 | −47.2 | 9.26 | 0.00–0.85 |
| `ho_fb001` | symmetric barrier 0.01 | 0.4464 | 0.4924 | −72.5 | 10.9 | 0.33–0.61 |
| `ho_fb01` | symmetric barrier 0.1 | 0.4491 | 0.4908 | −72.2 | 12.1 | 0.43–0.53 |
| `ho_fb0001` | symmetric barrier 0.001 | 0.4477 | 0.4919 | −71.6 | 12.7 | 0.18–0.57 |
| `ho_fb00001` | symmetric barrier 0.0001 | 0.4463 | 0.4927 | −54.8 | 9.18 | 0.15–0.46 |
| `ho_fd001` | chance-and-1 barrier 0.01 | 0.4453 | 0.5016 | −112.6 | 15.8 | 0.08–0.22 |
| `ho_fd01` | chance-and-1 barrier 0.1 | 0.4423 | 0.5020 | −74.3 | 15.0 | 0.08–0.19 |
| `ho_none_s1` | no second slope, seed 1 | 0.4427 | 0.4882 | – | – | 0.18–0.33 |
| `ho_two_sig_s1` | seed 1 | 0.4446 | 0.4912 | −88.8 | 16.2 | 0.16–0.28 |

Readings:

- **In every held-out fit with both conditions on, the idiocy rating fled far below the weakest user** (who sits near −22 to −30), so the second slope was switched off and the fit was functionally the plain model. Candidate cause, not separated: both conditions are bounds above the idiocy rating, so lowering it loosens them for everyone at once.
- **`ho_two_sig`'s 0.4866 is not evidence for the conditions.** It is the plain model in a different solution (floors split between near-zero and high, lapse 0.043). At seed 1 the same form is 3 millinats worse than the plain model, so the sign does not repeat.
- **The chance-and-1 barrier is 9 millinats worse on held-out rows** and compresses floors to 0.08–0.22: the `−log(1 − floor)` term has no scale of its own and pulls every floor down.
- Seeds hold aside different rows, so held-out levels compare only within a seed.

## Addendum (2026-10-03): the second slope removed; hunting the floor–slope instability

**Decision (user, after the floor analysis): the second slope (idiocy) is removed.** Attempts that sit on the floor are 0.1–0.2% of the log, and among them neither absolute ability nor ability relative to the problem predicts success (logistic coefficients −0.007 ± 0.016 and −0.005 ± 0.018 per rating unit in `baseline_all_rows`; the weakest fifth of floor-row users still succeeds 25% against a 2% chance level). However the data are served, there is nothing for a second slope to be fitted from. The idiocy code, ability/rating minimums and floor barrier are deleted from the trainer.

**Standing criteria (user):** cross-seed stability of the fitted shape (slope × difficulty spread, floor distribution, lapse, parameter agreement between seeds) outranks a few millinats of held-out loss. Tuned constants may probe an interaction but are not acceptable as the fix, because the data will shift (few users above 4d today).

**All fits below:** full rate, cosine, 7,000 epochs, 3% of rows held aside; a seed fixes the held-out rows, so held-out losses compare only within a seed. "Plain" = shipped settings; its seven seeds score 0.4926, 0.4882, 0.4983, 0.4935, 0.4884, 0.4846, 0.4950.

### The core finding: a floor–slope trade-off the data cannot see

- Rank agreement between seeds (plain, 5 seeds): ratings 0.97, slopes 0.69, floors 0.42 (one pair −0.09). Attempts per problem barely help (floors 0.36 at 0 attempts, 0.52 at 5–19).
- Not a data-volume problem: grouping problems into classes on their own activation inputs, even classes of ~11,000 problems move their average floor by ~0.40 (logit) between seeds, against a 0.15–0.20 spread between classes. The overall floor level moves too: median floor 0.21 / 0.26 / 0.24 / 0.14 / 0.31 across seeds 0–4.
- Predictions on the actual attempts agree well across seeds (logit correlation 0.90, median difference 2.4 points) although parameters do not: the disagreement lies in directions the served data cannot see.
- Weight decay on the network moves the instability instead of removing it: floors agree better, slopes worse (0.0001: floors 0.82, slopes 0.38; 0.0003: floors 0.71, slopes 0.40; plain: 0.53 / 0.71).
- Candidate mechanism (user): in `P = c + (1 − c − λ)·σ(a(θ − b))` the midpoint steepness is `(1 − c − λ)·a/4`, so floor and slope can stand in for each other. Under test as `slope_height`.

### Experiments and verdicts

| Experiment | Switch | Result | Verdict |
|---|---|---|---|
| Asymmetric main slope (σ(z)^p, one global p) | `link_power` | paired held-out +3.8 / +0.3 / −3.2; p fitted 1.45 / 1.39 / 1.04 | no effect |
| Straight ramp with rounded bevels | `link="ramp_soft"` | +9.6 / −2.7 / +0.7; 12–58% of floors near chance | rejected |
| Hockey stick (hard ramp) | `link="ramp_hard"` | −0.9 / −8.1 / +0.4; 16–29% of floors near chance | rejected (floors implausible, gain not established) |
| Lapse held at 0.02 / 0.04 / 0.06 | `lapse_fixed` | 0.04: −4.7 / −8.3 / −3.4 (p ≈ 0.06), but 2 of 3 seeds collapse into split floors and slope × spread 0.60 | probe only: the free lapse settles below the best-predicting value |
| Robust (IQR) difficulty gauge | `gauge="iqr"` | 5 seeds: held-out −1.4 mean (spread 3.0); seed 4 a different solution (median floor 0.52) | not a stabilizer |
| Empirical-Bayes floor prior (fitted mean and spread) | `floor_prior` | fitted spread collapses to ~5e-5: one floor for all problems (0.45–0.50); held-out +2.6 mean; lapse steady 0.023–0.039 | rejected (self-defeating), but shows one global floor costs only ~2.6 mnat |
| Floor in the slope's units | `floor_rel` | +2.8; 40% of floors near chance | rejected |
| Network weight decay 0.0001 / 0.0003 / 0.001 (with dropout) | `weight_decay` | moves instability floor → slope; 0.0003+ flattens floors to a constant | probe: confirms the trade-off |
| Snapshot average, cosine / constant rate | `weight_avg="swa"` | averaged held-out worse than live; floors agree 0.35 | rejected |
| Exponential average, cosine / constant rate | `weight_avg="ema"` | under cosine identical to final weights; constant rate: floors agree 0.28 | rejected |
| No dropout, weight decay 0.0001 / 0.001 | `dropout=0` | memorizes: train 0.26–0.28, held-out 0.68–0.77 | far too weak; stronger sweep running |

A bug was found and fixed along the way: re-measuring the difficulty gauge in eval mode after weight averaging rescaled difficulty under abilities and slopes fitted against the training-mode scale; the averaged buffers are now kept.

### Running / queued

- `slope_height` (floor-independent midpoint steepness), seeds 0–2 — the most promising structural candidate.
- No dropout with weight decay 0.01 / 0.03 / 0.1 / 0.3.
- Muon on the trunk matrices (`optimizer="muon"`), seeds 0–1.
- Dropout 0.4 / 0.5 and a doubled batch, seeds 0–2.

## Addendum (2026-10-04): the floor moves off the trunk — adopted

**Adopted (user): the floor is a linear map of the bot win-rate columns and the board flag** (`FLOOR_MODE = "bot"`), no longer a row of the trunk's head. The floor's source is now part of the ablation regime (`panel_regime`), and old records read as `"trunk"`.

Five paired seeds against the plain fit (trunk floor):

| Seed | Plain held-out | Bot-floor held-out | Difference |
|---|---|---|---|
| 0 | 0.4926 | 0.4918 | −0.8 |
| 1 | 0.4882 | 0.4837 | −4.5 |
| 2 | 0.4983 | 0.4956 | −2.7 |
| 3 | 0.4935 | 0.4921 | −1.4 |
| 4 | 0.4884 | 0.4871 | −1.3 |

Better on all five, mean −2.1 millinats. Agreement between seeds: floors 0.996 (plain 0.53), slopes 0.82 (plain 0.71), ratings 0.97 (unchanged). Lapse 0.004–0.026. High floors (above 0.6 in every seed: 11,526 problems) include the one-obvious-move positions the user judged legitimate (`3a3434b2`, `04aba1a0`).

**Known fault, open:** 5–10% of floors sit within 0.05 of chance, reproducibly (12,068 in all three of the first seeds; their attempted ones are solved 70.9% of the time). Some are positions the user judged should not be near chance (`0f58c3a7`, `107189c5`).

### Other results in this round

| Experiment | Result | Verdict |
|---|---|---|
| Floor-independent midpoint slope (`slope_height`) | held-out +9.8 / −5.3 / +7.3; floors collapse to chance (70k–112k); floor agreement −0.19 | rejected |
| Floor as a lower bound, sigmoid (`link="max_floor"`), 3 seeds | held-out +11.7 / +3.6 / +6.6; floors tidy (no extremes) but agreement 0.46; lapse → 0 | rejected |
| Curve clamped between floor and ceiling, sigmoid (`link="clamp"`), 3 seeds | +11.0 / −2.8 / +7.5; floor agreement 0.49; lapse → 0 | rejected |
| Bound / clamp with the hard ramp, seed 0 | −2.0 / +4.8; collapsed solution (lapse at cap, difficulties squeezed) | rejected |
| One global floor (`floor_mode="global"`) | +7.7 / +1.6 / +8.4 | rejected: per-problem floors carry real value |
| Floor linear over every input (`floor_mode="linear"`), 3 seeds | −1.4 / −8.0 / −2.0 vs plain; floors agree 0.993 but run higher (median 0.41–0.49, 2,200–3,900 above 0.8); slopes agree 0.76 | comparable to the bot floor; the bot floor kept (simpler, slopes steadier) |
| No dropout, coupled weight decay 0.01 / 0.03 | unstable mid-run; held-out 0.534 / 0.528; floors constant | rejected |
| No dropout, decoupled weight decay (AdamW) 0.1 / 1 | 0.1 memorizes (held-out 0.741); 1 breaks down mid-run (0.645) | rejected: dropout stays |
| Dropout 0.5 (trunk floor), 3 seeds | collapsed solutions on all three (slope × spread 0.11–0.66) | rejected |
| Doubled batch (trunk floor), 3 seeds | +7.1 / −0.7 / +7.6; floor agreement 0.43, slopes 0.54 | rejected |
| Muon on the trunk, step 0.02, 2 seeds | training stalls by epoch 1,000–2,000 (train 0.508); outputs constant | step size far too large; 0.002 / 0.005 queued |

### Queued under the new baseline

- Muon at step 0.002 and 0.005 (judged on convergence speed and final held-out loss, per the user's expectation).
- Slope from a linear map too (`slope_mode` = `bot` / `linear`), 3 seeds each.
- Bot-floor seeds 5–6.

### Update (2026-10-04, later)

- **Slope from a linear map**, 3 seeds on the bot-floor baseline: `slope_mode="linear"` held-out −1.8 / −0.8 / +0.5, slopes agree 0.997, slope × spread 1.27–1.42, lapse 0.009–0.019; `slope_mode="bot"` +0.4 / +0.6 / +4.3, slopes agree 1.000, lapse 0.004–0.006. Both raise near-chance floors to 28k–34k and squeeze the difficulty IQR (0.51–0.67 against 1.06–1.26). Promising on stability; blocked by the near-chance floors. Seeds 3–6 of the linear slope queued, plus floor and slope both linear over every input.
- **Bot-floor seeds 5–6** against plain: +3.7 and −3.1. Over seven seeds the held-out gain is −1.4 mean (spread 2.6), not significant; the case for the bot floor rests on stability (floors 0.996), not loss.
- **Muon**, step 0.002 / 0.005, 7,000 epochs: ahead early (−4.4 at epoch 1,000), then overfits past ~3,000 and ends +4.3 / +4.0. At a 3,000-epoch budget, Muon against Adam: +1.4 / −0.6 (no difference). Both 3,000-epoch fits are within ~1–2 millinats of the 7,000-epoch Adam fit, so a shorter budget may be nearly free — worth a proper budget panel later. Muon itself: no gain here.
- **Candidate-count floor bound**: SGF first-move variations enumerate nearly every legal move (median 63 against 63 legal cells), so one over their count is no useful lower bound for the floor.
- **Linear slope, seven seeds** against the bot-floor baseline: held-out −1.8 / −0.8 / +0.5 / +3.0 / +0.3 / −4.0 / −1.1 (mean −0.6, no effect); floors and slopes agree 0.998 between seeds. Near-chance floors 28k–37k against the baseline's 12k–26k.
- **Floor and slope both linear over every input**, seeds 0–1: −0.5 / −0.9; floors and slopes agree 0.998; near-chance floors 25k–30k and ~2,000 above 0.8.
- **Training-budget panel queued** on the baseline: 3,000 / 5,000 / 7,000 epochs, seeds 0–2 (the 3,000-epoch seeds 0–1 reuse `ho_adam3k`).
- **What the bot floor follows**: its logit correlates 0.92–0.93 with the pass rate of the human-imitation bot at kyu strengths. On problems where the floor sits near chance, those bots find the answer ~10% of the time (70% elsewhere), so the signal is real, but the floors (0.02–0.06) overshoot it. "Within 0.05 of chance" also misclassifies few-move problems (`0f58c3a7`, 8 legal moves, floor 0.124).
- **Floor from the kyu-strength human-imitation bots only** (`floor_mode="weak_human"`, 31 of 64 columns), 3 seeds: held-out +4.6 / −1.7 / +3.8 against the bot floor (mean +2.2); floors agree 0.999, slopes 0.84; near-chance floors 6.9k–9.5k (bot floor 12k–23k), 1st-percentile floor 0.036–0.041 (bot floor 0.016–0.028). Halves the fault at a small loss cost; seeds 3–4 and the combination with the linear slope queued.
- **Training budget** on the bot-floor baseline: 3,000 / 5,000 / 7,000 epochs held-out 0.4937 / 0.4946 / 0.4918 (seed 0), 0.4841 / 0.4898 / 0.4837 (seed 1), 0.5010 / 0.5011 / 0.4956 (seed 2). 7,000 is best on all three; the budget stays.
- **Fully linear model** (`hidden=[]`, floor from the shared head, no exception), 3 seeds: underfits (training 0.483–0.484 against 0.443) and held-out is 8–27 millinats worse than the bot-floor baseline (0.4997 / 0.5015 / 0.5223). Floors do NOT stabilize: agreement −0.13 (one pair −0.70), floors split between near-chance (31k–89k) and above 0.8 (up to 6.2k). Linearity alone is not what steadies the floor: a linear floor beside a NONLINEAR difficulty and slope reproduces (0.993–0.999), a linear floor beside a linear difficulty and slope over the same inputs does not. Candidate reading: the floor is identified only when it cannot share a representation with difficulty and slope.
- **One hidden layer of 64, floor from the shared head** (`hidden=[64]`, `floor_mode="trunk"`), 3 seeds: floors agree 0.92 (shipped two-layer trunk with the same head floor: 0.53), slopes 0.80, ratings 0.98 — a smaller shared trunk alone steadies the floor a lot. But held-out is +6.4 / +16.2 / +11.6 against the bot-floor baseline, lapse goes to 0, the bulk of difficulties is squeezed (slope × spread 0.81–1.04) and floors go to both extremes (18k–43k near chance, up to 3.5k above 0.8). Not usable as is; lighter trunks with the linear floor are queued.
- **Clamped hockey stick, trunk floor, seed 1** (user asked for more seeds of the original): overall shape repeats (lapse 0.015 / 0.012, slope × spread 1.32 / 1.22, floor median 0.41 / 0.40, almost no extremes), held-out +4.8 / −1.4 against plain. Per problem it agrees LESS than plain on the same seeds: floors 0.19 (plain 0.36), slopes 0.70 (0.81), ratings 0.91 (0.97), predicted success 0.79 (0.90). Seed 2 running.
- **Clamped hockey stick, trunk floor, seed 2**: held-out +4.7 against plain (three seeds: +4.8 / −1.4 / +4.7, mean +2.7). Over three seeds the overall shape repeats (lapse 0.012–0.021, slope × spread 1.22–1.32, floor median 0.40–0.42), but per problem it is no steadier than plain: floors 0.41 (plain 0.53), slopes 0.76 (0.71), ratings 0.93 (0.97), predicted success 0.83 (0.90). Seed 2 also has 560 floors above 0.8 where seeds 0–1 had almost none. Stopped at three seeds.

**Baseline note (user):** the adopted floor is the LINEAR map over every input (`FLOOR_MODE = "linear"`), not the bot-only one; comparisons from here are against the linear-floor fits (`ho_flin_*`: 0.4913 / 0.4803 / 0.4963 at seeds 0–2).

- **Floor with its own nonlinear trunk** (`floor_mode="own_trunk"`), 3 seeds: memorizes (training 0.400 against held-out 0.49–0.51); held-out +6.4 / +11.4 / +9.3 against the linear floor; floors swing high (median 0.61–0.64, 18–20% above 0.8, 7% near chance); seeds 0–1 agree on floors 0.85 but on ratings only 0.88. Separating the floor's parameters is not enough — its capacity is the problem. Rejected.
- **Hockey sticks with the linear floor**, seeds 0–1 (paired against the linear floor):

  | | Clamped ramp (`clamp_ramp`) | Floor-bound ramp (`max_ramp`) | Linear floor, sigmoid |
  |---|---|---|---|
  | Held-out difference | +7.2 / +5.3 | +4.7 / +8.6 | — |
  | Floor / slope / rating agreement (seeds 0–1) | 0.98 / 0.92 / 0.98 | 0.93 / 0.94 / 0.97 | 0.99 / 0.76 / 0.97 |
  | Lapse | 0.014 / 0.017 | 0.012 / 0.010 | 0.007 / 0.037 |
  | Floors near chance | 1.2k / 2.5k | 0.7k / 1.4k | 10.0k / 10.1k |
  | Floors above 0.8 | 3.1k / 3.7k | 3.4k / 3.3k | 3.9k / 3.0k |

  With the linear floor the hockey sticks are consistent: slopes agree far better than with the sigmoid (0.92–0.94 against 0.76), the lapse is steady, and near-chance floors fall by 4–13×. The cost is 5–9 millinats of held-out loss. Consistent, so seed 2 of both is queued.
- **Hockey sticks with the linear floor, three seeds** (paired against the linear floor): clamped ramp +7.2 / +5.3 / +6.1 (mean +6.2), floor-bound ramp +4.7 / +8.6 / +4.5 (mean +5.9). Agreement across the three seeds: clamped floors 0.95 / slopes 0.92 / ratings 0.97, floor-bound 0.95 / 0.94 / 0.97, against the sigmoid's 0.99 / 0.76 / 0.97. Predicted success on attempts agrees better too (logit correlation 0.92 against 0.90; median difference 1.2–1.3 points against 2.6). Lapse 0.014–0.017 and 0.010–0.014 (sigmoid 0.007–0.048). Near-chance floors 1.2k–2.5k (sigmoid 5.9k–10.1k). The most stable arrangement found so far, at ~6 millinats. Adopting it would change the client's curve (the ramp and the max/min form define slope and the 80% rating differently). Next: combined with the one-layer trunk, which was the best-scoring trunk at seed 0.
- **Per-problem linear lapse** (`lapse_mode="linear"`), seed 0: +6.6 against the linear floor, no stability or floor benefit; the client could not use it anyway. Dropped after one seed.
- **One hidden layer of 64 + linear floor**, seed 0: 0.4891 (−2.2), training 0.471 (less memorization), lapse 0.040, near-chance floors 4.2k (baseline 10k). Seed 1 running.

## Addendum: clamped ramp on a one-layer trunk, linear slope seed 2, odds clamp seed 0

- **Clamped ramp + one-layer (64) trunk + linear floor** (`ho_clmpr64_s0/s1`): held-out 0.4971 / 0.4894 against the clamped ramp on the full trunk's 0.4985 / 0.4856 (−1.4 / +3.8, seeds split), and +5.9 / +9.1 against the linear floor. Less stable than the full trunk under the same link: floors agree 0.90 (vs 0.95), slopes 0.74 (vs 0.92), ratings 0.96. Above-0.8 floors 4.4–6.1k. Arm ended.
- **Linear slope, seed 2** (`ho_lin2_s2`): 0.4962 vs linear floor 0.4963. Three seeds: 0.0 / +2.5 / −0.1 against the linear floor, and the shape reproduces: floors 0.99, slopes 0.998, ratings 0.97. The cost is near-chance floors at 25–32k (vs ~10k with the linear floor), so linear slope pushes the floor-slope trade-off onto the floor side.
- **Odds clamp seed 0** (`ho_oddsfl_s0`): 0.4879, −3.4 against the linear floor (−10.6 against the clamped ramp). But the floors polarize: 11.7k near-chance and 11.2k above 0.8 (linear floor: 10k / 3.9k), with lapse 0.043.
- **Odds clamp seed 1** (`ho_oddsfl_s1`): 0.4812, +0.9 against the linear floor (seed 0 was −3.4). The two seeds agree on shape, and more closely than the linear floor does: floors 0.996, slopes 0.86 (linear floor 0.76), ratings 0.98 (0.97). Lapse 0.043 / 0.044 (linear floor ranged 0.007–0.048 across seeds). The polarized floors reproduce: near-chance 11.7k / 14.6k, above 0.8 11.2k / 11.3k. Note that `stab.py` recovers difficulty with the sigmoid's 80% formula, so its slope × IQR column is approximate under this link. Seed 2 queued.
- **Linear floor seeds 3–4** (`ho_flin_s3/s4`): 0.4914 / 0.4896, −2.1 / +1.2 against the plain trunk floor at the same seeds. Over five seeds the floors agree 0.994 (min 0.989), slopes 0.77 (min 0.70), ratings 0.97. What wanders is the lapse and the slope scale: lapse 0.002–0.048, slope × difficulty IQR 1.5–2.9, and the two move together (low lapse ↔ low slope scale). That spread is the floor–slope–lapse trade-off left over once the floor is pinned; the odds clamp's two seeds sat at lapse 0.043 / 0.044.
- **Linear floor seeds 5–6**: 0.4813 / 0.4918, −3.3 / −3.2 against the plain trunk floor. Seven paired seeds: −1.3, −7.9, −2.0, −2.1, +1.2, −3.3, −3.2 (mean −2.7, paired t −2.6, p ≈ 0.04). Seven-seed agreement: floors 0.994 (min 0.989), slopes 0.76 (min 0.67), ratings 0.97. Lapse 0.002–0.048 across seeds.
- **Two-layer light trunk 64→32, linear floor** (`ho_t6432fl_s0/s1`): 0.5139 / 0.4969, +22.6 / +16.7 against the linear floor, lapse collapses to 0, slopes agree 0.69. Arm ended.
- **Odds clamp seed 2**: 0.4955, −0.8 against the linear floor. Three seeds: −3.4 / +0.9 / −0.8 (parity). Agreement: floors 0.994, slopes 0.80, ratings 0.98; lapse 0.043 / 0.044 / 0.039. The double-ended floors reproduce again (near-chance 14.3k, above 0.8 12.1k). Next: the odds clamp with the floor back on the trunk (`ho_oddstr_s0/s1`), to see whether the odds link alone is enough to stop the trunk floor wandering.
- **Odds clamp with the floor on the trunk** (`ho_oddstr_s0/s1`): 0.4996 / 0.4832, +7.0 / −5.0 against the plain trunk floor (+8.3 / +2.9 against the linear floor). Floors agree 0.34 across seeds; the odds link does not stop the trunk floor wandering, so the linear floor stays. Slopes agree 0.81 and lapse falls to 0.000 / 0.006. Arm ended. Odds clamp seeds 3–4 with the linear floor queued to firm up its stability reading.
- **Odds clamp seed 3**: 0.4919, +0.5 against the linear floor. Four seeds: −3.4 / +0.9 / −0.8 / +0.5 (parity). With four seeds the slope agreement drops to 0.78 (min 0.75), level with the linear floor's 0.76; the earlier 0.86 was two seeds. What stays distinct is the lapse (0.035–0.044 over four seeds, against 0.002–0.048) and the double-ended floors (near-chance 11.8k, above 0.8 10.1k).
- **Odds clamp + linear slope** (`ho_oddslin2_s0/s1`): 0.4872 / 0.4794, −4.1 / −0.9 against the linear floor, −0.7 / −1.8 against the odds clamp. Most reproducible arm yet: floors 0.997, slopes 0.999, ratings 0.981, lapse 0.037 / 0.037. Unlike the linear slope under the sigmoid, the rating spread does not halve (sd 2.2, as under the odds clamp). But the slope is still a stand-in for difficulty (slope–rating correlation −0.61, slope–floor +0.59; odds clamp with trunk slope −0.34 / +0.32) and near-chance floors rise to 19–22k beside 11k above 0.8. Seed 2 running; odds clamp with the weak-human floor queued (that floor halved near-chance floors under the sigmoid).
- **Odds clamp seed 4**: 0.4868, −2.7 against the linear floor. Five seeds: −3.4 / +0.9 / −0.8 / +0.5 / −2.7 (mean −1.1). Agreement over five: floors 0.993, slopes 0.77, ratings 0.98; lapse 0.035–0.044; near-chance floors 12–19k, above 0.8 10–12k.
- **Odds clamp + linear slope seed 2**: 0.4956, −0.7. Three seeds −4.1 / −0.9 / −0.7 (mean −1.9); floors 0.997, slopes 0.998, ratings 0.978. Lapse 0.037 / 0.037 / 0.021, so not pinned. Near-chance 18k, above 0.8 9k. Seeds 3–4 queued.
- **Odds clamp + weak-human floor** (`ho_oddswh_s0/s1`): 0.4999 / 0.4820, +8.6 / +1.7 against the linear floor (+3.6 / 0.0 against the weak-human floor under the sigmoid). Near-chance floors 14k, not halved as under the sigmoid; lapse 0.000 / 0.022. Arm ended.
- **Odds clamp + linear slope seeds 3–4**: 0.4904 / 0.4868, −1.0 / −2.7. Five seeds: −4.1 / −0.9 / −0.7 / −1.0 / −2.7 against the linear floor (mean −1.9, paired t ≈ −2.9, p ≈ 0.04). Five-seed agreement: floors 0.997, slopes 0.998, ratings 0.979; lapse 0.021–0.038. Near-chance floors 17–23k, above 0.8 9–10k. Seeds 5–6 queued.
- **Odds clamp seeds 5–6**: 0.4830 / 0.4900, +1.7 / −1.8. Seven seeds against the linear floor: −3.4 / +0.9 / −0.8 / +0.5 / −2.7 / +1.7 / −1.8 (mean −0.8, parity). The stability reading did not hold: lapse 0.009 / 0.016 on these two (0.009–0.044 over seven), and slope agreement over seven seeds is 0.57 (min 0.13: seeds 5–6 land on a different slope ordering). Floors 0.992, ratings 0.98. Arm ended; the odds clamp stays only in combination with the linear slope. Linear-slope seeds 3–4 under the sigmoid queued, to compare the two links at matched seeds.
- **Odds clamp + linear slope, seven seeds**: seeds 5–6 at 0.4800 / 0.4897, −1.3 / −2.1. Seven seeds against the linear floor: −4.1 / −0.9 / −0.7 / −1.0 / −2.7 / −1.3 / −2.1 (mean −1.8, 7 of 7 negative, paired t ≈ −3.9, p ≈ 0.01). Seven-seed agreement: floors 0.997, slopes 0.998, ratings 0.979; lapse 0.021–0.040. Near-chance floors 17–23k, above 0.8 9–12k.
- **Linear slope under the sigmoid, seeds 3–4**: 0.4949 / 0.4882, +3.5 / −1.4. Five seeds against the linear floor: 0.0 / +2.5 / −0.1 / +3.5 / −1.4 (mean +0.9). At matched seeds 0–4 the odds link beats the sigmoid under the linear slope on every seed: −4.1 / −3.4 / −0.6 / −4.5 / −1.4 (mean −2.8).
- **Correction to the rating-spread readings above.** The written rating is the sigmoid's 80% point under every link, and a problem whose floor is above 0.8 has no 80% point, so its rating comes out of a clamp far below everyone (1st percentile among them −6 to −21). Those problems inflated the rating sd, and the odds arms have 16–17k of them. Excluding floors above 0.78 (seed 0): rating sd 1.23 linear floor, 0.92 linear slope, 1.30 odds clamp, 1.46 odds + linear slope; so the linear slope narrows the spread by about a quarter, not half. The slope–rating correlation on the same subset is −0.21 linear floor, −0.75 under either linear-slope arm, −0.49 odds clamp. An odds-link candidate needs its own 80% point written (closed form in the link's comment) and a decision on what a problem with no 80% point ships.
- Queued: odds clamp + bot-only slope (`ho_oddsbot2_s0/s1`).

## Addendum: the linear floor parks users below the scale

Checked while weighing a proposal to anchor the floor on the worst-gap 0.1% of attempts. Under the linear floor that tail is not "weak users on hard problems": it is 1–7 users at ability −31 to −36 (median user +0.4), and it is dominated by ONE user with 7,813 attempts (≈2% of all rows), 13×13 only, 71% success. That user's success has an ordinary difficulty gradient (0.95 on the easiest eighth of their problems down to 0.40 on the hardest), and the plain trunk floor and the sigmoid linear slope both place them near −0.7. The linear-floor fits put them at −33 / −36 / (seed 2 likewise), the odds clamp at −26, and explain every one of their answers through the floor: floors average 0.64 on their 4,000 problems, against 0.41–0.43 corpus-wide.

| Fit | Users below −10 | Their attempts | Largest |
|---|---|---|---|
| Plain trunk floor s0 / s1 | 37 / 41 | 250 / 704 | 46 / 359 |
| Linear floor s0 / s1 / s2 | 132 / 129 / 117 | ~10,100 each | 7,813 |
| Linear slope (sigmoid) s0 | 26 | 415 | 53 |
| Odds clamp s0 | 78 | 10,267 | 7,813 |
| Odds clamp + linear slope s0 | 67 | 10,376 | 7,813 |
| Clamped ramp + linear floor s0 | 0 | 0 | — |

The linear floor reproduces this parking across seeds, so part of its 0.994 floor agreement may be several seeds landing on the same degenerate solution. Candidate route, not separated: a floor that is a linear map of features correlated with difficulty can itself carry a difficulty gradient, so a user can ride the floor, and below the curve the sigmoid passes almost no gradient back to the user's ability. Over more seeds (heavy user's ability; users below −10):

| Fit | Heavy user | Users below −10 |
|---|---|---|
| Plain trunk floor s0 / s1 | −0.7 / −0.9 | 37 / 41 |
| Clamped ramp, trunk floor s0 / s1 | 0.0 / 0.0 | 2 / 1 |
| Linear slope (sigmoid, linear floor) s0 | −0.8 | 26 |
| Clamped ramp + linear floor s0 / s1 / s2 | −6.8 / −8.5 / −11.3 | 0 / 6 / 1 |
| Linear floor s0–s6 | −24 to −36 | 117–158 |
| Odds clamp + linear slope s0 / s1 / s2 | −25.8 / −24.3 / −4.7 | 67 / 62 / 35 |

So parking goes with the linear floor beside a free (trunk) slope; the clamped ramp shrinks it to that one user at −7 to −11, and the linear slope under the sigmoid removes it. Bot-only slope arms dropped by decision (never to be used). Probe queued: the linear floor with that user excluded (`exclude_users`, `ho_flinx_s0/s1`; not paired, since excluding rows changes the held-out draw).

## Addendum: held-out rows on problems with no training rows

The row split leaves some held-out rows on problems none of whose rows were trained on (mostly single-attempt problems): 1,419–1,496 of the ~12,100 held-out rows per seed. Reconstructed after the fact (`/tmp/cold.py` on windtunnel: same filters, same `default_rng(seed)` draw; held-out counts match the logs exactly and the all-held-out loss matches within 0.2 millinats, the axis term left out). Those problems are still seen by the synthetic random user, so "cold" means no human rows. Deltas against the linear floor at the same seed, millinats:

| Arm | Cold s0–s4 | Cold mean | Warm s0–s4 | Warm mean |
|---|---|---|---|---|
| Plain trunk floor | +1.9 / −1.1 / −1.4 / −4.4 / +1.7 | −0.7 | +1.4 / +9.6 / +2.7 / +2.9 / −1.7 | +3.0 |
| Linear slope (sigmoid) | +3.0 / +1.6 / −0.7 / +2.7 / +2.2 | +1.8 | −0.4 / +2.7 / −0.1 / +3.7 / −2.0 | +0.8 |
| Odds clamp + linear slope | −8.2 / −2.6 / +6.2 / −5.7 / +6.1 | −0.8 | −3.6 / −0.6 / −1.7 / −0.4 / −4.1 | −2.1 |
| Clamped ramp + linear floor (s0–s2) | −2.0 / +3.9 / +8.9 | +3.6 | +8.2 / +5.8 / +5.0 | +6.3 |

About 1,400 rows per seed, so the cold column is noisy (the odds arm swings ±8). No arm separates from the linear floor on cold problems; the linear slope is slightly worse on 4 of 5 seeds. The trunk floor's warm-row cost against the linear floor does not carry over to cold problems.
- **Linear floor with the 7,813-attempt user excluded** (`ho_flinx_s0/s1`, not paired): the parking is structural, not that one user. About 100 users still sit below −10 (≈2,000 attempts, largest 118–121; plain trunk floor 37–41 users, 250–700 attempts). Floors drop (median 0.41–0.45 → 0.35–0.38) and near-chance floors rise from ~10k to 17.6k / 21.3k, so the heavy user had been holding floors up. Floor agreement unchanged at 0.996; slopes 0.82; lapse 0.013 / 0.000 still wanders.
- **Linear slope (sigmoid) seeds 5–6**: 0.4828 / 0.4939, +1.5 / +2.1 against the linear floor. Seven seeds: 0.0 / +2.5 / −0.1 / +3.5 / −1.4 / +1.5 / +2.1 (mean +1.2). Seven-seed agreement: floors 0.994, slopes 0.998, ratings 0.968; lapse 0.003–0.012; near-chance floors 23–30k. No parking (23–24 users below −10, ≤422 attempts). Odds + linear slope beats it at every matched seed (s5 / s6: −2.8 / −4.2).
- **Sharpened odds clamp, k = 4** (`odds_sharpness=4`, linear floor, `ho_oddsk4_s0/s1`): 0.5072 / 0.4804, +15.9 / +0.2 against the linear floor. The seeds split: lapse 0.000 vs 0.043, near-chance floors 2.5k vs 11.0k, slopes agree 0.47 (floors 0.985, ratings 0.98). Above-0.8 floors 4.9k / 6.4k (k = 1: 11k). Parking narrows to 18–19 users below −10, but the heavy user is still parked (−12.7 / −20.3). Arm ended (two disagreeing seeds).
- Queued: the global-to-local curriculum (`blend_frac`, linear floor): w from 0 to 1 over the whole run (`ho_blend_s0/s1`), then over the first half (`ho_blend05_s0/s1`).
- **Global-to-local curriculum over the whole run** (`blend_frac=1`, linear floor, `ho_blend_s0/s1`): 0.4879 / 0.4820, −3.4 / +1.7 against the linear floor. Floors 0.991, slopes 0.73, ratings 0.975; lapse 0.018 / 0.044. It does not prevent parking: 173–184 users below −10, heavy user at −28 / −20, so the parked arrangement forms while the floor and slope are still mostly global. Half-run ramp running.
- **Clamped ramp + linear slope + linear floor** (`ho_clmprlin2_s0/s1`): 0.4930 / 0.4833, +1.7 / +3.0 against the linear floor (+1.7 / +0.5 against the sigmoid linear slope). Floors 0.965, slopes 0.999, ratings 0.971; lapse 0.037 / 0.031. It removes both faults seen so far: near-chance floors 1.8k / 1.6k (sigmoid linear slope 23–30k, linear floor 5–11k) and NO parking (no user below −10; heavy user at −5.3 / −3.4). Above-0.8 floors 3.6k / 4.8k. Unseen-problem rows: −10.0 / +2.8 against the linear floor. Still the slope stands in for difficulty (slope–rating −0.82 / −0.80, floor–slope +0.70 / +0.76 on floors ≤ 0.78; linear floor −0.21 / +0.14). Seeds 2–3 queued.

## Addendum: experiments move back to the trunk model

By decision, new changes are tested on the trunk model (floor and slope both from the trunk); the linear floor and the linear slope are fallbacks only. The k = 4 and blend results above ran on the linear floor (trunk slope) and are being redone on the trunk: `ho_k4tr`, `ho_blend1tr` (whole run), `ho_blendtr` (first half), plus the ability prior (`theta_prior`) alone (`ho_tptr`) and with k = 4 (`ho_tpk4tr`). Parked users are reported split by volume from here on: light (< 30 attempts, plausibly just uncertain abilities) and heavy (≥ 30, structural parking). Under the plain trunk floor the users below −10 are light (median 4 attempts; at most one with ≥ 100); k = 4 on the linear floor parked fewer but heavier users (18–19, median ~45, the same ones on both seeds).
- **Ability prior + k = 4, trunk model, seed 0** (`ho_tpk4tr_s0`): the learned-spread prior COLLAPSES. Every user's ability ends at 1.44 (sd 0.00); the spread went to zero. Held-out is only +8.0 against the plain trunk floor (0.5006 vs 0.4926), so the user side must be carried by the axis positions instead. Lapse 0.050, near-chance floors 19.5k. Parking trivially gone (no one is anywhere). This is the joint-MAP degeneracy of a learned variance: at the optimum the spread equals the abilities' own spread, so once other parameters can explain users the abilities and the spread shrink together. Seed 1 and the prior alone (`ho_tptr`) are running to confirm whether k = 4 matters.
- Seed 1 of the learned-spread prior collapsed the same way (ability sd 0.0); the prior is removed. Replaced by a squared penalty on the population's ability moments, holding them to the difficulty's own gauge (`theta_moments`: weight × (mean² + variance), pulling both toward zero against the likelihood with no target spread asserted; a first version holding sd to 1 was withdrawn before it finished, since nothing says 1 is right). mean² + variance is the population's second moment, so this is the same ridge on every user; weight 0.01 (probe) gives each user prior information worth roughly ten attempts, negligible for heavy users and noticeable for sparse ones.: a far outlier inflates the spread quadratically, so a parked user is held near the edge where the fit can still pull them back, instead of drifting to −30 where no later change recovers them. Queued on the trunk model: with k = 4 (`ho_tm2k4tr`) and alone (`ho_tm2tr`), then the trunk redos of k = 4 and both blends.
- **Moments penalty 0.01, trunk model** (`ho_tm2k4tr_s0/s1` with k = 4, `ho_tm2tr_s0` alone): parking is gone (no user below −5, heavy user at −0.1 / −0.1 / −0.2), but the whole scale collapses. Ability sd 0.15–0.19 (IQR 0.20–0.22, plain 1.08–1.14), and the difficulty bulk shrinks with it (problem IQR 0.06–0.17, plain 1.42–1.44): the sd gauge on difficulty is met by a few far problems while the bulk contracts, which is the failure the `iqr` gauge option was written for. Held-out is BETTER, −4.5 / −8.5 / −5.4 against the plain trunk floor (0.4881 / 0.4797 / 0.4872), candidate route the shrinkage of sparse users' abilities (not separated). The shape does not reproduce: floors 0.66, slopes 0.77, ratings 0.87 across the k = 4 pair; near-chance floors 6k–46k. Next: the same penalty under the interquartile gauge (`gauge="iqr"`), with and without k = 4, so the bulk cannot contract.
- **Moments penalty alone, seed 1** (`ho_tm2tr_s1`): 0.4752, −13.0 against the plain trunk floor (seed 0 −5.4). Same collapse (ability sd 0.18, problem IQR 0.07, near-chance floors 50k). Across the pair the trunk floor agrees 0.92 (plain trunk 0.36), slopes 0.89, ratings 0.96 — the first trunk-floor arm whose floors reproduce, but on a collapsed scale.
- **k = 4 on the trunk model, seed 0** (`ho_k4tr_s0`): 0.4971, +4.5. Little parking: 6 light users below −10, 6 heavy below −5, heavy user at −0.3. Floors stay off chance (1st percentile 0.23, 2 near-chance). Lapse 0.001. Seed 1 queued.
- **Moments penalty + interquartile gauge + k = 4, seed 0** (`ho_tm2iqrk4_s0`): 0.4866, −6.0. The interquartile gauge holds the difficulty bulk (problem IQR 1.88) and the abilities contract less (IQR 0.68, sd 0.47). No user below −5; heavy user at −2.9. But floors run high: median 0.61, 8.9k above 0.8, 1st percentile 0.31. Seed 1 running.
