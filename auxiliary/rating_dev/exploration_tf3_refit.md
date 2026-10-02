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
