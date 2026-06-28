# Displayed-change contradiction simulator

After every answer the PWA shows the user one number on the result overlay
(`pwa/js/overlays.js`):

- **first-time / challenge** problem → a **rank change** (`Rank ±x.xxx`, plus an
  optional "rank up/down to <label>"), and
- **repeat / recall** problem → a **time-to-recall change** (`±N repeat delay`).

Both are *non-deterministic* in a way users notice: a problem you got **right** can
show your rank ticking **down**, or a recall you **missed** can show the repeat
delay getting **longer**. This harness measures how often that happens by replaying
**real** user histories through the **actual** client code.

It does **not** re-implement the model. `engine.mjs` is bundled (by esbuild) from the
real `pwa/js/model.js` (the two-head recall model + its warm-started Adam optimiser)
and `pwa/js/rank-decomposition.js` (the linear-information window 4PL MLE), with the
shipped `rating_to_rank.json` / `population_constants.json` inlined exactly as the
browser bundle does. Only the DOM-coupled edges are stubbed: `state.js` →
`state_shim.mjs`, `@sentry/browser` → `sentry_stub.mjs`, `math` (mathjs) loaded from
`pwa/lib/math.min.js` as a global. So the Adam trajectory, the window MLE and the
retention curve are the production ones, not a port.

## What it measures

For each attempt the runner mirrors `updateUserStats` (pwa/js/user-stats.js) step
for step: build the answer-history entry, refit the window-MLE rank, and — for a
repeat — run the **real retrain** (`calculateAndStoreStatsResults`) and read
`freshFailureReviewHours` **before vs after** (the genuine before/after the overlay
shows). The retrain runs after **every** attempt because the Adam state is carried
forward and the trajectory is path-dependent (this is the point the result hinges
on — see below).

A **contradiction** = the displayed change's sign opposes the answer:

| surface | shown for | expected | contradiction |
|---|---|---|---|
| rank | first-time | correct ⇒ ≥0, miss ⇒ ≤0 | correct→down / miss→up |
| recall delay | repeat | correct ⇒ later (+), miss ⇒ sooner (−) | correct→sooner / miss→later |

The recall delay saturates at the `MIN_STABILITY` floor (18 h) for weak users and
the `MAX_STABILITY` ceiling (365 d) for very durable items — there the overlay shows
`+0 s` and cannot contradict. The report therefore splits the repeat regime into
floor / middle / ceiling and reports the contradiction rate **among middle-regime
visible moves** as the honest denominator.

## Run

```bash
# 1. export real attempts + per-problem rating/slope/floor (reads attempts/database.sqlite + ratings/)
./.venv/bin/python3 -m auxiliary.displayed_change_sim.export_sim_data \
    --users <redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>,<redacted-user-id>
#    or auto-select:  --top 40 --min-total 300 --min-repeats 100

cd auxiliary/displayed_change_sim
node build.mjs                       # bundle the real client modules → engine.bundle.mjs
node runner.mjs --out results.json   # replay; prints the report, writes per-attempt recs
#   --users a,b        restrict to some users     --max-per-user N   cap attempts (quick checks)
#   --refine-steps N   extra one-at-a-time Adam steps to take when the displayed recall
#                      delay contradicts the answer (0 = shipped baseline). Drives the real
#                      model.js recallImpactWithRefine; run N=0 vs N=5 for an A/B.
```

## Recall-delay refine experiment

A mid-regime user's recall-delay overlay swings ±tens of minutes per attempt with a
sign largely decoupled from the answer (the 500-row window slides + one converged
Adam fit replaces another). `model.js recallImpactWithRefine(isCorrect, maxExtraSteps)`
(shared by `updateUserStats` and this sim) tests a fix: after the default retrain,
take **one extra Adam step at a time** (continuing the optimiser on the same window)
until the generic fresh-failure delay moves the way the answer implies, up to
`maxExtraSteps` (5). The runner's `--refine-steps` sets `maxExtraSteps`; the report's
`refine` line shows how often the loop fired and resolved.

Replaying is ~30 ms/attempt (dominated by the per-attempt retrain), so a heavy user
is several minutes; run it in the background for the full set.

## Files

| file | role |
|---|---|
| `export_sim_data.py` | DB + ratings → `sim_input.json` (one row per attempt; per-problem rating/slope/floor) |
| `build.mjs` | esbuild bundle of `engine.mjs` with the headless aliases → `engine.bundle.mjs` |
| `engine.mjs` | re-exports the real model / rank functions the runner drives |
| `state_shim.mjs`, `sentry_stub.mjs` | headless stand-ins aliased in by `build.mjs` |
| `runner.mjs` | replay + classify + report; writes `results.json` (aggregate + per-user + per-attempt recs) |

`engine.bundle.mjs`, `sim_input.json`, `results*.json` are generated (gitignored).

## Fidelity caveats

- Attempts to problems absent from the current `ratings/` are dropped (≈20 % — a
  regenerated corpus); the live client likewise can't rank/score an unrated problem.
- `review_time` in D1 is the cumulative-prior review total reported with each
  attempt, so it maps directly to the client's `accumulatedReview`; the per-encounter
  increment is back-filled onto the prior entry when the next encounter arrives
  (matching the live timeline).
- Working set capped at `ANSWER_HISTORY_WORKING_SET` (3000), matching the client's
  `getMergedAnswerHistory`; a repeat whose prior encounter aged out is treated as the
  client would (scan the working set only).
- `is_repeat` is taken from D1 (what the live scheduler actually decided); the harness
  does not re-run the recall scheduler.
