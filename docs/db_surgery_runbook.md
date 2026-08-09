# DB surgery runbook (freeze → operate → unfreeze)

How to run a destructive change against the production D1 without the table moving underneath you. Written for `auxiliary/purge_tsumegos.mjs`, but the freeze applies to any surgery.

## Why a freeze is needed

The surgery is computed against a **snapshot**. `pipeline.fetch_database` pulls the table locally, the script plans the change there, verifies it, and emits SQL to apply remotely. The digest fix-ups in that SQL are stated *relative to the values the snapshot held* — `user_seq = user_seq - 3` and so on. A row inserted between snapshot and apply is shifted against a chain it was never part of, and the corruption is silent: the row still has non-NULL values, they are just wrong, and the affected user then fails every digest check forever.

Reads are unaffected, so only writes are frozen.

## Why freezing is safe

`sendReport` (`pwa/js/user-stats.js`) only acknowledges an attempt when `res.ok`. A 503 leaves the attempt un-acked in `answerHistory`, and the durable outbox re-uploads it later through `/report-backfill` **with its original event time** — the same path that already handles a phone being offline. So a frozen write is a delayed write, not a lost one, and the client needs no maintenance mode, no banner, and no coordination. The freeze deliberately reuses the offline path instead of adding a parallel one.

The cost of the window is that attempts land late: they reach D1 on the next outbox flush (load / tab-visible / `online`, 30-minute throttle). Keep the window short anyway, because a long one means a large backfill burst on lift.

## The switch

`MAINTENANCE_MODE` in `pwa/wrangler.toml` (`[vars]`). `"1"` makes every POST route return `503 {error: "maintenance"}` with `Retry-After`, and skips the cron. It is checked in the `fetch` dispatcher **before the rate limiter and before any handler touches D1** (`maintenanceFrozen` in `pwa/src/guards.js`), so a frozen request costs one Worker invocation and zero D1.

Coverage:

- **Frozen** — every entry in `WRITE_ROUTES`: `/report`, `/report-backfill`, `/sync`, `/schedule`, `/rank-survey`, `/share-event`, `/push/*`, `/register`, `/invite/*`, `/profile/hide`. `/sync` is included even though it is a read, so no client can observe a half-applied surgery and reconcile against it.
- **Frozen** — the `scheduled()` cron, which writes recall notification stamps and leaderboard buffers. Skipping ticks is harmless: reminders are re-evaluated the next hour and the weekly jobs are idempotent.
- **Not frozen** — GETs, so the app keeps serving problems. The one writer left is the lazy `server_stats` recompute behind `GET /server-stats`; it only rewrites its own cache row, never a table under surgery, and the cron reconciles it afterwards.

## Procedure

Steps 2–7 are the window. Do them back to back.

1. **Prepare and rehearse.** With `MAINTENANCE_MODE = "0"` still live, take a snapshot and rehearse on a throwaway copy until the run is clean:
   ```bash
   ./.venv/bin/python3 -m pipeline.fetch_database
   cp db/d1.sqlite /tmp/rehearsal.sqlite      # rm -f /tmp/rehearsal.sqlite* first
   node auxiliary/purge_tsumegos.mjs --db /tmp/rehearsal.sqlite --ids <ids.txt> --apply
   ```
   The rehearsal must end in `post-verification PASSED`.

2. **Freeze.** Set `MAINTENANCE_MODE = "1"` in `pwa/wrangler.toml`, then:
   ```bash
   cd pwa && npm run deploy
   ```
   Confirm it took: `node auxiliary/check_live_status.mjs --expect-frozen`. It retries, which a bare `curl` does not — a deploy propagates over a few seconds, and until it does the previous version still answers, so a single probe can report the old state either way.

3. **Let in-flight requests drain.** A few seconds. Requests already inside a handler are not interrupted by the deploy.

4. **Take the authoritative snapshot — after the freeze, not before.**
   ```bash
   cp db/d1.sqlite db/d1.backup-pre-surgery.sqlite   # rm -f the -wal/-shm first
   ./.venv/bin/python3 -m pipeline.fetch_database
   ```
   This is the snapshot the emitted SQL is valid against. A fetch taken *before* the freeze is not, because the gap is exactly the hazard. Keep the backup: it is the exact pre-state.

5. **Plan, verify, emit — against the canonical local DB, not a copy.**
   ```bash
   node auxiliary/purge_tsumegos.mjs --db db/d1.sqlite --ids <ids.txt> \
        [--rows <row-ids.txt>] [--purge-broken-digest] --apply \
        --emit-sql surgery/purge.sql --emit-corpus-map surgery/purge_map.json
   ```
   Stop unless it prints `post-verification PASSED`; the script fails closed and emits no SQL otherwise.

   **Put every deletion in ONE run.** `--ids` removes whole problems, `--rows` removes individual `tsumego_reports` rows whose problem survives (an attempt whose `size` names a different board than the problem it reports, say). Both feed the same row set, so one run plans one digest repair. Splitting them into two surgeries means two *relative* repairs, and the second is stated against a snapshot the first has already moved — the same hazard the freeze exists to prevent, reintroduced by hand. Operating on `db/d1.sqlite` itself is deliberate — see *the local mirror cannot re-sync itself* below — and step 4's backup is the undo.

6. **Apply remotely.**
   ```bash
   npx wrangler d1 execute tsumego-reports --remote --file=surgery/purge.sql
   ```
   The file must run **exactly once**: the digest updates are relative, so applying it twice double-shifts every chain. If it fails partway, do not re-run it — restore the local DB from the backup, re-fetch with `--full`, and regenerate from the new state.

7. **Verify production against the local mirror.** Both sides should now hold the identical table, so one aggregate settles it:
   ```bash
   npx wrangler d1 execute tsumego-reports --remote --command \
     "SELECT COUNT(*) AS rows, SUM(user_seq) AS seq_sum, SUM(running_sum) AS sum_sum FROM tsumego_reports;"
   sqlite3 db/d1.sqlite \
     "SELECT COUNT(*), SUM(user_seq), SUM(running_sum) FROM tsumego_reports;"
   ```
   All three numbers must match. Do **not** verify with `pipeline.fetch_database` — it cannot see this change (below).

8. **Refresh the leaderboard — still frozen.** This is the right place for it, not after the lift: `update_leaderboard_ranks.mjs` rebuilds each row's `buffer` from a snapshot and writes it back, and any attempt landing between its read and its write is silently truncated out of that user's buffer permanently. Under the freeze nothing can land, so the race is not merely unlikely, it is impossible.
   ```bash
   node auxiliary/update_leaderboard_ranks.mjs --remote
   ```
   Its other ordering rule — run it only after the deploy carrying the new ratings, or the next `/report` recomputes on the old scale and overwrites the work — is satisfied twice over: the freeze deploy already shipped them, and no `/report` can run anyway. It reads the local `db/d1.sqlite` and the local `pwa/tsumegos/`, both of which are now the intended post-state, and it supersedes the purge's own leaderboard edits (which exist so the purge is correct when run standalone).

9. **Unfreeze and release.** Set `MAINTENANCE_MODE = "0"`, bump `CURRENT_STATE_EPOCH` if this release includes one, `npm run deploy`. Confirm the lift with `node auxiliary/check_live_status.mjs --expect-unfrozen` — `npm run deploy` already runs the same checks without the freeze assertion. Writes reopening is the one thing worth confirming rather than assuming: a lift that silently did not take leaves every client's outbox queuing behind a 503.

10. **After the window** — corpus and pipeline artifacts, if the ids are leaving the corpus for good and this was not already done as part of a larger release:
    ```bash
    ./.venv/bin/python3 -m auxiliary.corpus_surgery --map surgery/purge_map.json --apply
    ```

### The local mirror cannot re-sync itself after a purge

`pipeline.fetch_database`'s incremental path is built on `tsumego_reports` being **append-only** — it pulls `WHERE id > MAX(local id)` and never re-reads an existing row (`_fetch_incremental`). A purge breaks that premise in both directions:

- **deleted rows stay** in the local DB forever — the cursor is already past their ids, so they are never re-examined;
- **updated rows stay stale** — the digest repair rewrites `user_seq`/`running_sum` on a few hundred thousand *existing* rows, which an incremental fetch never re-reads.

Both are silent. So after any surgery, the local mirror is only correct if you either apply the same change locally (what step 5 does — deterministic, free, and proven byte-identical to the emitted SQL) or pay for a full re-export:

```bash
./.venv/bin/python3 -m pipeline.fetch_database --full
```

This is also why step 7 verifies with a direct remote query rather than a fetch, and why step 8 can trust the local DB it reads.

## When the surgery is part of a bigger release

The common case: an epoch bump, a re-rate, a corpus removal and the DB purge all ship together. They interact, and the ordering below is what keeps the freeze at minutes rather than hours.

**Keep the heavy work outside the window.** Corpus surgery, the retrain chain and packaging need no freeze, and — importantly — they do **not** need the DB purge to have run first. Once `corpus_surgery` has taken the SGFs and per-id artifacts, the removed ids have no features and no rating, so the trainer never joins their attempts; leaving those rows in D1 for another day changes nothing about the fit.

**Never ship the epoch bump while frozen.** The epoch wipe routes init through the adopt path, which needs a full pull over `POST /sync` — frozen, so it 503s. `accountSwitchPending` stays set and the device retries on the next boot, so nothing is corrupted, but until the freeze lifts the user sits on wiped local state. The bump belongs in the **unfreeze** deploy.

**Purge before the release deploy, not after.** The epoch re-adopt is a full history pull; if the purge has already landed, that single pull lands on the final state. Purging afterwards makes every device reconcile twice — once for the epoch, once for the digest divergence the purge creates.

Order:

1. *(unfrozen, hours)* `corpus_surgery` → `extract_activations` → `cache_rating_features` → `train_rating` → `map_rating_to_rank` → `build_axes` → `fit_population_constants` → `package_for_pwa`. Do **not** deploy yet.

   Take the `db/d1.sqlite` this chain reads from a **pre-freeze** fetch; it is only training input, and it does not need to be the surgical snapshot.

   `package_for_pwa` is the long pole and there is no cheap path here: removing ids changes `lookup.txt`, which changes every shard's `metadata.json`, and a re-rate moves `ratings_version` and re-emits `problem_params.bin` + `lookup.embeddings.bin` — so the per-size change detection rebuilds everything rather than skipping. That cost is deliberately on this side of the freeze. It is also why the ordering matters: you want to pay it **once**, and nothing downstream invalidates it — the DB surgery reads `ratings/` rather than feeding it, and `update_leaderboard_ranks.mjs` consumes `problem_params.bin` rather than regenerating it.

   The retrain steps read `db/d1.sqlite` while it still holds the doomed rows, and that is fine: `corpus_surgery` ran first, so those ids are already absent from `ratings/` and `per_problem.csv` and every join drops them. The rows are inert until the purge.
2. Freeze (step 2 above). This deploy ships whatever is in `pwa/dist` — including the newly packaged data, which is fine — but **not** the epoch bump.
3. Snapshot, surgery, apply, verify, leaderboard refresh (steps 4–8 above) — all inside the freeze.
4. Unfreeze **and** release in one deploy: `MAINTENANCE_MODE = "0"` together with the epoch bump.

One consequence worth planning for: **a trickle of attempts can still land on purged ids.** Clients keep serving the removed problems until they reload onto the new bundle, so between the purge and that reload a few attempts arrive for ids that no longer exist — plus whatever the outbox flushes from the frozen window. At ~3% of the corpus and current volume that is a handful of rows. They are harmless (the client strips rating fields for problems that left the corpus, and every estimate already skips rating-less rows), and a second, tiny purge pass mops them up whenever convenient. Note that those late arrivals also mean the local mirror needs a `--full` fetch before that mop-up pass, per the section above.

**Budget the window from the remote apply.** Everything else is fast — an incremental `fetch_database` right after a freeze pulls only the rows since the last one (seconds), and the planning script runs in a couple of minutes on ~380k rows. The unknown is `wrangler d1 execute --remote --file` on a few thousand statements; measure it in rehearsal against a scratch DB before committing to a window. If it turns out slow, the verification fetch in step 7 can move outside the freeze (unfreeze as soon as the apply reports success, verify immediately after, re-freeze if it disagrees) — the local run already proved the SQL equivalent, so this trades a small risk for a shorter window.

## If it goes wrong

The emitted SQL has no undo. D1 Time Travel is the recovery path — restore to a timestamp just before step 6 — so note the UTC time before applying. Keep the pre-surgery `db/d1.sqlite` as well; it is the exact pre-state and can be diffed against whatever production ends up holding.

Leaving the freeze on is the safe failure: the app still serves problems, and attempts queue client-side in the outbox. Do not leave it on for days — the outbox has a 30-minute throttle but the backlog is bounded by localStorage.

## What the client does about deleted rows

Nothing that needs arranging. Deleting server rows makes each device's stored digest disagree on the next `/report` or `/sync`; the mismatch escalates to the full mirror reconcile, whose deletion pass (`mirrorDeletePass`) drops local acked rows the server no longer has, from both the working set and the IndexedDB tail. Every divergence direction converges to an exact match. Do not add a bespoke migration for it.
