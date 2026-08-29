# analysis_server

KataGo behind a small HTTP server, serving `https://api.onlymove.net`. The PWA calls it only as a **fallback**: a tsumego SGF normally carries its own evaluation in each branch comment (`"<winrate>%/<score>/<visits>"`), and `pwa/js/analysis.js` reads that directly. The server is asked only when a position has no embedded number — a line the user explored themselves.

Two programs, and only the first is required:

| | runs on | role |
|---|---|---|
| `server.py` | the cloud box | answers every request, and hands work to a worker when one is connected |
| `worker.py` | any machine with a GPU | dials in and answers with real search instead of raw policy |

`server.py` is deliberately **stdlib + sgfmill only** — it is deployed on its own, with no repo beside it. `worker.py` lives in the repo and reuses `utils.katago`.

## What the worker is for

Not speed. Measured, the box spends about 200 ms of compute per answer and a GPU about 45 ms — but reaching the GPU costs a round trip that spends nearly the whole saving, so at equal visits the two are a wash.

What the box **cannot** do is search. One visit is the raw policy: an opinion with no reading behind it. Thirty visits there would take seconds. So the worker **replaces** the visit count on every query it takes (`WORKER_VISITS`, default 30), answering in roughly a second end to end against the box's third of one, and returns a genuinely read answer.

The reply carries `visits`, and the client shows its blinking-red *"evaluated on the spot, may contain mistakes"* marker only for a one-visit answer. A searched answer has earned its number.

## How the worker connects

It **dials out**. The worker parks a few long polls against `POST /worker/poll` and takes jobs down the open responses, so it needs no inbound port, no address, no DNS record and no certificate. Nothing about the deployment changes while it is connected, and stopping it at any instant is safe — an unanswered job is one the box answers itself once its own timeout passes.

**The pool's capacity is the number of parked polls.** A job is only ever handed to a poll that is already waiting, so a busy or absent worker degrades to local service rather than queueing behind itself. That is the whole failover rule, and it needs no health check to state.

Failure is bounded rather than absent. A worker that dies holding a poll is usually caught on the write and the job comes straight back; a dropped network gives no such error, so one request pays `WORKER_JOB_TIMEOUT` before `WORKER_COOLDOWN` takes the worker out of rotation.

## Running the worker

```bash
./.venv/bin/python3 -m analysis_server.worker
```

The shared secret is read from `~/.config/onlymove/worker_token` (0600, outside the repo so it cannot be committed) and must match the `worker_token` file beside `server.py` on the box. `ONLYMOVE_WORKER_TOKEN` overrides it. **An empty token on the box means the feature is off with no second switch** — the three `/worker` routes 404 — so the server can be deployed before any worker exists.

A secret is never passed on a command line: `/proc` makes that world-readable for as long as the process runs.

### As a service, starting at boot

`onlymove-worker.service` here is a **user** unit — it needs this user's venv, the token, and the OpenCL tuning cache in `~/.katago`, and no privilege at all. Install it as a symlink so edits track git:

```bash
ln -s ~/Repositories/onlymove/analysis_server/onlymove-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now onlymove-worker
loginctl enable-linger $USER          # start at BOOT, not at first login
journalctl --user -u onlymove-worker -f
```

Without the linger step a user unit waits for a login, which is not what "on boot" means on a machine nobody logs into. `systemctl --user stop onlymove-worker` hands analysis back to the box within a second; `Restart=always` with a 30 s delay covers the cold-boot case where the network is not up yet when the preflight runs.

Useful environment overrides: `ONLYMOVE_WORKER_VISITS` (30), `ONLYMOVE_WORKER_THREADS` (4 parked polls), `ONLYMOVE_WORKER_SEARCH_THREADS` (8 — a *latency* setting, since visits are sequential within one query unless several threads explore at once), `ONLYMOVE_ANALYSIS_URL`.

## Deploying

```bash
analysis_server/deploy.sh              # ship server.py + analysis.cfg, restart, verify
analysis_server/deploy.sh --token      # also install the shared secret
analysis_server/deploy.sh --no-restart # ship only
```

It ships by `scp` (the box has no rsync) and verifies by md5 what actually landed, backs up the previous `server.py` first, and prints a rollback command if the server does not answer. It edits nothing under `/etc`.

## The box

| | |
|---|---|
| instance | `instance-20250818-100543`, zone `us-east1-d`, project `onlymove-467201` |
| machine | `e2-micro` — free-tier eligible in us-east1 |
| disk | 30 GB `pd-standard` — free-tier eligible; `pd-balanced` is not |
| address | `onlymove-api-ip`, **reserved static** — an ephemeral IP is released by a stop, which would break DNS and cert renewal |
| service | `katago.service`, `WorkingDirectory=/home/typo_hh/server`, `Restart=always` |
| proxy | nginx terminates TLS (Certbot) and proxies `location /` to `127.0.0.1:8080` |

Two accounts, both with passwordless sudo and both keyed through project metadata (a hand-appended `authorized_keys` is not durable — the guest agent owns that file): `typo_hh` owns the service and its working directory, and `ootakamoku` is the gcloud-provisioned admin.

`analysis.cfg` here is **the box's**, tuned for a 1 GB micro — one thread, batch 1. It is emphatically not `bots/analysis.cfg`, which is the pipeline's, tuned for a GPU, and sets `requireMaxBoardSize = true`.

**The board-size ceiling is stated twice and the two must move together**: `MAX_BOARD_SIZE` in `server.py` refuses anything larger with a 400 before any evaluation, and `maxBoardXSizeForNNBuffer` in `analysis.cfg` sizes the NN buffer to the same bound, so an oversized board fails inside KataGo regardless. Nothing above 13x13 is in the corpus.

## Checking it

```bash
curl -X POST --data-binary '(;FF[4]GM[1]CA[UTF-8]SZ[9]RU[japanese]KM[6.5];B[gf];W[ef])' \
     https://api.onlymove.net/
```

`"visits": 1` means the box answered; anything higher means a worker is connected and searching. `GET /worker/status` with the token reports parked polls and jobs in flight.
