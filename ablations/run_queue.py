#!/usr/bin/env python3
"""Keep a fixed pool of arms training, drawn from a QUEUE that spans panels.

    ./.venv/bin/python3 -m ablations.run_queue                # reads ablations/queue.txt
    ./.venv/bin/python3 -m ablations.run_queue --queue other.txt

`run_panel` runs the arms of ONE panel and exits, so a chain of panels restarts
its ramp at every boundary and runs the card down to one arm at each panel's
tail. This runs the same ramp over a queue instead, so the pool stays full
across panel boundaries and the boundaries stop existing.

THE QUEUE FILE IS RE-READ EVERY TIME A SLOT OPENS. That is the point of it being
a file: appending work, deleting a line, or reordering what has not started yet
takes effect on the next slot, with nothing killed and nothing relaunched.
Editing it is the supported way to change priorities mid-run — which matters
because the arms that decide a baseline should never queue behind arms measured
against it.

Queue lines are `<panel> [<arm> ...]`; a bare panel name means every arm of that
panel, in manifest order. Blank lines and `#` comments are ignored. An arm that
already has a record is skipped when its slot comes up, so a finished queue can
be re-run and only trains what is missing.

Concurrency FILLS TO A KNOWN-GOOD LEVEL AND THEN RAMPS. The ceiling varies with
arm shape and is still discovered by measurement — an addition is kept only
while it buys aggregate throughput, and nothing is ever killed. The floor is
not in doubt and is no longer rediscovered one settle-interval at a time, which
used to leave the card underused for the first ten minutes of every start. GPU
utilization gates nothing here — a card reads busy long before small-kernel
fits stop scaling — and VRAM headroom gates every launch, fill included.

With `--gpu-lock` (or `ONLYMOVE_GPU_LOCK`) every arm takes a shared card's lock
for itself and the pool holds ONE arm; see `ablations.run_panel`.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations.run_panel import (  # noqa: E402
    POLL_SECS, RAMP_GAIN_MIN, RAMP_INTERVAL_SECS, RATE_WINDOW_SECS,
    VRAM_HEADROOM_FACTOR, _epochs_in_log, _gpu_mem, add_gpu_lock_arg,
    resolve_gpu_lock, with_gpu_lock)

# WHERE THE RAMP STARTS. The ceiling genuinely varies with arm shape — short
# arms have saturated at three or four, long ones at six — so the top is still
# discovered. The BOTTOM is not in doubt: every panel run so far has sustained
# this many, and climbing to it one step per settle interval leaves the card
# underused for the first ten minutes of every pool start for no information.
# So the pool fills to here immediately, subject to the same VRAM check, and
# ramps upward from there.
START_LEVEL = 4

PANELS = os.path.join(PROJECT_ROOT, "ablations", "panels")
DEV_RUNS = os.path.join(PROJECT_ROOT, "ablations", "dev_runs")
QUEUE = os.path.join(PROJECT_ROOT, "ablations", "queue.txt")


_ARM_RE = re.compile(r"--panel\s+(\S+)\s+--arm\s+(\S+)")


def live_arms():
    """{(panel, arm)} with a training process alive on this machine, whoever
    launched it. A pool restart, or a second pool started by mistake, must not
    hand out an arm that is already running: the record does not exist until
    the run ends, so the record check alone cannot see it, and two runs of one
    arm race to write the same file while sharing the card."""
    try:
        out = subprocess.run(["pgrep", "-fa", "ablations.ablation_mirt"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return set()
    return {m.groups() for m in _ARM_RE.finditer(out)}


def live_arm_count():
    return len(live_arms())


def read_queue(path):
    """[(panel, arm, cfg)] in queue order, skipping arms that have a record OR
    a live process.

    Read fresh on every call: the file is the live control surface, so a line
    added while the pool is running is picked up at the next free slot."""
    items = []
    if not os.path.exists(path):
        return items
    running_now = live_arms()
    for raw in open(path):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        panel, *arms = line.split()
        manifest_path = os.path.join(PANELS, f"{panel}.json")
        if not os.path.exists(manifest_path):
            print(f"  queue: no manifest for panel {panel!r}, skipping line",
                  flush=True)
            continue
        manifest = json.load(open(manifest_path))
        for arm in (arms or list(manifest["arms"])):
            if arm not in manifest["arms"]:
                print(f"  queue: {panel} has no arm {arm!r}, skipping",
                      flush=True)
                continue
            if os.path.exists(os.path.join(DEV_RUNS, panel, f"{arm}.json")):
                continue
            if (panel, arm) in running_now:
                continue
            items.append((panel, arm, manifest["arms"][arm]))
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queue", default=QUEUE,
                    help="queue file (default ablations/queue.txt)")
    ap.add_argument("--concurrency", default="auto",
                    help="'auto' (default) ramps in place on measured "
                         "throughput; an integer pins the level")
    add_gpu_lock_arg(ap)
    args = ap.parse_args(argv)
    lock = resolve_gpu_lock(args.gpu_lock)
    auto = lock is None and str(args.concurrency).lower() == "auto"
    fixed_level = None if auto else (1 if lock else max(1, int(args.concurrency)))

    running = {}          # (panel, arm) -> (Popen, log handle, log path)
    started = set()       # everything launched this run, so a re-read cannot
                          # hand the same arm out twice before its record lands
    retried = set()       # a failed arm is handed out ONCE more — a launch that
                          # died on a full card deserves a second slot, a broken
                          # arm does not deserve an endless one
    done_epochs = 0
    failures = []
    ceiling = None
    last_launch = 0.0
    rate_at_launch = None
    samples = []

    def launch(panel, arm, cfg):
        nonlocal last_launch
        panel_dir = os.path.join(DEV_RUNS, panel)
        os.makedirs(panel_dir, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "ablations.ablation_mirt",
               "--panel", panel, "--arm", arm]
        for k, v in cfg.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        log_path = os.path.join(panel_dir, f"{arm}.log")
        log = open(log_path, "w")
        print(f"  {panel}/{arm}: launched", flush=True)
        running[(panel, arm)] = (subprocess.Popen(
            with_gpu_lock(lock, panel, arm, cmd), cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT),
            log, log_path)
        started.add((panel, arm))
        last_launch = time.time()

    def reap():
        nonlocal done_epochs
        for key in list(running):
            proc, log, log_path = running[key]
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            done_epochs += _epochs_in_log(log_path)
            del running[key]
            print(f"  {key[0]}/{key[1]}: {'ok' if rc == 0 else f'FAILED rc={rc}'}",
                  flush=True)
            if rc != 0:
                failures.append(f"{key[0]}/{key[1]}")
                if key not in retried:
                    retried.add(key)
                    started.discard(key)
                    print(f"  {key[0]}/{key[1]}: will be retried once",
                          flush=True)

    def rate_now():
        total = done_epochs + sum(_epochs_in_log(p)
                                  for _, _, p in running.values())
        now = time.time()
        samples.append((now, total))
        while samples and samples[0][0] < now - RATE_WINDOW_SECS:
            samples.pop(0)
        if len(samples) < 2 or samples[-1][0] - samples[0][0] < 30:
            return None
        span = samples[-1][0] - samples[0][0]
        return (samples[-1][1] - samples[0][1]) / span

    def vram_allows_more():
        # Per-arm usage is read off EVERY arm process on the card, not only the
        # ones this pool launched: arms outlive the pool that started them, and
        # a pool that sized headroom by its own count alone launched into a
        # full card and had the launches die.
        mem = _gpu_mem()
        if mem is None:
            return True
        used, total = mem
        n = max(live_arm_count(), len(running))
        if n == 0:
            return True
        return (total - used) >= VRAM_HEADROOM_FACTOR * (used / n)

    def want_more():
        if not auto:
            return len(running) < fixed_level
        if ceiling is not None:
            return len(running) < ceiling
        if not running:
            return True
        # Fill to the known-good level without waiting: the interval exists to
        # let a NEW level prove itself, and these levels have already proved
        # themselves on every panel run. VRAM still gates each one.
        # Arms left running by a previous pool count toward the fill: the card
        # does not know which pool launched them.
        if max(len(running), live_arm_count()) < START_LEVEL:
            return vram_allows_more()
        if time.time() - last_launch < RAMP_INTERVAL_SECS:
            return False
        return vram_allows_more()

    print(f"queue {args.queue}, concurrency "
          f"{f'auto (fill to {START_LEVEL}, then ramp)' if auto else fixed_level}"
          + (f", each arm behind {lock}" if lock else ""), flush=True)
    idle_reported = False
    while True:
        reap()
        rate = rate_now()
        if (auto and ceiling is None and rate_at_launch is not None
                and rate is not None
                and time.time() - last_launch >= RAMP_INTERVAL_SECS):
            if rate < rate_at_launch * RAMP_GAIN_MIN:
                ceiling = len(running)
                print(f"  ramp: holding at {ceiling} — aggregate {rate:.2f} "
                      f"epochs/s against {rate_at_launch:.2f} before the last "
                      f"addition", flush=True)
            else:
                print(f"  ramp: {len(running)} arms live, aggregate "
                      f"{rate:.2f} epochs/s (was {rate_at_launch:.2f}) — still "
                      f"paying", flush=True)
            rate_at_launch = None

        pending = [it for it in read_queue(args.queue)
                   if (it[0], it[1]) not in started]
        if pending and want_more():
            # Only additions ABOVE the known-good level are judged; the fill is
            # not an experiment and a rate measured mid-fill would judge the
            # first real addition against a card that was not yet busy.
            if auto and len(running) >= START_LEVEL:
                rate_at_launch = rate if rate is not None else rate_at_launch
            panel, arm, cfg = pending[0]
            launch(panel, arm, cfg)
            idle_reported = False
        elif not pending and not running:
            # THE POOL DOES NOT EXIT ON AN EMPTY QUEUE. Work is added by editing
            # the file, and a runner that stopped the moment it drained would
            # have to be relaunched — which is the boundary this exists to
            # remove. It idles cheaply and picks up whatever lands next.
            if not idle_reported:
                print(f"  queue empty, pool idle — add lines to "
                      f"{args.queue} and they start at the next poll",
                      flush=True)
                if failures:
                    print(f"  failed so far: {', '.join(failures)}", flush=True)
                idle_reported = True
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    sys.exit(main())
