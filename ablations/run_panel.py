#!/usr/bin/env python3
"""Run every arm of a refit-tier panel, skipping arms that already have a
record — the resume unit is the ARM, so a killed run costs one fit.

    ./.venv/bin/python3 -m ablations.run_panel barrier
    ./.venv/bin/python3 -m ablations.run_panel barrier --concurrency 3
    ./.venv/bin/python3 -m ablations.run_panel barrier --smoke-epochs 3

Arms are independent subprocesses; each arm's stdout goes to
`dev_runs/<panel>/<arm>.log`. `--smoke-epochs` overrides every arm's budget for
an end-to-end plumbing test; the short budget is recorded in each arm JSON and
producers refuse to quote it, so a smoke panel can never reach a document.

CONCURRENCY IS CALIBRATED IN PLACE, not by a separate phase: `--concurrency
auto` (the default) starts one arm and keeps adding another at a fixed cadence
for as long as each addition still buys aggregate throughput and VRAM holds a
safe margin — the epochs the ramp trains during calibration are the panel's own
epochs, so calibration costs nothing. When an addition stops paying, the ramp
stops ADDING and never kills: the extra arm keeps its GPU time and the level it
reached becomes the top-up ceiling as arms finish. Throughput is measured, not
inferred from utilization — a GPU reads 100% busy long before small-kernel fits
stop scaling, so utilization gates nothing here; VRAM is the one hard cap.
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

PANELS = os.path.join(PROJECT_ROOT, "ablations", "panels")
DEV_RUNS = os.path.join(PROJECT_ROOT, "ablations", "dev_runs")

# The ramp's clock: long enough for a freshly added arm to finish loading and
# put real epochs on the board before its contribution is judged.
RAMP_INTERVAL_SECS = 150
# An addition must buy at least this factor of aggregate epoch rate, else the
# ramp holds at the level it reached. Small on purpose: the cost of one arm too
# many is mild contention, the cost of stopping early is an idle GPU slice.
RAMP_GAIN_MIN = 1.05
# VRAM safety: adding an arm needs this many times the observed per-arm
# footprint free, so a spike during an arm's load phase cannot OOM the fleet.
VRAM_HEADROOM_FACTOR = 1.5
POLL_SECS = 15
RATE_WINDOW_SECS = 240

_EPOCH_RE = re.compile(r"epoch\s+(\d+)/")


def _gpu_mem():
    """(used_mb, total_mb) or None where nvidia-smi is unavailable — the ramp
    then leans on the throughput criterion alone."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        used, total = out.splitlines()[0].split(",")
        return float(used), float(total)
    except Exception:
        return None


def _epochs_in_log(path):
    """The last reported epoch counter in an arm's log, cheaply (tail bytes)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode(errors="replace")
        hits = _EPOCH_RE.findall(tail)
        return int(hits[-1]) if hits else 0
    except OSError:
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("panel", help="a manifest under ablations/panels/<name>.json")
    ap.add_argument("--concurrency", default="auto",
                    help="'auto' (default) ramps in place on measured "
                         "throughput; an integer pins the level")
    ap.add_argument("--smoke-epochs", type=int, default=None,
                    help="override every arm's budget (plumbing test only)")
    ap.add_argument("--force", action="store_true",
                    help="re-run arms whose records exist")
    args = ap.parse_args(argv)
    auto = str(args.concurrency).lower() == "auto"
    fixed_level = None if auto else max(1, int(args.concurrency))

    path = os.path.join(PANELS, f"{args.panel}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no panel manifest at {path}")
    manifest = json.load(open(path))
    panel = manifest["panel"]
    panel_dir = os.path.join(DEV_RUNS, panel)
    os.makedirs(panel_dir, exist_ok=True)

    todo = []
    for arm, cfg in manifest["arms"].items():
        meta = os.path.join(panel_dir, f"{arm}.json")
        if os.path.exists(meta) and not args.force:
            print(f"  {arm}: record exists, skipping", flush=True)
            continue
        todo.append((arm, cfg))
    if not todo:
        print("panel complete — nothing to run", flush=True)
        return 0
    print(f"panel {panel}: {len(todo)} arm(s) to run, concurrency "
          f"{'auto (ramp in place)' if auto else fixed_level}", flush=True)

    running = {}          # arm -> (Popen, log_handle, log_path)
    done_epochs = 0       # epochs banked by arms that already exited
    failures = []
    ceiling = None        # auto: set when an addition stops paying
    last_launch = 0.0
    rate_at_launch = None  # aggregate rate just before the most recent addition
    samples = []          # (t, total_epochs) for the sliding-window rate

    def launch(arm, cfg):
        nonlocal last_launch
        cmd = [sys.executable, "-u", "-m", "ablations.ablation_mirt",
               "--panel", panel, "--arm", arm]
        if args.force:
            cmd.append("--force")
        for k, v in cfg.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        if args.smoke_epochs is not None:
            cmd += ["--epochs", str(args.smoke_epochs)]
        log_path = os.path.join(panel_dir, f"{arm}.log")
        log = open(log_path, "w")
        print(f"  {arm}: launched ({' '.join(cmd[3:])})", flush=True)
        running[arm] = (subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT),
            log, log_path)
        last_launch = time.time()

    def reap():
        nonlocal done_epochs
        for arm in list(running):
            p, lh, lp = running[arm]
            rc = p.poll()
            if rc is None:
                continue
            lh.close()
            done_epochs += _epochs_in_log(lp)
            del running[arm]
            print(f"  {arm}: {'ok' if rc == 0 else f'FAILED rc={rc}'}",
                  flush=True)
            if rc != 0:
                failures.append(arm)

    def rate_now():
        total = done_epochs + sum(_epochs_in_log(lp)
                                  for _, _, lp in running.values())
        t = time.time()
        samples.append((t, total))
        while samples and samples[0][0] < t - RATE_WINDOW_SECS:
            samples.pop(0)
        if len(samples) < 2 or samples[-1][0] - samples[0][0] < 30:
            return None
        dt = samples[-1][0] - samples[0][0]
        return (samples[-1][1] - samples[0][1]) / dt

    def vram_allows_more():
        mem = _gpu_mem()
        if mem is None or not running:
            return True
        used, total = mem
        per_arm = used / len(running)
        return (total - used) >= VRAM_HEADROOM_FACTOR * per_arm

    def want_more():
        if not todo:
            return False
        if not auto:
            return len(running) < fixed_level
        if ceiling is not None:
            return len(running) < ceiling
        if not running:
            return True
        if time.time() - last_launch < RAMP_INTERVAL_SECS:
            return False
        return vram_allows_more()

    while todo or running:
        reap()
        r = rate_now()
        if (auto and ceiling is None and rate_at_launch is not None
                and r is not None
                and time.time() - last_launch >= RAMP_INTERVAL_SECS):
            # The most recent addition has had its settle time: judge it. The
            # arm it added is never killed either way — only further additions
            # stop.
            if r < rate_at_launch * RAMP_GAIN_MIN:
                ceiling = len(running)
                print(f"  ramp: holding at {ceiling} — aggregate "
                      f"{r:.2f} epochs/s vs {rate_at_launch:.2f} before the "
                      f"last addition (below the {RAMP_GAIN_MIN:.2f}x bar)",
                      flush=True)
            else:
                print(f"  ramp: {len(running)} arms live, aggregate "
                      f"{r:.2f} epochs/s (was {rate_at_launch:.2f}) — "
                      f"still paying", flush=True)
            rate_at_launch = None      # each addition is judged exactly once
        if todo and want_more():
            if auto and running:
                rate_at_launch = r if r is not None else rate_at_launch
            arm, cfg = todo.pop(0)
            launch(arm, cfg)
        time.sleep(POLL_SECS)

    if auto:
        print(f"  ramp: finished at level "
              f"{ceiling if ceiling is not None else 'unsaturated'}",
              flush=True)
    if failures:
        print(f"PANEL INCOMPLETE — failed arms: {', '.join(failures)}",
              flush=True)
        return 1
    print("PANEL COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
