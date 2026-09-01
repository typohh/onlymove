#!/usr/bin/env python3
"""Run every arm of a refit-tier panel, skipping arms that already have a
record — the resume unit is the ARM, so a killed run costs one fit.

    ./.venv/bin/python3 -m ablations.run_panel barrier
    ./.venv/bin/python3 -m ablations.run_panel barrier --concurrency 3
    ./.venv/bin/python3 -m ablations.run_panel barrier --smoke-epochs 3

Arms are independent, so `--concurrency N` runs N as subprocesses to saturate a
GPU whose single fit does not. Each arm's stdout goes to
`dev_runs/<panel>/<arm>.log`. `--smoke-epochs` overrides every arm's budget for
an end-to-end plumbing test; the short budget is recorded in each arm JSON and
producers refuse to quote it, so a smoke panel can never reach a document.
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

PANELS = os.path.join(PROJECT_ROOT, "ablations", "panels")
DEV_RUNS = os.path.join(PROJECT_ROOT, "ablations", "dev_runs")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("panel", help="a manifest under ablations/panels/<name>.json")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--smoke-epochs", type=int, default=None,
                    help="override every arm's budget (plumbing test only)")
    ap.add_argument("--force", action="store_true",
                    help="re-run arms whose records exist")
    args = ap.parse_args(argv)

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
    print(f"panel {panel}: {len(todo)} arm(s) to run, "
          f"concurrency {args.concurrency}", flush=True)

    running = {}          # arm -> (Popen, log_handle)
    failures = []

    def reap():
        for arm in list(running):
            p, lh = running[arm]
            rc = p.poll()
            if rc is None:
                continue
            lh.close()
            del running[arm]
            print(f"  {arm}: {'ok' if rc == 0 else f'FAILED rc={rc}'}",
                  flush=True)
            if rc != 0:
                failures.append(arm)

    for arm, cfg in todo:
        while len(running) >= args.concurrency:
            reap()
            time.sleep(10)
        cmd = [sys.executable, "-u", "-m", "ablations.ablation_mirt",
               "--panel", panel, "--arm", arm]
        if args.force:
            cmd.append("--force")
        for k, v in cfg.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        if args.smoke_epochs is not None:
            cmd += ["--epochs", str(args.smoke_epochs)]
        log = open(os.path.join(panel_dir, f"{arm}.log"), "w")
        print(f"  {arm}: launched ({' '.join(cmd[3:])})", flush=True)
        running[arm] = (subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT), log)

    while running:
        reap()
        time.sleep(10)
    if failures:
        print(f"PANEL INCOMPLETE — failed arms: {', '.join(failures)}",
              flush=True)
        return 1
    print("PANEL COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
