"""Run the figure producers and bring the documents into line with what they measured.

    ./.venv/bin/python3 -m ablations --tier about        # minutes, no GPU: what pwa/about.html quotes
    ./.venv/bin/python3 -m ablations --tier methodology  # + the per-cell / torch measurements
    ./.venv/bin/python3 -m ablations --only corpus_figures
    ./.venv/bin/python3 -m ablations --check             # documents vs ledger, no measurement
    ./.venv/bin/python3 -m ablations --list

A run measures, merges the figures into ablations/figures.json, prints every key
whose value moved, rewrites the `<span data-fig="…">` values in the documents and
prints those edits too — that diff is the review step. A figure that moves can
change the sentence around it ("inside its interval", "most of"), which no
rewriter can see, so read the diff before committing.

Producers are the `*_figures.py` modules listed in PRODUCERS, cheapest tier first.
Each exposes `TIER`, `figures()` (→ list of `_figures.Figure`) and its own `main()`
so it also runs alone with a self-describing table. Any surprise in a producer's
inputs raises and stops the run; nothing here catches it.
"""

import argparse
import importlib
import sys

from ablations._figures import (
    TIERS, check_documents, keys_in_documents, load_ledger, save_ledger,
    snapshot, sync_documents, update_ledger)

# Module name (under ablations.) in run order. The module's own TIER decides
# whether a `--tier` run includes it.
PRODUCERS = (
    "corpus_figures",
    "rank_calibration_figures",
    "published_params_figures",
)


def _load(name):
    mod = importlib.import_module(f"ablations.{name}")
    for attr in ("TIER", "figures"):
        if not hasattr(mod, attr):
            raise SystemExit(f"ablations.{name} has no {attr}")
    if mod.TIER not in TIERS:
        raise SystemExit(f"ablations.{name}: unknown tier {mod.TIER!r}")
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--tier", choices=TIERS,
                   help="run every producer at this tier or cheaper")
    g.add_argument("--only", choices=PRODUCERS, help="run one producer")
    g.add_argument("--check", action="store_true",
                   help="only verify that every document span matches the ledger")
    g.add_argument("--list", action="store_true", help="list producers and their tiers")
    ap.add_argument("--no-sync", action="store_true",
                    help="update the ledger but leave the documents untouched")
    args = ap.parse_args(argv)

    if args.list:
        used = keys_in_documents()
        ledger = load_ledger()["figures"]
        for name in PRODUCERS:
            mod = _load(name)
            owned = sorted(k for k, e in ledger.items() if e["producer"] == f"ablations.{name}")
            print(f"{name:<32} tier={mod.TIER}")
            for k in owned:
                where = ", ".join(used.get(k, [])) or "(not quoted)"
                print(f"    {k:<34} {ledger[k]['text']:>12}   {where}")
        return 0

    if args.check:
        n = check_documents(load_ledger())
        print(f"{n} figure spans agree with the ledger")
        return 0

    if not (args.tier or args.only):
        ap.error("one of --tier, --only, --check, --list is required")

    if args.only:
        selected = [args.only]
    else:
        cap = TIERS.index(args.tier)
        selected = [n for n in PRODUCERS if TIERS.index(_load(n).TIER) <= cap]
    print(f"producers: {', '.join(selected)}\n", flush=True)

    ledger = load_ledger()
    measured = snapshot()
    all_changes = []
    for name in selected:
        mod = _load(name)
        print(f"{'=' * 78}\n{name}  [tier {mod.TIER}]\n{'=' * 78}", flush=True)
        figs = mod.figures()
        if not figs:
            raise SystemExit(f"ablations.{name} emitted no figures")
        changes = update_ledger(ledger, figs, f"ablations.{name}", mod.TIER, measured)
        all_changes.extend(changes)
        for fig in figs:
            print(f"  {fig.key:<34} {fig.text:>12}   {fig.note}")
        print(flush=True)
    save_ledger(ledger)

    print(f"{'=' * 78}\nledger: {len(all_changes)} figure(s) changed"
          f"  [git {measured['git']}, db max id {measured['db_max_id']}, "
          f"data {measured['data_version']}]")
    for key, old, new in all_changes:
        print(f"  {key:<34} {old if old is not None else '(new)':>12} -> "
              f"{new if new is not None else '(retired)'}")

    if args.no_sync:
        return 0
    edits = sync_documents(ledger, write=True)
    print(f"\ndocuments: {len(edits)} span(s) rewritten")
    # One line per (document, key): about.html repeats each key per language block.
    seen = {}
    for doc, key, old, new in edits:
        seen.setdefault((doc, key, old, new), 0)
        seen[(doc, key, old, new)] += 1
    for (doc, key, old, new), n in seen.items():
        print(f"  {doc:<22} {key:<34} {old:>12} -> {new}" + (f"   (×{n})" if n > 1 else ""))
    if edits:
        print("\nReview the sentences around each rewritten span before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
