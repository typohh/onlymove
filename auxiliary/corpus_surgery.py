"""One-off corpus surgery: apply the id map to every id-keyed pipeline artifact.

Consumes the map emitted by `auxiliary.check_corpus_ids --emit-map` and performs, per entry:

  rename — mv the SGF and its same-named per-id artifacts to the canonical id
  merge  — back up, then DELETE the loser's SGF + per-id artifacts; drop its
           rows/keys from every table-shaped artifact (the survivor's stay)

Table-shaped artifacts (CSV rows, json dict keys, aligned npz arrays) follow
the PREFER-SURVIVOR rule: after mapping every id to its final form, a final id
claimed by more than one row keeps the row that originated from the surviving
file; if the survivor had no row, the first loser's row is kept under the new
id (same position, equivalent content) and counted as a fallback.

The `joint_pca_*.npz` memo caches are deleted outright: they are fingerprinted
on the raw caches' mtime + exact tid subset (utils.rating.features), so the
rewrite orphans them and the next training run refits automatically. The stale
`contrastive_vecs_*.npy` aux caches (already row-count-desynced from
correct_ids, unused by production) are left untouched.

Usage:
    ./.venv/bin/python3 -m auxiliary.corpus_surgery --map surgery/id_map.json           # dry run
    ./.venv/bin/python3 -m auxiliary.corpus_surgery --map surgery/id_map.json --apply
    # optional roots override for rehearsal on a copied tree (the corpus is then
    # <root>/tsumegos unless --tsumegos-dir says otherwise):
    ./.venv/bin/python3 -m auxiliary.corpus_surgery --map ... --root /tmp/rehearsal --apply

Idempotent: a re-run after an interruption treats an entry whose old file is
gone and whose new file exists as already applied; table rewrites are atomic
(tmp + os.replace) and map-lookup misses leave already-rewritten rows alone.
Renames are exactly reversible via the kept map; only losers are destructive,
hence the backup (SGF + activations + per_cell npz + the small table artifacts;
loser rows dropped from correct_vecs_<size>.npy are re-derivable from the
backed-up activation npzs via the incremental cache builder).
"""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np  # noqa: E402

from utils.paths import TSUMEGOS_DIR  # noqa: E402
from utils.rating.ratings_io import FLOOR_CSV, RATINGS_CSV, SLOPE_CSV  # noqa: E402

# Per-id file artifacts: (dir template, extension). {corpus} is the tsumego
# corpus root, which need not sit under {root}.
FILE_ARTIFACTS = [
    ("{corpus}/{size}", ".sgf"),
    ("{root}/activations/{size}", ".npz"),
    ("{root}/axes/per_cell/{size}", ".npz"),
]


def _label(tmpl, size):
    """An artifact template as a log key: tsumegos/9, activations/9, ..."""
    return tmpl.format(root="", corpus="tsumegos", size=size).lstrip("/")

TMP_SUFFIX = ".surgery-tmp"


class Log:
    def __init__(self):
        self.counts = defaultdict(int)

    def hit(self, key, n=1):
        self.counts[key] += n

    def dump(self):
        for k in sorted(self.counts):
            print(f"  {k}: {self.counts[k]}")


def load_map(path):
    with open(path) as f:
        m = json.load(f)
    entries = m["entries"]
    assert all(e["action"] in ("rename", "merge", "quarantine") for e in entries)
    # Combined old->final map (renames + merges). Old ids are unique
    # corpus-wide (zero truncation collisions at scan time), so a flat map is
    # safe; assert anyway. Quarantined ids have NO final id — their table rows
    # are dropped and their SGF moves to tsumegos/broken/.
    old_to_final = {}
    quarantined = set()
    for e in entries:
        assert e["old_id"] not in old_to_final and e["old_id"] not in quarantined, \
            f"old id twice: {e['old_id']}"
        if e["action"] == "quarantine":
            quarantined.add(e["old_id"])
        else:
            old_to_final[e["old_id"]] = e["new_id"]
    # survivor_old[final_id] = the old id of the file that survives at final_id.
    # A rename entry's old_id is the survivor unless the entry is a merge; a
    # group whose survivor was already canonical has no rename entry, so the
    # survivor's old id IS the final id.
    survivor_old = {}
    for e in entries:
        if e["action"] == "rename" and e["new_id"] is not None:
            survivor_old[e["new_id"]] = e["old_id"]
    for e in entries:
        if e["new_id"] is not None:
            survivor_old.setdefault(e["new_id"], e["new_id"])
    return m, entries, old_to_final, survivor_old, quarantined


# ---------------------------------------------------------------- file ops

def backup_file(src, backup_root, log, apply, root, corpus):
    """Copy src under backup_root at its path relative to the tree it belongs
    to: corpus files under tsumegos/, everything else relative to root."""
    if os.path.commonpath([os.path.abspath(src), corpus]) == corpus:
        rel = os.path.join("tsumegos", os.path.relpath(src, corpus))
    else:
        rel = os.path.relpath(src, root)
    dst = os.path.join(backup_root, rel)
    if apply:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    log.hit("backup: files copied")


def delete_losers(entries, root, corpus, backup_root, log, apply):
    for e in entries:
        if e["action"] not in ("merge", "quarantine"):
            continue
        for tmpl, ext in FILE_ARTIFACTS:
            p = os.path.join(tmpl.format(root=root, corpus=corpus, size=e["size"]),
                             e["old_id"] + ext)
            is_sgf = ext == ".sgf"
            if e["action"] == "quarantine" and is_sgf:
                # The SGF is preserved in the quarantine dir, not deleted.
                dst_dir = os.path.join(corpus, "broken", str(e["size"]))
                dst = os.path.join(dst_dir, e["old_id"] + ext)
                if os.path.exists(p):
                    if apply:
                        os.makedirs(dst_dir, exist_ok=True)
                        shutil.move(p, dst)
                    log.hit("quarantine sgf -> tsumegos/broken")
                elif os.path.exists(dst):
                    log.hit("quarantine already-applied")
                else:
                    log.hit("quarantine MISSING sgf")
                continue
            if os.path.exists(p):
                backup_file(p, backup_root, log, apply, root, corpus)
                if apply:
                    os.remove(p)
                log.hit(f"delete {e['action']} artifact {_label(tmpl, e['size'])}")
            else:
                log.hit(f"delete {e['action']} artifact MISSING {_label(tmpl, e['size'])}")


def rename_files(entries, root, corpus, log, apply):
    """Two-phase: mv every old -> new+TMP_SUFFIX (freeing all old names), then
    tmp -> final. Immune to rename chains and to targets equal to deleted loser
    names. An entry with old gone + new present counts as already applied."""
    renames = [e for e in entries if e["action"] == "rename"]
    for tmpl, ext in FILE_ARTIFACTS:
        phase2 = []
        for e in renames:
            d = tmpl.format(root=root, corpus=corpus, size=e["size"])
            old = os.path.join(d, e["old_id"] + ext)
            new = os.path.join(d, e["new_id"] + ext)
            if not os.path.exists(old):
                if os.path.exists(new):
                    log.hit(f"rename already-applied {_label(tmpl, e['size'])}")
                else:
                    log.hit(f"rename MISSING {_label(tmpl, e['size'])}")
                continue
            assert not os.path.exists(new), f"rename target occupied: {new}"
            if apply:
                os.rename(old, new + TMP_SUFFIX)
            phase2.append(new)
            log.hit(f"rename {_label(tmpl, e['size'])}")
        for new in phase2:
            if apply:
                os.rename(new + TMP_SUFFIX, new)


# ------------------------------------------------------------- table ops

def drop_quarantined(rows, quarantined, log, label):
    """Filter out rows whose ORIGINAL id is quarantined (no final id)."""
    if not quarantined:
        return rows
    kept = [r for r in rows if r[0] not in quarantined]
    n = len(rows) - len(kept)
    if n:
        log.hit(f"{label}: quarantined rows dropped", n)
    return kept


def collapse_rows(rows, survivor_old, old_to_final):
    """rows: [(orig_id, final_id, payload)] -> {final_id: payload} preferring
    the survivor-origin row. When the survivor has no row, fall back to a row
    that MAPS to the final id (a merge loser — same position, equivalent
    content). A colliding row that is neither (a STALE line for a long-removed
    problem whose id a rename now targets) is always dropped — keeping it
    would put a different position's value under the live id. Returns
    (kept ordered list, n_dropped, n_fallback)."""
    by_final = defaultdict(list)
    order = []
    for orig, final, payload in rows:
        if final not in by_final:
            order.append(final)
        by_final[final].append((orig, payload))
    kept, dropped, fallback = [], 0, 0
    for final in order:
        cands = by_final[final]
        if len(cands) == 1:
            kept.append((final, cands[0][1]))
            continue
        surv = survivor_old.get(final, final)
        chosen = next((p for o, p in cands if o == surv), None)
        if chosen is None:
            chosen = next((p for o, p in cands if old_to_final.get(o) == final), None)
            if chosen is not None:
                fallback += 1
        dropped += len(cands) - (1 if chosen is not None else 0)
        if chosen is not None:
            kept.append((final, chosen))
    return kept, dropped, fallback


def rewrite_csv(path, old_to_final, survivor_old, quarantined, log, apply, tid_col="tsumego_id"):
    if not os.path.exists(path):
        log.hit(f"csv MISSING {os.path.relpath(path, PROJECT_ROOT)}")
        return
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw = list(reader)
    ti = header.index(tid_col)
    rel0 = os.path.relpath(path, PROJECT_ROOT)
    rows = [(r[ti], old_to_final.get(r[ti], r[ti]), r) for r in raw]
    rows = drop_quarantined(rows, quarantined, log, f"csv {rel0}")
    changed = sum(1 for o, fnl, _ in rows if o != fnl)
    kept, dropped, fallback = collapse_rows(rows, survivor_old, old_to_final)
    rel = os.path.relpath(path, PROJECT_ROOT)
    log.hit(f"csv {rel}: ids rewritten", changed)
    log.hit(f"csv {rel}: rows dropped", dropped)
    if fallback:
        log.hit(f"csv {rel}: FALLBACK loser rows kept", fallback)
    if not apply or (changed == 0 and dropped == 0 and len(kept) == len(raw)):
        return
    tmp = path + ".tmp-surgery"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for final, r in kept:
            r = list(r)
            r[ti] = final
            w.writerow(r)
    os.replace(tmp, path)


def rewrite_json_dict(path, old_to_final, survivor_old, quarantined, log, apply):
    if not os.path.exists(path):
        log.hit(f"json MISSING {os.path.relpath(path, PROJECT_ROOT)}")
        return
    with open(path) as f:
        d = json.load(f)
    rel0 = os.path.relpath(path, PROJECT_ROOT)
    n_in = len(d)
    rows = [(t, old_to_final.get(t, t), v) for t, v in d.items()]
    rows = drop_quarantined(rows, quarantined, log, f"json {rel0}")
    changed = sum(1 for o, fnl, _ in rows if o != fnl)
    kept, dropped, fallback = collapse_rows(rows, survivor_old, old_to_final)
    rel = os.path.relpath(path, PROJECT_ROOT)
    log.hit(f"json {rel}: ids rewritten", changed)
    log.hit(f"json {rel}: keys dropped", dropped)
    if fallback:
        log.hit(f"json {rel}: FALLBACK loser keys kept", fallback)
    if not apply or (changed == 0 and dropped == 0 and len(kept) == n_in):
        return
    tmp = path + ".tmp-surgery"
    with open(tmp, "w") as f:
        json.dump(dict(kept), f)
    os.replace(tmp, path)


def rewrite_aligned_ids(ids, old_to_final, survivor_old, quarantined, log, label):
    """For an id list aligned to array rows: returns (final_ids, keep_row_idx,
    n_changed, n_dropped, n_fallback) applying the prefer-survivor collapse."""
    rows = [(t, old_to_final.get(t, t), i) for i, t in enumerate(ids)]
    rows = drop_quarantined(rows, quarantined, log, label)
    changed = sum(1 for o, fnl, _ in rows if o != fnl)
    kept, dropped, fallback = collapse_rows(rows, survivor_old, old_to_final)
    final_ids = [fnl for fnl, _ in kept]
    keep_idx = [i for _, i in kept]
    return final_ids, keep_idx, changed, dropped, fallback


def rewrite_correct_cache(size, old_to_final, survivor_old, quarantined, log, apply, root):
    ids_path = os.path.join(root, "activations/_signal_cache", f"correct_ids_{size}.json")
    vecs_path = os.path.join(root, "activations/_signal_cache", f"correct_vecs_{size}.npy")
    if not (os.path.exists(ids_path) and os.path.exists(vecs_path)):
        log.hit(f"correct cache {size} MISSING")
        return
    with open(ids_path) as f:
        ids = json.load(f)
    final_ids, keep_idx, changed, dropped, fallback = rewrite_aligned_ids(
        ids, old_to_final, survivor_old, quarantined, log, f"correct_ids_{size}")
    log.hit(f"correct_ids_{size}: ids rewritten", changed)
    log.hit(f"correct_ids_{size}: rows dropped", dropped)
    if fallback:
        log.hit(f"correct_ids_{size}: FALLBACK loser rows kept", fallback)
    if not apply or (changed == 0 and dropped == 0 and len(keep_idx) == len(ids)):
        return
    vecs = np.load(vecs_path, mmap_mode="r")
    assert len(ids) == vecs.shape[0], f"correct cache {size} misaligned"
    tmp = vecs_path + ".tmp-surgery.npy"
    np.save(tmp, np.asarray(vecs[np.array(keep_idx, dtype=np.intp)]))
    os.replace(tmp, vecs_path)
    tmp = ids_path + ".tmp-surgery"
    with open(tmp, "w") as f:
        json.dump(final_ids, f)
    os.replace(tmp, ids_path)


def rewrite_move_evals(size, old_to_final, survivor_old, quarantined, log, apply, root):
    path = os.path.join(root, "activations/_signal_cache", f"move_evals_{size}.npz")
    if not os.path.exists(path):
        log.hit(f"move_evals_{size} MISSING")
        return
    d = np.load(path, allow_pickle=False)
    tids = [str(t) for t in d["tids"]]
    final_ids, keep_idx, changed, dropped, fallback = rewrite_aligned_ids(
        tids, old_to_final, survivor_old, quarantined, log, f"move_evals_{size}")
    log.hit(f"move_evals_{size}: ids rewritten", changed)
    log.hit(f"move_evals_{size}: rows dropped", dropped)
    if fallback:
        log.hit(f"move_evals_{size}: FALLBACK loser rows kept", fallback)
    if not apply or (changed == 0 and dropped == 0 and len(keep_idx) == len(tids)):
        return
    # Carry EVERY row-aligned array through, not just `feats`: the cache holds one row per tid
    # across several blocks (`feats`, `cont`, …), and naming them here means a block added later
    # is dropped by a purge — which reads downstream as "this cache predates that block" and
    # forces a full re-extract, or worse is read as a legitimately narrower cache. Selecting by
    # first-axis length keeps the rule to "row-aligned arrays are re-indexed".
    idx = np.array(keep_idx, dtype=np.intp)
    arrays = {name: d[name][idx] for name in d.files
              if name != "tids" and d[name].shape[0] == len(tids)}
    unaligned = [n for n in d.files if n != "tids" and n not in arrays]
    if unaligned:
        raise RuntimeError(
            f"move_evals_{size}: arrays not aligned to tids ({unaligned}); refusing to rewrite "
            f"— surgery cannot know how their rows map to the surviving problems")
    tmp = path + ".tmp-surgery.npz"
    np.savez(tmp, tids=np.array(final_ids), **arrays)
    os.replace(tmp, path)


def rewrite_label_anchors(old_to_final, survivor_old, quarantined, log, apply, root):
    path = os.path.join(root, "axes", "label_anchors.npz")
    if not os.path.exists(path):
        log.hit("label_anchors MISSING")
        return
    d = np.load(path, allow_pickle=False)
    tids = [str(t) for t in d["tids"]]
    final_ids, keep_idx, changed, dropped, fallback = rewrite_aligned_ids(
        tids, old_to_final, survivor_old, quarantined, log, "label_anchors")
    log.hit("label_anchors: ids rewritten", changed)
    log.hit("label_anchors: columns dropped", dropped)
    if fallback:
        log.hit("label_anchors: FALLBACK loser columns kept", fallback)
    if not apply or (changed == 0 and dropped == 0 and len(keep_idx) == len(tids)):
        return
    M = d["M"][:, np.array(keep_idx, dtype=np.intp)]
    tmp = path + ".tmp-surgery.npz"
    np.savez(tmp, tids=np.array(final_ids), ids=d["ids"], M=M)
    os.replace(tmp, path)


def rewrite_bot_attempts(size, old_to_final, survivor_old, quarantined, log, apply, root):
    d = os.path.join(root, "attempts", str(size))
    if not os.path.isdir(d):
        return
    for name in sorted(os.listdir(d)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(d, name)
        with open(path) as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        rows = []
        for ln in lines:
            tid = ln.split(" ", 1)[0]
            rows.append((tid, old_to_final.get(tid, tid), ln))
        n_in = len(rows)
        rows = drop_quarantined(rows, quarantined, log, f"bot attempts {size}")
        changed = sum(1 for o, fnl, _ in rows if o != fnl)
        kept, dropped, fallback = collapse_rows(rows, survivor_old, old_to_final)
        log.hit(f"bot attempts {size}: ids rewritten", changed)
        log.hit(f"bot attempts {size}: lines dropped", dropped)
        if fallback:
            log.hit(f"bot attempts {size}: FALLBACK loser lines kept", fallback)
        if not apply or (changed == 0 and dropped == 0 and len(kept) == n_in):
            continue
        tmp = path + ".tmp-surgery"
        with open(tmp, "w") as f:
            for final, ln in kept:
                parts = ln.split(" ", 1)
                f.write(final + (" " + parts[1] if len(parts) > 1 else "") + "\n")
        os.replace(tmp, path)


def delete_joint_pca_caches(log, apply, root):
    d = os.path.join(root, "activations/_signal_cache")
    for name in sorted(os.listdir(d)):
        if name.startswith("joint_pca_") and name.endswith(".npz"):
            if apply:
                os.remove(os.path.join(d, name))
            log.hit("joint_pca cache deleted (auto-orphaned; rebuilds on next fit)")


# ---------------------------------------------------------------- verify

def verify(entries, corpus, log):
    bad = 0
    for e in entries:
        d = os.path.join(corpus, str(e["size"]))
        old = os.path.join(d, e["old_id"] + ".sgf")
        new = os.path.join(d, (e["new_id"] or e["old_id"]) + ".sgf")
        if os.path.exists(old):
            print(f"VERIFY FAIL: old file still present: {old}")
            bad += 1
        if e["action"] == "rename" and not os.path.exists(new):
            print(f"VERIFY FAIL: rename target absent: {new}")
            bad += 1
        if e["action"] == "merge" and not os.path.exists(new):
            print(f"VERIFY FAIL: merge survivor absent: {new}")
            bad += 1
        if e["action"] == "quarantine" and not os.path.exists(
                os.path.join(corpus, "broken", str(e["size"]), e["old_id"] + ".sgf")):
            print(f"VERIFY FAIL: quarantined sgf absent from broken/: {e['old_id']}")
            bad += 1
    print(f"verify: {'OK' if bad == 0 else f'{bad} FAILURES'}")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="perform the surgery (default: dry-run report only)")
    ap.add_argument("--root", default=PROJECT_ROOT,
                    help="tree to operate on (default: the repo; point at a "
                         "copy for rehearsal)")
    ap.add_argument("--tsumegos-dir", default=None,
                    help="corpus to operate on (default: the configured corpus, "
                         "or <root>/tsumegos when --root is given)")
    ap.add_argument("--backup-dir", default=None,
                    help="loser backup destination (default: <map dir>/loser_backup)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    corpus = os.path.abspath(args.tsumegos_dir or (
        TSUMEGOS_DIR if root == PROJECT_ROOT else os.path.join(root, "tsumegos")))

    m, entries, old_to_final, survivor_old, quarantined = load_map(args.map)
    backup_root = args.backup_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.map)), "loser_backup")
    sizes = sorted({e["size"] for e in entries})
    log = Log()
    print(f"map: {m['renames']} renames, {m['merges']} merges, "
          f"{m.get('quarantines', 0)} quarantines (generated {m['generated_at']}); "
          f"{'APPLY' if args.apply else 'DRY RUN'} on {args.root} (corpus {corpus})")

    # Back up the small table artifacts once, before any rewrite.
    table_files = ([os.path.join(args.root, "axes", n)
                    for n in ("per_problem.csv", "label_anchors.npz")]
                   + [os.path.join(args.root, "activations/_signal_cache", n)
                      for size in sizes
                      for n in (f"correct_ids_{size}.json",
                                f"game_phase_{size}.json",
                                f"move_evals_{size}.npz")]
                   + [os.path.join(args.root, "ratings", str(size), n)
                      for size in sizes
                      for n in (RATINGS_CSV, SLOPE_CSV, FLOOR_CSV)]
                   + [os.path.join(args.root, "attempts", str(size), n)
                      for size in sizes
                      if os.path.isdir(os.path.join(args.root, "attempts", str(size)))
                      for n in os.listdir(os.path.join(args.root, "attempts", str(size)))
                      if n.endswith(".txt")])
    for p in table_files:
        if os.path.exists(p):
            backup_file(p, backup_root, log, args.apply, root, corpus)

    delete_losers(entries, args.root, corpus, backup_root, log, args.apply)
    rename_files(entries, args.root, corpus, log, args.apply)

    for size in sizes:
        for n in (RATINGS_CSV, SLOPE_CSV, FLOOR_CSV):
            rewrite_csv(os.path.join(args.root, "ratings", str(size), n),
                        old_to_final, survivor_old, quarantined, log, args.apply)
        rewrite_json_dict(
            os.path.join(args.root, "activations/_signal_cache",
                         f"game_phase_{size}.json"),
            old_to_final, survivor_old, quarantined, log, args.apply)
        rewrite_correct_cache(size, old_to_final, survivor_old, quarantined, log,
                              args.apply, args.root)
        rewrite_move_evals(size, old_to_final, survivor_old, quarantined, log,
                           args.apply, args.root)
        rewrite_bot_attempts(size, old_to_final, survivor_old, quarantined, log,
                             args.apply, args.root)
    rewrite_csv(os.path.join(args.root, "axes", "per_problem.csv"),
                old_to_final, survivor_old, quarantined, log, args.apply)
    rewrite_label_anchors(old_to_final, survivor_old, quarantined, log, args.apply, args.root)
    delete_joint_pca_caches(log, args.apply, args.root)

    print()
    log.dump()
    if args.apply:
        print()
        ok = verify(entries, corpus, log)
        print("\nNext: re-run auxiliary.check_corpus_ids (expect 0 misnamed, "
              "0 dup groups), then pipeline.package_for_pwa --force.")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
