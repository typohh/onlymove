"""Choose the frozen activation-encoder width M by measurement.

Two questions, both answered from RAW per-problem npz (candidates at full trunk
width). The shipped store holds the encoder's OUTPUT, so a re-run needs a raw
extraction (`extract_activations --raw --out-dir`) in place of it:

1. Spectrum scan — what fraction of the fit-set variance does an M-dim PCA
   keep, for M in {128, 256, 368, 512}? Bounds what any M-space refit can lose.
2. Simulated future refit — fit the encoder at M on HALF the problems ("the
   past"), then fit the downstream 128-PCA inside that M-space on ALL rows, and
   compare against the counterfactual full-space 128-PCA on the same rows.
   Scored on rows from problems the encoder never saw: variance-weighted R² of
   recovering the true 128 features from the M-space ones (best linear map fit
   on half the eval rows, scored on the other half), plus raw capture
   (reconstruction R² of standardized eval rows through the encoder span).

Fit set (the design the encoder will ship with): three thirds of ~N rows each —
correct cells uniform over the corpus, correct cells attempt-weighted (the
served distribution, via weighted sampling without replacement), and distinct
human-picked wrong cells resolved from tsumego_reports GTP coords against the
npz (x, y) map. Stratification across board sizes is inherited from each
frame's natural proportions.

Usage:
  ./.venv/bin/python3 -m ablations.encoder_m_sweep [--n-third 20000] [--rows-cache PATH]

The extracted row matrix is cached (--rows-cache) so the analysis half can be
re-run without re-reading the npz set.
"""

import argparse
import os
import sqlite3
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.constants import ALL_BOARD_SIZES
from utils.katago.coords import gtp_to_xy

DB_PATH = os.path.join(PROJECT_ROOT, "db", "d1.sqlite")
ACT_DIR = os.path.join(PROJECT_ROOT, "activations")
M_GRID = (128, 256, 368, 512)
K_DOWNSTREAM = 128
SEED = 0


def list_problems():
    """{(size, tid)} for every problem with a raw activation file."""
    frame = []
    for size in ALL_BOARD_SIZES:
        d = os.path.join(ACT_DIR, str(size))
        for e in os.scandir(d):
            if e.name.endswith(".npz"):
                frame.append((size, e.name[:-4]))
    return frame


def build_manifest(n_third, rng):
    """[(size, tid, kind, xy-or-None)] — xy None means the correct cell."""
    frame = list_problems()
    have = set(frame)
    print(f"corpus frame: {len(frame)} problems "
          f"({', '.join(f'{s}: {sum(1 for z in frame if z[0] == s)}' for s in ALL_BOARD_SIZES)})")

    con = sqlite3.connect(DB_PATH)
    attempts = {}
    for tid, size, n in con.execute(
            "SELECT problem, size, COUNT(*) FROM tsumego_reports GROUP BY problem, size"):
        if (size, tid) in have:
            attempts[(size, tid)] = n
    picked = []
    for tid, size, move in con.execute(
            "SELECT DISTINCT problem, size, move FROM tsumego_reports "
            "WHERE success=0 AND move IS NOT NULL AND move != ''"):
        xy = gtp_to_xy(move)
        if xy is not None and (size, tid) in have:
            picked.append((size, tid, xy))
    con.close()
    print(f"attempted problems: {len(attempts)}; distinct wrong picks: {len(picked)}")

    manifest = []

    idx = rng.choice(len(frame), size=min(n_third, len(frame)), replace=False)
    for i in idx:
        size, tid = frame[i]
        manifest.append((size, tid, "uniform", None))

    akeys = list(attempts)
    w = np.array([attempts[k] for k in akeys], dtype=np.float64)
    idx = rng.choice(len(akeys), size=min(n_third, len(akeys)), replace=False, p=w / w.sum())
    for i in idx:
        size, tid = akeys[i]
        manifest.append((size, tid, "served", None))

    idx = rng.choice(len(picked), size=min(n_third, len(picked)), replace=False)
    for i in idx:
        size, tid, xy = picked[i]
        manifest.append((size, tid, "picked", xy))

    return manifest


def extract_rows(manifest):
    """Read every manifest row's activation vector; one npz open per problem."""
    by_file = {}
    for j, (size, tid, kind, xy) in enumerate(manifest):
        by_file.setdefault((size, tid), []).append((j, xy))

    X = None
    tids_out = np.empty(len(manifest), dtype=object)
    kinds_out = np.empty(len(manifest), dtype=object)
    unresolved = 0
    t0 = time.time()
    for n_done, ((size, tid), wants) in enumerate(by_file.items()):
        with np.load(os.path.join(ACT_DIR, str(size), f"{tid}.npz")) as z:
            cand = z["candidates"]
            xys = z["xys"]
            ci = int(z["correct_idx"])
            if X is None:
                X = np.empty((len(manifest), cand.shape[1]), dtype=np.float16)
            for j, xy in wants:
                if xy is None:
                    row = ci
                else:
                    hit = np.where((xys[:, 0] == xy[0]) & (xys[:, 1] == xy[1]))[0]
                    if len(hit) == 0:
                        unresolved += 1
                        row = ci  # count it; substitute the correct cell so shapes stay fixed
                    else:
                        row = int(hit[0])
                X[j] = cand[row]
                tids_out[j] = f"{size}/{tid}"
                kinds_out[j] = manifest[j][2]
        if (n_done + 1) % 2000 == 0:
            rate = (n_done + 1) / (time.time() - t0)
            print(f"  read {n_done + 1}/{len(by_file)} files "
                  f"({rate:.0f}/s, eta {(len(by_file) - n_done - 1) / rate / 60:.1f} min)",
                  flush=True)
    print(f"unresolved picked coords (substituted correct cell): {unresolved}")
    return X, tids_out, kinds_out


def standardize(X, m=None, s=None):
    if m is None:
        m = X.mean(axis=0)
        s = X.std(axis=0)
        s[s < 1e-6] = 1e-6
    return (X - m) / s, m, s


def fit_pca(X, k, seed=SEED):
    from sklearn.decomposition import PCA
    return PCA(n_components=k, random_state=seed, svd_solver="randomized").fit(X)


def r2_map(zin_fit, ztrue_fit, zin_ev, ztrue_ev, weights):
    """Variance-weighted R² of predicting true features from M-space features
    via the best linear map (fit on one half, scored on the other)."""
    A = np.hstack([zin_fit, np.ones((len(zin_fit), 1))])
    W, *_ = np.linalg.lstsq(A, ztrue_fit, rcond=None)
    pred = np.hstack([zin_ev, np.ones((len(zin_ev), 1))]) @ W
    ss_res = ((ztrue_ev - pred) ** 2).sum(axis=0)
    ss_tot = ((ztrue_ev - ztrue_ev.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    return float(np.average(r2, weights=weights)), float(r2.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-third", type=int, default=20000)
    ap.add_argument("--rows-cache", default=os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "encoder_m_sweep_rows.npz"))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    if os.path.exists(args.rows_cache):
        print(f"loading cached rows: {args.rows_cache}")
        d = np.load(args.rows_cache, allow_pickle=True)
        X16, tids, kinds = d["X"], d["tids"], d["kinds"]
    else:
        manifest = build_manifest(args.n_third, rng)
        print(f"manifest: {len(manifest)} rows over "
              f"{len({(s, t) for s, t, _, _ in [(a, b, c, d_) for a, b, c, d_ in manifest]})} problems")
        X16, tids, kinds = extract_rows(manifest)
        np.savez(args.rows_cache, X=X16, tids=tids, kinds=kinds)
        print(f"cached rows -> {args.rows_cache}")

    X = X16.astype(np.float32)
    del X16
    n, d = X.shape
    print(f"fit matrix: {n} x {d}")

    # ---- 1. spectrum scan on the full fit set --------------------------------
    Z, _, _ = standardize(X)
    pca_full = fit_pca(Z, max(M_GRID))
    evr = np.cumsum(pca_full.explained_variance_ratio_)
    print("\nspectrum (cumulative explained variance of the fit set):")
    for m in M_GRID:
        print(f"  M={m:4d}: {evr[m - 1]:.4f}")

    # True arm for both scenarios: full-space 128 on ALL rows, own frame.
    ztrue_all = pca_full.transform(Z)[:, :K_DOWNSTREAM]
    ev_w = pca_full.explained_variance_[:K_DOWNSTREAM]

    def run_scenario(label, enc_mask, eval_idx):
        """Encoder fit on enc_mask rows; downstream 128 fit inside its M-space
        on ALL rows; scored against the true full-space 128 on eval_idx rows
        (map fit on one half, scored on the other)."""
        Zp, m1a, s1a = standardize(X[enc_mask])
        enc = fit_pca(Zp, max(M_GRID))       # top-M slice = the M-dim encoder
        del Zp
        Za = (X - m1a) / s1a                 # all rows in the encoder's frame
        eval_idx = eval_idx.copy()
        rng.shuffle(eval_idx)
        e1, e2 = eval_idx[: len(eval_idx) // 2], eval_idx[len(eval_idx) // 2 :]

        print(f"\n{label}:")
        print(f"  {'M':>4} {'raw capture R2':>15} {'128-feature R2 (wtd)':>21} {'worst comp':>11}")
        for m in M_GRID:
            C = enc.components_[:m]
            Ym = (Za - enc.mean_) @ C.T      # M-dim stored encoding, all rows
            recon = Ym[e1] @ C + enc.mean_
            ss_res = ((Za[e1] - recon) ** 2).sum()
            ss_tot = ((Za[e1] - Za[e1].mean(axis=0)) ** 2).sum()
            raw_r2 = 1.0 - ss_res / ss_tot

            pca_m = fit_pca(Ym, K_DOWNSTREAM)  # downstream 128 inside M-space
            zm_all = pca_m.transform(Ym)
            r2w, r2min = r2_map(zm_all[e1], ztrue_all[e1], zm_all[e2], ztrue_all[e2], ev_w)
            print(f"  {m:>4} {raw_r2:>15.4f} {r2w:>21.4f} {r2min:>11.4f}", flush=True)

    # ---- 2a. iid corpus growth: encoder on a random problem half ------------
    problems = np.array(sorted({t for t in tids}))
    half = rng.permutation(len(problems))
    past = set(problems[half[: len(problems) // 2]])
    is_past = np.array([t in past for t in tids])
    print(f"\nsplit: {is_past.sum()} past rows / {(~is_past).sum()} new rows "
          f"({len(past)} / {len(problems) - len(past)} problems)")
    run_scenario("simulated refit A (encoder fit on past half; scored on unseen problems)",
                 is_past, np.where(~is_past)[0])

    # ---- 2b. fit-set redesign: encoder on correct cells only ----------------
    # The historical shift this design just made: today's basis is fit on
    # correct cells; the new fit set adds picked cells. Scores whether an
    # encoder frozen BEFORE such a redesign spans the redesigned top-128.
    is_corr = np.array([k != "picked" for k in kinds])
    run_scenario("simulated refit B (encoder fit on correct cells only; scored on picked cells)",
                 is_corr, np.where(~is_corr)[0])


if __name__ == "__main__":
    main()
