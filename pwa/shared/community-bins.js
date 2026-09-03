// pwa/shared/community-bins.js
//
// The community statistics' display bins over the Worker's fine-bucket time
// histograms, shared by the Worker (which folds the histograms for the
// /statistics page) and the client (which places the reader's own median in the
// same bins for that page's "you are here" marker). One table, so a bar and the
// marker on it can never disagree about where a bin's edges lie.

// Display bins in SECONDS. Solving time is log-normal (the hump sits at ~5–20s),
// so the low end is split finely and the long tail coarsely — this reveals the
// bell instead of collapsing it into one tall "0–10s" bar. Every edge is a
// multiple of the server's bucket width, so a bucket maps cleanly to exactly one
// display bin.
export const TIME_BINS = [
    { lo: 0, hi: 5, label: '0–5s' },
    { lo: 5, hi: 10, label: '5–10s' },
    { lo: 10, hi: 15, label: '10–15s' },
    { lo: 15, hi: 20, label: '15–20s' },
    { lo: 20, hi: 30, label: '20–30s' },
    { lo: 30, hi: 45, label: '30–45s' },
    { lo: 45, hi: 60, label: '45–60s' },
    { lo: 60, hi: 90, label: '1–1.5m' },
    { lo: 90, hi: 120, label: '1.5–2m' },
    { lo: 120, hi: 180, label: '2–3m' },
    { lo: 180, hi: 300, label: '3–5m' },
    { lo: 300, hi: Infinity, label: '5m+' },
];

// Fold a server fine-bucket histogram ([{bucket, n}] rows, bucket width
// `bucketSec` seconds) into the TIME_BINS display bins.
export function binHist(hist, bucketSec) {
    const rows = TIME_BINS.map(b => ({ label: b.label, value: 0 }));
    for (const r of (hist || [])) {
        const sec = r.bucket * bucketSec;   // lower edge of the server bucket
        const i = TIME_BINS.findIndex(b => sec >= b.lo && sec < b.hi);
        if (i >= 0) rows[i].value += r.n;
    }
    return rows;
}
