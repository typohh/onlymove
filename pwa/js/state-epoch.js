// js/state-epoch.js
//
// The state-epoch wipe gate — the PERMANENT mechanism that replaced the
// migration ladders (docs/db_surgery_and_legacy_removal_plan.md section 6).
// A single `stateEpoch` localStorage key is checked at state.js IMPORT time,
// before any persisted property is read. Epoch current → cheap no-op (the
// steady-state path). Epoch absent/stale → wipe the account-derived state and
// re-adopt the device's own account from the server: keep only the userId
// (truncated to 8 chars — the server-side user-id normalization made the
// truncated id THE account), device prefs (board size, audio, onboarding
// flags, survey/reload throttles), and the curated ignored-problems list
// (extracted from the old history into its own key, re-applied post-restore);
// everything else — history, rank, models, sync bookkeeping, version stamps,
// and every legacy migration flag — is removed, and the pending-adopt flag
// routes init through the SAME maybeFinishAccountSwitch path an account
// switch uses (from-0 pull, tail re-split, silent refit). No bespoke restore
// code.
//
// For a future breaking state change: bump CURRENT_STATE_EPOCH and, if some
// state can survive, add a keep-path here. Both epochs so far wipe universally:
//   1 — the tsumego-id rewrite. Already-synced devices held old problem ids in
//       local copies, and only a full wipe-and-re-adopt propagates the rewrite.
//   2 — the single AGA-style rank scale (shared/rank-math.js). Every stored rank
//       changed units, and a stale value is not detectably wrong: the old
//       normalized userRank/problemRank are perfectly plausible NEW ranks (a
//       stored 0 meant 20k and now reads as 1d). The per-user recall and weakness
//       models are worse still — their fitted coefficients are expressed in the
//       old units, so they must be refit, which the wipe's re-adopt does.
//
// Dependency-light on purpose (no imports): runs before state.js initializes.

const STATE_EPOCH_KEY = 'stateEpoch';
const CURRENT_STATE_EPOCH = 2;

// Keys that survive the wipe. The userId is the account credential;
// the rest are device prefs no server pull can restore.
const KEEP_KEYS = new Set([
    'userId',
    'boardSize',
    'audioDisabled',
    'lastReloadTime',
    'rankSurveyShownAt_v2',
    'ignoredProblems_v1',
    STATE_EPOCH_KEY,
    'accountSwitchPending',
]);
const KEEP_PREFIXES = ['onb']; // onboarding flags (onbShown_*, onbInitialized, …)

// Same key state.js uses (duplicated literal — this module imports nothing).
const IGNORED_PROBLEMS_KEY = 'ignoredProblems_v1';

// Pull the curated ignored-problem ids out of the raw pre-wipe answerHistory
// (unified array, or the ancient {9:[…],13:[…]} bucket shape) into the
// standalone key that survives the wipe; maybeFinishAccountSwitch re-applies
// them to the restored history. Only the working set is readable synchronously
// here — an ignored problem whose last attempt sits in the IndexedDB cold tail
// is out of the recall window anyway.
function extractIgnoredProblems() {
    let parsed;
    try { parsed = JSON.parse(localStorage.getItem('answerHistory')); } catch { return; }
    const arrays = Array.isArray(parsed) ? [parsed]
        : (parsed && typeof parsed === 'object')
            ? Object.values(parsed).filter(Array.isArray)
            : [];
    const ignored = new Set();
    try {
        const prev = JSON.parse(localStorage.getItem(IGNORED_PROBLEMS_KEY));
        if (Array.isArray(prev)) for (const id of prev) ignored.add(id);
    } catch { /* fresh set */ }
    for (const arr of arrays) {
        for (const a of arr) {
            if (a && a.ignored && typeof a.problemId === 'string') ignored.add(a.problemId);
        }
    }
    if (ignored.size > 0) {
        try { localStorage.setItem(IGNORED_PROBLEMS_KEY, JSON.stringify([...ignored])); } catch { /* best effort */ }
    }
}

export function runStateEpochGate() {
    let stored;
    try { stored = parseInt(localStorage.getItem(STATE_EPOCH_KEY), 10); } catch { stored = NaN; }
    if (stored === CURRENT_STATE_EPOCH) return;

    extractIgnoredProblems();

    // Wipe: everything except the keep list. A snapshot first — removal
    // mutates the store under iteration.
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) keys.push(localStorage.key(i));
    for (const k of keys) {
        if (k === null || KEEP_KEYS.has(k)) continue;
        if (KEEP_PREFIXES.some((p) => k.startsWith(p))) continue;
        localStorage.removeItem(k);
    }
    // The resume slot references a pre-epoch problem/orientation.
    try { sessionStorage.removeItem('resumeProblem'); } catch { /* unavailable */ }

    // Truncate a legacy long userId — the server-side normalization (S7)
    // merged the long id's history onto its 8-char prefix, so the truncated id
    // IS this device's account.
    const userId = localStorage.getItem('userId');
    if (userId && userId.length > 8) {
        localStorage.setItem('userId', userId.slice(0, 8));
    }

    // An existing account re-adopts itself through the standard adopt path
    // (init's maybeFinishAccountSwitch: from-0 pull, tail re-split, silent
    // refit). 'epoch' (vs an account switch's '1') = on a failed pull keep the
    // flag and retry next boot rather than falling back to a fresh account —
    // the userId is the user's credential. A device already mid-switch keeps
    // its '1' (the switch's own fresh-id failure semantics stand); a fresh
    // device (no userId yet) has nothing to restore.
    if (userId && localStorage.getItem('accountSwitchPending') !== '1') {
        localStorage.setItem('accountSwitchPending', 'epoch');
    }

    localStorage.setItem(STATE_EPOCH_KEY, String(CURRENT_STATE_EPOCH));
    if (userId) console.log(`state-epoch: wiped to epoch ${CURRENT_STATE_EPOCH}; account ${localStorage.getItem('userId')} re-adopts from the server`);
}
