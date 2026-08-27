// js/constants.js


export const APP_MODES = {
  SOLVING: 'solving',
  ANALYZING: 'analyzing',
  LINKED: 'linked',
};

export const ANALYSIS_STATUS = {
  RETRY: 'Retry',
  ERROR: 'Error',
};

export const MODEL_INPUT_HISTORY_LENGTH = 500;

// True on devices that play a move in TWO taps: the first sets a ghost stone,
// a second tap on the same point confirms it (board.js). Desktop plays on one
// click. Defined here (not board.js) because onboarding.js words the intro
// bubble from it and importing board.js there would close a cycle through
// ui.js — the copy and the branch it describes must not drift apart.
export const usesTapToConfirm = () =>
  'ontouchstart' in window || navigator.maxTouchPoints > 0;

// Set once account-sync's one-off historical backfill has confirmed every
// local attempt is represented server-side (set immediately on devices with
// nothing to reconcile). The Account menu entry is gated on it: until the
// server log is complete, logging the account in elsewhere would silently
// lose the not-yet-uploaded local history. Defined here (not account-sync.js)
// because ui.js reads it too and account-sync already imports ui.
export const HISTORY_BACKFILL_DONE_KEY = 'historyBackfillDone_v1';
export const RECALL_CANDIDATE_HISTORY_LENGTH = 3000;

// Number of most-recent attempts (unified across board sizes) kept in
// localStorage as the synchronous working set. Everything older is evicted to
// IndexedDB (see history-store.js + state.evictColdTail) and read back only for
// the full-history charts. Set equal to the largest hot window the app uses on
// the *merged* history (the recall/weakness models warm over the last 3000),
// so the working set is a superset of every synchronous read.
export const ANSWER_HISTORY_WORKING_SET = 3000;

// The answer-history fields that carry an embedding fit's output. Split in two
// because the passes that drop them want different halves:
//   VECTOR  — what a refit invalidates and a refill can restore. Besides the
//             two per-move vectors this is the availability null of a wrong
//             attempt's problem (altMean/altSd/altAbove/altBelow — the
//             uniform-over-available error-null moments the tendency and
//             judgment readings stand against), which is a function of the
//             same per-cell bin and moves with the fit exactly as the vectors
//             do.
//   ALL     — the vectors plus `pickedCellIndex` and `correctCellIndex`, which
//             are NOT fit-derived (they address the SGF's root variations, so
//             a refit cannot move them) and are the keys the refill is looked
//             up BY. A pass that intends to refill therefore keeps them, while
//             a pass whose rows are never coming back into model range — the
//             keep-window strip, the removed-problem prune, the cold-tail
//             eviction — drops them too.
// Lives here rather than beside any one of those passes because state.js and
// backfill.js both apply it and neither may import the other.
// `noEmb` marks a row the sidecar cannot supply a vector for — its problem left
// the corpus, or it was never in the axis fit. Without it the fill-as-you-go pass
// re-asks for the same unfillable rows on every load, forever. It clears with the
// vectors: a new fit (or a re-added problem) deserves a fresh attempt, and the
// wipe is the moment that decision is re-made. (`altMean: []` is the analogous
// computed-but-degenerate mark for the null — see computeAltStats.)
export const EMBEDDING_VECTOR_FIELDS = ['embedding', 'pickedEmbedding',
                                        'altMean', 'altSd', 'altAbove', 'altBelow', 'noEmb'];
export const EMBEDDING_ALL_FIELDS = [...EMBEDDING_VECTOR_FIELDS, 'pickedCellIndex', 'correctCellIndex'];

// The device-level record of WHICH motif fit the stored vectors came from. One
// value for the whole history, which is what makes the scheme cheap: a vector is
// current if this says so, so nothing is stored per row. It holds only while
// every writer respects it — see motifDataIsCurrent (data-version.js) — and while
// the wipe that advances it reaches every holder of a vector (backfill.js).
// Lives here because the backfill writes it and the write-time check reads it.
export const EMBEDDING_VERSION_KEY = 'weaknessEmbeddingVersion';
// The same for the other two generations of derived per-attempt data: the rating
// fit (rating, slope, floor, and the quality derived from them) and the rank curve
// (problemRank, derived from the rating through M). Three generations, one rule —
// see storedVersionMatches (data-version.js).
export const RATINGS_VERSION_KEY = 'ratingsVersion';
export const RANK_CURVE_VERSION_KEY = 'rankCurveVersion';

// The shared one-standard-deviation motif gate, deliberately ONE constant for
// three readings that must not drift apart: a problem's motif is DISPLAYED when
// its loading clears this many corpus standard deviations (motif-bubble.js); a
// user's weakness becomes a SERVE CATEGORY and a training-focus entry when its
// centred-contrast posterior clears this many posterior standard errors
// (weakness.js); and the stats pane ADMITS a finding past the same bar in
// probability form, normCdf(MOTIF_Z) (stats-weaknesses.js). The three are
// z-scores of different statistics (corpus salience vs posterior confidence),
// so this lock is a deliberate coupling: retuning the gate for one reading
// retunes all of them.
export const MOTIF_Z = 1.0;
