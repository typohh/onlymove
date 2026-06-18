// js/constants.js

export const SGF_COMMENTS = {
  VITAL_POINT: 'Q',
};

export const TOTAL_RANKS = 29;

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
// the *merged* history (the momentum/recall replay warms over the last 3000),
// so the working set is a superset of every synchronous read.
export const ANSWER_HISTORY_WORKING_SET = 3000;

// Minimum merged answer-history entries (both sizes, repeats included) before
// the detailed-stats panel will open. The `statsNotEnough` copy interpolates
// this same value, so they can't drift apart (the string used to hardcode "5"
// while the gate already required 10).
export const MIN_STATS_ATTEMPTS = 10;
