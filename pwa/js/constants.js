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
