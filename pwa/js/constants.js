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
