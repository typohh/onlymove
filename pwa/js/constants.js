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
