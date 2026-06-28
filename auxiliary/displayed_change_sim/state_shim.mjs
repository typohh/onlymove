// Minimal headless stand-in for pwa/js/state.js, aliased in by build.mjs's
// esbuild plugin so the REAL model.js / rank-decomposition.js can run under Node
// with no DOM, localStorage or IndexedDB. It exports exactly the surface those
// two modules import:
//
//   rank-decomposition.js: { state }                       (reads state.boardSize)
//   model.js:              { state, setStatsResults, setModelParams, setAdamState,
//                            getMergedAnswerHistory, getAnswerHistoryForSize }
//
// `state.answerHistory` is the unified (cross-size) working set the runner pushes
// onto; getMergedAnswerHistory returns shallow copies tagged with size, matching
// the real accessor's no-mutation contract. DEFAULT_USER_RANK = 0 (≈20k), as in
// the real state.js.

export const state = {
  boardSize: 9,
  userRank: 0,          // DEFAULT_USER_RANK in the real state.js
  serveRank: 0,
  modelParams: null,
  adamState: null,
  statsResults: null,
  answerHistory: [],    // unified working set (each entry tagged with `size`)
};

export function getMergedAnswerHistory() {
  return state.answerHistory.map(a => (a.size != null ? { ...a } : { ...a, size: 9 }));
}

export function getAnswerHistoryForSize(size) {
  return state.answerHistory.filter(a => a.size === size);
}

export function setStatsResults(results) { state.statsResults = results; }
export function setModelParams(params) { state.modelParams = params; }
export function setAdamState(adamState) { state.adamState = adamState; }
