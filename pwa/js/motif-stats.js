// js/motif-stats.js
//
// The per-motif readings bound to the live app. The readings themselves — the
// tendency sign test against the availability null and the residual-accuracy
// regression, both computed ENTIRELY from the user's own attempts — are the
// pure function in motif-readings.js; this supplies what that function is
// deliberately not allowed to reach for itself: the merged history, the user's
// rank (the posterior mean, the value everything calibrated reads), the shipped
// axis count and the per-size rating→rank curve.

import { state, getMergedAnswerHistory } from './state.js';
import { SHIPPED_K } from './data-version.js';
import { curveForSize } from './rank-decomposition.js';
import { motifReadings, isFlagged } from './motif-readings.js';

export { isFlagged };

// Per-axis statistics over the user's own window. Returns null when there is no
// usable embedding data at all; otherwise an array of K entries, each of which
// may carry a null `tendency` or `strength` if that half lacks data.
export function computeMotifStats(history = null) {
    return motifReadings(history || getMergedAnswerHistory(), state.userRank,
                         SHIPPED_K, curveForSize);
}
