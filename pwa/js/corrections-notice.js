// js/corrections-notice.js
//
// One-time "your ratings/ranks were corrected retroactively" notice.
//
// When a deploy re-rates the corpus, re-fits the rating→rank curve, or re-fits
// the population constants, every EXISTING user's stored ranks/stats shift under
// them (backfill.js re-derives them). This tells those users why — once per
// change — via the grey confirm overlay (grey, not white/red: those are the
// answer cues).
//
// The ratings + curve signals are the SAME stamps backfill.js gates its re-derive
// on (RATINGS_VERSION_KEY / RANK_CURVE_VERSION_KEY). The notice fires purely on
// "version changed", NOT also on "a prior stamp existed" — deliberately, so a
// long-dormant returning user who predates the stamp (null stamp) is STILL told
// their ranks moved. The exclusions come from elsewhere, so this doesn't over-fire:
//   - a freshly-ADOPTED device is stamped to the CURRENT versions by
//     maybeFinishAccountSwitch (account-sync.js) BEFORE its history is pulled, so
//     its ratings/curve read as unchanged (flags false) — it is never notified;
//   - a brand-NEW device is likewise seeded to the CURRENT versions on its first
//     boot: main.js calls stampVersionsForFreshHistory() (a no-op once any history
//     exists) BEFORE the version gate, so a fresh device's flags are false too. So
//     the fresh-user exclusion is STRUCTURAL, not merely the < MIN_ATTEMPTS gate
//     below — which a new user could otherwise cross (play 2, reload) before being
//     stamped. The attempt gate stays as a backstop.
// Because existing users already carry the ratings/curve stamps, the very first
// ship of this feature (which lands with a re-rate) notifies them for free.
//
// Population constants get their OWN seen-stamp here (not a backfill gate — they
// don't touch stored per-attempt fields, so they must not trigger the expensive
// shard walk). It KEEPS the prior-stamp guard, because unlike ratings/curve the
// adoption path does NOT seed it — so a null-guarded pop axis is what stops an
// adopted device (null pop stamp) from firing on it. Every established load seeds
// pop, so after this feature's first ship every device can detect the next
// pop-only change; and this deploy is a ratings change anyway, so pop is not the
// axis that carries it.

import { getMergedAnswerHistory } from './state.js';
import { displayOverlay } from './overlays.js';
import { getTerm } from './i18n.js';
import POPULATION_CONSTANTS from '../tsumegos/population_constants.json';

const POP_VERSION_SEEN_KEY = 'popConstantsVersionSeen';
// A user with only a handful of attempts hasn't formed a sense of their rank yet,
// so a "your rank shifted" notice is just noise. 2 = the threshold the product
// asked for (any established user).
const MIN_ATTEMPTS_FOR_NOTICE = 2;
const GREY = 'rgb(128, 128, 128)';

let noticePending = false;

// Called from backfill.js right after it computes ratingsChanged/curveChanged,
// BEFORE it re-stamps those versions. `ratingsCorrected`/`curveCorrected` are the
// raw "version changed" flags (see the module header for why they carry no
// prior-stamp guard). This adds the population-constants axis and the
// >= MIN_ATTEMPTS gate, then arms the overlay for maybeShowRatingsCorrectionNotice().
export function detectRatingsCorrection({ ratingsCorrected, curveCorrected }) {
    const popVersion = (POPULATION_CONSTANTS && typeof POPULATION_CONSTANTS.version === 'string')
        ? POPULATION_CONSTANTS.version : null;
    const storedPop = localStorage.getItem(POP_VERSION_SEEN_KEY);
    // Population KEEPS the prior-stamp guard (null prior ⇒ not a correction, seed
    // silently): the adoption path doesn't seed pop, so this is what stops an
    // adopted device (null pop) from firing on the population axis.
    const populationCorrected = storedPop !== null && popVersion !== null
        && storedPop !== popVersion;
    // Seed/advance the population stamp now so the NEXT change is detectable.
    if (popVersion !== null) localStorage.setItem(POP_VERSION_SEEN_KEY, popVersion);

    if (!(ratingsCorrected || curveCorrected || populationCorrected)) return;

    // Established users only. getMergedAnswerHistory() is the cross-size working
    // set (a superset of any 2-attempt check).
    if (getMergedAnswerHistory().length < MIN_ATTEMPTS_FOR_NOTICE) return;

    noticePending = true;
}

// Called from main.js at the tail of the backfill chain — i.e. AFTER the
// re-derive that actually shifted the stored ranks — so the grey overlay lifts
// onto the already-corrected board. No-op unless detectRatingsCorrection armed it.
export function maybeShowRatingsCorrectionNotice() {
    if (!noticePending) return;
    noticePending = false;
    displayOverlay(
        getTerm('ratingsCorrectedMessage'),
        0,                       // duration ignored once buttonText is set
        null,                    // no post-dismiss callback
        GREY,
        null,                    // no stats block — plain message + button
        getTerm('ratingsCorrectedButton'),
    );
}
