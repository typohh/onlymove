// js/motif-explain.js
//
// The sentence behind a motif's name: what the motif IS, and what one of the
// two readings says about the user on it, with the number in it. Opened from
// the front page's Weaknesses list, from each row of the Weaknesses and
// Tendencies panes, and from the review-mode "what this tests" list (where it
// is the description alone — no reading is about the problem).
//
// The description is keyed by the axis's CANONICAL English name (labels.json
// `positive`/`negative`), which is what package_for_pwa resolves a shipped tail
// through; the localized name the user sees comes from localizedMotif as
// everywhere else. A tail with no description entry gets the reading alone.
//
// Two readings, two different subjects, and the sentence keeps them apart:
//   strength — about the PROBLEM, whose answer is a move of this type: accuracy
//              beyond what difficulty predicts, in ranks;
//   tendency — about the user's own MOVE against the answer: the share of wrong
//              answers that reached for the motif, against what the board
//              offered.

import { getTerm } from './i18n.js';
import { MOTIF_Z } from './constants.js';
import { isFlagged } from './motif-readings.js';

// Past this the reading is stated flatly; between the admission gate and it
// the sentence adds that the reading is still uncertain. The two-sigma point,
// the same one motif-bubble.js hedges its list on.
const CONFIDENT_Z = 2;

const DESCRIPTION_KEYS = {
    'life and death': 'motifDescLifeAndDeath',
    'semeai':         'motifDescSemeai',
    'ko':             'motifDescKo',
    'sente':          'motifDescSente',
    'tenuki':         'motifDescTenuki',
    'yose':           'motifDescYose',
    'opening':        'motifDescOpening',
};

function descriptionFor(component, side) {
    if (!component) return null;
    const canonical = side === 'negative' ? component.negative : component.positive;
    const key = DESCRIPTION_KEYS[canonical];
    return key ? getTerm(key) : null;
}

const fmtRanks = (r) => Math.abs(r).toFixed(2);
const fmtPct = (p) => String(Math.round(p * 100));

/**
 * Text for a motif hint. `kind` is 'strength', 'tendency' or 'problem';
 * `reading` is the oriented reading for that kind (orientReadings), or null.
 */
export function explainMotif({ component, side = 'positive', name, kind, reading }) {
    const parts = [];
    const desc = descriptionFor(component, side);
    if (desc) parts.push(desc);

    if (kind === 'strength' || kind === 'tendency') {
        const r = reading;
        if (!r || !Number.isFinite(r.z)) {
            parts.push(getTerm('motifExplainNoData', { motif: name }));
        } else if (!isFlagged(r)) {
            parts.push(kind === 'strength'
                ? getTerm('motifExplainStrengthUnresolved', { motif: name, ranks: fmtRanks(r.ranks) })
                : getTerm('motifExplainTendencyUnresolved',
                          { motif: name, share: fmtPct(r.share), expected: fmtPct(r.nullShare) }));
        } else {
            if (kind === 'strength') {
                parts.push(getTerm(r.z < 0 ? 'motifExplainWeak' : 'motifExplainStrong',
                                   { motif: name, ranks: fmtRanks(r.ranks) }));
            } else {
                parts.push(getTerm(r.z < 0 ? 'motifExplainUnder' : 'motifExplainOver',
                                   { motif: name, share: fmtPct(r.share), expected: fmtPct(r.nullShare) }));
            }
            if (Math.abs(r.z) <= CONFIDENT_Z) parts.push(getTerm('motifExplainTentative'));
        }
    }
    return parts.join('\n\n');
}

