// js/weakness.js
//
// The flagged SERVE CATEGORIES — which motifs selection should tilt toward, and
// in which direction. Everything here reads motif-stats.js, whose readings are
// computed from the user's own attempts and their problems' own geometry
// alone; no population centre, no fitted prior, and therefore none of the rank
// confound a population centre carries (motif prevalence varies several-fold
// across the ladder, so a centre averaged over users partly encodes which
// problems that user's rank is served).
//
// A problem's motif is the loading of its CORRECT move, which is what makes the
// serve directions distinct:
//
//   weak       worse than the 3PL predicts on motif-heavy problems
//              -> serve MORE of them: drill the deficit.
//   underplay  your errors miss the motif beyond what the boards offered
//              -> serve MORE of them: drill recognising when it applies.
//   overplay   your errors reach for the motif beyond what the boards offered
//              -> serve problems whose answer is NOT the motif. That is where
//                 the habit actually costs you, so this is a REDIRECT toward
//                 low-loading problems, not a de-prioritisation of the axis.
//   condition  your errors lean toward the motif where it is NOT the answer
//              AND away from it where it IS (the judgment reading's positive
//              slope — both directional deficits at once, which the tendency
//              share alone cannot represent)
//              -> serve BOTH ends: two slots, one drilling recognition where
//                 it applies, one drilling restraint where it does not.
//   strong     better than the 3PL predicts on motif-heavy problems
//              -> serve fewer of them: nothing to drill here.
//   (a NEGATIVE judgment slope — errors tracking the condition better than a
//   typical error — is, like `strong`, nothing to drill, and emits no
//   category.)
//
// The gate is CONFIDENCE ONLY (|z| > MOTIF_Z), with no minimum effect size: a
// bias the user is reliably making is worth acting on however small it is.
//
// Only a NAMED tail can be a category — an unnamed tail is untaught residual
// that the report tab is structurally unable to describe, so acting on it would
// target a campaign we cannot explain. The test is on the label EXISTING, not on
// its localization, so selection is locale-independent.

import { state } from './state.js';
import { computeMotifStats, isFlagged } from './motif-stats.js';

// Motifs barred from driving SELECTION, by their untranslated positive label.
//
// `tenuki` is excluded because under the FIXED 50% null it flagged for nearly
// every user, and a category almost everyone carries makes the served pool
// homogeneous and, being near-constant, is collinear with the intercept, so
// the deficit it describes launders into the rank estimate instead of steering
// anything. The availability null has since shown most of that universality
// was the MENU, not the players — the answer to a tsumego is by construction
// not a tenuki, so nearly every alternative loads higher — and under the
// corrected test the flag rate drops to ordinary levels. The exclusion stays
// until the flag-rate homogeneity argument is re-measured under the corrected
// null; reporting was never affected either way.
const SELECTION_EXCLUDED = new Set(['tenuki']);

// One {k, sign, tail, kind, z} per (axis, direction) whose reading clears the
// confidence gate. `sign` is the SERVE direction the selector argmaxes, hinged at
// zero: +1 targets high loadings on the named tail, -1 targets low ones. `tail`
// is the NAMED tail the category is DESCRIBED through, which for an overplay is
// the opposite of its serve direction. `kind` says which of the four readings
// produced it, so a consumer can treat them differently — the training-focus
// bubble shows the three that are training directions and drops `strong`.
// Every flagged (axis, direction), INCLUDING the selection-excluded motifs. This
// is the reporting list: an excluded motif is barred from steering the pool, not
// from being told to the user — the exclusion is about a category almost everyone
// carries being useless as a serve tilt, which says nothing about whether the
// finding is worth reading.
export function flaggedMotifCategories() {
    const labels = state.embeddingLabels;
    if (!labels) return [];
    const stats = computeMotifStats();
    if (!stats) return [];

    const categories = [];
    for (const { k, tendency, condition, strength } of stats) {
        const comp = labels[String(k)];
        if (!comp) continue;
        // Describe the axis from whichever tail carries a name, preferring the
        // positive one, and mirror both readings so "weak" and "overplay" keep
        // meaning the named motif throughout.
        const tail = comp.positive ? 'positive' : (comp.negative ? 'negative' : null);
        if (!tail) continue;
        const flip = tail === 'negative';
        const dir = flip ? -1 : 1;
        const excluded = SELECTION_EXCLUDED.has(comp.positive) ||
                         SELECTION_EXCLUDED.has(comp.negative);

        if (strength && isFlagged(strength)) {
            const z = dir * strength.z;
            // Worse than predicted on motif-heavy problems -> more of them.
            if (z < 0) categories.push({ k, sign: dir, tail, kind: 'weak', excluded, z: -z });
            // Better than predicted -> fewer of them.
            else categories.push({ k, sign: -dir, tail, kind: 'strong', z });
        }
        if (tendency && isFlagged(tendency)) {
            const z = dir * tendency.z;
            // Errors reach FOR the motif -> serve where it is not the answer.
            if (z > 0) categories.push({ k, sign: -dir, tail, kind: 'overplay', z });
            // Errors MISS the motif -> serve where it is the answer.
            else categories.push({ k, sign: dir, tail, kind: 'underplay', excluded, z: -z });
        }
        // The judgment slope is mirror-invariant (motif-stats.js), so no dir
        // factor: positive is the deficit whichever tail names the axis. It is
        // two-sided by construction, so it takes two slots — one per end. The
        // motif-answer slot carries the exclusion (it would serve MORE of the
        // barred motif); the anti-motif slot serves the norm and does not,
        // matching how overplay escapes the exclusion.
        if (condition && isFlagged(condition) && condition.z > 0) {
            categories.push({ k, sign: dir, tail, kind: 'condition', excluded, z: condition.z });
            categories.push({ k, sign: -dir, tail, kind: 'condition', z: condition.z });
        }
    }
    return categories;
}

// The categories SELECTION may draw from — the flagged set minus the motifs
// barred from steering the pool.
export function weaknessServeCategories() {
    return flaggedMotifCategories().filter(c => !c.excluded);
}
