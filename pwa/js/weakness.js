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
//   overplay   your errors reach for the motif beyond what the boards offered
//              -> serve problems whose answer is NOT the motif. That is where
//                 the habit actually costs you, so this is a REDIRECT toward
//                 low-loading problems, not a de-prioritisation of the axis.
//   strong     better than the 3PL predicts on motif-heavy problems
//              -> serve fewer of them: nothing to drill here.
//
// The tendency reading's OTHER sign — errors that carry LESS of the motif than
// the answer did, beyond what the boards offered — is NOT a category. It says
// only that the answer had the motif and the pick did not, which is the deficit
// the `weak` reading already measures (and measures against the rating, where
// this one sees failures alone); the pick itself names nothing the user was
// drawn to, so there is no habit to redirect. Scored out of sample on the
// attempt log, the problems it would serve are ones the user does no worse on
// than usual (auxiliary/serve_criteria_backtest.py). The stats panel still
// draws it; nothing is listed or served on it.
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

// No motif is barred from driving selection. The hazard to watch for if one
// ever seems to deserve it: a category carried by most users at once makes the
// served pool homogeneous and, being near-constant, is collinear with the
// intercept, so the deficit it describes launders into the rank estimate
// instead of steering anything. A fixed 50% tendency null realised exactly
// that on `tenuki` (flagged for ~94% of users); the availability null
// attributes that universality to the problem MENU — the answer to a tsumego
// is by construction not a tenuki, so nearly every alternative loads higher —
// and with the menu subtracted the flag rates are ordinary, so no category is
// near-universal and there is nothing to bar.

// One {k, sign, tail, kind, z} per (axis, direction) whose reading clears the
// confidence gate. `sign` is the SERVE direction the selector argmaxes, hinged at
// zero: +1 targets high loadings on the named tail, -1 targets low ones. `tail`
// is the NAMED tail the category is DESCRIBED through, which for an overplay is
// the opposite of its serve direction. `kind` says which reading
// produced it, so a consumer can treat them differently — the training-focus
// bubble shows the ones that are training directions and drops `strong`.
// This is the reporting list, and selection draws from all of it — see
// weaknessServeCategories below.
export function flaggedMotifCategories() {
    const labels = state.embeddingLabels;
    if (!labels) return [];
    const stats = computeMotifStats();
    if (!stats) return [];

    const categories = [];
    for (const { k, tendency, strength } of stats) {
        const comp = labels[String(k)];
        if (!comp) continue;
        // Describe the axis from whichever tail carries a name, preferring the
        // positive one, and mirror both readings so "weak" and "overplay" keep
        // meaning the named motif throughout.
        const tail = comp.positive ? 'positive' : (comp.negative ? 'negative' : null);
        if (!tail) continue;
        const flip = tail === 'negative';
        const dir = flip ? -1 : 1;

        if (strength && isFlagged(strength)) {
            const z = dir * strength.z;
            // Worse than predicted on motif-heavy problems -> more of them.
            if (z < 0) categories.push({ k, sign: dir, tail, kind: 'weak', z: -z });
            // Better than predicted -> fewer of them.
            else categories.push({ k, sign: -dir, tail, kind: 'strong', z });
        }
        if (tendency && isFlagged(tendency)) {
            const z = dir * tendency.z;
            // Errors reach FOR the motif -> serve where it is not the answer.
            // The opposite lean is deliberately no category (see the header).
            if (z > 0) categories.push({ k, sign: -dir, tail, kind: 'overplay', z });
        }
    }
    return categories;
}

// The categories SELECTION draws from. Today this is the flagged list whole;
// the separate name marks the seam where a serve-side filter would live, so a
// future bar on a category changes one function without touching reporting.
export function weaknessServeCategories() {
    return flaggedMotifCategories();
}
