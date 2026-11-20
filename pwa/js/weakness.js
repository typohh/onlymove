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
//   strong     better than the 3PL predicts on motif-heavy problems
//              -> serve fewer of them: nothing to drill here.
//
// Only the STRENGTH reading makes a category. The TENDENCY reading — which way
// the user's errors lean against what the boards offered — makes none in either
// direction: replayed over the attempt log out of sample, the problems either
// lean would have served are ones the user does no worse on than on any others,
// where a `weak` flag finds problems they do measurably worse on
// (ablations.motif_serve_figures). The stats panel draws the lean; nothing is
// listed or served on it.
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
// instead of steering anything. The tendency readings are where that lives
// (`tenuki`: the answer to a tsumego is by construction not one, so a lean flag
// there is carried by most users), and they make no category.

// One {k, sign, tail, kind, z} per (axis, direction) whose reading clears the
// confidence gate. `sign` is the SERVE direction the selector argmaxes, hinged at
// zero: +1 targets high loadings on the named tail, -1 targets low ones. `tail`
// is the NAMED tail the category is DESCRIBED through. `kind` is `weak` or
// `strong`, so a consumer can treat them differently — the training-focus
// bubble shows the ones that are training directions and drops `strong`.
// This is the reporting list, and selection draws from all of it — see
// weaknessServeCategories below.
export function flaggedMotifCategories() {
    const labels = state.embeddingLabels;
    if (!labels) return [];
    const stats = computeMotifStats();
    if (!stats) return [];

    const categories = [];
    for (const { k, strength } of stats) {
        const comp = labels[String(k)];
        if (!comp) continue;
        // Describe the axis from whichever tail carries a name, preferring the
        // positive one, and mirror the reading so "weak" keeps
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
    }
    return categories;
}

// The categories SELECTION draws from. Today this is the flagged list whole;
// the separate name marks the seam where a serve-side filter would live, so a
// future bar on a category changes one function without touching reporting.
export function weaknessServeCategories() {
    return flaggedMotifCategories();
}
