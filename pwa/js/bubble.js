// js/bubble.js
//
// The app's speech bubble: a small floating panel with an arrow pointing at
// whatever anchored it. Two callers, deliberately sharing one implementation —
// the first-run onboarding tips (onboarding.js) and the hint affordances that
// carry a control's explanation out of the layout (hint-bubble.js). Anything
// that needs to say a sentence about a thing on screen should use this rather
// than growing a third bubble.
//
// This module owns the DOM shape and the placement arithmetic and NOTHING else:
// no timing, no gating, no dismissal policy, no anchor registry. Those differ
// completely between a tip that appears unbidden and a hint the user asked for,
// and mixing them is what would make this un-reusable. Styling is the `ui-bubble`
// block in style.css.

const MARGIN = 8;    // keep the bubble this far inside the viewport
const GAP = 10;      // clearance between the anchor edge and the arrow tip
const ARROW_H = 14;  // protrusion of the rotated-square arrow (see .ui-bubble-arrow)
const ARROW_HALF = 14;

/**
 * Build a bubble carrying `text`. Returned detached and invisible: append it,
 * place it, then add `ui-bubble-visible` (on the next frame, so the transition
 * runs). `topLayer` raises it above modals — right for a hint the user opened
 * from inside one, wrong for an ambient tip that must not cover a dialog.
 */
export function createBubble(text, { topLayer = false } = {}) {
    const bubble = document.createElement('div');
    bubble.className = topLayer ? 'ui-bubble ui-bubble-top' : 'ui-bubble';
    const body = document.createElement('div');
    body.className = 'ui-bubble-body';
    body.textContent = text;
    const arrow = document.createElement('div');
    arrow.className = 'ui-bubble-arrow';
    bubble.appendChild(body);
    bubble.appendChild(arrow);
    return bubble;
}

/**
 * Place `bubble` against `rect` (a viewport-space DOMRect), arrow pointing at
 * the anchor's horizontal centre. `placement` ('above' | 'below' | 'auto') is a
 * PREFERENCE — it flips whenever the preferred side would run off screen, so a
 * caller never has to know how much room there is. `overAnchor` treats the
 * anchor's CENTRE as the edge, which puts the bubble on top of a large target
 * (the board) pointing inward instead of floating clear of it.
 *
 * The bubble must already be in the document: its own width and height are
 * measured here.
 */
export function placeBubble(bubble, rect, { placement = 'auto', overAnchor = false } = {}) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const anchorX = rect.left + rect.width / 2;
    const centreY = rect.top + rect.height / 2;

    const bw = Math.min(bubble.offsetWidth, vw - 2 * MARGIN);
    const bh = bubble.offsetHeight;

    const edgeTop = overAnchor ? centreY : rect.top;
    const edgeBottom = overAnchor ? centreY : rect.bottom;
    let below;
    if (placement === 'below') below = true;
    else if (placement === 'above') below = false;
    else below = (edgeBottom + GAP + bh + ARROW_H) <= (vh - MARGIN);
    // Flip if the preferred side overflows and the other one doesn't.
    if (below && (edgeBottom + GAP + bh + ARROW_H) > vh - MARGIN && (edgeTop - GAP - bh - ARROW_H) >= MARGIN) below = false;
    if (!below && (edgeTop - GAP - bh - ARROW_H) < MARGIN && (edgeBottom + GAP + bh + ARROW_H) <= vh - MARGIN) below = true;

    let top = below ? (edgeBottom + GAP + ARROW_H) : (edgeTop - GAP - ARROW_H - bh);
    top = Math.max(MARGIN, Math.min(top, vh - MARGIN - bh));

    let left = anchorX - bw / 2;
    left = Math.max(MARGIN, Math.min(left, vw - MARGIN - bw));

    bubble.style.left = `${Math.round(left)}px`;
    bubble.style.top = `${Math.round(top)}px`;
    bubble.style.maxWidth = `${Math.round(vw - 2 * MARGIN)}px`;

    // Arrow: on the edge nearest the anchor, horizontally under the anchor
    // centre — but never past the bubble's own rounded corners, so a bubble
    // shoved sideways by the viewport clamp still points as close as it can.
    const arrow = bubble.querySelector('.ui-bubble-arrow');
    let arrowLeft = anchorX - left;
    arrowLeft = Math.max(ARROW_HALF + 4, Math.min(arrowLeft, bw - ARROW_HALF - 4));
    arrow.style.left = `${Math.round(arrowLeft)}px`;
    bubble.classList.toggle('ui-arrow-top', below);       // arrow on the bubble's TOP edge when it sits below
    bubble.classList.toggle('ui-arrow-bottom', !below);
}
