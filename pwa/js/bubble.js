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
const ARROW_H = 14;   // clearance reserved for the arrow between the anchor and the bubble
// Half the arrow's span ALONG the bubble's edge: the rotated square's diagonal,
// halved. Plus the slack that keeps its base off the body's rounded corners,
// which it would otherwise nick — it is painted in FRONT of the body.
const ARROW_HALF = 17;
const ARROW_CORNER_SLACK = 8;

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
 * the anchor. `placement` ('above' | 'below' | 'auto') is a PREFERENCE for the
 * VERTICAL side and nothing more: the placer tries the preferred side, its
 * flip, and then the LEFT and RIGHT of the anchor, taking the first that lets
 * the whole bubble sit CLEAR of the anchor. Only if none does is the preference
 * honoured anyway and the bubble clamped into the viewport, where it may land
 * on its own anchor. This is why a tall bubble beside a control in a side
 * column goes out across the free space rather than up into the top margin.
 * `overAnchor` treats the anchor's CENTRE as its edge, which puts the bubble on
 * top of a large target (the board) pointing inward; a target one deliberately
 * covers has no "beside" worth having, so that mode stays vertical.
 *
 * The bubble must already be in the document: its own width and height are
 * measured here.
 */
export function placeBubble(bubble, rect, { placement = 'auto', overAnchor = false } = {}) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const anchorX = rect.left + rect.width / 2;
    const anchorY = rect.top + rect.height / 2;

    const bw = Math.min(bubble.offsetWidth, vw - 2 * MARGIN);
    const bh = bubble.offsetHeight;

    const edgeTop = overAnchor ? anchorY : rect.top;
    const edgeBottom = overAnchor ? anchorY : rect.bottom;
    const edgeLeft = overAnchor ? anchorX : rect.left;
    const edgeRight = overAnchor ? anchorX : rect.right;

    // What the arrow + its clearance cost on the side the bubble stands out on.
    const OUT = GAP + ARROW_H;
    const room = {
        below: (vh - MARGIN) - (edgeBottom + OUT),
        above: (edgeTop - OUT) - MARGIN,
        right: (vw - MARGIN) - (edgeRight + OUT),
        left: (edgeLeft - OUT) - MARGIN,
    };
    const fits = (side) => (side === 'below' || side === 'above')
        ? room[side] >= bh
        // Beside the anchor the bubble is centred on it, so its full height has
        // to fit the viewport as well as its width fitting the gap.
        : !overAnchor && room[side] >= bw && bh <= vh - 2 * MARGIN;

    const vertical = (placement === 'below' || (placement === 'auto' && fits('below')))
        ? ['below', 'above'] : ['above', 'below'];
    const horizontal = room.right >= room.left ? ['right', 'left'] : ['left', 'right'];
    const order = [...vertical, ...horizontal];
    const side = order.find(fits) || vertical[0];
    const beside = side === 'left' || side === 'right';

    let left, top;
    if (beside) {
        left = side === 'right' ? (edgeRight + OUT) : (edgeLeft - OUT - bw);
        top = anchorY - bh / 2;
    } else {
        left = anchorX - bw / 2;
        top = side === 'below' ? (edgeBottom + OUT) : (edgeTop - OUT - bh);
    }
    top = Math.max(MARGIN, Math.min(top, vh - MARGIN - bh));
    left = Math.max(MARGIN, Math.min(left, vw - MARGIN - bw));

    bubble.style.left = `${Math.round(left)}px`;
    bubble.style.top = `${Math.round(top)}px`;
    bubble.style.maxWidth = `${Math.round(vw - 2 * MARGIN)}px`;

    // Arrow: on the bubble edge nearest the anchor, and along that edge it
    // tracks the anchor's centre — but never past the bubble's own rounded
    // corners, so a bubble shoved by the viewport clamp still points as close
    // as it can. The CSS class fixes the edge and the inline coordinate the
    // position along it, so the other coordinate is cleared rather than left
    // over from a previous placement.
    const arrow = bubble.querySelector('.ui-bubble-arrow');
    if (beside) {
        arrow.style.left = '';
        arrow.style.top = `${Math.round(clampAlong(anchorY - top, bh))}px`;
    } else {
        arrow.style.top = '';
        arrow.style.left = `${Math.round(clampAlong(anchorX - left, bw))}px`;
    }
    // Arrow edge = the one facing the anchor, so it is the OPPOSITE of the side
    // the bubble was placed on.
    bubble.classList.toggle('ui-arrow-top', side === 'below');
    bubble.classList.toggle('ui-arrow-bottom', side === 'above');
    bubble.classList.toggle('ui-arrow-left', side === 'right');
    bubble.classList.toggle('ui-arrow-right', side === 'left');
}

// Keep the arrow inside the straight run of an edge, clear of both corners.
function clampAlong(offset, extent) {
    const inset = ARROW_HALF + ARROW_CORNER_SLACK;
    return Math.max(inset, Math.min(offset, extent - inset));
}
