// js/hint-bubble.js
//
// The hint affordance: a small "?" beside a control, carrying the sentence that
// would otherwise have to sit in the layout. It exists so a dialog can be a list
// of choices rather than a list of choices interleaved with paragraphs — the
// explanation is one tap away for whoever wants it and invisible to everyone
// else.
//
// One interaction, two idioms, no device sniffing: a CLICK always toggles (which
// is the whole story on a touchscreen, and the keyboard story everywhere), and
// where the browser reports a real hovering pointer, hover opens and closes it
// too. A click PINS the bubble open so moving the pointer away doesn't take back
// what the user deliberately asked for.
//
// At most one hint is open at a time, app-wide. It closes on Escape, on a
// pointerdown anywhere else, and on a resize or scroll — the last because the
// bubble is placed once against the anchor's position and does not follow it.
//
// The bubble itself is js/bubble.js, shared with the onboarding tips; styling is
// the `ui-bubble` + `.hint-btn` blocks in style.css.

import { createBubble, placeBubble } from './bubble.js';

// True where the primary input can hover (a mouse or trackpad). Coarse-pointer
// devices report false, and so do hybrids being driven by touch.
const canHover = () => window.matchMedia('(hover: hover)').matches;

let openFor = null;    // the "?" whose bubble is up, if any
let openEl = null;     // that bubble
let pinned = false;    // opened by click -> hover-out must not close it

export function closeHint() {
    if (openEl) openEl.remove();
    openEl = null;
    openFor = null;
    pinned = false;
    document.removeEventListener('pointerdown', onOutside, true);
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('resize', closeHint);
    window.removeEventListener('scroll', closeHint, true);
}

function onOutside(e) {
    // A press on the button itself is its own toggle; anything else dismisses.
    if (openFor && openFor.contains(e.target)) return;
    closeHint();
}

function onKey(e) {
    if (e.key === 'Escape') closeHint();
}

function openHint(button, text) {
    closeHint();
    const bubble = createBubble(text, { topLayer: true });
    document.body.appendChild(bubble);
    placeBubble(bubble, button.getBoundingClientRect());
    requestAnimationFrame(() => bubble.classList.add('ui-bubble-visible'));
    openFor = button;
    openEl = bubble;
    document.addEventListener('pointerdown', onOutside, true);
    document.addEventListener('keydown', onKey, true);
    window.addEventListener('resize', closeHint);
    // Capture phase: a scroll inside any container should close it too, and
    // scroll events on an element don't bubble.
    window.addEventListener('scroll', closeHint, true);
}

/**
 * A "?" button that shows `text` in a bubble. Drop it next to any control whose
 * meaning needs a sentence. The text doubles as the accessible name, so a screen
 * reader gets the explanation from the button without opening anything.
 */
export function hintButton(text) {
    const button = document.createElement('span');
    button.className = 'hint-btn';
    button.textContent = '?';
    button.setAttribute('role', 'button');
    button.setAttribute('tabindex', '0');
    button.setAttribute('aria-label', text);

    const toggle = () => {
        if (openFor === button && pinned) { closeHint(); return; }
        openHint(button, text);
        pinned = true;
    };

    button.addEventListener('click', (e) => {
        // The hint sits inside dialogs and rows that have their own click
        // handling; asking for an explanation must never also pick the thing.
        e.preventDefault();
        e.stopPropagation();
        toggle();
    });
    button.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
    });
    button.addEventListener('pointerenter', () => {
        if (canHover() && openFor !== button) { openHint(button, text); pinned = false; }
    });
    button.addEventListener('pointerleave', () => {
        if (canHover() && openFor === button && !pinned) closeHint();
    });
    return button;
}
