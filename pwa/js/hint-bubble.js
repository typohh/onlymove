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
// The pointerdown that closes it is absorbed only when it landed ON the bubble;
// see onOutside for why that one case cannot be allowed through.
//
// A hint's content is a sentence, or a Node where the reading is a small table
// (the rank translation). Either way it is resolved AT OPEN, and a resolver
// that returns nothing DECLINES: the anchor stays inert for that moment rather
// than opening an empty bubble. That is what lets a permanently-wired control
// carry a hint that only applies in one app mode.
//
// The bubble itself is js/bubble.js, shared with the onboarding tips; styling is
// the `ui-bubble` + `.hint-btn` blocks in style.css.

import { createBubble, placeBubble, absorbDismissTap } from './bubble.js';

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
    // A press ON the bubble is absorbed: closing it here detaches the press's
    // target, so the click the browser synthesizes afterwards would land on
    // whatever the bubble was covering — for a hint opened from the motif
    // bubble in portrait, that is the board, and the tap that dismissed the
    // explanation would play a move. Presses ANYWHERE ELSE keep falling
    // through: the hint was asked for rather than imposed, and eating them
    // would cost a click on every button and every other hint anchor while one
    // is open (including one merely hovered open, which the user never
    // deliberately raised).
    if (openEl && e.target instanceof Node && openEl.contains(e.target)) {
        closeHint();
        absorbDismissTap(e);
        return;
    }
    closeHint();
}

function onKey(e) {
    if (e.key === 'Escape') closeHint();
}

function openHint(button, text, passthrough) {
    closeHint();
    const bubble = createBubble(text, { topLayer: true });
    // A hover-opened bubble must not take the pointer. The placer looks for a
    // side with room to sit clear — beside the anchor as well as above and
    // below it — but where none has any it clamps into the viewport and lands
    // ON its own anchor, and a painted surface appearing under a
    // stationary cursor re-hit-tests, firing pointerleave on the anchor, which
    // closes it, which puts the cursor back on the anchor, which reopens it.
    // The hint then flickers and never settles, on exactly the longest
    // explanations. Pointer-transparent until PINNED, which costs nothing: a
    // tap always pins (touch has no hover), so the tap-absorption the painted
    // parts exist for is untouched.
    bubble.classList.toggle('ui-bubble-passthrough', !!passthrough);
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
 * Make `anchor` (any element) open a hint bubble with `content` — a string or a
 * Node, or a function returning one, resolved each time the bubble opens so a
 * reading that is costly or that changes with state is computed on demand
 * rather than at render. A resolver returning null, undefined or '' declines,
 * and nothing opens. The anchor keeps its own content; this wires the
 * interaction only. A string doubles as the accessible name; anything else
 * leaves the anchor to be named by its own content, which is what a hint ON a
 * readout wants — the readout's value is the name, in every mode.
 */
export function hintAnchor(anchor, content) {
    const resolve = () => (typeof content === 'function' ? content() : content);
    anchor.setAttribute('role', 'button');
    anchor.setAttribute('tabindex', '0');
    if (typeof content === 'string') anchor.setAttribute('aria-label', content);

    const toggle = () => {
        if (openFor === anchor && pinned) { closeHint(); return; }
        // Already up from hover: pin it in place rather than tearing it down and
        // rebuilding it, which would re-run the transition under the pointer.
        if (openFor === anchor && openEl) {
            openEl.classList.remove('ui-bubble-passthrough');
            pinned = true;
            return;
        }
        const resolved = resolve();
        if (!resolved) { closeHint(); return; }
        openHint(anchor, resolved, false);
        pinned = true;
    };

    anchor.addEventListener('click', (e) => {
        // The hint sits inside dialogs and rows that have their own click
        // handling; asking for an explanation must never also pick the thing.
        e.preventDefault();
        e.stopPropagation();
        toggle();
    });
    anchor.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
    });
    anchor.addEventListener('pointerenter', () => {
        if (!canHover() || openFor === anchor) return;
        const resolved = resolve();
        if (!resolved) return;
        openHint(anchor, resolved, true);
        pinned = false;
    });
    anchor.addEventListener('pointerleave', () => {
        if (canHover() && openFor === anchor && !pinned) closeHint();
    });
    return anchor;
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
    return hintAnchor(button, text);
}
