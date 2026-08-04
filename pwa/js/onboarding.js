// js/onboarding.js
//
// First-run onboarding bubbles. Small, non-blocking speech bubbles that appear
// ONE AT A TIME, drip-fed as the user meets each part of the UI — most fire off
// a concrete trigger (first review, a recall serve, the score estimate blinking
// red), a few "ambient" ones (front-page rank, komi) are paced roughly one per
// few problems. Each fires ONCE EVER (a localStorage flag), never two at once,
// and a rapid-dismiss streak backs the whole system off for the session.
//
// Deliberately isolated: this is the ONLY file that knows about the bubbles.
// The rest of the app calls the thin trigger hooks below (onSolvingProblemShown,
// onEnterReview, onReviewScoreShown, onScoreUnreliable, onResume) at existing UI
// seams; everything else — anchoring, arrow, gating, session and
// progress counting — lives here. Styling is the `onb-*` block in style.css.
//
// The whole system self-disables for anyone who isn't a genuinely new user when
// it first runs (see initOnboarding): existing players never get "Black to move".

import { state, getMergedAnswerHistory } from './state.js';
import { getTerm } from './i18n.js';
import { APP_MODES, HISTORY_BACKFILL_DONE_KEY, usesTapToConfirm } from './constants.js';
import { supportsNotifications, notificationPermission, getCurrentSubscription } from './notifications.js';

// --- tunables ---------------------------------------------------------------

// A device with more than this many recorded attempts when onboarding first
// runs is treated as an existing user and never shown any bubble.
const NEW_USER_MAX_HISTORY = 3;
// Two boots more than this far apart count as separate "sessions" (drives the
// install / notify bubbles, which wait for the 2nd / 3rd visit). A
// service-worker-update reload lands well inside the gap, so it never counts.
const SESSION_GAP_MS = 2 * 60 * 60 * 1000;   // 2 hours
// Minimum wall-clock gap between any two bubbles — pure anti-stacking, so a
// burst of triggers (answer wrong → review opens → score shows) can't pile up.
const MIN_GAP_MS = 15 * 1000;
// Ambient bubbles additionally wait this many completed problems since the last
// bubble of any kind — the "roughly one every few problems" drip.
const SPACING_PROBLEMS = 3;
// At most this many bubbles in one session, whatever the triggers.
const SESSION_CAP = 4;
// A dismiss within this window of a bubble appearing reads as "swatted away".
const QUICK_DISMISS_MS = 3500;
// This many swats in a row silences the rest of the session.
const BACKOFF_QUICK_DISMISSALS = 3;
// A shown bubble self-dismisses after this long even if untouched.
const AUTO_DISMISS_MS = 14 * 1000;
// Delay from a trigger to the bubble actually painting; the overlay-clear poll
// (see scheduleShow) extends this past any transition fade that's still up.
const SHOW_DELAY_MS = 450;

// --- persisted flags --------------------------------------------------------

const SHOWN_PREFIX = 'onbShown_';
const K_INITIALIZED = 'onbInitialized';
const K_SUPPRESSED = 'onbSuppressed';           // set for existing users → never show
const K_SESSION_COUNT = 'onbSessionCount';
const K_LAST_ACTIVE = 'onbLastActiveAt';
const K_LAST_SHOWN_AT = 'onbLastShownAt';
const K_LAST_AMBIENT_COUNT = 'onbLastAmbientCount';

const lsGet = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, String(v)); } catch {} };
const lsNum = (k) => { const v = parseInt(lsGet(k), 10); return Number.isFinite(v) ? v : 0; };

const wasShown = (id) => !!lsGet(SHOWN_PREFIX + id);
const markShown = (id) => lsSet(SHOWN_PREFIX + id, '1');

// --- runtime state ----------------------------------------------------------

let _ready = false;              // init() has run
let _suppressed = true;          // until init() decides otherwise
let _sessionCount = 1;
let _shownThisSession = 0;
let _quickDismissals = 0;
let _backedOff = false;
let _pending = false;            // a bubble is scheduled or on screen
let _pendingTimer = 0;
let _shownAtMs = 0;              // when the current bubble painted (for swat detection)

const historyLen = () => {
    try { return getMergedAnswerHistory().length; } catch { return 0; }
};

const isInstalled = () =>
    window.matchMedia('(display-mode: standalone)').matches ||
    window.navigator.standalone === true;

// notify-bubble gate reads a live push subscription; cached so the async probe
// doesn't gate the synchronous eligibility check.
let _subscribed = false;
async function refreshSubscribed() {
    try { _subscribed = supportsNotifications() ? !!(await getCurrentSubscription()) : false; }
    catch { _subscribed = false; }
}

// --- element helpers ---------------------------------------------------------

const el = (id) => document.getElementById(id);

// The header is duplicated for portrait (#title/#text1/#capture-score) and
// landscape (#…-h); exactly one set is rendered per orientation. Return the
// visible one.
function firstVisible(ids) {
    for (const id of ids) {
        const node = el(id);
        if (!node) continue;
        const r = node.getBoundingClientRect();
        if (r.width > 0 && r.height > 0 && node.offsetParent !== null) return node;
    }
    return null;
}

const anchors = {
    rank: () => firstVisible(['text1', 'text1-h']),
    komi: () => firstVisible(['capture-score', 'capture-score-h']),
    title: () => firstVisible(['title', 'title-h']),
    unreliableScore: () => document.querySelector('.unreliable'),
    board: () => el('board-container'),
    previous: () => el('button1'),
    share: () => el('button-share'),
    install: () => el('button4'),
    // The hamburger button itself (portrait ☰ / landscape "Menu"): the
    // menu-item bubbles point here and tell the user to open it, rather than
    // firing only once the menu is already open.
    menu: () => firstVisible(['menu-toggle-h', 'menu-toggle']),
};

const isVisible = (node) => {
    if (!node) return false;
    const r = node.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && node.offsetParent !== null;
};

// --- bubble catalogue --------------------------------------------------------
//
// kind: 'trigger' fires when its context appears (subject only to the anti-stack
// time gap); 'ambient' additionally waits SPACING_PROBLEMS. `anchor` names an
// entry in `anchors`; 'board' places the bubble over the board pointing at its
// centre. `eligible(ctx)` is the context gate on top of the fire-once flag.
// `text` is an i18n key, or a function returning one when the wording depends
// on the device (resolved at render time, not module-eval).

const BUBBLES = {
    // The one bubble that describes the input itself, so it has to match the
    // branch board.js takes: two taps to play on touch, one click on desktop.
    intro: {
        kind: 'trigger', anchor: 'board', placement: 'above',
        text: () => (usesTapToConfirm() ? 'onbIntroTouch' : 'onbIntro'),
        eligible: () => historyLen() <= 1,
    },
    recall: {
        kind: 'trigger', anchor: 'title', placement: 'below', text: 'onbRecall',
        eligible: (ctx) => !!ctx.isRepeat,
    },
    // Encouragement after a rough start: the 3rd miss within the first 10
    // attempts. Fired from the review entry a wrong answer auto-opens (the
    // failed attempt is already in history by then) and re-attempted on the
    // next problem if a gate held it back there.
    earlyMistakes: {
        kind: 'trigger', anchor: 'board', placement: 'above', text: 'onbEarlyMistakes',
        eligible: () => {
            const h = getMergedAnswerHistory();
            return h.length <= 10 && h.filter(e => !e.correct).length >= 3;
        },
    },
    // Teaches the Previous button. Ambient and gated past the early drip so it
    // doesn't compete with the first-problems bubbles. A user who has already
    // clicked Previous themselves never sees it — onPreviousClicked marks it
    // shown. Having been in review mode does NOT count (a wrong answer
    // auto-enters review without the button, teaching nothing about it).
    previous: {
        kind: 'ambient', anchor: 'previous', placement: 'below', text: 'onbPrevious',
        eligible: () => historyLen() >= 8 &&
            !!(state.previousProblem && state.previousProblem.sgf) &&
            isVisible(anchors.previous()),
    },
    rank: {
        kind: 'ambient', anchor: 'rank', placement: 'below', text: 'onbRank',
        eligible: () => historyLen() >= 5,
    },
    komi: {
        kind: 'ambient', anchor: 'komi', placement: 'below', text: 'onbKomi',
        eligible: () => historyLen() >= 20,
    },
    install: {
        kind: 'ambient', anchor: 'install', placement: 'below', text: 'onbInstall',
        eligible: () => _sessionCount >= 2 && !isInstalled() && isVisible(anchors.install()),
    },
    notify: {
        kind: 'ambient', anchor: 'install', placement: 'below', text: 'onbNotify',
        eligible: () => _sessionCount >= 3 && isInstalled() &&
            notificationPermission() !== 'denied' && !_subscribed && isVisible(anchors.install()),
    },
    // Review-mode, at the root position.
    reviewIntro: {
        kind: 'trigger', anchor: 'board', placement: 'above', text: 'onbReviewIntro',
        eligible: () => state.appMode === APP_MODES.ANALYZING,
    },
    reviewRank: {
        kind: 'trigger', anchor: 'rank', placement: 'below', text: 'onbReviewRank',
        eligible: () => state.appMode === APP_MODES.ANALYZING && wasShown('reviewIntro'),
    },
    share: {
        kind: 'trigger', anchor: 'share', placement: 'below', text: 'onbShare',
        eligible: () => state.appMode === APP_MODES.ANALYZING && wasShown('reviewIntro') &&
            isVisible(anchors.share()),
    },
    // Review-mode, after stepping into a variation (the score readout is live).
    reviewScore: {
        kind: 'trigger', anchor: 'rank', placement: 'below', text: 'onbReviewScore',
        eligible: () => state.appMode === APP_MODES.ANALYZING,
    },
    reviewScoreUnreliable: {
        kind: 'trigger', anchor: 'unreliableScore', placement: 'below', text: 'onbScoreUnreliable',
        eligible: () => state.appMode === APP_MODES.ANALYZING && isVisible(anchors.unreliableScore()),
    },
    // Menu-contained features. We do NOT open the menu for the user (intrusive);
    // instead these point at the Menu button and the text names the item, so the
    // bubble teaches both that the menu exists and what's inside it.
    stats: {
        kind: 'ambient', anchor: 'menu', placement: 'below', text: 'onbStats',
        eligible: () => Array.isArray(state.weaknessParams) && state.weaknessParams.length >= 4 &&
            isVisible(anchors.menu()),
    },
    account: {
        // Cross-device sync only matters to an invested, returning user — few
        // people multi-account right away. Gate it well past the early drip:
        // a returning player (3rd+ session) with real history.
        kind: 'ambient', anchor: 'menu', placement: 'below', text: 'onbAccount',
        eligible: () => historyLen() >= 60 && _sessionCount >= 3 &&
            !!lsGet(HISTORY_BACKFILL_DONE_KEY) && isVisible(anchors.menu()),
    },
    boardSize: {
        kind: 'ambient', anchor: 'menu', placement: 'below', text: 'onbBoardSize',
        eligible: () => (_sessionCount >= 2 || historyLen() >= 12) && isVisible(anchors.menu()),
    },
};

// --- gating ------------------------------------------------------------------

function spacingOk(kind) {
    if (_shownThisSession >= SESSION_CAP) return false;
    const lastShown = lsNum(K_LAST_SHOWN_AT);
    if (lastShown && (Date.now() - lastShown) < MIN_GAP_MS) return false;
    if (kind === 'ambient') {
        const since = historyLen() - lsNum(K_LAST_AMBIENT_COUNT);
        if (since < SPACING_PROBLEMS) return false;
    }
    return true;
}

// Try the given bubble ids in priority order; schedule the first that is
// unshown, contextually eligible, and passes the spacing gate.
function tryShow(ids, ctx = {}) {
    if (!_ready || _suppressed || _backedOff || _pending) return;
    if (document.hidden) return;
    for (const id of ids) {
        const def = BUBBLES[id];
        if (!def || wasShown(id)) continue;
        try { if (!def.eligible(ctx)) continue; } catch { continue; }
        if (!spacingOk(def.kind)) return;   // gate is global — don't fall through to a lower-priority one
        scheduleShow(id, def);
        return;
    }
}

// --- rendering ---------------------------------------------------------------

let _bubbleEl = null;

function removeBubble() {
    if (_pendingTimer) { clearTimeout(_pendingTimer); _pendingTimer = 0; }
    if (_bubbleEl) { _bubbleEl.remove(); _bubbleEl = null; }
    _pending = false;
    document.removeEventListener('pointerdown', onDismiss, true);
    document.removeEventListener('keydown', onDismiss, true);
    window.removeEventListener('resize', onReposition);
}

function onDismiss() {
    if (_shownAtMs && (Date.now() - _shownAtMs) < QUICK_DISMISS_MS) {
        _quickDismissals++;
        if (_quickDismissals >= BACKOFF_QUICK_DISMISSALS) _backedOff = true;
    } else {
        _quickDismissals = 0;   // an unhurried read breaks the swat streak
    }
    removeBubble();
}

let _repositionArgs = null;
function onReposition() {
    if (_bubbleEl && _repositionArgs) position(_bubbleEl, _repositionArgs.def);
}

// A fullscreen transition/loading/resume overlay is up — hold the bubble back
// (the menu overlay is deliberately NOT blocking: menu bubbles need it open).
const overlayBlocking = () =>
    !!document.querySelector('.app-blocking-overlay, .loading-overlay, .resume-overlay');

function scheduleShow(id, def) {
    _pending = true;
    let tries = 0;
    const attempt = () => {
        _pendingTimer = 0;
        // Re-verify: the world may have moved on during the delay.
        if (!_ready || _suppressed || _backedOff) { _pending = false; return; }
        if (document.hidden || overlayBlocking() || !isVisible(anchors.board())) {
            if (tries++ < 30) { _pendingTimer = setTimeout(attempt, 300); return; }
            _pending = false; return;   // gave up waiting for a clear board
        }
        let target;
        try { if (!def.eligible({})) { _pending = false; return; } } catch { _pending = false; return; }
        target = def.anchor === 'board' ? anchors.board() : anchors[def.anchor]();
        if (!isVisible(target)) { _pending = false; return; }
        render(id, def);
    };
    _pendingTimer = setTimeout(attempt, SHOW_DELAY_MS);
}

function render(id, def) {
    const bubble = document.createElement('div');
    bubble.className = 'onb-bubble';
    const body = document.createElement('div');
    body.className = 'onb-bubble-body';
    body.textContent = getTerm(typeof def.text === 'function' ? def.text() : def.text);
    const arrow = document.createElement('div');
    arrow.className = 'onb-bubble-arrow';
    bubble.appendChild(body);
    bubble.appendChild(arrow);
    document.body.appendChild(bubble);
    _bubbleEl = bubble;
    _repositionArgs = { def };

    position(bubble, def);
    requestAnimationFrame(() => bubble.classList.add('onb-visible'));

    // Record + arm dismissal.
    markShown(id);
    lsSet(K_LAST_SHOWN_AT, Date.now());
    if (def.kind === 'ambient') lsSet(K_LAST_AMBIENT_COUNT, historyLen());
    _shownThisSession++;
    _shownAtMs = Date.now();

    // Any tap/keypress dismisses. pointer-events:none on the bubble means the
    // tap that dismisses also reaches whatever's underneath (a board move, a
    // menu item) — exactly the "click anywhere to continue" behaviour wanted.
    document.addEventListener('pointerdown', onDismiss, true);
    document.addEventListener('keydown', onDismiss, true);
    window.addEventListener('resize', onReposition);
    _pendingTimer = setTimeout(onDismiss, AUTO_DISMISS_MS);
}

// Place the bubble relative to the anchor, arrow pointing at the anchor's
// centre. `def.placement` ('above'|'below'|'auto') is a preference; we flip if
// there isn't room. For the board anchor the "anchor point" is its centre, so
// the bubble sits over the board pointing inward.
function position(bubble, def) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const margin = 8;
    const gap = 10;
    const arrowH = 14;   // protrusion of the rotated-square arrow (see .onb-bubble-arrow)

    const target = def.anchor === 'board' ? anchors.board() : anchors[def.anchor]();
    if (!isVisible(target)) return;
    const r = target.getBoundingClientRect();
    const overBoard = def.anchor === 'board';
    const anchorX = r.left + r.width / 2;
    const anchorY = overBoard ? (r.top + r.height / 2) : (r.top + r.height / 2);

    const bw = Math.min(bubble.offsetWidth, vw - 2 * margin);
    const bh = bubble.offsetHeight;

    // Decide above vs below the anchor edge (or centre, for the board).
    const edgeTop = overBoard ? anchorY : r.top;
    const edgeBottom = overBoard ? anchorY : r.bottom;
    let below;
    if (def.placement === 'below') below = true;
    else if (def.placement === 'above') below = false;
    else below = (edgeBottom + gap + bh + arrowH) <= (vh - margin);
    // Flip if the preferred side overflows.
    if (below && (edgeBottom + gap + bh + arrowH) > vh - margin && (edgeTop - gap - bh - arrowH) >= margin) below = false;
    if (!below && (edgeTop - gap - bh - arrowH) < margin && (edgeBottom + gap + bh + arrowH) <= vh - margin) below = true;

    let top = below ? (edgeBottom + gap + arrowH) : (edgeTop - gap - arrowH - bh);
    top = Math.max(margin, Math.min(top, vh - margin - bh));

    let left = anchorX - bw / 2;
    left = Math.max(margin, Math.min(left, vw - margin - bw));

    bubble.style.left = `${Math.round(left)}px`;
    bubble.style.top = `${Math.round(top)}px`;
    bubble.style.maxWidth = `${Math.round(vw - 2 * margin)}px`;

    // Arrow: on the edge nearest the anchor, horizontally under the anchor centre.
    const arrow = bubble.querySelector('.onb-bubble-arrow');
    const arrowHalf = 14;   // half the rotated-square arrow's bounding width
    let arrowLeft = anchorX - left;
    arrowLeft = Math.max(arrowHalf + 4, Math.min(arrowLeft, bw - arrowHalf - 4));
    arrow.style.left = `${Math.round(arrowLeft)}px`;
    bubble.classList.toggle('onb-arrow-top', below);      // arrow on bubble's TOP edge when bubble is below
    bubble.classList.toggle('onb-arrow-bottom', !below);
}

// --- session / init ----------------------------------------------------------

function markActive() { lsSet(K_LAST_ACTIVE, Date.now()); }

export function initOnboarding() {
    // First ever run: decide new-user vs existing-user once.
    if (!lsGet(K_INITIALIZED)) {
        lsSet(K_INITIALIZED, '1');
        if (historyLen() > NEW_USER_MAX_HISTORY) lsSet(K_SUPPRESSED, '1');
        lsSet(K_SESSION_COUNT, '1');
    } else {
        // Returning boot: bump the session counter if we've been away a while.
        const last = lsNum(K_LAST_ACTIVE);
        let count = lsNum(K_SESSION_COUNT) || 1;
        if (last && (Date.now() - last) > SESSION_GAP_MS) count++;
        lsSet(K_SESSION_COUNT, count);
    }
    _sessionCount = lsNum(K_SESSION_COUNT) || 1;
    _suppressed = !!lsGet(K_SUPPRESSED);
    markActive();

    if (_suppressed) return;   // existing user — stay silent forever
    _ready = true;

    // Keep the push-subscription snapshot fresh for the notify gate.
    refreshSubscribed();

    // Evaluate whatever's already on screen (the first problem at boot).
    if (state.appMode === APP_MODES.SOLVING) onSolvingProblemShown(state.currentProblem);
}

// --- public trigger hooks ----------------------------------------------------

export function onSolvingProblemShown(problem) {
    if (!_ready || _suppressed) return;
    markActive();
    const isRepeat = !!(problem && problem.isRepeat);
    tryShow(['intro', 'earlyMistakes', 'recall', 'rank', 'previous', 'stats', 'account', 'install', 'notify', 'boardSize', 'komi'], { isRepeat });
}

export function onEnterReview() {
    tryShow(['earlyMistakes', 'reviewIntro', 'reviewRank', 'share']);
}

// The user clicked the Previous button themselves — they know it; retire the
// teaching bubble (a no-op once the flag is set). Deliberately NOT keyed off
// entering review mode: a wrong answer auto-enters review without the button.
export function onPreviousClicked() {
    if (!wasShown('previous')) markShown('previous');
}

export function onReviewScoreShown() {
    tryShow(['reviewScore']);
}

export function onScoreUnreliable() {
    tryShow(['reviewScoreUnreliable']);
}

export function onResume() {
    if (!_ready || _suppressed) return;
    markActive();
    refreshSubscribed();
    if (state.appMode === APP_MODES.SOLVING) onSolvingProblemShown(state.currentProblem);
}
