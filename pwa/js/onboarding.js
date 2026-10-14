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
// onEnterReview, onReviewScoreShown, onScoreUnreliable, onRecallBadgeShown,
// onLinkedProblemShown, onResume) at existing UI seams; everything else —
// anchoring, gating, session and progress counting — lives here. The bubble itself (its DOM and its placement) is js/bubble.js,
// shared with the hint affordances; styling is the `ui-bubble` block in
// style.css.
//
// The whole system self-disables for anyone who isn't a genuinely new user when
// it first runs (see initOnboarding): existing players never get "Black to move".

import { state, getMergedAnswerHistory } from './state.js';
import { createBubble, placeBubble, absorbDismissTap } from './bubble.js';
import { getTerm } from './i18n.js';
import { APP_MODES, HISTORY_BACKFILL_DONE_KEY } from './constants.js';
import { waitingRecallCount } from './recall-model.js';
import { supportsNotifications, notificationPermission, getCurrentSubscription } from './notifications.js';

// --- tunables ---------------------------------------------------------------

// A device with more than this many recorded attempts when onboarding first
// runs is treated as an existing user and never shown any bubble.
const NEW_USER_MAX_HISTORY = 3;
// Two spells of ACTIVITY more than this far apart count as separate "sessions"
// (drives the install / stats / account bubbles, which wait for a return
// visit). Activity-based rather than boot-based: a kept-open tab or a resumed
// PWA never cold-boots, so markActive() does the counting wherever it is
// called. A service-worker-update reload lands well inside the gap.
const SESSION_GAP_MS = 2 * 60 * 60 * 1000;   // 2 hours
// The notify bubble waits this long after the recall-button bubble, so the
// "get notified when problems come due" pitch lands on someone who has already
// seen recalls accumulate.
const NOTIFY_AFTER_RECALL_MS = 48 * 60 * 60 * 1000;
// Minimum wall-clock gap between any two bubbles — pure anti-stacking, so a
// burst of triggers (answer wrong → review opens → score shows) can't pile up.
const MIN_GAP_MS = 15 * 1000;
// Ambient bubbles additionally wait this many completed problems since the last
// bubble of any kind — the "roughly one every few problems" drip.
const SPACING_PROBLEMS = 3;
// At most this many AMBIENT bubbles in one session. Trigger bubbles are
// exempt and don't count: they answer something the user just did (first
// review, a shared problem, the weakness list appearing), and a long first
// sitting must not spend the budget on the drip and then miss those moments.
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

// The shown flag holds the paint timestamp (older installs hold a literal '1',
// which reads as "long ago" — fine for every gate that consults the time).
const wasShown = (id) => !!lsGet(SHOWN_PREFIX + id);
const markShown = (id) => lsSet(SHOWN_PREFIX + id, Date.now());
const shownAt = (id) => { const v = parseInt(lsGet(SHOWN_PREFIX + id), 10); return Number.isFinite(v) ? v : 0; };

// --- runtime state ----------------------------------------------------------

let _ready = false;              // init() has run
let _featureReady = false;       // init() has run; set even for suppressed users
let _suppressed = true;          // until init() decides otherwise
let _sessionCount = 1;
let _ambientThisSession = 0;     // ambient bubbles shown this session (SESSION_CAP)
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
    recall: () => el('button-recall'),
    share: () => el('button-share'),
    install: () => el('button4'),
    // The hamburger button itself (portrait ☰ / landscape "Menu"): the
    // menu-item bubbles point here and tell the user to open it, rather than
    // firing only once the menu is already open.
    menu: () => firstVisible(['menu-toggle-h', 'menu-toggle']),
    motifBubble: () => el('motif-bubble'),
};

// The answer's intersection as a stone-sized viewport rect, or null when there
// is none to point at (no board, a pass answer) or it must not be shown (a
// linked problem: a shared puzzle is someone else's to give away). The answer
// is the decision node's first child by the corpus contract (isCorrectMove's
// rule). WGo's getX/getY are in canvas pixels relative to its element, so they
// come back to CSS pixels through its pixelRatio.
function answerRect() {
    if (state.currentProblem?.linked) return null;
    const board = state.board;
    const move = state.kifuReader?.node?.children?.[0]?.move;
    if (!board || !board.element || !move || move.x == null || move.y == null) return null;
    const r = board.element.getBoundingClientRect();
    if (!(r.width > 0 && r.height > 0)) return null;
    const ratio = board.pixelRatio || 1;
    const cx = r.left + board.getX(move.x) / ratio;
    const cy = r.top + board.getY(move.y) / ratio;
    const half = Math.min(board.fieldWidth, board.fieldHeight) / ratio / 2;
    return new DOMRect(cx - half, cy - half, 2 * half, 2 * half);
}

const isVisible = (node) => {
    if (!node) return false;
    const r = node.getBoundingClientRect();
    if (!(r.width > 0 && r.height > 0)) return false;
    // offsetParent is null for position:fixed even while rendered (the motif
    // bubble in portrait); display:none is already caught by the zero rect.
    return node.offsetParent !== null || getComputedStyle(node).position === 'fixed';
};

// --- bubble catalogue --------------------------------------------------------
//
// kind: 'trigger' fires when its context appears (subject only to the anti-stack
// time gap); 'ambient' additionally waits SPACING_PROBLEMS and is what the
// per-session SESSION_CAP counts. `anchor` names an
// entry in `anchors`; 'board' places the bubble over the board pointing at its
// centre, or at the answer's intersection with `pointAt: 'answer'`. `eligible(ctx)` is the context gate on top of the fire-once flag.
// `text` is an i18n key, or a function returning one when the wording depends
// on the device (resolved at render time, not module-eval).

const BUBBLES = {
    // The one bubble that describes the input itself, so it has to match
    // board.js: two clicks to play, the first setting a ghost stone.
    // Offered on a shared problem too — for a newcomer arriving through a link
    // that IS the first board.
    // Its arrow points at the answer's intersection.
    intro: {
        kind: 'trigger', anchor: 'board', pointAt: 'answer', placement: 'above',
        text: 'onbIntro',
        eligible: () => historyLen() <= 1,
    },
    recall: {
        kind: 'trigger', anchor: 'title', placement: 'below', text: 'onbRecall',
        eligible: (ctx) => !!ctx.isRepeat,
    },
    // The Recall button, the first time its waiting-count badge paints — so the
    // bubble points at a badge with a number in it, never at a bare button.
    recallButton: {
        kind: 'trigger', anchor: 'recall', placement: 'below', text: 'onbRecallButton',
        eligible: () => waitingRecallCount() > 0 && isVisible(anchors.recall()),
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
    // Waits for the recall system to have visibly engaged — the recall-button
    // bubble, which itself needed a waiting recall — plus a couple of days, so
    // "get notified when problems come due" describes something the user has
    // already watched happen. Installed only: uninstalled, button4 reads Install.
    notify: {
        kind: 'ambient', anchor: 'install', placement: 'below', text: 'onbNotify',
        eligible: () => isInstalled() && wasShown('recallButton') &&
            (Date.now() - shownAt('recallButton')) >= NOTIFY_AFTER_RECALL_MS &&
            notificationPermission() !== 'denied' && !_subscribed && isVisible(anchors.install()),
    },
    // Review-mode, at the root position. Must land on the FIRST review, which a
    // wrong answer usually opens seconds after the intro — so it ignores the
    // anti-stack gap (that gap is for bursts within one context; entering
    // review is a context switch, and onEnterReview clears the old bubble).
    reviewIntro: {
        kind: 'trigger', anchor: 'board', placement: 'above', text: 'onbReviewIntro',
        ignoreGap: true,
        eligible: () => state.appMode === APP_MODES.ANALYZING,
    },
    // The review readout is overloaded — the problem's difficulty rank at the
    // root, the score estimate once inside a variation — and ONE bubble covers
    // both meanings, offered on a review entry after reviewIntro or from the
    // score-paint hook, whichever comes first. The two legacy flags belong to
    // the pre-merge pair; anyone who met either has been told.
    reviewReadout: {
        kind: 'trigger', anchor: 'rank', placement: 'below', text: 'onbReviewReadout',
        eligible: () => state.appMode === APP_MODES.ANALYZING && wasShown('reviewIntro') &&
            !wasShown('reviewRank') && !wasShown('reviewScore'),
    },
    share: {
        kind: 'trigger', anchor: 'share', placement: 'below', text: 'onbShare',
        eligible: () => state.appMode === APP_MODES.ANALYZING && wasShown('reviewIntro') &&
            isVisible(anchors.share()),
    },
    reviewScoreUnreliable: {
        kind: 'trigger', anchor: 'unreliableScore', placement: 'below', text: 'onbScoreUnreliable',
        eligible: () => state.appMode === APP_MODES.ANALYZING && isVisible(anchors.unreliableScore()),
    },
    // Menu-contained features. We do NOT open the menu for the user (intrusive);
    // instead these point at the Menu button and the text names the item, so the
    // bubble teaches both that the menu exists and what's inside it.
    // The Stats modal as a whole, for a returning user with enough history that
    // its charts show something. The weaknesses inside it are NOT this bubble's
    // trigger — trainingFocus announces those when a finding first paints.
    stats: {
        kind: 'ambient', anchor: 'menu', placement: 'below', text: 'onbStats',
        eligible: () => _sessionCount >= 2 && historyLen() >= 15 && isVisible(anchors.menu()),
    },
    // Every active player has an anonymous leaderboard row from their first
    // report, so this can come early; the text names the way to a named row.
    leaderboard: {
        kind: 'ambient', anchor: 'menu', placement: 'below', text: 'onbLeaderboard',
        eligible: () => historyLen() >= 10 && isVisible(anchors.menu()),
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
    // A NEW-FEATURE announcement rather than a first-run tip: fires the first
    // time the training-focus list (motif-bubble.js, solving mode) paints. The
    // list appears only for a converged user with a trained weakness model —
    // i.e. mostly EXISTING users — so this is the one bubble exempt from the
    // existing-user suppression (bypassSuppression, honored in scheduleShow and
    // entered via its own hook below). Once-ever, the anti-stack gap and the
    // backoff still apply.
    trainingFocus: {
        kind: 'trigger', anchor: 'motifBubble', placement: 'above', text: 'onbTrainingFocus',
        bypassSuppression: true,
        eligible: () => state.appMode === APP_MODES.SOLVING && isVisible(anchors.motifBubble()),
    },
    // A shared (linked) problem: say up front that it is a try-out — linked
    // attempts enter neither history nor the rank. Shared links reach existing
    // users as much as new ones, hence bypassSuppression. Only for someone who
    // HAS a rank to worry about: a newcomer's first bubble should be the intro,
    // and a bubble here would hold it back through the anti-stack gap.
    linked: {
        kind: 'trigger', anchor: 'board', placement: 'above', text: 'onbLinked',
        bypassSuppression: true,
        eligible: () => state.appMode === APP_MODES.LINKED && historyLen() >= 5,
    },
};

// --- gating ------------------------------------------------------------------

function spacingOk(def) {
    const lastShown = lsNum(K_LAST_SHOWN_AT);
    if (!def.ignoreGap && lastShown && (Date.now() - lastShown) < MIN_GAP_MS) return false;
    if (def.kind === 'ambient') {
        if (_ambientThisSession >= SESSION_CAP) return false;
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
        if (!spacingOk(def)) return;   // gate is global — don't fall through to a lower-priority one
        scheduleShow(id, def, ctx);
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

// The tap that dismisses a tip must never also PLAY. Two targets are absorbed
// (absorbDismissTap in bubble.js): the bubble itself, and the board — the
// second because a tip arrives unbidden and covers live intersections, so a tap
// there is a dismissal aimed at a board the user cannot even see, and the move
// it would otherwise play is recorded as an attempt and cannot be taken back.
// Every OTHER target keeps falling through, which is the "tap anywhere to
// continue" behaviour: a menu item or a button reached through a tip does what
// it says, and the tip goes away with it.
const absorbsDismiss = (target) => {
    if (!(target instanceof Node)) return false;
    if (_bubbleEl && _bubbleEl.contains(target)) return true;
    const board = anchors.board();
    return !!(board && board.contains(target));
};

function onDismiss(e) {
    if (e && absorbsDismiss(e.target)) absorbDismissTap(e);
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

// A fullscreen transition/loading overlay is up — hold the bubble back
// (the menu overlay is deliberately NOT blocking: menu bubbles need it open).
const overlayBlocking = () =>
    !!document.querySelector('.app-blocking-overlay, .loading-overlay');

// `ctx` is carried through the delay and re-checked with, NOT dropped: a gate
// like recall's `ctx.isRepeat` is a fact about the problem on screen that
// nothing on the page restates, so re-verifying against an empty context would
// refuse every context-gated bubble unconditionally.
function scheduleShow(id, def, ctx = {}) {
    _pending = true;
    let tries = 0;
    const attempt = () => {
        _pendingTimer = 0;
        // Re-verify: the world may have moved on during the delay. Feature
        // announcements (bypassSuppression) need init to have run but ignore
        // the existing-user suppression.
        const blocked = _backedOff ||
            (def.bypassSuppression ? !_featureReady : (!_ready || _suppressed));
        if (blocked) { _pending = false; return; }
        if (document.hidden || overlayBlocking() || !isVisible(anchors.board())) {
            if (tries++ < 30) { _pendingTimer = setTimeout(attempt, 300); return; }
            _pending = false; return;   // gave up waiting for a clear board
        }
        let target;
        try { if (!def.eligible(ctx)) { _pending = false; return; } } catch { _pending = false; return; }
        target = def.anchor === 'board' ? anchors.board() : anchors[def.anchor]();
        if (!isVisible(target)) { _pending = false; return; }
        render(id, def);
    };
    _pendingTimer = setTimeout(attempt, SHOW_DELAY_MS);
}

function render(id, def) {
    const bubble = createBubble(getTerm(typeof def.text === 'function' ? def.text() : def.text),
                                { topLayer: !!def.topLayer });
    document.body.appendChild(bubble);
    _bubbleEl = bubble;
    _repositionArgs = { def };

    position(bubble, def);
    requestAnimationFrame(() => bubble.classList.add('ui-bubble-visible'));

    // Record + arm dismissal.
    markShown(id);
    lsSet(K_LAST_SHOWN_AT, Date.now());
    if (def.kind === 'ambient') {
        lsSet(K_LAST_AMBIENT_COUNT, historyLen());
        _ambientThisSession++;
    }
    _shownAtMs = Date.now();

    // Any tap/keypress dismisses. A tap outside the bubble still reaches
    // whatever's underneath — the "click anywhere to continue" behaviour —
    // except on the bubble itself and on the board, which are absorbed (see
    // absorbsDismiss).
    document.addEventListener('pointerdown', onDismiss, true);
    document.addEventListener('keydown', onDismiss, true);
    window.addEventListener('resize', onReposition);
    _pendingTimer = setTimeout(onDismiss, AUTO_DISMISS_MS);
}

// Resolve this tip's anchor to a rect and hand it to the shared placer. The
// 'board' anchor is the one that wants `overAnchor`: it is far too big to sit
// clear of, so the bubble goes on top of it pointing inward.
function position(bubble, def) {
    const target = def.anchor === 'board' ? anchors.board() : anchors[def.anchor]();
    if (!isVisible(target)) return;
    // A single intersection is small enough to stand clear of, like any other
    // control; without one (a pass answer, a linked problem) the tip falls back
    // to the centre.
    const point = def.pointAt === 'answer' ? answerRect() : null;
    if (point) {
        placeBubble(bubble, point, { placement: def.placement });
        return;
    }
    placeBubble(bubble, target.getBoundingClientRect(), {
        placement: def.placement,
        overAnchor: def.anchor === 'board',
    });
}

// --- session / init ----------------------------------------------------------

// Records activity, and counts a session: a gap longer than SESSION_GAP_MS
// since the previous activity bumps the counter. Called at boot, on resume and
// on every solving paint, so a return after hours counts whether or not the
// app cold-booted or the browser delivered a visibility event.
function markActive() {
    const last = lsNum(K_LAST_ACTIVE);
    if (last && (Date.now() - last) > SESSION_GAP_MS) {
        _sessionCount = (lsNum(K_SESSION_COUNT) || 1) + 1;
        lsSet(K_SESSION_COUNT, _sessionCount);
    }
    lsSet(K_LAST_ACTIVE, Date.now());
}

export function initOnboarding() {
    // First ever run: decide new-user vs existing-user once.
    if (!lsGet(K_INITIALIZED)) {
        lsSet(K_INITIALIZED, '1');
        if (historyLen() > NEW_USER_MAX_HISTORY) lsSet(K_SUPPRESSED, '1');
        lsSet(K_SESSION_COUNT, '1');
    }
    markActive();
    _sessionCount = lsNum(K_SESSION_COUNT) || 1;
    _suppressed = !!lsGet(K_SUPPRESSED);
    // Feature-announcement bubbles (bypassSuppression) run for everyone; the
    // first-run ones only for a genuinely new user.
    _featureReady = true;
    if (!_suppressed) {
        _ready = true;
        // Keep the push-subscription snapshot fresh for the notify gate.
        refreshSubscribed();
    }

    // Evaluate whatever is already on screen — the first problem at boot. A
    // shared problem painted BEFORE this init ran, so its paint hook was a
    // no-op. Both hooks are silent for whoever they don't apply to, and this
    // has to come AFTER readiness is decided or the regular path refuses.
    if (state.appMode === APP_MODES.SOLVING) onSolvingProblemShown(state.currentProblem);
    else if (state.appMode === APP_MODES.LINKED) onLinkedProblemShown();
}

// --- public trigger hooks ----------------------------------------------------

export function onSolvingProblemShown(problem) {
    if (!_ready || _suppressed) return;
    markActive();
    const isRepeat = !!(problem && problem.isRepeat);
    tryShow(['intro', 'earlyMistakes', 'recall', 'rank', 'previous', 'leaderboard', 'stats', 'account', 'install', 'notify', 'boardSize', 'komi'], { isRepeat });
}

export function onEnterReview() {
    // A context switch: a solving-mode bubble still up or scheduled describes
    // a board that is gone, and left alone it would hold the review bubbles
    // back (`_pending`). Clear it. Not counted as a dismissal.
    removeBubble();
    tryShow(['reviewIntro', 'earlyMistakes', 'reviewReadout', 'share']);
}

// The Recall button's waiting-count badge just painted with a non-zero count
// (ui.js, on every solving repaint) — teach the button the first time there is
// something behind it.
export function onRecallBadgeShown() {
    tryShow(['recallButton']);
}

// The user clicked the Previous button themselves — they know it; retire the
// teaching bubble (a no-op once the flag is set). Deliberately NOT keyed off
// entering review mode: a wrong answer auto-enters review without the button.
export function onPreviousClicked() {
    if (!wasShown('previous')) markShown('previous');
}

export function onReviewScoreShown() {
    tryShow(['reviewReadout']);
}

export function onScoreUnreliable() {
    tryShow(['reviewScoreUnreliable']);
}

// tryShow for a FEATURE ANNOUNCEMENT: bypasses the existing-user suppression
// (an announcement is new to existing users too, and they are usually the only
// ones who reach its context at all) while keeping every other gate —
// once-ever, anti-stacking, backoff.
function tryShowFeature(id, ctx = {}) {
    if (!_featureReady || _backedOff || _pending) return;
    if (document.hidden) return;
    if (wasShown(id)) return;
    const def = BUBBLES[id];
    try { if (!def.eligible(ctx)) return; } catch { return; }
    if (!spacingOk(def)) return;
    scheduleShow(id, def, ctx);
}

// The training-focus list just painted (motif-bubble.js calls this on every
// repaint that shows it) — the users converged enough to have one are mostly
// existing ones, hence the announcement path.
export function onTrainingFocusShown() {
    tryShowFeature('trainingFocus');
}

// A shared (linked) problem just painted (js/problem-loader.js, and init for
// the boot-time case). A newcomer gets the intro (regular path, so it is
// silent for existing users); anyone with a rank gets the try-out notice
// (feature path — shared links reach existing users too). The two gates
// (≤1 and ≥5 attempts) never both hold, and a scheduled intro makes the
// second call a no-op anyway.
export function onLinkedProblemShown() {
    tryShow(['intro']);
    tryShowFeature('linked');
}

export function onResume() {
    if (!_ready || _suppressed) return;
    markActive();
    refreshSubscribed();
    if (state.appMode === APP_MODES.SOLVING) onSolvingProblemShown(state.currentProblem);
    else if (state.appMode === APP_MODES.LINKED) onLinkedProblemShown();
}
