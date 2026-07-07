// js/share.js
//
// The "share this problem" flow, reachable from the analysis-mode hamburger
// menu. Builds the share URL (with orientation), records create/open analytics,
// and offers link / image / both via the Web Share API with clipboard and
// toast fallbacks. Pulled out of main.js so the entry point stays wiring-only.

import * as Sentry from "@sentry/browser";
import { state } from './state.js';
import { tsumegoIdOf } from './sgf.js';
import { getTerm } from './i18n.js';
import { recordShareEvent } from './user-stats.js';
import { getBoardImageBlob } from './board.js';
import { showModal, showToast } from './overlays.js';

let isSharing = false;

// Legacy synchronous copy (works in insecure contexts and old browsers where
// navigator.clipboard is unavailable). Requires the active user gesture we
// already have from the share click. Returns whether the copy succeeded.
function legacyCopy(text) {
    try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.top = '0';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        return ok;
    } catch (_) {
        return false;
    }
}

// Last resort when no clipboard path works: show the URL pre-selected so the
// user can long-press/copy it by hand. Never throws.
function showShareUrlFallback(url) {
    showModal({
        build: (modal) => {
            const p = document.createElement('p');
            p.textContent = getTerm('copyLinkManually');
            const input = document.createElement('input');
            input.type = 'text';
            input.readOnly = true;
            input.value = url;
            input.style.width = '100%';
            input.addEventListener('focus', () => input.select());
            modal.appendChild(p);
            modal.appendChild(input);
            setTimeout(() => { input.focus(); input.select(); }, 0);
        },
    });
}

export function performShare() {
    if (isSharing) return;
    if (!state.previousProblem) {
        console.error('No current problem to share');
        return;
    }
    isSharing = true;
    const releaseGuard = () => { isSharing = false; };

    const tsumegoId = tsumegoIdOf(state.previousProblem.filename);
    const orientation = state.previousProblem.orientation;
    // Web Share + the async Clipboard API are secure-context only, and a link
    // generated while on plain http propagates that insecure origin to whoever
    // opens it (which then breaks *their* share too). Pin generated links to
    // https — localhost dev aside — so an http session can't poison the link.
    const h = window.location.hostname;
    const isLocalhost = h === 'localhost' || h === '::1' || h.startsWith('127.');
    const origin = isLocalhost ? window.location.origin
                               : window.location.origin.replace(/^http:/, 'https:');
    let shareUrl = `${origin}${window.location.pathname}?id=${tsumegoId}`;
    if (typeof orientation === 'number' && orientation !== 0) {
        shareUrl += `&o=${orientation}`;
    }

    // Record that a share link was generated (fire-and-forget analytics).
    recordShareEvent({ type: 'create', problemId: tsumegoId });

    let clipboardImageSupported = false;
    try {
        clipboardImageSupported = !!(navigator.clipboard && navigator.clipboard.write && window.ClipboardItem);
    } catch {
        clipboardImageSupported = false;
    }

    const copyToClipboard = () => {
        const copied = () => showToast(getTerm('linkCopied'));
        const localFallback = () => { if (legacyCopy(shareUrl)) copied(); else showShareUrlFallback(shareUrl); };
        // Only touch the clipboard while we still hold transient user activation.
        // Calling navigator.clipboard.writeText() WITHOUT it makes Android Chrome
        // raise the clipboard *permission* prompt ("see text and images copied to
        // the clipboard") — hit on the post-share copy fallback, where the tap's
        // activation has already lapsed. With no activation, skip the clipboard
        // API entirely and show the URL for manual copy: no API call, no prompt.
        // (execCommand copy is gesture-gated too, so it sits behind the same check.)
        // navigator.clipboard is also undefined in an insecure context / older
        // browsers, which the manual modal covers as well.
        const active = !navigator.userActivation || navigator.userActivation.isActive;
        if (!active) { showShareUrlFallback(shareUrl); return Promise.resolve(); }
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(shareUrl).then(copied).catch(err => {
                console.error('Failed to copy URL: ', err);
                localFallback();
            });
        }
        localFallback();
        return Promise.resolve();
    };

    const copyImageToClipboard = (file) => {
        const copied = () => showToast(getTerm('imageReady'));
        const active = !navigator.userActivation || navigator.userActivation.isActive;
        if (!active) {
            showToast(getTerm('shareFailed'), { isError: true });
            return Promise.resolve();
        }
        const item = new ClipboardItem({ [file.type]: file });
        return navigator.clipboard.write([item]).then(copied).catch(err => {
            console.error('Failed to copy image: ', err);
            showToast(getTerm('shareFailed'), { isError: true });
        });
    };

    const shareLink = () => {
        if (navigator.share && navigator.canShare && navigator.canShare({ url: shareUrl })) {
            return navigator.share({ url: shareUrl }).catch(err => {
                if (err.name === 'AbortError') return;
                console.error('Share link failed: ', err);
                reportShareRefusal('link', err);
                showToast(getTerm('shareFailed'), { isError: true });
                return copyToClipboard();
            });
        }
        return copyToClipboard();
    };

    // Board-image and Web Share failures are otherwise invisible (we swallow them
    // and fall back to a clipboard copy), so log enough to tell an expired-
    // activation refusal (NotAllowedError with userActivation.isActive === false)
    // apart from other causes (share-in-progress lock, OS sheet state). One source
    // of truth for the three share() catch handlers; the activation hypothesis is
    // still under investigation via this telemetry, not yet confirmed.
    let captureMs = null; // wall-clock of the board-image capture, for telemetry
    const reportShareRefusal = (action, err) => {
        try {
            const act = navigator.userActivation;
            Sentry.captureMessage('Web Share refused', {
                level: 'warning',
                extra: {
                    action,
                    errName: err && err.name,
                    errMessage: err && err.message,
                    captureMs,
                    activeAtShare: act ? act.isActive : null,
                    hasBeenActive: act ? act.hasBeenActive : null,
                },
            });
        } catch (_) {}
    };

    // If share() is refused specifically for lack of transient user activation
    // (NotAllowedError) — i.e. the image capture ran long enough that the tap's
    // ~5s activation expired — we STILL hold the ready File. Re-offer it behind an
    // explicit button: that tap is a fresh activation and the File is already in
    // hand, so share() fires straight from the handler (no await between) and the
    // native sheet opens. Reached only on the real too-slow-capture failure, so no
    // image is ever generated for a share the user didn't ask for.
    const offerReadyShare = (data) => {
        showModal({
            onDismiss: releaseGuard,
            build: (modal, close) => {
                const p = document.createElement('p');
                p.textContent = getTerm('imageReady');
                const btn = document.createElement('button');
                btn.className = 'btn menu-item';
                btn.textContent = getTerm('share');
                btn.addEventListener('click', () => {
                    close();
                    navigator.share(data).catch(err => {
                        if (err.name === 'AbortError') return;
                        console.error('Re-share failed: ', err);
                        reportShareRefusal('retry', err);
                        showToast(getTerm('shareFailed'), { isError: true });
                        return copyToClipboard();
                    });
                });
                modal.appendChild(p);
                modal.appendChild(btn);
            },
        });
    };

    // Warm the board-image capture as soon as the chooser opens, while the user
    // is still reading the options. getBoardImageBlob() is async and yields at
    // its awaits (image load, a 50ms tick, toBlob encode), so it doesn't freeze
    // the chooser — only brief synchronous WGo-render bursts. The head start
    // shrinks the gap between the user's pick and the navigator.share() call, so
    // share() is far more likely to land inside the ~5s transient-activation
    // window (the gap shrinks by the reading time even when capture isn't done
    // by the pick; if it IS done, the gap is ~0). Cached so the open-time warm
    // and the pick-time resolve share a single capture.
    let imageFilePromise = null;
    const warmImageFile = () => {
        if (!imageFilePromise) {
            const t0 = performance.now();
            imageFilePromise = (async () => {
                const blob = await getBoardImageBlob();
                if (!blob) throw new Error('No blob returned from getBoardImageBlob');
                return new File([blob], 'tsumego.png', { type: 'image/png' });
            })();
            const stamp = () => { captureMs = Math.round(performance.now() - t0); };
            imageFilePromise.then(stamp, stamp);
            imageFilePromise.catch(() => {}); // avoid an unhandled rejection if unused
        }
        return imageFilePromise;
    };

    // Resolve the warmed File for an image-bearing pick: show a "preparing…"
    // toast only if it isn't ready yet, with a hard timeout so a hung capture
    // surfaces as an error rather than a silent forever-spinner.
    const resolveImageFile = async () => {
        const p = warmImageFile();
        let ready = false;
        p.then(() => { ready = true; }, () => { ready = true; });
        await Promise.resolve();
        const toast = ready ? null : showToast(getTerm('preparingImage'), { autoDismiss: false });
        const cleanup = () => { if (toast && toast.parentNode) toast.parentNode.removeChild(toast); };
        try {
            const file = await Promise.race([
                p,
                new Promise((_, reject) => setTimeout(() => reject(new Error('image generation timed out')), 10000))
            ]);
            cleanup();
            return file;
        } catch (err) {
            cleanup();
            console.error('Failed to capture image: ', err);
            showToast(getTerm('imageCaptureFailed'), { isError: true });
            throw err;
        }
    };

    const shareImage = async () => {
        let file;
        try {
            file = await resolveImageFile();
        } catch {
            return shareLink();
        }
        const data = { files: [file], title: getTerm('tsumegoMode') };
        if (navigator.share && navigator.canShare && navigator.canShare(data)) {
            return navigator.share(data).catch(err => {
                if (err.name === 'AbortError') return;
                console.error('Share image failed: ', err);
                reportShareRefusal('image', err);
                if (err.name === 'NotAllowedError') return offerReadyShare(data);
                showToast(getTerm('shareFailed'), { isError: true });
                if (clipboardImageSupported) {
                    return copyImageToClipboard(file);
                }
                return shareLink();
            });
        }
        if (clipboardImageSupported) {
            return copyImageToClipboard(file);
        }
        return shareLink();
    };

    const shareBoth = async () => {
        let file;
        try {
            file = await resolveImageFile();
        } catch {
            return shareLink();
        }
        const data = { files: [file], url: shareUrl, title: getTerm('tsumegoMode') };
        if (navigator.share && navigator.canShare && navigator.canShare(data)) {
            return navigator.share(data).catch(err => {
                if (err.name === 'AbortError') return;
                console.error('Share both failed: ', err);
                reportShareRefusal('both', err);
                if (err.name === 'NotAllowedError') return offerReadyShare(data);
                showToast(getTerm('shareFailed'), { isError: true });
                return copyToClipboard();
            });
        }
        return shareLink();
    };

    const showShareChooser = () => {
        // Start the board-image capture now, while the user reads the options —
        // the head start keeps the eventual share() inside the activation window.
        warmImageFile();
        // Full-width stacked .btn buttons (not .btn-inline) — the chooser
        // intentionally presents three equal-width options.
        const mkBtn = (term, action) => {
            const btn = document.createElement('button');
            btn.className = 'btn menu-item';
            btn.textContent = getTerm(term);
            return [btn, action];
        };
        showModal({
            onDismiss: releaseGuard,
            build: (modal, close) => {
                const buttons = [];
                if (filesSupported) {
                    buttons.push(mkBtn('shareBoth', shareBoth));
                }
                buttons.push(mkBtn('shareLink', shareLink));
                buttons.push(mkBtn('shareImage', shareImage));

                for (const [btn, action] of buttons) {
                    btn.addEventListener('click', () => { close(); action().finally(releaseGuard); });
                    modal.appendChild(btn);
                }
            },
        });
    };

    // Probe whether the browser can share files at all, without actually generating one.
    let filesSupported = false;
    try {
        filesSupported = !!(navigator.share && navigator.canShare &&
            navigator.canShare({ files: [new File([], 'p.png', { type: 'image/png' })] }));
    } catch {
        filesSupported = false;
    }

    if (filesSupported || clipboardImageSupported) {
        showShareChooser();
    } else {
        shareLink().finally(releaseGuard);
    }
}
