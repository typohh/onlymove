// js/share.js
//
// The "share this problem" flow, reachable from the analysis-mode hamburger
// menu. Builds the share URL (with orientation), records create/open analytics,
// and offers link / image / both via the Web Share API with clipboard and
// toast fallbacks. Pulled out of main.js so the entry point stays wiring-only.

import { state } from './state.js';
import { getTerm } from './i18n.js';
import { recordShareEvent } from './user-stats.js';
import { getBoardImageBlob } from './board.js';
import { showModal, showToast } from './overlays.js';

let isSharing = false;

export function performShare() {
    if (isSharing) return;
    if (!state.problemHistory[1]) {
        console.error('No current problem to share');
        return;
    }
    isSharing = true;
    const releaseGuard = () => { isSharing = false; };

    const tsumegoId = state.problemHistory[1].filename.replace('.sgf', '');
    const orientation = state.problemHistory[1].orientation;
    let shareUrl = `${window.location.origin}${window.location.pathname}?id=${tsumegoId}`;
    if (typeof orientation === 'number' && orientation !== 0) {
        shareUrl += `&o=${orientation}`;
    }

    // Record that a share link was generated (fire-and-forget analytics).
    recordShareEvent({ type: 'create', problemId: tsumegoId });

    const copyToClipboard = () => {
        return navigator.clipboard.writeText(shareUrl).then(() => {
            showToast(getTerm('linkCopied'));
        }).catch(err => {
            console.error('Failed to copy URL: ', err);
            showToast(getTerm('error'), { isError: true });
        });
    };

    const shareLink = () => {
        if (navigator.share && navigator.canShare && navigator.canShare({ url: shareUrl })) {
            return navigator.share({ url: shareUrl }).catch(err => {
                if (err.name === 'AbortError') return;
                console.error('Share link failed: ', err);
                showToast(getTerm('shareFailed'), { isError: true });
                return copyToClipboard();
            });
        }
        return copyToClipboard();
    };

    // Generate the board image with a visible "preparing…" toast and a
    // hard timeout so canvas/render hangs surface as an error rather than
    // a silent forever-spinner.
    const generateImageFile = async () => {
        const toast = showToast(getTerm('preparingImage'), { autoDismiss: false });
        const cleanup = () => { if (toast.parentNode) toast.parentNode.removeChild(toast); };
        try {
            const blob = await Promise.race([
                getBoardImageBlob(),
                new Promise((_, reject) => setTimeout(() => reject(new Error('image generation timed out')), 10000))
            ]);
            cleanup();
            if (!blob) throw new Error('No blob returned from getBoardImageBlob');
            return new File([blob], 'tsumego.png', { type: 'image/png' });
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
            file = await generateImageFile();
        } catch {
            return shareLink();
        }
        const data = { files: [file], title: getTerm('tsumegoMode') };
        if (navigator.share && navigator.canShare && navigator.canShare(data)) {
            return navigator.share(data).catch(err => {
                if (err.name === 'AbortError') return;
                console.error('Share image failed: ', err);
                showToast(getTerm('shareFailed'), { isError: true });
                return shareLink();
            });
        }
        return shareLink();
    };

    const shareBoth = async () => {
        let file;
        try {
            file = await generateImageFile();
        } catch {
            return shareLink();
        }
        const data = { files: [file], url: shareUrl, title: getTerm('tsumegoMode') };
        if (navigator.share && navigator.canShare && navigator.canShare(data)) {
            return navigator.share(data).catch(err => {
                if (err.name === 'AbortError') return;
                console.error('Share both failed: ', err);
                showToast(getTerm('shareFailed'), { isError: true });
                return copyToClipboard();
            });
        }
        return shareLink();
    };

    const showShareChooser = () => {
        // Full-width stacked .btn buttons (not .btn-inline) — the chooser
        // intentionally presents three equal-width options.
        const mkBtn = (term, action) => {
            const btn = document.createElement('button');
            btn.className = 'btn';
            btn.textContent = getTerm(term);
            return [btn, action];
        };
        showModal({
            onDismiss: releaseGuard,
            build: (modal, close) => {
                for (const [btn, action] of [
                    mkBtn('shareBoth', shareBoth),
                    mkBtn('shareLink', shareLink),
                    mkBtn('shareImage', shareImage),
                ]) {
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

    if (filesSupported) {
        showShareChooser();
    } else {
        shareLink().finally(releaseGuard);
    }
}
