// js/coord-input.js
//
// Keyboard coordinate entry — an accessibility path for users without fine
// pointer control. The input is permanently invisible (it must never cost
// screen space for anyone else): typing a coordinate-looking character
// anywhere focuses it, board notation ("C3"; columns skip I, rows count from
// the bottom) submits on Enter through the same path as a board click, and
// the board itself is the feedback. Because the buffer is unseen, it is
// cleared on every Enter and on blur so stale characters can never corrupt
// the next entry.

import { state } from './state.js';
import { boardNotationToXy } from './sgf.js';
import { playTypedMove } from './board.js';
import { getTerm } from './i18n.js';

export function initCoordInput() {
    const input = document.getElementById('coord-input');
    input.setAttribute('aria-label', getTerm('coordInputLabel'));

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            input.blur();
            return;
        }
        if (e.key !== 'Enter') return;
        e.preventDefault();
        const move = boardNotationToXy(input.value.trim(), state.boardSize);
        input.value = '';
        if (move) playTypedMove(move.x, move.y);
    });

    input.addEventListener('blur', () => {
        input.value = '';
    });

    // Type-to-activate: a bare coordinate-looking keystroke while no text
    // field is focused moves focus into the input, and the browser's default
    // action then inserts that same character there — so a physical keyboard
    // works from anywhere without tabbing first. Touch never fires keydown,
    // so mobile (and its soft keyboard) is unaffected by construction, and
    // focus is never taken from a real text-entry target.
    const isTextEntry = (el) => !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable);
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.altKey || e.metaKey) return;
        if (!/^[a-hj-t0-9]$/i.test(e.key)) return;
        if (isTextEntry(document.activeElement)) return;
        input.focus();
    });
}
