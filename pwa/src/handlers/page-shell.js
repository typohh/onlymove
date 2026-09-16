// pwa/src/handlers/page-shell.js
//
// Shared chrome for the Worker's server-rendered pages (/leaderboard,
// /statistics, /tournament): the base typography/layout CSS and the esc() helper each page
// embeds in its inline <script>. Page-specific styling stays in the
// handlers; this owns only the pieces that shouldn't drift apart.

// The document stylesheet (pwa/pages.css, a static asset) carries the palette,
// type and table rules every document page shares; a page adds its width here
// and its own rules after.
export function baseCss(maxWidthPx) {
  return `@import url('/pages.css');
  body { max-width: ${maxWidthPx}px; padding-top: 2rem; }
  h1 { font-size: 1.6rem; margin-bottom: 0.4rem; }`;
}

// HTML-escape helper injected into each page's inline script (both pages
// render their rows client-side from inlined JSON).
export const ESC_SNIPPET = `function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}`;

// --- leaderboard-row display helpers ----------------------------------------------
// Country code -> flag emoji. Re-exported from the shared module so the Worker page
// and the client's Community-stats country distribution use ONE definition.
export { flag } from '../../shared/flag.js';
