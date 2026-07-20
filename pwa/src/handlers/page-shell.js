// pwa/src/handlers/page-shell.js
//
// Shared chrome for the Worker's server-rendered pages (/leaderboard,
// /invite): the base typography/layout CSS and the esc() helper each page
// embeds in its inline <script>. Page-specific styling stays in the
// handlers; this owns only the pieces that shouldn't drift apart.

export function baseCss(maxWidthPx) {
  return `:root { color-scheme: light dark; }
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0; padding: 1.5rem 1rem 3rem;
         max-width: ${maxWidthPx}px; margin-inline: auto; }
  h1 { font-size: 1.4rem; margin: 0 0 .3rem; }
  p.sub { margin: 0 0 1rem; opacity: .6; font-size: .85rem; }
  table { width: 100%; border-collapse: collapse; }
  tbody tr:nth-child(even) { background: color-mix(in srgb, currentColor 5%, transparent); }`;
}

// HTML-escape helper injected into each page's inline script (both pages
// render their rows client-side from inlined JSON).
export const ESC_SNIPPET = `function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}`;

// --- leaderboard-row display helpers ----------------------------------------------
// Country code -> flag emoji. Re-exported from the shared module so the Worker page
// and the client's Community-stats country distribution use ONE definition.
export { flag } from '../../shared/flag.js';
