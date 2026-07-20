// pwa/shared/flag.js
//
// Country code -> flag emoji, shared between the Worker (the /leaderboard page)
// and the client (the Community-stats country distribution) so the two never
// drift — same pattern as shared/rank-math.js. A valid ISO 3166-1 alpha-2 code
// becomes its regional-indicator pair; anything else returns '' (the caller
// decides whether to substitute a fallback glyph).
export const flag = (cc) => (typeof cc === 'string' && /^[A-Za-z]{2}$/.test(cc))
    ? cc.toUpperCase().replace(/./g, (c) => String.fromCodePoint(0x1F1E6 + c.charCodeAt(0) - 65))
    : '';
