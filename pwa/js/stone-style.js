// js/stone-style.js
//
// The stone draw handler: WGo's shell stones with per-stone variation. Each
// stone gets its own highlight (offset, shape and strength move within a
// band) and a small chance of a flaw — a chip at the rim, or a hairline
// crack — so a board reads as a set of real stones rather than one stamp.
//
// Everything is a PURE FUNCTION of (board.stoneSeed, x, y). WGo redraws every
// object on each repaint, so a stone must come out identical draw after draw
// or it flickers; and the seed is the PROBLEM's id (board.js sets it), so a
// problem always shows the same stones in the same places, and two problems
// differ. Nothing here reads Math.random.
//
// A handler is `{ stone: { draw(stone, board) }, shadow }` with `this` bound
// to the 2D context (see WGo.Board.drawHandlers). The shadow is WGo's own.

const RIM_CHIP_RATE = 1 / 8;     // stones with a chip at the rim
const CRACK_RATE = 1 / 20;       // stones with a hairline crack
const SHELL_PATTERNS = [
    { lines: [0.10, 0.12, 0.11, 0.10, 0.09, 0.09, 0.09, 0.09], factor: 0.25, thickness: 1.75 },
    { lines: [0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06], factor: 0.20, thickness: 1.5 },
    { lines: [0.12, 0.14, 0.13, 0.12, 0.12, 0.12], factor: 0.30, thickness: 2 },
];

// FNV-1a over a string, folded with the intersection; then a small PRNG
// (mulberry32) seeded from it hands out the stone's numbers in a fixed order.
export function hashSeed(text) {
    let h = 0x811c9dc5;
    const s = String(text || '');
    for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
}

function rngFor(seed, x, y) {
    let a = (seed ^ Math.imul(x + 1, 0x9e3779b1) ^ Math.imul(y + 1, 0x85ebca6b)) >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// The shell's grain: a fan of faint arcs across the disc, as WGo draws it, at
// an angle and pattern of the stone's own.
function drawShellLines(ctx, cx, cy, r, angle, pattern) {
    let up = angle, down = angle;
    ctx.strokeStyle = 'rgba(64,64,64,0.2)';
    ctx.lineWidth = (r / 30) * pattern.thickness;
    const rr = r - Math.max(1, ctx.lineWidth);
    for (const step of pattern.lines) {
        up += step; down -= step;
        const x1 = cx + rr * Math.cos(up * Math.PI), y1 = cy + rr * Math.sin(up * Math.PI);
        const x2 = cx + rr * Math.cos(down * Math.PI), y2 = cy + rr * Math.sin(down * Math.PI);
        let theta;
        if (x2 > x1) theta = Math.atan((y2 - y1) / (x2 - x1));
        else if (x2 === x1) theta = Math.PI / 2;
        else theta = Math.atan((y2 - y1) / (x2 - x1)) - Math.PI;
        const m = rr * pattern.factor;
        const dx = Math.sin(theta) * m, dy = Math.cos(theta) * m;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.bezierCurveTo(x1 + dx, y1 - dy, x2 + dx, y2 - dy, x2, y2);
        ctx.stroke();
    }
}

// A highlight that is not quite round: the context is scaled about the
// highlight's centre so the radial gradient becomes an ellipse, then restored.
function drawHighlight(ctx, cx, cy, r, { ox, oy, rx, ry, peak, stops }) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(0, r - 0.5), 0, 2 * Math.PI, true);
    ctx.clip();
    const hx = cx + ox * r, hy = cy + oy * r;
    ctx.translate(hx, hy);
    ctx.scale(rx, ry);
    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
    for (const [at, alpha] of stops) g.addColorStop(at, `rgba(255,255,255,${(peak * alpha).toFixed(3)})`);
    ctx.fillStyle = g;
    ctx.fillRect(-r / rx * 2, -r / ry * 2, (r / rx) * 4, (r / ry) * 4);
    ctx.restore();
}

// A shallow bite at the rim: a darker crescent where the surface falls away,
// with a thin light lip on its inner edge where the new face catches light.
function drawChip(ctx, cx, cy, r, rnd, isWhite) {
    const a = rnd() * 2 * Math.PI;                 // where on the rim
    const w = r * (0.28 + rnd() * 0.22);           // bite width
    const depth = r * (0.06 + rnd() * 0.06);
    const bx = cx + Math.cos(a) * (r - depth * 0.4);
    const by = cy + Math.sin(a) * (r - depth * 0.4);
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(0, r - 0.5), 0, 2 * Math.PI, true);
    ctx.clip();
    ctx.beginPath();
    ctx.ellipse(bx, by, w, depth * 2, a, 0, 2 * Math.PI);
    ctx.fillStyle = isWhite ? 'rgba(90,85,75,0.28)' : 'rgba(0,0,0,0.55)';
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(bx - Math.cos(a) * depth * 0.9, by - Math.sin(a) * depth * 0.9, w * 0.85, depth * 1.2, a, 0, 2 * Math.PI);
    ctx.strokeStyle = isWhite ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.14)';
    ctx.lineWidth = Math.max(0.6, r / 40);
    ctx.stroke();
    ctx.restore();
}

// A hairline: a short bezier in the lower half of the disc, away from the
// highlight, drawn dark with a fainter light stroke one pixel off so it reads
// as a groove rather than a pen mark.
function drawCrack(ctx, cx, cy, r, rnd, isWhite) {
    const a0 = Math.PI * (0.15 + rnd() * 0.7);     // start angle, lower half
    const len = r * (0.45 + rnd() * 0.5);
    const x1 = cx + Math.cos(a0) * r * 0.85, y1 = cy + Math.sin(a0) * r * 0.85;
    const dir = a0 + Math.PI + (rnd() - 0.5) * 0.9;
    const x2 = x1 + Math.cos(dir) * len, y2 = y1 + Math.sin(dir) * len;
    const bend = (rnd() - 0.5) * r * 0.5;
    const nx = -Math.sin(dir) * bend, ny = Math.cos(dir) * bend;
    const c1x = x1 + (x2 - x1) * 0.35 + nx, c1y = y1 + (y2 - y1) * 0.35 + ny;
    const c2x = x1 + (x2 - x1) * 0.7 - nx * 0.5, c2y = y1 + (y2 - y1) * 0.7 - ny * 0.5;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(0, r - 1), 0, 2 * Math.PI, true);
    ctx.clip();
    ctx.lineCap = 'round';
    const lw = Math.max(0.5, r / 55);
    ctx.lineWidth = lw;
    ctx.strokeStyle = isWhite ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.12)';
    ctx.beginPath();
    ctx.moveTo(x1 + 0.7, y1 + 0.7);
    ctx.bezierCurveTo(c1x + 0.7, c1y + 0.7, c2x + 0.7, c2y + 0.7, x2 + 0.7, y2 + 0.7);
    ctx.stroke();
    ctx.strokeStyle = isWhite ? 'rgba(70,65,55,0.45)' : 'rgba(0,0,0,0.7)';
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(c1x, c1y, c2x, c2y, x2, y2);
    ctx.stroke();
    ctx.restore();
}

function drawStone(ctx, stone, board) {
    const r = board.stoneRadius;
    const cx = board.getX(stone.x) - board.ls;
    const cy = board.getY(stone.y) - board.ls;
    const isWhite = stone.c === WGo.W;
    const rnd = rngFor(board.stoneSeed >>> 0, stone.x, stone.y);

    // Draw order of the random numbers is fixed: pattern, angle, highlight
    // (4), flaw roll, then whatever the flaw needs.
    const pattern = SHELL_PATTERNS[Math.floor(rnd() * SHELL_PATTERNS.length)];
    const angle = rnd() * 2;
    const hl = {
        ox: -0.42 + (rnd() - 0.5) * 0.16,
        oy: -0.42 + (rnd() - 0.5) * 0.16,
        rx: 0.85 + rnd() * 0.35,
        ry: 0.85 + rnd() * 0.35,
    };
    const strength = 0.75 + rnd() * 0.35;
    const flawRoll = rnd();

    ctx.beginPath();
    ctx.fillStyle = isWhite ? '#aaa' : '#000';
    ctx.arc(cx, cy, Math.max(0, r - 0.5), 0, 2 * Math.PI, true);
    ctx.fill();

    if (isWhite) {
        drawShellLines(ctx, cx, cy, r, angle, pattern);
        drawHighlight(ctx, cx, cy, r, {
            ...hl, peak: 0.9 * strength,
            stops: [[0, 1], [0.35, 0.85], [1, 0]],
        });
    } else {
        // Slate: a soft lift toward the lower right, then the highlight.
        const g1 = ctx.createRadialGradient(cx + 0.4 * r, cy + 0.4 * r, 0, cx + 0.5 * r, cy + 0.5 * r, r);
        g1.addColorStop(0, 'rgba(32,32,32,1)');
        g1.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.beginPath();
        ctx.fillStyle = g1;
        ctx.arc(cx, cy, Math.max(0, r - 0.5), 0, 2 * Math.PI, true);
        ctx.fill();
        drawHighlight(ctx, cx, cy, r, {
            ...hl, peak: 0.28 * strength,
            stops: [[0, 1], [0.4, 0.55], [1, 0]],
        });
    }

    if (flawRoll < CRACK_RATE) drawCrack(ctx, cx, cy, r, rnd, isWhite);
    else if (flawRoll < CRACK_RATE + RIM_CHIP_RATE) drawChip(ctx, cx, cy, r, rnd, isWhite);
}

/** The handler board.js hands to WGo.Board. `board.stoneSeed` picks the set. */
export const FLAWED_SHELL = {
    stone: { draw(stone, board) { drawStone(this, stone, board); } },
    get shadow() { return WGo.Board.drawHandlers.SHELL.shadow; },
};
