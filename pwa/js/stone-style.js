// js/stone-style.js
//
// The stone draw handler: WGo's shell stones with per-stone variation. Each
// stone gets its own highlight (offset, shape and strength move within a
// band) and a chance of a flaw — a smooth-edged chip at the rim or on the
// face, or a jagged crack running in from the rim — so a board reads as a set
// of real stones rather than one stamp.
//
// Everything is a PURE FUNCTION of (board.stoneSeed, x, y). WGo redraws every
// object on each repaint, so a stone must come out identical draw after draw
// or it flickers; and the seed is the PROBLEM's id (board.js sets it), so a
// problem always shows the same stones in the same places, and two problems
// differ. Nothing here reads Math.random.
//
// A handler is `{ stone: { draw(stone, board) }, shadow }` with `this` bound
// to the 2D context (see WGo.Board.drawHandlers). The shadow is WGo's own.

const CRACK_RATE = 0.3;          // stones with a crack from the rim
// Chips are mostly at the rim: nine in ten a bite that cuts the outline, the
// rest a shallow pit on the face.
const BITE_RATE = 0.315;         // stones with a bite out of the rim (not a circle any more)
const PIT_RATE = 0.035;          // stones with a shallow pit on the face
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
    const hx = cx + ox * r, hy = cy + oy * r;
    ctx.translate(hx, hy);
    ctx.scale(rx, ry);
    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
    for (const [at, alpha] of stops) g.addColorStop(at, `rgba(255,255,255,${(peak * alpha).toFixed(3)})`);
    ctx.fillStyle = g;
    ctx.fillRect(-r / rx * 2, -r / ry * 2, (r / rx) * 4, (r / ry) * 4);
    ctx.restore();
}

// A pit: a smooth-edged, shallow loss of surface on the face. Never jagged —
// a chip is a conchoidal facet, and the eye tells it from a crack by that
// smoothness. Faint on purpose: a darker oval and a thin light lip.
function drawPit(ctx, cx, cy, r, pit, isWhite) {
    const { bx, by, w, h, tilt } = pit;
    ctx.beginPath();
    ctx.ellipse(bx, by, w, h, tilt, 0, 2 * Math.PI);
    ctx.fillStyle = isWhite ? 'rgba(90,85,75,0.12)' : 'rgba(0,0,0,0.25)';
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(bx - w * 0.08, by - h * 0.3, w * 0.9, h * 0.75, tilt, Math.PI * 1.05, Math.PI * 1.95);
    ctx.strokeStyle = isWhite ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.08)';
    ctx.lineWidth = Math.max(0.6, r / 40);
    ctx.stroke();
}

// The stone's outline: a disc, less a bite where an ellipse centred just
// outside the rim overlaps it. Two clips: the disc, then the disc-with-
// ellipse under even-odd. The intersection is disc minus ellipse — a single
// even-odd clip would keep the ellipse's part OUTSIDE the disc, which is
// inside that path once. The same function clips the stone, its shadow (at
// the shadow's larger radius) and every mark on the face.
function clipOutline(ctx, cx, cy, r, bite, extra = 0) {
    const rr = Math.max(0, r - 0.5 + extra);
    ctx.beginPath();
    ctx.arc(cx, cy, rr, 0, 2 * Math.PI, true);
    ctx.clip();
    if (!bite) return;
    ctx.beginPath();
    ctx.arc(cx, cy, rr, 0, 2 * Math.PI, true);
    ctx.moveTo(bite.bx + bite.w, bite.by);
    ctx.ellipse(bite.bx, bite.by, bite.w, bite.h, bite.tilt, 0, 2 * Math.PI);
    ctx.clip('evenodd');
}

// The broken face along a bite: a thin dark line on the stone side of the
// new edge, and a lighter one just inside it where the fracture catches light.
function drawBiteEdge(ctx, cx, cy, r, bite, isWhite) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(0, r - 0.5), 0, 2 * Math.PI, true);
    ctx.clip();
    ctx.lineWidth = Math.max(0.8, r / 30);
    ctx.strokeStyle = isWhite ? 'rgba(120,112,98,0.45)' : 'rgba(0,0,0,0.7)';
    ctx.beginPath();
    ctx.ellipse(bite.bx, bite.by, bite.w + ctx.lineWidth * 0.5, bite.h + ctx.lineWidth * 0.5, bite.tilt, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.lineWidth = Math.max(0.6, r / 45);
    ctx.strokeStyle = isWhite ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.12)';
    ctx.beginPath();
    ctx.ellipse(bite.bx, bite.by, bite.w + ctx.lineWidth * 2.2, bite.h + ctx.lineWidth * 2.2, bite.tilt, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
}

// A crack: starts at the rim and wanders inward as a jagged polyline, each
// segment turning by a random amount, with a chance of one short side branch.
// Drawn dark with a fainter light stroke a pixel off so it reads as a groove.
function drawCrack(ctx, cx, cy, r, rnd, isWhite) {
    const a0 = rnd() * 2 * Math.PI;                // where on the rim it starts
    let x = cx + Math.cos(a0) * (r - 0.5), y = cy + Math.sin(a0) * (r - 0.5);
    let heading = a0 + Math.PI + (rnd() - 0.5) * 0.8;  // inward, roughly
    const segments = 4 + Math.floor(rnd() * 4);
    let step = r * (0.16 + rnd() * 0.12);
    const pts = [[x, y]];
    let branch = null;
    for (let i = 0; i < segments; i++) {
        heading += (rnd() - 0.5) * 1.3;
        x += Math.cos(heading) * step;
        y += Math.sin(heading) * step;
        pts.push([x, y]);
        step *= 0.85;
        if (!branch && i >= 1 && rnd() < 0.35) {
            const bh = heading + (rnd() < 0.5 ? 1 : -1) * (0.7 + rnd() * 0.6);
            const bl = step * (0.8 + rnd() * 0.6);
            branch = [[x, y], [x + Math.cos(bh) * bl * 0.6, y + Math.sin(bh) * bl * 0.6],
                      [x + Math.cos(bh + (rnd() - 0.5)) * bl, y + Math.sin(bh + (rnd() - 0.5)) * bl]];
        }
    }
    const trace = (line, dx, dy) => {
        ctx.beginPath();
        ctx.moveTo(line[0][0] + dx, line[0][1] + dy);
        for (let i = 1; i < line.length; i++) ctx.lineTo(line[i][0] + dx, line[i][1] + dy);
        ctx.stroke();
    };
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = Math.max(0.6, r / 45);
    ctx.strokeStyle = isWhite ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.14)';
    trace(pts, 0.8, 0.8);
    if (branch) trace(branch, 0.8, 0.8);
    ctx.strokeStyle = isWhite ? 'rgba(70,65,55,0.55)' : 'rgba(0,0,0,0.75)';
    trace(pts, 0, 0);
    if (branch) trace(branch, 0, 0);
}

// Every number a stone needs, drawn from its own generator in a FIXED order.
// Computed once per draw and shared by the stone and its shadow, which WGo
// draws on separate layers through separate calls.
function stoneParams(board, stone) {
    const r = board.stoneRadius;
    const cx = board.getX(stone.x) - board.ls;
    const cy = board.getY(stone.y) - board.ls;
    const rnd = rngFor(board.stoneSeed >>> 0, stone.x, stone.y);
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
    let bite = null, pit = null, crack = false;
    if (flawRoll < CRACK_RATE) {
        crack = true;
    } else if (flawRoll < CRACK_RATE + BITE_RATE) {
        // An ellipse centred a little outside the rim; its overlap with the
        // disc is what goes missing.
        const a = rnd() * 2 * Math.PI;
        const w = r * (0.22 + rnd() * 0.22);
        const h = r * (0.12 + rnd() * 0.12);
        const dist = r + h * (0.35 + rnd() * 0.3);
        bite = { bx: cx + Math.cos(a) * dist, by: cy + Math.sin(a) * dist, w, h, tilt: a + Math.PI / 2 };
    } else if (flawRoll < CRACK_RATE + BITE_RATE + PIT_RATE) {
        const a = rnd() * 2 * Math.PI;
        const w = r * (0.14 + rnd() * 0.16);
        const h = w * (0.55 + rnd() * 0.4);
        const dist = r * (0.15 + rnd() * 0.5);
        pit = { bx: cx + Math.cos(a) * dist, by: cy + Math.sin(a) * dist, w, h, tilt: rnd() * Math.PI };
    }
    return { r, cx, cy, rnd, pattern, angle, hl, strength, bite, pit, crack };
}

function drawStone(ctx, stone, board) {
    const p = stoneParams(board, stone);
    const { r, cx, cy, rnd, pattern, angle, hl, strength, bite, pit, crack } = p;
    const isWhite = stone.c === WGo.W;

    ctx.save();
    clipOutline(ctx, cx, cy, r, bite);

    ctx.fillStyle = isWhite ? '#aaa' : '#000';
    ctx.fillRect(cx - r - 1, cy - r - 1, 2 * r + 2, 2 * r + 2);

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
        ctx.fillStyle = g1;
        ctx.fillRect(cx - r - 1, cy - r - 1, 2 * r + 2, 2 * r + 2);
        drawHighlight(ctx, cx, cy, r, {
            ...hl, peak: 0.28 * strength,
            stops: [[0, 1], [0.4, 0.55], [1, 0]],
        });
    }

    if (crack) drawCrack(ctx, cx, cy, r, rnd, isWhite);
    else if (pit) drawPit(ctx, cx, cy, r, pit, isWhite);
    ctx.restore();

    if (bite) drawBiteEdge(ctx, cx, cy, r, bite, isWhite);
}

// WGo's own shadow, clipped to the stone's outline (at the shadow's larger
// radius) so a bitten rim casts a bitten shadow.
function drawShadow(ctx, stone, board) {
    const { r, cx, cy, bite } = stoneParams(board, stone);
    if (!bite) { WGo.Board.drawHandlers.SHELL.shadow.draw.call(ctx, stone, board); return; }
    ctx.save();
    clipOutline(ctx, cx, cy, r, bite, r);   // generous: the blur is well under r
    WGo.Board.drawHandlers.SHELL.shadow.draw.call(ctx, stone, board);
    ctx.restore();
}

/** The handler board.js hands to WGo.Board. `board.stoneSeed` picks the set. */
export const FLAWED_SHELL = {
    stone: { draw(stone, board) { drawStone(this, stone, board); } },
    shadow: {
        draw(stone, board) { drawShadow(this, stone, board); },
        clear(stone, board) { WGo.Board.drawHandlers.SHELL.shadow.clear.call(this, stone, board); },
    },
};
