// js/board-wear.js
//
// Wear on the wood: hairline cracks along the grain and tea-coloured
// splotches, on a canvas layer of their own UNDER the grid, so the lines run
// over them the way they do on a real board. The layer is a WGo CanvasLayer;
// the board redraws it with every repaint.
//
// The picture is a PURE FUNCTION of (board.wearSeed, board.wearLevel). The
// seed is the SESSION's — board.js hashes the first problem of the user's
// current run without a break — so the marks stay in the same places for the
// whole sitting; the level is that run's length, and every mark is drawn
// from it monotonically: each has a birth level, then grows in length,
// spread and depth with the levels since. So consecutive problems differ
// only a little, and nothing ever moves or vanishes mid-session. Nothing here
// reads Math.random.

const FULL_AT = 200;            // the level at which the set of marks is complete
const CRACK_COUNT = 9;          // marks that CAN exist per session, each with its own birth
const SPLOTCH_COUNT = 8;

function rngFor(seed, salt) {
    let a = (seed ^ Math.imul(salt + 1, 0x9e3779b1)) >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Age of a mark in [0, 1]: 0 before its birth level, 1 once it has had a full
// third of the session to develop. Eased so growth is quick at first and slows.
function ageOf(level, birth) {
    const t = Math.max(0, level - birth) / (FULL_AT / 3);
    const u = Math.min(1, t);
    return 1 - (1 - u) * (1 - u);
}

// A crack along the grain: mostly vertical (the wood texture's grain runs
// top to bottom), wandering a little, growing from its origin in both
// directions as it ages, and darkening. Drawn as a dark hairline with a
// lighter line beside it where the split's far edge catches light.
function drawCrack(ctx, W, H, rnd, level) {
    const birth = rnd() * FULL_AT * 0.85;
    const ox = W * (0.05 + rnd() * 0.9), oy = H * (0.1 + rnd() * 0.8);
    const fullLen = H * (0.12 + rnd() * 0.3);
    const lean = (rnd() - 0.5) * 0.25;                // off-vertical
    const segs = 6 + Math.floor(rnd() * 6);
    const wobble = [];
    for (let i = 0; i < segs; i++) wobble.push((rnd() - 0.5) * 1.1);
    const depthMax = 0.25 + rnd() * 0.45;
    const age = ageOf(level, birth);
    if (age <= 0) return;
    const len = fullLen * (0.15 + 0.85 * age);
    const depth = depthMax * (0.35 + 0.65 * age);
    const half = len / 2, step = len / segs;
    const pts = [];
    let x = ox, y = oy - half, heading = Math.PI / 2 + lean;
    for (let i = 0; i <= segs; i++) {
        pts.push([x, y]);
        heading = Math.PI / 2 + lean + wobble[i % segs] * 0.5;
        x += Math.cos(heading) * step;
        y += Math.sin(heading) * step;
    }
    const lw = Math.max(0.6, W / 900);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = lw * 1.6;
    ctx.strokeStyle = `rgba(255,235,200,${(0.35 * depth).toFixed(3)})`;
    ctx.beginPath();
    ctx.moveTo(pts[0][0] + lw, pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] + lw, pts[i][1]);
    ctx.stroke();
    ctx.lineWidth = lw;
    ctx.strokeStyle = `rgba(60,35,10,${(0.9 * depth).toFixed(3)})`;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
}

// A splotch: an irregular stain, its outline a circle with a few harmonics,
// filled with a soft radial falloff. Spreads and deepens with age.
function drawSplotch(ctx, W, H, rnd, level) {
    const birth = rnd() * FULL_AT * 0.85;
    const cx = W * (0.08 + rnd() * 0.84), cy = H * (0.08 + rnd() * 0.84);
    const fullR = Math.min(W, H) * (0.03 + rnd() * 0.07);
    const harmonics = [];
    for (let k = 0; k < 4; k++) harmonics.push([2 + k, 0.05 + rnd() * 0.18, rnd() * 2 * Math.PI]);
    const tone = rnd();                                // 0 tea, 1 grey-brown
    const alphaMax = 0.10 + rnd() * 0.18;
    const age = ageOf(level, birth);
    if (age <= 0) return;
    const r = fullR * (0.3 + 0.7 * age);
    const alpha = alphaMax * (0.4 + 0.6 * age);
    ctx.save();
    ctx.beginPath();
    const N = 48;
    for (let i = 0; i <= N; i++) {
        const th = (i / N) * 2 * Math.PI;
        let rr = 1;
        for (const [k, amp, ph] of harmonics) rr += amp * Math.sin(k * th + ph);
        const px = cx + Math.cos(th) * r * rr, py = cy + Math.sin(th) * r * rr;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.clip();
    const rgb = tone < 0.5 ? '110,70,25' : '70,55,40';
    const g = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r * 1.25);
    g.addColorStop(0, `rgba(${rgb},${(alpha * 0.55).toFixed(3)})`);
    g.addColorStop(0.75, `rgba(${rgb},${alpha.toFixed(3)})`);
    g.addColorStop(1, `rgba(${rgb},${(alpha * 0.35).toFixed(3)})`);
    ctx.fillStyle = g;
    ctx.fillRect(cx - r * 1.5, cy - r * 1.5, r * 3, r * 3);
    ctx.restore();
}

/** A WGo layer that paints the wood's wear. Add it below the grid (weight < 100). */
export function makeBoardWearLayer() {
    const layer = new WGo.Board.CanvasLayer();
    layer.draw = function (board) {
        const ctx = this.context;
        const W = this.element.width, H = this.element.height;
        const seed = board.wearSeed >>> 0;
        const level = board.wearLevel || 0;
        if (level <= 0) return;
        for (let i = 0; i < CRACK_COUNT; i++) drawCrack(ctx, W, H, rngFor(seed, 100 + i), level);
        for (let i = 0; i < SPLOTCH_COUNT; i++) drawSplotch(ctx, W, H, rngFor(seed, 200 + i), level);
    };
    return layer;
}
