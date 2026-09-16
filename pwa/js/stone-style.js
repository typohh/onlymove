// js/stone-style.js
//
// The stone draw handler: WGo's shell stones with per-stone variation. Each
// stone gets its own highlight (offset, shape and strength move within a
// band) and its own NUMBER of flaws — each one a smooth-edged chip at the rim
// or on the face, or a jagged crack running in from the rim — so a board
// reads as a set of real stones rather than one stamp, and a worn set carries
// more than one defect per stone.
//
// Everything is a PURE FUNCTION of (board.stoneSeed, board.stoneFlawRate, x,
// y). WGo redraws every object on each repaint, so a stone must come out
// identical draw after draw or it flickers; and the seed is the PROBLEM's id
// (board.js sets it), so a problem always shows the same stones in the same
// places, and two problems differ. Nothing here reads Math.random.
//
// A handler is `{ stone: { draw(stone, board) }, shadow }` with `this` bound
// to the 2D context (see WGo.Board.drawHandlers). The shadow is WGo's own.

// How worn the set is, which board.js sets per repaint from the user's
// current run of problems without a break (flawRateForStreak). It is the
// MEAN of a per-stone count, not a per-stone share: each stone samples its
// own Poisson number of flaws independently, so wear makes stones carry
// SEVERAL defects rather than making a larger fraction carry exactly one.
// The set starts CLEAN — no streak, no flaws — and wears in linearly, so by
// FULL_AT problems in one sitting nearly every stone has a defect and many
// have more. FLAWS_PER_WEAR is the knob for how disfigured that end is.
const FLAW_RATE_WORN = 0.97;
const FULL_AT = 200;
const FLAWS_PER_WEAR = 2;        // mean flaws per stone at full wear
const MAX_FLAWS = 6;             // the Poisson tail past this is far rarer than a board is wide
const MAX_RIM_CHIPS = 2;         // a real stone with three chips on its rim is not seen
export function flawRateForStreak(streak) {
    const t = Math.min(1, Math.max(0, streak) / FULL_AT);
    return FLAW_RATE_WORN * t;
}
// What a flaw is, given that there is one. Rolled independently for EVERY
// flaw on the stone, so one stone can carry a crack, several bites and a pit.
// Chips are mostly at the rim: nine in ten a bite that cuts the outline, the
// rest a shallow pit on the face.
const CRACK_SHARE = 0.46;
const BITE_SHARE = 0.485;        // the remainder is a face pit
const V_BITE_SHARE = 0.2;        // of the rim bites, a small V notch rather than an oval
const SHALLOW_RIM_SHARE = 0.1;   // of the oval bites, a large facet on the face with a narrow notch
const COMPOUND_SHARE = 0.4;      // of the oval bites, one to three smaller chips along the inner edge
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

// How many flaws one stone carries: a Poisson draw from the stone's own
// generator, so every stone samples its count independently. Knuth's method —
// trials until a product of uniforms drops under e^-lambda. It consumes a
// variable number of draws, which is fine because nothing after it depends on
// the stream's position.
function poisson(rnd, lambda) {
    const L = Math.exp(-lambda);
    let k = 0, p = 1;
    do { k++; p *= rnd(); } while (p > L);
    return k - 1;
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

// The stone's outline: a disc, less each bite where an ellipse centred just
// outside the rim overlaps it. Clip to the disc, then subtract one ellipse
// per bite — each as a path of a box XOR the ellipse under even-odd, which
// removes the ellipse from the box and leaves the rest of the box intact.
// Successive clips INTERSECT, so overlapping bites stay correct: putting two
// ellipses in ONE even-odd path would RE-ADMIT their overlap, where the
// containments are disc + ellipse + ellipse = three, which is odd. The same
// function clips the stone, its shadow (at the shadow's larger radius) and
// every mark on the face.
function clipOutline(ctx, cx, cy, r, bites, extra = 0) {
    const rr = Math.max(0, r - 0.5 + extra);
    ctx.beginPath();
    ctx.arc(cx, cy, rr, 0, 2 * Math.PI, true);
    ctx.clip();
    if (!bites || !bites.length) return;
    for (const bite of bites) {
        ctx.beginPath();
        ctx.rect(cx - rr - 1, cy - rr - 1, 2 * rr + 2, 2 * rr + 2);
        biteSubpath(ctx, bite, 0);
        ctx.clip('evenodd');
    }
}

// One bite's outline as a subpath: an oval centred outside the rim, or a V —
// a small notch with two points on the rim and its apex inside, closed by a
// point well outside the disc so the polygon certainly reaches past the
// edge. `grow` pads an oval outward, for the fracture line drawn around it.
function biteSubpath(ctx, bite, grow) {
    if (bite.shape === 'v') {
        const [p1, apex, p2, outer] = bite.pts;
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(apex[0], apex[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.lineTo(outer[0], outer[1]);
        ctx.closePath();
        return;
    }
    ctx.moveTo(bite.bx + bite.w + grow, bite.by);
    ctx.ellipse(bite.bx, bite.by, bite.w + grow, bite.h + grow, bite.tilt, 0, 2 * Math.PI);
}

// The broken face along one bite: a thin dark line on the stone side of the
// new edge, and a lighter one just inside it where the fracture catches light.
// Called once per bite, clipped to the stone's outline so a line crossing
// another bite's empty notch is cut away instead of floating over the board.
function drawBiteEdge(ctx, cx, cy, r, bites, i, isWhite) {
    const bite = bites[i];
    ctx.save();
    clipOutline(ctx, cx, cy, r, bites);
    const dark = isWhite ? 'rgba(120,112,98,0.45)' : 'rgba(0,0,0,0.7)';
    const light = isWhite ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.12)';
    if (bite.shape === 'v') {
        // The notch's two inner edges (never the closing side outside the
        // disc): a dark line on the edge, a light one a little further in.
        const [p1, apex, p2] = bite.pts;
        ctx.lineJoin = 'miter';
        for (const [lw, style, inset] of [[Math.max(0.7, r / 36), dark, 0], [Math.max(0.5, r / 50), light, Math.max(0.8, r / 30)]]) {
            const len = Math.hypot(apex[0] - cx, apex[1] - cy) || 1;
            const dx = (cx - apex[0]) / len * inset, dy = (cy - apex[1]) / len * inset;
            ctx.lineWidth = lw;
            ctx.strokeStyle = style;
            ctx.beginPath();
            ctx.moveTo(p1[0] + dx, p1[1] + dy);
            ctx.lineTo(apex[0] + dx, apex[1] + dy);
            ctx.lineTo(p2[0] + dx, p2[1] + dy);
            ctx.stroke();
        }
        ctx.restore();
        return;
    }
    ctx.lineWidth = Math.max(0.8, r / 30);
    ctx.strokeStyle = dark;
    ctx.beginPath();
    biteSubpath(ctx, bite, ctx.lineWidth * 0.5);
    ctx.stroke();
    ctx.lineWidth = Math.max(0.6, r / 45);
    ctx.strokeStyle = light;
    ctx.beginPath();
    biteSubpath(ctx, bite, ctx.lineWidth * 2.2);
    ctx.stroke();
    ctx.restore();
}

// The fracture surface of a top-face chip: an oval inside the rim, adjoining
// the notch, with the smooth glassy look of a conchoidal break. On slate it
// reads lighter than the matte face and carries a sharp reflection; on shell
// it is a slightly darker, translucent patch with the same reflection. The
// caller has already clipped to the stone's outline, so the oval's part in
// the notch is simply gone.
function drawFacet(ctx, cx, cy, r, bite, isWhite) {
    const { a, facet } = bite;
    const { fx, fy } = facet;
    // The shading radiates from the notch: the facet's outermost point.
    const ex = fx + Math.cos(a) * facet.D * 0.9, ey = fy + Math.sin(a) * facet.D * 0.9;
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(fx, fy, facet.A, facet.D, a + Math.PI / 2, 0, 2 * Math.PI);
    ctx.clip();
    // Base of the facet: a concave sphere, so the shading is radial from
    // the notch — brightest where the break is deepest, fading toward the
    // facet's inner edge where it runs out to the face.
    const g = ctx.createRadialGradient(ex, ey, 0, ex, ey, Math.max(facet.A, facet.D * 1.6));
    if (isWhite) {
        g.addColorStop(0, `rgba(150,145,130,${(0.45 * facet.gloss).toFixed(3)})`);
        g.addColorStop(1, `rgba(150,145,130,${(0.12 * facet.gloss).toFixed(3)})`);
    } else {
        g.addColorStop(0, `rgba(120,122,128,${(0.55 * facet.gloss).toFixed(3)})`);
        g.addColorStop(1, `rgba(70,72,78,${(0.25 * facet.gloss).toFixed(3)})`);
    }
    ctx.fillStyle = g;
    ctx.fillRect(cx - r - 1, cy - r - 1, 2 * r + 2, 2 * r + 2);
    // The reflection: a small hard ellipse off the facet's centre, toward the
    // light (upper left), the polish showing as a sharper highlight than the
    // face carries anywhere.
    const hx = fx - facet.A * 0.25, hy = fy - facet.D * 0.3;
    const hg = ctx.createRadialGradient(hx, hy, 0, hx, hy, facet.A * 0.45);
    hg.addColorStop(0, `rgba(255,255,255,${(isWhite ? 0.75 : 0.5) * facet.gloss})`);
    hg.addColorStop(0.5, `rgba(255,255,255,${(isWhite ? 0.25 : 0.15) * facet.gloss})`);
    hg.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = hg;
    ctx.fillRect(cx - r - 1, cy - r - 1, 2 * r + 2, 2 * r + 2);
    ctx.restore();
    // The facet's inner rim: a faint line where the polished surface meets
    // the matte face.
    ctx.beginPath();
    ctx.ellipse(fx, fy, facet.A, facet.D, a + Math.PI / 2, 0, 2 * Math.PI);
    ctx.lineWidth = Math.max(0.5, r / 50);
    ctx.strokeStyle = isWhite ? 'rgba(110,100,85,0.35)' : 'rgba(160,162,170,0.35)';
    ctx.stroke();
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
    // How deep the crack reads: most are faint, a few plainly there. The
    // product of two uniforms leans the draw toward the faint end.
    const depth = 0.3 + rnd() * rnd() * 0.7;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = Math.max(0.6, r / 45);
    ctx.strokeStyle = isWhite ? `rgba(255,255,255,${(0.5 * depth).toFixed(3)})` : `rgba(255,255,255,${(0.14 * depth).toFixed(3)})`;
    trace(pts, 0.8, 0.8);
    if (branch) trace(branch, 0.8, 0.8);
    ctx.strokeStyle = isWhite ? `rgba(70,65,55,${(0.55 * depth).toFixed(3)})` : `rgba(0,0,0,${(0.8 * depth).toFixed(3)})`;
    trace(pts, 0, 0);
    if (branch) trace(branch, 0, 0);
}

// Every number a stone needs, drawn from its own generator. Computed once per
// draw and shared by the stone and its shadow, which WGo draws on separate
// layers through separate calls; because it is a pure function of the seed,
// the wear rate and the intersection, the two calls agree.
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
    // First how MANY flaws the stone has, then what each one is. The wear
    // rate is the Poisson mean scale; every stone draws its own count, so the
    // set wears by accumulating defects rather than by every stone acquiring
    // one. Each flaw then rolls its kind independently.
    const lambda = FLAWS_PER_WEAR * (board.stoneFlawRate ?? 0);
    const n = Math.min(MAX_FLAWS, poisson(rnd, lambda));
    const flaws = [];
    // Chips keep well apart: on a real stone two chips near each other are
    // practically never seen. Rim chips claim an arc of the rim (their angle
    // and half-span, the facet's width counted where it is wider than the
    // notch) plus a wide clear margin either side; face pits claim a disc
    // with a margin. A new chip tries a few positions and is dropped if none
    // is free, so a worn stone runs out of room before it crowds. A compound
    // chip's sub-chips are part of their parent and claim nothing of their own.
    const arcs = [];    // [angle, halfSpan]
    const discs = [];   // [x, y, radius]
    const arcGap = 1.1;                     // radians of clear rim between chips
    const arcFree = (a, half) => arcs.every(([b, bh]) => {
        let d = Math.abs(a - b) % (2 * Math.PI);
        if (d > Math.PI) d = 2 * Math.PI - d;
        return d > half + bh + arcGap;
    });
    const discFree = (x, y, rad) => discs.every(([px, py, pr]) => Math.hypot(x - px, y - py) > 2 * (rad + pr))
        && arcs.every(([b, bh]) => {
            // A pit must also clear the rim chips' facets, which reach inward.
            const fx = cx + Math.cos(b) * r * 0.8, fy = cy + Math.sin(b) * r * 0.8;
            return Math.hypot(x - fx, y - fy) > 2 * rad + bh * r + r * 0.25;
        });
    const freeAngle = (half) => {
        if (arcs.length >= MAX_RIM_CHIPS) return null;
        for (let t = 0; t < 6; t++) { const a = rnd() * 2 * Math.PI; if (arcFree(a, half)) return a; }
        return null;
    };
    for (let i = 0; i < n; i++) {
        const kindRoll = rnd();
        if (kindRoll < CRACK_SHARE) {
            flaws.push({ kind: 'crack' });
        } else if (kindRoll < CRACK_SHARE + BITE_SHARE) {
            if (rnd() < V_BITE_SHARE) {
                // A small V notch: two rim points a little apart, the apex
                // inside and a little off-centre. Much smaller than an oval.
                const half = 0.05 + rnd() * 0.08;           // half-width, radians
                const depth = r * (0.05 + rnd() * 0.09);
                const skew = (rnd() - 0.5) * 0.6;
                const a = freeAngle(half);
                if (a == null) continue;
                arcs.push([a, half]);
                const rimPt = (da) => [cx + Math.cos(a + da) * (r + 0.5), cy + Math.sin(a + da) * (r + 0.5)];
                const am = a + skew * half;
                flaws.push({ kind: 'bite', shape: 'v', pts: [
                    rimPt(-half),
                    [cx + Math.cos(am) * (r - depth), cy + Math.sin(am) * (r - depth)],
                    rimPt(half),
                    [cx + Math.cos(a) * r * 1.3, cy + Math.sin(a) * r * 1.3],
                ] });
            } else {
                // An ellipse centred a little outside the rim; its overlap
                // with the disc is what goes missing. Sizes vary widely.
                // A chip is always a piece of a sphere. Usually the sphere
                // is centred outside the stone and takes a bite out of the
                // rim; rarely (SHALLOW_RIM_SHARE) it sits mostly ON the face,
                // a large dome-shaped facet that crosses the rim only along a
                // narrow notch.
                const shallowRim = rnd() < SHALLOW_RIM_SHARE;
                // ONE construction for the notch and the facet, so their
                // edges meet at the rim. The facet ellipse is placed first:
                // centre on the chip's bearing at radius r - D + p, semi-axes
                // A along the rim and D into the stone, protruding p past the
                // rim. Its chord where it crosses the rim is then
                //   c = A * sqrt(1 - ((D - p) / D)^2),
                // and the notch oval — centred outside at r + h*k with
                // semi-axes w, h — is given the SAME chord, w = c / sqrt(1 -
                // k^2). Half the ordinary chips came off the underside and
                // have no facet; those size their notch directly. The
                // shallow-rim chip is a near-circular facet with a small p,
                // which is what makes its notch narrow.
                const k = 0.3 + rnd() * 0.35;
                let facet = null, w, h;
                const hasFacet = shallowRim || rnd() < 0.5;
                if (hasFacet) {
                    const D = r * (shallowRim ? 0.3 + rnd() * 0.2 : 0.14 + rnd() * 0.18);
                    const A = D * (shallowRim ? 0.9 + rnd() * 0.4 : 1.2 + rnd() * 1.0);
                    const pr = D * (shallowRim ? 0.08 + rnd() * 0.12 : 0.3 + rnd() * 0.3);
                    const c = A * Math.sqrt(Math.max(0, 1 - ((D - pr) / D) ** 2));
                    facet = { fr: r - D + pr, A, D, gloss: 0.6 + rnd() * 0.4 };
                    w = c / Math.sqrt(1 - k * k);
                    // The cut is a shallow cap: depth into the stone tied to
                    // the chord, never a narrow deep gouge.
                    h = w * (0.25 + rnd() * 0.35);
                } else {
                    w = r * (0.16 + rnd() * 0.32);
                    h = w * (0.25 + rnd() * 0.35);
                }
                const dist = r + h * k;
                const span = Math.asin(Math.min(1, Math.max(w, facet ? facet.A : 0) / r));
                const a = freeAngle(span);
                if (a == null) continue;
                arcs.push([a, span]);
                if (facet) {
                    facet.fx = cx + Math.cos(a) * facet.fr;
                    facet.fy = cy + Math.sin(a) * facet.fr;
                }
                const main = { kind: 'bite', shape: 'oval', bx: cx + Math.cos(a) * dist, by: cy + Math.sin(a) * dist, w, h, tilt: a + Math.PI / 2, a, facet };
                flaws.push(main);
                // Chips seldom come off cleanly: some carry a run of smaller
                // chips along their inner edge, each a smaller sphere biting
                // a little deeper, with a smaller facet where the parent has
                // one.
                if (!shallowRim && rnd() < COMPOUND_SHARE) {
                    const count = 1 + Math.floor(rnd() * 2);
                    for (let j = 0; j < count; j++) {
                        // Within the parent's own width, so the cluster reads
                        // as ONE ragged chip and never as neighbours.
                        const along = (rnd() - 0.5) * 0.9;
                        const sw = w * (0.2 + rnd() * 0.2);
                        const sh = h * (0.4 + rnd() * 0.4);
                        // On the parent's inner edge: step inward from the parent's
                        // centre by its depth, then sideways along its width.
                        const ax = Math.cos(a), ay = Math.sin(a), tx = -ay, ty = ax;
                        const inward = h * (0.7 + rnd() * 0.3);
                        const sbx = main.bx - ax * inward + tx * along * w;
                        const sby = main.by - ay * inward + ty * along * w;
                        // A sub-chip's facet is centred on the sub-notch
                        // itself with the same half-width, so the two ovals
                        // meet where the sub-notch is widest.
                        const sfacet = facet ? {
                            fx: sbx - ax * sh * 0.4, fy: sby - ay * sh * 0.4,
                            A: sw, D: facet.D * (0.35 + rnd() * 0.3),
                            gloss: facet.gloss,
                        } : null;
                        flaws.push({ kind: 'bite', shape: 'oval', bx: sbx, by: sby, w: sw, h: sh, tilt: a + Math.PI / 2 + (rnd() - 0.5) * 0.5, a, facet: sfacet, sub: true });
                    }
                }
            }
        } else {
            const w = r * (0.14 + rnd() * 0.16);
            const h = w * (0.55 + rnd() * 0.4);
            let placed = null;
            for (let t = 0; t < 6 && !placed; t++) {
                const a = rnd() * 2 * Math.PI;
                const dist = r * (0.15 + rnd() * 0.5);
                const x = cx + Math.cos(a) * dist, y = cy + Math.sin(a) * dist;
                if (discFree(x, y, w)) placed = [x, y];
            }
            if (!placed) continue;
            discs.push([placed[0], placed[1], w]);
            flaws.push({ kind: 'pit', bx: placed[0], by: placed[1], w, h, tilt: rnd() * Math.PI });
        }
    }
    return { r, cx, cy, rnd, pattern, angle, hl, strength, flaws, bites: flaws.filter((f) => f.kind === 'bite') };
}

function drawStone(ctx, stone, board) {
    const p = stoneParams(board, stone);
    const { r, cx, cy, rnd, pattern, angle, hl, strength, flaws, bites } = p;
    const isWhite = stone.c === WGo.W;

    ctx.save();
    clipOutline(ctx, cx, cy, r, bites);

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

    for (const b of bites) if (b.facet) drawFacet(ctx, cx, cy, r, b, isWhite);
    for (const fl of flaws) {
        if (fl.kind === 'crack') drawCrack(ctx, cx, cy, r, rnd, isWhite);
        else if (fl.kind === 'pit') drawPit(ctx, cx, cy, r, fl, isWhite);
    }
    ctx.restore();

    for (let i = 0; i < bites.length; i++) drawBiteEdge(ctx, cx, cy, r, bites, i, isWhite);
}

// WGo's own shadow, clipped to the stone's outline (at the shadow's larger
// radius) so a bitten rim casts a bitten shadow.
function drawShadow(ctx, stone, board) {
    const { r, cx, cy, bites } = stoneParams(board, stone);
    if (!bites.length) { WGo.Board.drawHandlers.SHELL.shadow.draw.call(ctx, stone, board); return; }
    ctx.save();
    clipOutline(ctx, cx, cy, r, bites, r);   // generous: the blur is well under r
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
