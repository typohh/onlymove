// js/sgf.js

/**
 * Tsumego id from its SGF filename: strips the trailing `.sgf` extension. The
 * id is the symmetry-canonical 8-hex hash the corpus, shards, and reports all
 * key on. Centralized so the ~15 `filename.replace('.sgf', '')` call sites
 * can't drift. Null-tolerant: a falsy filename
 * (e.g. an optional-chained `problem?.filename`) is returned unchanged.
 */
export function tsumegoIdOf(filename) {
  return filename ? filename.replace(/\.sgf$/, '') : filename;
}

/**
 * Parses the score from an SGF comment.
 * @param {string} comment - The comment string.
 * @returns {number|null} - The score as a float, or null if not found.
 */
export function getScoreFromComment(comment) {
  if (!comment) {
    return null;
  }
  // Extracts score like "3.1" from "86.4%/3.1"
  const match = comment.match(/%\/([\d.-]+)/);
  if (match && match[1]) {
    return parseFloat(match[1]);
  }
  return null;
}

/**
 * Parses the visit count an SGF comment's estimate rests on, from the third
 * field of "<winrate>%/<score>/<visits>".
 *
 * A comment carrying only the first two fields predates the field, and nothing
 * was ever written below the 30-visit floor, so it reports 30 — that is a fact
 * about the corpus, not a fallback. Anything that is not an evaluation reports
 * null. Mirrors parse_eval_visits in utils/katago/add_variations.py.
 * @param {string} comment - The comment string.
 * @returns {number|null} - Visits behind the estimate, or null if absent.
 */
export function getVisitsFromComment(comment) {
  if (!comment) {
    return null;
  }
  const match = comment.match(/[\d.-]+%\/[\d.-]+\/(\d+)/);
  if (match) {
    return parseInt(match[1], 10);
  }
  return comment.match(/-?[\d.]+%\/-?[\d.]+/) ? 30 : null;
}

/**
 * The VISIT WEIGHTS of a problem's root variations, in SGF order — one integer
 * per node, for the availability null's mass arithmetic.
 *
 * The whole weight rule lives here rather than at the call site: the comment
 * regex belongs to getVisitsFromComment above, and the "a node carrying no
 * evaluation weighs nothing" half belongs beside it, so neither can be restated
 * differently somewhere else. Mirrors what package_for_pwa's
 * _read_shard_variation_visits builds from the same SGFs through
 * parse_eval_visits, which likewise reports 0 for a non-evaluation comment.
 * @param {Array} nodes - The decision node's children, in SGF document order.
 * @returns {Array<number>|null} - One non-negative integer per node.
 */
export function variationVisitWeights(nodes) {
  if (!Array.isArray(nodes)) return null;
  return nodes.map((n) => getVisitsFromComment(n && n.comment) ?? 0);
}

/**
 * Parses the winrate from an SGF comment. Always Black's perspective
 * (bots/analysis.cfg sets reportAnalysisWinratesAs = BLACK).
 * @param {string} comment - The comment string.
 * @returns {number|null} - Black-perspective winrate in [0, 100], or null if absent.
 */
export function getWinrateFromComment(comment) {
  if (!comment) {
    return null;
  }
  // Extracts winrate like "86.4" from "86.4%/3.1"
  const match = comment.match(/([\d.]+)%\//);
  if (match && match[1]) {
    return parseFloat(match[1]);
  }
  return null;
}

/**
 * Formats an evaluation as an SGF branch comment: "<winrate>%/<score>/<visits>".
 *
 * `winrate` is a fraction in [0, 1] from BLACK's perspective, which is how both
 * engines report it (bots/analysis.cfg and analysis_server/analysis.cfg both set
 * reportAnalysisWinratesAs = BLACK), so nothing here converts a frame. Mirrors
 * format_eval_comment in utils/katago/add_variations.py — the two must agree,
 * because a comment written here is read back by the same parsers that read the
 * pipeline's.
 * @param {number} winrate - Black-perspective winrate in [0, 1].
 * @param {number} score - Black-perspective score lead.
 * @param {number} visits - Visits the estimate rests on.
 * @returns {string} - The comment.
 */
export function formatEvalComment(winrate, score, visits) {
  return `${(winrate * 100).toFixed(1)}%/${score.toFixed(1)}/${Math.trunc(visits)}`;
}

/**
 * Writes an evaluation onto `node`, but NEVER DOWNGRADES one.
 *
 * A comment is replaced only by an estimate resting on MORE visits than the one
 * already there, so a shallow answer can never overwrite a deeper one and
 * re-running a shallower pass over the same node does nothing. A node carrying
 * no evaluation counts as zero visits, so a first write always lands. Mirrors
 * set_eval_comment in utils/katago/add_variations.py, which is the pipeline's
 * only writer; this is the client's.
 * @param {object} node - The node to write onto.
 * @param {number} winrate - Black-perspective winrate in [0, 1].
 * @param {number} score - Black-perspective score lead.
 * @param {number} visits - Visits the estimate rests on.
 * @returns {boolean} - Whether the comment was written.
 */
export function setEvalComment(node, winrate, score, visits) {
  if (Math.trunc(visits) <= (getVisitsFromComment(node.comment) ?? 0)) {
    return false;
  }
  node.comment = formatEvalComment(winrate, score, visits);
  return true;
}

/**
 * Whether `node` is a branch the pipeline played out to a DOUBLE PASS.
 *
 * Follows children[0]: the pipeline appends children in its candidate order and
 * always recurses into the first, so the first child IS the continuation it
 * followed. Measured on rebuilt 13x13 files this holds for 120/120 answers and
 * 120/120 best refutations, and accepts 0 of 10,645 other root variations.
 *
 * Two cheaper-looking tests are both wrong, and were measured to be. "A pass
 * appears somewhere in the subtree" is the deduction behind two corpus purges.
 * "A pass-PAIR appears somewhere" is not much better: the pipeline's shallow
 * sweep writes a move and its best response, so a pass answered by a pass
 * manufactures the shape with nothing played out — 32,445 extra children. And
 * following the highest-visit child drifts off the line entirely, because the
 * engine ranks its candidates by LCB rather than raw visits.
 *
 * This says the line was PLAYED OUT, not that it resolves the way the problem
 * claims — 7.5% of principal lines end with the solver losing. Whether a
 * resolved line is a win is read from the evaluation, not from here.
 *
 * Mirrors subtree_reaches_conclusion in utils/katago/add_variations.py; the two
 * must agree, and tests/test_js_twins_sgf.py holds them to it.
 */
export function subtreeReachesConclusion(node) {
    if (!node) return false;
    const isPass = (n) => !!(n.move && n.move.pass);
    let n = node;
    let prevWasPass = isPass(node);
    // The pipeline caps a variation at MAX_VARIATION_DEPTH = 800 plies.
    for (let step = 0; step < 802; step++) {
        const kids = (n.children || []).filter(c => c.move);
        if (kids.length === 0) return false;
        n = kids[0];
        const cur = isPass(n);
        if (cur && prevWasPass) return true;
        prevWasPass = cur;
    }
    return false;
}

// Standard Go board notation: column letters skip 'I' (A..H, J..T), row is
// 1-based from the BOTTOM while WGo's y=0 is the TOP row (so row = size - y).
// toBoardNotation and boardNotationToXy are exact inverses over this alphabet;
// they live together here so the COL_LETTERS table and the row flip can't drift.
// Exported for the board's edge coordinate labels (board.js).
export const COL_LETTERS = 'ABCDEFGHJKLMNOPQRST';

// WGo {x, y} → board notation (e.g. {x:2,y:6} on 9x9 → "C3"); null for an
// off-board coordinate. Used by the live click path to report the picked move.
export function toBoardNotation(x, y, size) {
    if (x < 0 || y < 0 || x >= size || y >= size || x >= COL_LETTERS.length) return null;
    return COL_LETTERS[x] + (size - y);
}

// Board notation → WGo {x, y}; the inverse of toBoardNotation. Used by
// account-sync's field fill to map a reported move (e.g. "C3") back to WGo
// coordinates. null for a malformed or off-board move.
export function boardNotationToXy(move, size) {
    if (typeof move !== 'string') return null;
    const m = move.toUpperCase().match(/^([A-HJ-T])(\d{1,2})$/);
    if (!m) return null;
    const x = COL_LETTERS.indexOf(m[1]);
    const y = size - parseInt(m[2]);
    if (x < 0 || x >= size || y < 0 || y >= size) return null;
    return { x, y };
}

/**
 * Checks if a move is correct.
 *
 * The answer IS variation 0 of the problem's decision node. That is the corpus
 * contract, fixed by construction and not inferred: construct_initial_sgf
 * writes the solution ProblemFinder verified as the first child and every later
 * annotation pass only appends, so nothing can displace it. Every backend stage
 * reads the answer positionally — extract_activations' correct_idx,
 * benchmark_bots' grading, row 0 of the published per-cell block — so reading
 * it positionally here is what keeps the client on the same move they do.
 *
 * Do NOT re-derive this from the evaluation comments. They are a shallower
 * annotation than the search that verified the solution, so on any file where
 * the two disagree, picking by comment puts the app on a different answer from
 * the ratings, the axis vectors and the availability null.
 * @param {object} moveNode - The node of the move to check.
 * @param {Array} allChildren - All sibling nodes, in SGF order.
 * @returns {boolean} - True if the move is correct.
 */
export function isCorrectMove(moveNode, allChildren) {
    return !!allChildren && allChildren.length > 0 && allChildren[0] === moveNode;
}

/**
 * Transforms coordinates based on the given transformation.
 * @param {number} x - Original x coordinate (0-based).
 * @param {number} y - Original y coordinate (0-based).
 * @param {number} transformation - Type of transformation (0-7).
 * @param {number} size - Board edge length (9, 13, 19, …).
 * @returns {object} - Transformed {x, y}.
 */
function transformCoordinates(x, y, transformation, size) {
    switch (transformation) {
        case 0: // rotate0
            return { x, y };
        case 1: // rotate90
            return { x: y, y: size - 1 - x };
        case 2: // rotate180
            return { x: size - 1 - x, y: size - 1 - y };
        case 3: // rotate270
            return { x: size - 1 - y, y: x };
        case 4: // reflectHorizontal
            return { x: x, y: size - 1 - y };
        case 5: // reflectVertical
            return { x: size - 1 - x, y: y };
        case 6: // reflectDiagonal1
            return { x: y, y: x };
        case 7: // reflectDiagonal2
            return { x: size - 1 - y, y: size - 1 - x };
        default:
            return { x, y };
    }
}

// Inverse of transformCoordinates for the same D4 transformation index: maps a
// coordinate in the SERVED (reoriented) frame back to the canonical
// orientation-0 frame. rotate90 (1) and rotate270 (3) are each other's
// inverses; every other element of the group is self-inverse. Used by the
// report path so the `move` sent to the server is canonical regardless of the
// random orientation a repeat was served at.
const INVERSE_TRANSFORMATION = [0, 3, 2, 1, 4, 5, 6, 7];

export function toCanonicalXy(x, y, orientation, size) {
    const t = Number.isInteger(orientation) && orientation >= 0 && orientation <= 7
        ? INVERSE_TRANSFORMATION[orientation] : 0;
    return transformCoordinates(x, y, t, size);
}

function transformKifuTree(node, transformation, size) {
    if (node.move) {
        const transformed = transformCoordinates(node.move.x, node.move.y, transformation, size);
        node.move.x = transformed.x;
        node.move.y = transformed.y;
    }
    if (node.children) {
        node.children.forEach(child => transformKifuTree(child, transformation, size));
    }
}

/**
 * Reorients the SGF by applying a transformation.
 * Board size is read from the parsed kifu so this works for 9x9 and 13x13 alike.
 * @param {string} sgf - Original SGF string.
 * @param {number|null} forcedOrientation - Optional orientation index (0-7). If null, random is used.
 * @returns {object} - Object containing { sgf: string, orientation: number }.
 */
export function reorientSgf(sgf, forcedOrientation = null) {
    let transformation;
    if (typeof forcedOrientation === 'number' && forcedOrientation >= 0 && forcedOrientation <= 7) {
        transformation = forcedOrientation;
    } else {
        transformation = Math.floor(Math.random() * 8);
    }

    const kifu = WGo.Kifu.fromSgf(sgf);
    transformKifuTree(kifu.root, transformation, kifu.size);

    return {
        sgf: kifu.toSgf(),
        orientation: transformation
    };
}
