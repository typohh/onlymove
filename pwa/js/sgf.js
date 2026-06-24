// js/sgf.js
import { SGF_COMMENTS } from './constants.js';

/**
 * Tsumego id from its SGF filename: strips the trailing `.sgf` extension. The
 * id is the symmetry-canonical 8-hex hash the corpus, shards, and reports all
 * key on. Centralized so the ~15 `filename.replace('.sgf', '')` call sites
 * can't drift (some used the unanchored form). Null-tolerant: a falsy filename
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
 * Finds the winning move among a node's children by their actual evaluation —
 * NOT by SGF child order. Comments are "<winrate>%/<score>" with both numbers
 * reported from BLACK's perspective (bots/analysis.cfg sets
 * reportAnalysisWinratesAs = BLACK), so the best move for the side to play is:
 *   - Black to play -> the HIGHEST winrate (score breaks exact ties)
 *   - White to play -> the LOWEST  winrate (score breaks exact ties)
 *
 * Almost every generated problem happens to list its solution first, but a few
 * generator misfires don't (e.g. 13a38a3f: the listed-first move loses at 0.4%
 * while the real answer wins at 99.2%). Selecting by evaluation makes the
 * correctness check robust to ordering instead of trusting children[0].
 *
 * Falls back to the first child only when NO sibling carries a parseable
 * winrate/score comment (preserving the legacy behavior for such nodes).
 * @param {Array} children - Array of SGF nodes (all the same side to move).
 * @returns {object|null} - The winning move node, or null if there are none.
 */
export function findBestMoveFromChildren(children) {
    if (!children || children.length === 0) {
        return null;
    }

    // All children of a decision node share the side to move; read it off the
    // first child that carries a colored move.
    const sideToPlay = children.find(c => c.move && (c.move.c === WGo.B || c.move.c === WGo.W))?.move.c;
    const blackToPlay = sideToPlay === WGo.B;

    let best = null;
    let bestWinrate = blackToPlay ? -Infinity : Infinity;
    let bestScore = blackToPlay ? -Infinity : Infinity;
    for (const child of children) {
        const winrate = getWinrateFromComment(child.comment);
        if (winrate === null) continue;
        const score = getScoreFromComment(child.comment);
        // Higher winrate/score is better for Black, lower for White (comments
        // are Black-perspective). Score only matters on an exact winrate tie.
        const tieScore = score ?? (blackToPlay ? -Infinity : Infinity);
        const better = blackToPlay
            ? (winrate > bestWinrate || (winrate === bestWinrate && tieScore > bestScore))
            : (winrate < bestWinrate || (winrate === bestWinrate && tieScore < bestScore));
        if (better) {
            best = child;
            bestWinrate = winrate;
            bestScore = tieScore;
        }
    }

    if (best) {
        return best;
    }
    // No sibling had a parseable evaluation: preserve the legacy first-child behavior.
    return (children[0].comment && children[0].comment.match(/%\//)) ? children[0] : null;
}

// Standard Go board notation: column letters skip 'I' (A..H, J..T), row is
// 1-based from the BOTTOM while WGo's y=0 is the TOP row (so row = size - y).
// toBoardNotation and boardNotationToXy are exact inverses over this alphabet;
// they live together here so the COL_LETTERS table and the row flip can't drift.
const COL_LETTERS = 'ABCDEFGHJKLMNOPQRST';

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
 * @param {object} moveNode - The node of the move to check.
 * @param {Array} allChildren - All sibling nodes.
 * @returns {boolean} - True if the move is correct.
 */
export function isCorrectMove(moveNode, allChildren) {
    const bestMoveNode = findBestMoveFromChildren(allChildren);
    return bestMoveNode === moveNode;
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
