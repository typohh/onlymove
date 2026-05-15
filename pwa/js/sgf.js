// js/sgf.js
import { SGF_COMMENTS } from './constants.js';

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
 * Finds the best move from a list of child nodes (assumes first child is best if it has winrate/score comment).
 * @param {Array} children - Array of SGF nodes.
 * @returns {object|null} - The best move node, or null if not found.
 */
export function findBestMoveFromChildren(children) {
    if (!children || children.length === 0) {
        return null;
    }
    if (children[0].comment && children[0].comment.match(/%\//)) {
        return children[0];
    }
    return null;
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
