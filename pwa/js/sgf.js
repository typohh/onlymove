// js/sgf.js\nimport { SGF_COMMENTS } from './constants.js';

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
 * @param {number} x - Original x coordinate (0-8).
 * @param {number} y - Original y coordinate (0-8).
 * @param {number} transformation - Type of transformation (0-7).
 * @returns {object} - Transformed {x, y}.
 */
function transformCoordinates(x, y, transformation) {
    const size = 9; // Assuming 9x9 board
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

/**
 * Recursively transforms all move coordinates in the kifu tree.
 * @param {object} node - Current node in the kifu tree.
 * @param {string} transformation - Type of transformation.
 */
function transformKifuTree(node, transformation) {
    if (node.move) {
        const transformed = transformCoordinates(node.move.x, node.move.y, transformation);
        node.move.x = transformed.x;
        node.move.y = transformed.y;
    }
    if (node.children) {
        node.children.forEach(child => transformKifuTree(child, transformation));
    }
}

/**
 * Reorients the SGF by applying a random transformation.
 * @param {string} sgf - Original SGF string.
 * @returns {string} - Reoriented SGF string.
 */
export function reorientSgf(sgf) {
    const randomTransformation = Math.floor(Math.random() * 8);

    const kifu = WGo.Kifu.fromSgf(sgf);
    transformKifuTree(kifu.root, randomTransformation);

    // Assuming WGo.Kifu has toSgf method
    return kifu.toSgf();
}
