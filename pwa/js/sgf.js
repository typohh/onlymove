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
