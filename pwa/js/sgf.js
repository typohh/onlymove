// js/sgf.js
import { SGF_COMMENTS } from './constants.js';

/**
 * Parses the winrate from an SGF comment.
 * @param {string} comment - The comment string.
 * @returns {number|null} - The winrate as a float, or null if not found.
 */
export function getWinrateFromComment(comment) {
  if (!comment) {
    return null;
  }
  // Extracts winrate like "86.4%" from "86.4%/3.1"
  const match = comment.match(/([\d.]+)%/);
  if (match && match[1]) {
    return parseFloat(match[1]);
  }
  return null;
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
 * Finds the best move from a list of child nodes based on winrate.
 * @param {Array} children - Array of SGF nodes.
 * @returns {object|null} - The best move node, or null if not found.
 */
export function findBestMoveFromChildren(children) {
    if (!children || children.length === 0) {
        return null;
    }

    let bestWinrate = -1;
    let bestMoveNode = null;

    for (const child of children) {
        if (!child.move) continue;
        const winrate = getWinrateFromComment(child.comment);
        if (winrate !== null && winrate > bestWinrate) {
            bestWinrate = winrate;
            bestMoveNode = child;
        }
    }
    return bestMoveNode;
}

/**
 * Checks if a move is correct.
 * @param {object} moveNode - The node of the move to check.
 * @param {Array} allChildren - All sibling nodes.
 * @returns {boolean} - True if the move is correct.
 */
export function isCorrectMove(moveNode, allChildren) {
    const hasWinrateComments = allChildren.some(child => child.comment && getWinrateFromComment(child.comment) !== null);

    if (hasWinrateComments) {
        const bestMoveNode = findBestMoveFromChildren(allChildren);
        return bestMoveNode === moveNode;
    } else {
        // Fallback to old logic
        return moveNode.comment?.includes(SGF_COMMENTS.CORRECT) || false;
    }
}
