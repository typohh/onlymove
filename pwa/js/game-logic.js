// js/game-logic.js

import { state, setAcceptClicks, setCurrentPointLoss } from './state.js';
import { getCurrentNode, redraw, playMoveByIndex, playNextMove } from './board.js';
import { displayOverlay, fadeOverlay } from './ui.js';
import { playStoneSound } from './audio.js';
import { isCorrectMove, findBestMoveFromChildren, getScoreFromComment } from './sgf.js';
import { SGF_COMMENTS } from './constants.js';
import { getTerm } from './utils.js';
import { sendReport, updateUserStats } from './user-stats.js';
import { loadTsumego } from './problem-loader.js';
import { enterAnalysisMode } from './analysis.js';

function handleCorrectAnswer() {
    sendReport(true);
    updateUserStats(true);

    fadeOverlay(1000, () => {
        loadTsumego(true).catch(error => console.error("Error loading next tsumego:", error));
    });
}

function _calculatePointLossText(playedNode, currentKifuNode) {
    const children = currentKifuNode.children;
    if (!children || children.length === 0) return '';

    const correctNode = findBestMoveFromChildren(children);

    if (!correctNode) return '';

    const correctScore = getScoreFromComment(correctNode.comment);
    const playedScore = getScoreFromComment(playedNode.comment);

    if (correctScore !== null && playedScore !== null) {
        const pointLoss = Math.abs(correctScore - playedScore);
        return `-${pointLoss.toFixed(1)} ${getTerm('points')}`;
    }

    const correctScoreMatch = correctNode.comment?.match(/Score:\s*([+-]?\d+\.?\d*)/);
    const playedScoreMatch = playedNode.comment?.match(/Score:\s*([+-]?\d+\.?\d*)/);

    if (correctScoreMatch && playedScoreMatch) {
        const pointLoss = Math.abs(parseFloat(correctScoreMatch[1]) - parseFloat(playedScoreMatch[1]));
        return `-${pointLoss.toFixed(1)} ${getTerm('points')}`;
    }

    return '';
}

async function _handleIncorrectMoveSequence(moveIndex) {
    const delay = (ms) => new Promise(res => setTimeout(res, ms));

    playMoveByIndex(moveIndex);
    playStoneSound();
    await delay(1000);

    playNextMove();
    playStoneSound();

    const oldStatus = state.problemHistory[0]?.status;

    sendReport(false);
    updateUserStats(false);

    // Calculate overlay duration and message based on time spent
    const timeSpent = Date.now() - (state.problemStartTime || Date.now());
    let duration = 10000; // 10s default
    let message = '';
    const decrease = Math.floor(timeSpent / 2000);
    duration = Math.max(2000, 10000 - decrease * 1000);
    if (timeSpent >= 20000 && (oldStatus === 'failure' || oldStatus === 'in-between')) {
        duration = 10000;
        message = 'You should know this already, spend more time reviewing your mistakes';
    } else if (timeSpent < 18000) { // less than 18s
        message = 'Careless mistake, spend more time reading and evaluating';
    }

    await new Promise((resolve) => displayOverlay(message, duration, resolve));

    await loadTsumego(true);
    enterAnalysisMode();
}

export function handleBoardClick(x, y) {
    if (!state.acceptClicks) return;

    const node = getCurrentNode();
    const children = node.children;

    if (!children || children.length === 0) {
        return;
    }

    const moveNode = children.find(child => child.move && child.move.x === x && child.move.y === y);
    if (!moveNode) {
        return;
    }

    setAcceptClicks(false);

    const moveIndex = children.indexOf(moveNode);
    if (isCorrectMove(moveNode, children)) {
        playMoveByIndex(moveIndex);
        playStoneSound();
        handleCorrectAnswer();
    } else {
        setCurrentPointLoss(_calculatePointLossText(moveNode, node));
        _handleIncorrectMoveSequence(moveIndex);
    }
}
