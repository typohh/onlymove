// js/game-logic.js

import { state, setAcceptClicks, setAppMode } from './state.js';
import { getCurrentNode, getLegalMoveNode, playMoveByIndex, playNextMove } from './board.js';
import { displayOverlay, fadeOverlay } from './overlays.js';
import { playStoneSound } from './audio.js';
import { isCorrectMove } from './sgf.js';
import { APP_MODES, PROBLEM_STATUS } from './constants.js';
import { getTerm } from './i18n.js';
import { sendReport, updateUserStats } from './user-stats.js';
import { loadTsumego } from './problem-loader.js';
import { enterAnalysisMode } from './analysis.js';

const handleCorrectAnswer = () => {
    setAppMode(APP_MODES.SOLVING);
    sendReport(true);
    const stats = updateUserStats(true);

    fadeOverlay(1000, () => {
        loadTsumego(true).catch(error => console.error("Error loading next tsumego:", error));
    }, stats);
}

async function _handleIncorrectMoveSequence(moveIndex) {
    Sentry.addBreadcrumb({
        category: 'flow',
        message: '_handleIncorrectMoveSequence started',
        level: 'info',
        data: { moveIndex }
    });
    const delay = (ms) => new Promise(res => setTimeout(res, ms));

    playMoveByIndex(moveIndex);
    playStoneSound();
    await delay(1000);

    playNextMove();
    playStoneSound();

    const oldStatus = state.problemHistory[0]?.status;

    sendReport(false);
    const stats = updateUserStats(false);

    // Calculate overlay duration and message based on time spent
    const timeSpent = state.accumulatedSolvingTime + (state.currentSolvingStartTime ? Date.now() - state.currentSolvingStartTime : 0);
    let duration = 2000; // 2s default
    let message = '';
    if (oldStatus === PROBLEM_STATUS.FAILURE || oldStatus === PROBLEM_STATUS.IN_BETWEEN) {
        duration = 6000;
        message = getTerm('shouldKnowThis');
    } else if (timeSpent < 15000) { // less than 15s
        duration = 6000;
        message = getTerm('carelessMistake');
    }

    await new Promise((resolve) => displayOverlay(message, duration, resolve, undefined, stats));

    await loadTsumego(true);
    await enterAnalysisMode();
}

export function handleBoardClick(x, y) {
    if (!state.acceptClicks) return;

    const moveNode = getLegalMoveNode(x, y);
    if (!moveNode) {
        return;
    }

    setAcceptClicks(false);

    const children = getCurrentNode().children;
    const moveIndex = children.indexOf(moveNode);
    if (isCorrectMove(moveNode, children)) {
        playMoveByIndex(moveIndex);
        playStoneSound();
        handleCorrectAnswer();
    } else {
        _handleIncorrectMoveSequence(moveIndex);
    }
}