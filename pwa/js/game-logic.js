// js/game-logic.js

import { state, setAcceptClicks, setAppMode } from './state.js';
import { getCurrentNode, getLegalMoveNode, playMoveByIndex, playNextMove } from './board.js';
import { displayOverlay, fadeOverlay } from './overlays.js';
import { playStoneSound } from './audio.js';
import { isCorrectMove } from './sgf.js';
import { APP_MODES, PROBLEM_STATUS, OLS_INDICES } from './constants.js';
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
    const statsResults = state.statsResults;
    const statsAreAvailable = statsResults && statsResults.length > 0;



    if (oldStatus === PROBLEM_STATUS.FAILURE || oldStatus === PROBLEM_STATUS.IN_BETWEEN) {
        let reviewTimeIsBeneficial = false;
        let meanReviewTime = 0;
        let previousReviewTime = 0;
        let reviewStat;
        if (statsAreAvailable) {
            const reviewTimeValues = state.answerHistory.map(p => p.reviewTime).filter(t => t !== undefined && t > 0);
            meanReviewTime = reviewTimeValues.length > 0 ? reviewTimeValues.reduce((a, b) => a + b, 0) / reviewTimeValues.length : 0;

            // Get accumulated review time for this problem
            const currentProblem = state.problemHistory[0];
            const problemId = currentProblem.filename.replace('.sgf', '');
            const previousAnswers = state.answerHistory.filter(ans => ans.problemId === problemId);
            previousReviewTime = previousAnswers.reduce((sum, ans) => sum + ans.reviewTime, 0);

            const targetVarIndex = (oldStatus === PROBLEM_STATUS.FAILURE) ? OLS_INDICES.REVIEW_TIME_SHORT : OLS_INDICES.REVIEW_TIME_LONG;
            reviewStat = statsResults.find(r => r.varIndex === targetVarIndex);
            if (reviewStat && reviewStat.coeff > 0) {
                reviewTimeIsBeneficial = true;
            }
        }

        if (reviewTimeIsBeneficial && previousReviewTime < meanReviewTime) {
            duration = 6000;
            if (reviewStat) {
                const { coeff, se } = reviewStat;
                const z = Math.abs(coeff) / se;
                const cdf = 0.5 * (1 + math.erf(z / Math.sqrt(2)));
                const conf = (2 * cdf - 1) * 100;
                const x = Math.round(previousReviewTime / 1000);
                message = getTerm('insufficientReviewMessage').replace('{seconds}', x).replace('{confidence}', conf.toFixed(0));
            }
        }
    } else { // Careless mistake for new problems
        let timeIsBeneficial = false;
        let meanSolvingTime = 0;
        if (statsAreAvailable) {
            const timeTakenValues = state.answerHistory.map(p => p.timeTaken).filter(t => t !== undefined);
            meanSolvingTime = timeTakenValues.length > 0 ? timeTakenValues.reduce((a, b) => a + b, 0) / timeTakenValues.length : 0;
            const timeStat = statsResults.find(r => r.varIndex === OLS_INDICES.TIME_SOLVING);
            timeIsBeneficial = timeStat ? timeStat.coeff > 0 : false;
        }

        if (timeIsBeneficial && state.accumulatedSolvingTime < meanSolvingTime) {
            duration = 6000;
            const timeStat = statsResults.find(r => r.varIndex === OLS_INDICES.TIME_SOLVING);
            const { coeff, se } = timeStat;
            const z = Math.abs(coeff) / se;
            const cdf = 0.5 * (1 + math.erf(z / Math.sqrt(2)));
            const conf = (2 * cdf - 1) * 100;
            const x = Math.round(timeSpent / 1000);
            message = getTerm('rushingMessage').replace('{seconds}', x).replace('{confidence}', conf.toFixed(0));
        }
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