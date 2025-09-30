// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateTsumegoStats, updateAnswerHistory, addToAccumulatedSolvingTime, setStatsResults } from './state.js';
import { updateAllUI } from './ui.js';
import { TOTAL_RANKS, PROBLEM_STATUS } from './constants.js';
import { getTerm } from './i18n.js';

// Function to send report to Cloudflare Worker
export async function sendReport(success) {
    try {
        const problemFilename = state.problemHistory[0].filename;
        const problemId = problemFilename.replace(/\.[^/.]+$/, "");
        
        const reportData = {
            userId: state.userId,
            problem: problemId,
            success: success
        };
        
        await fetch('/report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(reportData)
        });
    } catch (error) {
        console.error('Failed to send report:', error);
    }
}

const getRankInfo = (rating) => {
    const rankValue = rating * TOTAL_RANKS;
    const rankIndex = Math.floor(rankValue);
    const progress = (rankValue - rankIndex) * 100;
    return { rankIndex, progress };
}

export function updateUserStats(isCorrect) {
    const history = state.problemHistory;
    const currentProblem = history[0];

    if (!currentProblem) return {};

    const oldStatus = currentProblem.status;

    if (isCorrect) {
        if (oldStatus === PROBLEM_STATUS.FAILURE) {
            currentProblem.status = PROBLEM_STATUS.IN_BETWEEN;
        } else if (oldStatus === PROBLEM_STATUS.IN_BETWEEN) {
            currentProblem.status = PROBLEM_STATUS.SUCCESS;
        }
    } else {
        currentProblem.status = PROBLEM_STATUS.FAILURE;
    }

    if (currentProblem.linked) {
        updateProblemHistory(history);
        return {};
    }

    const currentSessionTime = state.currentSolvingStartTime ? Date.now() - state.currentSolvingStartTime : 0;
    addToAccumulatedSolvingTime(currentSessionTime);
    const timeTaken = state.accumulatedSolvingTime;
    const answer = {
        problemId: currentProblem.filename.replace('.sgf', ''),
        problemRating: currentProblem.rating,
        problemQuality: currentProblem.quality || 0,
        timeTaken,
        reviewTime: 0,
        correct: isCorrect,
        date: Date.now(),
        problemType: !currentProblem.isRepeat
            ? 'novel'
            : oldStatus === PROBLEM_STATUS.FAILURE
                ? 'short-term-repeat'
                : oldStatus === PROBLEM_STATUS.IN_BETWEEN
                    ? 'long-term-repeat'
                    : 'repeat',
    };

    const answerHistory = [...state.answerHistory, answer];
    updateAnswerHistory(answerHistory);

    // Reset solving timer after recording
    state.currentSolvingStartTime = null;

    const oldRank = state.userRank;
    const oldRankInfo = getRankInfo(oldRank);
    let rank = oldRank;
    let momentum = state.momentum;
    const stats = { ...state.tsumegoStats };

    if (oldStatus === PROBLEM_STATUS.FAILURE) {
        stats.totalOnFail++;
        if (!isCorrect) {
            stats.failOnFail++;
        }
    } else if (oldStatus === PROBLEM_STATUS.IN_BETWEEN) {
        stats.totalOnInBetween++;
        if (!isCorrect) {
            stats.failOnInBetween++;
        }
    } else { // success or new
        stats.totalOnSuccess++;
        if (!isCorrect) {
            stats.failOnSuccess++;
        }
    }

    if (isCorrect ) {
        const maxIncrease = (1.0 - rank) / 9;
        let intendedIncrease = momentum / 600;
        if( oldStatus !== PROBLEM_STATUS.FAILURE && oldStatus !== PROBLEM_STATUS.IN_BETWEEN ) {
            if(intendedIncrease > maxIncrease) {
                momentum *= maxIncrease / intendedIncrease;
            } else {
                momentum *= 1.1;
            }
        } else {
            intendedIncrease *= 0.1
        }
        intendedIncrease *= (currentProblem.quality || 0.5);
        rank += Math.min(intendedIncrease, maxIncrease);
    } else if (!isCorrect) { // Incorrect
        momentum = (momentum + 1) / 3;
        let rankChange = 4 * (momentum / 600);
        rankChange *= (currentProblem.quality || 0.5);
        rank = Math.max(0, rank - rankChange);
    }

    updateUserRank(rank);
    updateMomentum(momentum);
    updateProblemHistory(history);
    updateTsumegoStats(stats);
    updateAllUI();

    const newRank = state.userRank;
    const newRankInfo = getRankInfo(newRank);

    const rankChange = newRank - oldRank;

    let rankChangeDirection = 'same';
    if (newRankInfo.rankIndex > oldRankInfo.rankIndex) {
        rankChangeDirection = 'up';
    } else if (newRankInfo.rankIndex < oldRankInfo.rankIndex) {
        rankChangeDirection = 'down';
    }

    // Trigger background calculation of detailed stats
    if (!isCalculatingStats) {
        isCalculatingStats = true;
        setTimeout(() => {
            calculateAndStoreStatsResults().finally(() => {
                isCalculatingStats = false;
            });
        }, 0);
    }

    return {
        rankChange,
        rankChangeDirection,
        isCorrect
    };
}

let isCalculatingStats = false;

async function calculateAndStoreStatsResults() {
    try {
        const validHistory = state.answerHistory.filter(p =>
            p.problemRating !== undefined &&
            p.problemQuality !== undefined &&
            p.timeTaken !== undefined &&
            p.reviewTime !== undefined &&
            p.date !== undefined
        );

        if (validHistory.length < 5) {
            setStatsResults([]); // Set empty results if not enough data
            return;
        }
        const history = validHistory.slice(-500);

        // ... [All existing percentile and OLS calculations] ...
        const timeTakenValues = history.map(p => p.timeTaken);
        const sortedTime = timeTakenValues.slice().sort((a, b) => a - b);
        const p_low = 0.45;
        const p_high = 0.55;
        const idx_low = Math.floor(p_low * (sortedTime.length - 1));
        const idx_high = Math.floor(p_high * (sortedTime.length - 1));
        const time_low = sortedTime[idx_low];
        const time_high = sortedTime[idx_high];
        const delta_time_ms = time_high - time_low;
        const time_per_percentile_min = delta_time_ms / (60000 * 0.1);
        const percentileTime = timeTakenValues.map(t => {
            const below = sortedTime.filter(s => s < t).length;
            const equal = sortedTime.filter(s => s === t).length;
            return (below + 0.5 * equal) / sortedTime.length;
        });

        const previousReviewTimes = history.map((p, i) => {
            if (p.problemType === 'short-term-repeat' || p.problemType === 'long-term-repeat') {
                const prev = history.slice(0, i).reverse().find(ans => ans.problemId === p.problemId && ans.reviewTime > 0);
                return prev ? prev.reviewTime : 0;
            }
            return 0;
        });

        const shortTermReviewTimes = history.map((p, i) => p.problemType === 'short-term-repeat' ? previousReviewTimes[i] : 0).filter(t => t > 0);
        const longTermReviewTimes = history.map((p, i) => p.problemType === 'long-term-repeat' ? previousReviewTimes[i] : 0).filter(t => t > 0);
        let time_per_percentile_min_r_short = 0;
        let time_per_percentile_min_r_long = 0;
        if (shortTermReviewTimes.length > 0) {
            const sortedReviewShort = shortTermReviewTimes.slice().sort((a, b) => a - b);
            const p_low_r = 0.45;
            const p_high_r = 0.55;
            const idx_low_r = Math.floor(p_low_r * (sortedReviewShort.length - 1));
            const idx_high_r = Math.floor(p_high_r * (sortedReviewShort.length - 1));
            const time_low_r = sortedReviewShort[idx_low_r];
            const time_high_r = sortedReviewShort[idx_high_r];
            const delta_time_ms_r = time_high_r - time_low_r;
            time_per_percentile_min_r_short = delta_time_ms_r / (60000 * 0.1);
        }
        if (longTermReviewTimes.length > 0) {
            const sortedReviewLong = longTermReviewTimes.slice().sort((a, b) => a - b);
            const p_low_r = 0.45;
            const p_high_r = 0.55;
            const idx_low_r = Math.floor(p_low_r * (sortedReviewLong.length - 1));
            const idx_high_r = Math.floor(p_high_r * (sortedReviewLong.length - 1));
            const time_low_r = sortedReviewLong[idx_low_r];
            const time_high_r = sortedReviewLong[idx_high_r];
            const delta_time_ms_r = time_high_r - time_low_r;
            time_per_percentile_min_r_long = delta_time_ms_r / (60000 * 0.1);
        }
        
        const percentileReviewShort = previousReviewTimes.map((t, i) => {
            if (t === 0 || history[i].problemType !== 'short-term-repeat' || shortTermReviewTimes.length === 0) return 0;
            const below = shortTermReviewTimes.filter(s => s < t).length;
            const equal = shortTermReviewTimes.filter(s => s === t).length;
            return (below + 0.5 * equal) / shortTermReviewTimes.length;
        });
        const percentileReviewLong = previousReviewTimes.map((t, i) => {
            if (t === 0 || history[i].problemType !== 'long-term-repeat' || longTermReviewTimes.length === 0) return 0;
            const below = longTermReviewTimes.filter(s => s < t).length;
            const equal = longTermReviewTimes.filter(s => s === t).length;
            return (below + 0.5 * equal) / longTermReviewTimes.length;
        });

        const y = history.map(p => p.correct ? 1 : 0);
        const X_original = history.map((p, i) => [
            1, percentileTime[i], percentileReviewShort[i], percentileReviewLong[i],
            p.problemRating, p.problemQuality, i,
            p.problemType === 'short-term-repeat' ? 1 : 0,
            p.problemType === 'long-term-repeat' ? 1 : 0
        ]);

        const stdDevs = [];
        for (let j = 1; j < X_original[0].length; j++) {
            const col = X_original.map(row => row[j]);
            stdDevs.push(math.std(col));
        }

        const includedIndices = [0];
        for (let j = 1; j < stdDevs.length + 1; j++) {
            if (stdDevs[j - 1] > 0) includedIndices.push(j);
        }

        const X = X_original.map(row => includedIndices.map(idx => row[idx]));
        const n = X.length;
        const k = X[0].length;

        const Xt = math.transpose(X);
        const XtX = math.multiply(Xt, X);
        const XtX_inv = math.inv(XtX);
        const Xty = math.multiply(Xt, y);
        const beta = math.multiply(XtX_inv, Xty);

        const y_hat = math.multiply(X, beta);
        const residuals = math.subtract(y, y_hat);
        const ssr = math.multiply(math.transpose(residuals), residuals);
        const s2 = ssr / (n - k);
        const varCovarMatrix = math.multiply(s2, XtX_inv);
        const stdErrors = math.map(math.diag(varCovarMatrix), math.sqrt);

        const results = [];
        for (let i = 1; i < includedIndices.length; i++) {
            const varIndex = includedIndices[i] - 1;
            let coeff = beta[i];
            let se = stdErrors[i];

            let scaleFactor = 1;
            if (varIndex === 0 && time_per_percentile_min > 0) scaleFactor = 1 / time_per_percentile_min;
            else if (varIndex === 1 && time_per_percentile_min_r_short > 0) scaleFactor = 1 / time_per_percentile_min_r_short;
            else if (varIndex === 2 && time_per_percentile_min_r_long > 0) scaleFactor = 1 / time_per_percentile_min_r_long;
            else if (varIndex === 3) scaleFactor = 1 / 29;
            else if (varIndex === 4) scaleFactor = -1;
            else if (varIndex === 5) scaleFactor = 100;

            coeff *= scaleFactor;
            se *= Math.abs(scaleFactor);
            
            results.push({ coeff, se, varIndex });
        }
        setStatsResults(results);

    } catch (error) {
        console.error('OLS Regression failed during background calculation:', error);
        setStatsResults([]); // Set empty results on failure
    }
}

// Function to show detailed stats when user clicks on rank
export function showDetailedStats() {
    if (state.statsResults === null) {
        alert(getTerm('statsCalculating'));
        return;
    }
    
    if (state.answerHistory.length < 5) {
        alert(getTerm('statsNotEnough').replace('{count}', state.answerHistory.length));
        return;
    }

    try {
        const results = state.statsResults;
        const observations = [];
        
        for (const result of results) {
            const { coeff, se, varIndex } = result;
            if (se > 0) {
                const z = Math.abs(coeff) / se;
                const cdf = 0.5 * (1 + math.erf(z / Math.sqrt(2)));
                const conf = (2 * cdf - 1) * 100;
                if (conf >= 50 && !([0, 1, 2].includes(varIndex) && coeff <= 0)) {
                    observations.push({ mean: (coeff * 100).toFixed(1), conf: conf.toFixed(0), varIndex });
                }
            }
        }
        observations.sort((a, b) => b.conf - a.conf);

        for (let obs of observations) {
            let termKey;
            let meanVal = obs.mean;
            let meanNum = parseFloat(obs.mean);
            if (obs.varIndex === 0) termKey = 'statsSolvingTime';
            else if (obs.varIndex === 1) termKey = 'statsReviewTimeShort';
            else if (obs.varIndex === 2) termKey = 'statsReviewTimeLong';
            else if (obs.varIndex === 4) {
                termKey = meanNum > 0 ? 'statsQualityAnalytical' : 'statsQualityIntuitive';
                meanVal = Math.abs(meanNum).toFixed(1);
            } else if (obs.varIndex === 5) {
                termKey = meanNum > 0 ? 'statsProgress' : 'statsRegression';
                meanVal = Math.abs(meanNum).toFixed(1);
            }
            if (termKey) {
                obs.message = getTerm(termKey).replace('{mean}', meanVal).replace('{conf}', obs.conf);
            }
        }

        const totalProblems = state.answerHistory.length;
        const analysisProblems = Math.min(totalProblems, 500);
        let resultText = "";

        const validObservations = observations.filter(o => o.message);
        if (validObservations.length > 0) {
            const intro = getTerm('statsIntroNew').replace('{total}', totalProblems).replace('{analysis}', analysisProblems);
            resultText = intro + "- " + validObservations.map(o => o.message).join("\n- ");
        } else {
            const intro = getTerm('statsNoTrends').replace('{total}', totalProblems).replace('{analysis}', analysisProblems);
            resultText = intro;
        }

        alert(resultText);
    } catch (error) {
        console.error('Failed to generate stats text:', error);
        alert('Could not display statistical analysis.');
    }
}