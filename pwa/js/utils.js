// js/utils.js

import { TOTAL_RANKS } from './constants.js';

const localization = {
  zh: {
    kyu: '级',
    dan: '段',
    review: '上一个',                // Previous
    points: '目',                    // use '点数' if generic
    about: '关于',
    version: '版本',
    analyzing: '分析中…',
    error: '错误',
    errorLoadingProblem: '加载题目时出错。',
    appTitle: 'OnlyMove.net',
    exit: '下一个',                  // Next
    score: '得分',
    undo: '撤销',
    share: '分享',
    linkCopied: '链接已复制到剪贴板',
    tsumegoMode: '死活题模式',
    reviewMode: '复习模式',
    viewMode: '查看模式',
    carelessMistake: '粗心失误，请花更多时间阅读与判断。',
    shouldKnowThis: '应当已经掌握，请多花时间复盘你的错误。',
    rankUpTo: '升级到{rank}',
    rankDownTo: '降级到{rank}',
    onTheWayToNextRank: '距离下一级',
    standardDeviation: '标准差',
    totalTsumegos: '总题目数',
    shortTermRecollection: '短期记忆',
    longTermRecollection: '长期记忆',
    userRank: '用户等级'
  },
  ja: {
    kyu: '級',
    dan: '段',
    review: '前へ',
    points: '目',                    // use '点' or '点数' if generic
    about: 'このアプリについて',
    version: 'バージョン',
    analyzing: '分析中…',
    error: 'エラー',
    errorLoadingProblem: '問題の読み込み中にエラーが発生しました。',
    appTitle: 'OnlyMove.net',
    exit: '次へ',
    score: 'スコア',
    undo: '元に戻す',
    share: '共有',
    linkCopied: 'リンクをクリップボードにコピーしました',
    tsumegoMode: '詰碁モード',
    reviewMode: '復習モード',
    viewMode: '閲覧モード',
    carelessMistake: '不注意によるミスです。読みと検討にもっと時間をかけましょう。',
    shouldKnowThis: '既知のはずです。ミスを振り返る時間をもう少し取りましょう。',
    rankUpTo: '{rank}に昇段',
    rankDownTo: '{rank}に降段',
    onTheWayToNextRank: '次の段級まで',
    standardDeviation: '標準偏差',
    totalTsumegos: '合計問題数',
    shortTermRecollection: '短期記憶',
    longTermRecollection: '長期記憶',
    userRank: 'ユーザーランク'
  },
  ko: {
    kyu: '급',
    dan: '단',
    review: '이전',
    points: '집',                    // use '점' or '점수' if generic
    about: '정보',
    version: '버전',
    analyzing: '분석 중…',
    error: '오류',
    errorLoadingProblem: '문제를 불러오는 중 오류가 발생했습니다.',
    appTitle: 'OnlyMove.net',
    exit: '다음',
    score: '점수',
    undo: '실행 취소',
    share: '공유',
    linkCopied: '링크가 클립보드에 복사되었습니다',
    tsumegoMode: '사활 문제 모드',
    reviewMode: '복습 모드',
    viewMode: '보기 모드',
    carelessMistake: '부주의한 실수입니다. 더 시간을 들여 읽고 판단하세요.',
    shouldKnowThis: '이미 알고 있어야 합니다. 실수를 더 꼼꼼히 복기하세요.',
    rankUpTo: '{rank}으로 승급',
    rankDownTo: '{rank}으로 강등',
    onTheWayToNextRank: '다음 등급까지',
    standardDeviation: '표준 편차',
    totalTsumegos: '총 문제 수',
    shortTermRecollection: '단기 기억',
    longTermRecollection: '장기 기억',
    userRank: '사용자 등급'
  },
  en: {
    kyu: 'kyu',
    dan: 'dan',
    review: 'Previous',
    points: 'points',
    about: 'About',
    version: 'Version',
    analyzing: 'Analyzing…',
    error: 'Error',
    errorLoadingProblem: 'Error loading problem.',
    appTitle: 'OnlyMove.net',
    exit: 'Next',
    score: 'Score',
    undo: 'Undo',
    share: 'Share',
    linkCopied: 'Link copied to clipboard',
    tsumegoMode: 'Tsumego mode',
    reviewMode: 'Review mode',
    viewMode: 'View mode',
    carelessMistake: 'Slow down! You could likely have avoided this mistake with a bit more effort.',
    shouldKnowThis: 'A repeat mistake! During reviews try to understand difference between your move and the correct solution.',
    rankUpTo: 'Rank up to {rank}',
    rankDownTo: 'Rank down to {rank}',
    onTheWayToNextRank: 'On the way to next rank',
    standardDeviation: 'Standard Deviation',
    totalTsumegos: 'Total Tsumegos',
    shortTermRecollection: 'Short Term Recollection',
    longTermRecollection: 'Long Term Recollection',
    userRank: 'User Rank',
    rankChangeFormat: 'Rank {change}'
  }
};

export function getTerm(key) {
    const lang = navigator.language || 'en';
    if (lang.startsWith('zh')) return localization.zh[key];
    if (lang.startsWith('ja')) return localization.ja[key];
    if (lang.startsWith('ko')) return localization.ko[key];
    return localization.en[key];
}

export function getDescriptiveRank(numericRank) {
    const rankIndex = Math.floor(Math.pow(numericRank, 1.0) * TOTAL_RANKS);
    const kyuTerm = getTerm('kyu');
    const danTerm = getTerm('dan');
    return (rankIndex < 20) ? `${20 - rankIndex} ${kyuTerm}` : `${rankIndex - 19} ${danTerm}`;
}

/**
 * Updates the URL with the current tsumego ID
 */
export function updateUrlWithTsumegoId(tsumegoId) {
    const url = new URL(window.location);
    if (tsumegoId) {
        url.searchParams.set('id', tsumegoId);
    } else {
        url.searchParams.delete('id');
    }
    window.history.replaceState({}, '', url);
}
