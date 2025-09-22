// js/utils.js

import { TOTAL_RANKS } from './constants.js';

const localization = {
  zh: {
    kyu: '级',
    dan: '段',
    review: '上一个',                // Previous
    about: '关于',
    analyzing: '分析中…',
    error: '错误',
    exit: '下一个',                  // Next
    score: '得分',
    undo: '撤销',
    share: '分享',
    stats: '统计',
    linkCopied: '链接已复制到剪贴板',
    tsumegoMode: '挑战模式',
    reviewMode: '复习模式',
    viewMode: '查看模式',
    carelessMistake: '粗心失误，请花更多时间阅读与判断。',
    shouldKnowThis: '应当已经掌握，请多花时间复盘你的错误。',
    rankUpTo: '升级到{rank}',
    rankDownTo: '降级到{rank}',
    installPrompt: '安装为应用以获得更好的体验！',
    install: '安装',
    rankChangeFormat: '等级 {change}',
    unreliableWarning: '警告：此位置没有预计算解决方案，最佳移动和得分不可靠。',
    komi: '贴目',
    blackCaptures: '黑方提子',
    whiteCaptures: '白方提子',
    whiteAdjustment: '白方调整',
    shortTermRepeat: '短期重复',
    longTermRepeat: '长期重复',
    statsIntro: '过去 {count} 个问题的结果影响因素：',
    timeSpent: '解题时间',
    reviewTime: '复习时间',
    problemDifficulty: '问题难度',
    problemQuality: '问题质量',
    date: '日期',
    statsNotEnough: '您已完成 {count} 个数独，启用统计需要5个。',
    statsPerMinuteSolving: ' % 每分钟解题',
    statsPerMinuteReviewing: ' % 每分钟复习',
    statsPerProblemRank: ' % 每问题等级',
    statsIfIntuitiveProblem: ' % 如果是直观问题',
    statsPerProblemSolved: ' % 每解决的问题',
    statsIfRecentlyFailed: ' % 如果最近失败',
    statsIfFailedLongTimeAgo: ' % 如果很久前失败',
    statsDefault: ' %'
  },
  ja: {
    kyu: '級',
    dan: '段',
    review: '前へ',
    about: 'このアプリについて',
    analyzing: '分析中…',
    error: 'エラー',
    exit: '次へ',
    score: 'スコア',
    undo: '元に戻す',
    share: '共有',
    stats: '統計',
    linkCopied: 'リンクをクリップボードにコピーしました',
    tsumegoMode: 'チャレンジモード',
    reviewMode: '復習モード',
    viewMode: '閲覧モード',
    carelessMistake: '不注意によるミスです。読みと検討にもっと時間をかけましょう。',
    shouldKnowThis: '既知のはずです。ミスを振り返る時間をもう少し取りましょう。',
    rankUpTo: '{rank}に昇段',
    rankDownTo: '{rank}に降段',
    installPrompt: 'アプリとしてインストールして、より良い体験を！',
    install: 'インストール',
    rankChangeFormat: 'ランク {change}',
    unreliableWarning: '警告：この位置には事前計算されたソリューションがありません、最良の移動とスコアは信頼できません。',
    komi: 'コミ',
    blackCaptures: '黒の取り',
    whiteCaptures: '白の取り',
    whiteAdjustment: '白の調整',
    shortTermRepeat: '短期繰り返し',
    longTermRepeat: '長期繰り返し',
    statsIntro: '過去 {count} 件の問題の結果に影響する要因：',
    timeSpent: '解く時間',
    reviewTime: 'レビュー時間',
    problemDifficulty: '問題の難易度',
    problemQuality: '問題の品質',
    date: '日付',
    statsNotEnough: '{count} つのツメゴを完了しました、統計を有効にするには5つが必要です。',
    statsPerMinuteSolving: ' % 毎分解く',
    statsPerMinuteReviewing: ' % 毎分レビュー',
    statsPerProblemRank: ' % 毎問題ランク',
    statsIfIntuitiveProblem: ' % 直感的な問題の場合',
    statsPerProblemSolved: ' % 毎解決問題',
    statsIfRecentlyFailed: ' % 最近失敗した場合',
    statsIfFailedLongTimeAgo: ' % ずっと前に失敗した場合',
    statsDefault: ' %'
  },
  ko: {
    kyu: '급',
    dan: '단',
    review: '이전',
    about: '정보',
    analyzing: '분석 중…',
    error: '오류',
    exit: '다음',
    score: '점수',
    undo: '실행 취소',
    share: '공유',
    stats: '통계',
    linkCopied: '링크가 클립보드에 복사되었습니다',
    tsumegoMode: '챌린지 모드',
    reviewMode: '복습 모드',
    viewMode: '보기 모드',
    carelessMistake: '부주의한 실수입니다. 더 시간을 들여 읽고 판단하세요.',
    shouldKnowThis: '이미 알고 있어야 합니다. 실수를 더 꼼꼼히 복기하세요.',
    rankUpTo: '{rank}으로 승급',
    rankDownTo: '{rank}으로 강등',
    installPrompt: '앱으로 설치하여 더 나은 경험을!',
    install: '설치',
    rankChangeFormat: '등급 {change}',
    unreliableWarning: '경고: 이 위치에 대해 사전 계산된 솔루션이 없으며, 최상의 이동과 점수는 신뢰할 수 없습니다.',
    komi: '코미',blackCaptures: '흑의 포획',
    whiteCaptures: '백의 포획',
    whiteAdjustment: '백의 조정',
    shortTermRepeat: '단기 반복',
    longTermRepeat: '장기 반복',
    statsIntro: '지난 {count}개 문제의 결과에 영향을 미치는 요인:',
    timeSpent: '풀이 시간',
    reviewTime: '리뷰 시간',
    problemDifficulty: '문제 난이도',
    problemQuality: '문제 품질',
    date: '날짜',
    statsNotEnough: '{count}개의 츠메고를 완료했습니다, 통계를 활성화하려면 5개가 필요합니다.',
    statsPerMinuteSolving: ' % 분당 풀이',
    statsPerMinuteReviewing: ' % 분당 리뷰',
    statsPerProblemRank: ' % 문제 등급당',
    statsIfIntuitiveProblem: ' % 직관적인 문제인 경우',
    statsPerProblemSolved: ' % 해결된 문제당',
    statsIfRecentlyFailed: ' % 최근에 실패한 경우',
    statsIfFailedLongTimeAgo: ' % 오래 전에 실패한 경우',
    statsDefault: ' %'
  },
  en: {
    kyu: 'kyu',
    dan: 'dan',
    review: 'Previous',
    about: 'About',
    analyzing: 'Analyzing…',
    error: 'Error',
    exit: 'Next',
    score: 'Score',
    undo: 'Undo',
    share: 'Share',
    stats: 'Stats',
    linkCopied: 'Link copied to clipboard',
    tsumegoMode: 'Challenge mode',
    reviewMode: 'Review mode',
    viewMode: 'View mode',
    carelessMistake: 'Slow down! You could likely have avoided this mistake with a bit more effort.',
    shouldKnowThis: 'A repeat mistake! During reviews try to understand difference between your move and the correct solution.',
    rankUpTo: 'Rank up to {rank}',
    rankDownTo: 'Rank down to {rank}',
    installPrompt: 'Install as an app for a better experience!',
    install: 'Install',
    rankChangeFormat: 'Rank {change}',
    unreliableWarning: 'WARNING: no precomputed solution available for this position, best move and score are unreliable.',
    komi: 'Komi',
    blackCaptures: 'Black Captures',
    whiteCaptures: 'White Captures',
    whiteAdjustment: 'White Adjustment',
    shortTermRepeat: 'Short-term Repeat',
    longTermRepeat: 'Long-term Repeat',
    statsIntro: 'Factors affecting your result of the past {count} challenges:',
    timeSpent: 'Time Solving',
    reviewTime: 'Time Reviewing',
    problemDifficulty: 'Problem Difficulty',
    problemQuality: 'Problem Quality',
    date: 'Date',
    statsNotEnough: 'You have completed {count} tsumegos, 5 needed to enable statistics.',
    statsPerMinuteSolving: ' % per minute solving',
    statsPerMinuteReviewing: ' % per minute reviewing',
    statsPerProblemRank: ' % per problem rank',
    statsIfIntuitiveProblem: ' % if intuitive problem',
    statsPerProblemSolved: ' % per problem solved',
    statsIfRecentlyFailed: ' % if recently failed',
    statsIfFailedLongTimeAgo: ' % if failed long time ago',
    statsDefault: ' %'
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
 * Cleans the URL by removing the 'id' parameter and 'index.html'.
 */
export function cleanUrl() {
    const url = new URL(window.location);
    url.searchParams.delete('id');
    const newPath = url.pathname.replace(/index\.html$/, '');
    window.history.replaceState({}, '', newPath + url.search + url.hash);
}
