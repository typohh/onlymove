// js/utils.js

import { TOTAL_RANKS } from './constants.js';

const localization = {
  zh: {
    kyu: '级',
    dan: '段',
    review: '复习',
    about: '关于',
    analyzing: '分析中…',
    error: '错误',
    exit: '下一个',
    score: '得分',
    undo: '撤销',
    share: '分享',
    stats: '统计',
    linkCopied: '链接已复制到剪贴板',
    tsumegoMode: '挑战模式',
    reviewMode: '复习模式',
    viewMode: '查看模式',
    rushingMessage: '您只用了{seconds}秒就回答了。根据您以往的结果，匆忙作答会导致更多错误（置信度：{confidence}%）。',
    insufficientReviewMessage: '重复错误！您之前只花了{seconds}秒复习这个问题。您的历史记录显示，花更多时间复习，您更有可能记住形状（置信度：{confidence}%）。',
    iWontRushIt: '我不会再仓促行事。',
    rankUpTo: '升至{rank}',
    rankDownTo: '降至{rank}',
    installPrompt: '安装为应用以获得更好的体验！',
    install: '安装',
    rankChangeFormat: '等级 {change}',
    unreliableWarning: '警告：此位置没有预计算解决方案，最佳移动和得分不可靠。',
    komi: '贴目',
    blackCaptures: '黑方提子',
    whiteCaptures: '白方提子',
    whiteAdjustment: '白方调整',
    statsNotEnough: '您已完成 {count} 个死活题，启用统计需要5个。',
    statsSolvingTime: '解题时间（+1分钟）：{rank_change} ± {stddev} 等级',
    statsReviewTime: '复习时间（+1分钟）：{rank_change} ± {stddev} 等级',
    statsProblemQuality: '直观问题：{rank_change} ± {stddev} 等级',
    statsForgettingCurveWithStdDev: '距离复习时间（+24小时）：{rank_change} ± {stddev} 等级',
    statsIntroNew: '近期观察：',
    statsNoTrends: '基于您最近的{analysis}个问题（共{total}个），我们没有发现明显的统计规律。请继续练习以获取更精确的分析！',
    statsFailed: '无法计算统计数据。',
    statsFailures: '您有 {count_recent} 个最近的失败，将在 {hours} 小时延迟后重复。',
    statsOldFailures: '另外还有 {count_other} 个过去的失败偶尔会重复。',
    statsNovelProblems: '您迄今为止尝试了 {count} 个独特的问题。',
    ignore: '忽略',
    ignoreConfirm: '您觉得从这个问题中无法学到任何东西，并且不想在未来再次看到它吗？'
  },
  ja: {
    kyu: '級',
    dan: '段',
    review: '復習',
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
    rushingMessage: 'あなたはわずか{seconds}秒で回答しました。過去の結果から、急ぐとミスが増えることが示唆されます（信頼度：{confidence}%）。',
    insufficientReviewMessage: '繰り返しのミス！あなたはこの問題を以前に{seconds}秒だけレビューしました。あなたの履歴は、より多くのレビュー時間を費やすことで形を覚える可能性が高くなることを示しています（信頼度：{confidence}%）。',
    iWontRushIt: 'もう急ぎません。',
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
    statsNotEnough: '{count} つのツメゴを完了しました、統計を有効にするには5つが必要です。',
    statsSolvingTime: '解答時間（+1分）：{rank_change} ± {stddev} ランク',
    statsReviewTime: '復習時間（+1分）：{rank_change} ± {stddev} ランク',
    statsProblemQuality: '直感的な問題：{rank_change} ± {stddev} ランク',
    statsForgettingCurveWithStdDev: '復習からの時間（+24時間）：{rank_change} ± {stddev} ランク',
    statsIntroNew: '最近の観察：',
    statsNoTrends: '最近の{analysis}問（合計{total}問）では、統計的に有意な傾向は見られませんでした。より多くのデータを集めるために、練習を続けてください！',
    statsFailed: '統計を計算できませんでした。',
    statsFailures: 'あなたには {count_recent} 件の最近の失敗があり、 {hours} 時間の遅延後に繰り返されます。',
    statsOldFailures: 'さらに {count_other} 件の過去の失敗が時折繰り返されます。',
    statsNovelProblems: 'これまでに {count} 個のユニークな問題を試しました。',
    ignore: '無視',
    ignoreConfirm: 'この問題から何も学べないと感じて、今後また見たくないですか？'
  },
    ko: {
      kyu: '급',
      dan: '단',
      review: '복습',
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
      rushingMessage: '당신은 단 {seconds}초 만에 답했습니다. 과거 결과에 따르면 서두르면 실수가 더 많아집니다 ({confidence}% 신뢰도).',
      insufficientReviewMessage: '반복 실수! 당신은 이전에 이 문제를 {seconds}초만 검토했습니다. 당신의 기록은 더 많은 검토 시간을 투자하면 모양을 기억할 가능성이 더 높아진다는 것을 보여줍니다 ({confidence}% 신뢰도).',
      iWontRushIt: '이제 서두르지 않겠습니다.',
      rankUpTo: '{rank}으로 승급',
      rankDownTo: '{rank}으로 강등',
      installPrompt: '앱으로 설치하여 더 나은 경험을!',
      install: '설치',
      rankChangeFormat: '등급 {change}',
      unreliableWarning: '경고: 이 위치에 대해 사전 계산된 솔루션이 없으며, 최상의 이동과 점수는 신뢰할 수 없습니다.',
      komi: '코미',
      blackCaptures: '흑의 포획',
      whiteCaptures: '백의 포획',
      whiteAdjustment: '백의 조정',
      statsNotEnough: '{count}개의 사활 문제를 완료했습니다, 통계를 활성화하려면 5개가 필요합니다.',
      statsSolvingTime: '풀이 시간 (+1분): {rank_change} ± {stddev} 등급',
      statsReviewTime: '복습 시간 (+1분): {rank_change} ± {stddev} 등급',
      statsProblemQuality: '직관적인 문제: {rank_change} ± {stddev} 등급',
      statsForgettingCurveWithStdDev: '복습 후 시간 (+24시간): {rank_change} ± {stddev} 등급',
      statsIntroNew: '최근 관찰:',
      statsNoTrends: '최근 {analysis}개 문제(총 {total}개)에서는 통계적으로 유의미한 경향을 발견하지 못했습니다. 더 많은 데이터를 모으기 위해 계속 풀어보세요!',
      statsFailed: '통계를 계산할 수 없습니다.',
      statsFailures: '{count_recent}개의 최근 실패가 있으며 {hours}시간 지연 후 반복됩니다.',
      statsOldFailures: '추가로 {count_other}개의 과거 실패가 때때로 반복됩니다.',
      statsNovelProblems: '지금까지 {count}개의 고유한 문제를 시도했습니다.',
      ignore: '무시',
      ignoreConfirm: '이 문제에서 아무것도 배울 수 없다고 생각하며 앞으로 다시 보고 싶지 않으신가요?'
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
    rushingMessage: 'You answered after only {seconds} seconds. Your past results suggest there is {confidence}% probability that rushing like this leads to more mistakes.',
    insufficientReviewMessage: 'Repeat mistake! You have spent only {seconds} seconds reviewing this problem. Your past results suggest there is a {confidence}% probability you will remember it better with more time spent reviewing.',
    iWontRushIt: "I won't rush it.",
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
    statsNotEnough: 'You have completed {count} tsumegos, 5 needed to enable statistics.',
    statsSolvingTimeNatural: 'When you have spent more time solving there has been a {percent_change}% chance finding the correct move.',
    statsReviewTimeNatural: 'When you spend more time reviewing, there’s a {percent_change}% chance that you will find the correct move when encounter the problem again later.',
    statsNoTrends: 'Based on your most recent {analysis} out of {total} problems, there are no statistically significant trends. Keep playing to gather more data!',
    statsFailed: 'Could not calculate stats.',
    statsFailures: 'You have {count_recent} recent failures that will be repeated after a {hours} hour delay.',
    statsOldFailures: 'There are an additional {count_other} past failures that are occasionally repeated.',
    statsNovelProblems: 'You have sofar attempted {count} unique problems.',
    statsIgnoredProblems: "You have ignored {count_ignored} problems, and another {count_slipped} were abandoned due to irregular practice.",
    statsModelCoefficients: 'Model Coefficients:',
    statsIntercept: 'Intercept:',
    ignore: 'Ignore',
    ignoreConfirm: 'Do you feel like you cannot learn anything from this problem and don\'t want to see it again in the future?'
  }
};

export function getTerm(key) {
    const lang = navigator.language || 'en';
    let langDb;
    if (lang.startsWith('zh')) {
        langDb = localization.zh;
    } else if (lang.startsWith('ja')) {
        langDb = localization.ja;
    } else if (lang.startsWith('ko')) {
        langDb = localization.ko;
    } else {
        langDb = localization.en;
    }

    if (langDb && key in langDb) {
        return langDb[key];
    }
    return localization.en[key];
}

export function getDescriptiveRank(numericRank) {
    const rankIndex = Math.floor(Math.pow(numericRank, 1.0) * TOTAL_RANKS);
    const kyuTerm = getTerm('kyu');
    const danTerm = getTerm('dan');
    return (rankIndex < 20) ? `${20 - rankIndex} ${kyuTerm}` : `${rankIndex - 19} ${danTerm}`;
}