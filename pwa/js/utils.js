// js/utils.js

const localization = {
    'zh': { kyu: '级', dan: '段', review: '上一个', points: '点', about: '关于', version: '版本', analyzing: '分析中...', error: '错误', errorLoadingProblem: '加载问题时出错。', appTitle: 'OnlyMove.net', exit: '下一个', score: '得分', undo: '撤销', share: '分享', linkCopied: '链接已复制到剪贴板', tsumegoMode: 'Tsumego模式', reviewMode: '复习模式' },
    'ja': { kyu: '級', dan: '段', review: '前へ', points: '点', about: 'について', version: 'バージョン', analyzing: '分析中...', error: 'エラー', errorLoadingProblem: '問題の読み込みエラー。', appTitle: 'OnlyMove.net', exit: '次へ', score: 'スコア', undo: '元に戻す', share: '共有', linkCopied: 'リンクがクリップボードにコピーされました', tsumegoMode: 'Tsumegoモード', reviewMode: '復習モード' },
    'ko': { kyu: '급', dan: '단', review: '이전', points: '점', about: '정보', version: '버전', analyzing: '분석 중...', error: '오류', errorLoadingProblem: '문제 로딩 오류.', appTitle: 'OnlyMove.net', exit: '다음', score: '점수', undo: '실행 취소', share: '공유', linkCopied: '링크가 클립보드에 복사됨', tsumegoMode: 'Tsumego 모드', reviewMode: '복습 모드' },
    'en': { kyu: 'kyu', dan: 'dan', review: 'Previous', points: 'points', about: 'About', version: 'Version', analyzing: '...', error: 'Error', errorLoadingProblem: 'Error loading problem.', appTitle: 'OnlyMove.net', exit: 'Next', score: 'Score', undo: 'Undo', share: 'Share', linkCopied: 'Link copied to clipboard', tsumegoMode: 'Tsumego mode', reviewMode: 'Review mode' }
};

export function getTerm(key) {
    const lang = navigator.language || 'en';
    if (lang.startsWith('zh')) return localization.zh[key];
    if (lang.startsWith('ja')) return localization.ja[key];
    if (lang.startsWith('ko')) return localization.ko[key];
    return localization.en[key];
}

export function getDescriptiveRank(numericRank) {
    const totalRanks = 29;
    const rankIndex = Math.floor(Math.pow(numericRank, 1.0) * totalRanks);
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
