// js/utils.js

const localization = {
    'zh': { kyu: '级', dan: '段', komi: '贴目', review: '复习', points: '点', about: '关于', version: '版本', chinese: '中国规则', komiValue: '7.5', ruleset: '规则集', analyzing: '分析中...', error: '错误', errorLoadingProblem: '加载问题时出错。', appTitle: 'OnlyMove.net', exit: '退出', score: '得分', undo: '撤销', share: '分享', linkCopied: '链接已复制到剪贴板' },
    'ja': { kyu: '級', dan: '段', komi: 'コミ', review: '復習', points: '点', about: 'について', version: 'バージョン', chinese: '中国ルール', komiValue: '7.5', ruleset: 'ルールセット', analyzing: '分析中...', error: 'エラー', errorLoadingProblem: '問題の読み込みエラー。', appTitle: 'OnlyMove.net', exit: '終了', score: 'スコア', undo: '元に戻す', share: '共有', linkCopied: 'リンクがクリップボードにコピーされました' },
    'ko': { kyu: '급', dan: '단', komi: '덤', review: '복습', points: '점', about: '정보', version: '버전', chinese: '중국 규칙', komiValue: '7.5', ruleset: '규칙 집합', analyzing: '분석 중...', error: '오류', errorLoadingProblem: '문제 로딩 오류.', appTitle: 'OnlyMove.net', exit: '나가기', score: '점수', undo: '실행 취소', share: '공유', linkCopied: '링크가 클립보드에 복사됨' },
    'en': { kyu: 'kyu', dan: 'dan', komi: 'komi', review: 'Review', points: 'points', about: 'About', version: 'Version', chinese: 'Chinese', komiValue: '7.5', ruleset: 'ruleset', analyzing: 'Analyzing...', error: 'Error', errorLoadingProblem: 'Error loading problem.', appTitle: 'OnlyMove.net', exit: 'Exit', score: 'Score', undo: 'Undo', share: 'Share', linkCopied: 'Link copied to clipboard' }
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
