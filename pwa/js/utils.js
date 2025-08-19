// js/utils.js

const localization = {
    'zh': { kyu: '级', dan: '段', komi: '贴目', previous: '上一题', points: '点', rules: '规则', version: '版本', chinese: '中国规则', komiValue: '7.5', ruleset: '规则集' },
    'ja': { kyu: '級', dan: '段', komi: 'コミ', previous: '前の問題', points: '点', rules: 'ルール', version: 'バージョン', chinese: '中国ルール', komiValue: '7.5', ruleset: 'ルールセット' },
    'ko': { kyu: '급', dan: '단', komi: '덤', previous: '이전 문제', points: '점', rules: '규칙', version: '버전', chinese: '중국 규칙', komiValue: '7.5', ruleset: '규칙 집합' },
    'en': { kyu: 'kyu', dan: 'dan', komi: 'komi', previous: 'Previous', points: 'points', rules: 'Rules', version: 'Version', chinese: 'Chinese', komiValue: '7.5', ruleset: 'ruleset' }
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
