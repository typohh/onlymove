// shared/alias-words.js
//
// The vocabulary for a player's self-chosen leaderboard alias, ONE entry per
// word with all four UI languages, plus the per-language join that turns three
// words into a name.
//
// This module is the SOURCE the D1 word tables are seeded from
// (scripts/gen-alias-seed.mjs; run it after any edit here, and its --check mode
// is the drift guard). Nothing reads D1 to DISPLAY an alias — the Worker and the
// client both import this file — because the words are static and the four
// translations must ride along with them. D1 is asked only "is this word legal",
// so a stale client bundle cannot widen the vocabulary.
//
// Stored state is always the English word (see user_aliases in schema.sql). The
// other three languages are DISPLAY ONLY: editing a translation, or retiring a
// word from this list, never renames an existing player, and composeAlias falls
// back to the English string for a word this list no longer holds.
//
// TRANSLATION IS DATA, NOT UI COPY. A word cannot be translated one word at a
// time and concatenated: Japanese needs the attributive form (勇敢な) and a の
// between the colour and the animal, Chinese needs the 的 on the adjective, and
// Korean needs the 관형형 (용감한). The English words are Title Case because they
// are shown verbatim; the other languages are whatever reads correctly in the
// join below. The lists are English-first on purpose — the ENGLISH name is the
// identity, and the triplet is unique on those words alone.
//
// Keep every English word distinct within its own list (the D1 tables are keyed
// on it), keep it short enough for the leaderboard's 16rem name cell, and keep
// it safe to attach to a stranger in public: this is a pseudonym, and a careless
// entry is one nobody can take back. The three lists must ALSO be disjoint — a
// colour sitting in the adjective list offers "Golden Golden Fox"-shaped names
// and makes the categories meaningless, so a word belongs to exactly one list.

export const ALIAS_LANGS = ['en', 'zh', 'ja', 'ko'];

export const ALIAS_ADJECTIVES = [
    { en: 'Brave',     zh: '勇敢的',     ja: '勇敢な',       ko: '용감한' },
    { en: 'Gentle',    zh: '温柔的',     ja: '優しい',       ko: '온화한' },
    { en: 'Swift',     zh: '迅捷的',     ja: '素早い',       ko: '빠른' },
    { en: 'Clever',    zh: '聪明的',     ja: '賢い',         ko: '영리한' },
    { en: 'Calm',      zh: '沉着的',     ja: '穏やかな',     ko: '차분한' },
    { en: 'Bold',      zh: '大胆的',     ja: '大胆な',       ko: '대담한' },
    { en: 'Quiet',     zh: '安静的',     ja: '静かな',       ko: '조용한' },
    { en: 'Bright',    zh: '明亮的',     ja: '明るい',       ko: '밝은' },
    { en: 'Dark',      zh: '幽暗的',     ja: '暗い',         ko: '어두운' },
    { en: 'Lucky',     zh: '幸运的',     ja: '幸運な',       ko: '행운의' },
    { en: 'Wise',      zh: '睿智的',     ja: '賢明な',       ko: '지혜로운' },
    { en: 'Fierce',    zh: '凶猛的',     ja: '獰猛な',       ko: '사나운' },
    { en: 'Noble',     zh: '高贵的',     ja: '高貴な',       ko: '고귀한' },
    { en: 'Merry',     zh: '欢快的',     ja: '陽気な',       ko: '명랑한' },
    { en: 'Silent',    zh: '沉默的',     ja: '物静かな',     ko: '고요한' },
    { en: 'Shy',       zh: '害羞的',     ja: '内気な',       ko: '수줍은' },
    { en: 'Proud',     zh: '骄傲的',     ja: '誇り高い',     ko: '자랑스러운' },
    { en: 'Curious',   zh: '好奇的',     ja: '好奇心旺盛な', ko: '호기심 많은' },
    { en: 'Sleepy',    zh: '困倦的',     ja: '眠たい',       ko: '졸린' },
    { en: 'Playful',   zh: '顽皮的',     ja: '遊び好きな',   ko: '장난스러운' },
    { en: 'Patient',   zh: '耐心的',     ja: '忍耐強い',     ko: '인내심 강한' },
    { en: 'Loyal',     zh: '忠诚的',     ja: '忠実な',       ko: '충실한' },
    { en: 'Wild',      zh: '狂野的',     ja: '荒々しい',     ko: '거친' },
    { en: 'Ancient',   zh: '古老的',     ja: '古い',         ko: '고대의' },
    { en: 'Frosty',    zh: '冰霜的',     ja: '氷のような',   ko: '서리 같은' },
    { en: 'Sunny',     zh: '晴朗的',     ja: '晴れやかな',   ko: '화창한' },
    { en: 'Misty',     zh: '朦胧的',     ja: '霞んだ',       ko: '안개 낀' },
    { en: 'Cosmic',    zh: '宇宙的',     ja: '宇宙の',       ko: '우주의' },
    { en: 'Hidden',    zh: '隐秘的',     ja: '隠れた',       ko: '숨겨진' },
    { en: 'Dancing',   zh: '舞蹈的',     ja: '踊る',         ko: '춤추는' },
    { en: 'Singing',   zh: '歌唱的',     ja: '歌う',         ko: '노래하는' },
    { en: 'Wandering', zh: '漫游的',     ja: 'さまよう',     ko: '떠도는' },
    { en: 'Dashing',   zh: '潇洒的',     ja: '颯爽とした',   ko: '멋진' },
    { en: 'Humble',    zh: '谦逊的',     ja: '謙虚な',       ko: '겸손한' },
    { en: 'Steady',    zh: '稳健的',     ja: '着実な',       ko: '꾸준한' },
    { en: 'Radiant',   zh: '光辉的',     ja: '輝く',         ko: '빛나는' },
    { en: 'Iron',      zh: '钢铁的',     ja: '鋼の',         ko: '강철의' },
    { en: 'Whirling',  zh: '旋转的',     ja: '回る',         ko: '소용돌이치는' },
];

export const ALIAS_COLORS = [
    { en: 'Crimson', zh: '深红',   ja: '深紅',       ko: '진홍' },
    { en: 'Azure',   zh: '蔚蓝',   ja: '紺碧',       ko: '하늘' },
    { en: 'Emerald', zh: '翠绿',   ja: 'エメラルド', ko: '에메랄드' },
    { en: 'Golden',  zh: '金色',   ja: '金色',       ko: '황금' },
    { en: 'Silver',  zh: '银色',   ja: '銀色',       ko: '은색' },
    { en: 'Violet',  zh: '紫罗兰', ja: '菫色',       ko: '제비꽃' },
    { en: 'Indigo',  zh: '靛蓝',   ja: '藍色',       ko: '남색' },
    { en: 'Scarlet', zh: '猩红',   ja: '緋色',       ko: '주홍' },
    { en: 'Amber',   zh: '琥珀',   ja: '琥珀',       ko: '호박' },
    { en: 'Jade',    zh: '碧玉',   ja: '翡翠',       ko: '옥색' },
    { en: 'Coral',   zh: '珊瑚',   ja: '珊瑚',       ko: '산호' },
    { en: 'Ivory',   zh: '象牙',   ja: '象牙',       ko: '상아' },
    { en: 'Onyx',    zh: '墨黑',   ja: '黒曜',       ko: '흑요석' },
    { en: 'Copper',  zh: '铜色',   ja: '銅色',       ko: '구리' },
    { en: 'Bronze',  zh: '青铜',   ja: '青銅',       ko: '청동' },
    { en: 'Pearl',   zh: '珍珠',   ja: '真珠',       ko: '진주' },
    { en: 'Rose',    zh: '玫瑰',   ja: '薔薇',       ko: '장미' },
    { en: 'Moss',    zh: '苔绿',   ja: '苔',         ko: '이끼' },
    { en: 'Slate',   zh: '石板灰', ja: '石板',       ko: '석판' },
    { en: 'Snow',    zh: '雪白',   ja: '雪',         ko: '설백' },
];

export const ALIAS_ANIMALS = [
    { en: 'Fox',      zh: '狐狸',   ja: '狐',         ko: '여우' },
    { en: 'Wolf',     zh: '狼',     ja: '狼',         ko: '늑대' },
    { en: 'Crane',    zh: '鹤',     ja: '鶴',         ko: '두루미' },
    { en: 'Tiger',    zh: '老虎',   ja: '虎',         ko: '호랑이' },
    { en: 'Dragon',   zh: '龙',     ja: '龍',         ko: '용' },
    { en: 'Panda',    zh: '熊猫',   ja: 'パンダ',     ko: '판다' },
    { en: 'Otter',    zh: '水獭',   ja: '川獺',       ko: '수달' },
    { en: 'Falcon',   zh: '猎鹰',   ja: '隼',         ko: '매' },
    { en: 'Heron',    zh: '苍鹭',   ja: '鷺',         ko: '왜가리' },
    { en: 'Badger',   zh: '獾',     ja: 'アナグマ',   ko: '오소리' },
    { en: 'Koala',    zh: '考拉',   ja: 'コアラ',     ko: '코알라' },
    { en: 'Raven',    zh: '渡鸦',   ja: '鴉',         ko: '까마귀' },
    { en: 'Salmon',   zh: '鲑鱼',   ja: '鮭',         ko: '연어' },
    { en: 'Turtle',   zh: '乌龟',   ja: '亀',         ko: '거북' },
    { en: 'Rabbit',   zh: '兔子',   ja: '兎',         ko: '토끼' },
    { en: 'Sparrow',  zh: '麻雀',   ja: '雀',         ko: '참새' },
    { en: 'Beetle',   zh: '甲虫',   ja: '甲虫',       ko: '딱정벌레' },
    { en: 'Dolphin',  zh: '海豚',   ja: '海豚',       ko: '돌고래' },
    { en: 'Panther',  zh: '黑豹',   ja: '豹',         ko: '표범' },
    { en: 'Bison',    zh: '野牛',   ja: '野牛',       ko: '들소' },
    { en: 'Lynx',     zh: '猞猁',   ja: '山猫',       ko: '스라소니' },
    { en: 'Swan',     zh: '天鹅',   ja: '白鳥',       ko: '백조' },
    { en: 'Owl',      zh: '猫头鹰', ja: '梟',         ko: '부엉이' },
    { en: 'Deer',     zh: '鹿',     ja: '鹿',         ko: '사슴' },
    { en: 'Seal',     zh: '海豹',   ja: '海豹',       ko: '물개' },
    { en: 'Moth',     zh: '飞蛾',   ja: '蛾',         ko: '나방' },
    { en: 'Cricket',  zh: '蟋蟀',   ja: '蟋蟀',       ko: '귀뚜라미' },
    { en: 'Magpie',   zh: '喜鹊',   ja: '鵲',         ko: '까치' },
    { en: 'Stork',    zh: '鹳',     ja: '鸛',         ko: '황새' },
    { en: 'Boar',     zh: '野猪',   ja: '猪',         ko: '멧돼지' },
    { en: 'Hare',     zh: '野兔',   ja: '野兎',       ko: '산토끼' },
    { en: 'Lion',     zh: '狮子',   ja: '獅子',       ko: '사자' },
    { en: 'Bear',     zh: '熊',     ja: '熊',         ko: '곰' },
    { en: 'Horse',    zh: '马',     ja: '馬',         ko: '말' },
    { en: 'Eagle',    zh: '鹰',     ja: '鷲',         ko: '독수리' },
    { en: 'Serpent',  zh: '蛇',     ja: '蛇',         ko: '뱀' },
    { en: 'Whale',    zh: '鲸',     ja: '鯨',         ko: '고래' },
    { en: 'Phoenix',  zh: '凤凰',   ja: '鳳凰',       ko: '봉황' },
    { en: 'Griffin',  zh: '狮鹫',   ja: 'グリフォン', ko: '그리핀' },
    { en: 'Mongoose', zh: '猫鼬',   ja: 'マングース', ko: '몽구스' },
];

// The part grammar the words cannot carry: Chinese suffixes the adjective with
// 的, Japanese takes the attributive form plus の before the animal, English and
// Korean separate with spaces. A translator reviewing a list should review this
// alongside it — the same adjective reads wrong under the wrong join.
export const ALIAS_JOIN = {
    en: (a, c, n) => `${a} ${c} ${n}`,
    zh: (a, c, n) => `${a}${c}${n}`,
    ja: (a, c, n) => `${a}${c}の${n}`,
    ko: (a, c, n) => `${a} ${c} ${n}`,
};

const BY_EN = new Map();
for (const list of [ALIAS_ADJECTIVES, ALIAS_COLORS, ALIAS_ANIMALS]) {
    for (const w of list) BY_EN.set(w.en, w);
}

// The word in one language, falling back to English for a word this list no
// longer holds (a retired word must still render for whoever claimed it).
export function localizedWord(en, lang) {
    const w = BY_EN.get(en);
    return (w && w[lang]) || en;
}

// Compose a stored English triplet into a display string. Unknown pieces (a
// triplet from a build that knew words this one does not) are shown verbatim in
// English rather than dropped, so a name never renders as a hole.
export function composeAlias(adjective, color, animal, lang) {
    const join = ALIAS_JOIN[lang] || ALIAS_JOIN.en;
    return join(localizedWord(adjective, lang), localizedWord(color, lang), localizedWord(animal, lang));
}

// Every language at once, for the server-rendered leaderboard page, which
// resolves its language on the CLIENT (the page is edge-cached, so it cannot
// pick one). Returns null for an absent alias.
export function composeAliasAll(alias) {
    if (!alias || !alias.adjective || !alias.color || !alias.animal) return null;
    const out = {};
    for (const lang of ALIAS_LANGS) {
        out[lang] = composeAlias(alias.adjective, alias.color, alias.animal, lang);
    }
    return out;
}
