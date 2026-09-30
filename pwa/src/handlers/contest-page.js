// pwa/src/handlers/contest-page.js
//
// GET /contest — the contribution-rewards page: the rules of the 2026 content
// creators category (videos) and every entry so far, rendered from the
// cached contest blob (src/contest.js), which recomputes on request once it is 12
// hours old. Localized on the CLIENT from navigator.language like /statistics
// and /leaderboard, so one response serves every viewer.
//
// The page ranks nothing: entries are listed in submission order, the order the
// rules favour, and the organiser judges on the figures shown.
//
// Ranks print as a fractional dan ("3.9 dan": the continuous rank plus one,
// FLOORED to a tenth so a value never reads as the next dan before it is one) —
// a reader outside the site has no key for the front page's "3d 97%". A kyu
// rank has no such reading, since kyu count down, so it prints as its band label.

import { loadContest, CONTEST_DEADLINE_ISO } from '../contest.js';
import {
    composeAliasAll, ALIAS_ADJECTIVES, ALIAS_COLORS, ALIAS_ANIMALS, ALIAS_JOIN, ALIAS_LANGS,
} from '../../shared/alias-words.js';
import { baseCss, flag, ESC_SNIPPET } from './page-shell.js';

const DISCORD = 'https://discord.gg/ZU7jJNnkbB';
const DISCORD_NAME = 'ootakamoku';

const STR = {
    en: {
        title: 'Content creator rewards 2026',
        intro: 'Videos introducing OnlyMove to new players and showing how a strong player actually reads a problem. The three best submissions win prizes.',
        first: '1st', second: '2nd', third: '3rd',
        deadline: 'Entries close 31 December 2026, 23:59 UTC.',
        daysLeft: '{n} days left.',
        closed: 'Entries are closed.',
        criteriaH: 'Judged with roughly equal weight on',
        prizesH: 'Prizes',
        videoH: 'What to record',
        videoWhat: 'Record yourself solving OnlyMove problems in a single session on a new account, thinking aloud and explaining each one, climbing as high as you can.',
        vLang: 'Record in English, Chinese, Japanese or Korean: the four languages OnlyMove itself is available in, so viewers can follow along in the app in the language they watched.',
        vCrit1: 'Views',
        vCrit2: 'The quality of the explanation, scored by the organiser',
        vCrit3: 'The highest OnlyMove rank reached during the video',
        judging: 'Henry Hemming, the author of OnlyMove, judges the entries on the criteria above and enters their view counts and quality scores, which may be revised until the deadline.',
        contact: 'Send questions to {name} on {discord}.',
        formH: 'Enter',
        formAccount: 'Account code',
        formAccountHint: 'Filled in from this browser. If you played on another device, enter the code shown in the app under menu → Account.',
        formAccountNone: 'No OnlyMove account in this browser. Enter the code shown in the app under menu → Account.',
        acctChecking: 'Checking the account…',
        acctAlias: 'Listed as {alias}.',
        acctNoAlias: 'Listed as Anonymous.',
        acctHidden: 'This account\'s profile is hidden, so it will be listed as Anonymous.',
        acctInvalid: 'That is not a valid account code.',
        formUrl: 'Video link',
        formProblems: 'Problems solved in the video',
        formLanguage: 'Language of the video',
        langEn: 'English', langZh: 'Chinese', langJa: 'Japanese', langKo: 'Korean',
        formAppLink: 'The organiser may link this video from the app, for players of a suitable strength',
        formContact: 'How to reach you about prizes',
        contactWhatsapp: 'WhatsApp', contactDiscord: 'Discord', contactEmail: 'Email',
        phWhatsapp: 'Phone number, e.g. +44 7700 900123', phDiscord: 'Discord username', phEmail: 'name@example.com',
        formContactNote: 'Seen only by the organiser, used only to reach you about this contest, and deleted once the prizes are paid.',
        formMessage: 'Message to the organiser (optional)',
        formSubmit: 'Submit entry',
        formSending: 'Sending…',
        okNew: 'Received. The organiser will verify your entry and add it to the list of entries.',
        okReplaced: 'Received. This replaces your earlier submission.',
        okListed: 'Received. Your entry is already on the list, so this updates your contact details, message and app-link choice. Ask the organiser to change anything else.',
        errContact: 'The contact detail does not look right for the chosen method.',
        errUrl: 'Enter the video link, starting with https://.',
        errProblems: 'Enter how many problems you solved in the video.',
        errRate: 'Too many attempts. Wait a minute and try again.',
        errMaint: 'The site is briefly under maintenance. Try again in a few minutes.',
        errServer: 'The entry could not be sent. Try again later, or contact the organiser on Discord.',
        entriesH: 'Entries so far',
        updated: 'Updated {ago}. Every figure on this page is updated every 12 hours.',
        agoJustNow: 'just now', agoMinutes: '{n} min ago', agoHours: '{n} h ago', agoDays: '{n} d ago',
        colSubmitted: 'Entered', colPlayer: 'Player',
        colProblems: 'Problems', colVideo: 'Video', colViews: 'Views', colQuality: 'Quality', colPeak: 'Highest OnlyMove',
        watch: 'Watch', anonymous: 'Anonymous',
        emptyVideo: 'No videos yet.',
        unavailable: 'The entry list could not be loaded. Reload the page to try again.',
        dan: '{n} dan', kyu: '{n} kyu',
    },
    zh: {
        title: '内容创作者奖励 2026',
        intro: '向新棋手介绍 OnlyMove、展示强手如何真正计算一道题的视频。最出色的三份参赛作品获得奖金。',
        first: '第一名', second: '第二名', third: '第三名',
        deadline: '截止时间：2026 年 12 月 31 日 23:59（UTC）。',
        daysLeft: '还剩 {n} 天。',
        closed: '参赛已截止。',
        criteriaH: '评判时大致同等考虑',
        prizesH: '奖金',
        videoH: '录制内容',
        videoWhat: '在一次连续的时段内，用新账户录制你解答 OnlyMove 题目的过程，边想边说，讲解每道题，尽可能攀升到更高的段位。',
        vLang: '视频可使用英语、中文、日语或韩语录制，也就是 OnlyMove 本身支持的四种语言，这样观众可以在应用中以同一种语言跟着练习。',
        vCrit1: '观看次数',
        vCrit2: '讲解质量，由组织者评分',
        vCrit3: '视频中达到的最高 OnlyMove 段位',
        judging: 'OnlyMove 作者 Henry Hemming 依据上述标准评判参赛作品，并录入观看次数和质量评分，截止前可能调整。',
        contact: '如有问题，请在 {discord} 上联系 {name}。',
        formH: '参赛',
        formAccount: '账户代码',
        formAccountHint: '已根据此浏览器自动填写。如果你在其他设备上解题，请输入应用中“菜单 → 账户”显示的代码。',
        formAccountNone: '此浏览器中没有 OnlyMove 账户。请输入应用中“菜单 → 账户”显示的代码。',
        acctChecking: '正在检查账户……',
        acctAlias: '将以“{alias}”列出。',
        acctNoAlias: '将以“匿名”列出。',
        acctHidden: '此账户的个人资料已隐藏，将以“匿名”列出。',
        acctInvalid: '账户代码无效。',
        formUrl: '视频链接',
        formProblems: '视频中解答的题目数量',
        formLanguage: '视频语言',
        langEn: '英语', langZh: '中文', langJa: '日语', langKo: '韩语',
        formAppLink: '组织者可以在应用中向棋力合适的棋手推荐此视频',
        formContact: '领奖联系方式',
        contactWhatsapp: 'WhatsApp', contactDiscord: 'Discord', contactEmail: '电子邮件',
        phWhatsapp: '电话号码，例如 +86 138 0000 0000', phDiscord: 'Discord 用户名', phEmail: 'name@example.com',
        formContactNote: '仅组织者可见，仅用于就本次活动与你联系，奖金发放后即删除。',
        formMessage: '给组织者的留言（可选）',
        formSubmit: '提交参赛',
        formSending: '正在发送……',
        okNew: '已收到。组织者核实后会将你加入参赛列表。',
        okReplaced: '已收到。此次提交将替换你之前的提交。',
        okListed: '已收到。你已在列表中，此次提交更新了联系方式、留言和应用推荐选项。如需更改其他内容，请联系组织者。',
        errContact: '联系方式与所选方式不符，请检查。',
        errUrl: '请输入以 https:// 开头的视频链接。',
        errProblems: '请输入视频中解答的题目数量。',
        errRate: '尝试次数过多，请稍等一分钟后重试。',
        errMaint: '网站正在短暂维护，请几分钟后重试。',
        errServer: '无法发送参赛信息。请稍后重试，或在 Discord 上联系组织者。',
        entriesH: '目前的参赛',
        updated: '更新于 {ago}。本页所有数值每 12 小时更新一次。',
        agoJustNow: '刚刚', agoMinutes: '{n} 分钟前', agoHours: '{n} 小时前', agoDays: '{n} 天前',
        colSubmitted: '参加日期', colPlayer: '棋手',
        colProblems: '题目数', colVideo: '视频', colViews: '观看', colQuality: '质量', colPeak: 'OnlyMove 最高',
        watch: '观看', anonymous: '匿名',
        emptyVideo: '暂无视频。',
        unavailable: '无法加载参赛列表，请刷新页面重试。',
        dan: '{n} 段', kyu: '{n} 级',
    },
    ja: {
        title: 'コンテンツクリエイター賞 2026',
        intro: 'OnlyMove を新しいプレイヤーに紹介し、強い人が実際にどう問題を読むのかを伝える動画。最も優れた 3 件の応募に賞金があります。',
        first: '1 位', second: '2 位', third: '3 位',
        deadline: '締め切り：2026 年 12 月 31 日 23:59（UTC）。',
        daysLeft: '残り {n} 日。',
        closed: '受付は終了しました。',
        criteriaH: '次をほぼ同じ重みで審査します',
        prizesH: '賞金',
        videoH: '録画する内容',
        videoWhat: '新しいアカウントで、1 回のセッションで OnlyMove の問題を解く様子を、考えを声に出して各問題を説明しながら録画し、できるだけ高いランクまで上がってください。',
        vLang: '動画は英語・中国語・日本語・韓国語のいずれかで録画してください。OnlyMove 自体が対応している 4 言語なので、視聴者は同じ言語のアプリで一緒に解くことができます。',
        vCrit1: '再生回数',
        vCrit2: '解説の質（主催者が採点）',
        vCrit3: '動画中に到達した最高 OnlyMove ランク',
        judging: 'OnlyMove の作者 Henry Hemming が上記の基準で審査し、締め切りまで見直すことのある再生回数と質の評価を入力します。',
        contact: 'ご質問は {discord} で {name} までお送りください。',
        formH: '応募する',
        formAccount: 'アカウントコード',
        formAccountHint: 'このブラウザから自動入力しました。別の端末で解いた場合は、アプリの「メニュー → アカウント」に表示されるコードを入力してください。',
        formAccountNone: 'このブラウザには OnlyMove のアカウントがありません。アプリの「メニュー → アカウント」に表示されるコードを入力してください。',
        acctChecking: 'アカウントを確認しています…',
        acctAlias: '「{alias}」として掲載されます。',
        acctNoAlias: '「匿名」として掲載されます。',
        acctHidden: 'このアカウントのプロフィールは非公開のため、「匿名」として掲載されます。',
        acctInvalid: 'アカウントコードが無効です。',
        formUrl: '動画のリンク',
        formProblems: '動画内で解いた問題数',
        formLanguage: '動画の言語',
        langEn: '英語', langZh: '中国語', langJa: '日本語', langKo: '韓国語',
        formAppLink: 'アプリから棋力の合うプレイヤーにこの動画を紹介してもよい',
        formContact: '賞金についての連絡先',
        contactWhatsapp: 'WhatsApp', contactDiscord: 'Discord', contactEmail: 'メール',
        phWhatsapp: '電話番号（例：+81 90 1234 5678）', phDiscord: 'Discord のユーザー名', phEmail: 'name@example.com',
        formContactNote: '主催者のみが見るもので、このコンテストに関する連絡にだけ使い、賞金の支払い後に削除します。',
        formMessage: '主催者へのメッセージ（任意）',
        formSubmit: '応募する',
        formSending: '送信しています…',
        okNew: '受け付けました。主催者が確認したうえで、応募一覧に追加します。',
        okReplaced: '受け付けました。以前の応募を置き換えます。',
        okListed: '受け付けました。すでに一覧に掲載されているため、連絡先、メッセージ、アプリでの紹介の可否を更新しました。その他の変更は主催者にご連絡ください。',
        errContact: '選んだ連絡方法に対して、連絡先の形式が正しくないようです。',
        errUrl: 'https:// で始まる動画のリンクを入力してください。',
        errProblems: '動画内で解いた問題数を入力してください。',
        errRate: '試行回数が多すぎます。1 分ほど待ってから再度お試しください。',
        errMaint: 'サイトは一時的にメンテナンス中です。数分後に再度お試しください。',
        errServer: '応募を送信できませんでした。後でもう一度試すか、Discord で主催者に連絡してください。',
        entriesH: 'これまでの応募',
        updated: '更新 {ago}。このページのすべての数値は 12 時間ごとに更新されます。',
        agoJustNow: 'たった今', agoMinutes: '{n} 分前', agoHours: '{n} 時間前', agoDays: '{n} 日前',
        colSubmitted: '応募日', colPlayer: 'プレイヤー',
        colProblems: '問題数', colVideo: '動画', colViews: '再生', colQuality: '質', colPeak: 'OnlyMove 最高',
        watch: '見る', anonymous: '匿名',
        emptyVideo: '動画の応募はまだありません。',
        unavailable: '応募一覧を読み込めませんでした。ページを再読み込みしてください。',
        dan: '{n} 段', kyu: '{n} 級',
    },
    ko: {
        title: '콘텐츠 크리에이터 보상 2026',
        intro: '새로운 기사들에게 OnlyMove를 소개하고, 강자가 실제로 문제를 어떻게 읽는지 보여 주는 영상. 가장 뛰어난 참가작 세 편에 상금을 드립니다.',
        first: '1위', second: '2위', third: '3위',
        deadline: '마감: 2026년 12월 31일 23:59 (UTC).',
        daysLeft: '{n}일 남음.',
        closed: '참가가 마감되었습니다.',
        criteriaH: '다음을 대략 같은 비중으로 심사합니다',
        prizesH: '상금',
        videoH: '녹화할 내용',
        videoWhat: '새 계정으로 한 번의 세션에서 OnlyMove 문제를 푸는 모습을, 생각을 소리 내어 말하며 각 문제를 설명하면서 녹화하고, 가능한 한 높은 랭크까지 올라가 보세요.',
        vLang: '영상은 영어, 중국어, 일본어, 한국어 중 하나로 녹화해 주세요. OnlyMove가 지원하는 네 가지 언어이므로 시청자가 같은 언어의 앱에서 함께 따라 풀 수 있습니다.',
        vCrit1: '조회수',
        vCrit2: '해설의 질(주최자가 채점)',
        vCrit3: '영상 중 도달한 최고 OnlyMove 랭크',
        judging: 'OnlyMove 제작자 Henry Hemming이 위 기준에 따라 심사하며, 마감 전까지 조정될 수 있는 조회수와 질 점수를 입력합니다.',
        contact: '문의는 {discord}에서 {name}에게 보내 주세요.',
        formH: '참가 신청',
        formAccount: '계정 코드',
        formAccountHint: '이 브라우저에서 자동으로 채웠습니다. 다른 기기에서 풀었다면 앱의 메뉴 → 계정에 표시된 코드를 입력하세요.',
        formAccountNone: '이 브라우저에는 OnlyMove 계정이 없습니다. 앱의 메뉴 → 계정에 표시된 코드를 입력하세요.',
        acctChecking: '계정을 확인하는 중…',
        acctAlias: '"{alias}"(으)로 표시됩니다.',
        acctNoAlias: '"익명"으로 표시됩니다.',
        acctHidden: '이 계정의 프로필은 숨김 상태라 "익명"으로 표시됩니다.',
        acctInvalid: '잘못된 계정 코드입니다.',
        formUrl: '영상 링크',
        formProblems: '영상에서 푼 문제 수',
        formLanguage: '영상 언어',
        langEn: '영어', langZh: '중국어', langJa: '일본어', langKo: '한국어',
        formAppLink: '주최자가 앱에서 기력이 맞는 기사들에게 이 영상을 소개해도 됩니다',
        formContact: '상금 관련 연락처',
        contactWhatsapp: 'WhatsApp', contactDiscord: 'Discord', contactEmail: '이메일',
        phWhatsapp: '전화번호, 예: +82 10 1234 5678', phDiscord: 'Discord 사용자 이름', phEmail: 'name@example.com',
        formContactNote: '주최자만 볼 수 있으며, 이 대회와 관련된 연락에만 사용하고 상금 지급 후 삭제합니다.',
        formMessage: '주최자에게 남길 메시지(선택)',
        formSubmit: '신청하기',
        formSending: '보내는 중…',
        okNew: '접수되었습니다. 주최자가 확인한 뒤 참가 목록에 추가합니다.',
        okReplaced: '접수되었습니다. 이전에 보낸 신청을 대체합니다.',
        okListed: '접수되었습니다. 이미 목록에 올라 있으므로 연락처, 메시지, 앱 소개 여부를 갱신했습니다. 그 밖의 변경은 주최자에게 요청해 주세요.',
        errContact: '선택한 연락 방법에 맞지 않는 연락처입니다. 확인해 주세요.',
        errUrl: 'https://로 시작하는 영상 링크를 입력하세요.',
        errProblems: '영상에서 푼 문제 수를 입력하세요.',
        errRate: '시도 횟수가 너무 많습니다. 1분 뒤에 다시 시도해 주세요.',
        errMaint: '사이트가 잠시 점검 중입니다. 몇 분 뒤에 다시 시도해 주세요.',
        errServer: '신청을 보내지 못했습니다. 나중에 다시 시도하거나 Discord로 주최자에게 연락해 주세요.',
        entriesH: '지금까지의 참가',
        updated: '{ago} 업데이트. 이 페이지의 모든 수치는 12시간마다 갱신됩니다.',
        agoJustNow: '방금 전', agoMinutes: '{n}분 전', agoHours: '{n}시간 전', agoDays: '{n}일 전',
        colSubmitted: '참가일', colPlayer: '기사',
        colProblems: '문제 수', colVideo: '영상', colViews: '조회수', colQuality: '질', colPeak: 'OnlyMove 최고',
        watch: '보기', anonymous: '익명',
        emptyVideo: '아직 영상이 없습니다.',
        unavailable: '참가 목록을 불러오지 못했습니다. 페이지를 새로 고쳐 주세요.',
        dan: '{n}단', kyu: '{n}급',
    },
};

export async function handleContest(env) {
    let data = null;
    try {
        data = await loadContest(env);
    } catch (e) {
        console.error('contest read failed:', e);
    }
    return new Response(renderPage(buildView(data)), {
        status: 200,
        headers: {
            'Content-Type': 'text/html; charset=utf-8',
            'Cache-Control': 'no-store',
        },
    });
}

const num = (v) => (Number.isFinite(v) ? v : null);

// The blob, reduced to what the page draws. Ranks stay continuous; the page
// formats them in the reader's language. The alias is composed in all four
// languages here; the page picks its language's form.
function buildView(data) {
    if (!data) return null;
    const who = (e) => ({
        name: e.name || null,
        alias: e.name ? null : composeAliasAll(e.alias),
        flag: e.country ? flag(e.country) : '',
        date: typeof e.listedAt === 'string' ? e.listedAt.slice(0, 10) : '',
    });
    return {
        computedAt: data.computedAt,
        video: (data.video || []).map((e) => ({
            ...who(e),
            url: typeof e.url === 'string' && /^https:\/\//.test(e.url) ? e.url : null,
            views: num(e.views),
            quality: num(e.quality),
            peak: num(e.peakRank),
            problems: num(e.problems),
        })),
    };
}

// One Go stone, drawn inline so the category mark needs no asset.
function stone(color, size) {
    const fill = color === 'black' ? 'url(#stoneB)' : 'url(#stoneW)';
    return `<svg class="stone" width="${size}" height="${size}" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9.2" fill="${fill}"/></svg>`;
}

// The alias vocabulary, for the entry form's account preview: /profile returns the
// account's alias as its English triplet, and the page composes it in the reader's
// language with the shared module's own words and join rules.
const ALIAS_WORDS_JSON = JSON.stringify(Object.fromEntries(
    [...ALIAS_ADJECTIVES, ...ALIAS_COLORS, ...ALIAS_ANIMALS].map((w) => [w.en, w])));
const ALIAS_JOIN_JS = `{${ALIAS_LANGS.map((l) => `${l}: ${ALIAS_JOIN[l].toString()}`).join(', ')}}`;

function renderPage(view) {
    const json = JSON.stringify(view).replace(/</g, '\\u003c');
    const strJson = JSON.stringify(STR).replace(/</g, '\\u003c');
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>onlymove — Content creator rewards 2026</title>
<meta name="description" content="Prizes for videos solving OnlyMove problems while explaining the reading. Until the end of 2026.">
<link rel="canonical" href="https://onlymove.net/contest">
<style>
  ${baseCss(760)}
  :root {
    --kaya: #dcb877;
    --kaya-line: rgba(58, 38, 12, .38);
    --kaya-ink: #2a1c08;
    --serif: 'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, 'Noto Serif', serif;
  }
  body { padding-top: 0; }
  h1, h2 { font-family: var(--serif); }
  h2 { font-size: 1.55rem; font-weight: 600; display: flex; align-items: center; gap: .6rem; }
  h3 { font-size: 1rem; font-weight: 600; margin: 1.6rem 0 .5rem; display: flex; align-items: center; gap: .5rem; }
  .stone { flex: none; filter: drop-shadow(0 1px 1.5px rgba(0,0,0,.45)); }
  .lead { font-size: 1.08rem; }
  .muted { color: var(--ink-muted); }

  /* hero: the title over a corner of a board; the two prizes as the two stones */
  .hero { position: relative; padding: 3.2rem 0 2.2rem; margin-bottom: .5rem; }
  .hero::before {
    content: ''; position: absolute; inset: 0 -1rem; z-index: -1; pointer-events: none;
    background-image: linear-gradient(var(--hairline) 1px, transparent 1px),
                      linear-gradient(90deg, var(--hairline) 1px, transparent 1px);
    background-size: 36px 36px; background-position: -1px -1px;
    -webkit-mask-image: linear-gradient(170deg, #000 10%, transparent 78%);
            mask-image: linear-gradient(170deg, #000 10%, transparent 78%);
  }
  .hero h1 { font-size: clamp(2.2rem, 7vw, 3.3rem); line-height: 1.05; margin: 0 0 .7rem; letter-spacing: -.015em; }
  .hero .intro { font-size: 1.15rem; margin: 0 0 1.8rem; }
  .prizes { display: grid; grid-template-columns: minmax(0, 24rem); gap: 1rem; margin: 0 0 1.2rem; }
  .prize {
    display: grid; grid-template-columns: auto 1fr; column-gap: .9rem; align-items: center;
    padding: 1rem 1.1rem; border-radius: 14px; background: var(--panel);
    border: 1px solid var(--hairline); color: var(--ink); text-decoration: none;
  }
  .prize:hover, .prize:focus-visible { border-color: var(--accent); text-decoration: none; color: var(--ink); }
  .prize .stone { grid-row: span 2; }
  .prize .cat { font-weight: 600; font-size: 1.05rem; }
  .prize .amounts { color: var(--ink-muted); font-variant-numeric: tabular-nums; }
  .prize .amounts b { color: var(--ink); font-weight: 600; font-size: 1.1rem; }
  .deadline { margin: 0; }
  .deadline strong { font-weight: 600; }

  /* criteria: three items of equal weight, as three equal cells of board wood
     divided by board lines */
  ul.criteria { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; margin: .4rem 0 .6rem; padding: 0;
                background: var(--kaya-line); border-radius: 6px; overflow: hidden;
                box-shadow: inset 0 0 0 1px rgba(0,0,0,.12), 0 4px 14px rgba(0,0,0,.22); }
  ul.criteria li { list-style: none; margin: 0; padding: .95rem 1rem; background: var(--kaya); color: var(--kaya-ink);
                   font-size: .97rem; font-weight: 500; line-height: 1.45; }
  a.inline { color: inherit; text-decoration: underline; text-decoration-color: var(--accent); text-underline-offset: .18em; }
  a.inline:hover { color: var(--accent-hi); }

  /* the fine print closing the page: when the figures update, how entries are
     judged, where to ask */
  .fine { margin-top: 2.2rem; }
  .fine p { margin: 0 0 .5rem; }

  .tbl { overflow-x: auto; margin: .3rem 0 0; }
  .tbl table { margin: 0; font-variant-numeric: tabular-nums; }
  .tbl th { text-transform: none; letter-spacing: 0; font-size: .82rem; }
  .tbl th, .tbl td { white-space: nowrap; }
  .tbl th.n, .tbl td.n { text-align: right; }
  .tbl td.pl { width: 99%; }
  .empty { color: var(--ink-muted); padding: .9rem 1rem; border: 1px dashed var(--hairline); border-radius: 10px; margin: .3rem 0 0; }

  /* entry form */
  form.enter { display: grid; gap: 1.1rem; margin: .6rem 0 0; padding: 1.3rem 1.25rem; border-radius: 14px;
               background: var(--panel); border: 1px solid var(--hairline); }
  form.enter fieldset { border: 0; padding: 0; margin: 0; min-width: 0; }
  form.enter legend, form.enter .lbl { display: block; font-weight: 600; font-size: .92rem; margin: 0 0 .4rem; padding: 0; }
  form.enter input[type=text], form.enter input[type=url], form.enter input[type=number],
  form.enter input[type=email], form.enter input[type=tel], form.enter textarea {
    font: inherit; width: 100%; box-sizing: border-box; padding: .5rem .65rem; color: var(--ink);
    background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px;
  }
  form.enter input:focus, form.enter textarea:focus, form.enter select:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  form.enter textarea { min-height: 4.5rem; resize: vertical; }
  form.enter select { font-size: .95rem; padding: .45rem 1.6rem .45rem .6rem; }
  form.enter input#fAccount { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; max-width: 14rem; }
  form.enter input#fProblems { max-width: 8rem; }
  .row2 { display: grid; grid-template-columns: auto 1fr; gap: .6rem; align-items: center; }
  .hint { font-size: .88rem; color: var(--ink-muted); margin: .35rem 0 0; }
  .hint.warn { color: var(--accent-hi); }
  .check { display: flex; gap: .55rem; align-items: flex-start; cursor: pointer; }
  .check input { margin-top: .25rem; accent-color: var(--accent); }
  form.enter button {
    justify-self: start; font: inherit; font-weight: 600; padding: .6rem 1.3rem; border-radius: 10px; cursor: pointer;
    border: 1px solid var(--accent); background: var(--accent); color: #fff;
  }
  form.enter button:hover { background: var(--accent-hi); border-color: var(--accent-hi); }
  form.enter button:disabled { opacity: .55; cursor: default; }
  .status { margin: 0; padding: .7rem .9rem; border-radius: 10px; background: var(--surface); }
  .status.err { border-left: 3px solid var(--accent); }
  .status.ok { border-left: 3px solid #6a9a5b; }
  .status:empty { display: none; }
  [hidden] { display: none !important; }

  @media (max-width: 560px) {
    .prizes, ul.criteria { grid-template-columns: 1fr; }
    .row2 { grid-template-columns: 1fr; }
    .hero { padding-top: 2.2rem; }
  }
  @media (prefers-color-scheme: light) {
    ul.criteria { box-shadow: inset 0 0 0 1px rgba(0,0,0,.12), 0 3px 10px rgba(0,0,0,.10); }
  }
</style>
</head>
<body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <defs>
    <radialGradient id="stoneB" cx="35%" cy="30%" r="75%"><stop offset="0%" stop-color="#5a5650"/><stop offset="55%" stop-color="#1a1917"/><stop offset="100%" stop-color="#050505"/></radialGradient>
    <radialGradient id="stoneW" cx="35%" cy="30%" r="80%"><stop offset="0%" stop-color="#ffffff"/><stop offset="70%" stop-color="#e9e6de"/><stop offset="100%" stop-color="#bdb8ab"/></radialGradient>
  </defs>
</svg>

<header class="hero">
  <h1 id="h1"></h1>
  <p class="intro" id="intro"></p>
  <div class="prizes">
    <a class="prize" href="#videos">${stone('white', 34)}<span class="cat" id="prizesH"></span><span class="amounts" id="pVa"></span></a>
  </div>
  <p class="deadline"><span id="deadline"></span> <strong id="daysLeft"></strong></p>
</header>

<section>
  <h2 id="entriesH"></h2>
  <div class="tbl" id="videoList"></div>
</section>

<section id="videos">
  <h2>${stone('white', 26)}<span id="videoH"></span></h2>
  <p id="videoWhat"></p>
  <p id="vLang"></p>
  <h3 id="vCritH"></h3>
  <ul class="criteria"><li id="vCrit1"></li><li id="vCrit2"></li><li id="vCrit3"></li></ul>
</section>


<section id="enter">
  <h2 id="formH"></h2>
  <form class="enter" id="form" novalidate>
    <div>
      <label class="lbl" for="fAccount" id="formAccount"></label>
      <input type="text" id="fAccount" autocomplete="off" autocapitalize="off" spellcheck="false" maxlength="8">
      <p class="hint" id="acctHint"></p>
      <p class="hint" id="acctInfo" aria-live="polite"></p>
    </div>
    <div>
      <div>
        <label class="lbl" for="fUrl" id="formUrl"></label>
        <input type="url" id="fUrl" placeholder="https://" maxlength="500">
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr));gap:1.1rem;margin-top:1.1rem">
        <div>
          <label class="lbl" for="fProblems" id="formProblems"></label>
          <input type="number" id="fProblems" min="1" max="10000" step="1" inputmode="numeric">
        </div>
        <div>
          <label class="lbl" for="fLang" id="formLanguage"></label>
          <select id="fLang"><option value="en"></option><option value="zh"></option><option value="ja"></option><option value="ko"></option></select>
        </div>
      </div>
      <label class="check" style="margin-top:1.1rem"><input type="checkbox" id="fAppLink"><span id="formAppLink"></span></label>
    </div>
    <div>
      <label class="lbl" for="fContact" id="formContact"></label>
      <div class="row2">
        <select id="fMethod" aria-label="Contact method"><option value="whatsapp"></option><option value="discord"></option><option value="email"></option></select>
        <input type="text" id="fContact" maxlength="254">
      </div>
      <p class="hint" id="formContactNote"></p>
    </div>
    <div>
      <label class="lbl" for="fMessage" id="formMessage"></label>
      <textarea id="fMessage" maxlength="1000"></textarea>
    </div>
    <p class="status" id="formStatus" role="status"></p>
    <button type="submit" id="formSubmit"></button>
  </form>
</section>

<section class="fine muted">
  <p id="updated"></p>
  <p id="judging"></p>
  <p id="contact"></p>
</section>

<script>
(function () {
  var STR = ${strJson};
  var VIEW = ${json};
  var DISCORD = ${JSON.stringify(DISCORD)};
  var DISCORD_NAME = ${JSON.stringify(DISCORD_NAME)};
  var DEADLINE = Date.parse(${JSON.stringify(CONTEST_DEADLINE_ISO)});
  ${ESC_SNIPPET}
  function getLang() {
    var l = navigator.language || 'en';
    if (l.indexOf('zh') === 0) return 'zh';
    if (l.indexOf('ja') === 0) return 'ja';
    if (l.indexOf('ko') === 0) return 'ko';
    return 'en';
  }
  var LANG = getLang();
  function t(key, params) {
    var db = STR[LANG] || STR.en;
    var s = (key in db) ? db[key] : STR.en[key];
    if (params) s = s.replace(/\\{(\\w+)\\}/g, function (m, n) { return (n in params) ? String(params[n]) : m; });
    return s;
  }
  function $(id) { return document.getElementById(id); }
  function set(id, key, params) { $(id).textContent = t(key || id, params); }
  document.documentElement.lang = LANG;
  document.title = 'onlymove — ' + t('title');

  // A continuous rank (0 = the 1k/1d edge, one unit per rank) as a fractional
  // dan, floored to a tenth; a kyu rank as its whole band.
  function fmtRank(v) {
    if (v === null || v === undefined || !isFinite(v)) return null;
    if (v >= 0) return t('dan', { n: (Math.floor((v + 1) * 10 + 1e-9) / 10).toFixed(1) });
    return t('kyu', { n: -Math.floor(v) });
  }

  ['intro', 'deadline', 'prizesH', 'videoH', 'videoWhat', 'vLang', 'vCrit1', 'vCrit2', 'vCrit3',
   'judging', 'entriesH'].forEach(function (k) { set(k); });
  set('h1', 'title');
  set('vCritH', 'criteriaH');
  var amounts = esc(t('first')) + ' <b>100 €</b>&emsp;' + esc(t('second')) + ' <b>50 €</b>&emsp;' + esc(t('third')) + ' <b>25 €</b>';
  $('pVa').innerHTML = amounts;
  var days = Math.ceil((DEADLINE - Date.now()) / 86400000);
  $('daysLeft').textContent = days > 0 ? t('daysLeft', { n: days }) : t('closed');
  $('contact').innerHTML = esc(t('contact'))
    .replace('{discord}', '<a href="' + DISCORD + '" target="_blank" rel="noopener">Discord</a>')
    .replace('{name}', '<strong>' + esc(DISCORD_NAME) + '</strong>');

  // The entry form. The account code is the app's own, read from this origin's
  // storage (the app keeps it under localStorage 'userId'); it stays editable
  // for a player who solved on another device, or in an installed app whose
  // storage is its own. The preview under it shows what the list will show.
  var ALIAS_WORDS = ${ALIAS_WORDS_JSON};
  var ALIAS_JOIN = ${ALIAS_JOIN_JS};
  var USER_ID_RE = /^[A-Za-z0-9_-]{8}$/;
  var form = $('form');
  ['formH', 'formAccount', 'formUrl', 'formProblems', 'formLanguage', 'formAppLink',
   'formContact', 'formContactNote', 'formMessage', 'formSubmit'].forEach(function (k) { set(k); });
  [['en', 'langEn'], ['zh', 'langZh'], ['ja', 'langJa'], ['ko', 'langKo']].forEach(function (p) {
    $('fLang').querySelector('option[value="' + p[0] + '"]').textContent = t(p[1]);
  });
  $('fLang').value = LANG;
  var PH = { whatsapp: 'phWhatsapp', discord: 'phDiscord', email: 'phEmail' };
  var INPUT_TYPE = { whatsapp: 'tel', discord: 'text', email: 'email' };
  [['whatsapp', 'contactWhatsapp'], ['discord', 'contactDiscord'], ['email', 'contactEmail']].forEach(function (p) {
    $('fMethod').querySelector('option[value="' + p[0] + '"]').textContent = t(p[1]);
  });
  function syncMethod() {
    var m = $('fMethod').value;
    $('fContact').placeholder = t(PH[m]);
    $('fContact').type = INPUT_TYPE[m];
  }
  $('fMethod').addEventListener('change', syncMethod);
  syncMethod();

  var stored = null;
  try { stored = localStorage.getItem('userId'); } catch (e) { stored = null; }
  if (stored && USER_ID_RE.test(stored)) $('fAccount').value = stored;
  $('acctHint').textContent = t($('fAccount').value ? 'formAccountHint' : 'formAccountNone');

  // The account preview: one /profile read per distinct code, cached here.
  var profiles = {}, lookupTimer = null;
  function lookup(id) {
    if (profiles[id]) return;
    profiles[id] = { pending: true };
    fetch('/profile?userId=' + encodeURIComponent(id), { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (p) { profiles[id] = p || { failed: true }; renderAccount(); })
      .catch(function () { profiles[id] = { failed: true }; renderAccount(); });
  }
  function aliasText(a) {
    if (!a || !a.adjective || !a.color || !a.animal) return null;
    function w(en) { var x = ALIAS_WORDS[en]; return (x && x[LANG]) || en; }
    return (ALIAS_JOIN[LANG] || ALIAS_JOIN.en)(w(a.adjective), w(a.color), w(a.animal));
  }
  function renderAccount() {
    var el = $('acctInfo'), id = $('fAccount').value.trim();
    el.className = 'hint';
    if (!id) { el.textContent = ''; return; }
    if (!USER_ID_RE.test(id)) { el.textContent = t('acctInvalid'); el.className = 'hint warn'; return; }
    var p = profiles[id];
    if (!p) { lookup(id); p = profiles[id]; }
    if (p.pending) { el.textContent = t('acctChecking'); return; }
    if (p.failed) { el.textContent = ''; return; }
    var L = p.linked;
    if (p.hidden) {
      el.textContent = t('acctHidden');
    } else if (L && L.name) {
      el.textContent = t('acctAlias', { alias: L.name });
    } else if (aliasText(p.alias)) {
      el.textContent = t('acctAlias', { alias: aliasText(p.alias) });
    } else {
      el.textContent = t('acctNoAlias');
    }
  }
  $('fAccount').addEventListener('input', function () {
    clearTimeout(lookupTimer);
    lookupTimer = setTimeout(renderAccount, 350);
  });
  renderAccount();

  function status(key, ok) {
    var el = $('formStatus');
    el.textContent = key ? t(key) : '';
    el.className = 'status ' + (ok ? 'ok' : 'err');
  }
  var ERR = {
    invalid_contact: 'errContact', invalid_url: 'errUrl',
    invalid_problems: 'errProblems', closed: 'closed', bad_request: 'acctInvalid',
  };
  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    var id = $('fAccount').value.trim();
    if (!USER_ID_RE.test(id)) { status('acctInvalid'); $('fAccount').focus(); return; }
    var body = {
      userId: id, contactMethod: $('fMethod').value,
      contact: $('fContact').value.trim(), message: $('fMessage').value.trim(),
      videoUrl: $('fUrl').value.trim(), videoProblems: parseInt($('fProblems').value, 10),
      videoLanguage: $('fLang').value, appLink: $('fAppLink').checked,
    };
    if (!/^https:\\/\\//.test(body.videoUrl)) { status('errUrl'); $('fUrl').focus(); return; }
    if (!(body.videoProblems >= 1)) { status('errProblems'); $('fProblems').focus(); return; }
    if (!body.contact) { status('errContact'); $('fContact').focus(); return; }
    var btn = $('formSubmit');
    btn.disabled = true; btn.textContent = t('formSending'); status(null);
    fetch('/contest/submit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) { return { code: r.status, j: j }; });
    }).then(function (res) {
      if (res.j && res.j.ok) {
        status(res.j.listed ? 'okListed' : res.j.replaced ? 'okReplaced' : 'okNew', true);
      } else if (res.code === 429) status('errRate');
      else if (res.code === 503) status('errMaint');
      else status(ERR[res.j && res.j.error] || 'errServer');
    }).catch(function () {
      status('errServer');
    }).then(function () {
      btn.disabled = DEADLINE < Date.now(); btn.textContent = t('formSubmit');
    });
  });
  if (DEADLINE < Date.now()) { $('formSubmit').disabled = true; status('closed'); }

  // Every "OnlyMove" in the prose links to the app. Run once the text is in
  // place; text already inside a link, and the table cells, are left alone.
  function linkBrand() {
    var nodes = [], walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      var n = walker.currentNode;
      if (n.nodeValue.indexOf('OnlyMove') < 0) continue;
      if (n.parentNode.closest('a, table, script, title')) continue;
      if (!n.parentNode.closest('p, li')) continue;
      nodes.push(n);
    }
    nodes.forEach(function (n) {
      var span = document.createElement('span');
      span.innerHTML = esc(n.nodeValue).split('OnlyMove').join('<a class="inline" href="/">OnlyMove</a>');
      while (span.firstChild) n.parentNode.insertBefore(span.firstChild, n);
      n.parentNode.removeChild(n);
    });
  }

  if (!VIEW) { $('videoList').innerHTML = '<p class="empty">' + esc(t('unavailable')) + '</p>'; linkBrand(); return; }

  function ago(iso) {
    var ms = Date.now() - Date.parse(iso);
    if (!isFinite(ms) || ms < 60000) return t('agoJustNow');
    var h = ms / 3600000;
    if (h < 1) return t('agoMinutes', { n: Math.round(ms / 60000) });
    if (h < 24) return t('agoHours', { n: Math.round(h) });
    return t('agoDays', { n: Math.round(h / 24) });
  }
  $('updated').textContent = t('updated', { ago: ago(VIEW.computedAt) });

  function playerCell(r) {
    var label = r.name ? esc(r.name)
      : r.alias ? esc(r.alias[LANG] || r.alias.en)
      : '<span class="muted">' + esc(t('anonymous')) + '</span>';
    return '<td class="pl">' + (r.flag ? r.flag + ' ' : '') + label + '</td>';
  }
  function val(v) { return v === null || v === undefined ? '<span class="muted">—</span>' : esc(v); }
  function table(heads, rows, emptyKey) {
    if (!rows.length) return '<p class="empty">' + esc(t(emptyKey)) + '</p>';
    return '<table><thead><tr>' + heads.map(function (h, i) {
      return '<th' + (i >= 2 ? ' class="n"' : '') + '>' + esc(t(h)) + '</th>';
    }).join('') + '</tr></thead><tbody>' + rows.join('') + '</tbody></table>';
  }

  $('videoList').innerHTML = table(
    ['colSubmitted', 'colPlayer', 'colVideo', 'colViews', 'colQuality', 'colPeak', 'colProblems'],
    VIEW.video.map(function (r) {
      var link = r.url ? '<a href="' + esc(r.url) + '" target="_blank" rel="noopener">' + esc(t('watch')) + '</a>' : val(null);
      return '<tr><td>' + esc(r.date) + '</td>' + playerCell(r) +
        '<td class="n">' + link + '</td><td class="n">' + val(r.views) + '</td><td class="n">' +
        val(r.quality === null ? null : r.quality + ' / 10') + '</td><td class="n">' + val(fmtRank(r.peak)) +
        '</td><td class="n">' + val(r.problems) + '</td></tr>';
    }), 'emptyVideo');
  linkBrand();
})();
</script>
</body>
</html>`;
}
