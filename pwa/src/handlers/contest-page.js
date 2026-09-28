// pwa/src/handlers/contest-page.js
//
// GET /contest — the contribution-rewards page: the rules of the two 2026
// categories (strong players, videos) and every entry so far, rendered from the
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

import { loadContest } from '../contest.js';
import { composeAliasAll } from '../../shared/alias-words.js';
import { baseCss, flag, ESC_SNIPPET } from './page-shell.js';

const DISCORD = 'https://discord.gg/ZU7jJNnkbB';
const DISCORD_NAME = 'ootakamoku';
const DEADLINE_ISO = '2026-12-31T23:59:59Z';

const STR = {
    en: {
        title: 'Contribution rewards 2026',
        intro: 'Two ways to help OnlyMove, with prizes for the best of each.',
        first: '1st', second: '2nd',
        deadline: 'Entries close 31 December 2026, 23:59 UTC.',
        daysLeft: '{n} days left.',
        closed: 'Entries are closed.',
        strengthH: 'Strong players',
        strengthLead: '[Hundreds of players are active on OnlyMove], but very few are stronger than the organiser, so the top of the rank scale rests on thin data. Verified strong players solving many problems near their best are what calibrates it.',
        strengthWho: 'Open to players with an EGF or AGA rank who are stronger than the organiser. New and existing accounts are both welcome.',
        strengthWhoOrg: 'Open to players with an EGF or AGA rank who are stronger than the organiser, currently {egf} in EGF (GoR {gor}) and {om} on OnlyMove; both figures update with the lists below. New and existing accounts are both welcome.',
        criteriaH: 'Judged with roughly equal weight on',
        sCrit1: 'Your EGF or AGA rank',
        sCrit2: 'Your account\'s average OnlyMove rank over its whole history',
        sCrit3: 'The number of problems your account has solved',
        howH: 'How to enter',
        sStep1: 'Link your EGD or AGA record in the app (menu → Account).',
        sStep2: 'Send the organiser your name or your EGD PIN / AGA id.',
        sStep3: 'The organiser verifies your identity through other Go players and adds you to the list.',
        sAnon: 'To stay anonymous here, hide your profile in the same Account dialog. You are listed as Anonymous, and your rank is still shown.',
        videoH: 'Videos',
        videoLead: 'Videos introduce OnlyMove to new players and show how a strong player actually reads a problem.',
        videoWhat: 'Record yourself solving OnlyMove problems in a single session, thinking aloud and explaining how you solve each one, starting from a new account and climbing as high as you can.',
        vLang: 'Record in English, Chinese, Japanese or Korean: the four languages OnlyMove itself is available in, so viewers can follow along in the app in the language they watched.',
        vCrit1: 'Views',
        vCrit2: 'The quality of the explanation, scored by the organiser and open to revision until the deadline',
        vCrit3: 'The highest OnlyMove rank reached during the video',
        vMin: 'A video needs at least 50 views to qualify.',
        vStep1: 'At the start of the recording, create a new account and pick an alias for it (menu → Account).',
        vStep2: 'Solve in one session, thinking aloud and explaining each problem.',
        vStep3: 'Publish the video and send the organiser the link, the alias, and how many problems you solved in it.',
        vLink: 'With your agreement, the organiser may link your video from the app for players of the matching strength.',
        judgingH: 'Judging',
        judging: 'Both categories are judged by Henry Hemming, the author of OnlyMove, on the criteria above. In both, earlier entries are slightly favoured, counted from the day they appear on the list below. Entries the organiser considers dishonest are disqualified.',
        contact: 'Send entries and questions to {name} on {discord}.',
        entriesH: 'Entries so far',
        updated: 'Updated {ago}. The lists refresh every 12 hours.',
        agoJustNow: 'just now', agoMinutes: '{n} min ago', agoHours: '{n} h ago', agoDays: '{n} d ago',
        colSubmitted: 'Entered', colPlayer: 'Player', colIrl: 'EGF / AGA', colAvg: 'Average OnlyMove',
        colProblems: 'Problems', colVideo: 'Video', colViews: 'Views', colQuality: 'Quality', colPeak: 'Highest OnlyMove',
        watch: 'Watch', anonymous: 'Anonymous',
        emptyStrength: 'No strong-player entries yet.',
        emptyVideo: 'No videos yet.',
        unavailable: 'The entry lists could not be loaded. Reload the page to try again.',
        dan: '{n} dan', kyu: '{n} kyu', pro: '{n} pro',
    },
    zh: {
        title: '贡献奖励 2026',
        intro: '两种帮助 OnlyMove 的方式，每类中最出色者获得奖金。',
        first: '第一名', second: '第二名',
        deadline: '截止时间：2026 年 12 月 31 日 23:59（UTC）。',
        daysLeft: '还剩 {n} 天。',
        closed: '参赛已截止。',
        strengthH: '强手棋手',
        strengthLead: 'OnlyMove 有[数百名活跃用户]，但比组织者更强的棋手寥寥无几，因此段位刻度的顶端数据很少。经过验证的强手在接近自身最佳水平时大量解题，才能校准这一段。',
        strengthWho: '面向拥有 EGF 或 AGA 段位、且棋力强于组织者的棋手。新账户和已有账户均可参加。',
        strengthWhoOrg: '面向拥有 EGF 或 AGA 段位、且棋力强于组织者的棋手；组织者目前为 EGF {egf}（GoR {gor}），OnlyMove {om}，两项数值随下方列表一同更新。新账户和已有账户均可参加。',
        criteriaH: '评判时大致同等考虑',
        sCrit1: '你的 EGF 或 AGA 段位',
        sCrit2: '账户全部历史的 OnlyMove 平均段位',
        sCrit3: '账户解答的题目数量',
        howH: '参加方法',
        sStep1: '在应用中关联你的 EGD 或 AGA 记录（菜单 → 账户）。',
        sStep2: '将你的姓名或 EGD PIN / AGA ID 发送给组织者。',
        sStep3: '组织者通过其他围棋棋手核实你的身份，并将你加入列表。',
        sAnon: '如希望在此匿名，可在同一账户对话框中隐藏个人资料。你将以“匿名”列出，段位仍会显示。',
        videoH: '视频',
        videoLead: '视频能向新棋手介绍 OnlyMove，并展示强手如何真正地计算一道题。',
        videoWhat: '录制你在一次连续的时段内解答 OnlyMove 题目的过程，边想边说，讲解每道题的解法；从新账户开始，尽可能攀升到更高的段位。',
        vLang: '视频可使用英语、中文、日语或韩语录制，也就是 OnlyMove 本身支持的四种语言，这样观众可以在应用中以同一种语言跟着练习。',
        vCrit1: '观看次数',
        vCrit2: '讲解质量，由组织者评分，截止前可能调整',
        vCrit3: '视频中达到的最高 OnlyMove 段位',
        vMin: '视频至少需要 50 次观看才有资格参评。',
        vStep1: '在录制开始时注册新账户，并为其选择一个别名（菜单 → 账户）。',
        vStep2: '在一次连续的时段内解题，边想边说，讲解每道题。',
        vStep3: '发布视频，并将链接、别名以及视频中解答的题目数量发送给组织者。',
        vLink: '经你同意，组织者可能会在应用中向相应棋力的棋手推荐你的视频。',
        judgingH: '评判',
        judging: '两个类别均由 OnlyMove 作者 Henry Hemming 依据上述标准评判。两个类别中较早参加者均略有优先，以出现在下方列表中的日期为准。组织者认为不诚实的参赛将被取消资格。',
        contact: '请在 {discord} 上联系 {name}，提交参赛和咨询问题。',
        entriesH: '目前的参赛',
        updated: '更新于 {ago}。列表每 12 小时刷新一次。',
        agoJustNow: '刚刚', agoMinutes: '{n} 分钟前', agoHours: '{n} 小时前', agoDays: '{n} 天前',
        colSubmitted: '参加日期', colPlayer: '棋手', colIrl: 'EGF / AGA', colAvg: 'OnlyMove 平均',
        colProblems: '题目数', colVideo: '视频', colViews: '观看', colQuality: '质量', colPeak: 'OnlyMove 最高',
        watch: '观看', anonymous: '匿名',
        emptyStrength: '暂无强手参赛。',
        emptyVideo: '暂无视频。',
        unavailable: '无法加载参赛列表，请刷新页面重试。',
        dan: '{n} 段', kyu: '{n} 级', pro: '职业 {n} 段',
    },
    ja: {
        title: '貢献者リワード 2026',
        intro: 'OnlyMove を手伝う 2 つの方法。それぞれ最も優れたものに賞金があります。',
        first: '1 位', second: '2 位',
        deadline: '締め切り：2026 年 12 月 31 日 23:59（UTC）。',
        daysLeft: '残り {n} 日。',
        closed: '受付は終了しました。',
        strengthH: '強豪プレイヤー',
        strengthLead: 'OnlyMove には[数百人のアクティブユーザー]がいますが、主催者より強いプレイヤーはごくわずかで、ランクの上位はデータが薄い状態です。実力が確認された強豪が自己ベスト付近で数多く解くことが、この範囲の較正につながります。',
        strengthWho: 'EGF または AGA の段級位を持ち、主催者より強い方が対象です。新規アカウントでも既存アカウントでも参加できます。',
        strengthWhoOrg: 'EGF または AGA の段級位を持ち、主催者より強い方が対象です。主催者は現在 EGF {egf}（GoR {gor}）、OnlyMove {om} で、どちらも下の一覧と一緒に更新されます。新規アカウントでも既存アカウントでも参加できます。',
        criteriaH: '次をほぼ同じ重みで審査します',
        sCrit1: 'EGF または AGA の段級位',
        sCrit2: 'アカウント全履歴での OnlyMove 平均ランク',
        sCrit3: 'アカウントが解いた問題数',
        howH: '参加方法',
        sStep1: 'アプリで EGD または AGA の記録を連携します（メニュー → アカウント）。',
        sStep2: '氏名または EGD PIN / AGA ID を主催者に送ります。',
        sStep3: '主催者が他の碁打ちを通じて本人確認を行い、一覧に追加します。',
        sAnon: 'ここで匿名にしたい場合は、同じアカウント画面でプロフィールを非公開にしてください。「匿名」として掲載され、段級位は表示されます。',
        videoH: '動画',
        videoLead: '動画は OnlyMove を新しいプレイヤーに紹介し、強い人が実際にどう問題を読むのかを伝えます。',
        videoWhat: '1 回のセッションで OnlyMove の問題を解く様子を、考えを声に出して解き方を説明しながら録画してください。新しいアカウントから始めて、できるだけ高いランクまで上がってください。',
        vLang: '動画は英語・中国語・日本語・韓国語のいずれかで録画してください。OnlyMove 自体が対応している 4 言語なので、視聴者は同じ言語のアプリで一緒に解くことができます。',
        vCrit1: '再生回数',
        vCrit2: '解説の質（主催者が採点し、締め切りまで見直すことがあります）',
        vCrit3: '動画中に到達した最高 OnlyMove ランク',
        vMin: '審査対象となるには 50 回以上の再生が必要です。',
        vStep1: '録画の最初に新しいアカウントを作り、別名を選びます（メニュー → アカウント）。',
        vStep2: '1 回のセッションで、考えを声に出し、各問題を説明しながら解きます。',
        vStep3: '動画を公開し、リンク、別名、動画内で解いた問題数を主催者に送ります。',
        vLink: '了承をいただければ、該当する棋力のプレイヤー向けにアプリから動画を紹介することがあります。',
        judgingH: '審査',
        judging: '両部門とも、OnlyMove の作者 Henry Hemming が上記の基準で審査します。どちらの部門でも早い参加がやや有利で、下の一覧に掲載された日を基準とします。不正と判断した応募は失格とします。',
        contact: '応募と質問は {discord} で {name} までお送りください。',
        entriesH: 'これまでの応募',
        updated: '更新 {ago}。一覧は 12 時間ごとに更新されます。',
        agoJustNow: 'たった今', agoMinutes: '{n} 分前', agoHours: '{n} 時間前', agoDays: '{n} 日前',
        colSubmitted: '応募日', colPlayer: 'プレイヤー', colIrl: 'EGF / AGA', colAvg: 'OnlyMove 平均',
        colProblems: '問題数', colVideo: '動画', colViews: '再生', colQuality: '質', colPeak: 'OnlyMove 最高',
        watch: '見る', anonymous: '匿名',
        emptyStrength: '強豪部門の応募はまだありません。',
        emptyVideo: '動画の応募はまだありません。',
        unavailable: '応募一覧を読み込めませんでした。ページを再読み込みしてください。',
        dan: '{n} 段', kyu: '{n} 級', pro: 'プロ {n} 段',
    },
    ko: {
        title: '기여 보상 2026',
        intro: 'OnlyMove를 돕는 두 가지 방법, 각 부문 최고의 기여에 상금을 드립니다.',
        first: '1위', second: '2위',
        deadline: '마감: 2026년 12월 31일 23:59 (UTC).',
        daysLeft: '{n}일 남음.',
        closed: '참가가 마감되었습니다.',
        strengthH: '강자 부문',
        strengthLead: 'OnlyMove에는 [수백 명의 활성 사용자]가 있지만 주최자보다 강한 기사는 매우 적어서, 랭크 척도의 최상단은 데이터가 부족합니다. 실력이 검증된 강자가 자신의 최고 수준 근처에서 많은 문제를 푸는 것이 이 구간을 보정합니다.',
        strengthWho: 'EGF 또는 AGA 급수가 있고 주최자보다 강한 기사가 대상입니다. 새 계정과 기존 계정 모두 참가할 수 있습니다.',
        strengthWhoOrg: 'EGF 또는 AGA 급수가 있고 주최자보다 강한 기사가 대상입니다. 주최자는 현재 EGF {egf}(GoR {gor}), OnlyMove {om}이며, 두 수치 모두 아래 목록과 함께 갱신됩니다. 새 계정과 기존 계정 모두 참가할 수 있습니다.',
        criteriaH: '다음을 대략 같은 비중으로 심사합니다',
        sCrit1: 'EGF 또는 AGA 급수',
        sCrit2: '계정 전체 기록에 걸친 OnlyMove 평균 랭크',
        sCrit3: '계정이 푼 문제 수',
        howH: '참가 방법',
        sStep1: '앱에서 EGD 또는 AGA 기록을 연결합니다(메뉴 → 계정).',
        sStep2: '이름 또는 EGD PIN / AGA ID를 주최자에게 보냅니다.',
        sStep3: '주최자가 다른 바둑 기사들을 통해 신원을 확인하고 목록에 추가합니다.',
        sAnon: '여기서 익명으로 남고 싶다면 같은 계정 화면에서 프로필을 숨기세요. "익명"으로 표시되며 급수는 그대로 표시됩니다.',
        videoH: '영상 부문',
        videoLead: '영상은 새로운 기사들에게 OnlyMove를 소개하고, 강자가 실제로 문제를 어떻게 읽는지 보여 줍니다.',
        videoWhat: '한 번의 세션에서 OnlyMove 문제를 푸는 모습을, 생각을 소리 내어 말하고 풀이를 설명하면서 녹화해 주세요. 새 계정으로 시작해 가능한 한 높은 랭크까지 올라가 보세요.',
        vLang: '영상은 영어, 중국어, 일본어, 한국어 중 하나로 녹화해 주세요. OnlyMove가 지원하는 네 가지 언어이므로 시청자가 같은 언어의 앱에서 함께 따라 풀 수 있습니다.',
        vCrit1: '조회수',
        vCrit2: '해설의 질(주최자가 채점하며 마감 전까지 조정될 수 있음)',
        vCrit3: '영상 중 도달한 최고 OnlyMove 랭크',
        vMin: '심사 대상이 되려면 조회수가 50회 이상이어야 합니다.',
        vStep1: '녹화를 시작할 때 새 계정을 만들고 별명을 고릅니다(메뉴 → 계정).',
        vStep2: '한 번의 세션에서 생각을 소리 내어 말하고 각 문제를 설명하며 풉니다.',
        vStep3: '영상을 공개하고 링크, 별명, 영상에서 푼 문제 수를 주최자에게 보냅니다.',
        vLink: '동의해 주시면 해당 기력의 기사들을 위해 앱에서 영상을 소개할 수 있습니다.',
        judgingH: '심사',
        judging: '두 부문 모두 OnlyMove 제작자 Henry Hemming이 위 기준에 따라 심사합니다. 두 부문 모두 먼저 참가할수록 약간 유리하며, 아래 목록에 오른 날을 기준으로 합니다. 주최자가 부정하다고 판단한 참가는 실격 처리됩니다.',
        contact: '참가 신청과 문의는 {discord}에서 {name}에게 보내 주세요.',
        entriesH: '지금까지의 참가',
        updated: '{ago} 업데이트. 목록은 12시간마다 갱신됩니다.',
        agoJustNow: '방금 전', agoMinutes: '{n}분 전', agoHours: '{n}시간 전', agoDays: '{n}일 전',
        colSubmitted: '참가일', colPlayer: '기사', colIrl: 'EGF / AGA', colAvg: 'OnlyMove 평균',
        colProblems: '문제 수', colVideo: '영상', colViews: '조회수', colQuality: '질', colPeak: 'OnlyMove 최고',
        watch: '보기', anonymous: '익명',
        emptyStrength: '아직 강자 부문 참가자가 없습니다.',
        emptyVideo: '아직 영상이 없습니다.',
        unavailable: '참가 목록을 불러오지 못했습니다. 페이지를 새로 고쳐 주세요.',
        dan: '{n}단', kyu: '{n}급', pro: '프로 {n}단',
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
        date: typeof e.submittedAt === 'string' ? e.submittedAt.slice(0, 10) : '',
    });
    const org = data.organiser || {};
    return {
        computedAt: data.computedAt,
        organiser: Number.isFinite(org.egfRank) && Number.isFinite(org.onlymoveRank)
            ? { egf: org.egfRank, gor: num(org.gor), om: org.onlymoveRank } : null,
        strength: (data.strength || []).map((e) => ({
            ...who(e),
            irl: e.irl ? { fed: e.irl.fed, label: e.irl.label } : null,
            avg: num(e.avgRank),
            problems: num(e.problems),
        })),
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

// One Go stone, drawn inline so the category marks need no asset: black marks
// the strong-player category and white the video one, wherever either appears.
function stone(color, size) {
    const fill = color === 'black' ? 'url(#stoneB)' : 'url(#stoneW)';
    return `<svg class="stone" width="${size}" height="${size}" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9.2" fill="${fill}"/></svg>`;
}

function renderPage(view) {
    const json = JSON.stringify(view).replace(/</g, '\\u003c');
    const strJson = JSON.stringify(STR).replace(/</g, '\\u003c');
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>onlymove — Contribution rewards 2026</title>
<meta name="description" content="Prizes for strong EGF/AGA players who climb the OnlyMove ranks, and for videos solving OnlyMove problems while explaining the reading. Until the end of 2026.">
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
  .hero .intro { font-size: 1.15rem; margin: 0 0 1.8rem; max-width: 34em; }
  .prizes { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 0 0 1.2rem; }
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

  /* how to enter: a real sequence, so numbered */
  ol.steps { counter-reset: step; list-style: none; padding: 0; margin: .4rem 0 .8rem; }
  ol.steps li { counter-increment: step; position: relative; padding: .1rem 0 .7rem 2.4rem; margin: 0; }
  ol.steps li::before {
    content: counter(step); position: absolute; left: 0; top: .05rem; width: 1.6rem; height: 1.6rem;
    border-radius: 50%; border: 1px solid var(--hairline); display: grid; place-items: center;
    font-size: .85rem; font-weight: 600; color: var(--ink-muted); font-variant-numeric: tabular-nums;
  }

  .judging { padding: 1.1rem 1.25rem; border-left: 3px solid var(--accent); background: var(--surface); border-radius: 0 10px 10px 0; }
  .judging p { margin: 0 0 .6rem; }
  .judging p:last-child { margin: 0; }

  .tbl { overflow-x: auto; margin: .3rem 0 0; }
  .tbl table { margin: 0; font-variant-numeric: tabular-nums; }
  .tbl th { text-transform: none; letter-spacing: 0; font-size: .82rem; }
  .tbl th, .tbl td { white-space: nowrap; }
  .tbl th.n, .tbl td.n { text-align: right; }
  .tbl td.pl { width: 99%; }
  .empty { color: var(--ink-muted); padding: .9rem 1rem; border: 1px dashed var(--hairline); border-radius: 10px; margin: .3rem 0 0; }

  @media (max-width: 560px) {
    .prizes, ul.criteria { grid-template-columns: 1fr; }
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
    <a class="prize" href="#strong">${stone('black', 34)}<span class="cat" id="pS"></span><span class="amounts" id="pSa"></span></a>
    <a class="prize" href="#videos">${stone('white', 34)}<span class="cat" id="pV"></span><span class="amounts" id="pVa"></span></a>
  </div>
  <p class="deadline"><span id="deadline"></span> <strong id="daysLeft"></strong></p>
</header>

<section id="strong">
  <h2>${stone('black', 26)}<span id="strengthH"></span></h2>
  <p class="lead" id="strengthLead"></p>
  <p id="strengthWho"></p>
  <h3 id="sCritH"></h3>
  <ul class="criteria"><li id="sCrit1"></li><li id="sCrit2"></li><li id="sCrit3"></li></ul>
  <h3 id="sHowH"></h3>
  <ol class="steps"><li id="sStep1"></li><li id="sStep2"></li><li id="sStep3"></li></ol>
  <p class="muted" id="sAnon"></p>
</section>

<section id="videos">
  <h2>${stone('white', 26)}<span id="videoH"></span></h2>
  <p class="lead" id="videoLead"></p>
  <p id="videoWhat"></p>
  <p id="vLang"></p>
  <h3 id="vCritH"></h3>
  <ul class="criteria"><li id="vCrit1"></li><li id="vCrit2"></li><li id="vCrit3"></li></ul>
  <p id="vMin"></p>
  <h3 id="vHowH"></h3>
  <ol class="steps"><li id="vStep1"></li><li id="vStep2"></li><li id="vStep3"></li></ol>
  <p class="muted" id="vLink"></p>
</section>

<section>
  <h2 id="judgingH"></h2>
  <div class="judging">
    <p id="judging"></p>
    <p id="contact"></p>
  </div>
</section>

<section>
  <h2 id="entriesH"></h2>
  <p class="muted" id="updated"></p>
  <h3>${stone('black', 18)}<span id="strengthH2"></span></h3>
  <div class="tbl" id="strengthList"></div>
  <h3>${stone('white', 18)}<span id="videoH2"></span></h3>
  <div class="tbl" id="videoList"></div>
</section>
<script>
(function () {
  var STR = ${strJson};
  var VIEW = ${json};
  var DISCORD = ${JSON.stringify(DISCORD)};
  var DISCORD_NAME = ${JSON.stringify(DISCORD_NAME)};
  var DEADLINE = Date.parse(${JSON.stringify(DEADLINE_ISO)});
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
  // A discrete federation grade ("5d", "3k", "1p") in the reader's language.
  function fmtGrade(label) {
    var m = /^(\\d{1,2})([dkp])$/.exec(label || '');
    if (!m) return label || null;
    return t(m[2] === 'd' ? 'dan' : m[2] === 'k' ? 'kyu' : 'pro', { n: m[1] });
  }

  ['intro', 'deadline', 'strengthH', 'strengthWho',
   'sCrit1', 'sCrit2', 'sCrit3', 'sStep1', 'sStep2', 'sStep3', 'sAnon', 'videoH', 'videoLead',
   'videoWhat', 'vLang', 'vCrit1', 'vCrit2', 'vCrit3', 'vMin', 'vStep1', 'vStep2', 'vStep3', 'vLink',
   'judgingH', 'judging', 'entriesH'].forEach(function (k) { set(k); });
  set('h1', 'title');
  set('pS', 'strengthH'); set('pV', 'videoH');
  set('strengthH2', 'strengthH'); set('videoH2', 'videoH');
  set('sCritH', 'criteriaH'); set('vCritH', 'criteriaH');
  set('sHowH', 'howH'); set('vHowH', 'howH');
  var amounts = esc(t('first')) + ' <b>100 €</b>&emsp;' + esc(t('second')) + ' <b>50 €</b>';
  $('pSa').innerHTML = amounts; $('pVa').innerHTML = amounts;
  var days = Math.ceil((DEADLINE - Date.now()) / 86400000);
  $('daysLeft').textContent = days > 0 ? t('daysLeft', { n: days }) : t('closed');
  // The lead's bracketed phrase links to the community statistics.
  $('strengthLead').innerHTML = esc(t('strengthLead'))
    .replace(/\\[(.+?)\\]/, '<a class="inline" href="/statistics">$1</a>');
  $('contact').innerHTML = esc(t('contact'))
    .replace('{discord}', '<a href="' + DISCORD + '" target="_blank" rel="noopener">Discord</a>')
    .replace('{name}', '<strong>' + esc(DISCORD_NAME) + '</strong>');

  if (!VIEW) { $('updated').textContent = t('unavailable'); return; }

  // The organiser's current ranks, inlined into the entry criterion.
  if (VIEW.organiser) {
    var o = VIEW.organiser;
    $('strengthWho').innerHTML = esc(t('strengthWhoOrg'))
      .replace('{egf}', '<strong>' + esc(fmtRank(o.egf)) + '</strong>')
      .replace('{gor}', esc(o.gor === null ? '—' : String(o.gor)))
      .replace('{om}', '<strong>' + esc(fmtRank(o.om)) + '</strong>');
  }

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

  $('strengthList').innerHTML = table(
    ['colSubmitted', 'colPlayer', 'colIrl', 'colAvg', 'colProblems'],
    VIEW.strength.map(function (r) {
      var irl = r.irl ? r.irl.fed + ' ' + fmtGrade(r.irl.label) : null;
      return '<tr><td>' + esc(r.date) + '</td>' + playerCell(r) +
        '<td class="n">' + val(irl) + '</td><td class="n">' + val(fmtRank(r.avg)) +
        '</td><td class="n">' + val(r.problems) + '</td></tr>';
    }), 'emptyStrength');
  $('videoList').innerHTML = table(
    ['colSubmitted', 'colPlayer', 'colVideo', 'colViews', 'colQuality', 'colPeak', 'colProblems'],
    VIEW.video.map(function (r) {
      var link = r.url ? '<a href="' + esc(r.url) + '" target="_blank" rel="noopener">' + esc(t('watch')) + '</a>' : val(null);
      return '<tr><td>' + esc(r.date) + '</td>' + playerCell(r) +
        '<td class="n">' + link + '</td><td class="n">' + val(r.views) + '</td><td class="n">' +
        val(r.quality === null ? null : r.quality + ' / 10') + '</td><td class="n">' + val(fmtRank(r.peak)) +
        '</td><td class="n">' + val(r.problems) + '</td></tr>';
    }), 'emptyVideo');
})();
</script>
</body>
</html>`;
}
