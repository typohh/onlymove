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

import { loadContest } from '../contest.js';
import { rankProgress } from '../../shared/rank-math.js';
import { composeAliasAll } from '../../shared/alias-words.js';
import { baseCss, flag, ESC_SNIPPET } from './page-shell.js';

const DISCORD = 'https://discord.gg/ZU7jJNnkbB';
const DISCORD_NAME = 'ootakamoku';

const STR = {
    en: {
        title: 'Contribution rewards 2026',
        intro: 'OnlyMove is looking for two kinds of help, and pays for the best of each.',
        deadline: 'Deadline: 31 December 2026, 23:59 UTC.',
        prizes: 'In each category: 1st place 100 €, 2nd place 50 €.',
        strengthH: 'Strong players',
        strengthWhy: 'Hundreds of players are active on OnlyMove, but very few are stronger than the organiser, so the top of the rank scale rests on thin data. Verified strong players solving many problems near their best are what calibrates it.',
        strengthWho: 'Open to players with an EGF or AGA rank who are stronger than the organiser. New and existing accounts are both welcome.',
        organiserNow: 'The organiser currently stands at EGF {egf} (GoR {gor}) and OnlyMove {om}; both figures update with the lists below.',
        strengthCriteria: 'Judged with roughly equal weight on your EGF/AGA rank, your account\'s average OnlyMove rank over its whole history, and the number of problems the account has solved. Earlier entries are favoured, counted from the day they appear on the list below.',
        strengthHow: 'To enter, link your EGD or AGA record in the app (menu → Account), then send the organiser your name or your EGD PIN / AGA id. The organiser verifies your identity through other Go players. To stay anonymous on this page, hide your profile in the same Account dialog: you are listed as Anonymous, and your rank is still shown.',
        videoH: 'Videos',
        videoWhy: 'Videos introduce OnlyMove to new players and show how a strong player actually reads a problem.',
        videoWhat: 'Record yourself solving OnlyMove problems in a single session, thinking aloud and explaining how you solve each one. Start a new account at the beginning of the video and climb as high as you can. English, Chinese, Japanese or Korean.',
        videoCriteria: 'Judged with roughly equal weight on views, the quality of the explanation (scored by the organiser; the score may be revised before the deadline), and the highest OnlyMove rank reached during the video. Earlier entries are slightly favoured. Minimum 50 views.',
        videoHow: 'To enter, pick an alias for the account you played on (menu → Account), then send the organiser the video link, the alias, and how many problems you solved in the video. With your agreement, the organiser may link your video from the app for players of the matching strength.',
        judgingH: 'Judging',
        judging: 'Both categories are judged by Henry Hemming, the author of OnlyMove, on the criteria above. Entries the organiser considers dishonest are disqualified.',
        contact: 'Send entries and questions to {name} on {discord}.',
        entriesH: 'Entries',
        updated: 'Updated {ago}. The lists refresh every 12 hours.',
        agoJustNow: 'just now', agoMinutes: '{n} min ago', agoHours: '{n} h ago', agoDays: '{n} d ago',
        colSubmitted: 'Entered', colPlayer: 'Player', colIrl: 'EGF / AGA', colAvg: 'Avg OnlyMove',
        colProblems: 'Problems', colVideo: 'Video', colViews: 'Views', colQuality: 'Quality', colPeak: 'Peak OnlyMove',
        watch: 'Watch', anonymous: 'Anonymous', empty: 'No entries yet.',
        unavailable: 'The entry lists could not be loaded. Please try again later.',
    },
    zh: {
        title: '贡献奖励 2026',
        intro: 'OnlyMove 正在寻求两类帮助，并为每类中最出色的贡献提供奖金。',
        deadline: '截止时间：2026 年 12 月 31 日 23:59（UTC）。',
        prizes: '每个类别：第一名 100 欧元，第二名 50 欧元。',
        strengthH: '强手棋手',
        strengthWhy: 'OnlyMove 有数百名活跃用户，但比组织者更强的棋手寥寥无几，因此段位刻度的顶端数据很少。经过验证的强手在接近自身最佳水平时大量解题，才能校准这一段。',
        strengthWho: '面向拥有 EGF 或 AGA 段位、且棋力强于组织者的棋手。新账户和已有账户均可参加。',
        organiserNow: '组织者目前为 EGF {egf}（GoR {gor}），OnlyMove {om}；两项数值随下方列表一同更新。',
        strengthCriteria: '评判时大致同等考虑：你的 EGF/AGA 段位、账户全部历史的 OnlyMove 平均段位，以及账户解答的题目数量。较早参加者优先，以出现在下方列表中的日期为准。',
        strengthHow: '参加方法：在应用中关联你的 EGD 或 AGA 记录（菜单 → 账户），然后将你的姓名或 EGD PIN / AGA ID 发送给组织者。组织者会通过其他围棋棋手核实你的身份。如希望在本页匿名，可在同一账户对话框中隐藏个人资料：你将以“匿名”列出，段位仍会显示。',
        videoH: '视频',
        videoWhy: '视频能向新棋手介绍 OnlyMove，并展示强手如何真正地计算一道题。',
        videoWhat: '录制你在一次连续的时段内解答 OnlyMove 题目的过程，边想边说，讲解每道题的解法。在视频开始时注册一个新账户，并尽可能攀升到更高的段位。可使用英语、中文、日语或韩语。',
        videoCriteria: '评判时大致同等考虑：观看次数、讲解质量（由组织者评分，截止前可能调整），以及视频中达到的最高 OnlyMove 段位。较早参加者略有优先。最低要求 50 次观看。',
        videoHow: '参加方法：为你所用的账户选择一个别名（菜单 → 账户），然后将视频链接、别名以及视频中解答的题目数量发送给组织者。经你同意，组织者可能会在应用中向相应棋力的棋手推荐你的视频。',
        judgingH: '评判',
        judging: '两个类别均由 OnlyMove 作者 Henry Hemming 依据上述标准评判。组织者认为不诚实的参赛将被取消资格。',
        contact: '请在 {discord} 上联系 {name}，提交参赛和咨询问题。',
        entriesH: '参赛列表',
        updated: '更新于 {ago}。列表每 12 小时刷新一次。',
        agoJustNow: '刚刚', agoMinutes: '{n} 分钟前', agoHours: '{n} 小时前', agoDays: '{n} 天前',
        colSubmitted: '参加日期', colPlayer: '棋手', colIrl: 'EGF / AGA', colAvg: 'OnlyMove 平均',
        colProblems: '题目数', colVideo: '视频', colViews: '观看', colQuality: '质量', colPeak: 'OnlyMove 最高',
        watch: '观看', anonymous: '匿名', empty: '暂无参赛。',
        unavailable: '无法加载参赛列表，请稍后再试。',
    },
    ja: {
        title: '貢献者リワード 2026',
        intro: 'OnlyMove では 2 種類の協力を募集しており、それぞれ最も優れたものに賞金を用意しています。',
        deadline: '締め切り：2026 年 12 月 31 日 23:59（UTC）。',
        prizes: '各部門：1 位 100 ユーロ、2 位 50 ユーロ。',
        strengthH: '強豪プレイヤー',
        strengthWhy: 'OnlyMove には数百人のアクティブユーザーがいますが、主催者より強いプレイヤーはごくわずかで、ランクの上位はデータが薄い状態です。実力が確認された強豪が自己ベスト付近で数多く解くことが、この範囲の較正につながります。',
        strengthWho: 'EGF または AGA の段級位を持ち、主催者より強い方が対象です。新規アカウントでも既存アカウントでも参加できます。',
        organiserNow: '主催者の現在の値は EGF {egf}（GoR {gor}）、OnlyMove {om} です。どちらも下の一覧と一緒に更新されます。',
        strengthCriteria: '審査では、EGF/AGA の段級位、アカウント全履歴での OnlyMove 平均ランク、アカウントが解いた問題数をほぼ同じ重みで考慮します。早い参加ほど有利で、下の一覧に掲載された日を基準とします。',
        strengthHow: '参加方法：アプリで EGD または AGA の記録を連携し（メニュー → アカウント）、氏名または EGD PIN / AGA ID を主催者に送ってください。主催者は他の碁打ちを通じて本人確認を行います。このページで匿名にしたい場合は、同じアカウント画面でプロフィールを非公開にしてください。「匿名」として掲載され、段級位は表示されます。',
        videoH: '動画',
        videoWhy: '動画は OnlyMove を新しいプレイヤーに紹介し、強い人が実際にどう問題を読むのかを伝えます。',
        videoWhat: '1 回のセッションで OnlyMove の問題を解く様子を、考えていることを声に出し、解き方を説明しながら録画してください。動画の冒頭で新しいアカウントを作り、できるだけ高いランクまで上がってください。英語・中国語・日本語・韓国語のいずれでも構いません。',
        videoCriteria: '審査では、再生回数、解説の質（主催者が採点し、締め切りまでに見直すことがあります）、動画中に到達した最高 OnlyMove ランクをほぼ同じ重みで考慮します。早い参加がやや有利です。最低 50 回の再生が必要です。',
        videoHow: '参加方法：プレイしたアカウントで別名を選び（メニュー → アカウント）、動画のリンク、別名、動画内で解いた問題数を主催者に送ってください。了承をいただければ、該当する棋力のプレイヤー向けにアプリから動画を紹介することがあります。',
        judgingH: '審査',
        judging: '両部門とも、OnlyMove の作者 Henry Hemming が上記の基準で審査します。不正と判断した応募は失格とします。',
        contact: '応募と質問は {discord} で {name} までお送りください。',
        entriesH: '応募一覧',
        updated: '更新 {ago}。一覧は 12 時間ごとに更新されます。',
        agoJustNow: 'たった今', agoMinutes: '{n} 分前', agoHours: '{n} 時間前', agoDays: '{n} 日前',
        colSubmitted: '応募日', colPlayer: 'プレイヤー', colIrl: 'EGF / AGA', colAvg: 'OnlyMove 平均',
        colProblems: '問題数', colVideo: '動画', colViews: '再生', colQuality: '質', colPeak: 'OnlyMove 最高',
        watch: '見る', anonymous: '匿名', empty: 'まだ応募はありません。',
        unavailable: '応募一覧を読み込めませんでした。時間をおいて再度お試しください。',
    },
    ko: {
        title: '기여 보상 2026',
        intro: 'OnlyMove는 두 가지 도움을 찾고 있으며, 각 부문에서 가장 뛰어난 기여에 상금을 드립니다.',
        deadline: '마감: 2026년 12월 31일 23:59 (UTC).',
        prizes: '각 부문: 1위 100유로, 2위 50유로.',
        strengthH: '강자 부문',
        strengthWhy: 'OnlyMove에는 수백 명의 활성 사용자가 있지만 주최자보다 강한 기사는 매우 적어서, 랭크 척도의 최상단은 데이터가 부족합니다. 실력이 검증된 강자가 자신의 최고 수준 근처에서 많은 문제를 푸는 것이 이 구간을 보정합니다.',
        strengthWho: 'EGF 또는 AGA 급수가 있고 주최자보다 강한 기사가 대상입니다. 새 계정과 기존 계정 모두 참가할 수 있습니다.',
        organiserNow: '주최자의 현재 수치는 EGF {egf}(GoR {gor}), OnlyMove {om}이며, 두 수치 모두 아래 목록과 함께 갱신됩니다.',
        strengthCriteria: '심사는 EGF/AGA 급수, 계정 전체 기록에 걸친 OnlyMove 평균 랭크, 계정이 푼 문제 수를 대략 같은 비중으로 고려합니다. 먼저 참가할수록 유리하며, 아래 목록에 오른 날을 기준으로 합니다.',
        strengthHow: '참가 방법: 앱에서 EGD 또는 AGA 기록을 연결한 뒤(메뉴 → 계정), 이름 또는 EGD PIN / AGA ID를 주최자에게 보내 주세요. 주최자는 다른 바둑 기사들을 통해 신원을 확인합니다. 이 페이지에서 익명으로 남고 싶다면 같은 계정 화면에서 프로필을 숨기세요. "익명"으로 표시되며 급수는 그대로 표시됩니다.',
        videoH: '영상 부문',
        videoWhy: '영상은 새로운 기사들에게 OnlyMove를 소개하고, 강자가 실제로 문제를 어떻게 읽는지 보여 줍니다.',
        videoWhat: '한 번의 세션에서 OnlyMove 문제를 푸는 모습을, 생각을 소리 내어 말하고 풀이를 설명하면서 녹화해 주세요. 영상 시작 시 새 계정을 만들고 가능한 한 높은 랭크까지 올라가 보세요. 영어, 중국어, 일본어, 한국어 중 어느 언어든 좋습니다.',
        videoCriteria: '심사는 조회수, 해설의 질(주최자가 채점하며 마감 전에 조정될 수 있음), 영상 중 도달한 최고 OnlyMove 랭크를 대략 같은 비중으로 고려합니다. 먼저 참가할수록 약간 유리합니다. 최소 조회수 50회가 필요합니다.',
        videoHow: '참가 방법: 플레이한 계정에서 별명을 고른 뒤(메뉴 → 계정), 영상 링크, 별명, 영상에서 푼 문제 수를 주최자에게 보내 주세요. 동의해 주시면 해당 기력의 기사들을 위해 앱에서 영상을 소개할 수 있습니다.',
        judgingH: '심사',
        judging: '두 부문 모두 OnlyMove 제작자 Henry Hemming이 위 기준에 따라 심사합니다. 주최자가 부정하다고 판단한 참가는 실격 처리됩니다.',
        contact: '참가 신청과 문의는 {discord}에서 {name}에게 보내 주세요.',
        entriesH: '참가 목록',
        updated: '{ago} 업데이트. 목록은 12시간마다 갱신됩니다.',
        agoJustNow: '방금 전', agoMinutes: '{n}분 전', agoHours: '{n}시간 전', agoDays: '{n}일 전',
        colSubmitted: '참가일', colPlayer: '기사', colIrl: 'EGF / AGA', colAvg: 'OnlyMove 평균',
        colProblems: '문제 수', colVideo: '영상', colViews: '조회수', colQuality: '질', colPeak: 'OnlyMove 최고',
        watch: '보기', anonymous: '익명', empty: '아직 참가자가 없습니다.',
        unavailable: '참가 목록을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.',
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

// An OnlyMove rank as the front page shows it: the band label plus the floored
// progress into it (the shared rankProgress decomposition), unclamped.
function rankCell(v) {
    if (!Number.isFinite(v)) return null;
    const p = rankProgress(v);
    return `${p.label} ${p.percent}%`;
}

// The blob, reduced to display strings. The alias is composed in all four
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
            ? { egf: rankCell(org.egfRank), gor: org.gor, om: rankCell(org.onlymoveRank) } : null,
        strength: (data.strength || []).map((e) => ({
            ...who(e),
            irl: e.irl ? `${e.irl.fed} ${e.irl.label}` : null,
            avg: rankCell(e.avgRank),
            problems: Number.isFinite(e.problems) ? e.problems : null,
        })),
        video: (data.video || []).map((e) => ({
            ...who(e),
            url: typeof e.url === 'string' && /^https:\/\//.test(e.url) ? e.url : null,
            views: Number.isFinite(e.views) ? e.views : null,
            quality: Number.isFinite(e.quality) ? e.quality : null,
            peak: rankCell(e.peakRank),
            problems: Number.isFinite(e.problems) ? e.problems : null,
        })),
    };
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
  .prize { font-weight: 600; }
  .tbl { overflow-x: auto; margin: .6rem 0 0; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { padding: .4rem .5rem; text-align: right; white-space: nowrap; }
  th { font-size: .8rem; font-weight: 600; color: var(--ink-muted); border-bottom: 1px solid var(--hairline); }
  td { border-bottom: 1px solid var(--hairline); }
  th.l, td.l { text-align: left; }
  td.pl { width: 99%; }
  .muted { color: var(--ink-muted); }
  h3 { font-size: 1rem; margin: 1.6rem 0 .2rem; }
</style>
</head>
<body>
<h1 id="h1"></h1>
<p id="intro"></p>
<p><span class="prize" id="prizes"></span> <span id="deadline"></span></p>

<h2 id="strengthH"></h2>
<p id="strengthWhy"></p>
<p id="strengthWho"></p>
<p id="strengthCriteria"></p>
<p id="strengthHow"></p>

<h2 id="videoH"></h2>
<p id="videoWhy"></p>
<p id="videoWhat"></p>
<p id="videoCriteria"></p>
<p id="videoHow"></p>

<h2 id="judgingH"></h2>
<p id="judging"></p>
<p id="contact"></p>

<h2 id="entriesH"></h2>
<p class="muted" id="updated"></p>
<h3 id="strengthH2"></h3>
<div class="tbl" id="strengthList"></div>
<h3 id="videoH2"></h3>
<div class="tbl" id="videoList"></div>
<script>
(function () {
  var STR = ${strJson};
  var VIEW = ${json};
  var DISCORD = ${JSON.stringify(DISCORD)};
  var DISCORD_NAME = ${JSON.stringify(DISCORD_NAME)};
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
  document.documentElement.lang = LANG;
  document.title = 'onlymove — ' + t('title');
  ['intro', 'prizes', 'deadline', 'strengthH', 'strengthWhy', 'strengthWho', 'strengthCriteria',
   'strengthHow', 'videoH', 'videoWhy', 'videoWhat', 'videoCriteria', 'videoHow', 'judgingH',
   'judging', 'entriesH'].forEach(function (k) { $(k).textContent = t(k); });
  $('h1').textContent = t('title');
  $('strengthH2').textContent = t('strengthH');
  $('videoH2').textContent = t('videoH');
  $('contact').innerHTML = esc(t('contact')).replace('{discord}', '<a href="' + DISCORD + '" target="_blank" rel="noopener">Discord</a>')
    .replace('{name}', '<strong>' + esc(DISCORD_NAME) + '</strong>');
  if (VIEW && VIEW.organiser) $('strengthWho').textContent += ' ' + t('organiserNow', VIEW.organiser);

  if (!VIEW) { $('updated').textContent = t('unavailable'); return; }
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
    return '<td class="l pl">' + (r.flag ? r.flag + ' ' : '') + label +
      '</td>';
  }
  function val(v) { return v === null || v === undefined ? '<span class="muted">—</span>' : esc(v); }
  function table(heads, rows) {
    if (!rows.length) return '<p class="muted">' + esc(t('empty')) + '</p>';
    return '<table><thead><tr>' + heads.map(function (h, i) {
      return '<th' + (i < 2 ? ' class="l"' : '') + '>' + esc(t(h)) + '</th>';
    }).join('') + '</tr></thead><tbody>' + rows.join('') + '</tbody></table>';
  }

  $('strengthList').innerHTML = table(
    ['colSubmitted', 'colPlayer', 'colIrl', 'colAvg', 'colProblems'],
    VIEW.strength.map(function (r) {
      return '<tr><td class="l">' + esc(r.date) + '</td>' + playerCell(r) +
        '<td>' + val(r.irl) + '</td><td>' + val(r.avg) + '</td><td>' + val(r.problems) + '</td></tr>';
    }));
  $('videoList').innerHTML = table(
    ['colSubmitted', 'colPlayer', 'colVideo', 'colViews', 'colQuality', 'colPeak', 'colProblems'],
    VIEW.video.map(function (r) {
      var link = r.url ? '<a href="' + esc(r.url) + '" target="_blank" rel="noopener">' + esc(t('watch')) + '</a>' : val(null);
      return '<tr><td class="l">' + esc(r.date) + '</td>' + playerCell(r) +
        '<td>' + link + '</td><td>' + val(r.views) + '</td><td>' + val(r.quality === null ? null : r.quality + ' / 10') +
        '</td><td>' + val(r.peak) + '</td><td>' + val(r.problems) + '</td></tr>';
    }));
})();
</script>
</body>
</html>`;
}
