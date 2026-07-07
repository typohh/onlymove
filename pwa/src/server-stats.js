import { interpExtrap, rankScalarToLabel, RANK_INDEX_OFFSET } from '../shared/rank-math.js';
// M (rating→rank curve) is inlined at build time — same source the client bundles
// (pwa/tsumegos/rating_to_rank.json) — so the Community rank histogram needs no
// asset fetch for it. Other tsumego assets (lookup.txt, metadata.json) stay served.
import RATING_RANK_MAP from '../tsumegos/rating_to_rank.json';
import { APP_VERSION } from './version.js';
import { CORS_HEADERS, jsonResponse } from './http.js';

const SEVEN_DAYS_MS = 7 * 24 * 3600 * 1000;

// Per-isolate recompute dedup: right after a deploy every Community-tab open
// sees a version-stale blob, and without this each concurrent request would run
// its own full 7-day window scan (the D1 read cost the created_at index exists
// to bound) until the first one commits. One in-flight compute per isolate;
// concurrent callers await the same promise.
let computeInFlight = null;
function computeServerStatsOnce(env) {
  if (!computeInFlight) {
    computeInFlight = computeServerStats(env).finally(() => { computeInFlight = null; });
  }
  return computeInFlight;
}

// Serve the cached server-wide stats. Lazily (re)computes when nothing is
// cached yet OR the cache was produced by an older code version (so a deploy
// that changes the computation refreshes on the next read, not 24h later); the
// cron keeps it fresh by age thereafter.
export async function handleServerStats(env) {
  try {
    if (!env.DB) return jsonResponse({ error: 'Database not configured' }, 500);
    const row = await env.DB.prepare(`SELECT data FROM server_stats WHERE key = 'latest'`).first();
    let stale = true;
    if (row && row.data) {
      try { stale = JSON.parse(row.data).version !== APP_VERSION; } catch { stale = true; }
    }
    const body = stale ? JSON.stringify(await computeServerStatsOnce(env)) : row.data;
    return new Response(body, {
      status: 200,
      // The worker already serves from a D1 cache and manages freshness (daily
      // cron + version invalidation); an edge/browser HTTP cache on top would
      // serve stale data and bypass that logic, so opt out of caching here.
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('Error serving server stats:', err);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}

// Recompute when the cache is older than ~23h (the cron is hourly, so this fires
// once a day with slack for tick jitter) OR when it was produced by an older
// code version (so a logic change is picked up on the next tick).
export async function maybeComputeServerStats(env) {
  if (!env.DB) return;
  try {
    const row = await env.DB.prepare(`SELECT data, computed_at FROM server_stats WHERE key = 'latest'`).first();
    let staleVersion = true;
    if (row && row.data) {
      try { staleVersion = JSON.parse(row.data).version !== APP_VERSION; } catch { staleVersion = true; }
    }
    const last = row?.computed_at ? Date.parse(row.computed_at) : 0;
    if (staleVersion || Date.now() - last >= 23 * 3600 * 1000) {
      await computeServerStatsOnce(env);
    }
  } catch (err) {
    console.error('Error computing server stats:', err);
  }
}


// Aggregate the preceding 7 days into the cached JSON blob the PWA renders.
// Everything is a cheap GROUP BY; time histograms cap at 600s (5s buckets) so
// idle-tab outliers don't blow out the distribution.
async function computeServerStats(env) {
  const since = new Date(Date.now() - SEVEN_DAYS_MS).toISOString();

  // ONE pass over the last 7 days, aggregated in JS. idx_tsumego_reports_created
  // turns `created_at >= ?` into a range scan (reads only the window's rows, not
  // the whole table), and sharing a single SELECT across every metric keeps D1
  // rows-read to roughly one window's worth per recompute.
  const rows = (await env.DB.prepare(
    `SELECT user_id, problem, success, size, time_used, review_time, is_repeat, created_at, substr(created_at, 1, 10) AS day
     FROM tsumego_reports WHERE created_at >= ?`
  ).bind(since).all()).results || [];

  const usersAll = new Set();
  let attemptsAll = 0, winsAll = 0;
  const byDay = new Map();     // day → { attempts, users:Set }
  const bySizeMap = new Map(); // size → { attempts, wins, users:Set }
  // Solving/review time is summarised PER USER (one sample per user) rather than
  // per attempt, so heavy users don't dominate the distribution. Solving time is
  // per-attempt, so we keep every sample. Review time is only meaningful on
  // REPEAT (review) attempts, so we gate on is_repeat. It is also CUMULATIVE
  // across a problem's repeats (each repeat re-reports the running total), so per
  // (user, problem) we keep only the last instance — the final cumulative value —
  // and earlier repeats are dropped to avoid massive double counting.
  const userSolve = new Map();   // user → number[] of time_used (ms)
  const userReview = new Map();  // user → Map<problem, { t, at }> last cumulative review_time
  const perUserActivity = new Map(); // user → { attempts }
  // Recall time: how soon a user comes back to a problem they FAILED. Collect
  // each (user, problem)'s attempt timeline; below we measure the gap from a
  // failure to the very next attempt on that problem (the "first recall after a
  // failure"). Recalls following a SUCCESS are excluded — their interval is set
  // by the spaced-repetition schedule (it lengthens with each success), so it
  // isn't comparable across users; restricting to post-failure recalls
  // standardizes the measure.
  const userProblemSeq = new Map(); // user → Map<problem, [{ t, success }]>
  // Answer-derived rank: per user, the set of DISTINCT first-time problems they
  // attempted (with board size). Problems are served difficulty-matched (~80%
  // clear) near the user's rank, so the mean internal rank of these problems
  // approximates the user's own rank — coarse but enough for the histogram.
  // Repeats (recall) are excluded: they're re-shown by the spaced-repetition
  // schedule, not freshly difficulty-matched. neededProblems is the union we
  // must resolve to a rating below.
  const userFirstProblems = new Map(); // user → Map<problem, size>
  const neededProblems = new Set();    // every distinct first-time problem id

  for (const r of rows) {
    attemptsAll++;
    if (r.success) winsAll++;
    usersAll.add(r.user_id);

    let d = byDay.get(r.day);
    if (!d) { d = { attempts: 0, users: new Set() }; byDay.set(r.day, d); }
    d.attempts++; d.users.add(r.user_id);

    let s = bySizeMap.get(r.size);
    if (!s) { s = { attempts: 0, wins: 0, users: new Set() }; bySizeMap.set(r.size, s); }
    s.attempts++; if (r.success) s.wins++; s.users.add(r.user_id);

    if (r.time_used != null && r.time_used > 0) {
      let arr = userSolve.get(r.user_id);
      if (!arr) { arr = []; userSolve.set(r.user_id, arr); }
      arr.push(r.time_used);
    }
    if (r.is_repeat && r.review_time != null && r.review_time > 0) {
      let m = userReview.get(r.user_id);
      if (!m) { m = new Map(); userReview.set(r.user_id, m); }
      const prev = m.get(r.problem);
      if (!prev || r.created_at > prev.at) m.set(r.problem, { t: r.review_time, at: r.created_at });
    }

    let u = perUserActivity.get(r.user_id);
    if (!u) { u = { attempts: 0 }; perUserActivity.set(r.user_id, u); }
    u.attempts++;

    const ts = Date.parse(r.created_at);
    if (Number.isFinite(ts)) {
      let pm = userProblemSeq.get(r.user_id);
      if (!pm) { pm = new Map(); userProblemSeq.set(r.user_id, pm); }
      let seq = pm.get(r.problem);
      if (!seq) { seq = []; pm.set(r.problem, seq); }
      seq.push({ t: ts, success: !!r.success, repeat: !!r.is_repeat });
    }

    if (!r.is_repeat) {
      let pm = userFirstProblems.get(r.user_id);
      if (!pm) { pm = new Map(); userFirstProblems.set(r.user_id, pm); }
      if (!pm.has(r.problem)) pm.set(r.problem, r.size);
      neededProblems.add(r.problem);
    }
  }

  // Per-user median time → fine bucket; the histogram then counts USERS per
  // bucket, so binHist (client) reads it with "users" as the y-axis. Solving
  // time is log-normal and most of the mass sits under ~30s, so a coarse 10s
  // bucket lumped the whole hump into one bar; SOLVE_BUCKET_SEC=5 keeps the
  // low-end detail (bucketSec ships in the blob so the client can decode it,
  // and any future width change stays self-describing). Capped at 600s.
  const SOLVE_BUCKET_SEC = 5;
  const SOLVE_BUCKET_CAP = Math.round(600 / SOLVE_BUCKET_SEC);
  const median = (arr) => {
    if (!arr.length) return null;
    const s = arr.slice().sort((a, b) => a - b);
    const mid = s.length >> 1;
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  };
  const timeBucket = (ms) => Math.min(Math.floor(ms / (SOLVE_BUCKET_SEC * 1000)), SOLVE_BUCKET_CAP);
  const solveB = new Map();    // 10s bucket → user count
  const reviewB = new Map();
  for (const arr of userSolve.values()) {
    const med = median(arr);
    if (med == null) continue;
    const b = timeBucket(med);
    solveB.set(b, (solveB.get(b) || 0) + 1);
  }
  for (const perProblem of userReview.values()) {
    const med = median([...perProblem.values()].map(v => v.t));
    if (med == null) continue;
    const b = timeBucket(med);
    reviewB.set(b, (reviewB.get(b) || 0) + 1);
  }

  // Recall time, one sample per user: the MEDIAN novel→first-repeat gap across
  // their problems. Per problem, the gap runs from the NOVEL (first non-repeat)
  // attempt to the very next attempt on that problem, and only counts when the
  // novel attempt FAILED (see userProblemSeq note — post-success intervals are
  // set by the spaced-repetition schedule, not comparable). Pre-binned
  // server-side (the spans run minutes to days, so the fine solving/review
  // buckets don't apply) and rendered with makeBarList like
  // attemptsPerUserDist. Gaps under 12h are ignored entirely (a same-session
  // retry, not a meaningful recall interval). The 7-day window right-censors
  // long gaps (a novel attempt whose first repeat falls outside the window
  // contributes nothing), which biases the median slightly short — accepted.
  const RECALL_MIN_MS = 12 * 3600e3;
  const RECALL_BINS = [
    { hi: 24 * 3600e3, label: '<1d' },
    { hi: 48 * 3600e3, label: '1–2d' },
    { hi: 72 * 3600e3, label: '2–3d' },
    { hi: 96 * 3600e3, label: '3–4d' },
    { hi: 120 * 3600e3, label: '4–5d' },
    { hi: Infinity, label: '5d+' },
  ];
  const recallDist = RECALL_BINS.map(b => ({ label: b.label, value: 0 }));
  for (const pm of userProblemSeq.values()) {
    const gaps = [];
    for (const seq of pm.values()) {
      if (seq.length < 2) continue;
      seq.sort((a, b) => a.t - b.t);
      // The novel attempt must itself be in-window; a leading repeat means the
      // novel attempt predates the window, so that problem contributes nothing.
      const ni = seq.findIndex(e => !e.repeat);
      if (ni < 0 || ni + 1 >= seq.length) continue;
      if (seq[ni].success) continue;              // recall after a success — skip
      const gap = seq[ni + 1].t - seq[ni].t;
      if (gap >= RECALL_MIN_MS) gaps.push(gap);
    }
    const med = median(gaps);
    if (med != null) {
      const bi = RECALL_BINS.findIndex(b => med < b.hi);
      if (bi >= 0) recallDist[bi].value += 1;
    }
  }

  const totals = { attempts: attemptsAll, wins: winsAll, users: usersAll.size };
  const daily = [...byDay.entries()]
    .map(([day, v]) => ({ day, attempts: v.attempts, users: v.users.size }))
    .sort((a, b) => (a.day < b.day ? -1 : 1));
  const bySize = [...bySizeMap.entries()]
    .map(([size, v]) => ({ size, attempts: v.attempts, wins: v.wins, users: v.users.size }));
  const solveHist = [...solveB.entries()].map(([bucket, n]) => ({ bucket, n })).sort((a, b) => a.bucket - b.bucket);
  const reviewHist = [...reviewB.entries()].map(([bucket, n]) => ({ bucket, n })).sort((a, b) => a.bucket - b.bucket);

  // Per-user attempts-per-day, binned: "how many users average ~2/day, ~5/day,
  // …". One sample per user — the week's attempts averaged over the full 7-day
  // window (attempts ÷ 7, NOT ÷ distinct active days), so a user who burst-solves
  // on one day reads as their true daily average, not their per-session count.
  // The lowest bin is sub-1 so low-activity users (< 7 attempts/week) still count.
  const PER_USER_BINS = [[0, 1], [1, 2], [2, 5], [5, 10], [10, 20], [20, 50], [50, Infinity]];
  const PER_USER_LABELS = ['<1', '1–2', '2–5', '5–10', '10–20', '20–50', '50+'];
  const attemptsPerUserDist = PER_USER_LABELS.map(label => ({ label, value: 0 }));
  for (const u of perUserActivity.values()) {
    const rate = u.attempts / 7;
    const i = PER_USER_BINS.findIndex(([lo, hi]) => rate >= lo && rate < hi);
    if (i >= 0) attemptsPerUserDist[i].value += 1;
  }

  // Answer-derived rank histogram (EGF scale, weakest→strongest). Each active
  // user's rank is estimated from the last 7 days of first-time attempts — the
  // mean internal rank of the difficulty-matched problems they were served —
  // not from self-reported server ranks. Users with too little data are dropped.
  const ranks = await computeAnswerRankHist(env, userFirstProblems, neededProblems);

  const data = {
    // Stamp the code version so a deploy that changes this computation
    // invalidates the cache on the next read/cron, even within the 24h window.
    version: APP_VERSION,
    computedAt: new Date().toISOString(),
    windowDays: 7,
    totals,
    bucketSec: SOLVE_BUCKET_SEC,
    daily, attemptsPerUserDist, bySize, solveHist, reviewHist, recallDist, ranks,
  };

  await env.DB.prepare(
    `INSERT INTO server_stats (key, data, computed_at) VALUES ('latest', ?, ?)
     ON CONFLICT(key) DO UPDATE SET data = excluded.data, computed_at = excluded.computed_at`
  ).bind(JSON.stringify(data), data.computedAt).run();

  return data;
}

// Minimum distinct first-time problems a user needs in the window before we
// trust their answer-derived rank. Below this we leave the user out entirely.
const MIN_RANK_PROBLEMS = 10;

function fetchTsumegoAsset(env, path) {
  return env.ASSETS
    .fetch(new Request('https://onlymove.net' + path))
    .then(res => (res.ok ? res : null))
    .catch(e => { console.error('asset fetch failed:', path, e); return null; });
}

// Build the Community "Ranks" histogram from answers. Each problem's coarse
// rating is the midpoint of the shard it lives in (lookup.txt resolves a
// problem id → shard hash; the per-size metadata.json gives that shard's
// min/max rating); that rating maps to an internal rank index via M
// (rating_to_rank.json). A user's rank is the mean index over their first-time
// problems — sound because selection serves problems near the user's rank.
// Returns [{ rank: label, n }] ordered weakest→strongest; [] if assets missing.
async function computeAnswerRankHist(env, userFirstProblems, neededProblems) {
  if (!neededProblems.size) return [];

  // 1. problem id → shard hash, parsed straight off lookup.txt ("id hash size"
  //    per line). Keep only the ids we need to avoid a 143k-entry map.
  const lookupRes = await fetchTsumegoAsset(env, '/lookup.txt');
  if (!lookupRes) return [];
  const text = await lookupRes.text();
  const idToShard = new Map();
  for (let start = 0; start < text.length;) {
    let nl = text.indexOf('\n', start);
    if (nl < 0) nl = text.length;
    const sp = text.indexOf(' ', start);
    if (sp > start && sp < nl) {
      const id = text.slice(start, sp);
      if (neededProblems.has(id)) {
        const rest = text.slice(sp + 1, nl).trim();
        idToShard.set(id, rest.split(/\s+/)[0]);
      }
    }
    start = nl + 1;
  }

  // 2. shard hash → midpoint raw rating (board-independent scale, so a single
  //    map across both sizes is fine).
  const shardRating = new Map();
  for (const size of [9, 13]) {
    const res = await fetchTsumegoAsset(env, `/tsumegos/${size}/metadata.json`);
    if (!res) continue;
    const shards = await res.json();
    for (const s of shards) {
      const hash = String(s.filename).split('.')[0];
      shardRating.set(hash, (s.min_rating + s.max_rating) / 2);
    }
  }

  // 3. M curve (raw rating → internal rank index), per board size — inlined at
  //    build time (see the import at the top), so no asset fetch is needed.
  const curves = RATING_RANK_MAP;
  const rankIndexOf = (rating, size) => {
    const c = curves[String(size)] || curves['9'];
    if (!c || !Array.isArray(c.ratings) || !Array.isArray(c.rank_indices)) return null;
    return interpExtrap(rating, c.ratings, c.rank_indices);
  };

  // 4. Per user: mean internal-rank index → EGF scalar (index 19 == 1k == 0).
  const rankHist = new Map();
  for (const pm of userFirstProblems.values()) {
    let sum = 0, n = 0;
    for (const [pid, size] of pm) {
      const hash = idToShard.get(pid);
      if (!hash) continue;
      const rating = shardRating.get(hash);
      if (rating == null) continue;
      const idx = rankIndexOf(rating, size);
      if (idx == null || !Number.isFinite(idx)) continue;
      sum += idx; n++;
    }
    if (n < MIN_RANK_PROBLEMS) continue;
    const scalar = Math.max(-29, Math.min(9, Math.round(sum / n) - RANK_INDEX_OFFSET));
    rankHist.set(scalar, (rankHist.get(scalar) || 0) + 1);
  }
  return [...rankHist.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([v, n]) => ({ rank: rankScalarToLabel(v), n }));
}
