// Offline "displayed change" simulator.
//
// Replays each real user's attempt history (export_sim_data.py → sim_input.json)
// through the REAL client recall model + rank estimator (engine.bundle.mjs) and
// measures how often the post-answer overlay shows a change whose SIGN contradicts
// the answer — the thing a user perceives as "I got it right but my rank went
// DOWN" / "I missed the recall but it says it'll come back LATER".
//
// Two display surfaces (see pwa/js/user-stats.js updateUserStats + overlays.js):
//   - FIRST-TIME (challenge) attempt → "Rank {±x.xxx}" + up/down-to-label.
//        rankChange = userRank_after − userRank_before  (estimateWindowRank refit).
//        Expectation: correct ⇒ ≥0, wrong ⇒ ≤0.
//   - REPEAT (recall) attempt → "{±N} repeat delay".
//        delta = freshFailureReviewHours(after retrain) − (before retrain), the
//        change in a generic problem's time-to-first-repeat from teaching the
//        model this one answer. Expectation: correct ⇒ later (+), wrong ⇒ sooner (−).
//
// The retrain is the genuine warm-started Adam two-head fit run after every
// attempt (model.js calculateAndStoreStatsResults), so the Adam momentum trajectory
// is reproduced exactly, not re-derived.
//
// Usage:
//   node runner.mjs [--input sim_input.json] [--bundle engine.bundle.mjs]
//                   [--out results.json] [--users a,b] [--max-per-user N]

import { readFileSync, writeFileSync } from 'fs';
import vm from 'vm';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../../');

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
const INPUT = arg('input', path.join(HERE, 'sim_input.json'));
const BUNDLE = arg('bundle', path.join(HERE, 'engine.bundle.mjs'));
const OUT = arg('out', path.join(HERE, 'results.json'));
const USERS_FILTER = arg('users', null);
const MAX_PER_USER = parseInt(arg('max-per-user', '0')) || 0;
// Experiment knob: extra one-at-a-time Adam steps to take when the displayed
// recall delay contradicts the answer (0 = shipped behaviour / baseline arm).
const REFINE_STEPS = parseInt(arg('refine-steps', '0')) || 0;
// Report a SECOND frequency cut over attempts on/after this date — the ratings
// were refit 2026-06-22, so only post-cutoff attempts used the rating/slope/floor
// this replay assigns (pre-cutoff attempts ran against now-superseded ratings).
const SINCE = arg('since', '2026-06-22');
const SINCE_MS = Date.parse(SINCE);

const TOTAL_RANKS = 29;
const WORKING_SET = 3000;                 // ANSWER_HISTORY_WORKING_SET in constants.js
// "visible" thresholds = what the overlay rounds to a nonzero number.
const RANK_VISIBLE_STONES = 0.0005;       // overlays: rankChange*29 toFixed(3) ⇒ ≥0.001
const RANK_CLEAR_STONES = 0.01;           // a clearly-perceptible rank move
const RECALL_VISIBLE_S = 0.5;             // overlays: round(|Δh|*3600) ⇒ ≥1 s

// ---- math global (mathjs UMD), needed by model.calculateResultsAndStdErrors ----
vm.runInThisContext(readFileSync(path.join(REPO, 'pwa', 'lib', 'math.min.js'), 'utf8'));

const E = await import(pathToFileURL(BUNDLE).href);
// freshFailureReviewHours saturation rails (model.js MIN/MAX_STABILITY_DAYS):
//   floor = (MIN_RECALL_HOURS/24)/(-ln .8) days → MIN_RECALL_HOURS h ;  ceil = 365 days → 365·24·(-ln .8) h
const RECALL_FLOOR_H = E.MIN_RECALL_HOURS;
const RECALL_CEIL_H = 365 * 24 * -Math.log(0.8);   // ≈ 1954.7 h
E.loadPopulationConstants();              // real boot path: KAPPA/Z0/GROWTH/LN_C_PRIOR + biases

const data = JSON.parse(readFileSync(INPUT, 'utf8'));
const problems = data.problems;
let userIds = Object.keys(data.users);
if (USERS_FILTER) {
  const want = USERS_FILTER.split(',').map(s => s.trim());
  userIds = userIds.filter(u => want.some(w => u.startsWith(w)));
}

// ---------------------------------------------------------------------------
// Per-attempt replay of ONE user — mirrors updateUserStats step for step.
// ---------------------------------------------------------------------------
async function replayUser(uid, attempts) {
  E.state.answerHistory = [];
  E.state.userRank = 0;          // DEFAULT_USER_RANK (≈20k cold start)
  E.state.modelParams = null;
  E.state.adamState = null;
  E.state.statsResults = null;

  const recs = [];
  let n = 0;
  for (const [tid, succ01, size, timeUsed, reviewTimeCum, isRepeat01, dateMs] of attempts) {
    if (MAX_PER_USER && n >= MAX_PER_USER) break;
    n++;
    const correct = succ01 === 1;
    const isRepeat = isRepeat01 === 1;
    E.state.boardSize = size;

    const meta = problems[tid];                      // [rating, slope, floor]
    const rating = meta[0], slope = meta[1], floor = meta[2];
    const problemRank = E.normalizedRankForSize(rating, size);

    const hist = E.state.answerHistory;
    let prev = null;
    for (let k = hist.length - 1; k >= 0; k--) { if (hist[k].problemId === tid) { prev = hist[k]; break; } }

    // DB review_time is the cumulative-prior review total reported WITH this
    // attempt (buildReportPayload sums prior entries' reviewTime), i.e. exactly
    // the client's accumulatedReview for this row. The increment that fills the
    // PRIOR encounter's own reviewTime becomes known now → back-fill it, matching
    // the live timeline (a problem's review is reported on its NEXT encounter).
    const accumReview = reviewTimeCum != null ? reviewTimeCum : 0;
    if (prev) {
      const inc = accumReview - (prev.accumulatedReview || 0);
      if (inc > 0) prev.reviewTime = inc;
    }

    // successSinceFailure — identical branching to updateUserStats.
    let ssf;
    if (isRepeat) {
      if (!correct) ssf = 0;
      else ssf = (prev && prev.successSinceFailure !== undefined) ? prev.successSinceFailure + 1 : 1;
    } else if (!correct) {
      ssf = 0;
    } // else first-time correct ⇒ left undefined

    const entry = {
      problemId: tid,
      problemRating: rating,
      problemRank,
      problemSlope: slope,
      problemFloor: floor,
      timeTaken: timeUsed != null ? timeUsed : 0,
      reviewTime: 0,                  // this attempt's own review not done yet
      accumulatedReview: accumReview,
      correct,
      date: dateMs,
      successSinceFailure: ssf,
      isRepeat,
      size,
    };
    hist.push(entry);
    if (hist.length > WORKING_SET) hist.shift();     // working-set eviction

    // ---- rank refit (updateUserStats: runs for every attempt; repeats are
    //      filtered out of flattenAttempts so they leave userRank unchanged) ----
    const merged = E.getMergedAnswerHistory();        // one copy, reused below
    const oldRank = E.state.userRank;
    const est = E.estimateWindowRank(merged, oldRank * TOTAL_RANKS);
    if (est) E.state.userRank = est.rankIndex / TOTAL_RANKS;
    const rankChange = E.state.userRank - oldRank;   // normalized; ×29 = stones

    // ---- recall delta (repeat only): the REAL shared path, incl. the optional
    //      one-step-at-a-time refine (recallImpactWithRefine in model.js) ----
    let recallDelta = null, recallReady = false, recallBefore = null, recallAfter = null, recallSteps = 0, defEpochs = null;
    if (isRepeat) {
      const r = await E.recallImpactWithRefine(correct, REFINE_STEPS);
      recallBefore = r.beforeHours; recallAfter = r.afterHours; recallSteps = r.extraSteps; defEpochs = r.defaultEpochs;
      if (recallBefore != null && recallAfter != null) { recallDelta = recallAfter - recallBefore; recallReady = true; }
    } else {
      const m = await E.calculateAndStoreStatsResults();   // idle retrain keeps model current
      defEpochs = m ? m.epochsRun : null;
    }

    recs.push({ isRepeat, correct, rankChange, recallDelta, recallReady, recallBefore, recallSteps, defEpochs, date: dateMs });
  }
  return recs;
}

// ---------------------------------------------------------------------------
// Classification + aggregation
// ---------------------------------------------------------------------------
function newAgg() {
  return {
    // rank (first-time / challenge)
    rank_n: 0, rank_correct: 0,
    rank_contra_raw: 0, rank_contra_visible: 0, rank_contra_clear: 0,
    rank_visible_n: 0,                       // # with a visibly nonzero displayed number
    rank_contra_raw_correct: 0, rank_contra_raw_wrong: 0,
    rank_dirflip: 0, rank_dirmove_n: 0,      // integer-rank-label moves & wrong-way moves
    // recall (repeat)
    rec_n: 0, rec_ready: 0, rec_correct: 0,
    rec_contra_raw: 0, rec_contra_visible: 0,
    rec_visible_n: 0,
    rec_contra_raw_correct: 0, rec_contra_raw_wrong: 0,
    // saturation regime of the generic fresh-failure delay (before retrain)
    rec_floor: 0, rec_ceil: 0, rec_mid: 0,
    // contradiction restricted to the un-saturated (middle) regime
    rec_mid_visible_n: 0, rec_mid_contra: 0,
    // refine experiment: how often the extra-Adam-steps loop fired & resolved
    rec_refine_fired: 0, rec_refine_resolved: 0, rec_steps_sum: 0,
    // cost: Adam epochs the DEFAULT update runs (so extra steps can be priced)
    def_epochs_sum: 0, def_retrains: 0,
  };
}

// sinceMs: only COUNT attempts on/after this timestamp (0 = all). The rank
// integer-label tracking still threads through every first-time attempt so the
// post-cutoff "before" rank is the true running rank, not a window restart.
function classify(recs, agg, prevUserRankRef, sinceMs = 0) {
  for (const r of recs) {
    const count = r.date >= sinceMs;
    // default-update cost (every attempt runs one default retrain)
    if (count && typeof r.defEpochs === 'number') { agg.def_epochs_sum += r.defEpochs; agg.def_retrains++; }
    if (!r.isRepeat) {
      const stones = r.rankChange * TOTAL_RANKS;
      const expSign = r.correct ? 1 : -1;
      const opp = (expSign > 0 && stones < 0) || (expSign < 0 && stones > 0);
      // integer-rank-label direction (overlays "Rank up/down to X") — compute &
      // advance the running rank for EVERY first-time attempt.
      const before = prevUserRankRef.v, after = before + r.rankChange;
      const iBefore = Math.floor(before * TOTAL_RANKS), iAfter = Math.floor(after * TOTAL_RANKS);
      prevUserRankRef.v = after;
      if (!count) continue;
      agg.rank_n++;
      if (r.correct) agg.rank_correct++;
      if (Math.abs(stones) >= RANK_VISIBLE_STONES) agg.rank_visible_n++;
      if (opp && stones !== 0) {
        agg.rank_contra_raw++;
        if (r.correct) agg.rank_contra_raw_correct++; else agg.rank_contra_raw_wrong++;
      }
      if (opp && Math.abs(stones) >= RANK_VISIBLE_STONES) agg.rank_contra_visible++;
      if (opp && Math.abs(stones) >= RANK_CLEAR_STONES) agg.rank_contra_clear++;
      if (iAfter !== iBefore) {
        agg.rank_dirmove_n++;
        const upWrong = r.correct ? (iAfter < iBefore) : (iAfter > iBefore);
        if (upWrong) agg.rank_dirflip++;
      }
    } else {
      if (!count) continue;
      agg.rec_n++;
      if (r.correct) agg.rec_correct++;
      if (!r.recallReady) continue;
      agg.rec_ready++;
      const dh = r.recallDelta;            // hours
      const expSign = r.correct ? 1 : -1;
      const opp = (expSign > 0 && dh < 0) || (expSign < 0 && dh > 0);
      const visible = Math.abs(dh) * 3600 >= RECALL_VISIBLE_S;
      if (visible) agg.rec_visible_n++;
      if (opp && dh !== 0) {
        agg.rec_contra_raw++;
        if (r.correct) agg.rec_contra_raw_correct++; else agg.rec_contra_raw_wrong++;
      }
      if (opp && visible) agg.rec_contra_visible++;
      // saturation regime of the generic fresh-failure delay shown
      const b = r.recallBefore;
      const isFloor = Math.abs(b - RECALL_FLOOR_H) < 1e-6;
      const isCeil = b >= RECALL_CEIL_H - 1e-3;
      if (isFloor) agg.rec_floor++; else if (isCeil) agg.rec_ceil++;
      else {
        agg.rec_mid++;
        if (visible) { agg.rec_mid_visible_n++; if (opp) agg.rec_mid_contra++; }
      }
      // refine loop: fired iff it took ≥1 extra step (i.e. default was incongruent);
      // resolved iff the FINAL displayed delta is no longer contradictory.
      const steps = r.recallSteps || 0;
      agg.rec_steps_sum += steps;
      if (steps >= 1) { agg.rec_refine_fired++; if (!opp) agg.rec_refine_resolved++; }
    }
  }
}

function pct(a, b) { return b ? (100 * a / b).toFixed(1) + '%' : '   – '; }

// ---------------------------------------------------------------------------
// Two windows: ALL attempts, and only those on/after SINCE (post-ratings-refit).
const W = [
  { key: 'all', label: 'ALL ATTEMPTS', since: 0, overall: newAgg(), perUser: {} },
  { key: 'since', label: `SINCE ${SINCE} (post-ratings-refit)`, since: SINCE_MS, overall: newAgg(), perUser: {} },
];
const allRecs = {};
const t0 = Date.now();
let done = 0;
for (const uid of userIds) {
  const attempts = data.users[uid];
  const ut0 = Date.now();
  const recs = await replayUser(uid, attempts);
  for (const w of W) {
    const agg = newAgg();
    classify(recs, agg, { v: 0 }, w.since);
    classify(recs, w.overall, { v: 0 }, w.since);   // pooled, own rank ref
    w.perUser[uid] = agg;
  }
  allRecs[uid] = recs.map(r => ({ ...r, rankChange: +r.rankChange.toFixed(6) }));
  done++;
  process.stderr.write(
    `[${done}/${userIds.length}] ${uid.slice(0, 12).padEnd(13)} ` +
    `att=${recs.length} ft=${W[0].perUser[uid].rank_n} rep=${W[0].perUser[uid].rec_n} ` +
    `(${((Date.now() - ut0) / 1000).toFixed(1)}s)\n`);
}

// ---- report ----
function reportBlock(label, a) {
  const L = [];
  const defMean = a.def_retrains ? (a.def_epochs_sum / a.def_retrains).toFixed(0) : '–';
  L.push(`  ${label}`);
  L.push(`    FIRST-TIME (rank): n=${a.rank_n}  correct=${pct(a.rank_correct, a.rank_n)}`);
  L.push(`      contradiction (sign opposes answer):`);
  L.push(`        raw sign           ${pct(a.rank_contra_raw, a.rank_n)}  (of all first-time)`);
  L.push(`        visible (≥0.001)   ${pct(a.rank_contra_visible, a.rank_n)}  | of visible moves ${pct(a.rank_contra_visible, a.rank_visible_n)}`);
  L.push(`        clear   (≥0.01)    ${pct(a.rank_contra_clear, a.rank_n)}`);
  L.push(`        by answer:  correct→down ${pct(a.rank_contra_raw_correct, a.rank_correct)}   wrong→up ${pct(a.rank_contra_raw_wrong, a.rank_n - a.rank_correct)}`);
  L.push(`      integer-rank label moves: n=${a.rank_dirmove_n}  wrong-way ${pct(a.rank_dirflip, a.rank_dirmove_n)}`);
  L.push(`    REPEAT (recall delay): n=${a.rec_n}  ready=${a.rec_ready}  correct=${pct(a.rec_correct, a.rec_n)}`);
  L.push(`      displayed-delay regime:  floor(${RECALL_FLOOR_H}h)=${pct(a.rec_floor, a.rec_ready)}  ceil(365d)=${pct(a.rec_ceil, a.rec_ready)}  middle=${pct(a.rec_mid, a.rec_ready)}`);
  L.push(`        (floor/ceil are saturated ⇒ overlay shows "+0 s"; only the middle regime moves)`);
  L.push(`      contradiction (sign opposes answer):`);
  L.push(`        raw sign           ${pct(a.rec_contra_raw, a.rec_ready)}  (of ready repeats)`);
  L.push(`        visible (≥1 s)     ${pct(a.rec_contra_visible, a.rec_ready)}  | of visible moves ${pct(a.rec_contra_visible, a.rec_visible_n)}`);
  L.push(`        MIDDLE regime only ${pct(a.rec_mid_contra, a.rec_mid_visible_n)}  (of middle-regime visible moves — the honest denominator)`);
  L.push(`        by answer:  correct→sooner ${pct(a.rec_contra_raw_correct, a.rec_correct)}   wrong→later ${pct(a.rec_contra_raw_wrong, a.rec_n - a.rec_correct)}`);
  L.push(`      refine (extra Adam steps when incongruent, max ${REFINE_STEPS}):  fired=${a.rec_refine_fired}  resolved=${pct(a.rec_refine_resolved, a.rec_refine_fired)}  avg_steps/ready=${a.rec_ready ? (a.rec_steps_sum / a.rec_ready).toFixed(2) : '–'}`);
  L.push(`      cost: default update ≈${defMean} Adam epochs/retrain;  ${a.rec_steps_sum} extra steps total over ${a.rec_ready} repeats` +
         (a.def_retrains && a.rec_ready ? `  (≈${(100 * a.rec_steps_sum / (a.def_epochs_sum / a.def_retrains * a.rec_ready)).toFixed(2)}% extra epochs on repeats)` : ''));
  return L.join('\n');
}

// Column helper.
const C = (s, w, right = true) => right ? String(s).padStart(w) : String(s).padEnd(w);

function rankTable(overall, perUser, ids) {
  const L = [];
  L.push(`  RANK — first-time / challenge attempts (overlay shows "Rank ±x.xxx" + up/down-to):`);
  L.push(`    ${C('user', 13, false)}${C('n', 7)}${C('%corr', 7)}${C('contra', 8)}${C('vis', 7)}${C('clear', 7)}${C('cor→dn', 8)}${C('wr→up', 7)}${C('dirflip', 9)}`);
  const row = (lbl, a) => `    ${C(lbl, 13, false)}${C(a.rank_n, 7)}${C(pct(a.rank_correct, a.rank_n), 7)}` +
    `${C(pct(a.rank_contra_raw, a.rank_n), 8)}${C(pct(a.rank_contra_visible, a.rank_n), 7)}${C(pct(a.rank_contra_clear, a.rank_n), 7)}` +
    `${C(pct(a.rank_contra_raw_correct, a.rank_correct), 8)}${C(pct(a.rank_contra_raw_wrong, a.rank_n - a.rank_correct), 7)}` +
    `${C(pct(a.rank_dirflip, a.rank_dirmove_n), 9)}`;
  L.push(row('OVERALL', overall));
  L.push('    ' + '-'.repeat(67));
  for (const uid of ids) L.push(row(uid.slice(0, 12), perUser[uid]));
  return L.join('\n');
}

function recallTable(overall, perUser, ids) {
  const L = [];
  L.push(`  TIME-TO-RECALL — repeat attempts (overlay shows "±N repeat delay"):`);
  L.push(`    regime: floor/ceil are saturated ⇒ overlay shows "+0 s"; only the MIDDLE regime visibly moves.`);
  L.push(`    ${C('user', 13, false)}${C('n', 7)}${C('%corr', 7)}${C('floor', 7)}${C('mid', 7)}${C('ceil', 7)}${C('visMov', 7)}${C('midContra', 10)}${C('rawContra', 10)}`);
  const row = (lbl, a) => `    ${C(lbl, 13, false)}${C(a.rec_ready, 7)}${C(pct(a.rec_correct, a.rec_n), 7)}` +
    `${C(pct(a.rec_floor, a.rec_ready), 7)}${C(pct(a.rec_mid, a.rec_ready), 7)}${C(pct(a.rec_ceil, a.rec_ready), 7)}` +
    `${C(a.rec_mid_visible_n, 7)}${C(pct(a.rec_mid_contra, a.rec_mid_visible_n), 10)}${C(pct(a.rec_contra_raw, a.rec_ready), 10)}`;
  L.push(row('OVERALL', overall));
  L.push('    ' + '-'.repeat(78));
  for (const uid of ids) L.push(row(uid.slice(0, 12), perUser[uid]));
  L.push('');
  L.push(`    midContra = of MIDDLE-regime visible moves, fraction whose sign opposes the answer`);
  L.push(`                (correct recall → delay shorter, or wrong recall → delay longer) — the honest denominator.`);
  return L.join('\n');
}

let out = '';
out += `\n================ DISPLAYED-CHANGE CONTRADICTION REPORT ================\n`;
out += `users=${userIds.length}  refine-steps=${REFINE_STEPS}  elapsed=${((Date.now() - t0) / 1000).toFixed(0)}s\n`;
out += `A "contradiction" = the post-answer overlay moves the WRONG way for the answer:\n`;
out += `  rank   — correct attempt lowers rank, or a miss raises it.\n`;
out += `  recall — a correct recall shortens the next-repeat delay, or a miss lengthens it.\n`;
for (const w of W) {
  out += `\n##################  WINDOW: ${w.label}  ##################\n\n`;
  out += reportBlock(`AGGREGATE (pooled across all users)`, w.overall) + '\n\n';
  out += rankTable(w.overall, w.perUser, userIds) + '\n\n';
  out += recallTable(w.overall, w.perUser, userIds) + '\n';
}
out += `\n======================================================================\n`;
console.log(out);

writeFileSync(OUT, JSON.stringify({
  refineSteps: REFINE_STEPS, since: SINCE,
  windows: W.map(w => ({ key: w.key, label: w.label, overall: w.overall, perUser: w.perUser })),
  recs: allRecs,
}, null, 1));
console.error(`wrote ${OUT}`);
