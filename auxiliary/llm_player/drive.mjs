// auxiliary/llm_player/drive.mjs
//
// Plays an onlymove account from the command line, as a fresh device would: the
// account's history comes from the server, the rank is re-fitted by the shipped
// client code after every answer, the next problem is chosen by the client's
// serve draw, and each answer is POSTed to /report like a click. The Worker's
// leaderboard then rates the account from those reports exactly as it rates a
// person — which is the point: an LLM gets a leaderboard rank by playing.
//
// Usage (run with `node --no-warnings`; see auxiliary/llm_player/README.md):
//   drive.mjs init   --server URL [--size 9|13] [--label TEXT]
//   drive.mjs next   --server URL --user ID          print the problem to solve
//   drive.mjs answer --server URL --user ID <MOVE>   answer it (e.g. C3)
//   drive.mjs status --server URL --user ID
//   drive.mjs name <TEXT> --server URL --user ID [--country CC]
//                                                    leaderboard name + flag (default UN), via wrangler
//   drive.mjs unname --server URL --user ID          remove them (the row falls back to anonymous)
//   drive.mjs play [N] --server URL --user ID [--model M] [--effort E]
//                                                    N problems (default 1), solved by `claude -p`;
//                                                    a call with a tool round trip, no parsable answer
//                                                    or an illegal move aborts the run unreported,
//                                                    leaving the problem pending
//
// What a person's client does that this does NOT:
//   - no recalls. A repeat never moves the rank (the fit reads first-time
//     attempts only), so it would only spend solver calls.
//   - no review mode after a miss; the solver starts every problem in a new
//     context, so there is nothing to learn from one.
// Everything else on the serve path is the client's own code or a named mirror.

import path from 'node:path';
import os from 'node:os';
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync, appendFileSync, renameSync, copyFileSync, readdirSync } from 'node:fs';
import {
    ROOT, DataSource, readPosition, describePosition, shardProblems, problemCells, altStatMasses,
    stateMod, rankMod, weaknessMod, sgfMod, quantizeEmbedding, rankToLabel, serverRanksFor,
    DATA_VERSION, EMBEDDING_META_INLINE, APP_VERSION,
} from './client.mjs';

const { state } = stateMod;
const BASE_DIR = path.join(ROOT, 'db', 'llm_player');
const IGNORED_USERS = path.join(ROOT, 'db', 'ignored_users.txt');

// --- arguments ------------------------------------------------------------------
const argv = process.argv.slice(2);
const command = argv[0];
const opt = (name, dflt = null) => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 ? argv[i + 1] : dflt;
};
const positional = argv.slice(1).filter((a, i, all) => !a.startsWith('--') && !(i > 0 && all[i - 1].startsWith('--')));

function die(msg, code = 1) {
    console.error(msg);
    process.exit(code);
}

const server = opt('server');
if (!server && command !== undefined) die('--server URL is required (e.g. http://localhost:8787)', 2);
const src = server ? new DataSource(server, path.join(BASE_DIR, 'cache')) : null;
const hostDir = server ? path.join(BASE_DIR, 'accounts', new URL(server).host.replace(/[^A-Za-z0-9.-]/g, '_')) : null;

// --- account state on disk --------------------------------------------------------
// The acked report rows (what /sync would return, plus the fields this client
// derived from the served data at `dataVersion`) and the problem currently on
// the board. The server is the record; this file only spares a full pull per
// run, and is rebuilt from one when missing.
const accountPath = (userId) => path.join(hostDir, `${userId}.json`);
const logPath = (userId) => path.join(hostDir, `${userId}.log.jsonl`);

function loadAccount(userId) {
    const p = accountPath(userId);
    if (!existsSync(p)) return { userId, server, size: Number(opt('size', 9)), rows: [], pending: null };
    return JSON.parse(readFileSync(p, 'utf8'));
}

function saveAccount(acct) {
    mkdirSync(hostDir, { recursive: true });
    const p = accountPath(acct.userId);
    writeFileSync(`${p}.tmp`, JSON.stringify(acct, null, 1));
    renameSync(`${p}.tmp`, p);
}

// --- the Worker -----------------------------------------------------------------
async function postJson(route, body) {
    for (let attempt = 0; ; attempt++) {
        const resp = await fetch(`${server}${route}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        // The per-IP rate limit answers 429; back off rather than fail the run.
        if (resp.status === 429 && attempt < 6) {
            await new Promise((r) => setTimeout(r, 5000 * (attempt + 1)));
            continue;
        }
        return resp;
    }
}

// The served data is read from the server, but the rank curve, the population
// constants and the axis labels are INLINED into the client bundle — here, the
// local tree's. The Worker stamps every write response with the APP_VERSION its
// bundle was built with, which folds in those inlined files, so equality is what
// makes the local client the one that server's players run.
// APP_VERSION is `<commit>[-dirty-<nonce>]-<data hash>`. The local client is the
// served one when the data hash matches and pwa/ in the working tree is identical
// to the commit the server was built from — the commit stamps can differ (a dirty
// build, or commits since that touch nothing under pwa/).
function sameClient(served) {
    if (served === APP_VERSION) return true;
    const parse = (v) => v.match(/^([0-9a-f]+)(?:-dirty-[^-]+)?-([^-]+)$/);
    const s = parse(served), l = parse(APP_VERSION);
    if (!s || !l || s[2] !== l[2]) return false;
    const diff = spawnSync('git', ['diff', '--quiet', s[1], '--', 'pwa'], { cwd: ROOT });
    const untracked = spawnSync('git', ['ls-files', '--others', '--exclude-standard', '--', 'pwa'],
                                { cwd: ROOT, encoding: 'utf8' });
    return diff.status === 0 && untracked.status === 0 && untracked.stdout.trim() === '';
}

function checkVersion(resp) {
    const v = resp.headers.get('X-App-Version');
    if (v && !sameClient(v) && !argv.includes('--allow-version-mismatch')) {
        die(`server runs APP_VERSION ${v}, the local tree is ${APP_VERSION}: the local client ` +
            `would rank and serve with different curves than that server's players get. ` +
            `Rebuild/redeploy so they match, or pass --allow-version-mismatch.`);
    }
}

// Pull every row the server has past the newest one held (sinceId > 0 is never a
// "full pull", so it does not count against the daily cap — only a missing
// account file does). Picks up anything played on the account elsewhere.
async function syncRows(acct) {
    let since = acct.rows.reduce((m, r) => Math.max(m, r.serverId), 0);
    for (;;) {
        const resp = await postJson('/sync', { userId: acct.userId, sinceId: since, limit: 1000 });
        if (!resp.ok) die(`/sync → ${resp.status} ${await resp.text()}`);
        checkVersion(resp);
        const body = await resp.json();
        if (body.capExceeded) die('/sync: daily full-pull cap reached for this account; retry tomorrow');
        for (const r of body.reports) {
            acct.rows.push({
                serverId: r.id, problem: r.problem, success: r.success, size: r.size, move: r.move,
                timeUsed: r.timeUsed, isRepeat: r.isRepeat, createdAt: r.createdAt,
            });
        }
        since = body.cursorId;
        if (!body.hasMore) break;
    }
}

// --- history ----------------------------------------------------------------------
const shardCache = new Map();
async function problemMeta(size, zipFilename, id) {
    const key = `${size}/${zipFilename}`;
    if (!shardCache.has(key)) shardCache.set(key, shardProblems(src, size, zipFilename));
    return (await shardCache.get(key)).find((p) => p.filename === `${id}.sgf`) || null;
}

// The fields attempt-recorder.js updateUserStats writes on a live answer, derived
// from one report row and the served data. A wrong attempt also carries the
// picked cell, its vector and the availability null (picked-embedding.js), which
// the motif readings — and through them the serve draw's weakness slots — read.
async function deriveFields(row) {
    const hit = (await src.lookup()).byId.get(row.problem);
    if (!hit) return null;      // retired: its attempts drop out, as on the client
    const p = await problemMeta(hit.size, hit.zipFilename, row.problem);
    if (!p) return null;
    const d = {
        dataVersion: DATA_VERSION,
        problemRating: p.rating, problemRank: p.rank, problemQuality: p.quality || 0,
        problemSlope: p.slope, problemFloor: p.floor,
        embedding: Array.isArray(p.embedding) ? quantizeEmbedding(p.embedding) : undefined,
    };
    if (!row.success && row.move) {
        const xy = sgfMod.boardNotationToXy(row.move, hit.size);
        const pos = readPosition(await src.sgf(hit.size, hit.zipFilename, row.problem));
        const c = xy && pos.candidates.find((k) => k.x === xy.x && k.y === xy.y);
        if (c) {
            d.pickedCellIndex = c.index;
            const cells = await problemCells(src, hit.size, hit.zipFilename, row.problem);
            if (cells && cells[c.index]) d.pickedEmbedding = quantizeEmbedding(cells[c.index]);
            const alt = await altStatMasses(src, row.problem);
            if (alt) Object.assign(d, alt);
        }
    }
    return d;
}

function entryOf(row) {
    const d = row.derived;
    return {
        problemId: row.problem,
        timeTaken: row.timeUsed,
        reviewTime: 0,
        accumulatedReview: 0,
        correct: !!row.success,
        date: Date.parse(row.createdAt),
        successSinceFailure: row.success ? undefined : 0,
        isRepeat: !!row.isRepeat,
        size: row.size,
        serverId: row.serverId,
        problemRating: d.problemRating, problemRank: d.problemRank, problemQuality: d.problemQuality,
        problemSlope: d.problemSlope, problemFloor: d.problemFloor,
        ...(d.embedding ? { embedding: d.embedding } : {}),
        ...(d.pickedCellIndex !== undefined ? { pickedCellIndex: d.pickedCellIndex } : {}),
        ...(d.pickedEmbedding ? { pickedEmbedding: d.pickedEmbedding } : {}),
        ...(d.altAbove ? { altAbove: d.altAbove, altBelow: d.altBelow } : {}),
    };
}

// Hand the account's history to the client's state and re-fit the rank —
// reanchorRank, the single writer of the three rank slots.
async function loadIntoClient(acct) {
    for (const row of acct.rows) {
        if (!row.derived || row.derived.dataVersion !== DATA_VERSION) row.derived = await deriveFields(row);
    }
    const bySize = new Map();
    for (const row of acct.rows) {
        if (!row.derived) continue;
        if (!bySize.has(row.size)) bySize.set(row.size, []);
        bySize.get(row.size).push(entryOf(row));
    }
    for (const [size, entries] of bySize) stateMod.setAnswerHistoryForSize(size, entries);
    stateMod.setBoardSize(acct.size);
    state.embeddingLabels = EMBEDDING_META_INLINE.labels.components;   // backfill.js
    rankMod.reanchorRank(stateMod.getMergedAnswerHistory());
}

// --- the serve draw ---------------------------------------------------------------
// problem-loader.js hashToUnit (private there).
function hashToUnit(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    h ^= h >>> 16;
    h = Math.imul(h, 0x85ebca6b);
    h ^= h >>> 13;
    h = Math.imul(h, 0xc2b2ae35);
    h ^= h >>> 16;
    return (h >>> 0) / 0x100000000;
}

// problem-loader.js INFO_TOP_FRACTION / selectFromTopFraction / selectBest.
const INFO_TOP_FRACTION = 0.2;
function selectFromTopFraction(items, scoreFn, seed) {
    const scored = items.map((item) => ({ item, score: scoreFn(item) })).filter((e) => Number.isFinite(e.score));
    if (scored.length === 0) return items[0];
    scored.sort((a, b) => b.score - a.score);
    const n = Math.max(1, Math.ceil(scored.length * INFO_TOP_FRACTION));
    return scored[Math.min(Math.floor(hashToUnit(seed) * n), n - 1)].item;
}
function selectBest(items, scoreFn) {
    let best = items[0], bestScore = -Infinity;
    for (const item of items) {
        const s = scoreFn(item);
        if (Number.isFinite(s) && s > bestScore) { bestScore = s; best = item; }
    }
    return best;
}

// problem-loader.js selectChallengeCandidate, with the shard pool the client
// converges to: loadedShardRun anchors on the band nearest the serve rank, and
// warmBeyondCandidateSpan keeps loading the shard just past either end of the
// 50 candidates' span — so the 50 are the nearest 50 available in the whole size.
// Loading until that span stops moving reproduces the client's steady state
// without its timing. Every answered problem is excluded (never a repeat), which
// the client's RECALL_CANDIDATE_HISTORY_LENGTH window covers at this volume.
async function selectChallenge(acct) {
    const size = acct.size;
    const zips = (await src.json(`tsumegos/${size}/metadata.json`, { busted: true }))
        .map((z) => ({ ...z, minRank: rankMod.rankForSize(z.min_rating, size), maxRank: rankMod.rankForSize(z.max_rating, size) }))
        .sort((a, b) => a.min_rating - b.min_rating);
    const serveRank = state.serveRank;
    const dist = (z) => Math.min(Math.abs(serveRank - z.minRank), Math.abs(serveRank - z.maxRank));
    let home = 0;
    for (let i = 1; i < zips.length; i++) if (dist(zips[i]) < dist(zips[home])) home = i;

    const answered = new Set(stateMod.getMergedAnswerHistory().map((a) => a.problemId));
    const loaded = new Map();
    const load = async (i) => { if (!loaded.has(i)) loaded.set(i, await shardProblems(src, size, zips[i].filename)); };
    await load(home);
    let candidates;
    for (;;) {
        let lo = home, hi = home;
        while (loaded.has(lo - 1)) lo--;
        while (loaded.has(hi + 1)) hi++;
        const available = [];
        for (let i = lo; i <= hi; i++) {
            for (const p of loaded.get(i)) {
                if (!answered.has(p.filename.replace(/\.sgf$/, ''))) available.push({ ...p, shard: i });
            }
        }
        available.sort((a, b) => Math.abs(a.rank - serveRank) - Math.abs(b.rank - serveRank));
        candidates = available.slice(0, 50);
        const span = candidates.length ? candidates.map((p) => p.shard) : [lo, hi];
        const want = [Math.min(...span) - 1, Math.max(...span) + 1]
            .filter((i) => i >= 0 && i < zips.length && !loaded.has(i));
        if (want.length === 0 && candidates.length > 0) break;
        if (want.length === 0) die('no unanswered problem left in this size');
        for (const i of want) await load(i);
    }

    // The ONE categorical draw: one slot per flagged weakness, the information
    // slot while the rank is still converging, and the neutral slot always.
    const categories = weaknessMod.weaknessServeCategories();
    const withInfo = !rankMod.rankConverged();
    const slots = categories.length + (withInfo ? 1 : 0) + 1;
    const merged = stateMod.getMergedAnswerHistory();
    const lastDate = (merged.length && merged[merged.length - 1].date) || 0;
    const signature = categories.map((c) => `${c.k}${c.sign > 0 ? '+' : '-'}`).join(',') + (withInfo ? ',i' : '');
    const pick = Math.min(Math.floor(hashToUnit(`weakness-category|${lastDate}|${signature}`) * slots), slots - 1);
    let chosen, slot;
    if (pick < categories.length) {
        const { k, sign, kind } = categories[pick];
        slot = `motif ${k}${sign > 0 ? '+' : '-'} (${kind})`;
        chosen = selectBest(candidates, (p) => (p.embedding && p.embedding.length > k) ? Math.max(sign * p.embedding[k], 0) : 0);
    } else if (withInfo && pick === categories.length) {
        slot = 'information';
        chosen = selectFromTopFraction(candidates, (p) => Math.max(0, p.quality || 0), `info-pick|${lastDate}`);
    } else {
        slot = 'neutral';
        chosen = candidates[0];
    }
    return { id: chosen.filename.replace(/\.sgf$/, ''), size, zipFilename: chosen.zipFilename, slot };
}

// --- one problem --------------------------------------------------------------------
async function pendingPosition(acct) {
    const p = acct.pending;
    return readPosition(await src.sgf(p.size, p.zipFilename, p.id));
}

async function ensurePending(acct) {
    if (!acct.pending) {
        const sel = await selectChallenge(acct);
        // The clock starts when the problem is put in front of the player.
        acct.pending = { ...sel, servedAt: Date.now() };
        saveAccount(acct);
    }
    return acct.pending;
}

const REASON_TEXT = {
    'off the board': 'not a point on this board',
    occupied: 'already occupied',
    suicide: 'suicide',
    ko: 'a ko violation (immediate recapture)',
};

// Why `move` cannot be played on the pending board, or null when it can. On the
// board a click there does nothing; a person sees that and tries again.
function rejection(pos, move, size) {
    const xy = sgfMod.boardNotationToXy(move, size);
    if (!xy) return 'off the board';
    if (pos.candidates.some((c) => c.x === xy.x && c.y === xy.y)) return null;
    return pos.illegalReason(xy.x, xy.y) || 'not playable';
}

// Grade and report one answer; `move` must be playable (see rejection).
async function answerPending(acct, move, extraLog = {}) {
    const p = acct.pending;
    const pos = await pendingPosition(acct);
    const xy = sgfMod.boardNotationToXy(move, p.size);
    const cand = xy && pos.candidates.find((c) => c.x === xy.x && c.y === xy.y);
    if (!cand) throw new Error(`answerPending: ${move} is not playable`);
    const success = cand.index === 0;
    const now = Date.now();
    // A person's time is the clock from the problem appearing to the click; a
    // solver's is the model's own (`timeUsed` from solveOne), not process start-up
    // or this script's bookkeeping.
    const timeUsed = Number.isFinite(extraLog.timeUsed) ? extraLog.timeUsed : now - p.servedAt;
    const row = {
        problem: p.id, success, size: p.size,
        // Novel problems are served at orientation 0, so the board's coordinate
        // is already the canonical one game-logic.js reports.
        move: sgfMod.toBoardNotation(cand.x, cand.y, p.size),
        timeUsed, isRepeat: false, createdAt: new Date(now).toISOString(),
    };
    const resp = await postJson('/report', {
        userId: acct.userId, problem: row.problem, success, size: row.size, move: row.move,
        timeUsed: row.timeUsed, reviewTime: 0, isRepeat: false, createdAt: row.createdAt,
    });
    if (!resp.ok) die(`/report → ${resp.status} ${await resp.text()} (problem stays pending)`);
    checkVersion(resp);
    row.serverId = (await resp.json()).serverId;
    row.derived = await deriveFields(row);
    const rankBefore = { display: state.displayRank, mean: state.userRank };
    acct.rows.push(row);
    acct.pending = null;
    saveAccount(acct);
    await loadIntoClient(acct);
    const answer = pos.candidates.find((c) => c.index === 0);
    const result = {
        problem: p.id, slot: p.slot, problemRank: row.derived?.problemRank,
        move: row.move, correct: success, answer: answer ? sgfMod.toBoardNotation(answer.x, answer.y, p.size) : null,
        timeUsed: row.timeUsed, serverId: row.serverId,
        rankBefore, rankAfter: { display: state.displayRank, mean: state.userRank, se: state.rankSe },
        ...extraLog,
    };
    appendFileSync(logPath(acct.userId), JSON.stringify({ at: row.createdAt, ...result }) + '\n');
    return result;
}

// --- the solver -----------------------------------------------------------------------
// ONE `claude -p` call per problem, in a fresh context: --safe-mode loads no
// CLAUDE.md, memory, skills, hooks or MCP (and keeps subscription auth, unlike
// --bare); --tools "" leaves it no way to read anything but the prompt; and it
// runs in an empty directory outside the repo.
const SOLVER_CWD = path.join(os.homedir(), '.cache', 'onlymove-llm-player', 'solver-cwd');
const SYSTEM_PROMPT =
    'You are solving Go (baduk/weiqi) problems. Each is a whole-board position from a game ' +
    'in which exactly one move wins. You see the position as text and answer with one move.';
const ANSWER_RE = /ANSWER:\s*\**\s*([A-HJ-Ta-hj-t]\d{1,2})\b/g;
const FORMAT = 'End your reply with your move on its own line, as: ANSWER: <coordinate>  (for example ANSWER: C3).';

function runClaude(prompt, { model, effort }) {
    mkdirSync(SOLVER_CWD, { recursive: true });
    const args = ['-p', '--safe-mode', '--tools', '', '--output-format', 'json',
                  '--model', model, '--system-prompt', SYSTEM_PROMPT];
    if (effort) args.push('--effort', effort);
    args.push(prompt);
    return new Promise((resolve, reject) => {
        const child = spawn('claude', args, { cwd: SOLVER_CWD, stdio: ['ignore', 'pipe', 'pipe'] });
        let out = '', err = '';
        child.stdout.on('data', (d) => { out += d; });
        child.stderr.on('data', (d) => { err += d; });
        child.on('error', reject);
        child.on('close', (code) => {
            let body = null;
            try { body = JSON.parse(out); } catch { /* reported below */ }
            if (!body) return reject(new Error(`claude exited ${code}: ${err.slice(0, 500) || out.slice(0, 500)}`));
            if (body.is_error) return reject(new Error(`claude error: ${String(body.result).slice(0, 500)}`));
            resolve(body);
        });
    });
}

const lastAnswer = (text) => {
    const all = [...String(text || '').matchAll(ANSWER_RE)];
    return all.length ? all[all.length - 1][1].toUpperCase() : null;
};

// A call that is not a clean answer is never reported and never retried
// automatically: its transcript is kept, the problem stays pending — so the next
// launch serves it again, in a fresh context — and the run stops for a look.
// Three kinds:
//   tool_use  the call took more than one model turn, which is how a tool round
//             trip shows from outside — not the model reading the board;
//   syntax    the reply names no move in the ANSWER format;
//   illegal   the move names a point a click could not play (occupied, suicide,
//             ko, off the board).
// The per-kind counts ride on the pending problem and into its eventual log line.
function abortCall(acct, kind, detail, call) {
    const dir = path.join(hostDir, `${acct.userId}.errors`);
    mkdirSync(dir, { recursive: true });
    const projects = path.join(os.homedir(), '.claude', 'projects');
    let kept = null;
    for (const proj of readdirSync(projects)) {
        const f = path.join(projects, proj, `${call.session}.jsonl`);
        if (existsSync(f)) {
            kept = path.join(dir, `${acct.pending.id}-${kind}-${call.session}.jsonl`);
            copyFileSync(f, kept);
            break;
        }
    }
    const aborts = acct.pending.aborts = acct.pending.aborts || {};
    aborts[kind] = (aborts[kind] || 0) + 1;
    saveAccount(acct);
    appendFileSync(logPath(acct.userId), JSON.stringify({
        at: new Date().toISOString(), event: kind, problem: acct.pending.id, detail, transcript: kept, call,
    }) + '\n');
    die(`ABORT (${kind}): ${acct.pending.id} — ${detail}. Not reported; the problem stays pending ` +
        `(aborts so far: ${JSON.stringify(aborts)}). ` +
        `Transcript: ${kept ?? `session ${call.session} not found under ~/.claude/projects`}`, 4);
}

async function solveOne(acct, { model, effort }) {
    const p = acct.pending;
    const pos = await pendingPosition(acct);
    const prompt = `${describePosition(pos)}\n\n${FORMAT} Passing is not possible.`;
    const body = await runClaude(prompt, { model, effort });
    const move = lastAnswer(body.result);
    const call = {
        session: body.session_id, reply: body.result, move,
        apiMs: body.duration_api_ms, numTurns: body.num_turns, cost: body.total_cost_usd, usage: body.usage,
    };
    if (body.num_turns !== 1) abortCall(acct, 'tool_use', `the call took ${body.num_turns} model turns`, call);
    const server = body.usage?.server_tool_use || {};
    const serverTools = Object.values(server).reduce((t, v) => t + (Number.isFinite(v) ? v : 0), 0);
    if (serverTools > 0) abortCall(acct, 'tool_use', `server-side tool use ${JSON.stringify(server)}`, call);
    if (!move) abortCall(acct, 'syntax', 'no ANSWER: <coordinate> line in the reply', call);
    const why = rejection(pos, move, p.size);
    if (why) abortCall(acct, 'illegal', `${move} is ${REASON_TEXT[why] || 'not playable'}`, call);
    // The solve time is the model's own API time for the call — not process
    // start-up, nor this script's bookkeeping or the POST.
    return answerPending(acct, move, { model, effort, aborts: p.aborts || {}, call, timeUsed: call.apiMs });
}

// --- output ---------------------------------------------------------------------------
// The rank badge (displayRank) and the EGF row of its bubble.
const rankText = () => {
    const egf = serverRanksFor(state.displayRank, stateMod.getMergedAnswerHistory()).EGF;
    return `onlymove ${rankToLabel(state.displayRank)} | EGF ${Number.isFinite(egf) ? rankToLabel(egf) : '-'}`;
};
const rankDetail = () => {
    const se = state.rankSe;
    return `mean ${state.userRank.toFixed(2)}${Number.isFinite(se) ? ` ± ${se.toFixed(2)}` : ''}, ` +
        `serve ${state.serveRank.toFixed(2)}, converged ${rankMod.rankConverged()}`;
};

function printResult(r, prefix = '') {
    const mark = r.correct ? 'correct' : `wrong (answer ${r.answer})`;
    const prior = r.aborts && Object.keys(r.aborts).length ? ` [after aborts ${JSON.stringify(r.aborts)}]` : '';
    // Tokens of the answering call: everything read in (cache writes and reads
    // included) and everything written, with the thinking share of the latter.
    const u = r.call?.usage || {};
    const tokIn = (u.input_tokens || 0) + (u.cache_creation_input_tokens || 0) + (u.cache_read_input_tokens || 0);
    const think = u.output_tokens_details?.thinking_tokens;
    const tokens = r.call ? ` ${tokIn} in/${u.output_tokens || 0} out${Number.isFinite(think) ? ` (${think} thinking)` : ''} tok` : '';
    console.log(`${prefix}${r.problem} (${rankToLabel(r.problemRank)}, ${r.slot}) ${r.move} ${mark}` +
        ` ${(r.timeUsed / 1000).toFixed(1)}s${tokens}${prior} | ${rankText()}`);
}

// --- commands -----------------------------------------------------------------------
async function openAccount() {
    const userId = opt('user');
    if (!userId) die('--user ID is required (create one with `init`)', 2);
    const acct = loadAccount(userId);
    if (opt('size')) acct.size = Number(opt('size'));
    await syncRows(acct);
    await loadIntoClient(acct);
    saveAccount(acct);
    return acct;
}

// The leaderboard shows a linked name (external_ids.display_name) before an alias,
// and a binding's country_code before the reporting machine's geo-IP, so one
// external_ids row with no egd_pin/aga_id names the account and sets its flag on
// the board (and the monthly boards, which resolve identity at read time). The EGD
// and AGA refreshes select only rows carrying a PIN/AGA id, so they never touch it.
// Written with wrangler into the D1 the --server uses: --remote for a deployed
// Worker, --local for `npm run dev`.
function setName(userId, name, country) {
    if (!/^[A-Za-z0-9_-]{8}$/.test(userId)) die(`${userId} is not a user id`, 2);
    if (!name || name.length > 40) die('name must be 1-40 characters', 2);
    if (!/^[A-Z]{2}$/.test(country)) die('--country must be a 2-letter code (UN for none)', 2);
    const q = (s) => `'${s.replace(/'/g, "''")}'`;
    const sql = `INSERT INTO external_ids (user_id, display_name, country_code) ` +
        `VALUES (${q(userId)}, ${q(name)}, ${q(country)}) ` +
        `ON CONFLICT(user_id) DO UPDATE SET display_name = excluded.display_name, ` +
        `country_code = excluded.country_code`;
    const where = isLocal(server) ? '--local' : '--remote';
    console.log(`${where}: ${sql}`);
    const r = spawnSync('npx', ['wrangler', 'd1', 'execute', 'tsumego-reports', where, '--json', '--command', sql],
                        { cwd: path.join(ROOT, 'pwa'), encoding: 'utf8' });
    // An exit status of 0 is not proof of a write, so the row is read back below.
    // Remote D1 also reports the statement's change count (local does not).
    let changes;
    try { changes = JSON.parse(r.stdout)[0].meta.changes; } catch { /* reported below */ }
    if (r.status !== 0 || changes === 0) {
        die(`wrangler did not write the row (exit ${r.status}, changes ${changes}):\n${r.stdout}${r.stderr}`);
    }
    const check = spawnSync('npx', ['wrangler', 'd1', 'execute', 'tsumego-reports', where, '--json', '--command',
                                    `SELECT display_name, country_code FROM external_ids WHERE user_id = ${q(userId)}`],
                            { cwd: path.join(ROOT, 'pwa'), encoding: 'utf8' });
    let row = null;
    try { row = JSON.parse(check.stdout)[0].results[0]; } catch { /* reported below */ }
    if (!row || row.display_name !== name || row.country_code !== country) {
        die(`the row did not read back as written: ${check.stdout}${check.stderr}`);
    }
    console.log(`${userId} → ${JSON.stringify(row)}`);
}

function d1(sql) {
    const where = isLocal(server) ? '--local' : '--remote';
    const r = spawnSync('npx', ['wrangler', 'd1', 'execute', 'tsumego-reports', where, '--json', '--command', sql],
                        { cwd: path.join(ROOT, 'pwa'), encoding: 'utf8' });
    let out = null;
    try { out = JSON.parse(r.stdout)[0]; } catch { /* reported below */ }
    if (r.status !== 0 || !out) die(`wrangler failed (exit ${r.status}):\n${r.stdout}${r.stderr}`);
    return out;
}

// Deleting the binding takes the name and the flag with it: the board then shows
// the account as Anonymous with its geo-IP flag, exactly as an unlink does.
function unsetName(userId) {
    if (!/^[A-Za-z0-9_-]{8}$/.test(userId)) die(`${userId} is not a user id`, 2);
    console.log(`${isLocal(server) ? '--local' : '--remote'}: DELETE FROM external_ids WHERE user_id = '${userId}'`);
    d1(`DELETE FROM external_ids WHERE user_id = '${userId}' AND egd_pin IS NULL AND aga_id IS NULL`);
    const left = d1(`SELECT display_name, egd_pin, aga_id FROM external_ids WHERE user_id = '${userId}'`).results;
    if (left.length) die(`the row is still there: ${JSON.stringify(left)}`);
    console.log(`${userId} → no name`);
}

const isLocal = (url) => ['localhost', '127.0.0.1', '[::1]'].includes(new URL(url).hostname);

async function main() {
    switch (command) {
    case 'init': {
        // state.js minted a fresh id at import (getOrCreateUserId), as a new device does.
        const acct = { userId: state.userId, server, size: Number(opt('size', 9)), rows: [], pending: null };
        saveAccount(acct);
        console.log(acct.userId);
        if (!isLocal(server)) {
            const label = opt('label', 'llm player');
            appendFileSync(IGNORED_USERS, `${acct.userId}  # ${label} (${server}, ${new Date().toISOString().slice(0, 10)})\n`);
            console.error(`added ${acct.userId} to db/ignored_users.txt — commit it`);
        }
        break;
    }
    case 'next': {
        const acct = await openAccount();
        const p = await ensurePending(acct);
        console.log(`problem ${p.id} (${p.size}x${p.size}, ${p.slot} slot)\n`);
        console.log(describePosition(await pendingPosition(acct)));
        break;
    }
    case 'answer': {
        const acct = await openAccount();
        if (!acct.pending) die('no problem pending — run `next` first', 2);
        const move = positional[0];
        if (!move) die('usage: answer <MOVE>', 2);
        const why = rejection(await pendingPosition(acct), move.toUpperCase(), acct.pending.size);
        if (why) die(`illegal move: ${move} is ${REASON_TEXT[why] || 'not playable'}`, 3);
        printResult(await answerPending(acct, move.toUpperCase()));
        break;
    }
    case 'status': {
        const acct = await openAccount();
        const n = acct.rows.filter((r) => !r.isRepeat).length;
        const ok = acct.rows.filter((r) => !r.isRepeat && r.success).length;
        console.log(`${acct.userId} on ${server}: ${n} first-time attempts, ${ok} correct`);
        console.log(`${rankText()} (${rankDetail()})`);
        if (acct.pending) {
            const a = acct.pending.aborts;
            console.log(`pending: ${acct.pending.id}${a ? ` (aborts so far: ${JSON.stringify(a)})` : ''}`);
        }
        break;
    }
    case 'name': {
        const userId = opt('user');
        if (!userId) die('--user ID is required', 2);
        setName(userId, positional[0], opt('country', 'UN').toUpperCase());
        break;
    }
    case 'unname': {
        const userId = opt('user');
        if (!userId) die('--user ID is required', 2);
        unsetName(userId);
        break;
    }
    case 'play': {
        // `play N` is `play` run N times: each problem is the whole cycle.
        const n = positional[0] === undefined ? 1 : Number(positional[0]);
        if (!Number.isInteger(n) || n <= 0) die('usage: play [N]', 2);
        const model = opt('model', 'claude-haiku-5-5');
        const effort = opt('effort', null);
        const acct = await openAccount();
        console.log(`${acct.userId}: ${rankText()} — playing ${n} with ${model}${effort ? ` @ ${effort}` : ''}`);
        for (let i = 1; i <= n; i++) {
            await ensurePending(acct);
            printResult(await solveOne(acct, { model, effort }), `[${i}/${n}] `);
        }
        break;
    }
    default:
        die('commands: init | next | answer <MOVE> | status | name <TEXT> | unname | play [N]   (all take --server URL; see the header)', 2);
    }
}

await main();
process.exit(0);
