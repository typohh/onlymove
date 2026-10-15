// auxiliary/llm_player/describe_problem.mjs
//
// A problem as text: the position a player is shown on the board, with the
// side to play, the last-move marker, any ko-banned point and the header's
// effective komi — and nothing else. The answer, the variations' evaluations and
// the problem's rating are deliberately absent; this is what an LLM solver gets.
//
// Usage:  node auxiliary/llm_player/describe_problem.mjs <problem-id> [--source DIR|URL]
// --source defaults to the local pwa/ tree; a URL reads a running Worker's
// served data instead (cached under db/llm_player/cache/).

import path from 'node:path';
import { ROOT, DataSource, readPosition, describePosition } from './client.mjs';

const args = process.argv.slice(2);
const id = args.find((a) => /^[0-9a-f]{8}$/.test(a));
const si = args.indexOf('--source');
const source = si >= 0 ? args[si + 1] : path.join(ROOT, 'pwa');
if (!id) {
    console.error('usage: describe_problem.mjs <problem-id> [--source DIR|URL]');
    process.exit(2);
}

const src = new DataSource(source, path.join(ROOT, 'db', 'llm_player', 'cache'));
const hit = (await src.lookup()).byId.get(id);
if (!hit) {
    console.error(`${id} is not in the served lookup.txt`);
    process.exit(1);
}
console.log(describePosition(readPosition(await src.sgf(hit.size, hit.zipFilename, id))));
