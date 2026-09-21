// pwa/src/aliases.js
//
// The D1 side of a self-chosen leaderboard alias: read the account's triplet,
// check a submitted one against the legal word lists, write it, clear it.
//
// Two rules this file exists to keep true, and neither is the client's to
// enforce:
//
//   LEGAL      every one of the three words must be a row in its own table. The
//              client picks from shared/alias-words.js, but that bundle can be
//              stale or hand-edited, so the tables — seeded from that same file
//              by scripts/gen-alias-seed.mjs — are the authority. The stored
//              value is the word the TABLE returned, never the raw input, so a
//              case- or spelling-different submission cannot enter the database.
//
//   UNIQUE     one alias per account (user_id is the primary key) and one account
//              per triplet (idx_user_aliases_triplet). The pre-check below cannot
//              close the race between two users submitting the same triplet, so
//              the UNIQUE index is the actual guarantee and a violation is
//              reported as 'taken' rather than swallowed.
//
// The tables are read, never written, here: adding a word is a schema change
// (gen-alias-seed --write + the migration), not a runtime action.

// The account's stored triplet, or null. Best-effort: a read failure reports
// "no alias" so /profile and the leaderboard can still render.
export async function readAlias(env, userId) {
    if (!env.DB) return null;
    try {
        const row = await env.DB.prepare(
            `SELECT adjective, color, animal FROM user_aliases WHERE user_id = ?1`
        ).bind(userId).first();
        return row ? { adjective: row.adjective, color: row.color, animal: row.animal } : null;
    } catch (e) {
        console.error('readAlias failed:', e);
        return null;
    }
}

// Validate all three words in ONE round trip (three correlated subqueries), and
// return the canonical stored spelling. null means "not a legal triplet" — a
// missing table also lands here, which is the safe direction: no alias is
// written while the migration is unapplied.
export async function canonicalAlias(env, adjective, color, animal) {
    const row = await env.DB.prepare(
        `SELECT (SELECT word FROM alias_adjectives WHERE word = ?1) AS a,
                (SELECT word FROM alias_colors     WHERE word = ?2) AS c,
                (SELECT word FROM alias_animals    WHERE word = ?3) AS n`
    ).bind(adjective, color, animal).first();
    if (row && row.a && row.c && row.n) {
        return { adjective: row.a, color: row.c, animal: row.n };
    }
    return null;
}

// Set or replace the alias. Returns 'ok', 'taken' (another account holds this
// triplet) or 'error'. ON CONFLICT(user_id) makes a re-pick a replace, and
// re-submitting the SAME triplet is a no-op rather than a self-conflict.
export async function setAlias(env, userId, alias) {
    const now = new Date().toISOString();
    try {
        await env.DB.prepare(
            `INSERT INTO user_aliases (user_id, adjective, color, animal, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)
             ON CONFLICT(user_id) DO UPDATE SET
                 adjective = excluded.adjective,
                 color = excluded.color,
                 animal = excluded.animal,
                 updated_at = excluded.updated_at`
        ).bind(userId, alias.adjective, alias.color, alias.animal, now).run();
        return 'ok';
    } catch (e) {
        const msg = String((e && e.message) || e);
        // The user_id primary key is handled by the upsert, so the only unique
        // violation reachable here is the triplet, i.e. somebody else's alias.
        if (msg.includes('UNIQUE constraint failed: user_aliases') ||
            msg.includes('idx_user_aliases_triplet')) {
            return 'taken';
        }
        console.error('setAlias failed:', e);
        return 'error';
    }
}

// Drop the alias. The account falls back to Anonymous on the board and the
// triplet becomes claimable by anyone else.
export async function clearAlias(env, userId) {
    await env.DB.prepare(`DELETE FROM user_aliases WHERE user_id = ?1`).bind(userId).run();
}
