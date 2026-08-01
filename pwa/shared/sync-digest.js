// shared/sync-digest.js
//
// Per-profile sync digest primitives shared by BOTH the Cloudflare Worker
// (src/handlers/{report,sync}.js, bundled by wrangler) and the client bundle
// (js/account-sync.js etc., bundled by esbuild). One source of truth so the
// server-maintained digest and the client's local mirror can never drift.
//
// The account-sync change-detection digest
// is two values maintained per user:
//
//   - user_seq    — 1-based count of the user's attempts (append-only, so it
//                   only ever increments by 1 per row). Divergence signal:
//                   client localCount != server user_seq means the device is
//                   missing (or holds extra) rows.
//   - running_sum — Σ rowHash(id) over the user's rows. CO-EQUAL divergence
//                   signal with the count (sum-first digest): a sum mismatch
//                   at equal counts is content divergence and triggers the
//                   same mirror reconcile the count does.
//
// The per-row hash is a cheap integer-arithmetic mix of the row's AUTOINCREMENT
// id (NOT created_at — no date parsing, no SQLite hash function needed). It is
// the same expression SQLite sums natively in the lazy-seed INSERT (see
// ID_HASH_SQL) and that this module reproduces in JS via BigInt, so SQLite
// int64 and JS agree exactly at any id scale (a plain Number multiply would
// lose precision past id ~3.4M; BigInt never does).

// 2654435761 = Knuth's multiplicative constant (2^32 * golden ratio).
// 2147483647 = 2^31 - 1 (Mersenne prime modulus). Keep these in lockstep with
// ID_HASH_SQL below — they are the same arithmetic in two languages.
const HASH_MULT = 2654435761n;
const HASH_MOD = 2147483647n;

// SQL fragment computing the same hash over a row's `id` column, inlined into
// the Worker's INSERT / digest queries. MUST stay arithmetically identical to
// rowHash(). Expressed over the bare column name `id` so it can drop into a
// subquery against tsumego_reports.
export const ID_HASH_SQL = '((id*2654435761)%2147483647)';

// rowHash(id) — integer in [0, 2^31). Exact match to SQLite's
// (id*2654435761)%2147483647 for any non-negative integer id.
export function rowHash(id) {
  return Number((BigInt(id) * HASH_MULT) % HASH_MOD);
}
