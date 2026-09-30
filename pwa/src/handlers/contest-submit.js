// pwa/src/handlers/contest-submit.js
//
// POST /contest/submit — the /contest page's entry form. Creates the account's
// contest_entries row in that category PENDING (listed_at NULL): the organiser
// verifies it and lists it with auxiliary/contest_entries.mjs, so the public list
// never shows an unverified entry.
//
// A second submission from the same account in the same category updates the
// row. While it is pending every field the form sends is replaced; once it is
// listed only the contact, the message and the app-link consent are, so a
// verified video link or problem count cannot be swapped after verification —
// those change through the organiser.
//
//   POST /contest/submit
//     { userId, category: 'strength'|'video',
//       contactMethod: 'whatsapp'|'discord'|'email', contact,
//       videoUrl?, videoProblems?, videoLanguage?, appLink?,   (video only)
//       message? }
//     -> { ok: true, replaced: bool, listed: bool }
//   Rejections: 400 bad_request | 400 invalid_<field> | 409 not_linked | 409 closed
//
// `replaced` says a pending submission was overwritten, `listed` that the entry
// is already public, so the page can say which of the three happened.
// A strong-player submission needs the account's linked EGD/AGA record, since
// that is what the organiser verifies and what the list shows; a video one needs
// nothing of the account, whose alias is recommended but optional.
//
// Format-validated before any D1 write, like every other write handler, and
// rate-limited by RL_WRITE at the route (src/index.js).

import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';
import { CONTEST_DEADLINE_ISO } from '../contest.js';

const CONTACT_METHODS = new Set(['whatsapp', 'discord', 'email']);
const VIDEO_LANGUAGES = new Set(['en', 'zh', 'ja', 'ko']);
const MAX_MESSAGE = 1000;
const MAX_URL = 500;

// The contact detail, per method: an address, a phone number (digits with the
// usual separators, at least six digits), or a Discord username (no spaces, an
// optional legacy #1234 discriminator). Returns the trimmed value or null.
function validContact(method, raw) {
    if (typeof raw !== 'string') return null;
    const v = raw.trim();
    if (method === 'email') {
        return v.length <= 254 && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v) ? v : null;
    }
    if (method === 'whatsapp') {
        const digits = v.replace(/\D/g, '');
        return /^\+?[0-9 ().-]+$/.test(v) && digits.length >= 6 && digits.length <= 16 ? v : null;
    }
    return /^[^\s@:#]{2,32}(#\d{4})?$/.test(v) ? v : null;
}

export async function handleContestSubmit(request, env) {
    let d;
    try { d = await request.json(); } catch { return jsonResponse({ ok: false, error: 'bad_request' }, 400); }
    if (!d || !isValidUserId(d.userId) || !['strength', 'video'].includes(d.category)) {
        return jsonResponse({ ok: false, error: 'bad_request' }, 400);
    }
    if (Date.now() > Date.parse(CONTEST_DEADLINE_ISO)) return jsonResponse({ ok: false, error: 'closed' }, 409);
    if (!CONTACT_METHODS.has(d.contactMethod)) return jsonResponse({ ok: false, error: 'invalid_contact' }, 400);
    const contact = validContact(d.contactMethod, d.contact);
    if (!contact) return jsonResponse({ ok: false, error: 'invalid_contact' }, 400);
    let message = null;
    if (d.message !== undefined && d.message !== null && d.message !== '') {
        if (typeof d.message !== 'string' || d.message.length > MAX_MESSAGE) {
            return jsonResponse({ ok: false, error: 'invalid_message' }, 400);
        }
        message = d.message.trim() || null;
    }

    let url = null, problems = null, language = null, appLink = 0;
    if (d.category === 'video') {
        url = typeof d.videoUrl === 'string' ? d.videoUrl.trim() : '';
        if (url.length > MAX_URL || !/^https:\/\/[^\s/]+\.[^\s]+$/.test(url)) {
            return jsonResponse({ ok: false, error: 'invalid_url' }, 400);
        }
        problems = d.videoProblems;
        if (!Number.isInteger(problems) || problems < 1 || problems > 10000) {
            return jsonResponse({ ok: false, error: 'invalid_problems' }, 400);
        }
        if (!VIDEO_LANGUAGES.has(d.videoLanguage)) return jsonResponse({ ok: false, error: 'invalid_language' }, 400);
        language = d.videoLanguage;
        appLink = d.appLink === true ? 1 : 0;
    }
    if (!env.DB) return jsonResponse({ ok: false, error: 'server' }, 500);

    try {
        const [linked, prior] = await Promise.all([
            d.category === 'strength'
                ? env.DB.prepare(`SELECT 1 AS x FROM external_ids WHERE user_id = ?1`).bind(d.userId).first()
                : null,
            env.DB.prepare(`SELECT id, listed_at FROM contest_entries WHERE category = ?1 AND user_id = ?2`)
                .bind(d.category, d.userId).first(),
        ]);
        if (d.category === 'strength' && !linked) return jsonResponse({ ok: false, error: 'not_linked' }, 409);
        const now = new Date().toISOString();
        if (!prior) {
            // The UNIQUE (category, user_id) index settles a double-submit race:
            // the loser's INSERT is ignored and its fields are the same form's.
            await env.DB.prepare(
                `INSERT OR IGNORE INTO contest_entries
                     (category, user_id, submitted_at, contact_method, contact,
                      video_url, video_problems, video_language, app_link, message)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)`
            ).bind(d.category, d.userId, now, d.contactMethod, contact,
                   url, problems, language, appLink, message).run();
        } else if (prior.listed_at === null) {
            await env.DB.prepare(
                `UPDATE contest_entries SET submitted_at = ?2, contact_method = ?3, contact = ?4,
                        video_url = ?5, video_problems = ?6, video_language = ?7, app_link = ?8, message = ?9
                  WHERE id = ?1`
            ).bind(prior.id, now, d.contactMethod, contact, url, problems, language, appLink, message).run();
        } else {
            await env.DB.prepare(
                `UPDATE contest_entries SET contact_method = ?2, contact = ?3, app_link = ?4, message = ?5
                  WHERE id = ?1`
            ).bind(prior.id, d.contactMethod, contact, appLink, message).run();
        }
        return jsonResponse({ ok: true, replaced: !!prior && prior.listed_at === null, listed: !!prior && prior.listed_at !== null });
    } catch (e) {
        console.error('handleContestSubmit failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}
