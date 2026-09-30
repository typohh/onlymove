// pwa/src/handlers/contest-submit.js
//
// POST /contest/submit — the /contest page's entry form. Creates the account's
// video contest_entries row PENDING (listed_at NULL): the organiser
// verifies it and lists it with auxiliary/contest_entries.mjs, so the public list
// never shows an unverified entry.
//
// A second submission from the same account updates the row. While it is pending every field the form sends is replaced; once it is
// listed only the contact, the message and the app-link consent are, so a
// verified video link or problem count cannot be swapped after verification —
// those change through the organiser.
//
//   POST /contest/submit
//     { userId, contactMethod: 'whatsapp'|'discord'|'email', contact,
//       videoUrl, videoProblems, videoLanguage, appLink?, message? }
//     -> { ok: true, replaced: bool, listed: bool }
//   Rejections: 400 bad_request | 400 invalid_<field> | 409 closed
//
// `replaced` says a pending submission was overwritten, `listed` that the entry
// is already public, so the page can say which of the three happened. Nothing of
// the account is required: its alias is what the list shows, and is optional.
//
// Format-validated before any D1 write, like every other write handler, and
// rate-limited by RL_WRITE at the route (src/index.js).
//
// Every accepted submission is announced to the organiser through a Discord
// webhook (secret CONTEST_DISCORD_WEBHOOK; absent, nothing is sent). The post
// carries the entry id and what is public or harmless — the video link, its
// problem count and language, the app-link choice, the contact METHOD — and
// never the contact detail, the message or the user id: the privacy policy
// promises the first two are seen only by the organiser, and a Discord channel
// is a third party; the third is the account credential.

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

const KIND_TEXT = {
    new: 'New contest submission',
    replaced: 'Contest submission replaced (still pending)',
    listed: 'Listed contest entry updated its contact details',
};

// Fire-and-forget: a failed post is logged and never fails the submission.
async function notifyOrganiser(env, { id, kind, url, problems, language, appLink, contactMethod, hasMessage }) {
    if (!env.CONTEST_DISCORD_WEBHOOK) return;
    // A listed entry's link, count and language are not changed by the form
    // (the organiser's script owns them), so they are left out of its post.
    const details = [
        ...(url ? [`${problems} problems`, language] : []),
        `contact by ${contactMethod}`,
        ...(appLink ? ['may be linked from the app'] : []),
        ...(hasMessage ? ['has a message'] : []),
    ];
    const lines = [
        `**${KIND_TEXT[kind]}** — entry #${id ?? '?'}`,
        ...(url ? [`<${url}>`] : []),
        details.join(' · '),
    ];
    try {
        const res = await fetch(env.CONTEST_DISCORD_WEBHOOK, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // No mentions of any kind, whatever the URL happens to contain.
            body: JSON.stringify({ content: lines.join('\n'), allowed_mentions: { parse: [] } }),
        });
        if (!res.ok) console.error('contest webhook failed:', res.status);
    } catch (e) {
        console.error('contest webhook failed:', e);
    }
}

export async function handleContestSubmit(request, env, ctx) {
    let d;
    try { d = await request.json(); } catch { return jsonResponse({ ok: false, error: 'bad_request' }, 400); }
    if (!d || !isValidUserId(d.userId)) {
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

    const url = typeof d.videoUrl === 'string' ? d.videoUrl.trim() : '';
    if (url.length > MAX_URL || !/^https:\/\/[^\s/]+\.[^\s]+$/.test(url)) {
        return jsonResponse({ ok: false, error: 'invalid_url' }, 400);
    }
    const problems = d.videoProblems;
    if (!Number.isInteger(problems) || problems < 1 || problems > 10000) {
        return jsonResponse({ ok: false, error: 'invalid_problems' }, 400);
    }
    if (!VIDEO_LANGUAGES.has(d.videoLanguage)) return jsonResponse({ ok: false, error: 'invalid_language' }, 400);
    const language = d.videoLanguage;
    const appLink = d.appLink === true ? 1 : 0;
    if (!env.DB) return jsonResponse({ ok: false, error: 'server' }, 500);

    try {
        const prior = await env.DB.prepare(
            `SELECT id, listed_at FROM contest_entries WHERE category = 'video' AND user_id = ?1`
        ).bind(d.userId).first();
        const now = new Date().toISOString();
        let id = prior ? prior.id : null;
        if (!prior) {
            // The UNIQUE (category, user_id) index settles a double-submit race:
            // the loser's INSERT is ignored and its fields are the same form's.
            const row = await env.DB.prepare(
                `INSERT OR IGNORE INTO contest_entries
                     (category, user_id, submitted_at, contact_method, contact,
                      video_url, video_problems, video_language, app_link, message)
                 VALUES ('video', ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 RETURNING id`
            ).bind(d.userId, now, d.contactMethod, contact,
                   url, problems, language, appLink, message).first();
            id = row ? row.id : null;
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
        const replaced = !!prior && prior.listed_at === null;
        const listed = !!prior && prior.listed_at !== null;
        const note = notifyOrganiser(env, {
            id, kind: listed ? 'listed' : replaced ? 'replaced' : 'new',
            url: listed ? null : url, problems, language, appLink,
            contactMethod: d.contactMethod, hasMessage: !!message,
        });
        if (ctx && ctx.waitUntil) ctx.waitUntil(note); else await note;
        return jsonResponse({ ok: true, replaced, listed });
    } catch (e) {
        console.error('handleContestSubmit failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}
