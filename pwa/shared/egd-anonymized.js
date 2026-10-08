// pwa/shared/egd-anonymized.js
//
// Whether an EGD player record is one EGD has ANONYMIZED. Such a record keeps
// its PIN and still answers both lookup endpoints with retcode Ok, but its name
// fields are replaced by the placeholder "Anonymous" / "Anonymous" (or
// "Anonymous#NNNN"), and its country may be the non-country "XX". There are many
// of them, so the test is on the record's DATA, never on a PIN list.
//
// It takes the COMPOSED display name ("<Name> <Last_Name>", the one form every
// holder has: the search row, the refresh payload, the /link body), so the
// browser and the Worker apply one rule. The Worker cannot fetch EGD and sees
// only what a client posts, which is why the name is what it must be able to
// judge. Three places consult it and must stay together: the identity search
// (such a record is not offered), POST /link (it is refused), and the weekly GoR
// refresh on both ends (no heartbeat is written, so the account's last real one
// stands).
export function isAnonymizedEgdName(name) {
    return typeof name === 'string' && /^Anonymous\s+Anonymous(#\d+)?$/i.test(name.trim());
}
