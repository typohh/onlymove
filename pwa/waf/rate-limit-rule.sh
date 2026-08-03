#!/usr/bin/env bash
# pwa/waf/rate-limit-rule.sh
#
# The OUTER edge layer of the write defense: a Cloudflare WAF rate-limiting rule
# that blocks POST floods before they ever invoke the Worker (zero Worker + zero
# D1 cost). It is the cheapest of the four layers described in CLAUDE.md; the
# other three (the RL_REPORT/RL_WRITE Workers bindings, the in-handler format
# validation, and the MAINTENANCE_MODE freeze) all live in code already.
#
# WHY THIS FILE EXISTS. The rule is zone config, not Worker config, so `wrangler
# deploy` neither creates nor touches it — it is applied out of band, exactly
# like pwa/migrations/*.sql. Before this file the rule's parameters were
# documented in wrangler.toml but its IDENTITY (zone / ruleset / rule ids) and
# its executable form existed only in the dashboard, so a deleted or silently
# edited rule left nothing in the repo to restore it or even notice. Keeping the
# desired state here makes the rule reviewable in git and reproducible.
#
# The ids below are identifiers, not credentials, and this repo is private. The
# credential is CF_AUTH_TOKEN and is read from the environment, never committed.
# It needs the "Zone → WAF → Edit" permission on this zone.
#
#   ./pwa/waf/rate-limit-rule.sh show     # print the LIVE rule + flag any drift
#   ./pwa/waf/rate-limit-rule.sh apply    # push the desired state below
#
# SCOPE LIMIT — `apply` is a PATCH of an EXISTING rule, so it updates and cannot
# recreate. If the rule has been deleted, `show` reports it missing and the fix
# is a POST to
#   https://api.cloudflare.com/client/v4/zones/$ZONE_ID/rulesets/$RULESET_ID/rules
# with the same body, which mints a NEW rule id — update RULE_ID here afterwards.
# Recreating is left manual on purpose: it is a once-ever event, and a script
# that silently creates a second overlapping flood rule is worse than one that
# refuses.
#
# The rule matches ALL POSTs by METHOD alone (no path list), so every write
# endpoint — /report, /sync, /schedule, /rank-survey, /share-event, /push/*,
# /register, and any future POST route — is covered automatically with nothing to
# add per endpoint. Rationale for the numbers lives in wrangler.toml next to the
# binding limits; keep the two in sync when either changes.
#
# NOTE pipeline.fetch_database is unaffected by this rule: `wrangler d1 export`
# and `d1 execute --remote` hit api.cloudflare.com, a different hostname than the
# onlymove.net zone this guards.

set -euo pipefail

ZONE_ID="4789975ea205b081e5aa8be950562662"
RULESET_ID="109ff804bd184bfd84abb5619e16d45d"
RULE_ID="04f0ddcef468436b96b99bd0181ec68c"

API="https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/rulesets/${RULESET_ID}"

# --- desired state (the single source of truth for this rule) ------------------
# 100 POST/10s per source IP, block for 10s. Well above any legitimate burst —
# the heaviest real user hit ~24 reports in a 60s window — while still dropping a
# scripted flood at the edge.
read -r -d '' DESIRED <<'JSON' || true
{
    "description": "Flood guard on write endpoints (POST)",
    "expression": "(http.request.method eq \"POST\")",
    "action": "block",
    "enabled": true,
    "position": {
        "index": 1
    },
    "ratelimit": {
        "characteristics": [
            "ip.src"
        ],
        "mitigation_timeout": 10,
        "period": 10,
        "requests_per_period": 100,
        "rate_exceeds": "request_base"
    },
    "customCounter": false
}
JSON

: "${CF_AUTH_TOKEN:?set CF_AUTH_TOKEN to a token with Zone -> WAF -> Edit on this zone}"

# Cloudflare answers 200 with {"success": false, "errors": [...]} on some
# failures, so check the envelope rather than trusting the HTTP status.
cf_call() {
    local method="$1" url="$2" body="${3:-}"
    local out
    if [[ -n "$body" ]]; then
        out=$(curl -sS -X "$method" "$url" \
            -H "Authorization: Bearer ${CF_AUTH_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "$body")
    else
        out=$(curl -sS -X "$method" "$url" -H "Authorization: Bearer ${CF_AUTH_TOKEN}")
    fi
    if [[ "$(jq -r '.success' <<<"$out")" != "true" ]]; then
        echo "Cloudflare API call failed:" >&2
        jq -r '.errors' <<<"$out" >&2
        exit 1
    fi
    printf '%s' "$out"
}

# The Rulesets API has no get-one-rule endpoint; fetch the ruleset and select.
live_rule() {
    cf_call GET "$API" | jq --arg id "$RULE_ID" '.result.rules[]? | select(.id == $id)'
}

# Compare only the fields we declare — the API echoes back server-managed extras
# (id, version, last_updated, ref) that are not ours to own. Inside ratelimit we
# compare just the four ENFORCEMENT parameters: those are echoed verbatim, while
# rate_exceeds/customCounter are normalized or omitted when they hold their
# defaults, so including them would report drift on a rule that is actually in
# sync. They are still SENT on apply — this narrowing affects drift reporting
# only. If a future `show` reports drift on a rule you have not touched, widen
# this projection rather than assuming the rule changed.
comparable() {
    jq -S '{description, expression, action, enabled,
            ratelimit: (if .ratelimit == null then null else {
                characteristics: .ratelimit.characteristics,
                period: .ratelimit.period,
                requests_per_period: .ratelimit.requests_per_period,
                mitigation_timeout: .ratelimit.mitigation_timeout
            } end)}'
}

case "${1:-show}" in
show)
    rule="$(live_rule)"
    if [[ -z "$rule" ]]; then
        echo "RULE MISSING: no rule with id ${RULE_ID} in ruleset ${RULESET_ID}." >&2
        echo "It was deleted or recreated — see the SCOPE LIMIT note in this file." >&2
        exit 1
    fi
    echo "--- live ---"
    jq '{id, description, expression, action, enabled, ratelimit, last_updated}' <<<"$rule"
    if diff -q <(comparable <<<"$rule") <(comparable <<<"$DESIRED") >/dev/null; then
        echo
        echo "in sync with the desired state in this file."
    else
        echo
        echo "DRIFT — live (<) vs desired (>):"
        diff <(comparable <<<"$rule") <(comparable <<<"$DESIRED") || true
        echo
        echo "Run './pwa/waf/rate-limit-rule.sh apply' to push the desired state."
        exit 1
    fi
    ;;
apply)
    cf_call PATCH "${API}/rules/${RULE_ID}" "$DESIRED" >/dev/null
    echo "applied. verifying..."
    exec "$0" show
    ;;
*)
    echo "usage: $0 [show|apply]" >&2
    exit 1
    ;;
esac
