"""The figure ledger: how a measured number gets from an ablation into a document.

Every figure a document quotes is a `Figure` some producer module in this
directory emits — a key, a raw value and a display format. `python -m ablations`
runs the producers, merges what they emit into `ablations/figures.json` (the
LEDGER: raw value, rendered text, which producer, and the snapshot it was measured
on), and rewrites the value inside every `<span data-fig="KEY">…</span>` in the
documents that quote it. The documents keep a single committed version each: the
key stays in the file, the text inside the span is the current value, and a
re-run's `git diff` shows exactly which figures moved. `pwa/build.mjs` refuses to
build while a span disagrees with the ledger, so a figure cannot reach the site
without the ledger backing it (`pwa/scripts/figures-check.mjs`).

A document also DATES its figures from the ledger: two `data-fig-measured` spans
hold the first and last day on which any figure that document quotes or guards was
measured, computed from the per-key `measured.at` stamps and never typed by hand
(`measured_range`). One overall claim per document, never a date beside a claim.

The ledger holds RAW numbers plus a format per key; the rendered `text` is derived
here and stored beside them only so the build check needs no formatter of its own.
Nothing else should read `text`.

Hard failure is the contract: a producer that cannot find its input, or finds it in
a shape it did not expect, raises rather than emitting a figure from a guess, so a
run doubles as a check on the shipped artifacts it reads. `require` is that
assertion; it is not `assert`, which `-O` strips.
"""

import datetime
import json
import math
import os
import re
import sqlite3
import subprocess

from dataclasses import dataclass

from utils.paths import DB_SQLITE, PROJECT_ROOT

LEDGER_PATH = os.path.join(PROJECT_ROOT, "ablations", "figures.json")

# Documents that may quote a figure, repo-relative. The build check scans the same
# two files (pwa/scripts/figures-check.mjs); add a document in both places.
DOCUMENTS = ("pwa/about.html", "docs/methodology.md")

# Cost tiers, cheapest first. A producer declares one; `--tier X` runs every
# producer at X or cheaper. `about` is minutes with no GPU (shipped artifacts, the
# local DB copy, the calibration scripts); `methodology` adds the per-cell /
# torch measurements; `refit` is the panel that retrains arms and is a GPU-day.
TIERS = ("about", "methodology", "refit")

# Keys are dotted lowercase identifiers. The last segment must not look like a
# file suffix, because docs/methodology.md's build-time scope check reads
# `something.txt` in the method chapters as a leaked file name.
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_SUFFIX_LOOKALIKES = {"js", "mjs", "py", "json", "npz", "csv", "sql", "cfg", "sgf", "txt"}
SPAN_RE = re.compile(r'<span data-fig="([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)">([^<]*)</span>')

# The dates a document says its figures were measured between, as two spans:
#     <span data-fig-measured="earliest">2026-09-01</span>
#     <span data-fig-measured="latest">2026-09-02</span>
# Each holds a UTC calendar day taken over the ledger's `measured.at` stamps of
# every key THAT DOCUMENT quotes or guards — so the range is the page's own, not
# the ledger's, and it widens whenever one tier is re-run without the others,
# which is the truthful reading (the cheap figures are fresher than the GPU-day
# ones). Nothing measures a date: a measuring run and `--sync` both rewrite these
# from the ledger, and `pwa/scripts/figures-check.mjs` recomputes the same range
# at build time and refuses a document that disagrees.
MEASURED_SPAN_RE = re.compile(r'<span data-fig-measured="(earliest|latest)">([^<]*)</span>')
_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# A GUARD is an invisible assertion a document places beside a sentence whose
# wording depends on a figure staying inside a range without quoting it — "and
# nothing else", "a handful", "on N of the K axes". A span keeps a quoted number
# current; a guard is for the prose around it, which no rewriter can fix, so a
# violated guard fails the run and the build until a person rewrites the sentence.
# Syntax, as an HTML comment adjacent to the claim:
#     <!-- fig-guard: KEY OP NUMBER -->        e.g.  some.count == 0
#     <!-- fig-guard: NUMBER OP KEY OP NUMBER -->    e.g.  0.5 <= some.share <= 0.9
# OP is one of == != < <= > >=, evaluated against the ledger's RAW value.
# `pwa/scripts/figures-check.mjs` evaluates the same grammar at build time, and
# the methodology page generator strips the comments before publishing.
GUARD_RE = re.compile(r"<!--\s*fig-guard:\s*(.*?)\s*-->")
_G_NUM = r"-?\d+(?:\.\d+)?"
_G_OP = r"==|!=|<=|>=|<|>"
_G_KEY = r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+"
_GUARD_EXPR_RE = re.compile(
    rf"^(?:(?P<lo>{_G_NUM})\s*(?P<lop>{_G_OP})\s*)?(?P<key>{_G_KEY})\s*"
    rf"(?:(?P<rop>{_G_OP})\s*(?P<hi>{_G_NUM}))?$")


class AblationError(RuntimeError):
    """A producer met something it did not expect. Never caught inside this package."""


def require(condition, message):
    if not condition:
        raise AblationError(message)


def panel_regime(records, ramp_may_vary=False, mutual_may_vary=False,
                 prior_may_vary=False):
    """The (epoch budget, learning-rate schedule) a panel's arms share, or a
    hard failure if they differ — and the opening ramp and the mutual-
    orthogonality term (weight, detachment), which the arms must also share
    unless the panel is one that SWEEPS them. These are what a BASELINE CHANGE
    moves.

    Producers used to pin the budget to the trainer's current `EPOCHS`, which
    breaks the moment that constant moves: every arm already on disk becomes
    unquotable, including panels whose internal contrast is untouched by the
    change — and a panel's arms are only ever compared to EACH OTHER. So what is
    required is that one panel ran under one regime. A finished panel measured
    under a retired baseline keeps its internal contrast; what it loses is
    comparability to arms measured under the current one, and a panel that
    STRADDLES a change is exactly that comparison, refused here.

    A SMOKE ARM IS STILL REFUSED, by a different check that is already there:
    every producer requires an arm's recorded config to equal its manifest
    entry, and `run_panel --smoke-epochs` writes its override into the config.
    A budget a manifest DECLARES is a deliberate arm — that is what a
    budget-sweep panel is made of — and such a panel reads its arms with its
    own producer rather than through this.
    """
    budgets = {r["metrics"]["epochs"] for r in records.values()}
    require(len(budgets) == 1,
            f"the panel's arms ran at {sorted(budgets)} epochs — a contrast "
            f"between arms stopped at different budgets is a contrast between "
            f"the budgets as much as between the arms")
    schedules = {r["metrics"].get("lr_schedule", "constant")
                 for r in records.values()}
    require(len(schedules) == 1,
            f"the panel's arms ran under {sorted(schedules)} schedules — the "
            f"panel straddles a baseline change, and a contrast across it is "
            f"between the schedules as much as between the arms")
    # THE OPENING RAMP IS PART OF THE REGIME. A record that predates the knob
    # ran none, which is what the default reads. A panel whose manifest
    # deliberately declares several lengths reads its arms with its own
    # producer and says so here; every other panel straddling two ramps is a
    # panel straddling a baseline change.
    ramps = {int(r["metrics"].get("lr_warmup_epochs", 0)) for r in records.values()}
    require(ramp_may_vary or len(ramps) == 1,
            f"the panel's arms ran opening ramps of {sorted(ramps)} epochs — "
            f"the panel straddles a baseline change")
    # THE MUTUAL TERM IS PART OF THE REGIME, as (weight, detached). A record
    # that predates the knob ran none, which is what the defaults read; a
    # weight of zero has no detachment to speak of. The panels that sweep the
    # term by design (`detach`, `axis_space`) say so and read it off their own
    # records.
    mutual = set()
    for r in records.values():
        w = float(r["metrics"].get("ortho_axes", 0.0))
        mutual.add((w, w > 0 and bool(r["metrics"].get("ortho_axes_detach", False))))
    require(mutual_may_vary or len(mutual) == 1,
            f"the panel's arms ran mutual-orthogonality terms of {sorted(mutual)} "
            f"(weight, detached) — the panel straddles a baseline change")
    # THE FLOOR'S SOURCE IS PART OF THE REGIME. A record that predates the
    # knob drew its floor from the trunk's head, which is what the default
    # reads.
    floors = {str(r["metrics"].get("floor_mode", "trunk")) for r in records.values()}
    require(len(floors) == 1,
            f"the panel's arms drew their floors from {sorted(floors)} — the "
            f"panel straddles a baseline change")
    # SO IS WHAT IS HELD ASIDE AND WHETHER THE CURVE IS CALIBRATED ON IT, per
    # stop unit — a panel may score some arms on held rows and others on held
    # problems, but within a unit one share and one switch. A record that
    # predates these keys held 3% aside and was not calibrated, which is what
    # the defaults read; its stop unit is in its config.
    held = {}
    for r in records.values():
        mt = r["metrics"]
        unit = str(mt.get("stop_unit", r.get("config", {}).get("stop_unit", "row")))
        held.setdefault(unit, set()).add(
            (round(float(mt.get("stop_row_frac", 0.03)), 6),
             bool(mt.get("calibrated", False))))
    for unit, vals in sorted(held.items()):
        require(len(vals) == 1,
                f"the panel's {unit}-held arms ran (held share, calibrated) = "
                f"{sorted(vals)} — the panel straddles a baseline change")
    # SO ARE THE ABILITY PRIOR AND THE BASE-RATE START. A record that predates
    # either knob ran without it. The panel that sweeps them by design
    # (`prior_start`) says so and reads them off its own records.
    for knob in ("theta_laplace", "init_base_rate"):
        vals = {bool(r["metrics"].get(knob, False)) for r in records.values()}
        require(prior_may_vary or len(vals) == 1,
                f"the panel's arms ran {knob} = {sorted(vals)} — the panel "
                f"straddles a baseline change")
    return budgets.pop(), schedules.pop()


def panel_budget(records):
    """The epoch budget a panel's arms share; see `panel_regime`."""
    return panel_regime(records)[0]


@dataclass(frozen=True)
class Figure:
    """One measured number. `fmt` is a display format (see `render`); `note` is a
    one-line statement of what the number IS, kept in the ledger beside it."""
    key: str
    value: float
    fmt: str
    note: str = ""

    def __post_init__(self):
        require(_KEY_RE.match(self.key) is not None, f"bad figure key {self.key!r}")
        require(self.key.rsplit(".", 1)[-1] not in _SUFFIX_LOOKALIKES,
                f"figure key {self.key!r} ends in a file-suffix lookalike")
        require(isinstance(self.value, (int, float)) and not isinstance(self.value, bool),
                f"{self.key}: value must be a number, got {type(self.value).__name__}")
        require(math.isfinite(self.value), f"{self.key}: value is not finite")
        render(self.value, self.fmt)       # a bad format fails at emission, not at sync

    @property
    def text(self):
        return render(self.value, self.fmt)


def render(value, fmt):
    """Display text for a raw value.

    `~N`     round to the nearest N, thousands separator: (220463, "~1000") → "220,000"
    `%SPEC`  value×100 through the Python format SPEC: (0.31, "%.0f") → "31"
    `rank`   an INTERNAL rank scalar as its kyu/dan label: (-24.3, "rank") → "25k"
    SPEC     the Python format spec itself: (0.8534, ".2f") → "0.85"; (22, "d") → "22"
    """
    if fmt == "rank":
        # Pure arithmetic on the one internal scale (0 = the 1k/1d boundary,
        # integers are band edges), so the label is a function of the stored
        # value alone — the curve that produced it is the producer's business.
        from utils.rating.rank_surveys import value_to_rank
        return value_to_rank(value)
    if fmt.startswith("~"):
        step = int(fmt[1:])
        require(step > 0, f"bad rounding step in format {fmt!r}")
        return f"{int(round(value / step)) * step:,d}"
    if fmt.startswith("%"):
        return format(value * 100.0, fmt[1:])
    if fmt == "d":
        require(float(value).is_integer(), f"format 'd' on a non-integer {value!r}")
        return format(int(value), "d")
    return format(value, fmt)


# --- provenance ------------------------------------------------------------------

def snapshot():
    """What the figures were measured against: the commit, the attempt-log frontier
    and the published data version. Every input is required — a ledger entry that
    cannot say what it was measured on is not evidence."""
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    require(os.path.exists(DB_SQLITE), f"no {DB_SQLITE} — run pipeline.fetch_database")
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        db_max_id = con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0]
    finally:
        con.close()
    require(db_max_id is not None, "tsumego_reports is empty")
    meta_path = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "embedding_meta.json")
    require(os.path.exists(meta_path), f"no {meta_path} — run pipeline.package_for_pwa")
    with open(meta_path) as f:
        data_version = json.load(f).get("data_version")
    require(isinstance(data_version, str) and data_version, "embedding_meta.json has no data_version")
    return {
        "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git + ("+dirty" if dirty else ""),
        "db_max_id": int(db_max_id),
        "data_version": data_version,
    }


# --- the ledger ------------------------------------------------------------------

def load_ledger():
    if not os.path.exists(LEDGER_PATH):
        return {"figures": {}}
    with open(LEDGER_PATH) as f:
        ledger = json.load(f)
    require(isinstance(ledger.get("figures"), dict), f"{LEDGER_PATH}: no 'figures' object")
    return ledger


def update_ledger(ledger, figures, producer, tier, measured):
    """Merge one producer's figures into the ledger in place. Keys the producer did
    not emit are untouched, so a cheap tier never clobbers an expensive one's
    figures. Returns [(key, old_text | None, new_text)] for every key that changed."""
    require(tier in TIERS, f"{producer}: unknown tier {tier!r}")
    changes = []
    entries = ledger["figures"]
    seen = set()
    # A key this producer owns but no longer emits is retired: a renamed or dropped
    # figure would otherwise sit in the ledger forever, still backing a span.
    emitted = {f.key for f in figures}
    for key in [k for k, e in entries.items()
                if e.get("producer") == producer and k not in emitted]:
        changes.append((key, entries.pop(key).get("text"), None))
    for fig in figures:
        require(fig.key not in seen, f"{producer} emitted {fig.key} twice")
        seen.add(fig.key)
        old = entries.get(fig.key)
        if old is not None and old.get("producer") != producer:
            raise AblationError(
                f"{fig.key} is owned by {old.get('producer')}, not {producer}")
        entry = {
            "value": fig.value, "fmt": fig.fmt, "text": fig.text,
            "producer": producer, "tier": tier, "note": fig.note,
            "measured": measured,
        }
        if old is None or old.get("text") != entry["text"] or old.get("value") != fig.value:
            changes.append((fig.key, None if old is None else old.get("text"), entry["text"]))
        entries[fig.key] = entry
    return changes


def save_ledger(ledger, ran=()):
    """Write the ledger back, keeping every key this run did not own.

    A run holds the whole ledger in memory from the moment it starts, so a
    straight rewrite would erase whatever another run landed meanwhile — and
    producers are run concurrently as a matter of course, since a cheap one
    reading arm records has no reason to wait behind a GPU measurement. What a
    run OWNS is the keys of the producers it actually executed: those it writes,
    retirements included. Every other key is taken from the file as it stands at
    write time rather than from the copy this run loaded.

    `ran` is the producers executed. Empty means the caller owns everything,
    which is the honest default for a caller that cannot say — it restores the
    old whole-file behaviour rather than silently half-merging."""
    if ran:
        owned = set(ran)
        try:
            with open(LEDGER_PATH, encoding="utf-8") as f:
                on_disk = json.load(f).get("figures", {})
        except (OSError, ValueError):
            on_disk = {}
        merged = dict(ledger["figures"])
        for key, entry in on_disk.items():
            if entry.get("producer") not in owned and key not in merged:
                merged[key] = entry
        # A key this run holds whose producer it did NOT run is stale in memory
        # if the disk copy moved under it: the disk wins, being the later write.
        for key, entry in on_disk.items():
            if entry.get("producer") not in owned:
                merged[key] = entry
        ledger["figures"] = merged
    ledger["_about"] = (
        "Measured figures the documents quote, one key per number. Written by "
        "`python -m ablations`; raw `value` + `fmt` are the record, `text` is "
        "derived. Never edit by hand — re-run the producer.")
    ledger["figures"] = dict(sorted(ledger["figures"].items()))
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")


# --- the documents ---------------------------------------------------------------

def _scan(doc_rel):
    path = os.path.join(PROJECT_ROOT, doc_rel)
    require(os.path.exists(path), f"document {doc_rel} is missing")
    with open(path, encoding="utf-8") as f:
        return path, f.read()


def document_keys(text):
    """Every ledger key a document depends on: the figures it quotes in spans and
    the ones its guards are evaluated against. A guard that does not parse
    contributes nothing here; `check_guards` is what reports it."""
    keys = {m.group(1) for m in SPAN_RE.finditer(text)}
    for m in GUARD_RE.finditer(text):
        g = _GUARD_EXPR_RE.match(m.group(1))
        if g is not None:
            keys.add(g.group("key"))
    return keys


def measured_range(ledger, text, doc_rel):
    """{"earliest": "YYYY-MM-DD", "latest": "YYYY-MM-DD"}: the UTC days bounding
    when the figures `text` depends on were measured, from the ledger's own
    stamps. A key the ledger lacks is skipped — the span and guard checks report
    it — but a ledgered key with no usable stamp fails, because a figure that
    cannot say when it was measured cannot be dated, and a document depending on
    no ledgered figure has no range to state."""
    entries = ledger["figures"]
    days = []
    for key in sorted(document_keys(text)):
        if key not in entries:
            continue
        at = entries[key].get("measured", {}).get("at")
        require(isinstance(at, str) and _AT_RE.match(at) is not None,
                f"{doc_rel}: {key} carries no measured.at stamp in the ledger")
        days.append(at[:10])
    require(days, f"{doc_rel}: depends on no ledgered figure, so it has no measurement dates")
    return {"earliest": min(days), "latest": max(days)}


def sync_documents(ledger, write=True):
    """Rewrite every span's text from the ledger — the figure spans and the
    measured-between dates. Returns [(doc, key, old_text, new_text)] for each
    span that changed, a date span reporting as `measured.earliest` or
    `measured.latest`; a span whose key the ledger does not hold is an error,
    because a key in a document is a claim that a producer backs it."""
    entries = ledger["figures"]
    changes = []
    for doc_rel in DOCUMENTS:
        path, text = _scan(doc_rel)

        def _sub(m):
            key, old = m.group(1), m.group(2)
            require(key in entries, f"{doc_rel}: <span data-fig=\"{key}\"> has no ledger entry")
            new = entries[key]["text"]
            if new != old:
                changes.append((doc_rel, key, old, new))
            return f'<span data-fig="{key}">{new}</span>'

        updated = SPAN_RE.sub(_sub, text)

        if MEASURED_SPAN_RE.search(text) is not None:
            dates = measured_range(ledger, text, doc_rel)

            def _sub_date(m):
                which, old = m.group(1), m.group(2)
                new = dates[which]
                if new != old:
                    changes.append((doc_rel, f"measured.{which}", old, new))
                return f'<span data-fig-measured="{which}">{new}</span>'

            updated = MEASURED_SPAN_RE.sub(_sub_date, updated)

        if write and updated != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated)
    return changes


def check_documents(ledger):
    """Every span in every document carries exactly the ledger's text — the
    figure spans and the measured-between dates. Returns the number of spans
    checked; raises listing every disagreement at once."""
    entries = ledger["figures"]
    problems, n = [], 0
    for doc_rel in DOCUMENTS:
        _, text = _scan(doc_rel)
        for m in SPAN_RE.finditer(text):
            n += 1
            key, shown = m.group(1), m.group(2)
            if key not in entries:
                problem = f"{doc_rel}: {key} has no ledger entry"
            elif entries[key]["text"] != shown:
                problem = f"{doc_rel}: {key} shows {shown!r}, ledger says {entries[key]['text']!r}"
            else:
                continue
            if problem not in problems:       # one line per (document, key), not per language block
                problems.append(problem)
        if MEASURED_SPAN_RE.search(text) is not None:
            dates = measured_range(ledger, text, doc_rel)
            for m in MEASURED_SPAN_RE.finditer(text):
                n += 1
                which, shown = m.group(1), m.group(2)
                if dates[which] != shown:
                    problems.append(f"{doc_rel}: its figures were measured {which} on "
                                    f"{dates[which]}, the document says {shown!r}")
    require(not problems, "documents disagree with the ledger:\n  " + "\n  ".join(problems))
    return n


_OPS = {
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
}


def eval_guard(expr, entries):
    """One guard expression against the ledger's raw values. Returns the failure
    as a string, or None when the guard holds. A malformed expression or an
    unledgered key is itself a failure — a guard nobody can evaluate guards nothing."""
    m = _GUARD_EXPR_RE.match(expr)
    if m is None or (m.group("lop") is None and m.group("rop") is None):
        return f"unparseable guard {expr!r}"
    key = m.group("key")
    if key not in entries:
        return f"guard {expr!r}: {key} has no ledger entry"
    value = entries[key]["value"]
    if m.group("lop") is not None and not _OPS[m.group("lop")](float(m.group("lo")), value):
        return f"guard {expr!r} violated: {key} is {value}"
    if m.group("rop") is not None and not _OPS[m.group("rop")](value, float(m.group("hi"))):
        return f"guard {expr!r} violated: {key} is {value}"
    return None


def check_guards(ledger):
    """Every fig-guard in every document holds against the ledger. Returns the
    number of guards checked; raises listing every violation at once. Evaluation
    reads the LEDGER only — never a producer — so this is as cheap as the span
    check regardless of what tier the guarded figure comes from."""
    entries = ledger["figures"]
    problems, n = [], 0
    for doc_rel in DOCUMENTS:
        _, text = _scan(doc_rel)
        for m in GUARD_RE.finditer(text):
            n += 1
            problem = eval_guard(m.group(1), entries)
            if problem is not None:
                problems.append(f"{doc_rel}: {problem} — rewrite the sentence beside it")
    require(not problems, "document claims disagree with the ledger:\n  " + "\n  ".join(problems))
    return n


def keys_in_documents():
    """{key: [documents quoting it]} — which ledger entries are actually shown."""
    used = {}
    for doc_rel in DOCUMENTS:
        _, text = _scan(doc_rel)
        for m in SPAN_RE.finditer(text):
            used.setdefault(m.group(1), []).append(doc_rel)
    return used
