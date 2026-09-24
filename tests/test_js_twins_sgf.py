"""SGF twins: the eval-comment format and the conclusion predicate.

Each pair has one writer or reader in the pipeline and one in the client, and
both read the same files:

  format_eval_comment / set_eval_comment / parse_eval_visits
      (utils/katago/add_variations.py)
  formatEvalComment / setEvalComment / getVisitsFromComment / variationVisitWeights
      (pwa/js/sgf.js)
  subtree_reaches_conclusion (utils/katago/add_variations.py)
  subtreeReachesConclusion   (pwa/js/sgf.js)

The conclusion tests parse the SAME SGF text on both sides — sgfmill here, the
vendored WGo.js in node — so a difference in how either parser represents a
pass or a property-only node is part of what they check.
"""

import random

from sgfmill import sgf

from utils.katago.add_variations import (MAX_VARIATION_DEPTH, format_eval_comment,
                                         parse_eval_visits, set_eval_comment,
                                         subtree_reaches_conclusion)

SGF_JS = "pwa/js/sgf.js"


# --- eval comments ----------------------------------------------------------

COMMENTS = [
    None, "", "96.4%/1.3/1000", "96.4%/1.3", "50.0%/-2.5/30", "-5.0%/2/10",
    "0.0%/-81.5/1", "100.0%/0.0/250000", "50%/3.2/abc", "note 50.0%/2.5/7 after",
    "50.0%/1.3/1000 then 60.0%/2.0/7", "12%/3", "no evaluation here",
    "abc%/def", "1.2.3%/4/5",
]


def test_visits_parse_agrees(js):
    got = js.run([(SGF_JS, "getVisitsFromComment", [c]) for c in COMMENTS])
    for comment, j in zip(COMMENTS, got):
        # The client reports null where the pipeline reports 0 for a comment
        # that is no evaluation; every client reader applies `?? 0`.
        assert (j or 0) == parse_eval_visits(comment), comment


def test_visit_weights_agree(js):
    assert js.call("bridge", "visitWeights", COMMENTS) == [parse_eval_visits(c) for c in COMMENTS]


def _format_cases():
    rng = random.Random(7)
    cases = [(0.964, 1.3, 1000), (0.5, -2.5, 30), (0.0, -81.5, 1), (1.0, 0.0, 250000),
             (0.123456, -0.04, 30), (0.9999, 12.96, 7.9)]
    cases += [(rng.random(), rng.uniform(-90, 90), rng.randint(1, 100000)) for _ in range(500)]
    return cases


def test_format_agrees(js):
    cases = _format_cases()
    got = js.run([(SGF_JS, "formatEvalComment", c) for c in cases])
    for c, j in zip(cases, got):
        assert j == format_eval_comment(*c), c


def _py_writes(initial, writes):
    game = sgf.Sgf_game(size=9)
    node = game.extend_main_sequence()
    if initial is not None:
        node.set("C", initial)
    wrote = [set_eval_comment(node, w, s, v) for w, s, v in writes]
    return {"wrote": wrote, "comment": node.get("C") if node.has_property("C") else None}


def test_set_eval_comment_agrees(js):
    """Sequences of writes onto one node: the same writes land, the same are
    refused as downgrades, and the node ends with the same comment."""
    rng = random.Random(11)
    visit_choices = [1, 29, 30, 31, 100, 1000, 1000, 5000]
    scenarios = []
    for initial in [None, "50.0%/1.0", "50.0%/1.0/1000", "not an eval", "50.0%/1.0/29"]:
        for _ in range(40):
            writes = [(rng.random(), rng.uniform(-30, 30), rng.choice(visit_choices))
                      for _ in range(rng.randint(1, 6))]
            scenarios.append((initial, writes))
    got = js.run([("bridge", "evalWrites", (i, w)) for i, w in scenarios])
    for (initial, writes), j in zip(scenarios, got):
        assert j == _py_writes(initial, writes), (initial, writes)


# --- the conclusion predicate -------------------------------------------------

def _py_conclusion(sgf_text, path):
    node = sgf.Sgf_game.from_string(sgf_text).get_root()
    for i in path:
        node = node[i]
    return subtree_reaches_conclusion(node)


def _both(js, cases):
    """[(sgf_text, child_path)] -> (python verdicts, js verdicts)."""
    got = js.run([("bridge", "conclusionAt", (t, p)) for t, p in cases])
    return [_py_conclusion(t, p) for t, p in cases], got


HAND_BUILT = [
    # Answer played out to a double pass along the first-child chain.
    ("(;SZ[9](;B[ee];W[dd];B[];W[])(;B[cc];W[];B[dd]))", [0], True),
    # A single pass, then play resumes: not a conclusion.
    ("(;SZ[9](;B[ee];W[dd];B[];W[])(;B[cc];W[];B[dd]))", [1], False),
    # The pass pair sits on the first child's first child.
    ("(;SZ[9];B[gg](;W[];B[])(;W[hh]))", [0], True),
    # The pass pair sits on a SECOND branch, off the followed line.
    ("(;SZ[9];B[gg](;W[hh];B[ih])(;W[];B[]))", [0], False),
    # The queried node is itself a pass and its first child passes back.
    ("(;SZ[9](;B[];W[])(;B[ee]))", [0], True),
    # "tt" is a pass on boards up to 19x19 in both parsers.
    ("(;SZ[9];B[ee];W[tt];B[tt])", [0], True),
    ("(;SZ[9];B[ee];W[dd];B[tt])", [0], False),
    # A property-only child is skipped; the move child after it is followed.
    ("(;SZ[9];B[ee](;C[note])(;W[];B[]))", [0], True),
    # A property-only node INSIDE the chain ends it: it has no move children.
    ("(;SZ[9];B[ee];C[note];W[];B[])", [0], False),
    # A leaf.
    ("(;SZ[9];B[ee])", [0], False),
]


def test_conclusion_hand_built(js):
    cases = [(t, p) for t, p, _ in HAND_BUILT]
    py, j = _both(js, cases)
    for (t, p, expected), a, b in zip(HAND_BUILT, py, j):
        assert a == expected, ("python", t)
        assert b == expected, ("js", t)


def _chain(n_moves, colour_first="B"):
    """A single line: n_moves alternating on-board moves, then a double pass."""
    points = [chr(97 + x) + chr(97 + y) for x in range(9) for y in range(9)]
    colours = [colour_first, "W" if colour_first == "B" else "B"]
    nodes = [f";{colours[i % 2]}[{points[i % len(points)]}]" for i in range(n_moves)]
    nodes += [f";{colours[n_moves % 2]}[]", f";{colours[(n_moves + 1) % 2]}[]"]
    return "(;SZ[9]" + "".join(nodes) + ")"


def test_conclusion_depth_cap(js):
    """Both walks stop after the same number of plies: a line whose double pass
    lies just past the cap is refused by both, and just inside it by both."""
    cases = [(_chain(n), [0]) for n in range(MAX_VARIATION_DEPTH - 4, MAX_VARIATION_DEPTH + 4)]
    py, j = _both(js, cases)
    assert py == j
    assert py[0] and not py[-1], "the cap no longer falls inside the probed range"


def _random_tree(rng, depth, colour):
    """SGF text for a random subtree: passes, property-only nodes, branching."""
    other = "W" if colour == "B" else "B"
    r = rng.random()
    if r < 0.08:
        node = ";C[note]"
        nxt = colour
    elif r < 0.40:
        node = f";{colour}[]"
        nxt = other
    else:
        node = f";{colour}[{chr(97 + rng.randrange(9))}{chr(97 + rng.randrange(9))}]"
        nxt = other
    if depth == 0 or rng.random() < 0.15:
        return node
    k = rng.choice([1, 1, 1, 2, 3])
    kids = [_random_tree(rng, depth - 1, nxt) for _ in range(k)]
    return node + (kids[0] if k == 1 else "".join(f"({c})" for c in kids))


def test_conclusion_random_trees(js):
    rng = random.Random(3)
    cases = []
    for _ in range(300):
        kids = [_random_tree(rng, rng.randint(1, 10), "B") for _ in range(rng.randint(1, 4))]
        text = "(;SZ[9]" + "".join(f"({c})" for c in kids) + ")"
        cases += [(text, [i]) for i in range(len(kids))]
    py, j = _both(js, cases)
    mismatches = [(t, p, a, b) for (t, p), a, b in zip(cases, py, j) if a != b]
    assert not mismatches, mismatches[:3]
    # The generator must actually produce both verdicts to test anything.
    assert any(py) and not all(py)
