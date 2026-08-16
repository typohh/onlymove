"""Board-coordinate conversions, in one place so the several coordinate spaces
the pipeline juggles are named explicitly. A past vertical-mirror bug came from
these conventions being implicit and scattered across benchmark_bots/
katago_queries; keeping them here — with the spaces spelled out — is the guard
against another one.

Coordinate spaces
-----------------
- **GTP / KataGo coordinate** — a string like ``"D4"`` (or ``"pass"``). Column
  letter from the I-skipped alphabet A,B,…,H,J,…,T (A = leftmost = 0); row
  NUMBER 1..N counting from the BOTTOM (row 1 = bottom edge). This is what
  KataGo speaks.
- **SGF coordinate** — a 2-char lowercase string like ``"pd"``. Column letter
  a..s (a = leftmost = 0); row letter a..s counting from the TOP (a = top edge).
  This is the raw text stored in SGF ``B[]``/``W[]``/``AB[]``/``AW[]``
  properties. The pass move is ``"tt"``.
- **sgfmill point** — an ``(row, col)`` tuple in sgfmill's convention: row 0 =
  BOTTOM (lower-left origin), col 0 = left. This is what sgfmill's
  ``Tree_node.set_move`` / ``Board.play`` / ``.get()`` consume and produce.
- **(x, y)** — x = column (0 = left), y = row counting from the BOTTOM
  (0 = bottom). The board-scan coordinate the activation/embedding extractor
  labels cells by.

Row origins: the GTP number, the sgfmill point, and (x, y) all count rows from
the BOTTOM; only the SGF text space counts rows from the TOP. ``gtp_to_sgfmill_point``
and ``sgf_to_gtp`` are exact inverses through sgfmill's own point<->letter
serialisation — a stone written with the former and read back with the latter
round-trips to the same GTP coordinate.
"""

GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # I-skipped, matching KataGo / GTP convention


def col_index_to_gtp_letter(col):
    """Column index (0 = left) -> GTP column letter (skips 'I')."""
    return GTP_COLS[col]


def gtp_letter_to_col_index(letter):
    """GTP column letter -> column index (0 = left); inverse of the above."""
    return GTP_COLS.index(letter.upper())


def sgf_to_gtp(sgf_coord, size):
    """SGF coordinate string (e.g. ``"pd"``, top-origin rows) -> GTP string
    (e.g. ``"D4"``, bottom-origin rows). ``"tt"`` is the SGF pass -> ``"pass"``."""
    if sgf_coord == "tt":
        return "pass"
    col = ord(sgf_coord[0]) - ord('a')
    row = ord(sgf_coord[1]) - ord('a')       # 0 = top (SGF row letter)
    gtp_col = col_index_to_gtp_letter(col)
    gtp_row = size - row                      # top-origin letter -> bottom-origin number
    gtp_coord = f"{gtp_col}{gtp_row}"
    if gtp_row < 0:
        raise Exception(f"Invalid SGF coordinate: {sgf_coord}->{gtp_coord}")
    return gtp_coord


def gtp_to_sgfmill_point(gtp_coord, size):
    """GTP string (e.g. ``"D4"``) -> sgfmill ``(row, col)`` point (row 0 =
    bottom). Returns None for a pass.

    Both spaces count rows from the bottom, so the row index is simply
    ``int(row_str) - 1``. This is the exact inverse of ``sgf_to_gtp`` via
    sgfmill's point<->letter serialisation: a stone written with this function
    and read back with ``sgf_to_gtp`` round-trips to the same GTP coordinate.
    (It historically used ``size - int(row_str)``, which stored every board
    vertically mirrored vs KataGo's frame — harmless because Go is
    reflection-symmetric and every consumer read the stored SGF consistently,
    but it made the writer and reader non-inverse.)
    """
    if gtp_coord == "pass":
        return None
    col = gtp_letter_to_col_index(gtp_coord[0])
    row = int(gtp_coord[1:]) - 1              # GTP row 1 = bottom -> sgfmill row 0
    return row, col


def sgfmill_point_to_sgf(point, size):
    """sgfmill ``(row, col)`` point (row 0 = bottom) -> SGF coordinate string
    (top-origin row letter).

    sgfmill points count rows from the BOTTOM; an SGF coordinate's row letter
    counts from the TOP. Flipping the row (``size - 1 - row``) is exactly how
    sgfmill serialises a point, so an AB/AW setup stone read this way resolves to
    the same square as a move-node coordinate (which comes straight from the raw
    top-origin SGF text). Without the flip the setup stones would be vertically
    mirrored against the moves — latent only because current tsumego SGFs place
    setup via move nodes and carry no AB/AW.
    """
    row, col = point
    return f"{chr(ord('a') + col)}{chr(ord('a') + (size - 1 - row))}"


def gtp_to_xy(gtp_coord):
    """GTP string -> ``(x, y)`` with x = column (0 = left), y = row from the
    BOTTOM (0 = bottom). Returns None for a pass. Size-independent."""
    if gtp_coord.lower() == "pass":
        return None
    x = gtp_letter_to_col_index(gtp_coord[0])
    y = int(gtp_coord[1:]) - 1               # GTP row 1 -> y=0 (bottom)
    return x, y


def xy_to_gtp(x, y):
    """``(x, y)`` -> GTP string. Inverse of gtp_to_xy, so y counts from the
    BOTTOM. Size-independent (the caller is responsible for in-range x, y)."""
    return f"{col_index_to_gtp_letter(x)}{y + 1}"
