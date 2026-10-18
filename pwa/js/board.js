// js/board.js

import { state, setBoard, setKifuReader, setGhostMove, setSketchStone, getMergedAnswerHistory } from './state.js';
// NOTE: the next three imports are deliberate ESM cycles — game-logic.js,
// analysis.js, and ui.js all import board primitives back from this module.
// They work only because every imported name is called at runtime, never at
// module-eval time; don't add top-level calls to any of them.
import { handleBoardClick as handleTsumegoClick } from './game-logic.js';
import { handleBoardClick as handleAnalysisClick } from './analysis.js';
import { updateBoardSize, updateCapturesUI } from './ui.js';
import { APP_MODES } from './constants.js';
import { COL_LETTERS, tsumegoIdOf } from './sgf.js';
import { FLAWED_SHELL, hashSeed, flawRateForStreak } from './stone-style.js';
import { makeBoardWearLayer, gridWear } from './board-wear.js';

WGo.DIR = 'lib/wgo/';

const boardContainer = document.getElementById('board-container');

const handleBoardClick = (x, y, event) => {
    if (event && event.shiftKey) {
        handleSketchClick(x, y);
        return;
    }
    // A sketch still standing without Shift means its keyup was missed
    // (released outside the window): drop it, then treat this as a plain click.
    if (state.sketchStone) {
        setSketchStone(null);
        redraw();
    }
    if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
        // Honor the click-lock here too. The confirming click delegates to
        // handleTsumegoClick (which guards on acceptClicks), but the first
        // click sets state.ghostMove directly — so without this guard a click
        // during the locked post-solve transition (board still showing the old
        // problem) registers a ghost that then bleeds onto the next problem.
        if (!state.acceptClicks) return;
        // Two clicks to play, on every device: the first sets a ghost stone,
        // a second on the same point confirms it, anywhere else cancels it.
        if (state.ghostMove) {
            if (state.ghostMove.x === x && state.ghostMove.y === y) {
                // confirm
                setGhostMove(null);
                handleTsumegoClick(x, y);
            } else {
                // cancel
                setGhostMove(null);
                redraw();
            }
        } else {
            // first click
            const legalNode = getLegalMoveNode(x, y);
            if (legalNode) {
                setGhostMove({ x, y });
            }
            // Always redraw to ensure board state is correct, even if click was invalid
            redraw();
        }
    } else if (state.appMode === APP_MODES.ANALYZING) {
        handleAnalysisClick(x, y);
    }
}

// Sketch stones: while Shift is held, each click shows ONE outline stone,
// alternating black, white, black… from black on every fresh press, so a
// sequence can be pointed out move by move; releasing Shift removes it. Pure
// display — it never touches the game, the two-click ghost, or a report, and
// works in every mode. Shift is the key because it is the one the typed
// coordinate entry (coord-input.js) can never want: that path is triggered by
// unmodified column letters and digits, and a bare Shift press is neither.
function handleSketchClick(x, y) {
    if (!state.kifuReader) return;
    if (x < 0 || x >= state.board.size || y < 0 || y >= state.board.size) return;
    if (state.kifuReader.game.getPosition().get(x, y) !== 0) return;
    const c = state.sketchStone && state.sketchStone.c === WGo.B ? WGo.W : WGo.B;
    setSketchStone({ x, y, c });
    redraw();
}

function clearSketch() {
    if (!state.sketchStone) return;
    setSketchStone(null);
    redraw();
}

export function initSketchKeys() {
    // A fresh press always restarts at black; auto-repeat keydowns must not.
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Shift' && !e.repeat) clearSketch();
    });
    document.addEventListener('keyup', (e) => {
        if (e.key === 'Shift') clearSketch();
    });
    // A release while the window is unfocused never delivers a keyup.
    window.addEventListener('blur', clearSketch);
}

// Tiny always-on coordinate labels at the very edge of the board. Geometry:
// the canvas margin outside a first-line intersection is 0.75·fieldWidth and a
// stone's radius is 0.5·fieldWidth, so the outer 0.25·fieldWidth is clear even
// with a stone on the first line. Labels are kept within 0.2·fieldWidth of the
// edge because a stone placement's partial redraw clears 1.1·stoneRadius
// around the point — anything deeper gets clipped when a first-line stone
// lands. Drawn on the grid layer (under stones) as a WGo custom object, which
// survives removeAllObjects() and re-renders on resize.
const coordinateLabels = {
    grid: {
        draw: function (args, board) {
            const fieldSize = Math.min(board.fieldWidth, board.fieldHeight);
            const fontSize = Math.max(6, 0.19 * fieldSize);
            this.fillStyle = 'rgba(0, 0, 0, 0.5)';
            this.font = fontSize + 'px ' + (board.font || 'Calibri');
            this.textAlign = 'center';
            this.textBaseline = 'middle';
            // 0.65 field-units past the outer lines = 0.1·fieldWidth from the
            // canvas edge (line centers sit 0.75·fieldWidth in).
            const before = -0.65;
            const after = board.size - 1 + 0.65;
            for (let i = 0; i < board.size; i++) {
                const letter = COL_LETTERS[i];
                const number = String(board.size - i);
                this.fillText(letter, board.getX(i), board.getY(before));
                this.fillText(letter, board.getX(i), board.getY(after));
                this.fillText(number, board.getX(before), board.getY(i));
                this.fillText(number, board.getX(after), board.getY(i));
            }
        },
    },
};

export function initializeBoard(size) {
    if (state.board) {
        state.board.removeEventListener('click', handleBoardClick);
    }

    boardContainer.innerHTML = '';
    const boardWrapper = document.createElement('div');
    boardWrapper.id = 'board-wrapper';
    boardWrapper.className = 'board-wrapper';
    boardContainer.appendChild(boardWrapper);
    // Shift+mousedown would otherwise extend the page's text selection.
    boardWrapper.addEventListener('mousedown', (e) => {
        if (e.shiftKey) e.preventDefault();
    });

    const board = new WGo.Board(boardWrapper, {
        size,
        background: WGo.DIR + "wood_1024.jpg",
        stoneHandler: FLAWED_SHELL,
    });
    applyWear(board);
    // Under the grid (weight 100), over the wood: the wear layer.
    board.addLayer(makeBoardWearLayer(), 50);

    board.addEventListener('click', handleBoardClick);
    board.addCustomObject(coordinateLabels);
    // After the labels, so a patch at the edge rubs them faint too.
    board.addCustomObject(gridWear);
    setBoard(board);
    updateBoardSize();
}

// Typed-coordinate entry (coord-input.js). Plays in one step — no
// ghost-stone confirmation, since a typed coordinate is already
// deliberate. Returns whether a move was actually played: a click's feedback
// is the board itself, but typing needs an explicit invalid signal.
export function playTypedMove(x, y) {
    if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
        if (state.ghostMove) {
            setGhostMove(null);
            redraw();
        }
        return handleTsumegoClick(x, y) === true;
    }
    if (state.appMode === APP_MODES.ANALYZING) {
        return handleAnalysisClick(x, y) === true;
    }
    return false;
}

// A white outline marker's opacity. A translucent stone is a blend with the wood
// under it, and the wood is far nearer white than black, so at WGo's shared
// default a white marker barely departs from the board while a black one stands
// clear; white is drawn more opaque to read as strongly as black does.
const WHITE_OUTLINE_ALPHA = 0.6;

// Draw a hollow "outline" stone marker at `move`, in the side-to-move's color
// unless `c` is given — used identically for the analysis hint, a candidate
// move, and the ghost-stone confirmation; the sketch stone passes its own. No-op for a missing or off-board coordinate.
const drawOutlineMarker = (move, c = state.kifuReader.game.turn) => {
    if (!move || typeof move.x === 'undefined' || typeof move.y === 'undefined') return;
    if (move.x < 0 || move.x >= state.board.size || move.y < 0 || move.y >= state.board.size) return;
    const marker = { x: move.x, y: move.y, c, type: 'outline' };
    if (c === WGo.W) marker.alpha = WHITE_OUTLINE_ALPHA;
    state.board.addObject(marker);
}

// Mark (as 'SQ') the points adjacent to `move` that are illegal to play due to
// ko — a `game.play(..., true) === 4` probe per orthogonal neighbour. Shared by
// the live redraw and the off-screen share-image render (different game/board).
const markKoPoints = (game, move, board) => {
    if (!move) return;
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    for (const [dx, dy] of directions) {
        const adjX = move.x + dx;
        const adjY = move.y + dy;
        if (game.play(adjX, adjY, game.turn, true) === 4) { // illegal due to ko
            board.addObject({ x: adjX, y: adjY, type: 'SQ' });
        }
    }
}

const drawIllegalDueToKoMarker = () => {
    if (!state.kifuReader || !state.kifuReader.node.move) return;
    markKoPoints(state.kifuReader.game, state.kifuReader.node.move, state.board);
}

// The stone set is the PROBLEM's: seeded from its id, so a problem always
// shows the same stones with the same flaws, and the next problem shows
// others (stone-style.js). A board with no problem (the size picker's empty
// board) draws the seed-0 set.
function currentStoneSeed() {
    const p = state.currentProblem;
    return hashSeed(p && p.filename ? tsumegoIdOf(p.filename) : '');
}

// The stones and the wood wear with PACE: the level is the number of quick
// answers in the current run, an answer being quick when it came within
// STREAK_GAP_MS (one minute) of the one before. Walked back from the newest attempt; a
// run whose last attempt is older than the gap has ended. The run's first
// answer has no predecessor, so it says nothing about pace and adds no wear.
// That first problem also names the sitting: the wood's wear is seeded from
// it, so the marks stay put for the whole run (with no run, the problem on
// screen, which becomes that first row once answered).
const STREAK_GAP_MS = 60 * 1000;
function solvingRun() {
    const rows = getMergedAnswerHistory();
    let last = Date.now();
    let n = 0;
    for (let i = rows.length - 1; i >= 0; i--) {
        const t = rows[i].date;
        if (!Number.isFinite(t) || last - t > STREAK_GAP_MS) break;
        n++;
        last = t;
    }
    const p = state.currentProblem;
    const firstId = n > 0 ? rows[rows.length - n].problemId
                          : (p && p.filename ? tsumegoIdOf(p.filename) : '');
    return { level: Math.max(0, n - 1), firstId };
}

// The run as it stood when the board's problem was loaded (loadSgf). Every
// repaint of that problem reads it, so the wear is fixed for as long as the
// problem is on screen: measured afresh on each repaint, a minute's thought
// would wipe the marks at the next click, the ghost stone's included.
let pinnedRun = null;

// Everything a board reads to draw its wear, set together so the stone set,
// the stones' wear and the wood's wear describe one sitting.
function applyWear(board) {
    const run = pinnedRun ?? solvingRun();
    board.stoneSeed = currentStoneSeed();
    board.stoneFlawRate = flawRateForStreak(run.level);
    board.wearSeed = hashSeed(run.firstId);
    board.wearLevel = run.level;
}

export function redraw() {
    if (!state.kifuReader) return;

    const position = state.kifuReader.game.position;
    if (!state.board || state.board.size !== position.size) {
        initializeBoard(position.size);
    }

    applyWear(state.board);
    state.board.removeAllObjects();

    for (let x = 0; x < position.size; x++) {
        for (let y = 0; y < position.size; y++) {
            if (position.get(x, y) !== 0) {
                state.board.addObject({ x, y, c: position.get(x, y) });
            }
        }
    }

    if (state.kifuReader.node.move) {
        state.board.addObject({ x: state.kifuReader.node.move.x, y: state.kifuReader.node.move.y, type: 'CR' });
    }

    drawIllegalDueToKoMarker();

    if (state.appMode === APP_MODES.ANALYZING) {
        for (const candidate of state.candidateMoves) {
            drawOutlineMarker(candidate);
        }
        if (state.analysisHintMove && !state.candidateMoves.some(c => c.x === state.analysisHintMove.x && c.y === state.analysisHintMove.y)) {
            drawOutlineMarker(state.analysisHintMove);
        }
    }

    if (state.ghostMove) {
        drawOutlineMarker(state.ghostMove);
    }
    if (state.sketchStone) {
        drawOutlineMarker(state.sketchStone, state.sketchStone.c);
    }
    updateCapturesUI();
}

export function loadSgf(sgfContent) {
    // A new board invalidates any pending two-click ghost stone from the
    // previous problem; otherwise it survives the load and redraw() paints it
    // (colored by the new game's turn) on top of whatever stone now sits there.
    setGhostMove(null);
    setSketchStone(null);
    pinnedRun = solvingRun();

    const kifu = WGo.Kifu.fromSgf(sgfContent);
    const kifuReader = new WGo.KifuReader(kifu);

    while (kifuReader.node.children.length === 1) {
        kifuReader.next();
    }
    
    setKifuReader(kifuReader);
    redraw();
}

export function playMoveOnBranch(x, y) {
    if (!state.kifuReader) return false;

    const game = state.kifuReader.game;
    const position = game.getPosition();

    if (position.get(x, y) !== 0) return false;

    // Check for illegal moves (like suicide or ko).
    // A legal move returns false. Illegal moves return a numeric code (e.g., 3 for suicide, 4 for ko).
    if (game.play(x, y, game.turn, true) !== false) {
        return false;
    }

    const currentNode = state.kifuReader.node;
    const children = currentNode.children;

    // Check if the move already exists in children
    const existingChild = children.find(child => child.move && child.move.x === x && child.move.y === y);
    if (existingChild) {
        const index = children.indexOf(existingChild);
        state.kifuReader.next(index);
    } else {
        // If not, create a new node
        const newNode = new WGo.KNode({
            move: { x: x, y: y, c: game.turn }
        });
        currentNode.appendChild(newNode);
        state.kifuReader.next(currentNode.children.length - 1);
    }
    redraw();
    return true;
}

export function undoMove() {
    if (!state.kifuReader || !state.kifuReader.node.parent) return false;

    state.kifuReader.previous();
    redraw();
    return true;
}

export function getLegalMoveNode(x, y) {
    const node = state.kifuReader.node;
    const children = node.children;
    return children.find(child => child.move && child.move.x === x && child.move.y === y) || null;
}

export function getCurrentNode() {
    return state.kifuReader?.node;
}

export function playMoveByIndex(index) {
    if (!state.kifuReader) return false;
    state.kifuReader.next(index);
    redraw();
    return true;
}

export function playNextMove() {
    if (!state.kifuReader) return false;
    state.kifuReader.next();
    redraw();
    return true;
}

export async function getBoardImageBlob() {
    if (!state.board) return null;

    const width = state.board.pixel_width || state.board.width || 600;
    const height = state.board.pixel_height || state.board.height || 600;

    const bgImg = new Image();
    bgImg.src = WGo.DIR + "wood_1024.jpg";
    await new Promise(resolve => {
        bgImg.onload = resolve;
        bgImg.onerror = resolve;
    });

    // Create a temporary hidden container for an off-screen board
    const tempContainer = document.createElement('div');
    tempContainer.style.position = 'absolute';
    tempContainer.style.top = '-9999px';
    tempContainer.style.left = '-9999px';
    tempContainer.style.width = width + 'px';
    tempContainer.style.height = height + 'px';
    document.body.appendChild(tempContainer);

    try {
        // Clone the kifu reader to find the start position without affecting the game
        let startPosition = state.kifuReader ? state.kifuReader.game.position : null;
        let cloneReader = null;
        if (state.kifuReader) {
            cloneReader = new WGo.KifuReader(state.kifuReader.kifu);
            while (cloneReader.node.children && cloneReader.node.children.length === 1) {
                cloneReader.next();
            }
            startPosition = cloneReader.game.position;
        }

        // Initialize temporary board
        const tempBoard = new WGo.Board(tempContainer, {
            size: startPosition ? startPosition.size : state.board.size,
            background: "", // We will draw the background manually
            stoneHandler: FLAWED_SHELL,
            width: width
        });
        tempBoard.stoneSeed = state.board.stoneSeed;
        tempBoard.stoneFlawRate = state.board.stoneFlawRate;
        tempBoard.wearSeed = state.board.wearSeed;
        tempBoard.wearLevel = state.board.wearLevel;
        tempBoard.addLayer(makeBoardWearLayer(), 50);
        tempBoard.addCustomObject(coordinateLabels);
        tempBoard.addCustomObject(gridWear);

        // Populate the temporary board
        if (startPosition) {
            for (let x = 0; x < startPosition.size; x++) {
                for (let y = 0; y < startPosition.size; y++) {
                    if (startPosition.get(x, y) !== 0) {
                        tempBoard.addObject({ x, y, c: startPosition.get(x, y) });
                    }
                }
            }
        }

        // Add markers for the latest move and ko
        if (cloneReader && cloneReader.node.move) {
            tempBoard.addObject({ x: cloneReader.node.move.x, y: cloneReader.node.move.y, type: 'CR' });
            markKoPoints(cloneReader.game, cloneReader.node.move, tempBoard);
        }

        // Wait a small tick to ensure WGo has asynchronously rendered to the canvas layers
        await new Promise(r => setTimeout(r, 50));

        const finalCanvas = document.createElement('canvas');
        finalCanvas.width = width;
        finalCanvas.height = height;
        const ctx = finalCanvas.getContext('2d');
        
        ctx.fillStyle = '#DCB35C';
        ctx.fillRect(0, 0, width, height);

        if (bgImg.complete && bgImg.naturalWidth > 0) {
            ctx.drawImage(bgImg, 0, 0, width, height);
        }

        // Copy all canvas layers from the temporary board, in stacking order:
        // WGo appends each layer's canvas in the order they were added, which
        // is not their z-order once a layer is added under the grid.
        const canvases = Array.from(tempContainer.querySelectorAll('canvas'))
            .sort((a, b) => (parseInt(a.style.zIndex, 10) || 0) - (parseInt(b.style.zIndex, 10) || 0));
        canvases.forEach(c => {
            ctx.drawImage(c, 0, 0, width, height);
        });

        return await new Promise(resolve => finalCanvas.toBlob(resolve, 'image/png'));
    } finally {
        document.body.removeChild(tempContainer);
    }
}