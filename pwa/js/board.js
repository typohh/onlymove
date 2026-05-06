// js/board.js

import { state, setBoard, setKifuReader } from './state.js';
import { handleBoardClick as handleTsumegoClick } from './game-logic.js';
import { handleBoardClick as handleAnalysisClick } from './analysis.js';
import { updateBoardSize, updateCapturesUI } from './ui.js';
import { SGF_COMMENTS, APP_MODES } from './constants.js';

WGo.DIR = 'lib/wgo/';

const boardContainer = document.getElementById('board-container');

const isMobile = () => 'ontouchstart' in window || navigator.maxTouchPoints > 0;

const handleBoardClick = (x, y) => {
    if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
        if (isMobile()) {
            if (state.ghostMove) {
                if (state.ghostMove.x === x && state.ghostMove.y === y) {
                    // confirm
                    state.ghostMove = null;
                    handleTsumegoClick(x, y);
                } else {
                    // cancel
                    state.ghostMove = null;
                    redraw();
                }
            } else {
                // first click
                const legalNode = getLegalMoveNode(x, y);
                if (legalNode) {
                    state.ghostMove = {x, y};
                }
                // Always redraw to ensure board state is correct, even if click was invalid
                redraw();
            }
        } else {
            // desktop: single click
            handleTsumegoClick(x, y);
        }
    } else if (state.appMode === APP_MODES.ANALYZING) {
        handleAnalysisClick(x, y);
    }
}

export function initializeBoard(size) {
    if (state.board) {
        state.board.removeEventListener('click', handleBoardClick);
    }

    boardContainer.innerHTML = '';
    const boardWrapper = document.createElement('div');
    boardContainer.appendChild(boardWrapper);

    const board = new WGo.Board(boardWrapper, {
        size,
        background: WGo.DIR + "wood_1024.jpg",
        stoneHandler: WGo.Board.drawHandlers.REALISTIC
    });

    board.addEventListener('click', handleBoardClick);
    setBoard(board);
    updateBoardSize();
}

const drawAnalysisHintMarker = (move) => {
    if (!move || typeof move.x === 'undefined' || typeof move.y === 'undefined') return;

    const x = move.x;
    const y = move.y;

    if (x < 0 || x >= state.board.size || y < 0 || y >= state.board.size) return;

    const currentPlayer = state.kifuReader.game.turn;

    state.board.addObject({ x, y, c: currentPlayer, type: 'outline' });
}

const drawCandidateMarker = (candidate) => {
    if (!candidate || typeof candidate.x === 'undefined' || typeof candidate.y === 'undefined') return;
    if (candidate.x < 0 || candidate.x >= state.board.size || candidate.y < 0 || candidate.y >= state.board.size) return;

    const currentPlayer = state.kifuReader.game.turn;
    state.board.addObject({ x: candidate.x, y: candidate.y, c: currentPlayer, type: 'outline' });
}

const drawGhostMarker = (move) => {
    if (!move || typeof move.x === 'undefined' || typeof move.y === 'undefined') return;
    
    const x = move.x;
    const y = move.y;
    
    if (x < 0 || x >= state.board.size || y < 0 || y >= state.board.size) return;
    
    const currentPlayer = state.kifuReader.game.turn;
    
    state.board.addObject({ x, y, c: currentPlayer, type: 'outline' });
}

const drawIllegalDueToKoMarker = () => {
    if (!state.kifuReader || !state.kifuReader.node.move) return;
    
    const game = state.kifuReader.game;
    
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    for (const [dx, dy] of directions) {
        const adjX = state.kifuReader.node.move.x + dx;
        const adjY = state.kifuReader.node.move.y + dy;
        const adjResult = game.play(adjX, adjY, game.turn, true);
        if (adjResult === 4) { // illegal due to ko
            state.board.addObject({ x: adjX, y: adjY, type: 'SQ' });
        }
    }
}

export function redraw() {
    if (!state.kifuReader) return;

    const position = state.kifuReader.game.position;
    if (!state.board || state.board.size !== position.size) {
        initializeBoard(position.size);
    }

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
            drawCandidateMarker(candidate);
        }
        if (state.analysisHintMove && !state.candidateMoves.some(c => c.x === state.analysisHintMove.x && c.y === state.analysisHintMove.y)) {
            drawAnalysisHintMarker(state.analysisHintMove);
        }
    }

    if (state.ghostMove) {
        drawGhostMarker(state.ghostMove);
    }
    updateCapturesUI();
}

export function loadSgf(sgfContent) {
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
            stoneHandler: WGo.Board.drawHandlers.REALISTIC,
            width: width
        });
        
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
            
            const game = cloneReader.game;
            const move = cloneReader.node.move;
            const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
            for (const [dx, dy] of directions) {
                const adjX = move.x + dx;
                const adjY = move.y + dy;
                const adjResult = game.play(adjX, adjY, game.turn, true);
                if (adjResult === 4) { // illegal due to ko
                    tempBoard.addObject({ x: adjX, y: adjY, type: 'SQ' });
                }
            }
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

        // Copy all canvas layers from the temporary board
        const canvases = Array.from(tempContainer.querySelectorAll('canvas'));
        canvases.forEach(c => {
            ctx.drawImage(c, 0, 0, width, height);
        });

        return await new Promise(resolve => finalCanvas.toBlob(resolve, 'image/png'));
    } finally {
        document.body.removeChild(tempContainer);
    }
}