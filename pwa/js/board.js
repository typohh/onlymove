// js/board.js

import { state, setBoard } from './state.js';
import { handleBoardClick as handleTsumegoClick } from './tsumego.js';
import { handleBoardClick as handleAnalysisClick } from './analysis.js';
import { updateBoardSize } from './ui.js';

WGo.DIR = 'lib/wgo/';

const boardContainer = document.getElementById('board-container');

function handleBoardClick(x, y) {
    if (state.appMode === 'solving') {
        handleTsumegoClick(x, y);
    } else if (state.appMode === 'analyzing') {
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
    setBoard(board); // Update board in global state
    updateBoardSize();
}

function drawIllegalDueToKoMarker() {
    if (!state.kifuReader || !state.kifuReader.node.move) return;
    
    // Get the game from the kifu reader
    const game = state.kifuReader.game;
    
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // up, down, left, right

    for (const [dx, dy] of directions) {
        const adjX = state.kifuReader.node.move.x + dx;
        const adjY = state.kifuReader.node.move.y + dy;
        const adjResult = game.play(adjX, adjY, game.turn, true);
        if (adjResult === 4) { // illegal due to ko
            state.board.addObject({ x: adjX, y: adjY, type: 'SQ' });
        }
    }
}

function drawCorrectMoveMarker() {
    if (!state.kifuReader) return;
    const correctNode = state.kifuReader.node.children.find(child => child.comment?.includes('Correct'));
    if (correctNode?.move) {
        state.board.addObject({ x: correctNode.move.x, y: correctNode.move.y, type: 'outline' });
    }
}

function drawAnalysisHintMarker(move) {
    // Assuming move is in SGF format like 'qa' (lowercase letters)
    if (!move || move.length < 2) return;
    
    // Convert SGF coordinates to board coordinates
    // SGF uses lowercase letters 'a'-'s' for coordinates 0-18
    const x = move.charCodeAt(0) - 97;  // 'a' = 0, 'b' = 1, etc.
    const y = move.charCodeAt(1) - 97;  // 'a' = 0, 'b' = 1, etc.
    
    // Validate coordinates for 9x9 board
    if (x < 0 || x >= 9 || y < 0 || y >= 9) return;
    
    // Use the current player's color for the hint marker
    const currentPlayer = state.kifuReader.game.turn;
    
    state.board.addObject({ x, y, c: currentPlayer, type: 'outline' });
}

export function updateBoard(analysisHintMove = null) {
    if (!state.board) return;
    
    if (!state.kifuReader) return;

    const position = state.kifuReader.game.position;
    if (state.board.size !== position.size) {
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

    drawIllegalDueToKoMarker();

    if (state.kifuReader.node.move) {
        state.board.addObject({ x: state.kifuReader.node.move.x, y: state.kifuReader.node.move.y, type: 'CR' });
    }
    
    if (state.appMode === 'solving' && state.showCorrectMove) {
        drawCorrectMoveMarker();
    }
    
    if (state.appMode === 'analyzing' && analysisHintMove) {
        drawAnalysisHintMarker(analysisHintMove);
    }
}

export function initializeKifuFromSGF(sgfContent) {
    const kifu = WGo.Kifu.fromSgf(sgfContent);
    const kifuReader = new WGo.KifuReader(kifu);

    // For tsumego, advance to the first move
    while (kifuReader.node.children.length === 1) {
        kifuReader.next();
    }
    
    return kifuReader;
}
