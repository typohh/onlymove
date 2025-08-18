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
    const reader = state.appMode === 'analyzing' ? state.analysisKifuReader : state.kifuReader;
    if (!reader || !reader.node.move) return;
    
    // Get the game from the appropriate reader
    const game = reader.game;
    
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // up, down, left, right

    for (const [dx, dy] of directions) {
        const adjX = reader.node.move.x + dx;
        const adjY = reader.node.move.y + dy;
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
    // Assuming move is in SGF format like 'Q16'
    if (!move || move.length < 2) return;
    
    // THIS IS THE FIX: Use 'let' instead of 'const'
    let x = move.charCodeAt(0) - 'A'.charCodeAt(0);
    
    // Correct for 'I' being skipped in SGF coordinates
    if (x > 7) x--; 
    const y = state.board.size - parseInt(move.substring(1));
    state.board.addObject({ x, y, type: 'outline' });
}

export function updateBoard(analysisHintMove = null) {
    if (!state.board) return;
    
    const reader = state.appMode === 'analyzing' ? state.analysisKifuReader : state.kifuReader;
    if (!reader) return;

    const position = reader.game.position;
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

    if (reader.node.move) {
        state.board.addObject({ x: reader.node.move.x, y: reader.node.move.y, type: 'CR' });
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
