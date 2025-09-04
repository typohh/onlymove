// js/board.js

import { state, setBoard, setKifuReader } from './state.js';
import { handleBoardClick as handleTsumegoClick } from './game-logic.js';
import { handleBoardClick as handleAnalysisClick } from './analysis.js';
import { updateBoardSize } from './ui.js';
import { SGF_COMMENTS } from './constants.js';

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
    setBoard(board);
    updateBoardSize();
}

function drawAnalysisHintMarker(move) {
    if (!move || typeof move.x === 'undefined' || typeof move.y === 'undefined') return;
    
    const x = move.x;
    const y = move.y;
    
    if (x < 0 || x >= state.board.size || y < 0 || y >= state.board.size) return;
    
    const currentPlayer = state.kifuReader.game.turn;
    
    state.board.addObject({ x, y, c: currentPlayer, type: 'outline' });
}

function drawIllegalDueToKoMarker() {
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

export function redraw(analysisHintMove = null) {
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

    if (state.appMode === 'analyzing' && analysisHintMove) {
        drawAnalysisHintMarker(analysisHintMove);
    }
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