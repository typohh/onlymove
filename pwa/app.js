const boardContainer = document.getElementById('board-container');
const rankElement = document.getElementById('rank');
const attemptedElement = document.getElementById('attempted');

let board;
let game;
let kifuReader;
let solutionMove;

function showResult(message) {
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    overlay.style.display = 'flex';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';
    overlay.style.zIndex = '1000';

    const messageBox = document.createElement('div');
    messageBox.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    messageBox.style.color = 'white';
    messageBox.style.padding = '20px';
    messageBox.style.borderRadius = '10px';
    messageBox.textContent = message;

    overlay.appendChild(messageBox);
    document.body.appendChild(overlay);

    setTimeout(() => {
        document.body.removeChild(overlay);
        loadTsumego();
    }, 3000);
}

function handleBoardClick(x, y) {
    if (!game || !kifuReader) return;

    const currentPlayer = game.turn;
    const isLegal = game.isValid(x, y, currentPlayer);

    if (isLegal) {
        console.log(`Legal move played at (${x}, ${y}) by ${currentPlayer === WGo.B ? 'Black' : 'White'}`);
        // Here you can add your logic for what happens after a legal move is played
        // For example, check if it's the solution move
        if (solutionMove && x === solutionMove.x && y === solutionMove.y) {
            showResult('Correct move!');
        } else {
            showResult('Incorrect move, try again.');
        }
    } else {
        console.log(`Illegal move at (${x}, ${y})`);
    }
}

async function loadTsumego() {
    try {
        const response = await fetch('tsumego.sgf');
        console.log('SGF fetch response:', response);
        const sgfContent = await response.text();
        console.log('SGF content:', sgfContent);

        const kifu = WGo.Kifu.fromSgf(sgfContent);
        console.log('Parsed Kifu object:', kifu);
        kifuReader = new WGo.KifuReader(kifu);
        console.log('KifuReader initialized:', kifuReader);

        // Go to the second to last move
        kifuReader.last();
        console.log('KifuReader after last():', kifuReader);
        kifuReader.previous();
        console.log('KifuReader after previous():', kifuReader);

        // Get the board state at the second to last move
        const position = kifuReader.game.position;
        console.log('Board position at second-to-last move:', position);

        // Clear previous board if exists
        if (board) {
            board.removeAllObjects();
            board.removeEventListener('click', handleBoardClick);
            boardContainer.innerHTML = ''; // Clear the container
            console.log('Cleared existing board.');
        }

        // Initialize WGo.Board
        board = new WGo.Board(boardContainer, {
            size: kifu.size,
            background: WGo.DIR + "wood_1024.jpg",
            stoneHandler: WGo.Board.drawHandlers.REALISTIC,
            width: boardContainer.clientWidth,
            height: boardContainer.clientHeight,
        });
        console.log('WGo.Board initialized:', board);

        // Add stones to the board based on the position
        let stonesAdded = 0;
        for (let x = 0; x < position.size; x++) {
            for (let y = 0; y < position.size; y++) {
                const color = position.get(x, y);
                if (color !== 0) {
                    board.addObject({ x: x, y: y, c: color });
                    stonesAdded++;
                }
            }
        }
        console.log(`Added ${stonesAdded} stones to the board.`);

        // Store the game object for move validation
        game = kifuReader.game;
        console.log('Game object for validation:', game);

        // Get the solution move (the first child of the current node)
        const currentNode = kifuReader.node;
        if (currentNode.children.length > 0 && currentNode.children[0].move) {
            solutionMove = currentNode.children[0].move;
            console.log('Solution move:', solutionMove);
        } else {
            solutionMove = null;
            console.log('No solution move found or current node has no children.');
        }

        // Add click listener to the board
        board.addEventListener('click', handleBoardClick);
        console.log('Click listener added to board.');

        // Update UI elements (rank, attempted) - placeholder for now
        rankElement.textContent = 'N/A';
        attemptedElement.textContent = '0';

    } catch (error) {
        console.error('Error loading or parsing SGF:', error);
    }
}

loadTsumego();