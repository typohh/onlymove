const boardContainer = document.getElementById('board-container');
const rankElement = document.getElementById('rank');
const attemptedElement = document.getElementById('attempted');

let board;
let game;
let kifuReader;
let userRank = '20k'; // Default starting rank
let problems = []; // Array to store problems from metadata.json

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
    }, 2000);
}

function updateBoard() {
    const position = kifuReader.game.position;
    board.removeAllObjects();
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
    console.log(`Updated board with ${stonesAdded} stones.`);

    const lastNode = kifuReader.node;
    if (lastNode.move) {
        board.addObject({x: lastNode.move.x, y: lastNode.move.y, type: 'CR'});
    }
}

function handleBoardClick(x, y) {
    if (!game || !kifuReader) return;

    const currentPlayer = game.turn;
    const isLegal = game.isValid(x, y, currentPlayer);

    if (isLegal) {
        console.log(`Legal move played at (${x}, ${y}) by ${currentPlayer === WGo.B ? 'Black' : 'White'}`);

        const branchingNode = kifuReader.node;
        const correctMoveNode = branchingNode.children.find(child => child.move && child.move.x === x && child.move.y === y);

        if (correctMoveNode) {
            // Correct move
            let comment = correctMoveNode.comment || 'Correct!';

            const moveIndex = branchingNode.children.indexOf(correctMoveNode);
            kifuReader.next(moveIndex);
            updateBoard();

            setTimeout(() => {
                // Advance to the next node to get white's response
                kifuReader.next();
                updateBoard();

                setTimeout(() => {
                    showResult(comment);
                }, 1000);
            }, 1000);

        } else {
            // Incorrect move
            showResult('Incorrect move, try again.');
        }
    } else {
        console.log(`Illegal move at (${x}, ${y})`);
    }
}

async function loadTsumego() {
    try {
        // First load the metadata to get the list of problems
        if (problems.length === 0) {
            const metadataResponse = await fetch('metadata.json');
            problems = await metadataResponse.json();
        }

        // Filter problems by user's rank
        const rankedProblems = problems.filter(problem => problem.rank === userRank);
        
        // Select a random problem from the filtered list
        const randomProblem = rankedProblems[Math.floor(Math.random() * rankedProblems.length)];
        
        // Load the selected problem
        const response = await fetch(`tsumegos/${randomProblem.filename}`);
        console.log('SGF fetch response:', response);
        const sgfContent = await response.text();
        console.log('SGF content:', sgfContent);

        const kifu = WGo.Kifu.fromSgf(sgfContent);
        console.log('Parsed Kifu object:', kifu);
        kifuReader = new WGo.KifuReader(kifu);
        console.log('KifuReader initialized:', kifuReader);

        // Find the branching node
        while (kifuReader.node.children.length === 1) {
            kifuReader.next();
        }
        console.log('Branching node found:', kifuReader.node);

        // Clear previous board if exists
        if (board) {
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

        updateBoard();

        // Store the game object for move validation
        game = kifuReader.game;
        console.log('Game object for validation:', game);

        // Add click listener to the board
        board.addEventListener('click', handleBoardClick);
        console.log('Click listener added to board.');

        // Update UI elements
        rankElement.textContent = userRank;
        attemptedElement.textContent = '0'; // This would be updated with actual stats in a full implementation

    } catch (error) {
        console.error('Error loading or parsing SGF:', error);
    }
}

// Add a function to change user rank and load a new problem
function changeRank(newRank) {
    userRank = newRank;
    loadTsumego();
}

loadTsumego();
