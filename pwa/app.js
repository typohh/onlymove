const boardContainer = document.getElementById('board-container');
const rankElement = document.getElementById('rank');

// App version - last modified date and time
const APP_VERSION = "2025-07-26 22:24";

console.log(`Yose Tsumego App v${APP_VERSION} loaded`);

// Force service worker update
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(registrations => {
    registrations.forEach(reg => reg.update());
  });
}

let board;
let game;
let kifuReader;
// Initialize user rank from localStorage or start at 0.5
let userRank = localStorage.getItem('userRank') ? parseFloat(localStorage.getItem('userRank')) : 0.5;
let problems = []; // Array to store problems from metadata.json
// Initialize problem frequencies from localStorage or default to 1
let problemFrequencies = localStorage.getItem('problemFrequencies') ? 
    JSON.parse(localStorage.getItem('problemFrequencies')) : {};
let currentProblemFilename = null; // Track current problem filename
let showCorrectMove = false; // Flag to indicate if we should show the correct move

// Function to update board size based on screen orientation
function updateBoardSize() {
    if (!board) return;
    
    const isLandscape = window.matchMedia("(orientation: landscape)").matches;
    const boardSize = isLandscape ? 400 : Math.min(window.innerWidth, window.innerHeight) * 0.95;
    board.setWidth(boardSize);
    board.setHeight(boardSize);
    
    // Ensure container takes full space
    boardContainer.style.display = 'block';
}

// Convert numeric rank to descriptive rank
function getDescriptiveRank(numericRank) {
    // There are 29 ranks from 20 kyu to 9 dan
    // 0.0 = 20 kyu, 1.0 = 9 dan
    const totalRanks = 29;
    const rankIndex = Math.floor(numericRank * totalRanks);
    
    if (rankIndex < 20) {
        // Kyu ranks (20 kyu to 1 kyu)
        const kyuRank = 20 - rankIndex;
        return `${kyuRank} kyu`;
    } else {
        // Dan ranks (1 dan to 9 dan)
        const danRank = rankIndex - 19;
        return `${danRank} dan`;
    }
}

function showResult(message, correct, problemFilename) {
    // Update problem frequency based on correctness
    if (correct) {
        // Decrease frequency to 1/5 of current value
        const currentFreq = problemFrequencies[problemFilename] || 1;
        problemFrequencies[problemFilename] = currentFreq / 5;
        userRank = Math.min(1, userRank + 0.001);
        // Skip overlay for correct moves, just load next tsumego
        localStorage.setItem('userRank', userRank.toString());
        localStorage.setItem('problemFrequencies', JSON.stringify(problemFrequencies));
        rankElement.textContent = `${getDescriptiveRank(userRank)} (${userRank.toFixed(3)})`;
        console.log(`Correct answer for ${problemFilename}. New frequency: ${problemFrequencies[problemFilename]}`);
        loadTsumego();
        return;
    } else {
        // Set frequency to 25 for incorrect answers
        problemFrequencies[problemFilename] = 25;
        userRank = Math.max(0, userRank - 0.004);
        localStorage.setItem('userRank', userRank.toString());
        localStorage.setItem('problemFrequencies', JSON.stringify(problemFrequencies));
        rankElement.textContent = `${getDescriptiveRank(userRank)} (${userRank.toFixed(3)})`;
        console.log(`Incorrect answer for ${problemFilename}. New frequency: ${problemFrequencies[problemFilename]}`);
        
        // For incorrect answers, reload the same tsumego with correct move highlighted
        console.log("Setting showCorrectMove flag to true in showResult");
        showCorrectMove = true;
        
        // Show the overlay message
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

        // After showing the message, reload the same tsumego
        setTimeout(() => {
            document.body.removeChild(overlay);
            loadTsumego(); // This will now reload the same tsumego because showCorrectMove is true
        }, 2000);
        
        // Return early to prevent the normal flow
        return;
    }
}

function updateBoard() {
    const position = kifuReader.game.position;
    board.removeAllObjects();
    for (let x = 0; x < position.size; x++) {
        for (let y = 0; y < position.size; y++) {
            const color = position.get(x, y);
            if (color !== 0) {
                board.addObject({ x: x, y: y, c: color });
            }
        }
    }

    const lastNode = kifuReader.node;
    if (lastNode.move) {
        board.addObject({x: lastNode.move.x, y: lastNode.move.y, type: 'CR'});
    }
    
    // If we're showing the correct move, add a GLOW marker
    if (showCorrectMove) {
        // Find the correct move in the branching node's children
        const branchingNode = kifuReader.node;
        const correctNode = branchingNode.children.find(child => 
            child.comment && child.comment.includes('Correct'));
        if (correctNode && correctNode.move) {
            console.log(`Showing correct move at (${correctNode.move.x}, ${correctNode.move.y})`);
            board.addObject({x: correctNode.move.x, y: correctNode.move.y, type: 'outline', stoneStyle:'NORMAL'});
        }
    }
}

function handleBoardClick(x, y) {
    // If we're showing the correct move, only allow clicks on the correct move
    if (showCorrectMove) {
        const branchingNode = kifuReader.node;
        const correctNode = branchingNode.children.find(child => 
            child.comment && child.comment.includes('Correct'));
        
        if (correctNode && correctNode.move && 
            correctNode.move.x === x && correctNode.move.y === y) {
            // User clicked on the correct move, proceed normally
            console.log("User clicked on the correct move");
            // Reset the flag since the user clicked the correct move
            showCorrectMove = false;
        } else {
            // User clicked on an incorrect move, show overlay
            console.log("User clicked on an incorrect move while showing correct move");
            showOverlay("Click on the correct move");
            return;
        }
    }
    
    if (!game || !kifuReader) return;

    const currentPlayer = game.turn;
    const isLegal = game.isValid(x, y, currentPlayer);

    if (isLegal) {
        const branchingNode = kifuReader.node;
        const correctMoveNode = branchingNode.children.find(child => child.move && child.move.x === x && child.move.y === y);

        if (correctMoveNode) {
            // Check if the comment contains the word "Correct"
            let comment = correctMoveNode.comment || '';
            const isCorrect = comment.includes('Correct');

            if (isCorrect) {
                // Correct move
                const moveIndex = branchingNode.children.indexOf(correctMoveNode);
                kifuReader.next(moveIndex);
                updateBoard();

                setTimeout(() => {
                    // Advance to the next node to get white's response
                    kifuReader.next();
                    updateBoard();

                    setTimeout(() => {
                        // Pass the current problem filename to showResult
                        showResult(comment, true, currentProblemFilename);
                    }, 1000);
                }, 1000);
            } else {
                // Incorrect move despite being a valid variation
                // Set flag to show correct move on reload
                console.log("Setting showCorrectMove flag to true");
                showCorrectMove = true;
                showResult('Incorrect move, try again.', false, currentProblemFilename);
            }
        } else {
            // Incorrect move
            // Set flag to show correct move on reload
            console.log("Setting showCorrectMove flag to true");
            showCorrectMove = true;
            showResult('Incorrect move, try again.', false, currentProblemFilename);
        }
    }
}

// Helper function to show an overlay message
function showOverlay(message) {
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
    }, 2000);
}

async function loadTsumego() {
    try {
        // First load the metadata to get the list of problems
        if (problems.length === 0) {
            const metadataResponse = await fetch('metadata.json');
            problems = await metadataResponse.json();
        }

        let selectedProblem;
        let wasShowingCorrectMove = false; // Track if we were showing the correct move
        
        // If we're showing the correct move, use the current problem
        if (showCorrectMove) {
            selectedProblem = problems.find(p => p.filename === currentProblemFilename);
            console.log(`Reloading same problem: ${currentProblemFilename}`);
            wasShowingCorrectMove = true; // Mark that we're showing the correct move
        } else {
            // Find the problem with the rank closest to the user's rank
            let minRankDiff = Infinity;

            for (const problem of problems) {
                const rankDiff = Math.abs(problem.rank - userRank);
                if (rankDiff < minRankDiff) {
                    minRankDiff = rankDiff;
                }
            }

            const closestProblems = problems.filter(p => Math.abs(p.rank - userRank) === minRankDiff);
            
            // Use weighted random selection based on frequencies
            let totalWeight = 0;
            const weightedProblems = closestProblems.map(problem => {
                const frequency = problemFrequencies[problem.filename] || 1;
                totalWeight += frequency; // Higher frequency = higher weight
                return { problem, weight: frequency };
            });

            // Select a problem based on weights
            let random = Math.random() * totalWeight;
            selectedProblem = null;
            
            for (const { problem, weight } of weightedProblems) {
                random -= weight;
                if (random <= 0) {
                    selectedProblem = problem;
                    break;
                }
            }
            
            // Fallback to random selection if weighting fails
            if (!selectedProblem) {
                selectedProblem = closestProblems[Math.floor(Math.random() * closestProblems.length)];
            }
        }

        // Store the current problem filename
        currentProblemFilename = selectedProblem.filename;

        // Load the selected problem
        const response = await fetch(`tsumegos/${selectedProblem.filename}`);
        const sgfContent = await response.text();

        const kifu = WGo.Kifu.fromSgf(sgfContent);
        kifuReader = new WGo.KifuReader(kifu);

        // Find the branching node
        while (kifuReader.node.children.length === 1) {
            kifuReader.next();
        }

        // Clear previous board if exists
        if (board) {
            board.removeEventListener('click', handleBoardClick);
            boardContainer.innerHTML = ''; // Clear the container
        }

        // Initialize WGo.Board
        const isLandscape = window.matchMedia("(orientation: landscape)").matches;
        const boardSize = isLandscape ? 400 : Math.min(window.innerWidth, window.innerHeight) * 0.95;
        
        // Clear any existing content and ensure container takes full space
        boardContainer.innerHTML = '';
        boardContainer.style.display = 'block';
        
        board = new WGo.Board(boardContainer, {
            size: kifu.size,
            background: WGo.DIR + "wood_1024.jpg",
            stoneHandler: WGo.Board.drawHandlers.REALISTIC,
            width: boardSize,
            height: boardSize,
        });

        updateBoard();

        // Store the game object for move validation
        game = kifuReader.game;

        // Add click listener to the board
        board.addEventListener('click', handleBoardClick);

        // Log current problem frequency
        const currentFrequency = problemFrequencies[currentProblemFilename] || 1;
        console.log(`Loaded problem: ${currentProblemFilename}, frequency: ${currentFrequency}`);

        // Update UI elements
        rankElement.textContent = `${getDescriptiveRank(userRank)} (${userRank.toFixed(3)})`;

        // Reset the flag after the tsumego is loaded
        if (wasShowingCorrectMove) {
            showCorrectMove = false; // Reset the flag
            console.log("Reset showCorrectMove flag to false");
        }

    } catch (error) {
        console.error('Error loading or parsing SGF:', error);
    }
}

loadTsumego();

// Add resize handler
window.addEventListener('resize', updateBoardSize);