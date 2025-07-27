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

// State variables
let board;
let game;
let kifuReader;
let problems = [];
let currentProblemFilename = null;
let showCorrectMove = false;

// Load state from localStorage or set defaults
let userRank = parseFloat(localStorage.getItem('userRank')) || 0; // Set initial rank to 0
let problemFrequencies = JSON.parse(localStorage.getItem('problemFrequencies')) || {};
// Initialize attempt history with all wins for new players
let attemptHistory = JSON.parse(localStorage.getItem('attemptHistory')) || 
    Array.from({length: 30}, () => true);

// --- Helper Functions ---

/**
 * Converts a numeric rank (0.0 to 1.0) to a descriptive Go rank (20k to 9d).
 * @param {number} numericRank - The rank value between 0.0 and 1.0.
 * @returns {string} The descriptive rank.
 */
function getDescriptiveRank(numericRank) {
    const totalRanks = 29; // 20 kyu ranks + 9 dan ranks
    const rankIndex = Math.floor(numericRank * totalRanks);
    if (rankIndex < 20) {
        return `${20 - rankIndex} kyu`;
    }
    return `${rankIndex - 19} dan`;
}

/**
 * Displays a message in a temporary overlay.
 * @param {string} message - The text to display.
 * @param {number} duration - How long to display the overlay in milliseconds.
 * @param {function} [callback] - An optional function to run after the overlay is removed.
 */
function displayOverlay(message, duration, callback) {
    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex',
        justifyContent: 'center', alignItems: 'center', zIndex: '1000'
    });

    const messageBox = document.createElement('div');
    messageBox.className = 'overlay-message-box';
    Object.assign(messageBox.style, {
        backgroundColor: 'rgba(0, 0, 0, 0.7)', color: 'white',
        padding: '20px', borderRadius: '10px'
    });
    messageBox.textContent = message;

    overlay.appendChild(messageBox);
    document.body.appendChild(overlay);

    setTimeout(() => {
        document.body.removeChild(overlay);
        if (callback) callback();
    }, duration);
}

/**
 * Updates the rank display on the UI.
 */
function updateRankDisplay() {
    // Calculate standard deviation
    const recentLosses = attemptHistory.filter(success => !success).length;
    const adjustmentFactor = (1 + Math.pow(recentLosses - 6, 2)) * 0.5;
    rankElement.textContent = `${getDescriptiveRank(userRank)} (±${adjustmentFactor.toFixed(1)})`;
}

/**
 * Updates user rank and problem frequency, then saves to localStorage.
 * @param {boolean} isCorrect - Whether the user's answer was correct.
 */
function updateUserStats(isCorrect) {
    // Update attempt history (shift array and add new result)
    attemptHistory.shift();
    attemptHistory.push(isCorrect);
    
    // Calculate recent losses (failures) in the last 30 attempts
    const recentLosses = attemptHistory.filter(success => !success).length;
    
    // Apply rank adjustment formula
    const adjustmentFactor = 1 + Math.pow(recentLosses - 6, 2);
    
    if (isCorrect) {
        const currentFreq = problemFrequencies[currentProblemFilename] || 1;
        problemFrequencies[currentProblemFilename] = Math.max(0.001, currentFreq / 5);
        userRank = Math.min(1, userRank + ((1/(5*3*29)) * adjustmentFactor));
    } else {
        problemFrequencies[currentProblemFilename] = 25;
        userRank = Math.max(0, userRank - ((1/(3*29)) * adjustmentFactor));
    }

    // Save to localStorage
    localStorage.setItem('userRank', userRank.toString());
    localStorage.setItem('problemFrequencies', JSON.stringify(problemFrequencies));
    localStorage.setItem('attemptHistory', JSON.stringify(attemptHistory));
    
    updateRankDisplay();

    const outcome = isCorrect ? "Correct" : "Incorrect";
    console.log(`${outcome} answer for ${currentProblemFilename}. New frequency: ${problemFrequencies[currentProblemFilename]}`);
}

// --- Core Application Logic ---

/**
 * Handles the logic for a correct user move.
 */
function handleCorrectAnswer() {
    updateUserStats(true);
    // Directly load the next tsumego without an overlay.
    loadTsumego();
}

/**
 * Handles the logic for an incorrect user move.
 */
function handleIncorrectAnswer() {
    updateUserStats(false);
    showCorrectMove = true;
    // Show an overlay, and after it disappears, reload the tsumego.
    displayOverlay('Incorrect move, try again.', 2000, () => {
        loadTsumego();
    });
}

/**
 * Handles user clicks on the Go board.
 * @param {number} x - The x-coordinate of the click.
 * @param {number} y - The y-coordinate of the click.
 */
function handleBoardClick(x, y) {
    if (!game || !kifuReader) return;

    if (showCorrectMove) {
        const correctNode = kifuReader.node.children.find(child => child.comment?.includes('Correct'));
        if (correctNode?.move && correctNode.move.x === x && correctNode.move.y === y) {
            showCorrectMove = false; // Allow play to proceed
        } else {
            displayOverlay("Click on the highlighted correct move.", 1500);
            return;
        }
    }

    if (game.isValid(x, y, game.turn)) {
        const moveNode = kifuReader.node.children.find(child => child.move && child.move.x === x && child.move.y === y);
        if (moveNode?.comment?.includes('Correct')) {
            const moveIndex = kifuReader.node.children.indexOf(moveNode);
            kifuReader.next(moveIndex);
            updateBoard();

            setTimeout(() => {
                kifuReader.next(); // Opponent's response
                updateBoard();
                setTimeout(() => handleCorrectAnswer(), 1000);
            }, 1000);
        } else {
            handleIncorrectAnswer();
        }
    }
}

// --- Tsumego Loading and Board Rendering ---

/**
 * Selects the next tsumego problem based on user rank and past performance.
 * @returns {object} The selected problem's metadata.
 */
function selectProblem() {
    // If we need to show the correct move, stick to the current problem.
    if (showCorrectMove && currentProblemFilename) {
        return problems.find(p => p.filename === currentProblemFilename);
    }

    // Find problems with the rank closest to the user's rank.
    let minRankDiff = Infinity;
    problems.forEach(p => minRankDiff = Math.min(minRankDiff, Math.abs(p.rank - userRank)));
    const closestProblems = problems.filter(p => Math.abs(p.rank - userRank) === minRankDiff);

    // Use weighted random selection based on problem frequencies.
    const weightedProblems = closestProblems.map(problem => ({
        problem,
        weight: problemFrequencies[problem.filename] || 1
    }));
    const totalWeight = weightedProblems.reduce((sum, p) => sum + p.weight, 0);

    let random = Math.random() * totalWeight;
    for (const { problem, weight } of weightedProblems) {
        random -= weight;
        if (random <= 0) return problem;
    }

    // Fallback to simple random selection if weighting fails.
    return closestProblems[Math.floor(Math.random() * closestProblems.length)];
}

/**
 * Initializes or re-initializes the WGo board.
 * @param {number} size - The size of the board (e.g., 19 for a 19x19 board).
 */
function initializeBoard(size) {
    if (board) {
        board.removeEventListener('click', handleBoardClick);
    }
    boardContainer.innerHTML = ''; // Clear the container
    boardContainer.style.display = 'block';

    board = new WGo.Board(boardContainer, {
        size: size,
        background: WGo.DIR + "wood_1024.jpg",
        stoneHandler: WGo.Board.drawHandlers.REALISTIC,
        width: 400, // Default size, will be updated immediately
        height: 400,
    });

    board.addEventListener('click', handleBoardClick);
    
    // Update board size based on current orientation
    updateBoardSize();
}

/**
 * Draws a marker on the board to indicate the correct move.
 */
function drawCorrectMoveMarker() {
    const correctNode = kifuReader.node.children.find(child => child.comment?.includes('Correct'));
    if (correctNode?.move) {
        console.log(`Highlighting correct move at (${correctNode.move.x}, ${correctNode.move.y})`);
        board.addObject({
            x: correctNode.move.x,
            y: correctNode.move.y,
            type: 'outline',
            stoneStyle: 'NORMAL'
        });
    }
}

/**
 * Redraws the board with the current position, moves, and markers.
 */
function updateBoard() {
    if (!board || !kifuReader) return;

    const position = kifuReader.game.position;
    board.removeAllObjects();

    // Draw stones
    for (let x = 0; x < position.size; x++) {
        for (let y = 0; y < position.size; y++) {
            if (position.get(x, y) !== 0) {
                board.addObject({ x, y, c: position.get(x, y) });
            }
        }
    }

    // Mark the last move
    if (kifuReader.node.move) {
        board.addObject({ x: kifuReader.node.move.x, y: kifuReader.node.move.y, type: 'CR' });
    }

    // Highlight the correct move if needed
    if (showCorrectMove) {
        drawCorrectMoveMarker();
    }
}

/**
 * Adjusts the board's dimensions on window resize or orientation change.
 */
function updateBoardSize() {
    if (!board) return;
    const isLandscape = window.matchMedia("(orientation: landscape)").matches;
    // For landscape, use 90% of screen height, for portrait use full width
    const boardSize = isLandscape ? window.innerHeight * 0.9 : Math.min(window.innerWidth, window.innerHeight);
    board.setWidth(boardSize);
    board.setHeight(boardSize);
}

/**
 * Fetches an SGF file, parses it, and prepares the kifu reader.
 * @param {string} filename - The SGF file to load from the 'tsumegos' directory.
 */
async function fetchAndLoadSgf(filename) {
    const response = await fetch(`tsumegos/${filename}`);
    const sgfContent = await response.text();
    const kifu = WGo.Kifu.fromSgf(sgfContent);

    kifuReader = new WGo.KifuReader(kifu);

    // Advance to the first branching node (where the user needs to move).
    while (kifuReader.node.children.length === 1) {
        kifuReader.next();
    }

    game = kifuReader.game;
    return kifu;
}

/**
 * Main function to load and display a tsumego problem.
 */
async function loadTsumego() {
    try {
        // Ensure problem metadata is loaded.
        if (problems.length === 0) {
            const metadataResponse = await fetch('metadata.json');
            problems = await metadataResponse.json();
        }

        const wasShowingCorrectMove = showCorrectMove;

        const selectedProblem = selectProblem();
        currentProblemFilename = selectedProblem.filename;

        const kifu = await fetchAndLoadSgf(currentProblemFilename);

        initializeBoard(kifu.size);
        updateBoard();
        updateRankDisplay();

        const currentFrequency = problemFrequencies[currentProblemFilename] || 1;
        console.log(`Loaded problem: ${currentProblemFilename}, rank: ${selectedProblem.rank.toFixed(3)}, frequency: ${currentFrequency}`);

        // Reset the flag only after the problem has been reloaded with the hint.
        if (wasShowingCorrectMove) {
            showCorrectMove = false;
        }

    } catch (error) {
        console.error('Error loading or parsing SGF:', error);
        displayOverlay('Error loading problem. Please refresh.', 5000);
    }
}

// --- Initial Load ---

loadTsumego();
updateRankDisplay(); // Initialize performance display
window.addEventListener('resize', updateBoardSize);
window.addEventListener('orientationchange', updateBoardSize);