const boardContainer = document.getElementById('board-container');

// App version - last modified date and time
const APP_VERSION = "2025-07-28 21:22";

console.log(`App Version: ${APP_VERSION}`);

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
let zipMetadata = []; // Metadata about zip files (filename, min_rank, max_rank)
let loadedZips = {}; // Track which zip files have been loaded {zipFilename: problemsArray}
let allProblems = []; // Flattened list of all problems from loaded zips
let currentProblemFilename = null;
let previousProblemFilename = localStorage.getItem('previousTsumego'); // Store previous tsumego filename
let currentSGF = null; // Store current tsumego SGF content
let previousSGF = localStorage.getItem('previousTsumegoContent'); // Store previous tsumego SGF content for download
let showCorrectMove = false;
let acceptClicks = false; // Track when we should accept clicks

// Load state from localStorage or set defaults
let userRank = parseFloat(localStorage.getItem('userRank')) || 0; // Set initial rank to 0
let problemFrequencies = JSON.parse(localStorage.getItem('problemFrequencies')) || {};
// Initialize momentum
let momentum = parseFloat(localStorage.getItem('momentum')) || 30;

// --- Helper Functions ---

/**
 * Converts a numeric rank (0.0 to 1.0) to a descriptive Go rank (20k to 9d).
 * @param {number} numericRank - The rank value between 0.0 and 1.0.
 * @returns {string} The descriptive rank in the appropriate language.
 */
function getDescriptiveRank(numericRank) {
//    numericRank = Math.pow(numericRank,0.4)
    const totalRanks = 29; // 20 kyu ranks + 9 dan ranks
    const rankIndex = Math.floor(numericRank * totalRanks);
    
    // Get browser language, default to 'en'
    const lang = navigator.language || 'en';
    
    // Define translations for kyu and dan
    let kyuTerm, danTerm, komiTerm;
    
    if (lang.startsWith('zh')) {
        // Chinese
        kyuTerm = '级';
        danTerm = '段';
        komiTerm = '贴目';
    } else if (lang.startsWith('ja')) {
        // Japanese
        kyuTerm = '級';
        danTerm = '段';
        komiTerm = 'コミ';
    } else if (lang.startsWith('ko')) {
        // Korean
        kyuTerm = '급';
        danTerm = '단';
        komiTerm = '덤';
    } else {
        // Default (English and others)
        kyuTerm = 'kyu';
        danTerm = 'dan';
        komiTerm = 'komi';
    }
    
    // Update the komi display with the translated term
    const komiElement = document.getElementById('komi');
    if (komiElement) {
        komiElement.textContent = `7.5 ${komiTerm}`;
    }
    
    if (rankIndex < 20) {
        return `${20 - rankIndex} ${kyuTerm}`;
    }
    return `${rankIndex - 19} ${danTerm}`;
}

/**
 * Displays a temporary blood red overlay that fades out.
 * @param {string} message - The text to display (not used in new implementation).
 * @param {number} duration - How long to display the overlay in milliseconds.
 * @param {function} [callback] - An optional function to run after the overlay is removed.
 */
function displayOverlay(message, duration, callback) {
    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: 'rgba(139, 0, 0, 1)', // Blood red completely opaque
        display: 'flex',
        justifyContent: 'center', alignItems: 'center', zIndex: '1000',
        opacity: '1',
        transition: 'opacity 3s ease-out'
    });

    document.body.appendChild(overlay);

    // Start fade-out after a short delay
    setTimeout(() => {
        overlay.style.opacity = '0';
    }, 50);

    // Remove overlay after fade-out completes and call callback
    setTimeout(() => {
        if (overlay.parentNode) {
            document.body.removeChild(overlay);
        }
        if (callback) callback();
    }, 3000);
}

/**
 * Updates the rank display on the UI.
 */
function updateRankDisplay() {
    // Calculate stddev from momentum
    const stddev = momentum / 2;
    
    // Update the rank element inside the board container
    const rankElement = document.getElementById('rank-display');
    if (rankElement) {
        rankElement.textContent = `${getDescriptiveRank(userRank)} (±${stddev.toFixed(1)})`;
    }
    
    // Update the previous tsumego link text based on language
    const prevTsumegoLink = document.getElementById('prev-tsumego');
    if (prevTsumegoLink) {
        // Get browser language, default to 'en'
        const lang = navigator.language || 'en';
        
        // Define translations
        let prevTsumegoText;
        if (lang.startsWith('zh')) {
            prevTsumegoText = '上一题';
        } else if (lang.startsWith('ja')) {
            prevTsumegoText = '前の詰碁';
        } else if (lang.startsWith('ko')) {
            prevTsumegoText = '이전 문제';
        } else {
            // Default (English and others)
            prevTsumegoText = 'Previous Tsumego';
        }
        
        prevTsumegoLink.textContent = prevTsumegoText;
    }
}




/**
 * Updates user rank and problem frequency, then saves to localStorage.
 * @param {boolean} isCorrect - Whether the user's answer was correct.
 * @param {boolean} [hintWasShown=false] - Whether a hint was shown before the answer.
 */
function updateUserStats(isCorrect, hintWasShown = false) {
    // If a hint was shown, do not update stats.
    if (hintWasShown) {
        return;
    }

    const oldRank = userRank;
    
    if (isCorrect) {
        const currentFreq = problemFrequencies[currentProblemFilename] || 1;
        problemFrequencies[currentProblemFilename] = currentFreq / 5;
        userRank = Math.min(1, userRank + (momentum / 300));
    } else {
        problemFrequencies[currentProblemFilename] = 25;
        momentum = (momentum + 1) / 2;
        userRank = Math.max(0, userRank - 4 * (momentum / 300));
    }

    // Log rank update
    console.log(`Rank updated from ${oldRank.toFixed(4)} to ${userRank.toFixed(4)}`);

    // Save to localStorage
    localStorage.setItem('userRank', userRank.toString());
    localStorage.setItem('problemFrequencies', JSON.stringify(problemFrequencies));
    localStorage.setItem('momentum', momentum.toString());
    
    updateRankDisplay();
}

// --- Core Application Logic ---

/**
 * Handles the logic for a correct user move.
 * @param {boolean} [hintWasShown=false] - Whether a hint was shown before the answer.
 */
function handleCorrectAnswer(hintWasShown = false) {
    updateUserStats(true, hintWasShown);
    // Directly load the next tsumego without an overlay.
    loadTsumego().catch(error => {
        console.error("Error loading next tsumego:", error);
    });
}

/**
 * Handles the logic for an incorrect user move.
 * @param {boolean} [hintWasShown=false] - Whether a hint was shown before the answer.
 */
function handleIncorrectAnswer(hintWasShown = false) {
    updateUserStats(false, hintWasShown);
    displayOverlay('', 3000, () => {
        showCorrectMove = true;
        loadTsumego().catch(error => {
            console.error("Error loading next tsumego:", error);
        });
    });
}

/**
 * Handles user clicks on the Go board.
 * @param {number} x - The x-coordinate of the click.
 * @param {number} y - The y-coordinate of the click.
 */
function handleBoardClick(x, y) {
    // Only process clicks when we're expecting them
    if (!acceptClicks) return;
    
    if (!game || !kifuReader) return;

    const hintWasShown = showCorrectMove;

    if (showCorrectMove) {
        const correctNode = kifuReader.node.children.find(child => child.comment?.includes('Correct'));
        if (correctNode?.move && correctNode.move.x === x && correctNode.move.y === y) {
            showCorrectMove = false; // Allow play to proceed
        } else {
            displayOverlay("", 1500);
            return;
        }
    }

    if (game.isValid(x, y, game.turn)) {
        // Disable clicks immediately after a valid move is played
        acceptClicks = false;
        
        // Find the move node that matches the user's click
        const moveNode = kifuReader.node.children.find(child => 
            child.move && child.move.x === x && child.move.y === y);
        
        if (moveNode) {
            // Check if this is the correct move
            const isCorrect = moveNode.comment?.includes('Correct') || false;
            
            // Play the user's move
            const moveIndex = kifuReader.node.children.indexOf(moveNode);
            kifuReader.next(moveIndex);
            updateBoard();

            // After 1 second, show the follow-up move
            setTimeout(() => {
                // Check if there's a follow-up move in this variation
                if (kifuReader.node.children.length > 0) {
                    kifuReader.next(); // Play the follow-up move
                    updateBoard();
                    
                    // After another 1 second, either show overlay or proceed to next tsumego
                    setTimeout(() => {
                        if (isCorrect) {
                            handleCorrectAnswer(hintWasShown);
                        } else {
                            handleIncorrectAnswer(hintWasShown);
                        }
                    }, 1000);
                } else {
                    // No follow-up move, proceed immediately
                    if (isCorrect) {
                        handleCorrectAnswer(hintWasShown);
                    } else {
                        handleIncorrectAnswer(hintWasShown);
                    }
                }
            }, 1000);
        }
    }
}

// --- Tsumego Loading and Board Rendering ---

// The binarySearchClosest function is no longer used with the new lazy loading system
// but kept for reference in case it's needed in the future.

/**
 * Selects the next tsumego problem based on user rank and past performance.
 * @returns {object|null} The selected problem's metadata, or null if no problems are available.
 */
function selectProblem() {
    // If we need to show the correct move, stick to the current problem.
    if (showCorrectMove && currentProblemFilename) {
        const problem = allProblems.find(p => p.filename === currentProblemFilename);
        return problem;
    }
    
    // If no problems are loaded, return null to trigger loading
    if (allProblems.length === 0) {
        return null;
    }

    // Create a list of available problems, excluding current and previous
    let availableProblems = allProblems.filter(problem =>
        problem.filename !== currentProblemFilename && 
        problem.filename !== previousProblemFilename
    );

    // If we've excluded all problems, use the full list (but still exclude current/previous)
    if (availableProblems.length === 0) {
        availableProblems = allProblems.filter(problem => 
            problem.filename !== currentProblemFilename && 
            problem.filename !== previousProblemFilename
        );
    }

    // If we still have no problems, fallback to all problems
    if (availableProblems.length === 0) {
        availableProblems = allProblems;
    }

    // Sort available problems by rank distance to user's rank
    availableProblems.sort((a, b) => {
        const diffA = Math.abs(a.rank - userRank);
        const diffB = Math.abs(b.rank - userRank);
        return diffA - diffB;
    });

    // Select from the closest 200 problems (or all if less than 200)
    const candidateProblems = availableProblems.slice(0, 200);

    // Use weighted random selection based on problem frequencies
    const weightedProblems = candidateProblems.map(problem => ({
        problem,
        weight: problemFrequencies[problem.filename] || 1
    }));
    
    const totalWeight = weightedProblems.reduce((sum, p) => sum + p.weight, 0);

    // Log the selection probability for the current problem (if it exists in candidates)
    if (currentProblemFilename) {
        const currentProblemEntry = weightedProblems.find(wp => wp.problem.filename === currentProblemFilename);
        if (currentProblemEntry) {
            const probability = totalWeight > 0 ? (currentProblemEntry.weight / totalWeight) * 100 : 0;
            console.log(`Current problem selection probability: ${probability.toFixed(2)}% (weight: ${currentProblemEntry.weight}, total: ${totalWeight})`);
        }
    }

    // If total weight is zero, fallback to uniform random selection
    if (totalWeight <= 0) {
        return candidateProblems[Math.floor(Math.random() * candidateProblems.length)];
    }

    let random = Math.random() * totalWeight;
    
    for (const { problem, weight } of weightedProblems) {
        random -= weight;
        if (random <= 0) {
            return problem;
        }
    }

    // Fallback to simple random selection if weighting fails
    return candidateProblems[Math.floor(Math.random() * candidateProblems.length)];
}

/**
 * Initializes or re-initializes the WGo board.
 * @param {number} size - The size of the board (e.g., 19 for a 19x19 board).
 */
function initializeBoard(size) {
    if (board) {
        board.removeEventListener('click', handleBoardClick);
        board = null; // Explicitly nullify the board reference
    }
    
    const boardWrapper = document.createElement('div');
    boardContainer.innerHTML = '';
    boardContainer.appendChild(boardWrapper);

    board = new WGo.Board(boardWrapper, {
        size: size,
        background: WGo.DIR + "wood_1024.jpg",
        stoneHandler: WGo.Board.drawHandlers.REALISTIC,
        width: 400, // Default size, will be updated immediately
        height: 400,
    });

    // Explicitly verify and set the board size
    if (board.size !== size) {
        board.size = size;
    }

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
    if (!board || !kifuReader) {
        return;
    }

    const position = kifuReader.game.position;
    
    // Verify board size matches position size
    if (board.size !== position.size) {
        // Reinitialize the board with correct size
        initializeBoard(position.size);
    }
    
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

    const appContainer = document.getElementById('app-container');
    const boardContainer = document.getElementById('board-container');
    const infoContainer = document.getElementById('info-container');
    const w = window.innerWidth;
    const h = window.innerHeight;

    infoSize = 1/13

    // Calculate the available space for the board
    const availableWidth = w;
    const availableHeight = h / (1+infoSize)

    // Determine the size of the board based on the available space
    const boardSize = Math.min(availableWidth, availableHeight);

    // Set the size of the board container
    boardContainer.style.width = boardSize + 'px';
    boardContainer.style.height = boardSize + 'px';

    // Set the size of the board
    board.setWidth(boardSize);
    board.setHeight(boardSize);

    // Style the info container
    if (infoContainer) {
        infoContainer.style.width = boardSize + 'px';
        infoContainer.style.height = (boardSize * infoSize) + 'px';
    }

    // Update font size for rank, komi, and previous tsumego link
    const rankDisplay = document.getElementById('rank-display');
    const komiDisplay = document.getElementById('komi');
    const prevTsumegoLink = document.getElementById('prev-tsumego');
    if (rankDisplay && komiDisplay && prevTsumegoLink) {
        const fontSize = boardSize / 25;
        rankDisplay.style.fontSize = fontSize + 'px';
        komiDisplay.style.fontSize = fontSize + 'px';
        prevTsumegoLink.style.fontSize = fontSize + 'px';
    }
}

// Cache for loaded zip files to avoid reloading
const zipCache = {};

/**
 * Fetches an SGF file from a zip archive, parses it, and prepares the kifu reader.
 * @param {string} filename - The SGF file to load.
 * @param {number} group - The group number indicating which zip file to load from.
 */
async function fetchAndLoadSgf(filename, group) {
    const zipFilename = `tsumegos${group}.zip`;
    
    // Ensure the zip is loaded in our tracking system
    if (!loadedZips[zipFilename]) {
        await loadZipProblems(zipFilename);
    }
    
    // Get the zip from cache (should be there after loadZipProblems)
    const zip = zipCache[group];
    if (!zip) {
        throw new Error(`Zip file ${zipFilename} not found in cache`);
    }
    
    const file = zip.file(filename);
    
    if (!file) {
        throw new Error(`File ${filename} not found in tsumegos/${zipFilename}`);
    }
    
    const sgfContent = await file.async('text');
    const kifu = WGo.Kifu.fromSgf(sgfContent);
    kifuReader = new WGo.KifuReader(kifu);

    // Advance to the first branching node (where the user needs to move).
    while (kifuReader.node.children.length === 1) {
        kifuReader.next();
    }

    game = kifuReader.game;
    return { kifu, sgfContent }; // Return both kifu and sgfContent
}

/**
 * Loads metadata about zip files.
 */
async function loadZipMetadata() {
    if (zipMetadata.length === 0) {
        const metadataResponse = await fetch('metadata.json');
        zipMetadata = await metadataResponse.json();
    }
}

/**
 * Loads problems from a specific zip file.
 * @param {string} zipFilename - The name of the zip file to load.
 */
async function loadZipProblems(zipFilename) {
    if (loadedZips[zipFilename]) {
        return loadedZips[zipFilename];
    }

    // Extract group number from filename (e.g., "tsumegos0.zip" -> 0)
    const groupMatch = zipFilename.match(/tsumegos(\d+)\.zip/);
    if (!groupMatch) {
        throw new Error(`Invalid zip filename format: ${zipFilename}`);
    }
    const group = parseInt(groupMatch[1]);

    // Fetch and parse the metadata.json inside the zip
    const zipResponse = await fetch(`tsumegos/${zipFilename}`);
    const zipData = await zipResponse.arrayBuffer();
    const zip = new JSZip();
    await zip.loadAsync(zipData);
    
    const metadataFile = zip.file('metadata.json');
    if (!metadataFile) {
        throw new Error(`metadata.json not found in ${zipFilename}`);
    }
    
    const metadataContent = await metadataFile.async('text');
    const problems = JSON.parse(metadataContent);
    
    // Add group information to each problem
    problems.forEach(problem => {
        problem.group = group;
    });
    
    loadedZips[zipFilename] = problems;
    allProblems = allProblems.concat(problems);
    
    // Also cache the loaded zip for future use to avoid redundant fetches
    zipCache[group] = zip;
    
    return problems;
}

/**
 * Background loads a single zip file.
 * @param {string} zipFilename - The name of the zip file to load.
 */
async function backgroundLoadZip(zipFilename) {
    if (!loadedZips[zipFilename]) {
        try {
            await loadZipProblems(zipFilename);
        } catch (error) {
            console.error(`Failed to background load ${zipFilename}:`, error);
        }
    }
}

/**
 * Background loads zip files that might be needed soon.
 * @param {Array} zipFilenames - Array of zip filenames to load in background.
 * @deprecated Use backgroundLoadZip instead for single file loading
 */
function backgroundLoadZips(zipFilenames) {
    if (zipFilenames.length > 0) {
        backgroundLoadZip(zipFilenames[0]);
    }
}

/**
 * Main function to load and display a tsumego problem.
 * @param {object} [selectedProblem] - Optional specific problem to load.
 */
async function loadTsumego(selectedProblem = null) {
    try {
        // Ensure zip metadata is loaded
        await loadZipMetadata();

        const wasShowingCorrectMove = showCorrectMove;

        // On startup, check if we have a saved tsumego
        if (!currentProblemFilename && !selectedProblem) {
            const savedTsumego = localStorage.getItem('currentTsumego');
            const savedTsumegoContent = localStorage.getItem('currentTsumegoContent');
            
            if (savedTsumego && savedTsumegoContent) {
                // Load the saved tsumego content directly
                const kifu = WGo.Kifu.fromSgf(savedTsumegoContent);
                kifuReader = new WGo.KifuReader(kifu);
                
                // Advance to the first branching node (where the user needs to move).
                while (kifuReader.node.children.length === 1) {
                    kifuReader.next();
                }
                
                game = kifuReader.game;
                
                // Set current values
                currentProblemFilename = savedTsumego;
                currentSGF = savedTsumegoContent;
                
                // Process the loaded tsumego
                initializeBoard(kifu.size);
                updateBoard();
                updateRankDisplay();
                
                // Update the previous tsumego link state
                const prevTsumegoLink = document.getElementById('prev-tsumego');
                if (prevTsumegoLink) {
                    // Enable the link if we have previous tsumego data
                    if (previousSGF) {
                        prevTsumegoLink.classList.remove('disabled');
                    }
                    // If no previous tsumego, it's already disabled by default
                }
                
                // Reset showCorrectMove
                showCorrectMove = false;
                
                // Enable clicks once the board is ready
                acceptClicks = true;
                
                // Log tsumego metadata
                const problem = allProblems.find(p => p.filename === currentProblemFilename);
                if (problem) {
                    console.log("Tsumego metadata:");
                    for (const [key, value] of Object.entries(problem)) {
                        console.log(`  ${key}: ${value}`);
                    }
                }
                
                // Background load zip files that might contain problems closer to user's rank
                backgroundLoadCloserZips();
                
                return;
            }
        }

        // If no specific problem was provided, select one
        if (!selectedProblem) {
            selectedProblem = selectProblem();
        }
        
        // If no problems are available, we need to load at least one zip
        if (!selectedProblem) {
            // Find the zip file closest to user's rank
            let closestZip = null;
            let minRankDiff = Infinity;
            
            zipMetadata.forEach(zip => {
                const rankDiff = Math.abs((zip.min_rank + zip.max_rank) / 2 - userRank);
                if (rankDiff < minRankDiff) {
                    minRankDiff = rankDiff;
                    closestZip = zip;
                }
            });
            
            if (closestZip) {
                await loadZipProblems(closestZip.filename);
                // Try selecting a problem again
                const retrySelectedProblem = selectProblem();
                if (!retrySelectedProblem) {
                    throw new Error("Failed to load any tsumego problems");
                }
                selectedProblem = retrySelectedProblem;
            } else {
                throw new Error("No zip files found");
            }
        }
        
        // Update previousSGF and previousProblemFilename with current values before loading new tsumego
        previousSGF = currentSGF;
        previousProblemFilename = currentProblemFilename;
        
        currentProblemFilename = selectedProblem.filename;
        
        const { kifu, sgfContent } = await fetchAndLoadSgf(currentProblemFilename, selectedProblem.group);
        currentSGF = sgfContent; // Store the current SGF content
        
        // Save current tsumego filename and content to localStorage
        // Also save previous tsumego info
        localStorage.setItem('currentTsumego', currentProblemFilename);
        localStorage.setItem('currentTsumegoContent', currentSGF);
        if (previousProblemFilename) {
            localStorage.setItem('previousTsumego', previousProblemFilename);
            localStorage.setItem('previousTsumegoContent', previousSGF);
        }
        
        // Log all available metadata fields for the tsumego in a readable format
        console.log("Tsumego metadata:");
        for (const [key, value] of Object.entries(selectedProblem)) {
            console.log(`  ${key}: ${value}`);
        }

        initializeBoard(kifu.size);
        updateBoard();
        updateRankDisplay();
        
        // Update the previous tsumego link state
        const prevTsumegoLink = document.getElementById('prev-tsumego');
        if (prevTsumegoLink) {
            if (previousSGF) {
                prevTsumegoLink.classList.remove('disabled');
            } else {
                prevTsumegoLink.classList.add('disabled');
            }
        }

        // Only reset showCorrectMove if we're not showing the correct move
        if (!wasShowingCorrectMove) {
            showCorrectMove = false;
        }

        // Enable clicks once the board is ready
        acceptClicks = true;
        
        // Background load zip files that might contain problems closer to user's rank
        backgroundLoadCloserZips();

    } catch (error) {
        console.error("Error loading tsumego:", error);
        displayOverlay('Error loading problem.', 3000);
    }
}

/**
 * Background loads the zip file with rank range closest to the user's current rank.
 */
function backgroundLoadCloserZips() {

    // Find the zip file with rank range closest to user's current rank
    let closestZip = null;
    let minRankDiff = Infinity;
    
    zipMetadata.forEach(zip => {
        // Calculate the distance from user's rank to the zip's min and max ranks
        const diffToMin = Math.abs(userRank - zip.min_rank);
        const diffToMax = Math.abs(userRank - zip.max_rank);
        const minDiff = Math.min(diffToMin, diffToMax);
        
        if (minDiff < minRankDiff) {
            minRankDiff = minDiff;
            closestZip = zip;
        }
    });
    
    if (closestZip) {

        // If the closest zip is not already loaded, background load it
        if (!loadedZips[closestZip.filename]) {
            backgroundLoadZip(closestZip.filename);
        }
    } else {
        console.log("No zip files found in metadata");
    }
}



// --- Initial Load ---

// Initialize performance display
updateRankDisplay();

// Initialize previous tsumego link and attach event listener
document.addEventListener('DOMContentLoaded', () => {
    const prevTsumegoLink = document.getElementById('prev-tsumego');
    if (prevTsumegoLink) {
        // Set initial state based on whether we have previous tsumego data
        if (previousSGF && previousProblemFilename) {
            prevTsumegoLink.classList.remove('disabled');
        } else {
            prevTsumegoLink.classList.add('disabled');
        }
        
        // Attach event listener for previous tsumego download
        prevTsumegoLink.addEventListener('click', (e) => {
            e.preventDefault();
            if (previousSGF && previousProblemFilename) {
                // Create a blob and download link
                const blob = new Blob([previousSGF], { type: 'application/x-go-sgf' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = previousProblemFilename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            } else {
                console.log("No previous tsumego available for download");
            }
        });
    }
});

// Load initial tsumego
loadTsumego().catch(error => {
    console.error("Failed to load initial tsumego:", error);
});

window.addEventListener('resize', updateBoardSize);
window.addEventListener('orientationchange', updateBoardSize);