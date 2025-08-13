// --- START OF REFACTORED FILE app.js ---

const boardContainer = document.getElementById('board-container');

// App version - last modified date and time (automatically updated by deployment script)
const APP_VERSION = "2025-08-12 19:50";
console.log(`App Version: ${APP_VERSION}`);

// Stone placement sound
let stoneSound = null;
// Fail sounds
let failSound1 = null;
let failSound2 = null;
// Success sound
let successSound = null;

// Initialize audio
function initAudio() {
    try {
        stoneSound = new Audio('sound/stone.wav');
        stoneSound.preload = 'auto';
        
        failSound1 = new Audio('sound/fail1.wav');
        failSound1.preload = 'auto';
        
        failSound2 = new Audio('sound/fail2.wav');
        failSound2.preload = 'auto';
        
        successSound = new Audio('sound/success.wav');
        successSound.preload = 'auto';
    } catch (e) {
        console.log('Audio initialization failed:', e);
    }
}

// Play stone placement sound
function playStoneSound() {
    if (stoneSound) {
        try {
            // Reset to beginning and play
            stoneSound.currentTime = 0;
            stoneSound.play().catch(e => console.log('Audio play failed:', e));
        } catch (e) {
            console.log('Audio play failed:', e);
        }
    }
}

// Play random fail sound
function playRandomFailSound() {
    if (!failSound1 || !failSound2) return;
    
    try {
        const sounds = [failSound1, failSound2];
        const randomSound = sounds[Math.floor(Math.random() * sounds.length)];
        
        // Reset to beginning and play
        randomSound.currentTime = 0;
        randomSound.play().catch(e => console.log('Fail sound play failed:', e));
    } catch (e) {
        console.log('Fail sound play failed:', e);
    }
}

// Play success sound
function playSuccessSound() {
    if (successSound) {
        try {
            // Reset to beginning and play
            successSound.currentTime = 0;
            successSound.play().catch(e => console.log('Success sound play failed:', e));
        } catch (e) {
            console.log('Success sound play failed:', e);
        }
    }
}

// Force service worker update
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(registrations => {
    registrations.forEach(reg => reg.update());
  });
}

// Add click handler for showing version
document.addEventListener('DOMContentLoaded', () => {
  const komiElement = document.getElementById('komi');
  if (komiElement) {
    komiElement.addEventListener('click', () => {
      // Show version briefly
      const versionDisplay = document.createElement('div');
      versionDisplay.textContent = `Version: ${APP_VERSION}`;
      versionDisplay.style.position = 'fixed';
      versionDisplay.style.top = '50%';
      versionDisplay.style.left = '50%';
      versionDisplay.style.transform = 'translate(-50%, -50%)';
      versionDisplay.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
      versionDisplay.style.color = 'white';
      versionDisplay.style.padding = '10px 20px';
      versionDisplay.style.borderRadius = '5px';
      versionDisplay.style.zIndex = '10000';
      versionDisplay.style.fontSize = '18px';
      versionDisplay.style.pointerEvents = 'none';
      
      document.body.appendChild(versionDisplay);
      
      // Remove after 2 seconds
      setTimeout(() => {
        if (versionDisplay.parentNode) {
          document.body.removeChild(versionDisplay);
        }
      }, 2000);
    });
  }
});

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
let userRank = parseFloat(localStorage.getItem('userRank')) || 0;
let problemFrequencies = JSON.parse(localStorage.getItem('problemFrequencies')) || {};
let momentum = parseFloat(localStorage.getItem('momentum')) || 30;

// Cache for loaded zip files to avoid redundant fetches
const zipCache = {};

// Cache name for tsumego files
const TSUMEGO_CACHE_NAME = 'tsumego-cache-v1';


// --- Helper Functions ---

/**
 * Converts a numeric rank (0.0 to 1.0) to a descriptive Go rank (20k to 9d).
 * @param {number} numericRank - The rank value between 0.0 and 1.0.
 * @returns {{rank: string, kyu: string, dan: string, komi: string}} An object with translated terms.
 */
function getDescriptiveRank(numericRank) {
    const totalRanks = 29; // 20 kyu ranks + 9 dan ranks
    const rankIndex = Math.floor(Math.pow(numericRank ,1.0) * totalRanks);
    const lang = navigator.language || 'en';

    let kyuTerm, danTerm, komiTerm;

    if (lang.startsWith('zh')) { // Chinese
        kyuTerm = '级'; danTerm = '段'; komiTerm = '贴目';
    } else if (lang.startsWith('ja')) { // Japanese
        kyuTerm = '級'; danTerm = '段'; komiTerm = 'コミ';
    } else if (lang.startsWith('ko')) { // Korean
        kyuTerm = '급'; danTerm = '단'; komiTerm = '덤';
    } else { // Default (English and others)
        kyuTerm = 'kyu'; danTerm = 'dan'; komiTerm = 'komi';
    }

    const rank = (rankIndex < 20) ? `${20 - rankIndex} ${kyuTerm}` : `${rankIndex - 19} ${danTerm}`;
    return { rank, kyuTerm, danTerm, komiTerm };
}

/**
 * Centralized function to update all UI elements related to rank, komi, and links.
 */
function updateAllUI() {
    const { rank, komiTerm } = getDescriptiveRank(userRank);
    const stddev = momentum / 2;

    // Update rank display
    const rankElement = document.getElementById('rank-display');
    if (rankElement) {
        rankElement.textContent = `${rank} (±${stddev.toFixed(1)})`;
    }

    // Update komi display
    const komiElement = document.getElementById('komi');
    if (komiElement) {
        komiElement.textContent = `7.5 ${komiTerm}`;
    }

    // Update "Previous" link text and state
    const prevTsumegoLink = document.getElementById('prev-tsumego');
    if (prevTsumegoLink) {
        const lang = navigator.language || 'en';
        let prevTsumegoText;
        if (lang.startsWith('zh')) { prevTsumegoText = '上一题'; }
        else if (lang.startsWith('ja')) { prevTsumegoText = '前の問題'; }
        else if (lang.startsWith('ko')) { prevTsumegoText = '이전 문제'; }
        else { prevTsumegoText = 'Previous'; }

        prevTsumegoLink.textContent = prevTsumegoText;

        if (previousSGF && previousProblemFilename) {
            prevTsumegoLink.classList.remove('disabled');
        } else {
            prevTsumegoLink.classList.add('disabled');
        }
    }
}


/**
 * Displays a temporary overlay that fades in to 90% opacity over 4 seconds.
 * Total duration is 5 seconds (1s wait + 4s fade in), then overlay is directly removed.
 * @param {string} message - The text to display (not used in new implementation).
 * @param {number} duration - How long to display the overlay in milliseconds (ignored for fixed duration).
 * @param {function} [callback] - An optional function to run after the overlay is removed.
 * @param {string} [color='rgba(139, 0, 0, 1)'] - The background color of the overlay.
 */
function displayOverlay(message, duration, callback, color = 'rgba(139, 0, 0, 1)') {
    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: color,
        display: 'flex', justifyContent: 'center', alignItems: 'center',
        zIndex: '1000', opacity: '0'
    });

    document.body.appendChild(overlay);

    // Start fade in after 1 second (to account for the 1s wait)
    setTimeout(() => {
        overlay.style.transition = 'opacity 4s linear';
        requestAnimationFrame(() => {
            requestAnimationFrame(() => { 
                overlay.style.opacity = '0.9'; 
                // Play random fail sound when overlay starts
                playRandomFailSound();
            });
        });
    }, 1000); // Start fade in after 1 second

    // Remove overlay directly after fade in completes (1s wait + 4s fade in)
    setTimeout(() => {
        if (overlay.parentNode) {
            document.body.removeChild(overlay);
        }
        if (callback) callback();
    }, 5000); // Remove overlay after 5 seconds total
}

/**
 * Displays a white overlay with fade in and fade out effects.
 * @param {number} duration - How long to display the overlay in milliseconds.
 * @param {function} [callback] - An optional function to run after the overlay is removed.
 */
function fadeOverlay(duration, callback) {
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: 'white',
        zIndex: '1000', opacity: '0', transition: 'opacity 1s ease-in-out'
    });

    document.body.appendChild(overlay);

    // Fade in and play success sound
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            overlay.style.opacity = '1';
            playSuccessSound(); // Play success sound when overlay starts fading in
        });
    });

    // Hold for duration, then call callback and fade out
    setTimeout(() => {
        // Call the callback (load next tsumego) when fully visible
        if (callback) callback();
        
        // Start fade out
        overlay.style.opacity = '0';
        
        // Remove overlay after fade out completes
        setTimeout(() => {
            if (overlay.parentNode) {
                document.body.removeChild(overlay);
            }
        }, 1000); // Match the 1s transition time
    }, duration);
}

/**
 * Updates user rank and problem frequency, then saves to localStorage.
 * @param {boolean} isCorrect - Whether the user's answer was correct.
 * @param {boolean} [hintWasShown=false] - Whether a hint was shown before the answer.
 */
function updateUserStats(isCorrect, hintWasShown = false) {
    if (hintWasShown) return;

    const oldRank = userRank;

    if (isCorrect) {
        const currentFreq = problemFrequencies[currentProblemFilename] || 1;
        problemFrequencies[currentProblemFilename] = currentFreq / 5;
        // Limit rank increase to half the distance to 1.0
        const maxIncrease = (1.0 - userRank) / 2;
        const intendedIncrease = momentum / 300;
        userRank = userRank + Math.min(intendedIncrease, maxIncrease);
    } else {
        problemFrequencies[currentProblemFilename] = 25;
        momentum = (momentum + 1) / 2;
        userRank = Math.max(0, userRank - 4 * (momentum / 300));
    }

    localStorage.setItem('userRank', userRank.toString());
    localStorage.setItem('problemFrequencies', JSON.stringify(problemFrequencies));
    localStorage.setItem('momentum', momentum.toString());

    updateAllUI();
}

// --- Core Application Logic ---

/**
 * Handles the logic for a correct user move.
 * @param {boolean} [hintWasShown=false] - Whether a hint was shown before the answer.
 */
function handleCorrectAnswer(hintWasShown = false) {
    updateUserStats(true, hintWasShown);
    showCorrectMove = false;
    
    // Create white overlay with fade in/out effect
    fadeOverlay(1000, () => {
        loadTsumego(true).catch(error => console.error("Error loading next tsumego:", error));
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
        loadTsumego(false).catch(error => console.error("Error loading next tsumego:", error));
    });
}

/**
 * Handles user clicks on the Go board.
 * Sequence: user move -> wait 1s -> opponent response (if any) -> wait 1s -> resolve.
 * Uses async/await with a delay helper and frame awaits to ensure paint before delays.
 * @param {number} x - The x-coordinate of the click.
 * @param {number} y - The y-coordinate of the click.
 */
async function handleBoardClick(x, y) {
    if (!acceptClicks || !game || !kifuReader) return;

    const delay = (ms) => new Promise(res => setTimeout(res, ms));
    const renderAndHold = async (ms) => {
        if (boardContainer) {
            // Force style/layout to flush any pending drawing
            window.getComputedStyle(boardContainer).getPropertyValue('width');
            void boardContainer.offsetHeight;
        }
        await Promise.resolve();
        await delay(ms);
    };

    const hintWasShown = showCorrectMove;

    if (showCorrectMove) {
        const correctNode = kifuReader.node.children.find(child => child.comment?.includes('Correct'));
        if (correctNode?.move && correctNode.move.x === x && correctNode.move.y === y) {
            showCorrectMove = false;
        } else {
            // For incorrect moves in hint mode, show the normal animation path but without rating update
            const moveNode = kifuReader.node.children.find(child => child.move && child.move.x === x && child.move.y === y);

            if (!moveNode) {
                displayOverlay("", 1500);
                return;
            }

            const isCorrect = moveNode.comment?.includes('Correct') || false;
            const moveIndex = kifuReader.node.children.indexOf(moveNode);

            // User move
            kifuReader.next(moveIndex);
            playStoneSound(); // Play stone sound for user move
            updateBoard();
            await renderAndHold(1000); // Show user move for 1s (after forcing render)

            // For incorrect moves, show opponent response and start red overlay simultaneously
            kifuReader.next();
            playStoneSound(); // Play stone sound for opponent response
            updateBoard();
            
            // Start the red overlay immediately (don't wait 1 second) but don't update ratings
            await new Promise((resolve) => {
                displayOverlay('', 5000, resolve);
            });
            showCorrectMove = true;
            loadTsumego(false).catch(error => console.error("Error loading next tsumego:", error));
            return;
        }
    }

    // Check if the move is valid first
    if (game.isValid(x, y, game.turn)) {
        acceptClicks = false;

        const moveNode = kifuReader.node.children.find(child => child.move && child.move.x === x && child.move.y === y);

        if (!moveNode) {
            acceptClicks = true;
            return;
        }

        const isCorrect = moveNode.comment?.includes('Correct') || false;
        const moveIndex = kifuReader.node.children.indexOf(moveNode);

        // User move
        kifuReader.next(moveIndex);
        playStoneSound(); // Play stone sound when user places a stone
        updateBoard();
        await renderAndHold(1000); // Show user move for 1s (after forcing render)

        // Resolve outcome sequentially
        if (isCorrect) {
            handleCorrectAnswer(hintWasShown);
        } else {
            // For incorrect moves, show opponent response and start red overlay simultaneously
            kifuReader.next();
            playStoneSound(); // Play stone sound for opponent's response
            updateBoard();
            
            // Start the red overlay immediately (don't wait 1 second)
            updateUserStats(false, hintWasShown);
            await new Promise((resolve) => {
                displayOverlay('', 5000, resolve);
            });
            showCorrectMove = true;
            loadTsumego(false).catch(error => console.error("Error loading next tsumego:", error));
        }
    } else {
        // Check if the move is illegal due to ko
        const koResult = game.play(x, y, game.turn, true); // noplay=true to only test
        if (koResult === 4) { // 4 indicates illegal due to ko
            // Mark the illegal ko move with a square
            board.addObject({ x, y, type: 'SQ' });
            
            // Also check adjacent squares for potential ko points
            const size = game.size;
            const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // up, down, left, right
            
            for (const [dx, dy] of directions) {
                const adjX = x + dx;
                const adjY = y + dy;
                
                // Check if adjacent square is within board boundaries
                if (adjX >= 0 && adjX < size && adjY >= 0 && adjY < size) {
                    const adjResult = game.play(adjX, adjY, game.turn, true); // noplay=true to only test
                    if (adjResult === 4) { // illegal due to ko
                        board.addObject({ x: adjX, y: adjY, type: 'SQ' });
                    }
                }
            }
        }
    }
}

// --- Tsumego Loading and Board Rendering ---

/**
 * Selects the next tsumego problem based on user rank and past performance.
 * @returns {object|null} The selected problem's metadata, or null if no problems are available.
 */
function selectProblem() {
    if (showCorrectMove && currentProblemFilename) {
        return allProblems.find(p => p.filename === currentProblemFilename);
    }

    if (allProblems.length === 0) return null;

    let availableProblems = allProblems.filter(problem =>
        problem.filename !== currentProblemFilename &&
        problem.filename !== previousProblemFilename
    );

    if (availableProblems.length === 0) {
        availableProblems = allProblems; // Fallback to all problems if filtering leaves none
    }

    availableProblems.sort((a, b) => Math.abs(a.rank - userRank) - Math.abs(b.rank - userRank));
    const candidateProblems = availableProblems.slice(0, 500);

    const weightedProblems = candidateProblems.map(problem => ({
        problem,
        weight: problemFrequencies[problem.filename] || 1
    }));

    const totalWeight = weightedProblems.reduce((sum, p) => sum + p.weight, 0);

    if (totalWeight <= 0) {
        return candidateProblems[Math.floor(Math.random() * candidateProblems.length)];
    }

    let random = Math.random() * totalWeight;
    for (const { problem, weight } of weightedProblems) {
        random -= weight;
        if (random <= 0) return problem;
    }

    return candidateProblems[Math.floor(Math.random() * candidateProblems.length)];
}

/**
 * Initializes or re-initializes the WGo board.
 * @param {number} size - The size of the board.
 */
function initializeBoard(size) {
    if (board) board.removeEventListener('click', handleBoardClick);

    boardContainer.innerHTML = '';
    const boardWrapper = document.createElement('div');
    boardContainer.appendChild(boardWrapper);

    board = new WGo.Board(boardWrapper, {
        size,
        background: WGo.DIR + "wood_1024.jpg",
        stoneHandler: WGo.Board.drawHandlers.REALISTIC
    });

    // Add click event listener
    board.addEventListener('click', handleBoardClick);
    
    // Add WebGL context event listeners to one of the canvas elements
    // We use the stone layer's first canvas element since it's always present
    if (board && board.stone && board.stone.elements && board.stone.elements[0]) {
        const canvas = board.stone.elements[0];
        
        // Listen for WebGL context loss
        canvas.addEventListener('webglcontextlost', function(event) {
            console.log('WebGL context lost');
            event.preventDefault();
            // Prevent the default behavior which would cause the browser to remove the canvas
        });
        
        // Listen for WebGL context restoration
        canvas.addEventListener('webglcontextrestored', function(event) {
            console.log('WebGL context restored');
            // Re-render the board when context is restored
            updateBoard();
        });
    }
    
    updateBoardSize();
}

/**
 * Draws a marker on the board to indicate the correct move.
 */
function drawCorrectMoveMarker() {
    const correctNode = kifuReader.node.children.find(child => child.comment?.includes('Correct'));
    if (correctNode?.move) {
        board.addObject({ x: correctNode.move.x, y: correctNode.move.y, type: 'outline' });
    }
}

/**
 * Redraws the board with the current position, moves, and markers.
 */
function updateBoard() {
    if (!board || !kifuReader) return;

    const position = kifuReader.game.position;
    if (board.size !== position.size) initializeBoard(position.size);

    board.removeAllObjects();

    for (let x = 0; x < position.size; x++) {
        for (let y = 0; y < position.size; y++) {
            if (position.get(x, y) !== 0) {
                board.addObject({ x, y, c: position.get(x, y) });
            }
        }
    }

    if (kifuReader.node.move) {
        board.addObject({ x: kifuReader.node.move.x, y: kifuReader.node.move.y, type: 'CR' });
        
        // Check if the last move created an illegal ko situation
        // We check all four adjacent positions for potential ko captures
        const x = kifuReader.node.move.x;
        const y = kifuReader.node.move.y;
        const size = game.size;
        const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // up, down, left, right
        
        for (const [dx, dy] of directions) {
            const adjX = x + dx;
            const adjY = y + dy;
            
            // Check if adjacent square is within board boundaries
            if (adjX >= 0 && adjX < size && adjY >= 0 && adjY < size) {
                // Check if playing at this position would be illegal due to ko
                const koResult = game.play(adjX, adjY, game.turn, true); // noplay=true to only test
                if (koResult === 4) { // 4 indicates illegal due to ko
                    board.addObject({ x: adjX, y: adjY, type: 'SQ' });
                }
            }
        }
    }

    if (showCorrectMove) drawCorrectMoveMarker();
}

/**
 * Adjusts the board's dimensions on window resize or orientation change.
 */
function updateBoardSize() {
    if (!board) return;

    const infoContainer = document.getElementById('info-container');
    const h = window.innerHeight;
    const infoSize = 1 / 13;

    const availableHeight = h / (1 + infoSize);
    const boardSize = Math.min(window.innerWidth, availableHeight);

    boardContainer.style.width = boardContainer.style.height = boardSize + 'px';
    board.setWidth(boardSize);
    board.setHeight(boardSize);

    if (infoContainer) {
        infoContainer.style.width = boardSize + 'px';
        infoContainer.style.height = (boardSize * infoSize) + 'px';

        const fontSize = boardSize / 25;
        document.getElementById('rank-display').style.fontSize = fontSize + 'px';
        document.getElementById('komi').style.fontSize = fontSize + 'px';
        document.getElementById('prev-tsumego').style.fontSize = fontSize + 'px';
    }
}

/**
 * Initializes the KifuReader from SGF content and advances to the first playable move.
 * @param {string} sgfContent - The SGF string to load.
 * @returns {object} The parsed WGo.Kifu object.
 */
function initializeKifuFromSGF(sgfContent) {
    const kifu = WGo.Kifu.fromSgf(sgfContent);
    kifuReader = new WGo.KifuReader(kifu);

    while (kifuReader.node.children.length === 1) {
        kifuReader.next();
    }

    game = kifuReader.game;
    return kifu;
}

/**
 * Fetches an SGF file from a zip archive.
 * @param {string} filename - The SGF file to load.
 * @param {string} zipFilename - The name of the zip file to load from.
 */
async function fetchSgfFromZip(filename, zipFilename) {
    if (!loadedZips[zipFilename]) {
        await loadZipProblems(zipFilename);
    }

    const zip = zipCache[zipFilename];
    if (!zip) throw new Error(`Zip file ${zipFilename} not found in cache`);

    const file = zip.file(filename);
    if (!file) throw new Error(`File ${filename} not found in ${zipFilename}`);

    return await file.async('text');
}

/**
 * Loads metadata about all zip files.
 */
async function loadZipMetadata() {
    if (zipMetadata.length === 0) {
        const response = await fetch('metadata.json');
        zipMetadata = await response.json();
    }
}

/**
 * Evicts zip files from cache that are not listed in the main metadata.json
 */
async function evictUnusedZipFiles() {
    try {
        // Load metadata first
        await loadZipMetadata();
        
        // Get all cache keys
        const cache = await caches.open(TSUMEGO_CACHE_NAME);
        const cacheKeys = await cache.keys();
        
        // Extract expected zip filenames from metadata
        const expectedZipFiles = new Set(zipMetadata.map(zip => `tsumegos/${zip.filename}`));
        
        // Check each cached file
        for (const request of cacheKeys) {
            const url = new URL(request.url);
            const path = url.pathname.substring(1); // Remove leading slash
            
            // If it's a tsumego zip file and not in our expected list, delete it
            if (path.startsWith('tsumegos/tsumegos') && path.endsWith('.zip') && !expectedZipFiles.has(path)) {
                console.log("Evicting unused zip from cache:", path);
                await cache.delete(request);
            }
        }
    } catch (error) {
        console.error("Error evicting unused zip files:", error);
    }
}

/**
 * Loads and caches problems from a specific zip file.
 * @param {string} zipFilename - The name of the zip file to load.
 */
async function loadZipProblems(zipFilename) {
    if (loadedZips[zipFilename]) return;

    // Try to get from cache first
    const cache = await caches.open(TSUMEGO_CACHE_NAME);
    const cachedResponse = await cache.match(`tsumegos/${zipFilename}`);
    
    let zipData;
    if (cachedResponse) {
        zipData = await cachedResponse.arrayBuffer();
    } else {
        const zipResponse = await fetch(`tsumegos/${zipFilename}`);
        zipData = await zipResponse.arrayBuffer();
        
        // Cache the response
        const responseToCache = new Response(zipData);
        await cache.put(`tsumegos/${zipFilename}`, responseToCache);
    }

    const zip = new JSZip();
    await zip.loadAsync(zipData);

    const metadataFile = zip.file('metadata.json');
    if (!metadataFile) throw new Error(`metadata.json not found in ${zipFilename}`);

    const metadataContent = await metadataFile.async('text');
    const problems = JSON.parse(metadataContent).map(p => ({ ...p, zipFilename }));

    loadedZips[zipFilename] = problems;
    allProblems.push(...problems);
    zipCache[zipFilename] = zip;
}

/**
 * Finds the zip file with a rank range closest to the user's current rank.
 * @returns {object|null} The metadata for the closest zip file.
 */
function findClosestZip() {
    let closestZip = null;
    let minRankDiff = Infinity;

    zipMetadata.forEach(zip => {
        const diffToMin = Math.abs(userRank - zip.min_rank);
        const diffToMax = Math.abs(userRank - zip.max_rank);
        const minDiff = Math.min(diffToMin, diffToMax);

        if (minDiff < minRankDiff) {
            minRankDiff = minDiff;
            closestZip = zip;
        }
    });

    return closestZip;
}

/**
 * Background loads the zip file closest to the user's rank if not already loaded.
 */
async function backgroundLoadCloserZip() {
    const closestZip = findClosestZip();
    if (closestZip && !loadedZips[closestZip.filename]) {
        try {
            await loadZipProblems(closestZip.filename);
        } catch (error) {
            console.error(`Failed to background load ${closestZip.filename}:`, error);
        }
    }
}

/**
 * Main function to load and display a tsumego problem.
 */
async function loadTsumego(update) {
    try {
        acceptClicks = false;
        await loadZipMetadata();
        
        // Set early-click block only when not in hint mode

        let kifu;
        let selectedProblem = null;

        const savedTsumegoFile = localStorage.getItem('currentTsumego');
        const savedTsumegoContent = localStorage.getItem('currentTsumegoContent');

        if (savedTsumegoFile && savedTsumegoContent && !update) {
            currentProblemFilename = savedTsumegoFile;
            currentSGF = savedTsumegoContent;
        } else {
            if (!selectedProblem) {
                selectedProblem = selectProblem();
            }

            if (!selectedProblem) {
                const closestZip = findClosestZip();
                if (closestZip) {
                    console.log("Blocking on zip download for problem selection");
                    await loadZipProblems(closestZip.filename);
                    selectedProblem = selectProblem();
                    if (!selectedProblem) throw new Error("Failed to select a problem after loading zip.");
                } else {
                    throw new Error("No zip files found in metadata.");
                }
            }

            previousSGF = currentSGF;
            previousProblemFilename = currentProblemFilename;

            currentProblemFilename = selectedProblem.filename;
            currentSGF = await fetchSgfFromZip(currentProblemFilename, selectedProblem.zipFilename);

            localStorage.setItem('currentTsumego', currentProblemFilename);
            localStorage.setItem('currentTsumegoContent', currentSGF);
            if (previousProblemFilename) {
                localStorage.setItem('previousTsumego', previousProblemFilename);
                localStorage.setItem('previousTsumegoContent', previousSGF);
            }

        }

        kifu = initializeKifuFromSGF(currentSGF);
        initializeBoard(kifu.size);
        updateBoard();
        updateAllUI();

        acceptClicks = true;
        backgroundLoadCloserZip();

    } catch (error) {
        // MODIFICATION START: Improved error logging
        let errorMessage = "Error loading tsumego";
        if (currentProblemFilename) {
            errorMessage += ` (file: ${currentProblemFilename})`;
        }
        errorMessage += ":";
        console.error(errorMessage, error);
        // MODIFICATION END

        displayOverlay('Error loading problem.', 3000);
    }
}


// --- Initial Load & Event Listeners ---

document.addEventListener('DOMContentLoaded', () => {
    updateAllUI();
    
    // Initialize audio
    initAudio();

    const prevTsumegoLink = document.getElementById('prev-tsumego');
    if (prevTsumegoLink) {
        prevTsumegoLink.addEventListener('click', (e) => {
            e.preventDefault();
            // When showing correct move (hint mode), use current tsumego as "previous"
            const sgfToDownload = showCorrectMove ? currentSGF : previousSGF;
            const filenameToDownload = showCorrectMove ? currentProblemFilename : previousProblemFilename;
            
            if (sgfToDownload && filenameToDownload) {
                const blob = new Blob([sgfToDownload], { type: 'application/x-go-sgf' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filenameToDownload;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
        });
    }

    // Load tsumego without waiting for eviction
    loadTsumego(false).catch(error => console.error("Failed to load initial tsumego:", error));
    
    // Evict unused zip files from cache in background
    evictUnusedZipFiles().catch(error => console.error("Error during cache eviction:", error));
});

// Handle page visibility changes to detect when the app was idle
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && board) {
        // Page became visible again, update the board in case of context loss
        console.log('Page became visible, checking board state');
        try {
            updateBoard();
        } catch (e) {
            console.log('Board update failed, likely due to context loss:', e);
        }
    }
});

window.addEventListener('resize', updateBoardSize);
window.addEventListener('orientationchange', updateBoardSize);