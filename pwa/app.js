// --- START OF REFACTORED FILE app.js ---

const boardContainer = document.getElementById('board-container');

// App version - last modified date and time (automatically updated by deployment script)
const APP_VERSION = "2025-08-12 19:50";
console.log(`App Version: ${APP_VERSION}`);

// --- Constants ---
const TSUMEGO_CACHE_NAME = 'tsumego-cache-v1';

// Local Storage Keys
const STORAGE_USER_RANK = 'userRank';
const STORAGE_PROBLEM_FREQUENCIES = 'problemFrequencies';
const STORAGE_MOMENTUM = 'momentum';
const STORAGE_PREVIOUS_TSUMEGO = 'previousTsumego';
const STORAGE_PREVIOUS_TSUMEGO_CONTENT = 'previousTsumegoContent';
const STORAGE_CURRENT_TSUMEGO = 'currentTsumego';
const STORAGE_CURRENT_TSUMEGO_CONTENT = 'currentTsumegoContent';


// --- Audio State ---
let stoneSound = null;
let failSound1 = null;
let failSound2 = null;
let successSound = null;

// --- State Variables ---
let board;
let game;
let kifuReader;
let zipMetadata = []; // Metadata about zip files (filename, min_rank, max_rank)
let loadedZips = {}; // Track which zip files have been loaded {zipFilename: problemsArray}
let allProblems = []; // Flattened list of all problems from loaded zips
let currentProblemFilename = null;
let previousProblemFilename = localStorage.getItem(STORAGE_PREVIOUS_TSUMEGO);
let currentSGF = null;
let previousSGF = localStorage.getItem(STORAGE_PREVIOUS_TSUMEGO_CONTENT);
let showCorrectMove = false;
let acceptClicks = false; // Track when we should accept clicks
let currentPointLoss = null; // Store point loss information for display in overlay

// Load state from localStorage or set defaults
let userRank = parseFloat(localStorage.getItem(STORAGE_USER_RANK)) || 0;
let problemFrequencies = JSON.parse(localStorage.getItem(STORAGE_PROBLEM_FREQUENCIES)) || {};
let momentum = parseFloat(localStorage.getItem(STORAGE_MOMENTUM)) || 30;

// MODIFICATION: Cache for loaded zip files using direct references to prevent garbage collection.
const zipCache = new Map();


// --- Localization Helper ---

const localization = {
    'zh': { kyu: '级', dan: '段', komi: '贴目', previous: '上一题', points: '点', rules: '规则', version: '版本', chinese: '中国规则', komiValue: '7.5贴目' },
    'ja': { kyu: '級', dan: '段', komi: 'コミ', previous: '前の問題', points: '点', rules: 'ルール', version: 'バージョン', chinese: '中国ルール', komiValue: '7.5コミ' },
    'ko': { kyu: '급', dan: '단', komi: '덤', previous: '이전 문제', points: '점', rules: '규칙', version: '버전', chinese: '중국 규칙', komiValue: '7.5 덤' },
    'en': { kyu: 'kyu', dan: 'dan', komi: 'komi', previous: 'Previous', points: 'points', rules: 'Rules', version: 'Version', chinese: 'Chinese', komiValue: '7.5' }
};

/**
 * Gets a localized term based on the browser's language.
 * @param {string} key - The key of the term to retrieve ('kyu', 'dan', 'komi', 'previous', 'points').
 * @returns {string} The translated term.
 */
function getTerm(key) {
    const lang = navigator.language || 'en';
    if (lang.startsWith('zh')) return localization.zh[key];
    if (lang.startsWith('ja')) return localization.ja[key];
    if (lang.startsWith('ko')) return localization.ko[key];
    return localization.en[key];
}


// --- Audio Functions ---

/**
 * Encapsulates the logic for playing an audio element.
 * @param {HTMLAudioElement} audioObject - The audio object to play.
 */
function _playSound(audioObject) {
    if (audioObject) {
        try {
            audioObject.currentTime = 0;
            audioObject.play().catch(e => console.log('Audio play failed:', e));
        } catch (e) {
            console.log('Audio play failed:', e);
        }
    }
}

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

function playStoneSound() {
    _playSound(stoneSound);
}

function playRandomFailSound() {
    if (!failSound1 || !failSound2) return;
    const sounds = [failSound1, failSound2];
    const randomSound = sounds[Math.floor(Math.random() * sounds.length)];
    _playSound(randomSound);
}

function playSuccessSound() {
    // Temporarily disabled
    // _playSound(successSound);
}


// --- Suspension Detector Class (Unchanged) ---
class SuspensionDetector {
  constructor(onResume) {
    this.onResume = onResume;
    this._init();
  }
  _init() {
    if ("onfreeze" in document) {
      document.addEventListener("freeze", this.onPause);
      document.addEventListener("resume", () => {
        this.onResume(document.wasDiscarded);
      });
    } else {
      this._initTimerHack();
    }
    this._initWeakRefCheck();
  }
  _initTimerHack() {
    let last = performance.now();
    setInterval(() => {
      const now = performance.now();
      if (now - last > 2000) this.onResume();
      last = now;
    }, 1000);
  }
  _initWeakRefCheck() {
    if (typeof WeakRef !== "undefined") {
      this.ref = new WeakRef({});
      const finalizer = new FinalizationRegistry(() => {
        this.onResume();
      });
      finalizer.register(this.ref.deref());
    }
  }
}


// --- Service Worker and Version Display ---
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(registrations => {
    registrations.forEach(reg => reg.update());
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const komiElement = document.getElementById('komi');
  if (komiElement) {
    komiElement.textContent = getTerm('rules'); // Use localized term for "Rules"
    komiElement.addEventListener('click', () => {
      const versionDisplay = document.createElement('div');
      // Show version, rules, and komi when clicked with localization
      versionDisplay.innerHTML = `${getTerm('version')}: ${APP_VERSION}<br>${getTerm('rules')}: ${getTerm('chinese')}<br>${getTerm('komi')}: ${getTerm('komiValue')}`;
      Object.assign(versionDisplay.style, {
          position: 'fixed', top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'rgba(0, 0, 0, 0.8)', color: 'white',
          padding: '10px 20px', borderRadius: '5px', zIndex: '10000',
          fontSize: '18px', pointerEvents: 'none',
          textAlign: 'center' // Center-align the text
      });
      document.body.appendChild(versionDisplay);
      setTimeout(() => versionDisplay.remove(), 2000);
    });
  }
});


// --- Helper Functions ---

/**
 * Converts a numeric rank (0.0 to 1.0) to a descriptive Go rank (20k to 9d).
 * @param {number} numericRank - The rank value between 0.0 and 1.0.
 * @returns {string} The descriptive rank string.
 */
function getDescriptiveRank(numericRank) {
    const totalRanks = 29; // 20 kyu ranks + 9 dan ranks
    const rankIndex = Math.floor(Math.pow(numericRank ,1.0) * totalRanks);
    const kyuTerm = getTerm('kyu');
    const danTerm = getTerm('dan');
    return (rankIndex < 20) ? `${20 - rankIndex} ${kyuTerm}` : `${rankIndex - 19} ${danTerm}`;
}

/**
 * Centralized function to update all UI elements related to rank, komi, and links.
 */
function updateAllUI() {
    const rank = getDescriptiveRank(userRank);
    const stddev = momentum / 4;

    const rankElement = document.getElementById('rank-display');
    if (rankElement) rankElement.textContent = `${rank} (±${stddev.toFixed(1)})`;

    const komiElement = document.getElementById('komi');
    if (komiElement) komiElement.textContent = getTerm('rules');

    const prevTsumegoLink = document.getElementById('prev-tsumego');
    if (prevTsumegoLink) {
        prevTsumegoLink.textContent = getTerm('previous');
        if (previousSGF && previousProblemFilename) {
            prevTsumegoLink.classList.remove('disabled');
        } else {
            prevTsumegoLink.classList.add('disabled');
        }
    }
}


/**
 * Displays a temporary overlay.
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

    if (currentPointLoss) {
        const pointLossElement = document.createElement('div');
        Object.assign(pointLossElement.style, {
            color: 'white', fontWeight: 'bold', textAlign: 'center',
            textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)',
            fontSize: boardContainer ? `${boardContainer.offsetWidth / 25}px` : '24px'
        });
        pointLossElement.textContent = currentPointLoss;
        overlay.appendChild(pointLossElement);
    }

    document.body.appendChild(overlay);

    setTimeout(() => {
        overlay.style.transition = 'opacity 4s linear';
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                overlay.style.opacity = '0.9';
                playRandomFailSound();
            });
        });
    }, 1000);

    setTimeout(() => {
        overlay.remove();
        currentPointLoss = null; // Clear after displaying
        if (callback) callback();
    }, 5000);
}

/**
 * Displays a white fade overlay for correct answers.
 */
function fadeOverlay(duration, callback) {
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: 'white',
        zIndex: '1000', opacity: '0', transition: 'opacity 1s ease-in-out'
    });

    document.body.appendChild(overlay);

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            overlay.style.opacity = '1';
            playSuccessSound();
        });
    });

    setTimeout(() => {
        if (callback) callback();
        overlay.style.opacity = '0';
        setTimeout(() => overlay.remove(), 1000);
    }, duration);
}

/**
 * Updates user rank and problem frequency, then saves to localStorage.
 * @param {boolean} isCorrect - Whether the user's answer was correct.
 */
function updateUserStats(isCorrect) {
    const oldRank = userRank;

    if (isCorrect) {
        momentum = momentum * 1.1;
        const currentFreq = problemFrequencies[currentProblemFilename] || 1;
        problemFrequencies[currentProblemFilename] = currentFreq / 5;
        const maxIncrease = (1.0 - userRank) / 2;
        const intendedIncrease = momentum / 600;
        userRank += Math.min(intendedIncrease, maxIncrease);
    } else {
        momentum = (momentum + 1) / 3;
        problemFrequencies[currentProblemFilename] = 25;
        userRank = Math.max(0, userRank - 4 * (momentum / 600));
    }

    localStorage.setItem(STORAGE_USER_RANK, userRank.toString());
    localStorage.setItem(STORAGE_PROBLEM_FREQUENCIES, JSON.stringify(problemFrequencies));
    localStorage.setItem(STORAGE_MOMENTUM, momentum.toString());

    updateAllUI();
}

// --- Core Application Logic ---

/**
 * Handles the logic for a correct user move.
 * @param {boolean} hintWasShown - Whether a hint was shown before the answer.
 */
function handleCorrectAnswer(hintWasShown) {
    if (!hintWasShown) updateUserStats(true);
    showCorrectMove = false;
    fadeOverlay(1000, () => {
        loadTsumego(true).catch(error => console.error("Error loading next tsumego:", error));
    });
}

/**
 * Calculates the point loss text for an incorrect move.
 * @param {object} playedNode - The kifu node for the move the user played.
 * @param {object} currentKifuNode - The current kifu node containing all possible next moves.
 * @returns {string} The formatted point loss text, or an empty string.
 */
function _calculatePointLossText(playedNode, currentKifuNode) {
    const correctNode = currentKifuNode.children.find(child => child.comment?.includes('Correct'));
    if (!correctNode) return '';

    const correctScoreMatch = correctNode.comment?.match(/Score:\s*([+-]?\d+\.?\d*)/);
    const playedScoreMatch = playedNode.comment?.match(/Score:\s*([+-]?\d+\.?\d*)/);

    if (correctScoreMatch && playedScoreMatch) {
        const correctScore = parseFloat(correctScoreMatch[1]);
        const playedScore = parseFloat(playedScoreMatch[1]);
        const pointLoss = Math.abs(correctScore - playedScore);
        return `-${pointLoss.toFixed(1)} ${getTerm('points')}`;
    }
    return '';
}

/**
 * Private helper to handle the animation and resolution of an incorrect move.
 * @param {number} moveIndex - The index of the incorrect move played.
 * @param {boolean} hintWasShown - Whether a hint was active.
 */
async function _handleIncorrectMoveSequence(moveIndex, hintWasShown) {
    const delay = (ms) => new Promise(res => setTimeout(res, ms));
    const renderAndHold = async (ms) => {
        if (boardContainer) void boardContainer.offsetHeight; // Force reflow
        await Promise.resolve();
        await delay(ms);
    };

    // User move
    kifuReader.next(moveIndex);
    playStoneSound();
    updateBoard();
    await renderAndHold(1000);

    // Opponent response
    kifuReader.next();
    playStoneSound();
    updateBoard();

    // Show overlay and load next problem
    if (!hintWasShown) updateUserStats(false);
    await new Promise((resolve) => displayOverlay('', 5000, resolve));

    showCorrectMove = true;
    loadTsumego(false).catch(error => console.error("Error loading next tsumego:", error));
}


/**
 * Handles user clicks on the Go board.
 * @param {number} x - The x-coordinate of the click.
 * @param {number} y - The y-coordinate of the click.
 */
async function handleBoardClick(x, y) {
    if (!acceptClicks || !game || !kifuReader) return;

    const hintWasShown = showCorrectMove;

    if (hintWasShown) {
        const correctNode = kifuReader.node.children.find(child => child.comment?.includes('Correct'));
        if (correctNode?.move && correctNode.move.x === x && correctNode.move.y === y) {
            // Show the stone immediately
            kifuReader.next(kifuReader.node.children.indexOf(correctNode));
            playStoneSound();
            updateBoard();
            // Then, after a delay, handle the success logic
            setTimeout(() => handleCorrectAnswer(hintWasShown), 1000);
            return;
        }
    }

    const moveNode = kifuReader.node.children.find(child => child.move && child.move.x === x && child.move.y === y);

    if (!moveNode) {
        if (hintWasShown) displayOverlay("", 1500);
        return;
    }

    acceptClicks = false;
    const isCorrect = moveNode.comment?.includes('Correct') || false;
    const moveIndex = kifuReader.node.children.indexOf(moveNode);

    if (isCorrect) {
        // Show the stone immediately
        kifuReader.next(moveIndex);
        playStoneSound();
        updateBoard();

        // Then, after a delay, handle the success logic
        setTimeout(() => {
            handleCorrectAnswer(hintWasShown);
        }, 1000);
    } else {
        // For incorrect moves, calculate point loss and then run the sequence
        currentPointLoss = _calculatePointLossText(moveNode, kifuReader.node);
        // The sequence itself handles the necessary delays internally
        await _handleIncorrectMoveSequence(moveIndex, hintWasShown);
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

    let availableProblems = allProblems.filter(p => p.filename !== currentProblemFilename && p.filename !== previousProblemFilename);
    if (availableProblems.length === 0) availableProblems = allProblems;

    availableProblems.sort((a, b) => Math.abs(a.rank - userRank) - Math.abs(b.rank - userRank));
    const candidateProblems = availableProblems.slice(0, 500);

    const weightedProblems = candidateProblems.map(problem => ({ problem, weight: problemFrequencies[problem.filename] || 1 }));
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

    board.addEventListener('click', handleBoardClick);
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
    }

    for( let xd = -1; xd < 2 ; xd += 2) {
        for( let yd = -1; yd < 2 ; yd += 2) {
            x = kifuReader.node.move.x + xd;
            y = kifuReader.node.move.y + yd;
            if (!game.isValid(x, y, game.turn)) {
                if (game.play(x, y, game.turn, true) === 4) { // 4 = illegal due to ko
                    board.addObject({ x, y, type: 'SQ' });
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


// --- Caching and Data Fetching ---

async function fetchSgfFromZip(filename, zipFilename) {
    if (!loadedZips[zipFilename]) await loadZipProblems(zipFilename);

    const zip = zipCache.get(zipFilename);
    if (!zip) {
        console.error(`Zip file ${zipFilename} was not found in cache. Forcing reload.`);
        await loadZipProblems(zipFilename);
        const reloadedZip = zipCache.get(zipFilename);
        if (!reloadedZip) throw new Error(`Failed to load and cache zip file: ${zipFilename}`);
        const fileInReloaded = reloadedZip.file(filename);
        if (!fileInReloaded) throw new Error(`File ${filename} not found in reloaded ${zipFilename}`);
        return await fileInReloaded.async('text');
    }

    const file = zip.file(filename);
    if (!file) throw new Error(`File ${filename} not found in ${zipFilename}`);
    return await file.async('text');
}

async function loadZipMetadata() {
    if (zipMetadata.length === 0) {
        const response = await fetch('metadata.json');
        zipMetadata = await response.json();
    }
}

async function evictUnusedZipFiles() {
    try {
        await loadZipMetadata();
        const cache = await caches.open(TSUMEGO_CACHE_NAME);
        const cacheKeys = await cache.keys();
        const expectedZipFiles = new Set(zipMetadata.map(zip => `tsumegos/${zip.filename}`));
        for (const request of cacheKeys) {
            const path = new URL(request.url).pathname.substring(1);
            if (path.startsWith('tsumegos/tsumegos') && path.endsWith('.zip') && !expectedZipFiles.has(path)) {
                console.log("Evicting unused zip from cache:", path);
                await cache.delete(request);
            }
        }
    } catch (error) {
        console.error("Error evicting unused zip files:", error);
    }
}

async function loadZipProblems(zipFilename) {
    if (loadedZips[zipFilename]) return;

    const cache = await caches.open(TSUMEGO_CACHE_NAME);
    const cacheKey = `tsumegos/${zipFilename}`;
    let zipData;
    const cachedResponse = await cache.match(cacheKey);

    if (cachedResponse) {
        zipData = await cachedResponse.arrayBuffer();
    } else {
        const zipResponse = await fetch(cacheKey);
        zipData = await zipResponse.arrayBuffer();
        await cache.put(cacheKey, new Response(zipData));
    }

    const zip = await new JSZip().loadAsync(zipData);
    const metadataFile = zip.file('metadata.json');
    if (!metadataFile) throw new Error(`metadata.json not found in ${zipFilename}`);

    const metadataContent = await metadataFile.async('text');
    const problems = JSON.parse(metadataContent).map(p => ({ ...p, zipFilename }));

    loadedZips[zipFilename] = problems;
    allProblems.push(...problems);
    zipCache.set(zipFilename, zip);
}

function findClosestZip() {
    return zipMetadata.reduce((closest, zip) => {
        const diff = Math.min(Math.abs(userRank - zip.min_rank), Math.abs(userRank - zip.max_rank));
        return diff < closest.minDiff ? { zip, minDiff: diff } : closest;
    }, { zip: null, minDiff: Infinity }).zip;
}

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
        
        const savedTsumegoFile = localStorage.getItem(STORAGE_CURRENT_TSUMEGO);
        const savedTsumegoContent = localStorage.getItem(STORAGE_CURRENT_TSUMEGO_CONTENT);
        let selectedProblem = null;

        if (savedTsumegoFile && savedTsumegoContent && !update) {
            currentProblemFilename = savedTsumegoFile;
            currentSGF = savedTsumegoContent;
        } else {
            selectedProblem = selectProblem();
            if (!selectedProblem) {
                const closestZip = findClosestZip();
                if (closestZip) {
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

            localStorage.setItem(STORAGE_CURRENT_TSUMEGO, currentProblemFilename);
            localStorage.setItem(STORAGE_CURRENT_TSUMEGO_CONTENT, currentSGF);
            if (previousProblemFilename) {
                localStorage.setItem(STORAGE_PREVIOUS_TSUMEGO, previousProblemFilename);
                localStorage.setItem(STORAGE_PREVIOUS_TSUMEGO_CONTENT, previousSGF);
            }
        }

        const kifu = initializeKifuFromSGF(currentSGF);
        initializeBoard(kifu.size);
        updateBoard();
        updateAllUI();

        acceptClicks = true;
        backgroundLoadCloserZip();

    } catch (error) {
        console.error(`Error loading tsumego (file: ${currentProblemFilename || 'unknown'}):`, error);
        displayOverlay('Error loading problem.', 3000);
    }
}


// --- Initial Load & Event Listeners ---

document.addEventListener('DOMContentLoaded', () => {
    updateAllUI();
    initAudio();

    new SuspensionDetector(() => {
        console.log('App resumed from suspension, reinitializing...');
        initAudio();
        try {
            if (board && currentSGF) {
                const kifu = initializeKifuFromSGF(currentSGF);
                initializeBoard(kifu.size);
                updateBoard();
                console.log('Board successfully reinitialized after suspension');
            }
        } catch (e) {
            console.log('Board reinitialization failed:', e);
            loadTsumego(false).catch(reloadError => console.error("Failed to reload tsumego after suspension:", reloadError));
        }
    });

    const prevTsumegoLink = document.getElementById('prev-tsumego');
    if (prevTsumegoLink) {
        prevTsumegoLink.addEventListener('click', (e) => {
            e.preventDefault();
            const sgfToDownload = showCorrectMove ? currentSGF : previousSGF;
            const filenameToDownload = showCorrectMove ? currentProblemFilename : previousProblemFilename;
            
            if (sgfToDownload && filenameToDownload) {
                const blob = new Blob([sgfToDownload], { type: 'application/x-go-sgf' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filenameToDownload;
                a.click();
                URL.revokeObjectURL(url);
            }
        });
    }

    loadTsumego(false).catch(error => console.error("Failed to load initial tsumego:", error));
    evictUnusedZipFiles().catch(error => console.error("Error during cache eviction:", error));
});

window.addEventListener('resize', updateBoardSize);
window.addEventListener('orientationchange', updateBoardSize);
