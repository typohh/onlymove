# Project Overview: Tsumego SPA

## 1. High-Level Summary

**Tsumego SPA** is a web-based, mobile-first Single Page Application (SPA) designed for practicing Go (Baduk, Weiqi) 9x9 life and death problems (tsumego).

The core experience revolves around presenting users with tsumego problems appropriate for their skill level. The application dynamically adjusts the user's rank based on their performance. When a user fails a problem, they automatically enter an "Analysis Mode" where they can freely explore variations. This analysis is powered by data embedded in the SGF file itself or, as a fallback, by a server-side AI engine.

The application is built with vanilla JavaScript (ESM), organized into a modular structure, and uses a modern build process for optimization and deployment.

## 2. Core Features

*   **Adaptive Tsumego Solving:** Selects problems from a large, pre-processed collection based on the user's estimated rank and problem quality, using weighted random sampling.
*   **Dynamic Rank System:** User rank and problem selection momentum are adjusted in real-time based on correct/incorrect answers.
*   **Automatic Analysis Mode:** Failing a problem automatically transitions the user into a review mode for that tsumego.
*   **Client-Side Analysis:** For many problems, analysis (best move, score) is determined instantly using win-rate information embedded in the SGF file, avoiding server latency.
*   **Progressive Web App (PWA):** Includes a service worker and manifest for offline capabilities and a native-like app experience.

## 3. Technology Stack

| Category      | Technology / Library                                       | Purpose                                                              |
|---------------|------------------------------------------------------------|----------------------------------------------------------------------|
| **Frontend**  | HTML5, CSS3, JavaScript (ES Modules)                       | Core application structure, styling, and logic.                      |
| **Libraries** | **WGo.js**                                                 | Rendering the Go board, stones, and handling SGF parsing.            |
|               | **JSZip.js**                                               | Reading tsumego problems from `.zip` archives on the client side.    |
| **Build Tools**| **Node.js**                                                | The runtime environment for the build script.                        |
|               | **npm**                                                    | Package management and running scripts.                              |
|               | **esbuild**                                                | High-speed JavaScript bundling, minification, and code transformation. |
| **Deployment**| **Git**                                                    | Version control.                                                     |
|               | **Cloudflare Pages**                                       | Hosting and automated, Git-connected deployments.                    |
| **Backend**   | **Cloudflare Functions**                                   | Serverless functions for handling user reports and AI analysis fallback. |
|               | **Cloudflare D1**                                          | SQLite-compatible database for storing user results.                 |

## 4. Project Structure

The codebase is organized into a modular structure to separate concerns.

```
/
├── dist/                     # The build output directory (auto-generated, not in git)
├── functions/                # Cloudflare Functions
│   └── report.js             # Function for handling tsumego reports and storing in D1
├── js/                       # All application source code lives here
│   ├── main.js               # Main entry point, event listeners, mode switching
│   ├── game-logic.js         # Core tsumego solving logic (click handling, correctness)
│   ├── problem-loader.js     # Tsumego problem loading, selection, and caching
│   ├── user-stats.js         # User rank and statistics management
│   ├── analysis.js           # Analysis mode logic
│   ├── board.js              # Abstraction layer for WGo.js board interaction
│   ├── state.js              # Centralized application state management
│   ├── sgf.js                # SGF data parsing, interpretation, and transformation
│   ├── api.js                # Server communication for analysis
│   ├── ui.js                 # All DOM manipulation and UI updates
│   ├── overlays.js           # Manages temporary informational overlays
│   ├── audio.js              # Audio playback functions
│   ├── i18n.js               # Internationalization and localization
│   └── constants.js          # Shared application constants
├── lib/                      # Third-party libraries
│   ├── wgo/                  # WGo.js library files and assets (stones, CSS)
│   └── jszip.min.js          # JSZip library file
├── sound/                    # UI sound effects
├── tsumegos/                 # Directory containing zipped tsumego collections
├── build.mjs                 # The Node.js build script using esbuild
├── index.html                # The single HTML page for the application
├── package.json              # Project dependencies and scripts
├── schema.sql                # D1 database schema
├── style.css                 # Main application styles
├── sw.js                     # The Service Worker file for PWA functionality
└── manifest.json             # The PWA manifest file
```

## 5. Key Concepts & Architecture

### 5.1. State Management (`js/state.js`)

The application uses a centralized state object, but all modifications are handled through exported setter functions (e.g., `setAppMode`, `setUserRank`). This provides a single source of truth while making state changes more explicit and trackable. The state is persisted in `localStorage` to allow session recovery.

### 5.2. Application Modes

The app operates in three primary modes, controlled by `state.appMode`:

1.  **Solving Mode (`'solving'`)**:
    *   This is the default mode for practicing tsumego.
    *   Logic is primarily handled by **`js/game-logic.js`**.
    *   User clicks are evaluated as correct or incorrect based on the SGF data.
    *   If a user fails a problem, the app automatically enters Analysis Mode.

2.  **Analysis Mode (`'analyzing'`)**:
    *   Entered automatically after failing a problem, or manually by clicking the "Review" button.
    *   Logic is handled by **`js/analysis.js`**.
    *   Users can place stones for either color to explore variations.
    *   The best move and score are determined client-side by parsing the SGF via **`js/sgf.js`**. If the SGF lacks this data, **`js/api.js`** sends the position to a server for AI evaluation.
    *   The UI shows controls to "Exit" and "Undo".

3.  **Linked Mode (`'linked'`)**:
    *   Entered when a user opens a URL with a specific problem ID.
    *   This mode allows viewing and analyzing a specific problem without affecting the user's rank or problem history.

### 5.3. SGF Data Format & Handling (`js/sgf.js`)

The application relies on a rich SGF format.
*   The starting position of the problem is marked with a `"Q"` comment.
*   Each branch after this point contains a comment with win-rate and score information (e.g., `"86.4%/3.1"`).
*   The **`sgf.js`** module parses this information to determine the correct move without needing a "Correct" text marker.
*   This module also handles random reorientation of problems to provide variety.

### 5.4. Tsumego Loading Pipeline (`js/problem-loader.js`)

The loading pipeline is a core component, managed by **`js/problem-loader.js`**. It uses `.zip` archives and a sophisticated selection algorithm:
*   **Weighted Random Selection**: Selects from a pool of 50 candidate problems closest to the user's rank. The selection is weighted by the problem's `quality` attribute, making higher-quality problems more likely to be chosen.
*   **Spaced Repetition**: Re-introduces failed problems into the selection pool at increasing intervals (after ~30 and ~180 problems).
*   **Background Loading**: To ensure a smooth user experience, the system proactively loads the next most relevant `.zip` archive in the background based on the user's current rank.

### 5.5. Board Interaction (`js/board.js`)

This module acts as an abstraction layer over the WGo.js library. It is responsible for:
*   Initializing and rendering the board.
*   Drawing stones, markers, and analysis hints.
*   Handling all click events and routing them to the appropriate handler (`game-logic.js` or `analysis.js`) based on the current `appMode`.
*   Implementing a "ghost stone" for double-click move confirmation on mobile devices.

### 5.6. Build Process (`build.mjs`)

The build process is unchanged. The project **must be built** before it can be run using `npm run build`.

### 5.7. Cloudflare Functions & D1 Database Integration

The backend integration for reporting results is also unchanged, now handled by the dedicated **`js/user-stats.js`** module on the frontend.

## 6. Development Workflow

### 6.1. Initial Setup

1.  Clone the repository.
2.  Run `npm install` to download the development dependencies (`esbuild`, `wrangler`).

### 6.2. Local Testing (Important)

The application **cannot be run by simply opening `index.html`**. It must be built first.

1.  Run the build command:
    ```sh
    npm run build
    ```
2.  Navigate into the output directory:
    ```sh
    cd dist
    ```
3.  Start a simple local web server:
    ```sh
    python -m http.server
    ```
4.  Open `http://localhost:8000` in your browser.

### 6.3. Deployment

Deployment is handled automatically by Cloudflare Pages. The process is configured in the Cloudflare dashboard to:

1.  Watch a specific branch (e.g., `main`).
2.  Use a specific sub-folder as the project's root directory.
3.  Run the build command: `npm install && npm run build`.
4.  Deploy the contents of the `dist` directory.
5.  Automatically deploy Cloudflare Functions from the `functions` directory.

Therefore, to deploy a new version, a developer only needs to **push their commits to the production Git branch**. Cloudflare Pages will automatically detect and deploy both the static assets and the serverless functions.

### 7.0. LLM instructions

- Rely on duduced file content whenever reasonable, instead of reading the content. Reading files "just in case" will slow down operation without improving performance.
- If you want to read data files, including text data files like csv, only read as few lines as possible, usually something like 1-3.
- For small code files, read the entire file, instead of reading parts of it.
- After you have made the requested changes, run npm build, provide summary of changes made since last request and stop and wait for further instructions.
- Do not remove debug outputs, etc. until the bug is fixed and is also TESTED that it is fixed, not just that it compiles.
- Confine your actions strictly to the user's explicit instructions. Do not perform any additional actions, even if you assess them as beneficial or corrective. If you identify potential issues or improvements outside the scope of the current task, complete the requested task first. Then, separately, present your observations and ask for permission before taking any further action.
- Try to rule out any "explanation" you have for a bug, by properly investigating the code before making any changes. Instead of hiding behind vague claims and jumping to conclusions.
- Always investigate code thoroughly for potential issues before assuming external causes, and avoid quick fixes or number adjustments without proper analysis.
- Avoid workarounds for issues that are not fully understood. Investigate the root cause of a problem before implementing a solution. Do not use 'just-in-case' code that can hide underlying issues or make the code harder to understand.
- When using the edit tool, avoid providing escaped special characters in the new_string; use literal newlines and unescaped quotes to prevent the tool from introducing syntax errors in code, HTML, or other files.
- Avoid performing broad project-level searches with searchText, as they may return too many results; instead, limit searches to specific files or subfolders where you know the content is relevant.
- Upon reading these LLM instructions, quote back any you understand and agree to abide by.

