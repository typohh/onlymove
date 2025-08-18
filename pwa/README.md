# Project Overview: Tsumego SPA

## 1. High-Level Summary

**Tsumego SPA** is a web-based, mobile-first Single Page Application (SPA) designed for practicing Go (Baduk, Weiqi) life and death problems (tsumego).

The core experience revolves around presenting users with tsumego problems appropriate for their skill level. The application dynamically adjusts the user's rank based on their performance. It also features an "Analysis Mode" where users can freely explore variations of the previous problem with the help of a server-side AI engine.

The application is built with vanilla JavaScript (ESM), organized into a modular structure, and uses a modern build process for optimization and deployment.

## 2. Core Features

*   **Adaptive Tsumego Solving:** Selects problems from a large, pre-processed collection based on the user's estimated rank.
*   **Dynamic Rank System:** User rank and problem selection momentum are adjusted in real-time based on correct/incorrect answers.
*   **Analysis Mode:** Users can enter a special mode to analyze the previous problem, place stones freely, and receive real-time feedback (score, best move) from a dedicated analysis server.
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

## 4. Project Structure

The codebase is organized into a modular structure to separate concerns.

```
/
├── dist/                     # The build output directory (auto-generated, not in git)
├── js/                       # All application source code lives here
│   ├── main.js               # Main entry point, event listeners, mode switching
│   ├── state.js              # Centralized application state management
│   ├── ui.js                 # All DOM manipulation and UI updates
│   ├── board.js              # WGo.js board logic and interaction
│   ├── tsumego.js            # Tsumego problem loading and solving logic
│   ├── analysis.js           # Analysis mode logic
│   ├── api.js                # Server communication for analysis
│   ├── audio.js              # Audio playback functions
│   └── utils.js              # General helper functions (localization, etc.)
├── lib/                      # Third-party libraries
│   ├── wgo/                  # WGo.js library files and assets (stones, CSS)
│   └── jszip.min.js          # JSZip library file
├── sound/                    # UI sound effects
├── tsumegos/                 # Directory containing zipped tsumego collections
├── build.mjs                 # The Node.js build script using esbuild
├── index.html                # The single HTML page for the application
├── package.json              # Project dependencies and scripts
├── style.css                 # Main application styles
├── sw.js                     # The Service Worker file for PWA functionality
└── manifest.json             # The PWA manifest file
```

## 5. Key Concepts & Architecture

### 5.1. State Management (`js/state.js`)

The application uses a simple, centralized state object. All mutable data (current user rank, loaded problems, game state, etc.) is stored in a single exported `state` object in `js/state.js`. This makes tracking and debugging data flow straightforward.

### 5.2. Application Modes

The app operates in two primary modes, controlled by `state.appMode`:

1.  **Solving Mode (`'solving'`)**:
    *   This is the default mode for practicing tsumego.
    *   Logic is primarily handled by **`js/tsumego.js`**.
    *   User clicks are evaluated as correct or incorrect, and stats are updated.
    *   The UI shows the user's rank and a link to analyze the previous problem.

2.  **Analysis Mode (`'analyzing'`)**:
    *   Entered by clicking the "Previous" button.
    *   Logic is handled by **`js/analysis.js`**.
    *   Users can place stones for either color to explore variations.
    *   After each move, **`js/api.js`** sends the current board position (SGF) to a hardcoded server endpoint to get an AI evaluation.
    *   The UI shows controls to "Exit", "Undo", and displays the current score.

### 5.3. Tsumego Loading Pipeline

Problems are stored in `.zip` files to reduce the number of network requests. The loading process is as follows:

1.  `metadata.json` (root) is fetched, containing a list of all available `.zip` files and the rank range they cover.
2.  Based on `userRank`, an appropriate `.zip` file is selected and fetched. The contents are cached in the browser's Cache Storage.
3.  The `.zip` file is opened in memory using **JSZip**. It contains another `metadata.json` with a list of all SGF files within.
4.  A problem (SGF filename) is selected from the zip's metadata based on a weighted algorithm.
5.  The content of the selected SGF file is extracted from the zip as a string and displayed on the board.

### 5.4. Build Process & Cache Busting (`build.mjs`)

The project **must be built** before it can be run. The `npm run build` command executes `build.mjs`, which performs these critical tasks:

1.  **Bundling:** All JavaScript modules in the `js/` directory are combined into a single file by `esbuild`.
2.  **Minification:** The bundled JavaScript is minified to reduce its size.
3.  **Versioning:** The current Git commit hash is injected into the code as a global `APP_VERSION` constant.
4.  **Cache Busting:** `esbuild` generates a unique, hashed filename for the final JavaScript bundle (e.g., `main.A1B2C3D4.js`).
5.  **HTML Injection:** The `build.mjs` script automatically updates `index.html` to point to this new, uniquely named JavaScript file.
6.  **Asset Copying:** All static assets (`lib/`, `sound/`, `style.css`, etc.) are copied into the final `dist` directory.

The output is a production-ready `dist` folder. The hashed filename guarantees that browsers will never serve a stale, cached version of the application's logic.

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

Therefore, to deploy a new version, a developer only needs to **push their commits to the production Git branch**.
