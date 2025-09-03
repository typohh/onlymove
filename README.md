# YoseTsumego

This project generates Go problems using KataGo.

- We generate a random position by playing 4 moves anywhere except first line.
- We then play two moves that attempt to get the game back to balance according to score and winrate.
- We evaluate if position is a good tsumego, if best move has a high winrate (70%), second best has a low winrate (30%), and score is 0-2 for white for the second best move.

## Usage

To generate Go problems, run the `main.py` script using the virtual environment's Python:

```bash
.venv/bin/python3 main.py
```

## PWA

This project includes a Progressive Web App (PWA) for solving tsumegos.

### Building the PWA

To build the PWA, navigate to the `pwa` directory and run the build command:

```bash
cd pwa
npm install
npm run build
```

### Problem Selection Logic

The PWA uses a history-based problem selection logic:

- It tracks the last 200 problems you've faced.
- Each problem is marked as `success`, `failure`, or `in-between`.
- Failed problems have a linearly increasing probability of being shown again.
- In-between problems also have a (slower) linearly increasing probability of being shown again.
- If no problem is selected from the history, a new problem close to your rating is chosen.

