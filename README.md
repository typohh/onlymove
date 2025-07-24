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

