# YoseTsumego

This project generates Go problems using KataGo.

Generation strategy is to play out a single game, then evaluate each black position during the game as posible tsumego candidate.

We may try to alter each position to ensure following goals are met:
- The correct move leads to black winning by 0-2 points and winrate exceeding 60%. This can be managed by having both play moves that have this score, and as high win rate as possible for black.
- There is only one correct first move, all other moves are at most 40%. This can be managed by having black play a move that is above 40% winrate, other than the best move, and having white play respond that aims at score of 0-2 and winrate for black as high as possible.
- 
- We want the tsumego to have large gap in winrate between best and second best move. We want the score of the second best move to be close to -0.5 as possible.

Also we evaluate the tsumego quality and difficulty, by using katago human network to see which rank humans have a reasonable chance of winning against the full strength katago network.

## Usage

To generate Go problems, run the `main.py` script using the virtual environment's Python:

```bash
.venv/bin/python3 main.py
```

