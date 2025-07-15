# YoseTsumego

This project generates Go problems using KataGo. 

Generation strategy is to play out a single game, then evaluate each black position during the game as posible tsumego candidate.
- We can start this evaluation at the last position, and proceed backwards in the game. When we reach 1 dan difficulty, we can stop as we do not want harder than 1 dan tsumegos.

We may try to alter the position to ensure following goals are met:
- There is only one correct first move. This can be managed by having black play a move that is above 50% winrate, other than the best move, and having white respond. Repeating this until there is only one remaining 50% winrate move for black.
- The correct move leads to black winning by 0.5 points. This can be managed by having black play a move that loses enough points to drop the point margin of win to 0.5, and have white respond.

Also we evaluate the tsumego quality and difficulty, by using katago human network to see which rank humans have a reasonable chance of winning against the full strength katago network.
- This should be done with multiple plays to the end with each rank human network.
- The tsumego rank is according to the the rank human network that exceeds 50% wins in plays to the ends.
- Next move as a single move should be at most 90% policy candidate for the tsumego rank human network. I.e we want it to be difficult enough.


## Setup

1.  **Install Python dependencies:**
    It is recommended to use a virtual environment (`venv`).

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install sgfmill
    ```

2.  **KataGo Executable and Models:**
    Ensure you have the KataGo executable compiled and the necessary `.bin.gz` model files downloaded. Update the `KATAGO_PATH`, `SL_MODEL_PATH`, and `RL_MODEL_PATH` variables in `main.py` to point to your KataGo executable and models.

## Usage

To generate Go problems, run the `main.py` script using the virtual environment's Python:

```bash
.venv/bin/python3 main.py
```

To see KataGo engine output during generation, add the `--verbose` flag:

```bash
.venv/bin/python3 main.py --verbose
```
