### **Design Plan: Go Problem Generation**

*Note: KataGo's documentation, including details on the analysis engine and its parameters, can be found in the `KataGo/docs` directory of this repository.*

#### **1. Overview**

This document specifies the design for a Python application that generates high-quality 9x9 endgame (yose) Go problems. The system's methodology is to perform a backward analysis of optimal-play games to identify "difficulty spikes"—moves that are demonstrably harder than their surrounding context. The difficulty of a move is determined by finding the precise rank threshold at which a simulated player can identify the correct move.

#### **2. Core Design Principles**

*   **Principle of Backward Analysis:** Problems are found by first generating a complete, optimal game sequence (the "ground truth"). The system then analyzes this sequence in reverse to evaluate the difficulty of each step.
*   **Principle of Direct Rank Simulation:** The difficulty of a move is assessed by instructing KataGo's human-trained (SL) network to simulate a specific rank. This is achieved by passing the full rank string (e.g., `"rank_10k"`, `"rank_3d"`) to the engine via the `humanSLProfile` parameter within the `overrideSettings` object of a query.
*   **Principle of Pure Intuition:** The query to the human-style network for difficulty assessment is always performed with `maxVisits: 1`. This is a fixed constant to ensure the analysis is based on the network's raw, trained intuition for the given rank profile, not on deep search.
*   **Principle of Difficulty as a Search Problem:** The difficulty of a move is not found by iterating through a predefined list. It is found by performing an efficient binary search over the entire ordered spectrum of ranks supported by the engine (e.g., from `rank_20k` to `rank_9d`). This allows the system to pinpoint the exact rank that acts as the tipping point for solving the problem.
*   **Principle of Superhuman Verification:** Before a problem is finalized, a superhuman (RL) network is used to confirm the solution move is objectively correct and that the player's other "intuitive" moves are not superior.
*   **Principle of the Difficulty Spike:** A high-quality problem at move M_i must satisfy two conditions:
    1.  **Entry Condition:** The move is significantly more difficult than the immediately preceding move (M_i-1).
    2.  **Crux Condition:** No subsequent move in the optimal follow-up sequence is significantly more difficult than M_i.

#### **3. System Architecture**

The application uses a two-network, two-phase architecture.

*   **Networks:**
    1.  **RL Network (`pro_level_model.bin.gz`):** The superhuman model, used as the arbiter of objective truth.
    2.  **SL Network (`human_sl_model.bin.gz`):** The human-trained model, which supports the `humanSLProfile` parameter.
*   **Workflow Phases:**
    1.  **Phase 1: Ground-Truth Generation:** The RL Network plays a game against itself at high strength to produce a perfect game sequence.
    2.  **Phase 2: Backward Problem Search:** A dedicated `ProblemFinder` component analyzes the game sequence to identify and extract positions that meet the "Difficulty Spike" criteria.

#### **4. Component Design**

##### **`DifficultyAssessor`**

This class finds the difficulty of a move by performing a binary search over the full range of player ranks.

*   **Initialization:** `__init__(self, human_controller, strong_controller)`
    *   Takes two KataGoController instances: `human_controller` (SL Network) and `strong_controller` (RL Network).
    *   Defines a single, comprehensive, ordered list of all ranks supported by the engine. This is the full spectrum required to enable the binary search. According to the `katago analysis doc`, these ranks require a prefix (e.g., `"rank_"`).

    ```python
    # This list represents the full, ordered space that can be searched.
    # Note the required "rank_" prefix for each profile.
    self._SUPPORTED_RANKS = [
        "rank_20k", "rank_19k", "rank_18k", "rank_17k", "rank_16k",
        "rank_15k", "rank_14k", "rank_13k", "rank_12k", "rank_11k",
        "rank_10k", "rank_9k", "rank_8k", "rank_7k", "rank_6k",
        "rank_5k", "rank_4k", "rank_3k", "rank_2k", "rank_1k",
        "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d",
        "rank_6d", "rank_7d", "rank_8d", "rank_9d"
    ]
    ```

*   **Helper Method:** `_check_rank_solves(self, rank_string, moves, player, correct_move)`
    *   **Purpose:** To check if a single, specific rank can solve the problem.
    *   **Process:**
        1.  **Build Query:** Construct a JSON query for the `human_controller`.
            *   Set `maxVisits` to `1`.
            *   Use the **`overrideSettings`** object to set the `humanSLProfile`. The parameter name `overrideSettings` is used.
        2.  **Execute Query:** Get the SL network's intuitive move proposals from the `humanPolicy` field in the response.
        3.  **Check Solution:** Check if the `correct_move` is among the top few proposals and is verified as non-losing by the `strong_controller`.
    *   **Output:** Returns `True` if the rank solves the problem, `False` otherwise.

*   **Primary Method:** `assess_move_difficulty(self, moves, player_to_move, correct_outcome)`
    *   **Purpose:** To find the precise rank that is the threshold for solving the problem.
    *   **Process:**
        1.  Initialize binary search pointers: `low = 0`, `high = len(self._SUPPORTED_RANKS) - 1`.
        2.  Initialize `best_found_rank = None`.
        3.  Perform a binary search:
            *   While `low <= high`, select `mid` index and its corresponding `rank_string`.
            *   Call `self._check_rank_solves` with the `rank_string`.
            *   If it returns `True` (the rank solved it), then the solution might be found by an even weaker rank. Store this rank as the current best (`best_found_rank = rank_string`) and search the weaker half: `high = mid - 1`.
            *   If it returns `False`, a stronger rank is needed. Search the stronger half: `low = mid + 1`.
    *   **Output:** Returns the `best_found_rank` found during the search.

#### **5. Correct Query Example**

To assess if a 5 kyu player would find the correct move, the following JSON query should be sent to the KataGo analysis engine. This example correctly uses the `overrideSettings` object and the full `humanSLProfile` name.

```json
{
    "id": "difficulty_check_yose_101",
    "moves": [["B", "A4"], ["W", "B3"], ["B", "C2"]],
    "initialStones": [],
    "rules": "tromp-taylor",
    "komi": 7.5,
    "boardXSize": 9,
    "boardYSize": 9,
    "maxVisits": 1,
    "includePolicy": true,
    "overrideSettings": {
        "humanSLProfile": "rank_5k"
    }
}
```

#### **6. Project Implementation Stages**

This project will be implemented in the following stages:

1.  **Stage 1: Core Engine Integration**
    *   Develop a Python wrapper for the KataGo analysis engine.
    *   Implement functionality to send queries and parse responses.
    *   Load and manage the RL and SL models.

2.  **Stage 2: Game Generation**
    *   Implement the `GroundTruthGenerator` class.
    *   Use the RL network to generate optimal game sequences.
    *   Store game records in a suitable format (e.g., SGF).

3.  **Stage 3: Difficulty Assessment**
    *   Implement the `DifficultyAssessor` class.
    *   Perform binary search over ranks to find move difficulty.
    *   Implement the `_check_rank_solves` helper method.

4.  **Stage 4: Problem Identification**
    *   Implement the `ProblemFinder` class.
    *   Analyze game sequences for "difficulty spikes".
    *   Extract and store identified problems.

5.  **Stage 5: User Interface and Output**
    *   Design and implement a simple command-line interface (CLI).
    *   Allow users to specify parameters (e.g., board size, number of problems).
    *   Output problems in a user-friendly format (e.g., SGF with annotations).
