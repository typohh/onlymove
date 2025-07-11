### **Task List: Go Problem Generation**

**My Instructions:**

I will implement the tasks one at a time. When a task is complete and tested, I will mark it as done. If a task seems too large, I will break it down into smaller subtasks and implement those individually. If I encounter any issues, or think of improvements, I will add them to this task list.

This task list is based on the stages outlined in the `design.md` document.

#### **Stage 1: Core Engine Integration**

*   [ ] Develop a Python wrapper for the KataGo analysis engine.
    *   [x] Create a class to manage the KataGo process.
    *   [x] Implement methods for sending JSON queries.
    *   [x] Implement methods for parsing JSON responses.
*   [x] Load and manage the RL and SL models.
    *   [x] Implement logic to specify and load the correct model files.
    *   [x] Handle potential errors during model loading.

#### **Stage 2: Game Generation**

*   [x] Implement the `GroundTruthGenerator` class.
    *   [x] Design the class structure.
    *   [x] Implement the game generation loop.
*   [x] Use the RL network to generate optimal game sequences.
    *   [x] Configure the RL engine for self-play.
    *   [x] Set appropriate parameters for high-quality game generation.
*   [x] Store game records in a suitable format (e.g., SGF).
    *   [x] Choose a Python SGF library.
    *   [x] Implement logic to convert game data to SGF format.
    *   [x] Save SGF files to a designated directory.

#### **Stage 3: Difficulty Assessment**

*   [x] Implement the `DifficultyAssessor` class.
    *   [x] Define the class structure and initialization.
    *   [x] Store the list of supported ranks.
*   [x] Perform binary search over ranks to find move difficulty.
    *   [x] Implement the binary search algorithm.
    *   [x] Call the helper method to check rank performance.
*   [x] Implement the `_check_rank_solves` helper method.
    *   [ ] Construct the JSON query with `overrideSettings`.
    *   [ ] Send the query to the SL network.
    *   [ ] Parse the response and check the `humanPolicy` field.
    *   [ ] Verify the proposed move with the RL network.

#### **Stage 4: Problem Identification**

*   [x] Implement the `ProblemFinder` class.
    *   [x] Design the class structure.
    *   [ ] Load and parse generated game records.
*   [x] Analyze game sequences for "difficulty spikes".
    *   [x] Iterate through moves in reverse order.
    *   [x] Use the `DifficultyAssessor` to get the difficulty of each move.
    *   [x] Implement the "Entry Condition" and "Crux Condition" logic.
*   [x] Extract and store identified problems.
    *   [x] Create a data structure to represent a problem.
    *   [x] Save problems to a file (e.g., JSON, YAML).

#### **Stage 5: User Interface and Output**

*   [x] Design and implement a simple command-line interface (CLI).
    *   [x] Use a library like `argparse` to handle command-line arguments.
    *   [x] Define arguments for user-configurable parameters.
*   [x] Allow users to specify parameters (e.g., board size, number of problems).
    *   [ ] Implement logic to pass user input to the appropriate components.
*   [x] Output problems in a user-friendly format (e.g., SGF with annotations).
    *   [x] Add annotations to SGF files to explain the problem and solution.
    *   [x] Provide clear output to the user indicating the location of the generated problems.
