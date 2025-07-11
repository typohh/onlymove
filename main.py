import argparse
import os

from src.katago_controller import KatagoController
from src.game_generator import GroundTruthGenerator
from src.difficulty_assessor import DifficultyAssessor
from src.problem_finder import ProblemFinder

def main():
    parser = argparse.ArgumentParser(description="Generate Go problems using KataGo.")

    args = parser.parse_args()

    # Hardcoded KataGo paths (assuming KataGo is compiled and models are in place)
    # IMPORTANT: These paths need to be adjusted based on your KataGo installation
    KATAGO_PATH = os.path.join(os.path.dirname(__file__), "KataGo", "cpp", "katago") # Assuming 'katago' executable in KataGo/cpp
    RL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "KataGo", "model", "pro_level_model.bin.gz") # Assuming models in KataGo/model
    SL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "KataGo", "model", "human_sl_model.bin.gz") # Assuming models in KataGo/model

    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tsumegos")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing KataGo controllers...")
    rl_controller = KatagoController(KATAGO_PATH, RL_MODEL_PATH)
    sl_controller = KatagoController(KATAGO_PATH, SL_MODEL_PATH)

    try:
        rl_controller.start_engine()
        sl_controller.start_engine()
        print("KataGo engines started.")

        game_generator = GroundTruthGenerator(rl_controller)
        difficulty_assessor = DifficultyAssessor(sl_controller, rl_controller)
        problem_finder = ProblemFinder(difficulty_assessor, game_generator)

        print(f"Generating game...")
        game_moves = game_generator.generate_game()
        game_generator.save_sgf(game_moves, os.path.join(OUTPUT_DIR, f"game_1.sgf"))
        print(f"Game generated and saved.")

        print(f"Finding problems in game...")
        problems = problem_finder.find_problems(game_moves)
        problem_finder.save_problems(problems, os.path.join(OUTPUT_DIR, f"problems_1.json"))
        print(f"Found {len(problems)} problems and saved.")

        for i, problem in enumerate(problems):
            problem_finder.save_problem_as_sgf(problem, os.path.join(OUTPUT_DIR, f"problem_{i+1}.sgf"))
            print(f"Problem {i+1} saved as SGF.")

    finally:
        rl_controller.stop_engine()
        sl_controller.stop_engine()
        print("KataGo engines stopped.")

if __name__ == "__main__":
    main()
