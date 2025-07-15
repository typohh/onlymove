import os
import sys
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from katago_controller import KatagoController
from game_generator import GroundTruthGenerator
from difficulty_assessor import DifficultyAssessor
from problem_finder import ProblemFinder

# Define paths - adjust these if your KataGo executable or models are elsewhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KATAGO_PATH = os.path.join(BASE_DIR, "KataGo", "cpp", "katago")
SL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "b18c384nbt-humanv0.bin.gz")
RL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")

def main():
    # Check if running in virtual environment
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    if not in_venv:
        print("ERROR: Please run this script inside a virtual environment as per README.md instructions")
        print("You can either:")
        print("1. Activate the virtual environment: source .venv/bin/activate")
        print("   then run: python3 main.py")
        print("2. Or run directly using: .venv/bin/python3 main.py")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate Go problems using KataGo.")
    parser.add_argument("--verbose", action="store_true", help="Show KataGo engine output")
    args = parser.parse_args()
    args.board_size = 9  # Fixed board size
    args.komi = 7.5      # Fixed komi

    # Always use existing game.sgf if present
    game_filename = "./generated_games/game.sgf"
    human_controller = None
    strong_controller = None
    try:
        print("Initializing KataGo controllers...")
        human_controller = KatagoController(KATAGO_PATH, SL_MODEL_PATH, human_model_path=SL_MODEL_PATH)
        strong_controller = KatagoController(KATAGO_PATH, RL_MODEL_PATH)

        print("Starting KataGo engines...")
        human_controller.start_engine(verbose=args.verbose)
        strong_controller.start_engine(verbose=args.verbose)
        print("KataGo engines started.")

        print("Initializing game generator...")
        game_generator = GroundTruthGenerator(strong_controller)

        print("Initializing difficulty assessor...")
        difficulty_assessor = DifficultyAssessor(human_controller, strong_controller)

        print("Initializing problem finder...")
        problem_finder = ProblemFinder(difficulty_assessor, game_generator)

        if os.path.exists(game_filename):
            print(f"Using existing game file: {game_filename}")
        else:
            print("Generating new game...")
            game_output_dir = "./generated_games"
            os.makedirs(game_output_dir, exist_ok=True)
            game_generator.generate_game(board_size=args.board_size, komi=args.komi, filename=game_filename)
            print(f"Generated game and saved to {game_filename}")

        print("Loading game for analysis...")
        game_moves = problem_finder.load_sgf(game_filename)
        print(f"Loaded game with {len(game_moves)} moves.")

        print("Finding problems...")
        problems = problem_finder.find_problems(game_moves)
        print(f"Found {len(problems)} problems.")

        if problems:
            output_dir = "./generated_problems"
            os.makedirs(output_dir, exist_ok=True)

            # Save problems to JSON
            json_filename = os.path.join(output_dir, "problems.json")
            problem_finder.save_problems(problems, json_filename)
            print(f"Problems saved to {json_filename}")

            # Save each problem as an SGF
            for i, problem in enumerate(problems):
                sgf_filename = os.path.join(output_dir, f"problem_{i+1}.sgf")
                problem_finder.save_problem_as_sgf(problem, sgf_filename)
                print(f"Problem {i+1} saved to {sgf_filename}")
        else:
            print("No problems found.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping KataGo engines...")
        if human_controller:
            human_controller.stop_engine()
        if strong_controller:
            strong_controller.stop_engine()
        print("KataGo engines stopped.")

if __name__ == "__main__":
    main()
