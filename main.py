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
    parser = argparse.ArgumentParser(description="Generate Go problems using KataGo.")
    parser.add_argument("--board_size", type=int, default=9, help="The size of the board (e.g., 9, 13, 19).")
    parser.add_argument("--komi", type=float, default=7.5, help="The komi for the game.")
    parser.add_argument("--sgf", type=str, default=None, help="Path to an existing SGF file to analyze.")
    args = parser.parse_args()
    human_controller = None
    strong_controller = None
    try:
        print("Initializing KataGo controllers...")
        human_controller = KatagoController(KATAGO_PATH, SL_MODEL_PATH, human_model_path=SL_MODEL_PATH)
        strong_controller = KatagoController(KATAGO_PATH, RL_MODEL_PATH)

        print("Starting KataGo engines...")
        human_controller.start_engine()
        strong_controller.start_engine()
        print("KataGo engines started.")

        print("Initializing game generator...")
        game_generator = GroundTruthGenerator(strong_controller)

        print("Initializing difficulty assessor...")
        difficulty_assessor = DifficultyAssessor(human_controller, strong_controller)

        print("Initializing problem finder...")
        problem_finder = ProblemFinder(difficulty_assessor, game_generator)

        if args.sgf:
            game_filename = args.sgf
            print(f"Using existing game file: {game_filename}")
        else:
            print("Generating game...")
            game_output_dir = "./generated_games"
            os.makedirs(game_output_dir, exist_ok=True)
            game_filename = os.path.join(game_output_dir, "game.sgf")
            game_generator.generate_game(board_size=args.board_size, komi=args.komi, filename=game_filename)
            print(f"Generated game and saved to {game_filename}.")

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