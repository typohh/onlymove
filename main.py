import os
import argparse
import traceback
import uuid

from katago_controller import KatagoController
from problem_finder import ProblemFinder
import game_generator

# Define paths - adjust these if your KataGo executable or models are elsewhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KATAGO_PATH = os.path.join(BASE_DIR, "KataGo", "cpp", "katago")
SL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "b18c384nbt-humanv0.bin.gz")
RL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")

def main():
    parser = argparse.ArgumentParser(description="Generate Go problems using KataGo.")
    parser.add_argument("--verbose", action="store_true", help="Show KataGo engine output")
    parser.add_argument("--game", type=str, help="Use existing game to create the tsumegos")
    args = parser.parse_args()

    # Always use existing game.sgf if present
    controller = None
    try:
        print("Starting KataGo engines...")
        controller = KatagoController(KATAGO_PATH, RL_MODEL_PATH, human_model_path=SL_MODEL_PATH)
        controller.start_engine(verbose=args.verbose)
        print("KataGo engines started.")

        problem_finder = ProblemFinder(controller, controller)

        moves = []

        if not args.game:
            print("Generating new game...")
            game_output_dir = "./generated_games"
            os.makedirs(game_output_dir, exist_ok=True)
            game_moves = game_generator.generate_game(controller)
            game_filename = os.path.join(game_output_dir, f"{uuid.uuid4()}.sgf")
            game_generator.save_sgf(game_moves, game_filename, 9, 7.5)
            print(f"Generated game and saved to {game_filename}")
        else:
            print(f"Loading game for analysis from {args.game}...")
            game_moves = game_generator.load_sgf(args.game)
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
        traceback.print_exc()
    finally:
        print("Stopping KataGo engines...")
        if controller:
            controller.stop_engine()
        print("KataGo engines stopped.")

if __name__ == "__main__":
    main()
