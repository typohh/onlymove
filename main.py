import os
import sys
import argparse
import traceback

from katago_controller import KatagoController
from difficulty_assessor import DifficultyAssessor
from problem_finder import ProblemFinder

# Define paths - adjust these if your KataGo executable or models are elsewhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KATAGO_PATH = os.path.join(BASE_DIR, "KataGo", "cpp", "katago")
SL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "b18c384nbt-humanv0.bin.gz")
RL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")

def main():
    parser = argparse.ArgumentParser(description="Generate Go problems using KataGo.")
    parser.add_argument("--verbose", action="store_true", help="Show KataGo engine output")
    # TODO: add --game argument to parser, which can take an sfd filename, help should be ""Use existing game to create the tsumegos"
    args = parser.parse_args()

    # Always use existing game.sgf if present
    human_controller = None
    strong_controller = None
    try:
        print("Starting KataGo engines...")
        human_controller = KatagoController(KATAGO_PATH, SL_MODEL_PATH, human_model_path=SL_MODEL_PATH)
        strong_controller = KatagoController(KATAGO_PATH, RL_MODEL_PATH)

        human_controller.start_engine(verbose=args.verbose)
        strong_controller.start_engine(verbose=args.verbose)
        print("KataGo engines started.")

        difficulty_assessor = DifficultyAssessor(human_controller, strong_controller)

        moves = []

        if _ :# TODO: if args doesnt have a game specificed, create one..
            print("Generating new game...")
            game_output_dir = "./generated_games"
            os.makedirs(game_output_dir, exist_ok=True)
            game_moves = game_generator.generate_game()
            game_filename = # TODO: create random filename .sgf, in the game output dir
            game_generator.save_sgf(game_moves,game_filename,9,7.5)
            print(f"Generated game and saved to {game_filename}")
        else :
            print("Loading game for analysis...")
            game_moves = game_generator.load_sgf(_) # TODO use the filename provided by argparser for this purpose
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
        if human_controller:
            human_controller.stop_engine()
        if strong_controller:
            strong_controller.stop_engine()
        print("KataGo engines stopped.")

if __name__ == "__main__":
    main()
