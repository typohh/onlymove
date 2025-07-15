import os
import sys
import json
import argparse
from src.katago_controller import KatagoController
from src.problem_finder import ProblemFinder

def main():
    parser = argparse.ArgumentParser(description="Debug KataGo analysis at move 27")
    args = parser.parse_args()

    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KATAGO_PATH = os.path.join(BASE_DIR, "KataGo", "cpp", "katago")
    RL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")
    GAME_FILE = "./generated_games/game.sgf"

    # Initialize problem finder for loading SGF
    problem_finder = ProblemFinder(None, None)

    # Load game moves
    if not os.path.exists(GAME_FILE):
        print(f"Error: Game file not found at {GAME_FILE}")
        return
    game_moves = problem_finder.load_sgf(GAME_FILE)
    
    # Get moves up to move 26 (index 26) for analysis at move 27
    if len(game_moves) < 27:
        print(f"Error: Game only has {len(game_moves)} moves, need at least 27")
        return
    moves_so_far = game_moves[:26]
    
    # Initialize and start KataGo
    controller = KatagoController(KATAGO_PATH, RL_MODEL_PATH)
    try:
        print("Starting KataGo engine...")
        controller.start_engine()
        print("KataGo engine started.")
        
        # Create analysis query
        query = {
            "id": "debug_move_27",
            "moves": moves_so_far,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 100,
            "analysisPVLen": 10
        }
        
        # Send query and print response
        print("Sending analysis query for position after 26 moves...")
        response = controller.send_query(query)
        print("\nAnalysis Output:")
        print(json.dumps(response, indent=2))
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nStopping KataGo engine...")
        controller.stop_engine()
        print("KataGo engine stopped.")

if __name__ == "__main__":
    main()
