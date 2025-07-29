# main.py
import time


def load_bot_classes():
    class PlaceholderBotA:
        def name(self): return "AggroBot"

        def start_bot(self, gs): pass

        def add_move(self, m, p): pass

        def get_next_move(self, gs): return 'paper'  # Beats rock

        def stop_bot(self, gs): pass

        def get_winner(self, gs): return gs.get('winner')

    class PlaceholderBotB:
        def name(self): return "Rocky"

        def start_bot(self, gs): pass

        def add_move(self, m, p): pass

        def get_next_move(self, gs): return 'rock'

        def stop_bot(self, gs): pass

        def get_winner(self, gs): return gs.get('winner')

    class PlaceholderBotC:
        def name(self): return "RandomBot"

        def start_bot(self, gs): pass

        def add_move(self, m, p): pass

        def get_next_move(self, gs): import random; return random.choice(['rock', 'paper', 'scissors'])

        def stop_bot(self, gs): pass

        def get_winner(self, gs): return gs.get('winner')

    return {"AggroBot": PlaceholderBotA, "Rocky": PlaceholderBotB, "RandomBot": PlaceholderBotC}


# -------------------------

from persistence import Persistence
from ranking import ProbitRanking
from matchmaking import Matchmaking
from game_runner import GameRunner


def main():
    """Main execution loop for the bot battling framework."""

    bot_classes = load_bot_classes()
    bot_names = list(bot_classes.keys())
    if len(bot_names) < 2:
        print("Error: At least two bots are required.")
        return

    # --- 1. Initialization ---
    db = Persistence()
    ranking_system = ProbitRanking(db, bot_names)
    matchmaking_system = Matchmaking()

    # --- Config ---
    total_battles = 500
    rank_update_interval = 10  # Update ranks every 10 battles for efficiency

    print(f"Starting simulation for {total_battles} games.")
    print(f"Competing bots: {', '.join(bot_names)}")

    # --- 2. Main Battle Cycle ---
    try:
        for i in range(total_battles):
            print(f"\n--- Battle {i + 1}/{total_battles} ---")

            # a. Get aggregated data for matchmaking
            aggregated_results = db.get_aggregated_pair_results()

            # b. Select best pair for battle
            pair = matchmaking_system.get_next_battle(bot_names, aggregated_results)
            bot_a_name, bot_b_name = pair
            print(f"Matchmaking selected: {bot_a_name} vs {bot_b_name}")

            # c. Instantiate and run battle
            bot_a = bot_classes[bot_a_name]()
            bot_b = bot_classes[bot_b_name]()
            runner = GameRunner(bot_a, bot_b)
            winner_name = runner.run_battle()

            # d. Record the result
            if winner_name != "disagreement":
                db.record_battle(bot_a_name, bot_b_name, winner_name)
            else:
                print("Skipping record due to winner disagreement.")

            # e. Update ranks periodically
            if (i + 1) % rank_update_interval == 0:
                print(f"\n--- Updating Ranks (Cycle {i + 1}) ---")
                ranking_system.update_ranks()
                ranking_system.print_leaderboard()
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        # --- 3. Finalization ---
        print("\n--- Final Simulation Results ---")
        ranking_system.update_ranks()  # Perform one last update
        ranking_system.print_leaderboard()
        db.close()
        print("Framework shut down gracefully.")


if __name__ == "__main__":
    main()
    ```