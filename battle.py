import subprocess
import sys

# --- Configuration ---
# Replace with the actual commands to run your bot engines
BOT1_COMMAND = ["KataGo/cpp/katago", "GTP", "-model", "weights/kata9x9-b18c384nbt-20231025.bin.gz", "-human-model", "weights/b18c384nbt-humanv0.bin.gz", "-config",]
BOT2_COMMAND = ["Pachi-12.86", "--gtp"]
BOARD_SIZE = 9


class GTPBot:
    """A class to manage and communicate with a Go bot process using GTP."""

    def __init__(self, command, bot_name="Bot"):
        """
        Initializes the bot process.
        :param command: The command to execute the bot engine.
        :param bot_name: A name for the bot for logging purposes.
        """
        self.bot_name = bot_name
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

    def send_command(self, command):
        """
        Sends a GTP command to the bot and gets the response.
        :param command: The GTP command string (e.g., "genmove b").
        :return: The bot's response.
        """
        if self.proc.stdin:
            self.proc.stdin.write(command + '\n')
            self.proc.stdin.flush()

        response = ""
        while True:
            line = self.proc.stdout.readline().strip()
            if not line:
                break
            if line.startswith('='):
                response += line[1:].strip()
                break
            elif line.startswith('?'):
                print(f"Error from {self.bot_name}: {line[1:].strip()}", file=sys.stderr)
                break
            else:
                response += line + "\n"
        return response

    def close(self):
        """Terminates the bot process."""
        if self.proc:
            self.proc.terminate()


def play_game(black_bot, white_bot):
    """
    Plays a single game of Go between two bots.
    :param black_bot: The GTPBot instance for the black player.
    :param white_bot: The GTPBot instance for the white player.
    :return: The winner ("B" or "W").
    """
    black_bot.send_command(f"boardsize {BOARD_SIZE}")
    white_bot.send_command(f"boardsize {BOARD_SIZE}")
    black_bot.send_command("clear_board")
    white_bot.send_command("clear_board")

    passes = 0
    current_player, opponent, color = black_bot, white_bot, "b"

    while passes < 2:
        move = current_player.send_command(f"genmove {color}")

        if "resign" in move.lower():
            return "W" if color == "b" else "B"
        if "pass" in move.lower():
            passes += 1
        else:
            passes = 0

        opponent.send_command(f"play {color} {move}")
        current_player, opponent = opponent, current_player
        color = "w" if color == "b" else "b"

    score_str = black_bot.send_command("final_score")
    if score_str.startswith("B+"):
        return "B"
    else:
        return "W"


def main():
    """Main function to run the bot matches."""
    bot1_wins = 0
    bot2_wins = 0

    while True:
        print(f"\n--- Starting Game {i + 1} ---")
        bot1 = GTPBot(BOT1_COMMAND, "Bot1")
        bot2 = GTPBot(BOT2_COMMAND, "Bot2")

        if i % 2 == 0:
            print("Bot 1 is Black, Bot 2 is White")
            winner = play_game(bot1, bot2)
            if winner == "B":
                bot1_wins += 1
                print("Game winner: Bot 1")
            else:
                bot2_wins += 1
                print("Game winner: Bot 2")
        else:
            print("Bot 2 is Black, Bot 1 is White")
            winner = play_game(bot2, bot1)
            if winner == "B":
                bot2_wins += 1
                print("Game winner: Bot 2")
            else:
                bot1_wins += 1
                print("Game winner: Bot 1")

        bot1.close()
        bot2.close()

        total_games = bot1_wins + bot2_wins
        bot1_winrate = bot1_wins / total_games

        print(f"--- Results after {total_games} games ---")
        print(f"Winrate {(100 * bot1_winrate):.2%} after {total_games} games.")


if __name__ == "__main__":
    main()
