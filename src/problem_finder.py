from src.difficulty_assessor import DifficultyAssessor
from src.game_generator import GroundTruthGenerator
import json
from sgfmill import sgf

class ProblemFinder:
    """
    Finds problems in a game sequence.
    """
    def __init__(self, difficulty_assessor: DifficultyAssessor, game_generator: GroundTruthGenerator):
        """
        Initializes the ProblemFinder.

        Args:
            difficulty_assessor (DifficultyAssessor): The difficulty assessor.
            game_generator (GroundTruthGenerator): The game generator.
        """
        self.difficulty_assessor = difficulty_assessor
        self.game_generator = game_generator

    def find_problems(self, game_moves):
        """
        Finds problems in a game sequence.

        Args:
            game_moves (list): The list of moves in the game.

        Returns:
            list: A list of identified problems.
        """
        problems = []
        difficulties = []

        # First, calculate the difficulty of all moves in the game
        for i in range(len(game_moves)):
            moves_so_far = game_moves[:i]
            player_to_move = game_moves[i][0]
            correct_move = game_moves[i][1]
            difficulty = self.difficulty_assessor.assess_move_difficulty(moves_so_far, player_to_move, correct_move)
            difficulties.append(difficulty)

        # Now, identify problems by looking for difficulty spikes
        for i in range(1, len(difficulties)):
            # Entry Condition
            if self._is_significant_increase(difficulties[i-1], difficulties[i]):
                # Crux Condition
                is_crux = True
                for j in range(i + 1, len(difficulties)):
                    if self._is_significant_increase(difficulties[i], difficulties[j]):
                        is_crux = False
                        break
                if is_crux:
                    problems.append({
                        "moves": game_moves[:i],
                        "solution": game_moves[i],
                        "difficulty": difficulties[i]
                    })

        return problems

    def save_problems(self, problems, filename):
        """
        Saves a list of problems to a JSON file.

        Args:
            problems (list): The list of problems to save.
            filename (str): The name of the JSON file.
        """
        with open(filename, 'w') as f:
            json.dump(problems, f, indent=4)

    def save_problem_as_sgf(self, problem, filename, board_size=9, komi=7.5):
        """
        Saves a single problem as an SGF file with annotations.

        Args:
            problem (dict): The problem dictionary.
            filename (str): The name of the SGF file.
            board_size (int): The size of the board.
            komi (float): The komi for the game.
        """
        game = sgf.Sgf_game(size=board_size)
        game.root.set("KM", str(komi))
        game.root.set("RU", "tromp-taylor")
        game.root.set("GN", f"Problem Difficulty: {problem["difficulty"]}")

        # Add initial moves
        current_node = game.root
        for player, move in problem["moves"]:
            current_node = current_node.extend_main_sequence()
            current_node.set_move(player.lower(), self.game_generator._katago_to_sgf_coords(move, board_size))

        # Add the solution move and annotations
        solution_player, solution_move = problem["solution"]
        solution_node = current_node.extend_main_sequence()
        solution_node.set_move(solution_player.lower(), self.game_generator._katago_to_sgf_coords(solution_move, board_size))
        solution_node.set("C", f"Solution move for {problem["difficulty"]} problem.")
        solution_node.set("MA", [self.game_generator._katago_to_sgf_coords(solution_move, board_size)])

        with open(filename, "wb") as f:
            f.write(game.serialise())

    def _is_significant_increase(self, rank1, rank2):
        """
        Checks if there is a significant increase in difficulty between two ranks.
        This is a simple implementation and can be improved.
        """
        if rank1 is None or rank2 is None:
            return False
            
        # For simplicity, we can compare the index of the ranks in the supported ranks list.
        # A larger index means a higher rank (more difficult).
        try:
            index1 = self.difficulty_assessor._SUPPORTED_RANKS.index(rank1)
            index2 = self.difficulty_assessor._SUPPORTED_RANKS.index(rank2)
            return (index2 - index1) > 3 # A jump of more than 3 ranks is significant
        except ValueError:
            return False
