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
        # Pre-allocate difficulties array
        difficulties = [None] * len(game_moves)
        adjusted_moves_list = [None] * len(game_moves)

        # Analyze all moves except first 10 and last 10
        for i in range(10, len(game_moves) - 10):
            if game_moves[i][0] != "B":
                continue
            moves_so_far = game_moves[:i]
            player_to_move = game_moves[i][0]
            correct_move = game_moves[i][1]

            print(f"\nValidating tsumego at move {i+1}/{len(game_moves)}...")
            is_valid, adjusted_score, adjusted_moves, solution_move = self.difficulty_assessor.validate_and_adjust_tsumego(moves_so_far, player_to_move, correct_move)
            if not is_valid or not solution_move:
                print(f"  ❌ Not a valid tsumego.")
                continue

            print(f"  ✅ Valid tsumego. Adjusted score: {adjusted_score:.1f}, Solution: {solution_move}")
            print(f"  Assessing difficulty on adjusted position...")
            difficulty = self.difficulty_assessor.assess_move_difficulty(adjusted_moves, player_to_move, correct_move)
            difficulties[i] = difficulty
            adjusted_moves_list[i] = adjusted_moves
            print(f"  Difficulty: {difficulty}")

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
        for player, move in problem["moves"]:
            node = game.extend_main_sequence()
            node.set_move(player.lower(), self.game_generator._katago_to_sgf_coords(move, board_size))

        # Add the solution move and annotations
        solution_player, solution_move = problem["solution"]
        solution_node = game.extend_main_sequence()
        solution_node.set_move(solution_player.lower(), self.game_generator._katago_to_sgf_coords(solution_move, board_size))
        solution_node.set("C", f"Solution move for {problem["difficulty"]} problem.")
        solution_node.set("MA", [self.game_generator._katago_to_sgf_coords(solution_move, board_size)])

        with open(filename, "wb") as f:
            f.write(game.serialise())

    def _get_position_score(self, moves):
        """Helper to get the current position score from strong network"""
        query = {
            "id": "position_score",
            "moves": moves,
            "initialStones": [],
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 100
        }
        response = self.difficulty_assessor.strong_controller.__send_query(query)
        if response and "rootInfo" in response:
            return response["rootInfo"]["scoreLead"]
        return None

    def _is_significant_increase(self, rank1, rank2):
        """
        Checks if there is a significant change in difficulty between two ranks.
        This is a simple implementation and can be improved.
        """
        if rank1 is None or rank2 is None:
            return False

        # For simplicity, we can compare the index of the ranks in the supported ranks list.
        # A larger index means a higher rank (more difficult).
        try:
            index1 = self.difficulty_assessor._SUPPORTED_RANKS.index(rank1)
            index2 = self.difficulty_assessor._SUPPORTED_RANKS.index(rank2)
            return abs(index2 - index1) > 1 # A jump of more than 1 rank is significant
        except ValueError:
            return False

    def load_sgf(self, filename):
        """
        Loads a game from an SGF file.

        Args:
            filename (str): The name of the SGF file.

        Returns:
            list: A list of moves representing the game sequence.
        """
        with open(filename, 'rb') as f:
            sgf_game = sgf.Sgf_game.from_bytes(f.read())

        moves = []
        for node in sgf_game.get_main_sequence():
            player, move = node.get_move()
            if player is not None:
                moves.append([player.upper(), self._sgf_to_katago_coords(move, sgf_game.get_size())])
        return moves

    def _sgf_to_katago_coords(self, sgf_move, board_size):
        """
        Converts SGF coordinates to KataGo coordinates.

        Args:
            sgf_move (tuple): The move in SGF coordinate format (row, col).
            board_size (int): The size of the board.

        Returns:
            str: The move in KataGo's coordinate format (e.g., 'A1', 'B2').
        """
        if sgf_move is None:
            return "pass"

        row, col = sgf_move
        col_str = chr(ord('A') + col)
        if col >= 8:
            col_str = chr(ord('A') + col + 1)
        row_str = str(board_size - row)
        return col_str + row_str
