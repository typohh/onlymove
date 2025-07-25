import json
from pprint import pprint

from katago_controller import KatagoController
from problem_finder import ProblemFinder, problem_to_sgf

from katago_queries import KatagoQueries


def main():
    controller = KatagoController()
    controller.start_engine(verbose=False)
    queries = KatagoQueries(controller)
    moves = ['D7', 'E4', 'E7', 'B6', 'C8', 'C6', 'C3', 'H6', 'D4', 'D3', 'D5', 'E5', 'E3', 'D2', 'F3', 'D6', 'B4', 'B3', 'C4', 'E6', 'G8']
    sgf = problem_to_sgf({
        "moves": moves
    })
    print(sgf)
    r = queries.query_children(moves)
    print(r)


if __name__ == "__main__":
    main()
