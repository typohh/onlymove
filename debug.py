import json
from pprint import pprint

from katago_controller import KatagoController
from problem_finder import ProblemFinder, problem_to_sgf

from katago_queries import KatagoQueries


def main():
    controller = KatagoController()
    controller.start_engine(verbose=False)
    queries = KatagoQueries(controller)
    moves = ['D7', 'B3', 'C7', 'F2', 'H2', 'F5', 'F4']
    sgf = problem_to_sgf({
        "moves": moves
    })
    print(sgf)
    r = queries.query_children(moves)
    print(r)


if __name__ == "__main__":
    main()
