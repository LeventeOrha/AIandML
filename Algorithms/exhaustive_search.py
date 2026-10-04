"""
Functiontionality of exhaustive search
"""
import itertools
import operator
from typing import Callable

class ExhaustiveSearch:
    """
    Exhaustive search algorithm
    """
    def __init__(self, evaluate_fnc: Callable[[list[int]], int], direction: str):
        """
        Initialize an Exhaustive Search Algorithm

        Parameters
        ---------
        evaluate_fnc: `function(list[float/int])`
            Function to evaluate the results
        direction: `str` - [min, max]
            Wheter it is a minimization or maximization problem
        """
        self.evaluate_fnc = evaluate_fnc
        if direction == "min":
            self.operator = operator.lt
            self.best_value = float('inf')
            self.original_goal = float('inf')
        elif direction == "max":
            self.operator = operator.gt
            self.best_value = float('-inf')
            self.original_goal = float('-inf')
        else:
            raise NotImplementedError(f"Only minimization and maximization problems are implemented, {direction} is not.")

    def search(self, data: list, starting_point = 0):
        """
        Search for the best version by re-arranging the list
        """
        self.best_option = data
        self.best_value = self.original_goal

        for option in itertools.permutations(data):
            option = (starting_point, ) + option
            value = self.evaluate_fnc(option)
            if self.operator(value, self.best_value):
                self.best_value = value
                self.best_option = option

        return self.best_option, self.best_value