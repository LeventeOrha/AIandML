"""
Functionality of Hill climbing algorithm
"""
import numpy as np
import operator
from typing import Callable

class HillClimb:
    def __init__(self, evaluate_fnc: Callable[[list[int]], int], direction: str, seed: int = None):
        """
        Initialize a Hill Climbing Algorithm

        Parameters
        ---------
        evaluate_fnc: `function(list[float/int])`
            Function to evaluate the results
        direction: `str` - [min, max]
            Wheter it is a minimization or maximization problem
        """
        self.evaluate_fnc = evaluate_fnc
        self.rng = np.random.default_rng(seed)
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

    def search(self, start_order: list, max_steps: int):
        """
        Search for a better solution, by random swap method

        Parameters
        ----------
        start_order: `list`
            Starting order to find a better solution from
        max_steps: `int`
            Maximum steps to make
        """
        self.best_option = start_order
        self.best_value = self.original_goal
        n = len(start_order)

        for _ in range(max_steps):
            # Generate two indexes (They can never be the same)
            a, b = self.rng.choice(n, size=2, replace=False)

            # Swap two elements
            temp = start_order[a]
            start_order[a] = start_order[b]
            start_order[b] = temp

            # Check if this is better or not
            value = self.evaluate_fnc(start_order)
            if self.operator(value, self.best_value):
                self.best_value = value
                self.best_option = start_order

        return self.best_option, self.best_value