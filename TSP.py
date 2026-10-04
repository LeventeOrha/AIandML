"""
Otherwise needed functions, specifically for the Travelling Salesman Problem
"""
import csv
import timeit
import numpy as np
from typing import Callable, Any
import matplotlib.pyplot as plt

class TSP:
    def __init__(self, source_file: str, europe_map: str):
        with open(source_file, "r") as f:
            self.data = list(csv.reader(f, delimiter=';'))
            self.cities = self.data[0]

        self.europe_map = plt.imread(europe_map)

        self.city_coords = {
            "Barcelona": [2.154007, 41.390205], "Belgrade": [20.46, 44.79], "Berlin": [13.40, 52.52], 
            "Brussels": [4.35, 50.85], "Bucharest": [26.10, 44.44], "Budapest": [19.04, 47.50],
            "Copenhagen": [12.57, 55.68], "Dublin": [-6.27, 53.35], "Hamburg": [9.99, 53.55], 
            "Istanbul": [28.98, 41.02], "Kyiv": [30.52, 50.45], "London": [-0.12, 51.51], 
            "Madrid": [-3.70, 40.42], "Milan": [9.19, 45.46], "Moscow": [37.62, 55.75],
            "Munich": [11.58, 48.14], "Paris": [2.35, 48.86], "Prague": [14.42, 50.07],
            "Rome": [12.50, 41.90], "Saint Petersburg": [30.31, 59.94], "Sofia": [23.32, 42.70],
            "Stockholm": [18.06, 60.33], "Vienna": [16.36, 48.21], "Warsaw": [21.02, 52.24]}

    def tour_length(self, city_order: list[int]):
        """
        Calculate the length of a tour, based on the source file.
        IMPORTANT - This also takes into accout going back to the original starting point.
        """
        total_length = 0.0

        cities = np.array(self.data[1:], dtype=float) # Cut off city names

        for i in range(len(city_order)-1):
            total_length += cities[city_order[i],city_order[i+1]]

        total_length += cities[city_order[-1], city_order[0]]

        return total_length

    def plot_plan(self, city_order: list[str]):
        """
        Plot a tour on the Europe map
        Code taken from description
        """
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.europe_map, extent=[-14.56, 38.43, 37.697 + 0.3, 64.344 + 2.0], aspect='auto')

        # Map (long, lat) to (x, y) for plotting
        for index in range(len(city_order) - 1):
            current_city_coords = self.city_coords[city_order[index]]
            next_city_coords = self.city_coords[city_order[index+1]]
            x, y = current_city_coords[0], current_city_coords[1]

            # Plotting a line to the next city
            next_x, next_y = next_city_coords[0], next_city_coords[1]
            plt.plot([x, next_x], [y, next_y])

            plt.plot(x, y, 'ok', markersize=5)
            plt.text(x, y, index, fontsize=12)

        # Finally, plotting from last city to first
        first_city_coords = self.city_coords[city_order[0]]
        first_x, first_y = first_city_coords[0], first_city_coords[1]
        plt.plot([next_x, x], [next_y, y])

        # Plotting a marker and index for the final city
        plt.plot(next_x, next_y, "ok", markersize=5)
        plt.text(next_x, next_y, index+1, fontsize=12)

        plt.show()

    def plot_fitness(self, fitnesses: list[list[float]], labels: list[str]):
        """
        Plot fitness across generations, for several fitness logs

        fitnesses: `list[list[float]]`
            [[fitness]across generations]several runs (assuming each have the same number of generations)
        labels: `list[str]`
            Labels for each run
        """
        generations = range(len(fitnesses[0]))

        colors = ["g-", "b--", "k:", "r.-"]

        for i in range(len(fitnesses)):
            fitness = fitnesses[i]
            label = labels[i]

            plt.plot(generations, fitness, colors[i], label=label)

        plt.legend(title="Population size")
        plt.xlabel("Generation")
        plt.ylabel("Fitness [km]")
        plt.tight_layout()
        plt.show()

    def report_runs(self, runs: list[tuple[list[int], float]]):
        """
        Report best, worst, mean and standard deviation over a set of runs
        
        runs: `list[tuple[list[int], float]]`
            For each run, store the best order (list of ints) and the total lengt (float) in a tuple
        """
        best = {"Order": None, "Length": float("inf")}
        worst = {"Order": None, "Length": float("-inf")}

        lengths = [] # Store the lengthes of each run

        for run in runs:
            length = run[1]
            lengths.append(length)
            if best["Length"] > length:
                best["Length"] = length
                best["Order"] = run[0]
            if worst["Length"] < length:
                worst["Length"] = length
                worst["Order"] = run[0]
        print(f"Best order: {best['Order']} with length of {best['Length']:.2f} km.")
        print(f"Worst order: {worst['Order']} with length of {worst['Length']:.2f} km.")
        print(f"Mean of lengths: {np.mean(lengths):.2f}, with standard deviation of {np.std(lengths):.2f}")

        return best, worst

    def time_function(self, title: str, func: Callable[..., Any], *args, **kwargs):
        start = timeit.default_timer()

        result = func(*args, **kwargs)

        elapsed = timeit.default_timer() - start
        print(f"{title} - {elapsed:.6f} s")

        return result