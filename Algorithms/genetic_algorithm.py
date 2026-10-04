"""
Functionality of a Genetic Algorithm
"""
import numpy as np
from typing import Callable
from hill_climb import HillClimb

class GeneticAlgorithm:
    def __init__(self, evaluate_fnc: Callable[[list[int]], int], tournament_size: int, direction: str, method: str = "", seed: int = None):
        """
        Genetic Algorithm setup

        Parameters
        ---------
        evaluate_fnc: `function(list[float/int])`
            Function to evaluate the results
        tournament_size: `int`
            Number of individuals in each tournament
        direction: `str` - [min, max]
            Wheter it is a minimization or maximization problem
        method: `str` - [L, B]
            Lamarckian or Baldwinian method (leave empty for neither)
        """
        self.evaluate_fnc = evaluate_fnc
        self.rng = np.random.default_rng(seed)
        self.tournament_size = tournament_size
        self.fitness_log = []
        self.method = method

        if method in ["L", "B"]:
            self.HillClimb = HillClimb(evaluate_fnc, direction, seed + 1 if seed is not None else seed)
            # Changing the seed a bit to have a bit of change there

        if direction == "min":
            self.operator = np.min
            self.argop = np.argmin
        elif direction == "max":
            self.operator = np.max
            self.argop = np.argmax
        else:
            raise NotImplementedError(f"Only minimization and maximization problems are implemented, {direction} is not.")

    def evaluate_population(self, population: np.ndarray):
        """
        Get the fitness of each individual in the population
        """
        fitness = []
        for pop in population:
            fitness.append(self.evaluate_fnc(pop))
        return np.array(fitness)

    def parent_selection(self, population: np.ndarray, fitness: np.ndarray):
        """
        Parent selection with tournament method
        """
        # Pick the candidates in the tournament
        idx = self.rng.choice(len(population), size=self.tournament_size, replace=False) # Cannot pick the same twice
        candidates = population[idx]
        cand_fitness = fitness[idx]

        # Select the parent based on the direction
        parent = candidates[self.argop(cand_fitness)]

        return parent

    def pmx(self, parent1: np.ndarray, parent2: np.ndarray, a: int, b: int):
        """
        Partially Mapped Crossover, with cutting points at `a` and `b`.
        """
        # Create an empty child
        child = np.full_like(parent1, 0)

        # Copy the segment
        child[a:b] = parent1[a:b]

        # Map the common parts
        mapping = {parent1[i]: parent2[i] for i in range(a,b)}

        # Fill in the rest of the positions
        for i in list(range(0,a)) + list(range(b, len(parent1))):
            gene = parent2[i]

            while gene in child[a:b]:
                gene = mapping[gene]
            child[i] = gene

        return child

    def crossover(self, parent_a: np.ndarray, parent_b: np.ndarray):
        """
        Create two children from two parents by crossover
        """
        # Create the two cutting points
        a, b = sorted(self.rng.choice(len(parent_a), size=2, replace=False))

        # Create the two children
        child1 = self.pmx(parent_a, parent_b, a, b)
        child2 = self.pmx(parent_b, parent_a, a, b)

        return child1, child2

    def mutation(self, individual: np.ndarray):
        """
        Mutate an individual by swapping method
        """
        # Pick two random indices
        a, b = self.rng.choice(len(individual), size=2, replace=False)

        # Swap
        tmp = individual[a]
        individual[a] = individual[b]
        individual[b] = tmp

        return individual

    def survivor_selection(self, population: np.ndarray, fitness: np.ndarray, offspring: np.ndarray, offspring_fitness: np.ndarray):
        """
        Select survivors, by taking the best from the population and the len(population) - 1 best offspring
        """
        # Preserve the best parent
        elite_idx = self.argop(fitness)
        elite = population[elite_idx].copy()
        elite_fitness = fitness[elite_idx]

        # Fill the remaining with N-1 best offspring
        N = len(population) - 1

        best_idx = np.argsort(offspring_fitness)[:N]

        selected_offspring = np.asarray(offspring)[best_idx]
        selected_fitness = np.asarray(offspring_fitness)[best_idx]

        next_population = np.vstack([elite, selected_offspring])
        next_fitness = np.concatenate([[elite_fitness], selected_fitness])

        return next_population, next_fitness

    def generate_population(self, Nmax: int, population_size: int):
        """
        Generate a population for ordering
        """
        population = []
        for _ in range(population_size):
            individual = np.array(self.rng.choice(Nmax, size=Nmax, replace=False))
            population.append(individual)

        return np.array(population)

    def lifetime_learning(self, individual, max_steps: int = 50):
        """
        Additional learning (Hill-climbing) during a lifetime
        """
        if self.method in ["L", "B"]:
            new_individual, fitness = self.HillClimb.search(individual.copy(), max_steps)
            # Lamarckian method
            if self.method == "L":
                return new_individual, fitness
            # Baldwinian method
            elif self.method == "B":
                return individual, fitness
        else:
            # Result would be a list, I need a single value
            fitness = self.evaluate_population([individual])[0]
            return individual, fitness


    def run_EA(self, population: np.ndarray, generations: int):
        """
        Run the evolutionary algorithm for `generations` generations, on the initial population
        """
        # Track the best fitness in each generation
        self.fitness_log = []

        # Best individual
        self.best_option = None

        # Get an initial fitness
        fitness = self.evaluate_population(population)

        # Save current best fitness
        self.fitness_log.append(self.operator(fitness))

        # Run the generations
        for _ in range(generations):
            # Create offspring
            offspring = []
            offspring_fitness = []

            while len(offspring) < (len(population) - 1):
                # Pick parents
                parent_a = self.parent_selection(population, fitness)
                parent_b = self.parent_selection(population, fitness)

                # Create offspring
                child_a, child_b = self.crossover(parent_a, parent_b)

                # Mutate these offspring
                child_a = self.mutation(child_a)
                child_b = self.mutation(child_b)

                # Additional learning for children
                child_a, fit_a = self.lifetime_learning(child_a)
                child_b, fit_b = self.lifetime_learning(child_b)

                offspring.extend([child_a, child_b])
                offspring_fitness.extend([fit_a, fit_b])

            population, fitness = self.survivor_selection(population, fitness, offspring, offspring_fitness)

            self.fitness_log.append(self.operator(fitness))

            self.best_value = self.operator(fitness)
            self.best_option = population[self.argop(fitness)]

        return self.best_option, self.best_value