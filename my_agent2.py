__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

import numpy as np
import random

agentName = "<my_agent>"
trainingSchedule = [("random_agent.py", 200), ("self", 200)]    # Train against random agent for 5 generations,
                                                            # then against self for 1 generation

# This is the class for your cleaner/agent
class Cleaner:

    def __init__(self, nPercepts, nActions, gridSize, maxTurns):
        # This is where agent initialisation code goes (including setting up a chromosome with random values)

        # Leave these variables as they are, even if you don't use them in your AgentFunction - they are
        # needed for initialisation of children Cleaners.
        self.nPercepts = nPercepts
        self.nActions = nActions
        self.gridSize = gridSize
        self.maxTurns = maxTurns

        population_size = 100  # Adjust the size of your population as needed
        population = []
        import random

        class Cleaner:

            def __init__(self, nPercepts, nActions, gridSize, maxTurns):
                # Set the properties
                self.nPercepts = nPercepts
                self.nActions = nActions
                self.gridSize = gridSize
                self.maxTurns = maxTurns

                # Initialize the chromosome with random values between 0 and 1
                self.chromosome = [random.uniform(0, 1) for _ in range(nPercepts)]

            # Other methods and functions for the Cleaner class

        for _ in range(population_size):
            cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)
            print(cleaner.chromosome)
            population.append(cleaner)



    def AgentFunction(self, percepts):
            visual, energy, bin, fails = percepts
            actions = []

            # Example logic:
            if energy < 0.5:
                actions.append("recharge")
            elif bin > 0.8:
                actions.append("empty")
            else:
                # Use chromosome values to make dynamic decisions
                # For example, if the first value in the chromosome is greater than 0.5, move forward
                if self.chromosome[0] > 0.5:
                    actions.append("move_forward")
                else:
                    actions.append("turn_left")

            return actions


def evalFitness(population):

    N = len(population)

    # Fitness initialiser for all agents
    fitness = np.zeros((N))

    # This loop iterates over your agents in the old population - the purpose of this boilerplate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent
    for n, cleaner in enumerate(population):
        # cleaner is an instance of the Cleaner class that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, each object have 'game_stats' attribute provided by the
        # game engine, which is a dictionary with the following information on the performance of the cleaner in
        # the last game:
        #
        cleaned = cleaner.game_stats['cleaned']
        emptied = cleaner.game_stats['emptied']
        num_turns = cleaner.game_stats['active_turns']
        num_succ = cleaner.game_stats['successful_actions']
        num_recharge = cleaner.game_stats['recharge_count']
        energy_recharge = cleaner.game_stats['recharge_energy']
        visited = cleaner.game_stats['visits']

        #define weights of each stat
        cleaning_weight = 1
        emptied_weight = 1
        energy_weight = 0.5
        visited_weight = 0.2

        fitness[n] = (cleaning_weight * cleaned) + (emptied_weight * emptied) + (energy_weight * energy_recharge) + (visited_weight * visited)

    return fitness


def newGeneration(old_population):
    # This function should return a tuple consisting of:
    # - a list of the new_population of cleaners that is of the same length as the old_population,
    # - the average fitness of the old population

    N = len(old_population)

    # Fetch the game parameters stored in each agent (we will need them to
    # create a new child agent)
    gridSize = old_population[0].gridSize
    nPercepts = old_population[0].nPercepts
    nActions = old_population[0].nActions
    maxTurns = old_population[0].maxTurns

    fitness = evalFitness(old_population)

    # Sort the old_population cleaners according to fitness, setting it up for parent selection
    sorted_population = [x for _, x in sorted(zip(fitness, old_population), reverse=True)]

    # Create new population list
    new_population = []

    # Implement elitism (keeping the top-performing agents in the new generation)
    num_elite = int(N * 0.1)  # Adjust the percentage of elite agents as needed

    # Copy the top-performing agents (elites) to the new population
    new_population.extend(sorted_population[:num_elite])

    # Create offspring to fill the rest of the population
    while len(new_population) < N:
        # Select two parents based on their fitness (you can use various selection methods)
        parent1 = selectParent(sorted_population)
        parent2 = selectParent(sorted_population)

        # Apply crossover to create a child's chromosome
        child_chromosome = crossover(parent1.chromosome, parent2.chromosome)

        # Apply mutation to the child's chromosome
        child_chromosome = mutate(child_chromosome)

        # Create a new cleaner with the child's chromosome
        new_cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)
        new_cleaner.chromosome = child_chromosome

        # Add the new cleaner to the new population
        new_population.append(new_cleaner)

    # Calculate the average fitness of the old population
    avg_fitness = np.mean(fitness)

    return (new_population, avg_fitness)

def selectParent(population):
    # Implement parent selection strategy (e.g., roulette wheel, tournament selection)
    # Choose one parent based on their fitness
    # You can experiment with different selection methods
    return random.choice(population)

def crossover(chromosome1, chromosome2):
    # Implement crossover strategy (e.g., single-point, two-point, uniform)
    # Combine two parent chromosomes to create a child chromosome
    # You can experiment with different crossover methods
    crossover_point = random.randint(0, len(chromosome1) - 1)
    child_chromosome = chromosome1[:crossover_point] + chromosome2[crossover_point:]
    return child_chromosome

def mutate(chromosome):
    # Implement mutation strategy
    # Make small random changes to the chromosome
    mutation_rate = 0.1  # Adjust the mutation rate as needed
    mutated_chromosome = chromosome.copy()
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = random.uniform(0, 1)
    return mutated_chromosome
