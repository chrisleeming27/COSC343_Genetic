__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

import numpy as np

agentName = "<my_agent>"
trainingSchedule = [("random_agent.py", 5), ("self", 1)]    # Train against random agent for 5 generations,
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

        for _ in range(population_size):
            cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)
            population.append(cleaner)



    def AgentFunction(self, percepts):
        visual, energy, bin, fails = percepts



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

    # At this point you should sort the old_population cleaners according to fitness, setting it up for parent
    # selection.
    # .
    # .
    # .

    # Create new population list...
    new_population = list()
    for n in range(N):

        # Create a new cleaner
        new_cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)

        # Here you should modify the new cleaner' chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_cleaner.chromosome

        # Consider implementing elitism, mutation and various other
        # strategies for producing a new creature.

        # .
        # .
        # .

        # Add the new cleaner to the new population
        new_population.append(new_cleaner)

    # At the end you need to compute the average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)

    return (new_population, avg_fitness)
