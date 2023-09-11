__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

import random

import numpy as np

agentName = "<my_agent>"
trainingSchedule = [("random_agent.py", 500)]    # Train against random agent for 5 generations,
                                                            # then against self for 1 generation

# This is the class for your cleaner/agent
class Cleaner:

    def __init__(self, nPercepts, nActions, gridSize, maxTurns):
        # This is where agent initialisation code goes (including setting up a chromosome with random values)
        chromosome_size = nPercepts + 1
        self.chromosomes = np.random.uniform(-1, 1, (nActions, chromosome_size))

        # Leave these variables as they are, even if you don't use them in your AgentFunction - they are
        # needed for initialisation of children Cleaners.
        self.nPercepts = nPercepts
        self.nActions = nActions
        self.gridSize = gridSize
        self.maxTurns = maxTurns



    def AgentFunction(self, percepts):

        # The percepts are a tuple consisting of four pieces of information
        #
        # visual - it information of the 3x5 grid of the squares in front and to the side of the cleaner; this variable
        #          is a 3x5x4 tensor, giving four maps with different information
        #          - the dirty,clean squares
        #          - the energy
        #          - the friendly and enemy cleaners that are able to traverse vertically
        #          - the friendly and enemy cleaners that are able to traverse horizontally
        #
        #  energy - int value giving the battery state of the cleaner -- it's effectively the number of actions
        #           the cleaner can still perform before it runs out of charge
        #
        #  bin    - number of free spots in the bin - when 0, there is no more room in the bin - must emtpy
        #
        #  fails - number of consecutive turns that the agent's action failed (rotations always successful, forward or
        #          backward movement might fail if it would result in a collision with another robot); fails=0 means
        #          the last action succeeded.


        visual, energy, bin, fails = percepts

        # You can further break down the visual information

        floor_state = visual[:,:,0]   # 3x5 map where -1 indicates dirty square, 0 clean one
        energy_locations = visual[:,:,1] #3x5 map where 1 indicates the location of energy station, 0 otherwise
        vertical_bots = visual[:,:,2] # 3x5 map of bots that can in this turn move up or down (from this bot's point of
                                      # view), -1 if the bot is an enemy, 1 if it is friendly
        horizontal_bots = visual[:,:,3] # 3x5 map of bots that can in this turn move up or down (from this bot's point
                                        # of view), -1 if the bot is an enemy, 1 if it is friendly

        #You may combine floor_state and energy_locations if you'd like: floor_state + energy_locations would give you
        # a mape where -1 indicates dirty square, 0 a clean one, and 1 an energy station.
        floor = floor_state + energy_locations

        #v=w1x1 +w2x2 +...+w63x63 +b,
        # where w1, w2, ..., w63 are the weights, x1, x2, ..., x63 are the inputs, and b is the bias.
        # The weights and the bias are the chromosome values.
        # The inputs are the flattened visual information (floor_state, energy_locations, vertical_bots, horizontal_bots)
        # concatenated with the energy, bin, and fails values.
        # The output is a vector of 4 values, each of which is the result of the above formula for each of the 4 actions.
        # The action with the largest value is the one that is taken.

        flattened_visual = visual.reshape(-1)

        # Create the input vector by concatenating flattened_floor with energy, bin, and fails
        weights = self.chromosomes[:, :-1]
        biases = self.chromosomes[:, -1]
        x = np.concatenate([flattened_visual, [energy, bin, fails]]) # Here we have combined all of the array values together.

        actions = []

        # We want to return an array of actions with each index representing the score for that action.
        for action in range(self.nActions): # Here we are looping through each action as an index e.g 1...2...3...4
            action_score = 0 # Start with the score equal to 0.
            for index in range(len(x)): #For every index in the input vector
                # Multiply the input vector by the weights
                action_score = action_score + x[index] * self.chromosomes[action][index]
                # Add the bias to the action score
            action_score += self.chromosomes[action][-1]
            # Append the action score to the actions array
            actions.append(action_score)

        return actions

        # You should implement a model here that translates from 'percepts' to 'actions' through 'self.chromosome'.
        # The 'actions' variable must be returned, and it must be a 4-item list or a 4-dim numpy vector

        # for values in self.chromosome:
        #     # Perform element-wise multiplication with self.flattenedVisuals
        #     new_values = np.array(flattenedWithBias) * np.array(values)
        #     # Sum the new_values and store it in the action_vector
        #     actions[i] = np.sum(new_values)
        #     i += 1


        # The index of the largest value in the 'actions' vector/list is the action to be taken,
        # with the following interpretation:
        # largest value at index 0 - move forward;
        # largest value at index 1 - turn right;
        # largest value at index 2 - turn left;
        # largest value at index 3 - move backwards;
        #
        # Different 'percepts' values should lead to different 'actions'.  This way the agent
        # reacts differently to different situations.
        #
        # Different 'self.chromosome' should lead to different 'actions'.  This way different
        # agents can exhibit different behaviour.

        # .
        # .
        # .

        # Right now this agent ignores percepts and chooses a random action.  Your agents should not
        # perform random actions - your agents' actions should be deterministic from
        # computation based on self.chromosome and percepts


def evalFitness(population):

    N = len(population)
    fitness = np.zeros((N))

    for n, cleaner in enumerate(population):
        cleaned = cleaner.game_stats['cleaned']
        emptied = cleaner.game_stats['emptied']
        num_turns = cleaner.game_stats['active_turns']
        num_succ = cleaner.game_stats['successful_actions']
        num_recharge = cleaner.game_stats['recharge_count']
        energy_recharge = cleaner.game_stats['recharge_energy']
        visited = cleaner.game_stats['visits']

        

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

    fitness_population_pairs = list(zip(fitness, old_population))
    sorted_pairs = sorted(fitness_population_pairs, key=lambda x: x[0], reverse=True)

    sorted_population = [pair[1] for pair in sorted_pairs]



    # Create new population list...
    new_population = list()

    #Elitism:
    #Normally //10
    nElitism = len(sorted_population)//10
    for n in range(nElitism):
        new_population.append(sorted_population[n])

    for n in range(N-nElitism):
        # Create a new cleaner
        new_cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)

        # Here you should modify the new cleaner' chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_cleaner.chromosome
        # parent1, parent2 = selectParents(old_population, fitness)
        parent1 = selectParentsTournament(old_population, fitness, 5)
        parent2 = selectParentsTournament(old_population, fitness, 5)

        # Consider implementing elitism, mutation and various other
        # strategies for producing a new creature.
        # .
        # .
        # .
        if n % 10 != 0:
            new_cleaner = crossover(new_cleaner,parent1, parent2, nActions)
            mutate_gene(new_cleaner, nActions)

        # Add the new cleaner to the new population
        new_population.append(new_cleaner)

    # At the end you need to compute the average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)
    return (new_population, avg_fitness)

def mutate_gene(cleaner, nActions):
    #Normally 0.1
    mutation_rate = 0.1
    if random.random() < mutation_rate:
        index = random.randint(0, nActions - 1) #Bias is not mutated
        chromosome = cleaner.chromosomes[index]
        chromosome_size = len(chromosome)
        new_chromosome = np.random.uniform(-1, 1, chromosome_size)
        cleaner.chromosomes[index] = new_chromosome
    return cleaner

# def crossover(cleaner, parent1, parent2,nActions): #One point cross over
#         num_elements = random.randint(0, nActions)
#         child = np.array(parent1.chromosomes[:num_elements].tolist() + parent2.chromosomes[num_elements:].tolist())
#         cleaner.chromosomes = child
#         return cleaner

# def crossover(cleaner, parent1, parent2, nActions): #two point cross over
#     point1 = random.randint(0, nActions - 1)
#     point2 = random.randint(point1, nActions)
#
#     child = np.concatenate([parent1.chromosomes[:point1],
#                             parent2.chromosomes[point1:point2],
#                             parent1.chromosomes[point2:]])
#
#     cleaner.chromosomes = child
#     return cleaner

#

def crossover(cleaner, parent1, parent2, nActions):
    child = np.empty_like(parent1.chromosomes)

    for i in range(nActions):
        gene_from_parent1 = parent1.chromosomes[i]
        gene_from_parent2 = parent2.chromosomes[i]

        if np.isscalar(gene_from_parent1) and np.isscalar(gene_from_parent2):
            child[i] = gene_from_parent1 if random.random() < 0.5 else gene_from_parent2
        else:
            # Handle the case when the genes are sequences (e.g., lists or arrays)
            # Here, we'll do a uniform crossover at a deeper level for each element of the sequence
            gene_length = len(gene_from_parent1)  # Assuming both genes have the same length
            child_gene = []
            for j in range(gene_length):
                child_gene.append(gene_from_parent1[j] if random.random() < 0.5 else gene_from_parent2[j])
            child[i] = np.array(child_gene)  # Convert child gene back to an array

    cleaner.chromosomes = child
    return cleaner


# def crossover(cleaner, parent1, parent2, nActions):
#     crossover_point = random.randint(0, nActions - 1)
#     child_chromosomes = np.concatenate([parent1.chromosomes[:crossover_point], parent2.chromosomes[crossover_point:]])
#     cleaner.chromosomes = child_chromosomes
#     return cleaner

# def selectParentsTournament(population, fitness, tournament_size):
#     selected_parents = []
#
#     for _ in range(len(population)):
#         # Randomly select 'tournament_size' individuals from the population
#         tournament_individuals = random.sample(list(enumerate(population)), tournament_size)
#
#         # Calculate their fitness scores and select the best one
#         tournament_fitness_scores = [fitness[i] for i, _ in tournament_individuals]
#         best_index = tournament_fitness_scores.index(max(tournament_fitness_scores))
#         selected_parents.append(tournament_individuals[best_index][1])
#
#     return selected_parents

# def selectParents(population, fitness):
#     # Calculate the total fitness of the population
#     total_fitness = sum(fitness)
#
#     # Calculate selection probabilities for each individual
#     selection_probs = [fit / total_fitness for fit in fitness]
#
#     # Use roulette wheel selection to choose two parents
#     parent1 = np.random.choice(population, p=selection_probs)
#     parent2 = np.random.choice(population, p=selection_probs)
#
#     return parent1, parent2

def selectParentsTournament(population, fitness, tournament_size):
    # Randomly select 'tournament_size' individuals from the population
    indices = list(range(len(population)))
    selected_indices = random.sample(indices, tournament_size)

    # Identify the individual with the highest fitness among the selected ones
    selected_fitnesses = [fitness[i] for i in selected_indices]
    best_index = selected_indices[np.argmax(selected_fitnesses)]

    return population[best_index]


# def selectParents(population, fitness, tournament_size):
#     selected_parents = []
#
#     for _ in range(2):  # Select two parents
#         # Randomly choose individuals for the tournament
#         tournament = np.random.choice(population, size=tournament_size, replace=False)
#
#         # Calculate the fitness values for the tournament participants
#         tournament_fitness = [fitness[population.index(individual)] for individual in tournament]
#
#         # Choose the individual with the highest fitness in the tournament
#         selected_parent = tournament[np.argmax(tournament_fitness)]
#
#         selected_parents.append(selected_parent)
#
#     return selected_parents