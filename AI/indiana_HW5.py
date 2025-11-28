import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Ant import UNIT_STATS
from GameState import *
from AIPlayerUtils import *
import numpy as np
import pickle


# Authors: Indiana Atwood, Luis Perez-Ruiz, Malory Morey
# Date: November 13, 2025


''' CODE FOR ARTIFICIAL NEURAL NETWORK '''

# True will train ANN, False will test ANN
training = False

# Pop file needed for training, not used in submission
#   instead, values are
POP_FILE = "./ANN_weights.txt"

# Activation function, using sigmiod for backpropogation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Generates weights with initial values between -1.0 and 1.0
def initialize_weights(layers):
    weights = {}
    biases = {}

    for i in range(len(layers) - 1):
        weights[i+1] = np.random.uniform(-1, 1, (layers[i+1], layers[i]))
        # Bias is one per output neuron
        biases[i+1] = np.ones((layers[i+1], 1))

    return weights, biases


##
# forward_prop
# Description: Performs forward propogation on ANN (saves weights, biases)
#              by comparing parentState with childState
#
# Parameters:
#   parentState - parent state of given move
#   childState - child state of given move
#
# Return: the ANN's neurons, weights and biases for back prop
## 
def forward_prop(parentState, childState):
    # convert game states into vectors
    parent_vec = encode_state(parentState)
    child_vec  = encode_state(childState)
    input_vec  = np.concatenate([parent_vec, child_vec]).reshape(-1, 1)

    # load last weights if we have them
    weights, biases = load_weights()

    # Using 2 arbitrary hidden layers, 16 neurons each
    if weights is None:
        input_dim = len(input_vec)
        layers = [input_dim, 16, 16, 1]
        weights, biases = initialize_weights(layers)

    # Neurons given label 0 (input layer)
    neurons = {0: input_vec}

    for i in range(1, len(weights) + 1):
        # Z = W*a_prev + b
        Z = np.dot(weights[i], neurons[i-1]) + biases[i]
        neurons[i] = sigmoid(Z)

    return neurons, weights, biases


##
# back_prop
# Description: Performs back propogation on ANN (saves weights, biases)
#
# Parameters:
#   neurons - The neuron values after sigmoid is applied
#   target - Expected value of ANN (provided by utility())
#   weights - Weights of the given neurons
#   biases - Biases of the given neurons
#   alpha - Learning rate (0.05 in this case)
#
# Return: the ANN's updated weights and biases
## 
def back_prop(neurons, target, weights, biases, alpha):
    updated_weights = {}
    updated_biases = {}
    length = len(weights)

    # Error of output layer (predicted - actual)
    #   using 'delta' for ease of computation in next lines
    delta = neurons[length] - target

    for i in range(length, 0, -1):
        # First part of backprop equation, delta*x (rest calculated later)
        updated_weights[i] = np.dot(delta, neurons[i-1].T)
        updated_biases[i] = delta.copy()

        if i > 1:
            # Finds delta value (error * a(1-a), where a is neurons's value)
            delta = np.dot(weights[i].T, delta) * (neurons[i-1] * (1 - neurons[i-1]))

    # Updates weights and biases with formula W + alpha*delta*x (delta*x occurs in earlier loop)
    for i in range(1, length + 1):
        weights[i] -= alpha * updated_weights[i]
        biases[i] -= alpha * updated_biases[i]

    # Save updated weights/biases if needed
    save_weights(weights, biases)

    return weights, biases


# Saves weights (not readable)
def save_weights(weights, biases):
    #write to popfile
    with open(POP_FILE, "wb") as f:
        pickle.dump((weights, biases), f)


# Loads weights (from hard-coded values at the end of file)
def load_weights():
    # Commented out code for actual ANN training (with a .txt file)
    #   instead, using hard-coded values
    return global_weights, global_biases

    # try:
    #     with open(POP_FILE, "rb") as f:
    #         return pickle.load(f)
    # except:
    #     return None, None


##
# encode_state
# Description: Converts a state into an array of decimal values
#
# Parameters:
#   state - The state of the current game
#
# Return: array of decimal values (input array for ANN)
## 
def encode_state(state):
    """
    Convert a GameState into a fixed-length float feature vector.
    The vector is normalized (0–1) so it can feed directly into an ANN.
    """
    player_id = state.whoseTurn

    my_inv = getCurrPlayerInventory(state)
    enemy_inv = getEnemyInv(player_id, state)

    features = []

    # 1. Food counts (normalize by reasonable max, e.g. 20)
    features.append(my_inv.foodCount / 11.0)
    features.append(enemy_inv.foodCount / 11.0)

    # 2. Unit counts (normalize by max 10 of each)
    my_workers  = getAntList(state, player_id, (WORKER,))
    my_soldiers = getAntList(state, player_id, (SOLDIER,))
    my_drones   = getAntList(state, player_id, (DRONE,))

    features.append(len(my_workers)  / 5.0)
    features.append(len(my_soldiers) / 5.0)
    features.append(len(my_drones)   / 5.0)

    # 3. Enemy workers and queen
    enemy_workers = getAntList(state, not player_id, (WORKER,))
    enemy_drones = getAntList(state, not player_id, (DRONE,))
    enemy_soldiers = getAntList(state, not player_id, (SOLDIER,))
    enemy_queen   = enemy_inv.getQueen()

    features.append(len(enemy_workers) / 5.0)

    if enemy_queen:
        features.append(enemy_queen.health / UNIT_STATS[QUEEN][HEALTH])
    else:
        features.append(0.0)

    # 4. Distances
    # Normalize Manhattan distance by board max = 18
    def norm_dist(a, b):
        return (abs(a[0]-b[0]) + abs(a[1]-b[1])) / 18.0

    # Worker to tunnel/food
    tunnel = getConstrList(state, player_id, (TUNNEL,))[0]
    food   = getConstrList(state, None, (FOOD,))[0]

    if my_workers:
        d_to_home = np.mean([norm_dist(w.coords, tunnel.coords) for w in my_workers])
        d_to_food = np.mean([norm_dist(w.coords, food.coords) for w in my_workers])
    else:
        d_to_home = 1.0
        d_to_food = 1.0

    if my_soldiers:
        if enemy_workers:
            d_to_worker = np.mean([norm_dist(e_w.coords, my_soldiers[0].coords) for e_w in enemy_workers])
        else:
            d_to_worker = 1.0
        if enemy_queen:
            d_to_queen = np.mean([norm_dist(s.coords, enemy_queen.coords) for s in my_soldiers])
        else:
            d_to_queen = 1.0
    else:
        d_to_worker = 1.0
        d_to_queen = 1.0

    features.append(d_to_home)
    features.append(d_to_food)
    features.append(d_to_worker)
    features.append(d_to_queen)

    # 5. Ratios and relative strength
    total_my_units = max(len(my_workers)+len(my_soldiers)+len(my_drones), 1)
    total_enemy_units = max(len(enemy_workers)+len(enemy_soldiers)+len(enemy_drones), 1)

    features.append(len(my_soldiers)/total_my_units)  # fraction soldiers
    features.append(len(my_workers)/total_my_units)   # fraction workers
    features.append(total_my_units / (total_my_units + total_enemy_units))  # relative unit strength

    # 6. Threat
    # Count of my units near enemy (within 1-2 tiles)
    threat_radius = 2
    threatened_workers = sum(1 for w in my_workers if any(abs(w.coords[0]-e.coords[0]) + abs(w.coords[1]-e.coords[1])
                                                          <= threat_radius for e in enemy_soldiers+enemy_drones))
    threatened_soldiers = sum(1 for s in my_soldiers if enemy_workers or enemy_queen)
    features.append(threatened_workers / max(len(my_workers), 1))
    features.append(threatened_soldiers / max(len(my_soldiers), 1))

    # Offensive to queen/worker
    offensive = getAntList(state, player_id, (DRONE, SOLDIER, R_SOLDIER))

    if offensive and (enemy_workers or enemy_queen):
        target = enemy_workers[0] if enemy_workers else enemy_queen
        d_target = np.mean([norm_dist(a.coords, target.coords) for a in offensive])
    else:
        d_target = 1.0

    features.append(d_target)

    return np.array(features, dtype=float)

#############################################################################
#############################################################################
#############################################################################

''' FUNCTIONALITY FOR UTILITY '''

# Determines the distance between selected Ant and its target
def getDistance(antID, target, inventory):
    # Ants are originally searched from the parentState, so their
    #   corresponding Ant in the childState is matched via the Ant.UniqueID
    for ant in inventory.ants:
        if ant.UniqueID == antID:
            antX, antY = ant.coords
            break

    # Returns a distance between the Ant and the Target's coordinates
    targetX, targetY = target.coords
    return abs(antX - targetX) + abs(antY - targetY)


# Determines the best course of action (most successful Move)
def utility(parentState, childState):
    me = childState.whoseTurn

    # My inventories and enemy inventory for comparisons
    #       (this drives the utility function)
    parentInventory = getCurrPlayerInventory(parentState)
    childInventory = getCurrPlayerInventory(childState)
    
    enemyInventory = getEnemyInv(not me, childState)
    enemyQueen = enemyInventory.getQueen()

    # If the enemy Queen will die, take the move (highest utility)
    if enemyQueen is None or enemyQueen.health <= 0:
        return 1.0

    ################
    ###  Functionality for targeting:
    ###     Each Ant in the list is evaluated for its distance to the target
    ###     in both the parentState and the childState (allows for comparison)
    ###
    ###     If the childState has a shorter distance than the parent, this move
    ###     is encouraged by getting a higher utility (moves ants toward target)
    ################

    # Targets for the Worker ants (so they will collect food)
    tunnel = getConstrList(parentState, me, (TUNNEL,))[0]
    if (len(getConstrList(parentState, None, (FOOD,))) > 0):
        food = getConstrList(parentState, None, (FOOD,))[0]

    # Workers target the tunnels and the food sources (collecting)
    workerBonus = 0.0
    workers = getAntList(parentState, me, (WORKER,))
    for worker in workers:
        if worker.carrying:
            parentDistance = getDistance(worker.UniqueID, tunnel, parentInventory)
            nextDistance = getDistance(worker.UniqueID, tunnel, childInventory)
        else:
            parentDistance = getDistance(worker.UniqueID, food, parentInventory)
            nextDistance = getDistance(worker.UniqueID, food, childInventory)

        improvement = parentDistance - nextDistance
        if improvement > 0:
            workerBonus += improvement

    # Attacker Ants attack the enemy Workers first, then the enemy Queen
    offensiveBonus = 0.0
    offensiveAnts = getAntList(parentState, me, (DRONE, SOLDIER, R_SOLDIER))
    enemyWorkers = getAntList(parentState, not me, (WORKER,))

    # Picks a target in the enemyWorkers list (or the Queen if no Workers)
    if enemyWorkers:
        target = enemyWorkers[0]
    else:
        target = enemyQueen

    for ant in offensiveAnts:
        parentDistance = getDistance(ant.UniqueID, target, parentInventory)
        nextDistance = getDistance(ant.UniqueID, target, childInventory)
        
        improvement = parentDistance - nextDistance
        if improvement > 0:
            offensiveBonus += improvement

    # Reward for increasing the food count
    foodBonus = childInventory.foodCount * 0.1

    # Reward for building Soldiers
    numSoldiers = len(getAntList(childState, me, (SOLDIER,)))
    soldierNumBonus = numSoldiers * 0.2

    # Reward for having exactly 2 Workers at all times (if possible)
    numWorkers = len(getAntList(childState, me, (WORKER,)))
    workerNumBonus = max(0, 1 - abs(numWorkers - 2) * 0.5)

    # Calculate combined utility (these values are somewhat arbitrary)
    value = (
        workerBonus * 0.01 +
        offensiveBonus * 0.4 +
        foodBonus * 0.01 +
        soldierNumBonus * 0.03 +
        workerNumBonus * 0.02
    )

    return min(1.0, value)


# Find best move in a given list of nodes
def bestNode(nodes):
    best_utility = 0
    best_node = None

    for node in nodes:
        utility = node["evaluation"]

        # Rank their utility and take the best
        if (utility > best_utility):
            best_utility = utility
            best_node = node

    return best_node

##
# AIPlayer
# Description: The responsbility of this class is to interact with the game by
# deciding a valid move based on a given game state. This class has methods that
# will be implemented by students in Dr. Nuxoll's AI course.
#
# Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##

    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "ANN_AGENT_test")
        self.episode_memory = []
    
    ##
    # getPlacement
    #
    # Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
        
    ##
    # getMove
    # Description: Gets the next move from the Player.
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##    
    def getMove(self, currentState):
        legal_moves = listAllLegalMoves(currentState)
        node_list = []

        for move in legal_moves:
            nextState = getNextState(currentState, move)
            depth = 1
            
            # evaluation is the sum of the utility and the depth
            if training:
                evaluation = utility(currentState, nextState)
            else:
                neurons, weights, _ = forward_prop(currentState, nextState)
                evaluation = neurons[len(weights)]

            node = {
                "move": move,
                "state": nextState,
                "depth": depth,
                "parent": currentState,
                "evaluation": evaluation
            }
            node_list.append(node)

        best_node = bestNode(node_list)

        if training:
            # Add state info to episode_memory (for later ANN training)
            parent, child, eval = best_node["parent"], best_node["state"], best_node["evaluation"]
            self.episode_memory.append((parent, child, eval))

        # return move
        return best_node["move"]

    
    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    # registerWin
    #
    # This agent learns by training the ANN at the end of every game
    #
    def registerWin(self, hasWon):
        #method templaste, not implemented
        if not training:
            return
        
        # If there were many movements (maybe got stuck in a loop),
        #   skip training for computational simplicity
        if len(self.episode_memory) > 250:
            print("Too many transitions, abandoning training\n")

            self.episode_memory.clear()
            return

        print(f"Game ended. {'Won' if hasWon else 'Lost'}. Training on {len(self.episode_memory)} transitions...")

        # Forward and back propogation for each move taken throughout the game
        for (parent, child, u_val) in self.episode_memory:
            neurons, weights, biases = forward_prop(parent, child)
            target = np.array([u_val])
            alpha = 0.05

            # Compute output and error for logging
            output = neurons[len(weights)]
            error = np.mean((target - output) ** 2)
            print(f"   Training sample: target={target[0]:.3f}, output={output[0][0]:.3f}, error={error:.5f}")

            back_prop(neurons, target, weights, biases, alpha)

        # Clear for next episode
        self.episode_memory.clear()
        print("Training complete. Memory cleared.\n")

# ------------------------------
# Hard-coded weights (from ANN_weights.txt)
# ------------------------------
global_weights = {}

global_weights[1] = np.array([
    [1.56578105, 0.43458508, 0.50767764, 1.98153452, 0.72231115, 1.01702473, 1.97355166, 0.24885487, -0.36710307, -0.40733689, 1.01322983, 0.74672240, -0.46666232, 0.39947281, 0.33856674, -0.12987705, 9.80083906, 0.00025622, 0.12747704, -1.20308752, -0.16957722, -0.18566166, -0.48102015, -1.55163888, 0.01565750, 0.28302214, 1.40169498, -0.02884943, -0.33930829, -0.48158388, -1.58907354, -0.11904494, -4.14679025, -7.02590817],
    [0.54877027, 0.03706042, 0.14361941, 2.44092180, -0.43164321, -1.64566792, 0.23142244, -0.12776482, 0.13434821, 1.05009833, 0.26217110, 0.69247860, -2.44874527, 1.17586225, -0.09222958, 0.88341325, 10.30244989, -1.75931991, -0.60615346, -0.36068415, -1.33846353, 0.49117809, 0.06310396, 0.52981241, -0.42668745, -0.05489006, -1.30787761, -0.62766200, 0.08915164, -0.72086712, -1.21381844, -0.37338957, -0.22647377, -11.30491833],
    [1.10785846, 0.52415790, 0.31782168, 2.11687161, 0.89677490, 0.65799964, -1.53810894, -0.16895480, 0.05186267, -1.12692531, 1.21893746, 1.82776069, -1.14978013, 0.52168103, 0.09478640, 2.56964639, 8.50760643, -1.02367888, -0.53121025, -0.08865236, 0.29726186, -0.08085057, -0.52378365, -0.65863236, -0.05289347, -0.48594231, 1.12138436, -3.44538314, -0.41919105, -1.96427085, 0.40446239, 0.16725530, -2.55665870, -8.02467102],
    [0.16030414, 0.41160992, 0.10599281, 0.13567349, -0.64687553, -0.13558121, 0.56713213, -0.17076264, -0.49576316, -1.26471957, -1.52902549, 0.30625506, 0.37073285, 0.43297775, 0.38344788, -1.76854930, -0.95761403, 0.68387241, -0.60120867, 0.72038219, -0.59496110, 0.22406453, 1.10165568, -1.21080767, 0.87017340, 0.50957290, 2.32732995, -0.14940719, -0.05544002, 0.36236467, 0.30323748, -0.57996990, -0.56558044, 1.02699656],
    [-0.88618684, 0.74969247, 1.07687424, 1.30933268, -0.14739570, 7.09425081, -0.21709307, -0.14526081, -0.11835506, -4.71104060, 0.22596405, 0.53495302, -0.67346209, -2.60882746, 0.15625358, 0.40166736, -2.19556968, 1.41704403, -0.51369265, -0.73226953, -1.20572568, -0.74560341, -7.50097403, 0.87503542, -0.01853833, 0.13015925, 4.65994648, -1.59629416, -0.70293584, 0.42038694, 2.47953919, -0.12269690, -3.03177051, 1.72635553],
    [-2.39988756, -0.14935685, -1.02602904, -1.24623817, -0.53423347, -2.46744342, -1.02516516, -0.98463918, -0.22874755, -0.73433334, -3.66628574, 0.83204338, -0.88991738, 0.43368651, 0.36766588, -2.77266188, -11.23913733, 0.48010816, -0.28731755, 1.15806408, 1.51780396, -0.49364256, 1.92102985, 1.06927403, -0.45427179, -0.25659540, 0.32808513, 1.67521834, -0.25981155, -2.12138815, 0.34019119, -0.46048097, 4.94316048, 7.27481699],
    [-0.10308235, 0.30447790, -0.46564318, -2.35129426, 0.14039335, -1.33541886, -0.10340669, 0.32256503, 0.29704944, 1.07682536, 0.85941244, -1.84083858, -0.32672117, -0.17191368, 0.19274184, -2.77896728, -0.27103807, 0.36931675, -0.56234680, -1.20219381, -0.66128203, 0.20648226, -0.09260479, 0.34979555, 0.24529757, 0.21456156, 0.23102676, -0.11815204, -0.15523439, 0.91735688, -0.94296201, -0.17491914, 1.29564500, 1.13878082],
    [0.82338522, -0.11508061, -0.83930505, -0.15532868, 0.52594189, -1.73386059, -0.19444800, -0.58578034, -0.46158885, -1.69153627, 0.65599903, -0.02674189, 0.99559698, 0.05863361, -0.16816102, 0.36701069, -1.30436878, -0.87914690, 0.46694252, 0.62612234, 0.37290978, -0.27098509, 1.01733953, -0.43062212, -0.17217526, 0.04117079, -0.46459824, -0.24319726, -1.23358456, -0.29681109, -1.17347480, 0.03937862, 0.70194556, 1.25769490],
    [-0.66283351, 0.45364417, 0.73618142, 0.18952287, -0.56220816, -0.65837202, -0.91320066, 0.40385470, 1.06033025, -0.53024933, 0.03498482, -0.28592384, 0.20457171, 0.34159319, 0.00052210, 0.25432415, -1.39422437, 1.32817449, 0.07667012, 0.52092402, -0.77470397, 0.79794270, 0.79343372, 0.65956275, -0.53664120, -0.50760545, 1.10877011, 0.59694221, -0.79374717, -0.33233257, -0.17697102, 0.16233269, -0.57565253, -0.06970889],
    [-0.14429679, 0.28861553, 0.35728954, 1.75874457, 0.98119715, 2.39816337, -1.45140953, -0.95131017, -0.47068623, -1.89341967, -0.61242311, 1.68206578, -1.04114846, -1.09035218, 0.06062565, 3.05707598, 6.13101222, -0.69490471, 0.03059241, 0.62623714, 0.86159328, -0.62439607, -1.75876841, 0.49138757, 0.16550084, 0.54230789, 0.17680858, -0.31665581, 0.84926933, -0.12866467, 0.30639813, -0.12892983, -3.81274235, -8.19703637],
    [-0.10924887, -0.29360017, 0.27777853, 0.86356660, -0.34115684, 2.23801288, -1.04819171, -0.16861030, 0.07626253, 1.69740363, -0.64703441, 0.02397512, -0.36437159, -0.99683518, -0.40445762, 0.72684477, -0.42011209, 2.10341773, -0.60470075, 0.83296853, -0.47667569, 0.44871561, -2.45899527, 0.16968820, 0.43088764, -0.47129892, -1.43940098, 1.37750642, -0.38644466, -1.10851438, 0.28549431, 0.32457250, -1.89505920, 0.73320942],
    [-0.33304997, -0.37832966, -0.89341819, -0.49775297, -0.95897217, -0.84492333, 1.51523872, 0.48230345, -0.47604189, 1.20283855, 1.03413710, -0.28599103, 0.96465805, 0.51950947, 0.72638191, -1.16421659, 1.67969055, -0.03492631, 0.44993072, -0.41630242, 0.22874513, -0.60365778, 0.45247610, 3.41786440, -0.28028460, 0.73993658, 0.60364316, -4.42063689, -1.29964167, 0.80726836, -0.10967799, -0.83696159, 2.22831566, -1.25938551],
    [-0.14682667, 0.03791984, 0.10879947, 0.27210555, -0.57059973, -0.73318235, 0.27065714, -0.81721207, -0.29150204, 1.63346284, -1.51904047, -0.25837815, -0.45388963, 0.90217662, 0.18522928, -0.90196192, 4.85689906, -0.75829702, -0.07410688, 0.65241941, -0.57551444, -0.63745363, 0.80552057, -0.01888185, 0.22431863, -0.63986182, 0.71477691, 1.84964940, 0.42922888, -0.04354055, 0.09799609, -0.24991843, -1.56514385, -1.23744215],
    [-0.82922658, 0.73876320, -1.39635200, -1.04300206, -0.26792484, 0.32314983, 0.06920321, -0.16865499, 0.01562738, -6.99689881, -1.35117285, -0.58251907, -1.92379853, -0.97964407, -0.11724973, -2.58641994, -18.93651207, 0.11155631, -0.63012004, -0.17641049, 2.52812064, 0.64139986, -0.40635894, -0.41464160, 0.40253493, 0.21899553, 6.83051687, 1.28643248, -0.15880064, -1.63041838, 0.43721333, -0.01851338, 5.67547507, 19.66328558],
    [-0.63975583, -0.04371454, 0.38089360, -0.18510385, -0.94865373, 0.30277923, -0.94583123, -0.50303904, -0.53887199, 0.00876892, 2.09567087, -2.22347799, 1.56814000, -1.29144453, -0.20270242, -3.07672606, 1.82370351, 0.01269968, -0.19193465, -1.16384687, 0.05395599, 0.67603896, -0.51040633, 0.52304485, 0.27153306, -0.10089646, 0.00881793, 0.51295443, -1.45243378, -0.23855654, -0.84210990, 0.57165799, -0.91801526, -0.36758898],
    [-0.50591137, 0.44347850, -0.56056558, -0.90181996, 0.80557503, -0.89457749, -0.28999409, -0.54596344, -0.00393979, 2.32134365, -1.50789157, -0.03462989, 0.38324238, 0.02116039, -0.55521526, -1.84437128, -6.14160687, 1.51376954, -0.14866343, -0.47115646, 0.27285936, 0.63528718, 1.42297392, -1.26566486, -0.10020850, -0.90880590, -0.93254664, 2.31095620, 0.38599998, -0.06804437, 0.10298861, 0.32969028, 0.88319602, 8.80919650],
])

global_weights[2] = np.array([
    [0.08829768, 0.59861920, -1.23879876, 0.21760794, 0.71020754, -0.28106882, -0.85249647, -1.08100059, 0.20421554, 0.62247586, -0.88179985, -0.29742715, -1.55094252, -0.49450851, 0.65007492, -0.08372035],      
    [0.14895979, 0.52815071, 1.22769405, -1.55237886, -0.53004631, -1.30812529, -1.85883462, 0.33927375, -1.43580990, 0.57011568, -1.03061478, -1.11929426, 0.29435540, -1.67096166, -0.21836193, -1.24608885],     
    [-0.06137795, -0.18755324, -1.03182136, 0.84474132, -0.51788960, -0.41817960, -0.64373343, -0.76962875, 0.45873406, -0.90212167, 0.94864709, -2.39081547, -0.10498164, -1.00499085, 0.66318269, 1.51455690],    
    [-5.02871817, -6.11447709, -5.31589516, 0.21716091, 2.20406245, 5.01179670, 0.80354532, 1.15658764, -0.25832913, -3.54378666, 1.32562102, 0.10131655, -1.81581202, 10.89846608, 0.78681558, 3.77124860],        
    [-0.07534533, -0.16333316, -1.05661352, -1.44624908, -1.38161171, 0.00771735, 2.23441981, -0.24984169, -2.27198639, -2.51974640, -2.11148946, -0.65764868, 0.36769957, 0.16096102, 2.79128099, -0.91667012],    
    [-2.49965795, -0.53522208, -2.10271543, -1.72623432, -3.11616825, 2.08404837, -0.26564868, -0.04591406, -0.64626875, -0.91272801, 0.37708345, 0.72519472, -2.30314952, 3.64947713, 0.46444116, 0.76782861],     
    [0.28571884, 0.74678051, 0.68851341, -1.08737924, 1.13906312, -0.90402638, -0.05324341, 0.01560964, -0.61999995, 0.26783885, -0.22705983, -0.83694380, -1.15578255, -1.69115972, -0.40793918, 0.15082776],      
    [1.51507164, 2.84112100, 0.87591833, -0.74404874, -2.72253034, -2.19404222, 0.90431960, 0.21171656, -0.55520585, 0.61681949, -1.11080256, 0.85494931, 1.86455394, -2.17590342, -0.99799482, -0.50449607],       
    [0.50905707, 0.25573659, 0.49451126, -0.42750866, 2.05010528, -0.76200633, 0.62160265, -0.62539850, -0.42668976, -0.74049450, -0.22075827, -1.11130132, 0.20677850, -2.10494264, -0.43948013, -0.49107520],     
    [-1.50433353, -3.04152624, -1.22013570, -1.17148113, 5.12255442, -2.25773993, -1.18178626, -1.45185031, -0.66467805, 0.39370890, 2.03784633, -4.28630487, -1.78278714, 3.73858982, -1.80828410, 0.76927081],    
    [-0.37146689, -1.48202690, -0.83063488, 0.30371772, 0.72811160, 0.00519789, -0.44156946, -0.12080993, -0.77678263, -0.45804814, 0.11675552, -2.22846243, 0.61702702, -0.19538118, -1.12786870, 0.41623659],     
    [-0.97382225, -1.33873134, -0.31305117, 0.04387258, -1.62760051, 1.14187233, -1.01893231, 0.27850262, -0.04666830, -0.94346947, -1.19004897, 0.19167031, -2.87154657, 2.00336104, 0.23502650, 0.34886485],      
    [-0.70205389, -0.98896079, -0.33255565, -0.49898153, -1.34665741, 0.69974100, 0.96205757, 0.68233362, -0.48752878, -1.11837492, -1.40309280, -0.01533670, -1.07693401, 0.72441907, 0.67761945, 0.88820750],     
    [0.04365986, 0.46403468, -1.13873095, -0.19731964, -0.33910356, -1.13254623, 0.72171712, -1.00878308, -1.10026620, -1.07752019, -0.20713250, -0.42046843, -1.43322437, -1.37317969, 1.95046340, -0.30060433],   
    [0.72028171, 2.13629677, 0.82532231, -1.70425006, 2.05873972, -1.88681651, -1.46405652, 0.98679207, 0.28871662, 1.44614601, 2.00376084, -3.07918391, 1.16882984, -3.66547710, -2.99927687, -1.38273476],        
    [-0.55874983, -0.59895451, -0.75746789, -0.91947893, 0.02047326, 0.81265208, 0.59457987, -0.74587561, -0.20540694, -0.94319147, 0.20330344, -1.44208712, -0.19575293, 0.25635572, -0.37656818, 0.04448823],     
])

global_weights[3] = np.array([
    [-0.55614913, 0.16054479, 2.78766135, -4.97856293, -3.55017704, -0.92279929, -0.68241483, -1.91696958, -0.90803057, 4.90307413, 1.49601092, -1.91805121, 0.11340151, -1.28227644, 2.57567615, 0.82881571],      
])

# ------------------------------
# Hard-coded biases
# ------------------------------

global_biases = {} 

global_biases[1] = np.array([
    [1.08739436],
    [-0.25133837],
    [0.57772615],
    [0.56285067],
    [1.53242721],
    [-0.25580611],
    [0.83076988],
    [0.69364795],
    [0.30103574],
    [0.89157481],
    [0.69447685],
    [0.76006098],
    [0.57677713],
    [-0.95656001],
    [0.72662690],
    [0.90779342],
])

global_biases[2] = np.array([
    [-0.00077790],
    [0.28085788],
    [0.82959478],
    [0.84969291],
    [-0.26963800],
    [-0.41998451],
    [0.25214718],
    [0.20299363],
    [-0.03849426],
    [1.56464722],
    [0.17050814],
    [-0.76187906],
    [-0.19142113],
    [-0.19086682],
    [2.42371823],
    [0.98309817],
])

global_biases[3] = np.array([
    [1.53008330],
])