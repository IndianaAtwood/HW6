import random
import pickle
import os
import numpy as np
from Player import *
from Constants import *
from GameState import *
from AIPlayerUtils import *

# General parameters
ALPHA = 0.1          # learning rate
GAMMA = 0.9          # discount
LAMBDA = 0.7         # eligibility decay
EPSILON = 0.1        # exploration
STEP_REWARD = -0.01  # small negative reward to discourage loops

WEIGHT_FILE = "./atwoodi26_anderale26_weights.txt"   # keep same convention as assignment

# Turn encoded states into categories
def bucketize(vec, bins):
    vec = np.clip(vec, 0.0, 1.0)
    return tuple((vec * (bins - 1)).astype(int).tolist())

##
# encodeState
# Description: Converts a state into an array of decimal values
#
# Parameters:
#   state - The state of the current game
#
# Return: array of decimal values (input array for ANN)
## 
def encodeState(state):
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


# AIPlayer Class
class AIPlayer(Player):

    def __init__(self, playerId):
        super(AIPlayer, self).__init__(playerId, "TD_AGENT")

        # value table: category -> utility estimate
        self.V = {}
        # eligibility traces
        self.E = {}

        # for remembering state between moves
        self.last_category = None

        # try loading previous utilities
        self.loadWeights()

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

    # Make sure category even exists
    def ensureCategory(self, cat):
        if cat not in self.V:
            self.V[cat] = 0.0
            self.E[cat] = 0.0

    # Load saved utilities
    def loadWeights(self):
        if os.path.exists(WEIGHT_FILE):
            try:
                with open(WEIGHT_FILE, "rb") as f:
                    self.V = pickle.load(f)
                    self.E = {c: 0.0 for c in self.V}
                print(f"Loaded {len(self.V)} categories from {WEIGHT_FILE}")
            except:
                print("Failed to load weights — starting fresh.")

    # Save utilities after each game
    def saveWeights(self):
        try:
            with open(WEIGHT_FILE, "wb") as f:
                pickle.dump(self.V, f)
            print(f"Saved {len(self.V)} categories.")
        except:
            print("Error saving weights.")

    # Turn states into categories
    def getCategory(self, state):
        vec = encodeState(state)
        return bucketize(vec, 6)

    ##
    # tempDiff
    # Description: Temporal Difference functionality
    #
    # Parameters:
    #   s_cat - current state category
    #   next_cat - next state category (after action is taken)
    #   reward - punishment/reward for win/loss
    #   gameEnd - If the game ended or not (boolean)
    ## 
    def tempDiff(self, s_cat, next_cat, reward, gameEnd):
        self.ensureCategory(s_cat)
        if next_cat is not None:
            self.ensureCategory(next_cat)

        V_s = self.V[s_cat]
        V_snext = 0.0 if gameEnd or next_cat is None else self.V[next_cat]

        delta = reward + GAMMA * V_snext - V_s

        # increment eligibility for current category
        self.E[s_cat] += 1.0

        # update all categories with eligibility traces
        for cat in self.E.keys():
            self.V[cat] += ALPHA * delta * self.E[cat]
            self.E[cat] *= GAMMA * LAMBDA

    # Evaluate move, predicts future state
    def evaluateMove(self, state, move):
        nextState = getNextState(state, move)

        enemyInv = getEnemyInv(self.playerId, nextState)
        q = enemyInv.getQueen()

        # If no enemy Queen, a win
        if q is None:
            return 1.0, True

        cat = self.getCategory(nextState)
        self.ensureCategory(cat)
        return self.V[cat], False

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

        legalMoves = listAllLegalMoves(currentState)

        # e-greedy exploration (if less than Epsilon, random choice)
        if random.random() < EPSILON:
            move = random.choice(legalMoves)
            nextCat = self.getCategory(getNextState(currentState, move))

            if self.last_category is not None:
                self.tempDiff(self.last_category, nextCat, STEP_REWARD, False)

            self.last_category = nextCat
            return move

        # exploitation: choose best predicted utility
        bestVal = -99999
        best = []

        for m in legalMoves:
            val, gameEnd = self.evaluateMove(currentState, m)
            if val > bestVal:
                bestVal = val
                best = [(m, gameEnd)]
            elif val == bestVal:
                best.append((m, gameEnd))

        chosen, gameEnd = random.choice(best)
        nextCat = None if gameEnd else self.getCategory(getNextState(currentState, chosen))

        if self.last_category is not None:
            reward = 1.0 if gameEnd else STEP_REWARD
            self.tempDiff(self.last_category, nextCat, reward, gameEnd)

        self.last_category = nextCat
        return chosen

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
    # Description: Resets variables and saves weights for a win
    ##
    def registerWin(self, hasWon):
        if self.last_category is not None:
            finalReward = 1.0 if hasWon else -1.0
            self.tempDiff(self.last_category, None, finalReward, True)

        # reset eligibilities
        for c in self.E:
            self.E[c] = 0.0

        self.last_category = None

        self.saveWeights()
