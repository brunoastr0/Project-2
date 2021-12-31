# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a game_state and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = game_state.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(game_state, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentgame_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        game_states (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a game_state (pacman.py)
        successorgame_state = currentgame_state.generatePacmanSuccessor(action)
        newPos = successorgame_state.getPacmanPosition()
        newFood = successorgame_state.getFood().asList()
        newGhostStates = successorgame_state.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        nearest_ghost_dis = 99999
        for ghost_state in newGhostStates:
            ghost_x, ghost_y = ghost_state.getPosition()
            if ghost_state.scaredTimer == 0:  # pacman will be eaten so he must stay away
                nearest_ghost_dis = min(nearest_ghost_dis, manhattanDistance((int(ghost_x), int(ghost_y)), newPos))
        nearest_food_dis = 99999 if newFood else 0
        if nearest_food_dis:
            for food in newFood:
                nearest_food_dis = min(nearest_food_dis, manhattanDistance(food, newPos))
        return successorgame_state.getScore() - 7 / (nearest_ghost_dis + 1) - nearest_food_dis * 0.25


def scoreEvaluationFunction(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.getLegalActions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generateSuccessor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.getNumAgents():
        Returns the total number of agents in the game

        game_state.isWin():
        Returns whether or not the game state is a winning state

        game_state.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxSearch(game_state, agent_index=0, depth=self.depth)[1]

    def minimaxSearch(self, game_state, agent_index, depth):
        if depth == 0 or game_state.isLose() or game_state.isWin():
            return self.evaluationFunction(game_state), Directions.STOP
        elif agent_index == 0:  # maximizer
            return self.get_agent_value(game_state, agent_index, depth)
        else:  # minimizer
            return self.get_agent_value(game_state, agent_index, depth)

    @staticmethod
    def update_arguments(game_state, agent_index, depth):
        if agent_index == game_state.getNumAgents() - 1:
            return 0, depth - 1
        return agent_index + 1, depth

    def get_agent_value(self, game_state, agent_index, depth):
        next_agent, next_depth = self.update_arguments(game_state, agent_index, depth)
        desired_score, desired_action = 99999 if agent_index != 0 else -99999, Directions.STOP
        actions = game_state.getLegalActions(agent_index)
        for action in actions:
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            new_score = self.minimaxSearch(successor_game_state, next_agent, next_depth)[0]
            if agent_index and new_score < desired_score:
                desired_score, desired_action = new_score, action
            elif not agent_index and new_score > desired_score:
                desired_score, desired_action = new_score, action
        return desired_score, desired_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alpha_beta_search(game_state, agent_index=0, depth=self.depth)[1]

    def alpha_beta_search(self, game_state, agent_index, depth, alpha=-9999, beta=9999):
        if depth == 0 or game_state.isLose() or game_state.isWin():
            return self.evaluationFunction(game_state), Directions.STOP
        elif agent_index == 0:  # maximizer
            return self.get_agent_value(game_state, agent_index, depth, alpha, beta)
        else:  # minimizer
            return self.get_agent_value(game_state, agent_index, depth, alpha, beta)

    @staticmethod
    def update_arguments(game_state, agent_index, depth):
        if agent_index == game_state.getNumAgents() - 1:
            return 0, depth - 1
        return agent_index + 1, depth

    def get_agent_value(self, game_state, agent_index, depth, alpha, beta):
        next_agent, next_depth = self.update_arguments(game_state, agent_index, depth)
        desired_score, desired_action = 99999 if agent_index != 0 else -99999, Directions.STOP
        actions = game_state.getLegalActions(agent_index)
        for action in actions:
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            new_score = self.alpha_beta_search(successor_game_state, next_agent, next_depth, alpha, beta)[0]
            if agent_index:  # minimizer
                if new_score < desired_score:
                    desired_score, desired_action = new_score, action
                if new_score < alpha:
                    return desired_score, desired_action
                beta = min(new_score, beta)
            elif not agent_index:  # maximizer
                if new_score > desired_score:
                    desired_score, desired_action = new_score, action
                if new_score > beta:
                    return desired_score, desired_action
                alpha = max(new_score, alpha)
        return desired_score, desired_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax_search(game_state, agent_index=0, depth=self.depth)[1]

    def expectimax_search(self, game_state, agent_index, depth):
        if depth == 0 or game_state.isLose() or game_state.isWin():
            return self.evaluationFunction(game_state), Directions.STOP
        elif agent_index == 0:  # maximizer
            return self.get_agent_value(game_state, agent_index, depth)
        else:  # minimizer
            return self.get_agent_value(game_state, agent_index, depth)

    @staticmethod
    def update_arguments(game_state, agent_index, depth):
        if agent_index == game_state.getNumAgents() - 1:
            return 0, depth - 1
        return agent_index + 1, depth

    def get_agent_value(self, game_state, agent_index, depth):
        next_agent, next_depth = self.update_arguments(game_state, agent_index, depth)
        desired_score, desired_action = 0 if agent_index != 0 else -99999, Directions.STOP
        actions = game_state.getLegalActions(agent_index)
        for action in actions:
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            new_score = self.expectimax_search(successor_game_state, next_agent, next_depth)[0]
            if agent_index:  # expectancy
                desired_score = desired_score + new_score
            elif not agent_index:  # maximizer
                if new_score > desired_score:
                    desired_score, desired_action = new_score, action
        if agent_index:  # expectancy
            desired_score, desired_action = desired_score/len(actions), random.choice(actions)
        return desired_score, desired_action


def betterEvaluationFunction(currentgame_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_pos = currentgame_state.getPacmanPosition()
    food = currentgame_state.getFood().asList()
    ghostStates = currentgame_state.getGhostStates()
    "*** YOUR CODE HERE ***"
    nearest_ghost_dis = 99999
    for ghost_state in ghostStates:
        ghost_x, ghost_y = ghost_state.getPosition()
        if ghost_state.scaredTimer == 0:  # pacman will be eaten so he must stay away
            nearest_ghost_dis = min(nearest_ghost_dis, manhattanDistance((int(ghost_x), int(ghost_y)), pacman_pos))
    nearest_food_dis = 99999 if food else 0
    if nearest_food_dis:
        for food in food:
            nearest_food_dis = min(nearest_food_dis, manhattanDistance(food, pacman_pos))
    return currentgame_state.getScore() - 7 / (nearest_ghost_dis + 1) - nearest_food_dis * 0.25


# Abbreviation
better = betterEvaluationFunction
