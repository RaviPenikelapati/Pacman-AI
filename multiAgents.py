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
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        import sys
        foodList = newFood.asList()
        ghostList = successorGameState.getGhostPositions()
        distancesToFood = []
        distancesToGhosts = []
        evaluation = 0

        for food in foodList:
            distancesToFood.append(util.manhattanDistance(newPos, food))

        for ghost in ghostList:
            distancesToGhosts.append(util.manhattanDistance(newPos, ghost))

        if distancesToFood:
            evaluation = - (min(distancesToFood) + 500*len(distancesToFood))

        for ghostDistance in distancesToGhosts:
            if ghostDistance < 3:
                return -sys.maxsize

        return evaluation


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        import sys
        numAgents = gameState.getNumAgents()
        prediction = {}

        def miniMaxHelper(state, count):
            currentAgent = count % numAgents

            # BASE CASE
            if state.isWin() or state.isLose() or (count >= self.depth * numAgents):
                return self.evaluationFunction(state)

            # MAX
            if currentAgent == 0:
                outcome = -sys.maxsize
                for action in state.getLegalActions(currentAgent):
                    successor = state.generateSuccessor(currentAgent, action)
                    outcome = max(outcome, miniMaxHelper(successor, count+1))
                    # Keeping track of the expected outcomes of all possible first moves by pacman
                    if count == 0:
                        prediction[action] = outcome
                return outcome

            # MIN
            else:
                outcome = sys.maxsize
                for action in state.getLegalActions(currentAgent):
                    successor = state.generateSuccessor(currentAgent, action)
                    outcome = min(outcome, miniMaxHelper(successor, count+1))
                return outcome

        miniMaxHelper(gameState, 0)
        return max(prediction, key=prediction.get)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import sys
        numAgents = gameState.getNumAgents()
        prediction = {}

        def miniMaxHelper(state, count, a=-sys.maxsize, b=sys.maxsize):
            currentAgent = count % numAgents

            # BASE CASE
            if state.isWin() or state.isLose() or (count >= self.depth * numAgents):
                return self.evaluationFunction(state)

            # MAX
            if currentAgent == 0:
                outcome = -sys.maxsize
                for action in state.getLegalActions(currentAgent):
                    successor = state.generateSuccessor(currentAgent, action)
                    outcome = max(outcome, miniMaxHelper(
                        successor, count+1, a, b))
                    a = max(a, outcome)
                    if count == 0:
                        prediction[action] = outcome
                    if b < a:
                        break
                return outcome

            # MIN
            else:
                outcome = sys.maxsize
                for action in state.getLegalActions(currentAgent):
                    successor = state.generateSuccessor(currentAgent, action)
                    outcome = min(outcome, miniMaxHelper(
                        successor, count+1, a, b))
                    b = min(b, outcome)
                    if b < a:
                        break
                return outcome

        miniMaxHelper(gameState, 0)
        return max(prediction, key=prediction.get)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        import sys
        import random
        numAgents = gameState.getNumAgents()
        prediction = {}

        def miniMaxHelper(state, count):
            currentAgent = count % numAgents

            # BASE CASE
            if state.isWin() or state.isLose() or (count >= self.depth * numAgents):
                return self.evaluationFunction(state)

            # MAX
            if currentAgent == 0:
                outcome = -sys.maxsize
                for action in state.getLegalActions(currentAgent):
                    successor = state.generateSuccessor(currentAgent, action)
                    outcome = max(outcome, miniMaxHelper(successor, count+1))
                    if count == 0:
                        prediction[action] = outcome
                return outcome

            # MIN
            else:
                outcome = 0
                for action in state.getLegalActions(currentAgent):
                    successor = state.generateSuccessor(currentAgent, action)
                    outcome += miniMaxHelper(successor, count+1)
                return outcome/len(state.getLegalActions(currentAgent))

        miniMaxHelper(gameState, 0)
        return max(prediction, key=prediction.get)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <Same thing as the previous evaluation function. 
    But I could also include pellets and scared states to make the scores better>
    """
    "*** YOUR CODE HERE ***"
    import sys

    foodList = currentGameState.getFood().asList()
    ghostList = currentGameState.getGhostPositions()
    distancesToFood = []
    distancesToGhosts = []
    evaluation = 0

    for food in foodList:
        distancesToFood.append(util.manhattanDistance(
            currentGameState.getPacmanPosition(), food))

    for ghost in ghostList:
        distancesToGhosts.append(util.manhattanDistance(
            currentGameState.getPacmanPosition(), ghost))

    if distancesToFood:
        evaluation = - (min(distancesToFood) + 500*len(distancesToFood))

    for ghostDistance in distancesToGhosts:
        if ghostDistance < 3:
            return -sys.maxsize

    return evaluation


# Abbreviation
better = betterEvaluationFunction
