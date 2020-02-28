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
import sys

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        food = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        if successorGameState.isWin(): # if action results in Win, we should take the action
          return float("Inf")
        elif successorGameState.isLose() or action == 'Stop': # avoiding getting eaten by a ghost
          return float("-Inf")

        # evaluating action based on the closest food, closer food results in higher evaluation score of the action 
        return max([-util.manhattanDistance(newPos, food) for food in food.asList()])

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        """

        # calculates value of node in mini-max tree
        # pseudocode taken from Lecture 6
        def value(gameState, agentIndex, depth):
          if agentIndex == gameState.getNumAgents(): # looped back to pacman's turn
            agentIndex = 0
            depth -= 1
          if depth == 0 or gameState.isWin() or gameState.isLose(): # is terminal state
            return self.evaluationFunction(gameState)
          elif agentIndex == 0: # pacman's turn, so we are in max node
            return maxValue(gameState, agentIndex, depth)
          else: # ghost's turn, so we are in min node
            return minValue(gameState, agentIndex, depth)

        # calculates value max node in mini-max tree
        # pseudocode taken from Lecture 6
        def maxValue(gameState, agentIndex, depth):
          v = ("Action", float("-Inf"))
          for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            tmpValue = value(successorState, agentIndex + 1, depth)
            if type(tmpValue) is not tuple: # value was returned from self.evaluationFunction(), so it is just a float
              if tmpValue > v[1]:
                v = (action, tmpValue)
            elif tmpValue[1] > v[1]:
              v = (action, tmpValue[1])
          return v
        
        # calculates value mini node in mini-max tree
        # pseudocode taken from Lecture 6
        def minValue(gameState, agentIndex, depth):
          v = ("Action", float("Inf"))
          for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            tmpValue = value(successorState, agentIndex + 1, depth)
            if type(tmpValue) is not tuple: # value was returned from self.evaluationFunction(), so it is just a float
              if tmpValue < v[1]:
                v = (action, tmpValue)
            elif tmpValue[1] < v[1]:
              v = (action, tmpValue[1])
          return v

        # start of recursive calculation of mini-max, returning just the correct action not the value we've calculated
        return value(gameState, 0, self.depth)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # calculates value of node in mini-max tree
        # pseudocode taken from Lecture 6
        def value(gameState, agentIndex, depth, a, b):
          if agentIndex == gameState.getNumAgents(): # looped back to pacman's turn
            agentIndex = 0
            depth -= 1
          if depth == 0 or gameState.isWin() or gameState.isLose(): # is terminal state
            return self.evaluationFunction(gameState)
          elif agentIndex == 0: # pacman's turn, so we are in max node
            return maxValue(gameState, agentIndex, depth, a, b)
          else: # ghost's turn, so we are in min node
            return minValue(gameState, agentIndex, depth, a, b)

        # calculates value max node in mini-max tree
        # pseudocode taken from Lecture 6
        def maxValue(gameState, agentIndex, depth, a, b):
          v = ("Action", float("-Inf"))
          for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            tmpValue = value(successorState, agentIndex + 1, depth, a, b)
            if type(tmpValue) is not tuple: # value was returned from self.evaluationFunction(), so it is just a float
              newValue = tmpValue
            else:
              newValue = tmpValue[1]
            if newValue > v[1]:
              v = (action, newValue)
            
            # prunning part
            if newValue > b: # if new value is higher than MIN's best choise we skip other children
              return (action, newValue)
            a = max(a, newValue) # update the best choise for MAX
            
          return v
        
        # calculates value mini node in mini-max tree
        # pseudocode taken from Lecture 6
        def minValue(gameState, agentIndex, depth, a, b):
          v = ("Action", float("Inf"))
          for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            tmpValue = value(successorState, agentIndex + 1, depth, a, b)
            if type(tmpValue) is not tuple: # value was returned from self.evaluationFunction(), so it is just a float
              newValue = tmpValue
            else:
              newValue = tmpValue[1]
            if newValue < v[1]:
              v = (action, newValue)
            if newValue < a: # if new value is lower than MAX's best choise we skip other children
              return (action, newValue)
            b = min(b, newValue) # update the best choise for MIN
            
          return v

        # start of recursive calculation of mini-max with prunning, returning just the correct action not the value we've calculated
        return value(gameState, 0, self.depth, float("-inf"), float("inf"))[0]

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
        # calculates value of node in expectimax tree
        # pseudocode taken from Lecture 7
        def value(gameState, agentIndex, depth):
          if agentIndex == gameState.getNumAgents(): # looped back to pacman's turn
            agentIndex = 0
            depth -= 1
          if depth == 0 or gameState.isWin() or gameState.isLose(): # is terminal state
            return self.evaluationFunction(gameState)
          elif agentIndex == 0: # pacman's turn, so we are in max node
            return maxValue(gameState, agentIndex, depth)
          else: # ghost's turn, so we are in min node
            return expValue(gameState, agentIndex, depth)

        # calculates value of max node in expectimax tree
        # pseudocode taken from Lecture 7
        def maxValue(gameState, agentIndex, depth):
          v = ("Action", float("-Inf"))
          for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            tmpValue = value(successorState, agentIndex + 1, depth)
            if type(tmpValue) is not tuple: # value was returned from self.evaluationFunction() or expValue(), so it is just a float
              if tmpValue > v[1]:
                v = (action, tmpValue)
            elif tmpValue[1] > v[1]:
              v = (action, tmpValue[1])
          return v
        
        # calculates value of exp node in expectimax tree
        # pseudocode taken from Lecture 7
        def expValue(gameState, agentIndex, depth):
          v = 0
          actions = gameState.getLegalActions(agentIndex)
          p = 1.0 / len(actions)
          for action in actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            tmpValue = value(successorState, agentIndex + 1, depth)
            if type(tmpValue) is not tuple: # value was returned from self.evaluationFunction() or expValue(), so it is just a float
              v += p * tmpValue # update expected value
            else:
              v += p * tmpValue[1] # update expected value
          return v

        # start of recursive calculation of expectimax, returning just the correct action not the value we've calculated
        return value(gameState, 0, self.depth)[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin(): # if action results in Win, we should take the action
      return float("Inf")
    elif currentGameState.isLose(): # avoiding getting eaten by a ghost
      return float("-Inf")

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodCount = food.count()
    ghosts = currentGameState.getGhostStates()
    scaredTimes = sum([ghostState.scaredTimer for ghostState in ghosts])

    foodDistance = min([util.manhattanDistance(pos, food) for food in food.asList()])
    ghostDistance = min([util.manhattanDistance(pos, ghost.getPosition()) for ghost in ghosts])
    
    if ghostDistance < 5: 
      # if ghost is relatively close we take the distance into consideration
      return currentGameState.getScore() + ghostDistance + scaredTimes - foodCount - foodDistance 
    else:
      # if ghost is not nearby we evaluate the state with food being the top priority
      return currentGameState.getScore() - foodCount - foodDistance 

# Abbreviation
better = betterEvaluationFunction

