import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        # print(scores, legal_moves)
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best
        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """
        value = 0
        # Influence of conscentraiting around one corner
        if board[3][3] == max_tile:
            value += 2048
        if board[2][3] != 0:
            value += np.log2(board[2][3])
        if board[3][2] != 0:
            value += np.log2(board[3][2])
        if board[2][2] != 0:
            value += np.log2(board[2][2]) / 4
        num_of_empty_tiles = len(successor_game_state.get_empty_tiles()[0])
        value += num_of_empty_tiles / 16
        # Having mostly large value tiles
        sum = 0
        rather_big_tile = 32
        while rather_big_tile <= max_tile:
            sum += rather_big_tile * len(np.where(board == rather_big_tile)[0])
            rather_big_tile = rather_big_tile * 2
        value += sum
        return value


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):

    def max_value(self, state, action, depth, agent_index):
        """
        Max - our part of the minimax algo
        :param state: game state
        :param action: action that lead to this state
        :param depth: depth of the game (1, 2, etc)
        :param agent_index: 0, our player
        :return: max of the minimums of cost
        """
        if depth == self.depth or state.done or action == Action.STOP:
            return self.evaluation_function(state)
        v = float('-inf')
        for new_action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, new_action)
            v = max(v, self.min_value(successor, new_action, depth + 0.5, (agent_index + 1) % 2))
        return v

    def min_value(self, state, action, depth, agent_index):
        """
        Min - adversary's part of the minimax algo
        :param state: game state
        :param action: action that lead to this state
        :param depth: half-depth (0.5, 1.5, etc)
        :param agent_index: 1, adversary
        :return: min of the maximums of cost
        """
        if depth == self.depth or state.done or action == Action.STOP:
            return self.evaluation_function(state)
        v = float('inf')
        for new_action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, new_action)
            v = min(v, self.max_value(successor, new_action, depth + 0.5, (agent_index + 1) % 2))
        return v

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        no_actions = True
        max_value = 0
        best_action = Action.STOP
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            # half a step in depth to min
            value = self.min_value(successor, action, 0.5, agent_index=1)

            # if it's the first successor, save it as maximal, else compare to current max
            if no_actions or value > max_value:
                no_actions = False
                max_value = value
                best_action = action
        # return best action, STOP if there were no legal moves
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alpha_beta_helper(self, state, depth, alpha, beta, player):

        if depth == 0 or state.done:
            # print("Depth is ", depth)
            # print("Returning evaluation function, value is ", self.evaluation_function(state))
            return self.evaluation_function(state)

        for action in state.get_legal_actions(player):
            successor = state.generate_successor(player, action)
            if player == 0:
                # Max
                alpha = max(alpha, self.alpha_beta_helper(successor, depth - 1,
                                                          alpha, beta, 1))
            else:
                # Min
                beta = min(beta, self.alpha_beta_helper(successor, depth - 1,
                                                        alpha, beta, 0))
            if beta <= alpha:
                break
        # Max
        if player == 0:
            return alpha
        # Min
        return beta

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        max_value = float('-inf')
        index = 0
        depth = self.depth * 2
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            value = self.alpha_beta_helper(
                successor, depth - 1, float('-inf'), float('inf'), 1)
            if max_value < value:
                max_value = value
                index = action
        if index == 0:
            return Action.STOP
        return Action(index)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def max_value(self, state, action, depth, agent_index):
        """
        Player
        """
        if depth == self.depth or state.done or action == Action.STOP:
            return self.evaluation_function(state)
        v = float('-inf')
        for new_action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, new_action)
            v = max(v, self.exp_value(successor, new_action, depth + 0.5, (agent_index + 1) % 2))
        return v

    def exp_value(self, state, action, depth, agent_index):
        """
        Adversary
        uniform random legal move
        """
        if depth == self.depth or state.done or action == Action.STOP:
            return self.evaluation_function(state)

        actions = state.get_legal_actions(agent_index)
        if len(actions) == 0:
            return float('inf')

        v = 0
        for new_action in actions:
            successor = state.generate_successor(agent_index, new_action)
            v += self.max_value(successor, new_action, depth + 0.5, (agent_index + 1) % 2)
        return v / len(actions)  # uniform probability to get the action

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        no_actions = True
        max_value = 0
        best_action = Action.STOP
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            # half a step in depth to min
            value = self.exp_value(successor, action, 0.5, agent_index=1)

            # if it's the first successor, save it as maximal, else compare to current max
            if no_actions or value > max_value:
                no_actions = False
                max_value = value
                best_action = action
        # return best action, STOP if there were no legal moves
        return best_action


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    board = current_game_state.board
    max_tile = current_game_state.max_tile
    score = current_game_state.score

    sum = 0
    empty = 0
    same = 0
    max_where = 0
    if board[3][3] == max_tile:
        max_where = 1

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                empty += 1
            if j != len(board) - 1:
                if board[i][j] != 0 and board[i][j + 1] != 0:
                    diff = np.absolute(board[i][j] - board[i][j + 1])
                    if diff == 0:
                        # same += np.log2(board[i][j])
                        same += board[i][j]
                    if board[i][j] > board[i][j + 1]:
                        sum += diff
                    else:
                        sum += diff / 2

            if i != len(board) - 1:
                if board[i][j] != 0 and board[i + 1][j] != 0:
                    diff = np.absolute(board[i][j] - board[i + 1][j])
                    if diff == 0:
                        # same += np.log2(board[i][j])
                        same += board[i][j]
                    if board[i][j] > board[i + 1][j]:
                        sum += diff
                    else:
                        sum += diff / 2

    # optimal WEIGHTS for now: (on seed 1, where was the lowest when w_max was 0)
    w_monotone = -1/max_tile  # if down or right there is smth bigger - this penalty ("sum")
    w_empty = 1/32  # number of empty
    w_same = 0  # same tile is close-by
    w_max = 5/max_tile  # max tile is in right lower corner (if 0 score drops from 30k to 6k on
                        # seed 1)

    # CURRENT ATTEMPTS
    # w_monotone = -1/max_tile
    # w_empty = 1/32
    # w_same = 1/max_tile
    # w_max = board[3][3]/max_tile

    # print("%2.2f %2.2f %2.2f %2.2f "% (empty*w_empty, sum*w_monotone, same*w_same, max_where*w_max))
    return empty*w_empty + sum*w_monotone + same*w_same + max_where*w_max

# Abbreviation
better = better_evaluation_function
