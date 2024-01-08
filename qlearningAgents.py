# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        maxQ = -float('inf')
        for a in legalActions:
            q = self.getQValue(state, a)
            if q > maxQ:
                maxQ = q
        if maxQ == -float('inf'):
            return 0.0
        else:
            return maxQ

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        action = None
        maxQ = -float('inf')
        for a in legalActions:
            q = self.getQValue(state, a)
            if q > maxQ:
                maxQ = q
                action = a
        return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            return action
        if util.flipCoin(self.epsilon):  # random action
            action = random.choice(legalActions)
        else:  # best policy
            action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        currQ_value = self.getQValue(state, action)
        self.q_values[(state, action)] = (1 - self.alpha) * currQ_value + self.alpha * (
                reward + self.discount * self.getValue(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.2, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        q_val = 0
        features = self.featExtractor.getFeatures(state, action)
        for f in features.sortedKeys():
            q_val += self.weights[f] * features[f]
        return q_val

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        diff = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        for f in features.sortedKeys():
            self.weights[f] += self.alpha * diff * features[f]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

class QLearnAgent(Agent):
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining=10):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.q_value = util.Counter()
        self.score = 0
        self.lastState = []
        self.lastAction = []
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1
    def getEpisodesSoFar(self):
        return self.episodesSoFar
    def getNumTraining(self):
        return self.numTraining
    def setEpsilon(self, value):
        self.epsilon = value
    def getAlpha(self):
        return self.alpha
    def setAlpha(self, value):
        self.alpha = value
    def getGamma(self):
        return self.gamma
    def getMaxAttempts(self):
        return self.maxAttempts
    def getQValue(self, state, action):
        return self.q_value[(state, action)]
    def getMaxQ(self, state):
        q_list = []
        for a in state.getLegalPacmanActions():
            q = self.getQValue(state, a)
            q_list.append(q)
            if len(q_list) == 0:
                return 0
            return max(q_list)
    def updateQ(self, state, action, reward, qmax):
        q = self.getQValue(state, action)
        self.q_value[(state, action)] = q + self.alpha * (reward + self.gamma * qmax - q)
    def doTheRightThing(self, state):
        legal = state.getLegalPacmanActions()
        if self.getEpisodesSoFar() * 1.0 / self.getNumTraining() < 0.5:
            if Directions.STOP in legal:
                legal.remove(Directions.STOP)
            if len(self.lastAction) > 0:
                last_action = self.lastAction[-1]
                distance0 = state.getPacmanPosition()[0] - state.getGhostPosition(1)[0]
                distance1 = state.getPacmanPosition()[1] - state.getGhostPosition(1)[1]
            if math.sqrt(distance0 ** 2 + distance1 ** 2) > 2:
                if (Directions.REVERSE[last_action] in legal) and len(legal) > 1:
                    legal.remove(Directions.REVERSE[last_action])
                    tmp = util.Counter()
        for action in legal:
            tmp[action] = self.getQValue(state, action)
            return tmp.argMax()
    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            reward = state.getScore() - self.score
        if len(self.lastState) > 0:
            last_state = self.lastState[-1]
            last_action = self.lastAction[-1]
            max_q = self.getMaxQ(state)
            self.updateQ(last_state, last_action, reward, max_q)
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)
        else:
            action = self.doTheRightThing(state)
            self.score = state.getScore()
            self.lastState.append(state)
            self.lastAction.append(action)
            return action
    def final(self, state):
        reward = state.getScore() - self.score
        last_state = self.lastState[-1]
        last_action = self.lastAction[-1]
        self.updateQ(last_state, last_action, reward, 0)
        self.score = 0
        self.lastState = []
        self.lastAction = []
        ep = 1 - self.getEpisodesSoFar() * 1.0 / self.getNumTraining()
        self.setEpsilon(ep * 100.0)
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() % 100 == 0:
            print("Completed %s runs of training" % self.getEpisodesSoFar())
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)