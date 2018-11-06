# valueIterationAgents.py
# -----------------------
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

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        nextStVals = util.Counter()
        iterNum = 0
        res = None
        lst = []

        # Going through the number of iterations
        while iterNum < self.iterations:

            # Going through each possible state in the given MDP
            for currState in mdp.getStates():
                qVals = []
                maxNum = float('-inf')

                # All the possible actions in the given current state. All the computed Q-values from each action are added to the qVals list
                for possAction in mdp.getPossibleActions(currState):
                    qVals.insert(len(qVals)-1,self.computeQValueFromValues(currState,possAction))

                # If the final state of the problem has been reached, 0 is appended to the Q-value
                if self.mdp.isTerminal(currState):
                    qVals.insert(len(qVals)-1,0)

                # Finding the maximum
                for i in qVals:
                    if i > maxNum:
                        maxNum = i

                # Storing the maximum of the Q-values of the state to the dictionary that stores the values of the next state
                nextStVals[currState] = maxNum

            # Copying the list with the highest Q-values to self.values l
            lst.append(nextStVals.copy())
            iterNum += 1
            self.values = lst.pop()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        sumQ = 0

        # Goes through the transition states and the problems and adds up all the values in the transitions to compute the total Q-value
        for trans1,trans2 in self.mdp.getTransitionStatesAndProbs(state, action):
            sumQ += (trans2*(self.mdp.getReward(state, action, trans1) + self.discount * self.values[trans1]))

        return sumQ

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # If the actions are possible
        if self.mdp.getPossibleActions(state):
            qValue = float('-inf')
            actions = []

            # Goes through the possible actions from the MDP and finds out the highest possible Q val for each action. The actions associated with the Q-value are appended to the list
            for possActions in self.mdp.getPossibleActions(state):
                if self.computeQValueFromValues(state,possActions) > qValue:
                    qValue = self.computeQValueFromValues(state,possActions)
                    actions.append(possActions)

            return actions[len(actions)-1]

        return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
