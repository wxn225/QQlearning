# -*- coding: utf-8 -*-
"""
@author: Hugo Gilbert and Bruno Zanuttini
"""

import random 

# ===============================================================================================
# A class defining a generic "SSB MDP". Nothing here is specific to Gardner's dice or the
# SSB-Q-learning algorithm whatsoever.
# ===============================================================================================

class MDP ():

    # states, actions, wealthLevels: lists of strings (atoms over which the MDP is built)
    # allowedActionsFunction: function from state to list of actions
    # finalStates: list of states
    # wealthFunction: function from final state to wealthLevel
    # transitionFunction: function from state, action to dictionary (nextState,probability)
    # ssbFunction: function from wealthLevel, otherWealthLevel to number
    def __init__ (self, states, actions, wealthLevels, allowedActionsFunction, finalStates, wealthFunction, transitionFunction, ssbFunction, initialState, mdpType, horizon):
        self.states = states
        self.actions = actions
        self.wealthLevels = wealthLevels
        self.allowedActionsFunction = allowedActionsFunction
        self.finalStates = finalStates
        self.wealthFunction = wealthFunction
        self.transitionFunction = transitionFunction
        self.ssbFunction = ssbFunction
        self.initialState = initialState
        self.mdpType = mdpType
        self.counter = 0
        #horizon defines the maximum num of actions can be taken in one history
        self.horizon = horizon


    def getStates (self):
        return self.states

    def getActions (self):
        return self.actions

    def getWealthLevels (self):
        return self.wealthLevels

    def getAllowedActions (self, state):
        return self.allowedActionsFunction(state)

    def getMdpAllowedActions (self, state):
        return self.allowedMdpActionsFunction(state)

    def getFinalStates (self):
        return self.finalStates

    def isFinal (self, state):
        return state in self.finalStates

    def isAllowed (self, state, action):
        return action in self.allowedActionsFunctions(state)

    def getWealthLevel (self, finalState):
        return self.wealthFunction(finalState)

    # Returns a dictionary (nextState,probability)
    def getDistribution (self, state, action):
        return self.transitionFunction(state,action)

    def getProbability (self, state, action, nextState):
        if not nextState in self.transitionFunction(state,action).keys():
            return 0.
        return self.transitionFunction(state,action)[nextState]

    def drawNextState (self, state, action):
        if action == "reinit":
            self.counter = 0
        distribution = self.transitionFunction(state,action)
        randomNumber = random.random()
        mass = 0
        for nextState in distribution.keys():
            probability = distribution[nextState]
            mass = mass+probability
            if randomNumber <= mass:
                return nextState

    # Returns phi(wealthLevel,otherWealthLevel), where phi is the SSB utility function
    def getPreference (self, wealthLevel, otherWealthLevel):
        return self.ssbFunction(wealthLevel,otherWealthLevel)

    def solveMDP(self,reward):
        k = 0
        for value in reward.values():
            if value == 1:
                k += 1
        if k == len(reward):
            return 1
        values = {}
        for state in self.getStates():
            if self.isFinal(state):
                values[state] = reward[state]
            else:
                values[state] = 0
        converged = False 
        while(converged is False):
            delta = 0
            for state in self.getStates():
                if not(self.isFinal(state)):
                    nv = -1000
                    for action in self.getAllowedActions(state):
                        Q = 0
                        for new_state,prob in self.transitionFunction( state, action).items():
                            Q += prob*values[new_state]
                        nv = max(nv,Q)
                    delta = max(delta, abs(values[state] - nv))
                    values[state] = nv
            if delta == 0:
                converged = True
        return values[self.initialState] 
