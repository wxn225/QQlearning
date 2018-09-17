# -*- coding: utf-8 -*-
"""
@author: Hugo Gilbert and Bruno Zanuttini
"""

import random
import numpy as np

from vectors import *

# ===============================================================================================
# A class defining a generic "SSB Q learner". Nothing here is specific to Gardner's dice
# whatsoever.
#
# The algorithm uses the following:
#
# - nbVisits[state] is the number of visits to state so far
# - nbExperiences[state][action] is the number of times action has been taken in state so far
#
# It also uses the following quantities:
#
# - alpha(t) is the weight given to the t-th experience in a given couple (state,action), when
#   updating wealthExpectations[state][action]
# - beta(t) is the weight given to the t-th action taken in a given state, when updating the
#   current mixed policy
# ===============================================================================================

class QLearning ():

    # Initial state: from which it tries to maximise reward
    # Epsilon: if random(1)<epsilon, exploration
    def __init__ (self, mdp, initialState, epsilon):
        self.mdp = mdp
        self.initialState = initialState
        self.epsilon = epsilon
        self.wealth_frequencies = {}
        self.real_wealth_frequencies = {}
        self.bestResponseValue = []
        self.nbVisits = {}
        self.nbExperiences = {}
        self.QValues = {}
        self.score = []
        self.histories = []
        
        for state in self.mdp.getStates():
            self.nbVisits[state] = 0
            self.nbExperiences[state] = {}
            self.QValues[state] = {}
            for action in self.mdp.getAllowedActions(state):
                self.nbExperiences[state][action] = 0
                self.QValues[state][action] = 0
                #self.QValues[state]["stop"] = 0
        for state in self.mdp.getStates():
            self.QValues[state]["reinit"] = 0
            
        nb_wealth_levels = sum(1 for i in self.mdp.getWealthLevels())
        for wealth_level in self.mdp.getWealthLevels():
            self.wealth_frequencies[wealth_level] = 1/float(nb_wealth_levels)
            self.real_wealth_frequencies[wealth_level] = 1/float(nb_wealth_levels)
        # Debug information
        self.debug = False
        self.nbEpsilons = 0
        self.nbSoleActions = 0
        self.nbExploitations = 0
        self.nbWealthObtained = 0
        self.realNbWealthObtained = 0
        self.strategy = "epsilon-greedy"  #boltzmann or epsilon-greedy or epsilon-greedy-traj
        self.isRandomTraj = 0
        self.temperature = 5


    # If several optimal actions, random choice
    def chooseAction (self, state):
        self.mdp.counter += 1
        if self.mdp.counter == self.mdp.horizon:
            self.mdp.cumulatedCost = 0
            return "reinit"
        # Debug
        if self.debug and not self.mdp.isFinal(state):
            print "---------------"
            print "State",state
        allActions = self.mdp.getAllowedActions(state)
        # Only one action available
        if self.debug and not self.mdp.isFinal(state):
            print "Chooses",
        if len(allActions)==1:
            self.nbSoleActions += 1
            if self.debug and not self.mdp.isFinal(state):
                print allActions[0],"(sole action)"
            return allActions[0]
        
        # Exploration (epsilon-greedy)
        if self.strategy == "epsilon-greedy":
            choice = random.random()
            if choice<self.epsilon:
                self.nbEpsilons += 1
                allActions=self.mdp.getAllowedActions(state)[:]
                res = random.choice(allActions)
                if self.debug and not self.mdp.isFinal(state):
                    print res,"(exploring because of epsilon)"
                return res
            # Exploitation
            self.nbExploitations += 1
            return self.getBestAction(state)
            if self.debug and not self.mdp.isFinal(state):
                print res,"(best)"
        

    # Informs the algorithm of an experienced transition
    def inform (self, state, action, nextState):
        # Debug
        if self.debug and not self.mdp.isFinal(state):
            print "Informed of next state",nextState
        # Wealth obtained, if any
        wealthLevel = None
        reward = 0
        if self.mdp.isFinal(nextState):
            wealthLevel = self.mdp.wealthFunction(nextState)
            if self.isRandomTraj==0:
                self.nbWealthObtained += 1
            self.realNbWealthObtained += 1

        #self.rewards_obtained.append(reward)

        max_Q_next_state = None
        for nextAction in self.mdp.getAllowedActions(nextState):
            if self.QValues[nextState][nextAction] > max_Q_next_state or max_Q_next_state == None:
                max_Q_next_state = self.QValues[nextState][nextAction]

        
        # # Update of information piece by piece
        self.addVisit(state)
        self.addExperience(state,action)
        if self.mdp.mdpType == "DataCenter" and action != "reinit":
            activeServer = int(state[1:3])
            activeServerNext = int(nextState[1:3])
            arrivalNum = int(state[-2:])
            arrivalNumNext = int(nextState[-2:])
            self.mdp.cumulatedCost += activeServer*1 + 1*abs(activeServer-activeServerNext)
            if arrivalNum < activeServer:
                self.mdp.cumulatedCost += 1 * arrivalNum**2 / float(activeServer)
            else:
                self.mdp.cumulatedCost += 1 * arrivalNum * (activeServer + 10*(arrivalNum-activeServer)) / float(activeServer)
            if self.mdp.counter+1 == self.mdp.horizon:
                max_Q_next_state = 0
                if arrivalNumNext < activeServerNext:
                    self.mdp.cumulatedCost += 1 * arrivalNumNext**2 / float(activeServerNext)
                else:
                    self.mdp.cumulatedCost += 1 * arrivalNumNext * float(activeServerNext + 10*(arrivalNumNext-activeServerNext)) / activeServerNext
                reward = 2000 - self.mdp.cumulatedCost
                self.histories.append(reward)
            self.QValues[state][action] = self.QValues[state][action] + self.getAlpha(self.nbExperiences[state][action])*(reward + max_Q_next_state - self.QValues[state][action])

            #print "update cumulatedCost", self.mdp.cumulatedCost
        else: 
            if not self.mdp.isFinal(state):
                self.QValues[state][action] = self.QValues[state][action] + self.getAlpha(self.nbExperiences[state][action])*(reward + max_Q_next_state - self.QValues[state][action])
        
        # Debug
        if self.debug and not self.mdp.isFinal(state):
            print "Current reward in",self.initialState+":",
            print self.wealth_frequencies
    
    
    # Relative to some vector and the current reward function (stateWealthExpectations of initial state)
    # Expected reward of each action (i.e., vector of expected amount of each wealth level) is given
    # by information, which is supposed to be indexed by [state][action]
    # Random choice between actions with best value
    def getBestAction (self, state):
        # Maximisation
        bestValue = None
        bestActions = []
        for action in self.mdp.getAllowedActions(state):
            expectedReward = self.QValues[state][action]
            if bestValue==None or expectedReward>bestValue:
                bestValue = expectedReward
                bestActions = [action]
            elif expectedReward==bestValue:
                bestActions += [action]
        return random.choice(bestActions)

    # Handling of data structures ===================================================
    # alpha_t: update of wealth expectations
    def getAlpha (self, nbExperiences):
        return 1./(nbExperiences**(11/float(20)))

    # beta_t: update of policy coefficients
    def getBeta (self, nbExperiences):
        return 1./(nbExperiences**(4/float(6)))

    def getGamma (self, nbExperiences):
        return 1./nbExperiences

    # Handling of visits and experiences ===============================================
    def addVisit (self, state):
        if not state in self.nbVisits.keys():
            self.nbVisits[state] = 0
        self.nbVisits[state] += 1

    def getNbVisits (self, state):
        if not state in self.nbVisits.keys():
            return 0
        return self.nbVisits[state]

    def addExperience (self, state, action):
        if not state in self.nbExperiences.keys():
            self.nbExperiences[state] = {}
        if not action in self.nbExperiences[state].keys():
            self.nbExperiences[state][action] = 0
        self.nbExperiences[state][action] += 1

    def getNbExperiences (self, state, action):
        if not state in self.nbExperiences.keys() or not action in self.nbExperiences[state]:
            return 0
        return self.nbExperiences[state][action]

        
    # ==================================================================================

    def __str__ (self):
        return "SSQ Q-Learner"
