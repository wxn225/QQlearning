# -*- coding: utf-8 -*-
"""
@author: Hugo Gilbert and Bruno Zanuttini
"""

from gardner import *
from qqlearning import *

# ===============================================================================================
# The following observation functions are intended to be called by the simulation module every
# step-th step of the simulation.
# Each of them prints different information on the current state of the SSB-Q-learning algorithm.
# Some functions are specific to the Gardner's dice problems (as suggested by their name).
# ===============================================================================================

# Initial state must be s0
def observeGardner (mdp, ssbQLearning, initialState, step):
    if initialState!="s0":
        raise "Cannot observe Gardner from initial state other than s0"
    print "Plays",
    print "A wp", "%0.2f" % (ssbQLearning.getPolicy(initialState,"throw(A)")*13),
    print "B wp", "%0.2f" % (ssbQLearning.getPolicy(initialState,"throw(B)")*13),
    print "C wp", "%0.2f" % (ssbQLearning.getPolicy(initialState,"throw(C)")*13),
    print "after",step,"steps (probabilities rescaled to [0,13])"

# Initial state must be s0
def observeSequentialGardner (mdp, ssbQLearning, initialState, step):
    if initialState!="s0":
        raise "Cannot observe sequential Gardner from initial state other than s0"
    print "Plays",
    print "A wp", "%0.2f" % (ssbQLearning.getPolicy("s0","throw(A)")*13),
    print "B wp", "%0.2f" % (ssbQLearning.getPolicy("s0","not-throw(A)")*ssbQLearning.getPolicy("sBC","throw(B)")*13),
    print "C wp", "%0.2f" % (ssbQLearning.getPolicy("s0","not-throw(A)")*ssbQLearning.getPolicy("sBC","throw(C)")*13),
    print "after",step,"steps (probabilities rescaled to [0,13])"

# Ignores the given initial state
def observeWealthExpectations (mdp, ssbQLearning, initialState, step):
    print "----"
    print "At step",step
    for state in mdp.getStates():
        if state=="s0" or state=="sBC":
            print "Wealth expectation in",state,":",
            for wealthLevel in mdp.getWealthLevels():
                print wealthLevel,"=",ssbQLearning.getStateWealthExpectation(state,wealthLevel),
            print ""
            for action in mdp.getAllowedActions(state):
                print "Wealth expectation in",state,"for action",action,":",
                for wealthLevel in mdp.getWealthLevels():
                    print wealthLevel,"=",str(ssbQLearning.getWealthExpectation(state,action,wealthLevel)),"|",
                print ""
    print "----"

# Ignores the given initial state
def observeVisitsAndExperiences (mdp, ssbQLearning, initialState, step):
    print "----"
    print "At step",step
    for state in mdp.getStates():
        print "Number of visits to",state," = ",ssbQLearning.getNbVisits(state)
        if state=="s0" or state=="sBC":
            print "Number of experiences in",state,":",
            for action in mdp.getAllowedActions(state):
                print action,"=",str(ssbQLearning.getNbExperiences(state,action)),"|",
            print ""
    print "----"

# Ignores the given initial state
def observeReturnSequentialGardner (mdp, ssbQLearning, initialState, step):
    # Computes the real expected proportion of a given wealth level
    # given that the current mixed policy is played starting from state
    def getReturn (state,wealthLevel):
        if mdp.isFinal(state):
            if mdp.getWealthLevel(state)==wealthLevel:
                return 1.
            return 0.
        res = 0.
        for action in mdp.getAllowedActions(state):
            for nextState in mdp.getStates():
                if mdp.getProbability(state,action,nextState)!=0.:
                    res += ssbQLearning.getPolicy(state,action)*mdp.getProbability(state,action,nextState)*getReturn(nextState,wealthLevel)
        return res
    # This function itself
    for state in ["s0","sBC"]:
        print "Estimated return of current policy in",state,"after",step,"steps (vs real return):"
        for wealthLevel in mdp.getWealthLevels():
            print wealthLevel,"=","%0.2f" % ssbQLearning.getStateWealthExpectation(state,wealthLevel),
            print "(vs","%0.2f" % getReturn(state,wealthLevel)+")","|",
        print ""

# Ignores the given initial state
def observeBRReturnSequentialGardner (mdp, ssbQLearning, initialState, step):
    # Computes the real expected proportion of a given wealth level
    # given that the current BR policy is played starting from state
    def getReturn (state,wealthLevel):
        if mdp.isFinal(state):
            if mdp.getWealthLevel(state)==wealthLevel:
                return 1.
            return 0.
        res = 0.
        # Repeat because choice of best action is stochastic
        for i in xrange(1000):
            action = ssbQLearning.getBestAction(state,ssbQLearning.wealthExpectations)
            for nextState in mdp.getStates():
                if mdp.getProbability(state,action,nextState)!=0.:
                    res += mdp.getProbability(state,action,nextState)*getReturn(nextState,wealthLevel)
        return res/1000.
    # This function itself
    for state in ["s0","sBC"]:
        print "Estimated return of BR policy in",state,"after",step,"steps (vs real return):"
        action = ssbQLearning.getBestAction(state,ssbQLearning.wealthExpectations)
        for wealthLevel in mdp.getWealthLevels():
            print wealthLevel,"=","%0.2f" % ssbQLearning.getBRExpectation(state,action,wealthLevel),
            print "(vs","%0.2f" % getReturn(state,wealthLevel)+")","|",
        print ""

# Ignores the given initial state
def observePolicySequentialGardner (mdp, ssbQLearning, initialState, step):
    for state in ["s0","sBC"]:
        print "Policy for",state+":",
        stateString = ""
        for action in mdp.getAllowedActions(state):
            probability = ssbQLearning.getPolicy(state,action)
            if probability!=0:
                stateString += action+" = "+str(probability)+" | "
        if stateString!="":
            print stateString,
        else:
            print "state not visited",
        print "after",step,"steps"

# Ignores the given initial state
def observeExplorationExploitation (mdp, ssbQLearning, initialState, step):
    if step!=0:
        print "After",step,"steps,",
        print "exploration: epsilon","%0.2f" % (ssbQLearning.nbEpsilons*100/float(step)),"%",
        print "currentEpsilon","%0.2f" % (ssbQLearning.nbCurrentEpsilons*100/float(step)),"%,",
        print "exploitation: sole action","%0.2f" % (ssbQLearning.nbSoleActions*100/float(step)),"%",
        print "real exploitation","%0.2f" % (ssbQLearning.nbExploitations*100/float(step)),"%"
