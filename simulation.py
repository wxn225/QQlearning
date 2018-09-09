# -*- coding: utf-8 -*-
"""
@author: Hugo Gilbert and Bruno Zanuttini
"""

from mdp import *

# ===============================================================================================
# Various functions for launching simulations, analysing traces and printing statistics.
# Nothing here is specific to Gardner's dice or the SSB-Q-Learning algorithm whatsoever.
# ===============================================================================================

# Runs a simulation of an MDP starting from a given state, with an agent choosing actions in each state,
# and for a given number of steps (choices of actions along a unique trajectory).
# Each observationFunction in list observationFunctions: takes mdp, agent, initialState, step and is called after every
# step-th step and after the last one
# (can be used, e.g., for printing or for updating global variables)
# Returns a list of triples (state,action,nextState).
def simulate (mdp, agent, initialState, numberOfSteps, step, observationFunctions):
    res = []
    currentState = initialState
    currentAction = None
    for i in xrange(numberOfSteps):
        # Run observers
        if i%step==0:
            print i ,step
            for observationFunction in observationFunctions:
                observationFunction(mdp,agent,initialState,i)
            reward_dict = {}
            if not mdp.mdpType == "DataCenter" and not mdp.mdpType == "Garnets":
                for final_state in mdp.finalStates:
                    reward_dict[final_state] = agent.getCurrentRewardWealthLevel(mdp.wealthFunction(final_state))
                brv =  mdp.solveMDP(reward_dict)
                print "best response value ",brv
                agent.bestResponseValue.append(brv)
            state = agent.initialState
            sum_p = max(agent.QValues[state][action] for action in agent.mdp.getAllowedActions(state))
            agent.score.append(sum_p)
            cumulative = []
            sump = 1
            for w in agent.mdp.getWealthLevels():
                sump-= agent.real_wealth_frequencies[w]
                if(sump < agent.q):
                    print 'q ',w
                    break
            print sum(agent.getCurrentRewardWealthLevel(w)*agent.real_wealth_frequencies[w] for w in agent.mdp.getWealthLevels())
            print agent.theta
        # Execute transition
        currentAction = agent.chooseAction(currentState)
        nextState = mdp.drawNextState(currentState,currentAction)
        agent.inform(currentState,currentAction,nextState)
        res.append((currentState,currentAction,nextState))
        currentState = nextState
        print "debug", currentAction, nextState
    # Run observers after last step
    for observationFunction in observationFunctions:
        observationFunction(mdp,agent,initialState,numberOfSteps)
    return res

# Prints various statistics over a given execution trace
def printStats (mdp, trace):
    # print ""
    # print "Underlying distribution of next states for"
    # print ""
    # for state in mdp.getStates():
    #     for action in mdp.getAllowedActions(state):
    #         print "*",action,"in",state+":",
    #         distribution = mdp.getDistribution(state,action)
    #         for nextState in mdp.getStates():
    #             if nextState in distribution.keys():
    #                 if distribution[nextState]!=0:
    #                     print nextState,"=",str(distribution[nextState]),"|",
    #         print ""

    # print ""
    # print "Empirical distribution of next states for"
    # print ""
    # for state in mdp.getStates():
    #     for action in mdp.getAllowedActions(state):
    #         if isExperienced(state,action,trace):
    #             print "*",action,"in",state+":",
    #             for nextState in mdp.getStates():
    #                 frequency = empiricalFrequency(state,action,nextState,trace)
    #                 if frequency!=0:
    #                     print nextState,"=",str(frequency),"|",
    #             print ""
    #         else:
    #             print "*",action,"in",state+":","not visited"

    print ""
    print "Policy actually played for"
    print ""
    for state in mdp.getStates():
        if isVisited(state,trace):
            print "*",state+":",
            for action in mdp.getAllowedActions(state):
                probability = actualPolicy(state,action,trace)
                if probability!=0:
                    print action,"=",str(probability),"|",
            print ""
        else:
            print "*",state+":","not visited"

# Whether there is at least one triple (state,.,.) in a given trace for given state.
def isVisited (state, trace):
    for (s,a,ns) in trace:
        if s==state:
            return True
    return False

# Whether there is at least one triple (state,action,.) in a given trace for given
# state and action.
def isExperienced (state, action, trace):
    for (s,a,ns) in trace:
        if s==state and a==action:
            return True
    return False

# Frequency of given next state in triples (state,action,.) for given state and action,
# in trace given as list of triples (state,action,nextState). Assumes that there is
# at least one experience (state,action,.) in the trace (use isExperienced to decide this).
def empiricalFrequency (state, action, nextState, trace):
    totalNumber = sum([1 for (s,a,ns) in trace if s==state and a==action])
    number = sum([1 for (s,a,ns) in trace if s==state and a==action and ns==nextState])
    return number/float(totalNumber)

# Frequency of play of given action in triples (state,action,.) for given state,
# in trace given as list of triples (state,action,nextState). Assumes that there is
# at least one experience (state,.,.) in the trace (use isVisited to decide this).
def actualPolicy (state, action, trace):
    totalNumber = sum([1 for (s,a,ns) in trace if s==state])
    number = sum([1 for (s,a,ns) in trace if s==state and a==action])
    return number/float(totalNumber)
