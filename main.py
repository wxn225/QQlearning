# -*- coding: utf-8 -*-
"""
@author: Hugo Gilbert and Bruno Zanuttini
"""

# ===============================================================================================
# A main program for launching simulations on Gardner's dice and SSQ-Q-learning (or random)
# agents. Usage is as given by the printUsageAndExit function.
# Lines marked "optional" give control about what information is displayed every step-th step.
# The agent may be changed to/from random/SSB-Q-learning in the dedicated section.
# ===============================================================================================

import sys

from gardner import *
from million import *
from grid import *
from garnets import *
from datacenter import *
from qqlearning import *
from qlearning import *
from observation import *
from simulation import *
import matplotlib.pyplot as plt

# Main functions ===========================================================

def printUsageAndExit ():
    print "Usage: nbSimulationSteps step epsilon <sequential>"
    print "Observation function is called after every step-th simulation step (and after the last one)"
    print "Exploration probability epsilon must be in [0,1]"
    print "If last argument is not given, nonsequential Gardner's dice are played"
    sys.exit(1)

def plotResults(agent, nbSteps, step):
    print '--------------------------------------'
    print 'Here is the vector of wealth frequancies when exploiting: '
    print  agent.wealth_frequencies
    print '--------------------------------------'
    
    print '--------------------------------------'
    print 'Here is the vector of wealth frequancies when exploiting and exploring: '
    print agent.real_wealth_frequencies
    print '--------------------------------------'
    
    if not agent.mdp.mdpType == "DataCenter" and not agent.mdp.mdpType == "Garnets":
        n =  agent.mdp.getWealthLevels()
        for w in n:
            print agent.getCurrentRewardWealthLevel(w)
        print sum(agent.getCurrentRewardWealthLevel(w)*agent.real_wealth_frequencies[w] for w in n)
          
        plt.figure()
        #nbSteps = 20000000
        #step = 1000
        np.save("brv.npy", agent.bestResponseValue)
        plt.plot(range(0,nbSteps,step),agent.bestResponseValue)
        plt.ylabel(ur"$V^*_{\theta}(s_0)$")
        plt.xlabel('learning steps')

    #plt.figure()
    #plt.plot(agent.bestResponseValue)

    plt.figure()
    np.save("agentScore.npy", agent.score)
    plt.plot(range(0,nbSteps,step),agent.score)
    plt.ylabel('score')
    plt.xlabel('learning steps')
    plt.xlim((-20000,nbSteps))
    
    plt.figure()
    np.save("agentTheta.npy", agent.theta)
    plt.plot(agent.thetas)
    plt.ylabel(ur"$\theta$")
    plt.xlabel('learning steps')
    plt.xlim((-20000,nbSteps))
    plt.ylim((11,21))

    
    plt.show()
    

def plotResultsQ(agent,agentQ):
    plt.figure()
    plotDataset = [[],[]]
    agent.histories = [i/100. for i in agent.histories[-10000:]]
    history = list(set(agent.histories))
    history.sort()
    count = len(agent.histories)
    cumNum = 0
    for data in history:
        cumNum += agent.histories.count(data)
        if data > 0:
            plotDataset[0].append(data)
            plotDataset[1].append(float(cumNum)/count)
    plt.plot(plotDataset[0],plotDataset[1],label="QQ-learning")
    plotDataset = [[],[]]

    agentQ.histories = [i/100. for i in agentQ.histories[-10000:]]
    history = list(set(agentQ.histories))
    history.sort()
    count = len(agentQ.histories)
    cumNum = 0
    for data in history:
        cumNum += agentQ.histories.count(data)
        if data > 0:
            plotDataset[0].append(data)
            plotDataset[1].append(float(cumNum)/count)
    plt.plot(plotDataset[0],plotDataset[1],label="Standard Q-learning")
    plt.legend(loc=0)
    plt.ylabel("cumulated probability")
    plt.xlabel("wealth for histories")
    plt.ylim((0,1.1))
    plt.show()

# Called when all arguments have been retrieved and variables initialised
def main (mdp, agent, agentQ, epsilon, initialState, nbSteps, observationFunctions):
    print "Running simulation..."
    print "* problem:", mdp
    print "* agent:", agent
    print "* epsilon:", epsilon
    print "* initial state:", initialState
    print "* number of steps:", nbSteps
    if observationFunctions!=[]:
        print ""
    trace = simulate(mdp,agent,initialState,nbSteps,step,observationFunctions)
    #simulate(mdp,agentQ,initialState,nbSteps,step,observationFunctions)
    if observationFunctions!=[]:
        print ""
    print "Simulation: done."
    #printStats(mdp,trace)

    plotResults(agent, nbSteps, step)
    #plotResults(agentQ, nbSteps, step)
    #plotResultsQ(agent,agentQ)

# Handling program arguments =================================================

if len(sys.argv)!=4 and len(sys.argv)!=5:
    printUsageAndExit()

# Retrieving number of steps

try:
    nbSteps = int(sys.argv[1])
except ValueError:
    printUsageAndExit()

# Retrieving step between observations

try:
    step = int(sys.argv[2])
except ValueError:
    printUsageAndExit()

# Retrieving epsilon

try:
    epsilon = float(sys.argv[3])
except ValueError:
    printUsageAndExit()
if epsilon<0 or epsilon>1:
    printUsageAndExit()

# Retrieving MDP and defining observation function ===========================

mdp = None
observationFunctions = []

if len(sys.argv)==5:
    if sys.argv[4]=="sequential":
        mdp = SequentialGardnerDiceMDP()   
        #observationFunctions.append(observeSequentialGardner) # optional
        #observationFunctions.append(observeReturnSequentialGardner) # optional
        #observationFunctions.append(observeBRReturnSequentialGardner) # optional
        #observationFunctions.append(observePolicySequentialGardner) # optional
        #observationFunctions.append(observeExplorationExploitation) # optional
        #observationFunctions.append(observeWealthExpectations) # optional
        #observationFunctions.append(observeVisitsAndExperiences) # optional
    elif sys.argv[4]=="million":
        mdp = MillionMDP() 
    elif sys.argv[4]=="grid":
        mdp = GridMDP() 
    elif sys.argv[4]=="garnets":
        mdp = GarnetsMDP()
    elif sys.argv[4]=="datacenter":
        mdp = DataCenterMDP()
    else:
        printUsageAndExit()
else:
    mdp = GardnerDiceMDP()
    #observationFunctions.append(observeGardner) # optional
    #observationFunctions.append(observeExplorationExploitation) # optional
    #observationFunctions.append(observeWealthExpectations) # optional
    #observationFunctions.append(observeVisitsAndExperiences) # optional

# Choosing agent =============================================================

#agent = RandomAgent(mdp) # one of this... (then comment out all observation functions)
agent = SSBQLearning(mdp,mdp.initialState,epsilon) # ... or that
agentQ = QLearning(mdp,mdp.initialState,epsilon)

# Simulating ==================================================================

main(mdp,agent,agentQ,epsilon,mdp.initialState,nbSteps,observationFunctions)






