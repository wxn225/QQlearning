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

from garnets import *
from datacenter import *
from qqlearning import *
from qlearning import *
from observation import *
from simulation import *
import matplotlib.pyplot as plt

# Main functions ===========================================================

def printUsageAndExit ():
    print "Total step: nbSimulationSteps step epsilon <sequential>"
    print "Middle results every n step: Observation function is called after every step-th simulation step (and after the last one)"
    print "epsilon: Exploration probability epsilon must be in [0,1]"
    print "mdpType"
    print "tau: should be in [0,1]"
    print "optional: input <compare> to compare the performance of standard Q and QQ learning"
    sys.exit(1)

def plotResults(agent, nbSteps, step):
    for i in agent:
        print '--------------------------------------'
        print 'Here is the vector of wealth frequancies when exploiting and exploring: '
        print i.real_wealth_frequencies
        print '--------------------------------------'
        
        if not i.mdp.mdpType == "DataCenter" and not i.mdp.mdpType == "Garnets":
            n =  i.mdp.getWealthLevels()
            for w in n:
                print i.getCurrentRewardWealthLevel(w)
            print sum(i.getCurrentRewardWealthLevel(w)*i.real_wealth_frequencies[w] for w in n)
              
            plt.figure()
            #nbSteps = 20000000
            #step = 1000
            np.save("brv.npy", i.bestResponseValue)
            plt.plot(range(0,nbSteps,step),i.bestResponseValue)
            plt.ylabel(ur"$V^*_{\theta}(s_0)$")
            plt.xlabel('learning steps')

        #plt.figure()
        #plt.plot(agent.bestResponseValue)

        plt.figure()
        plt.title(str(i.initTheta)+" "+str(i.constant))
        np.save(str(agent.index(i))+"Score.npy", i.score)
        plt.plot(range(0,nbSteps,step),i.score)
        plt.ylabel('maxQ')
        plt.xlabel('learning steps')
        plt.xlim((-20000,nbSteps))
        
        plt.figure()
        plt.title(str(i.initTheta)+" "+str(i.constant))
        np.save(str(agent.index(i))+"Theta.npy", i.theta)
        plt.plot(i.thetas)
        plt.ylabel(ur"$\theta$")
        plt.xlabel('learning steps')
        plt.xlim((-20000,nbSteps))
        #plt.ylim((11,21))

    
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

#change the coefficient here
def produceAgent(mdp, initialState, epsilon, tau):
    #garnets theta:20 constant:5
    #datacenter theta:550 constant:100
    theta = [480]
    constant = [50]
    agent = []
    agentQ = []
    test = "theta"
    if test == "theta":
        for i in theta:
            agent.append(QQLearning(mdp,mdp.initialState,epsilon,tau,i,constant[0]))
    elif test == "const":
        for i in constant:
             agent.append(QQLearning(mdp,mdp.initialState,epsilon,tau,theta[0],i))
    else:
        agent.append(QQLearning(mdp,mdp.initialState,epsilon,tau,theta[0],constant[0]))
        agentQ.append(QLearning(mdp,mdp.initialState,epsilon))

    return agent, agentQ


# Called when all arguments have been retrieved and variables initialised
def main (mdp, epsilon, initialState, nbSteps, observationFunctions, tau, compare):

    agent,agentQ = produceAgent(mdp,mdp.initialState,epsilon,tau)

    print "Running simulation..."
    print "* problem:", mdp
    print "* agent:", agent
    print "* epsilon:", epsilon
    print "* initial state:", initialState
    print "* number of steps:", nbSteps
    if observationFunctions!=[]:
        print ""
    
    for i in agent:
        simulate(mdp,i,initialState,nbSteps,step,observationFunctions)
    if compare:
        simulate(mdp,agentQ[0],initialState,nbSteps,step,observationFunctions)

    print "Simulation: done."
    #printStats(mdp,trace)

    #plotResults(agentQ, nbSteps, step)
    if compare:
        plotResultsQ(agent,agentQ)
    else:
        plotResults(agent, nbSteps, step)

# Handling program arguments =================================================


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
compare = False
tau = 0

try:
    tau = float(sys.argv[5])
except ValueError:
    printUsageAndExit()
if tau<0 or tau>1:
    printUsageAndExit()

#Add model here
if sys.argv[4]=="garnets":
       mdp = GarnetsMDP()
elif sys.argv[4]=="datacenter":
    mdp = DataCenterMDP()
else:
    printUsageAndExit()

if len(sys.argv) > 6:
    if sys.argv[6] == "compare":
        compare = True

# Simulating ==================================================================

main(mdp,epsilon,mdp.initialState,nbSteps,observationFunctions,tau,compare)






