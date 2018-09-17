# -*- coding: utf-8 -*-
"""
@author: Xining Wang
"""

from mdp import *
import random


# ===============================================================================================
# class for representing the Garnets as an 
# MDPs. Nothing here is specific to the SSB-Q-Learning algorithm whatsoever.
# ===============================================================================================

# Garnets is an MDP set with ns states, nA actions in each states and ceil(log2 ns) successor
# states of each action

class GarnetsMDP (MDP):

    def __init__ (self):
        states = ["s"+"0"+str(i) for i in xrange(10)] + ["s"+str(i) for i in xrange(10,100)]
        actions = ["reinit"] + ["A"+str(i) for i in xrange(1,6)]
        wealthLevels = ["w"+str(i) for i in xrange(1,21)]
        finalStates = ["s" + str(i) for i in xrange(80,100)]
        #self.real_nash_equilibrium = {"w1":1./3,"w2":1./3, "w3":1./3}
        MDP.__init__(self,states,actions,wealthLevels,self.allowedActionsFunction,finalStates,self.wealthFunction,self.transitionFunction,self.ssbFunction,"s00","Garnets",50)
        self.transitionTable = {}
        self.generateTransitionTable()

    def allowedActionsFunction (self, state):
        if state in self.finalStates:
            return ["reinit"]
        return self.actions[1:]

    def allowedMdpActionsFunction (self, state):
        if state in self.finalStates:
            return ["reinit"]
        return self.actions[1:]

    def wealthFunction (self, finalState):
        return "w" + str(int(finalState[1:])-79)

    def generateTransitionTable(self):
        random.seed(8)
        realState = self.states[:]
        for i in self.states[:-20]:
            realState = self.states[:]
            realState.remove(i)
            self.transitionTable[i] = {}
            for j in self.actions[1:]:
                self.transitionTable[i][j] = {}
                pRest = 1
                for k in xrange(7):
                    nextState = random.choice(realState)
                    while nextState in self.transitionTable[i][j].keys():
                        nextState = random.choice(realState)
                    if k < 6:
                        p = random.uniform(0,pRest)
                        pRest -= p
                    else:
                        p = pRest
                    self.transitionTable[i][j][nextState] = p
        return



    def transitionFunction (self, state, action):
        if action == "reinit":
            return {"s00":1.}

        return self.transitionTable[state][action]

   
    def ssbFunction():
        pass


    def __str__ (self):
        return "Garnets"


def main():
    mdp = GarnetsMDP()
    print(mdp.transitionTable)

if __name__ == "__main__":
    main()
        
