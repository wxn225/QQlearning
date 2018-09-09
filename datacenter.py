# -*- coding: utf-8 -*-
"""
@author: Xining Wang
"""

from mdp import *
import random
import math


# ===============================================================================================
# class for representing the data center control as an 
# MDPs. Nothing here is specific to the SSB-Q-Learning algorithm whatsoever.
# ===============================================================================================


class DataCenterMDP (MDP):

    def __init__ (self):
        self.server = 30
        self.poisson = {"low": self.server/2, "normal": self.server*3/2, "high": self.server*5/2}
        self.cumulatedCost = 0
        states = ["s"+"0"+str(i) + "0" + str(j) for i in xrange(1,10) for j in xrange(10)] + \
                    ["s"+"0"+str(i)+str(j) for i in xrange(1,10) for j in xrange(10,91)] + \
                    ["s"+str(i)+"0"+str(j) for i in xrange(10,31) for j in xrange(10)] + \
                    ["s"+str(i)+str(j) for i in xrange(10,31) for j in xrange(10,91)] + ["s0"]
        actions = ["reinit"] + ["A"+ "0" +str(i) for i in xrange(1,10)] + ["A"+str(i) for i in xrange(10,31)]# + ["stop"]
        wealthLevels = ["w"+str(i) for i in xrange(1,21)]  # xrange(1,22) with horizons
        finalStates = ["s0"]
        random.seed(10)
        #self.real_nash_equilibrium = {"w1":1./3,"w2":1./3, "w3":1./3}
        MDP.__init__(self,states,actions,wealthLevels,self.allowedActionsFunction,finalStates,self.wealthFunction,self.transitionFunction,self.ssbFunction,random.choice(states[:-1]),"DataCenter")
        self.transitionTable = {}
        self.generateTransitionTable()

    def allowedActionsFunction (self, state):
        return self.actions[1:]

    def wealthFunction (self, finalState):
        #if finalState == "s0":
        #   return "w1"
        return "w" + str(int(finalState[1:])-79) #78 with horizon

    def generateTransitionTable(self):
        #realState = self.states # [:-1] with horizon
        for i in self.states[:-1]:
            self.transitionTable[i] = {}
            for j in self.actions[1:]:
                self.transitionTable[i][j] = {}
                lam = 0
                if int(i[-1:]) < 20:
                    lam = self.poisson["low"]
                elif int(i[-1:]) < 40:
                    lam = self.poisson["normal"]
                else:
                    lam = self.poisson["high"]
                for nextArrival in xrange(91):
                    if nextArrival < 10:
                        nextArrivalStr = "0" + str(nextArrival)
                    else:
                        nextArrivalStr = str(nextArrival)
                    self.transitionTable[i][j]["s"+j[1:]+nextArrivalStr] = lam**nextArrival*math.exp(-lam)/math.factorial(nextArrival)
        return



    def transitionFunction (self, state, action):
        if action == "reinit":
            return {self.initialState:1.}

        if action == "stop":
            return {"s0":1.}
        
        return self.transitionTable[state][action]

   
    def ssbFunction():
        pass


    def __str__ (self):
        return "Datacenter"


def main():
    mdp = DataCenterMDP()
    print(mdp.transitionTable)

if __name__ == "__main__":
    main()
        
