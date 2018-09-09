# -*- coding: utf-8 -*-
"""
@author: Hugo Gilbert and Bruno Zanuttini
"""

from mdp import *
import random

# ===============================================================================================
# Two classes for representing the standard and the sequential versions of Gardner's dice as
# MDPs. Nothing here is specific to the SSB-Q-Learning algorithm whatsoever.
# ===============================================================================================

# Non sequential version of Gardner's dice
# There are 7 states, "s0" from which actions "throw(A)", "throw(B)", and "throw(C)" are available,
# and "rolled1",...,"rolled6" with wealthLevels "w1",...,"w6", resp., where "wI" is preferred to "wJ" if and only if
# I > J holds (always with magnitude 1). All states but "s0" are final, and in other states there is only the
# deterministic action "reinit", which leads to "s0"
class GridMDP (MDP):

    def __init__ (self):
        states = ["s"+str(i)+str(j) for i in xrange(3) for j in xrange(3)]
        actions = ["reinit","left","up"]
        wealthLevels = ["w"+str(i) for i in xrange(1,4)]
        finalStates = ["s20","s02","s22"]
        self.real_nash_equilibrium = {"w1":1./3,"w2":1./3, "w3":1./3}
        MDP.__init__(self,states,actions,wealthLevels,self.allowedActionsFunction,finalStates,self.wealthFunction,self.transitionFunction,self.ssbFunction,"s00","grid")

    def allowedActionsFunction (self, state):
        if state in self.finalStates:
            return ["reinit"]
        return ["left","up"]

    def wealthFunction (self, finalState):
        if finalState == "s20":
            return "w1"
        if finalState == "s22":
            return "w2"
        if finalState == "s02":
            return "w3"


    def transitionFunction (self, state, action):
        if action=="reinit":
            return {"s00":1.}
        if action=="left":
            if int(state[1]) == 2:
                return {state:0.8,"s22":0.2}
            if int(state[2]) == 2:
                return {state:0.2,"s22":0.8}
            return{"s"+ str(int(state[1])+1) +state[2]:0.8, "s"+state[1] +str(int(state[2])+1):0.2}
        if action=="up":
            if int(state[2]) == 2:
                return {state:0.8,"s22":0.2}
            if int(state[1]) == 2:
                return {state:0.2,"s22":0.8}
            return{"s"+ str(int(state[1])+1) +state[2]:0.2, "s"+state[1] +str(int(state[2])+1):0.8}
            

    def ssbFunction (self, wealthLevel, otherWealthLevel):
        asInt = int(wealthLevel[1:])
        otherAsInt = int(otherWealthLevel[1:])
        sign = 1
        if otherAsInt>asInt:
            sign=-1            
        if asInt==otherAsInt:
            return 0
        if abs(asInt-otherAsInt)==1:
            return sign*10
        if abs(asInt-otherAsInt)==2:
            return -sign*10
        return 0

    def __str__ (self):
        return "Grid"
        
