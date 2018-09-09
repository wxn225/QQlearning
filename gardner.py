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
class GardnerDiceMDP (MDP):

    def __init__ (self):
        states = ["s0"]+["rolled"+str(i) for i in xrange(1,7)]
        actions = ["reinit","throw(A)","throw(B)","throw(C)"]
        wealthLevels = ["w"+str(i) for i in xrange(1,7)]
        finalStates = ["rolled"+str(i) for i in xrange(1,7)]
        self.real_nash_equilibrium = {"w1":1./26,"w2":7./26, "w3":5./26, "w4":5./26, "w5":7./26, "w6":1./26}
        MDP.__init__(self,states,actions,wealthLevels,self.allowedActionsFunction,finalStates,self.wealthFunction,self.transitionFunction,self.ssbFunction,"s0","gardner")

    def allowedActionsFunction (self, state):
        if state!="s0":
            return ["reinit"]
        return ["throw(A)","throw(B)","throw(C)"]

    def wealthFunction (self, finalState):
        return "w"+finalState[6:]

    def transitionFunction (self, state, action):
        if state=="s0":
            if action=="throw(A)":
                return {"rolled1":1/6.,"rolled4":5/6.}
            if action=="throw(B)":
                return {"rolled3":5/6.,"rolled6":1/6.}
            if action=="throw(C)":
                return {"rolled2":1/2.,"rolled5":1/2.}
        if state!="s0" and action=="reinit":
            return {"s0":1.}

    def ssbFunction (self, wealthLevel, otherWealthLevel):
        asInt = int(wealthLevel[1:])
        otherAsInt = int(otherWealthLevel[1:])
        if asInt>otherAsInt:
            return +10
        if otherAsInt>asInt:
            return -10
        return 0

    def __str__ (self):
        return "Gardner's dice"
        
# Sequential version of Gardner's dice
# There are 8 states, "s0" from which actions "throw(A)", "not-throw(A)" are available, with "not-throw(A)" leading
# deterministically to "sBC", from which "throw(B)" and "throw(C)" are available. The rest is similar to the nonsequential
# version of the problem.
class SequentialGardnerDiceMDP (MDP):

    def __init__ (self):
        states = ["s0"]+["sBC"]+["rolled"+str(i) for i in xrange(1,7)]
        actions = ["reinit","throw(A)","not-throw(A)","throw(B)","throw(C)"]
        wealthLevels = ["w"+str(i) for i in xrange(1,7)]
        finalStates = ["rolled"+str(i) for i in xrange(1,7)]
        self.real_nash_equilibrium = {"w1":1./26,"w2":7./26, "w3":5./26, "w4":5./26, "w5":7./26, "w6":1./26}
        MDP.__init__(self,states,actions,wealthLevels,self.allowedActionsFunction,finalStates,self.wealthFunction,self.transitionFunction,self.ssbFunction, "s0", "sequentialGardner")

    def allowedActionsFunction (self, state):
        if state!="s0" and state!="sBC":
            return ["reinit"]
        if state=="s0":
            return ["throw(A)","not-throw(A)"]
        return ["throw(B)","throw(C)"]

    def wealthFunction (self, finalState):
        return "w"+finalState[6:]

    def transitionFunction (self, state, action):
        if state=="s0":
            if action=="throw(A)":
                return {"rolled1":1/6.,"rolled4":5/6.}
            if action=="not-throw(A)":
                return {"sBC":1.}
        if state=="sBC":
            if action=="throw(B)":
                return {"rolled3":5/6.,"rolled6":1/6.}
            if action=="throw(C)":
                return {"rolled2":1/2.,"rolled5":1/2.}
        if state!="s0" and state!="sBC" and action=="reinit":
            return {"s0":1.}

    def ssbFunction (self, wealthLevel, otherWealthLevel):
        asInt = int(wealthLevel[1:])
        otherAsInt = int(otherWealthLevel[1:])
        if asInt>otherAsInt:
            return +10
        if otherAsInt>asInt:
            return -10
        return 0

    def __str__ (self):
        return "Sequential Gardner's dice"
