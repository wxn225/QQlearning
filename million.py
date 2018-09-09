# -*- coding: utf-8 -*-
"""
@author: Hugo Gilbert and Bruno Zanuttini
"""

from mdp import *
import random
import numpy as np

# ===============================================================================================
# class for representing the game who wants to be a millionnaire as an 
# MDPs. Nothing here is specific to the SSB-Q-Learning algorithm whatsoever.
# ===============================================================================================

# Who wants to be a millionnaire.
# There are 15 questions, two guarentee points (questions 5 and 10) and three lifelines
# There are 16 wealth levels according to the numbers of questions correctly answered.
# A state is final if the game is lost, if the player decided to stop playing or if all questions have been answered successfully
class MillionMDP (MDP):

    #Conventions for states
    #losti means that the player has lost and returns to the wealth level of question i
    #stopi means that the player decided tu stop playing and keeps the wealth level of question i
    #answeredall: all questions where correctly answered and the player wins 330000 euros
    #play_101_t means that the player is currently playing, he has succesfully answered t questions and he still has lifelines 1 and 3 (the _xxx_ in the middle of the state name represents three booleans saying which lifelines are still usable)
    #play_with_011 means that the players answers the current question using lifelines 2 and 3. 
    def __init__ (self):
        states = ["lost"+str(i) for i in [0,5,10]] + ["stop"+str(i) for i in xrange(15)] + ["play_" + jokers + str(i) for jokers in ["111_","110_","101_","011_","100_","010_","001_","000_"] for i in xrange(15)] + ["answered_all"]
        actions = ["reinit","stop","play_with_000","play_with_100", "play_with_010", "play_with_001", "play_with_110", "play_with_101", "play_with_011", "play_with_111"]
        wealthLevels = ["w"+str(i) for i in [0, 150, 300, 450, 900, 1800, 2100, 2700, 3600, 4500, 9000, 18000, 36000, 72000, 144000, 330000]]
        finalStates = ["lost"+str(i) for i in [0,5,10]] + ["stop"+str(i) for i in xrange(15)] + ["answered_all"]
        self.real_nash_equilibrium = {"w0":0.32873626, "w150":0.,"w300":0,"w450":0.,"w900":0.,"w1800":0.36379174,"w2100":0.,"w2700" : 0.21313447,"w3600" : 0.09433753,"w4500" : 0. , "w9000":0.,"w18000":0., "w36000":0.,"w72000":0.,"w144000":0.,  "w330000":0.  }
        MDP.__init__(self,states,actions,wealthLevels,self.allowedActionsFunction,finalStates,self.wealthFunction,self.transitionFunction,self.ssbFunction, "play_111_0", "million")

    
    def allowedActionsFunction (self, state):
        if state[0:4]!="play":
            return ["reinit"]
	if state[5:8]=="111":
            return self.actions[1:]
	if state[5:8]=="110":
            return ["stop","play_with_000","play_with_100", "play_with_010", "play_with_110"]
	if state[5:8]=="101":
            return ["stop","play_with_000","play_with_100", "play_with_001", "play_with_101"]
	if state[5:8]=="011":
            return ["stop","play_with_000","play_with_010", "play_with_001", "play_with_011"]
	if state[5:8]=="100":
            return ["stop","play_with_000","play_with_100"]
	if state[5:8]=="010":
            return ["stop","play_with_000","play_with_010"]
	if state[5:8]=="001":
            return ["stop","play_with_000","play_with_001"]
	if state[5:8]=="000":
            return ["stop","play_with_000"]
	print "I did not recognize the state " + state + " in allowedActionsFunction"
    	sys.exit(1)

    def wealthFunction (self, finalState):
	if finalState == "answered_all":
	    return "w330000"
        return self.wealthLevels[int(finalState[4:])]

    def transitionFunction (self, state, action):
	jokerFactors = np.array([0,0,0, 0.672,0.527,0.745, 0.698,0.547,0.773, 0.707,0.554,0.783, 0.711,0.557,0.788, 0.714,0.559,0.791, 0.716,0.561,0.793,0.717,0.562,0.795,0.718,0.563,0.796,0.719,0.563,0.796,0.719,0.564,0.797,0.720,0.564,0.798,0.720,0.564,0.798,0.721,0.565,0.799,0.721,0.565,0.799]) 
	jokerFactors = jokerFactors.reshape(15,3)
	#jokerFactors[t][i] is the multiplicative coefficient that reduces the failure probability if lifeline i is used at question t
	failure = [(0.004+0.051*t) for t in xrange(15)]
	
	if action == "reinit":
	   return {"play_111_0":1.}
	
	time = state[9:]
	if action == "stop":  
	   return {"stop"+time:1.} 	
	j1, j2, j3 = int(state[5]), int(state[6]), int(state[7])
	is_used_j1, is_used_j2, is_used_j3 = int(action[10]), int(action[11]), int(action[12]) 
        t = int(time)

	jf = 1
	if (is_used_j1 == 1):
		jf *= jokerFactors[t,0]
	if (is_used_j2 == 1):
		jf *= jokerFactors[t,1]
	if (is_used_j3 == 1):
		jf *= jokerFactors[t,2]
	
	last_checked_point = 0
	if t >= 5:
	    last_checked_point = 5
	if t >= 10:
	    last_checked_point = 10
	
	if(t == 14):
	    return {"lost"+str(last_checked_point) : jf*failure[t],"answered_all":1-jf*failure[t]}
	return {"lost"+str(last_checked_point) : jf*failure[t],"play_"+str(j1-is_used_j1)+str(j2-is_used_j2)+str(j3-is_used_j3)+"_"+str(t+1):1-jf*failure[t]}		
        
    def ssbFunction (self, wealthLevel, otherWealthLevel):
        asInt = int(wealthLevel[1:])
        otherAsInt = int(otherWealthLevel[1:])
        if asInt>otherAsInt:
            return +10
        if otherAsInt>asInt:
            return -10
        return 0

    def __str__ (self):
        return "Who wants to be a millionnaire"
