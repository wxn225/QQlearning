# -*- coding: utf-8 -*-
"""
@author: Hugo Gilbert and Bruno Zanuttini
"""

import random

# ===============================================================================================
# A class for an agent which always chooses the action at random in an MDP (among those allowed).
# Nothing here is specific to Gardner's dice or the SSB-Q-Learning algorithm.
# ===============================================================================================

class RandomAgent ():

    def __init__ (self, mdp):
        self.mdp = mdp

    def chooseAction (self, state):
        return random.choice(self.mdp.getAllowedActions(state))

    def inform (self, state, action, nextState):
        pass

    def __str__ (self):
        return "Random Agent"
