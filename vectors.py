# -*- coding: utf-8 -*-
"""
@author: Hugo Gilbert and Bruno Zanuttini
"""

# ===============================================================================================
# Utilities for manipulation of vectors. Vectors are represented by lists.
# ===============================================================================================

def innerProduct (vector, otherVector):
    if len(vector)!=len(otherVector):
        raise "Cannot compute inner product of "+str(vector)+" and "+str(otherVector)+", they have different lengths"
    res = 0.
    for i in xrange(len(vector)):
        res += vector[i]*otherVector[i]
    return res

# ===============================================================================================
# Utilities for manipulation of distributions (of probabilities). Distributions are represented
# by dictionaries (sparse, with null entries omitted).
# ===============================================================================================

# Updates a distribution D by (1-coef)D(elem)+coef*value for some element, and (1-coef)D(y) for others
# Sets element to value if distribution is null everywhere
def dynamic_mean (distribution, element, coefficient):
    isNull = True
    for oneElement in distribution.keys():
        if distribution[oneElement]!=0.:
            isNull = False
            break
    if isNull:
        distribution[element] = 1
        return
    if not element in distribution.keys():
        distribution[element] = 0.
    for oneElement in distribution.keys():
        if oneElement==element:
            distribution[oneElement] = (1.-coefficient)*distribution[oneElement]+coefficient
        else:
            distribution[oneElement] = (1.-coefficient)*distribution[oneElement]
    # elements not in the distribution remain at 0


# Turns a distribution to a vector of probabilities, given some order on all elements (as a list).
def distributionToVector (distribution, allElements):
    res = []
    for element in allElements:
        if element in distribution.keys():
            res.append(distribution[element])
        else:
            res.append(0.)
    return res

# Turns a distribution to a string, given some order on all elements (as a list).
def distributionToString (distribution, allElements):
    return vectorToString(distributionToVector(distribution,allElements))

# Turns a vector of floats to a string
def vectorToString (vector):
    res = "| "
    for element in vector:
        res += ("%0.2f" % element)+" | "
    return res
