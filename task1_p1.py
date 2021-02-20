"""
Chris Um Task 1 Problem 1



"""

import numpy as np
import math
import random

initial_state = np.array([[1],[0]])
# The above state is an initial state we multiply our
# variational (parametric) circuit to generate the most
# general 1 qubit state.

def U3(theta, phi, lambda_):
    # This is a U3 gate that acts as a variational circuit to transform
    # our initial state to a Bloch sphere state, where it has
    # two parameters (theta and phi) that define a point on the Bloch
    # sphere.
    
    return [[math.cos(theta/2),-complex(math.cos(lambda_),math.sin(lambda_))*math.sin(theta/2)],[complex(math.cos(phi),
            math.sin(phi))*math.sin(theta/2), complex(math.cos(phi+lambda_),math.sin(phi+lambda_))*math.cos(theta/2)]]

def Bloch(theta, phi, lambda_):
    # generates the most generalized 1-qubit state,
    # a state where it covers every point lying
    # on the Bloch sphere.
    
    return np.dot(U3(theta, phi, lambda_),initial_state)


def check(final_state):
    # This is an error bound function, where it tests
    # whether the final state generated from a random
    # set of parameters lies on the Bloch sphere
    # We declare the maximum error bound as 0.01
    
    if(1-(abs(final_state[0])+abs(final_state[1])) < 0.00002):
        return True

cnt = 0

# We now randomly generate our state 1000 times with random parameters.
# We count the number of times our randomly generated state lies on the
# Bloch sphere.

for x in range(0,5000):
    final_state = Bloch(random.uniform(0,2*math.pi),random.uniform(0,2*math.pi),random.uniform(0,2*math.pi))
    if(check(final_state)):
        cnt += 1

print(cnt)