"""
Chris Um Task 1 Problem 2

"""

import numpy as np
import math
import random

initial_state = np.array([[1],[0]])

unknown = Bloch(random.uniform(0,2*math.pi),random.uniform(0,2*math.pi),random.uniform(0,2*math.pi))
# This is the randomly generated unknown state that
# we wish to reproduce. 

H = (1/math.sqrt(2))*np.array([[1,1],[1,-1]]) # Hadamard gate
P0x0 = np.array([[1,0],[0,0]]) # \ket{0}\bra{0} projection
P1x1 = np.array([[0,0],[0,1]]) # \ket{1}\bra{1} projection
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) # SWAP gate

known = np.array([[0],[1]])
# This is the state we SWAP test
# with the unknown state to examine
# how much they differ, allowing us
# to find the unknown from the
# difference. 

I = np.array([[1,0],[0,1]]) # Identity size 2
CSWAP = np.kron(np.kron(P0x0,I),I)+np.kron(P1x1,SWAP) # CSWAP gate
ancilla_bit = np.array([[1],[0]]) # ancilla bit initialized to 0

def U3(theta, phi, lambda_):
    # This is a U3 gate that acts as a variational circuit to transform
    # our initial state to a Bloch sphere state, where it has
    # two parameters (theta and phi) that define a point on the Bloch
    # sphere.
    
    return [[math.cos(theta/2),-complex(math.cos(lambda_),math.sin(lambda_))*math.sin(theta/2)],[complex(math.cos(phi),math.sin(phi))*math.sin(theta/2), complex(math.cos(phi+lambda_),math.sin(phi+lambda_))*math.cos(theta/2)]]
    

def Bloch(theta, phi, lambda_):
    # generates the most generalized 1-qubit state,
    # a state where it covers every point lying
    # on the Bloch sphere.
    
    return np.dot(U3(theta, phi, lambda_),initial_state)

def rev_H(col_vector):
    cnt = 0
    for i in range(int(len(col_vector)/2)):
        cnt += abs(col_vector[i])**2
    return np.array([[math.sqrt(cnt)],[math.sqrt(1-cnt)]])

def complexx(final_state):
    # converts amplitudes to probabilities
    
    for x in range(len(final_state)):
        final_state[x] = abs(final_state[x])**2

sum = 0
# The following is the algorithmic procedure of the SWAP test.
# We start with initializing P = 100, the number of times
# the algorithm will be executed. 
for i in range(100):
    
    ancilla_bit = np.dot(H, ancilla_bit) # Apply a Hadamard gate to the
                                         # ancilla qubit. 
    for i in range(1): # since "unknown" is an 1 qubit state, n = 1
        col_vector8 = np.dot(CSWAP,(np.kron(ancilla_bit,np.kron(known,unknown))))
        
        # Applies CSWAP to (ancilla, known, unknown)
    
    col_vector2 = rev_H(col_vector8) # Since col_vector8 is a tensor product
                                     # we have to look in terms of the single
                                     # qubit (ancilla) only to properly
                                     # apply Hadamard to ancilla.
    
    col_vector2 = np.dot(H, col_vector2) # Hadamard is applied to the
                                         # ancilla qubit.
                                         
    col_vector2 = complexx(col_vector2)  # Before we do the measurement,
                                         # we have to convert amplitudes
                                         # to probabilities in order for
                                         # col_vector2 to be a prob. distribution,
                                         # since we apply a random sampling method.
                                         
    M_i = random.choices(np.array([[0],[1]]), weights = col_vector2, k=1)
    # We measure the ancilla qubit in the Z basis and record the measurement. 
    
    sum += M_i[0]

s = 1-(2/100)*sum # We compute the magnitude squared of the inner product of unknown and known state.
                  # s is how much unknown and known differ, scaling from 1 as equal, and 0 being orthogonal.
                  
print("unknown state is:" + str(math.sqrt(1-abs(s)))+ "\ket{0}"+ " " + str(math.sqrt(abs(s)))+ "\ket{1}")

# By obtaining s, we are able to examine how much of a difference there is between known and unknown.
# since known is [[0],[1]], we know that s would be 1 if unknown was [[0],[1]]. Therefore,
# whatever s is, we can express the amplitudes of unknown in terms of s now.
# s is the amplitude of \ket{1} state of the unknown. 
# We finally print our prediction of the randomly generated unknown state. 

        