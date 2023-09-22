#!/usr/bin/env python3

import random
import math
import numpy as np

def initweights():                      # Random weights initialization
    if inityp == 'uniform':
        return [random.uniform(-1,1) for i in range(9)]
    if inityp == 'normal':
        return [np.random.normal(0,0.2) for i in range(9)]

def actfun(val):                        # Activation functions
    if actyp == "sigmoid":
        return (1/(1+math.exp(-val)))
    if actyp == "tanh":
        return math.tanh(val)
    if actyp == "relu":
        return max(0,val)
    
def activations(weights, input):        # Calculates activations of hidden and output nodes
    net0 = input[0]*weights[0] + input[1]*weights[1] + weights[2]
    y0 = actfun(net0)
    net1 = input[0]*weights[3] + input[1]*weights[4] + weights[5]
    y1 = actfun(net1)
    net = y0*weights[6] + y1*weights[7] + weights[8]
    y = actfun(net)
    return(y0, y1, y)

def xor_net(weights, input):            # Returns value of output node
    act = activations(weights, input)
    return act[2]

def mse(weights, inputs, targets):      # Calculates MSE + number of misclassified inputs
    tot, misit = 0, 0
    for index, input in enumerate(inputs):
        tot += (1/2)*(xor_net(weights, input)-targets[index])**2
        if (targets[index] == 0) and (xor_net(weights,input) > 0.5):    # Outcome > 0.5 is considered 1
            misit += 1
        if (targets[index] == 1) and (xor_net(weights,input) <= 0.5):   # Outcome <= 0.5 is considered 0
            misit += 1
    return tot, misit

def grdmse(weights, input, target):    # Output length == input vector weights
    act = activations(weights, input)
    pw = (act[2]-target)*act[2]*(1-act[2])
    pu = (act[2]-target)*act[2]*(1-act[2])*act[0]*(1-act[0])*weights[6]
    pv = (act[2]-target)*act[2]*(1-act[2])*act[1]*(1-act[1])*weights[7]
    return [pu*input[0], pu*input[1], pu, pv*input[0], pv*input[1], pv, pw*act[0], pw*act[1], pw]

def gradec(weights, inputs, targets):                       # Gradient descent 
    it = 0
    curmse, misit = mse(weights, inputs, targets)           # Calculate MSE and #misclassified items
    while (curmse > traincut) and (it < trainit):           
        it += 1
        for index, input in enumerate(inputs):
            tgrad = grdmse(weights, input, targets[index])  # Gradient of MSE
            for index1 in range(len(weights)):
                weights[index1] -= eta * tgrad[index1]      # Update rule
        curmse, misit = mse(weights, inputs, targets)
    print("Training complete")
    print("Iterations: ", it, " MSE: ", curmse, " Misclassified: ", misit)
    print("Weights after training: ", weights)
    return

eta = 20                                    # Learning rate
actyp = "relu"                           # Activation function type: sigmoid, tanh or relu
inityp = "uniform"                          # Initialization strategy: normal or uniform
typ = "training"                            # Lazy random trial & error or training
inputs = [[0,0], [1,0], [0,1], [1,1]]       # Training inputs
targets = [0,1,1,0]                         # Training outputs
trainit = 1000                              # Maximum number of training iterations
traincut = 0.1                              # Upper bound for target MSE

if typ == "lazy":
    for i in range(trainit):
        weights = initweights()                         # Weight initialization
        curmse, misit = mse(weights, inputs, targets)   # Calculate MSE and #misclassified items
        print("Iteration ", i,": MSE = ", curmse, " Misclassified inputs: ", misit)
        if curmse < traincut:
            break
else:
    weights = initweights()                         # Weight initialization
    print("Weights before training: ", weights)
    gradec(weights, inputs, targets)
