'''
    Define some activation function in the module
'''

import math
import numpy as np

def Sign(WX):
    '''
        Sign
    '''
    return -1 if WX < 0 else 1

def Sigmoid(WX):
    '''
        Sigmoid activition function
        WX : dot(W, X)
        X : Feature Array
        W : Weight Array
        Sigmoid:
          F(WX) = 1 / (1 + e^-WX)
    '''
    return 1 / (1 + np.exp(-WX))
    
def Tanh(WX):
    '''
        Tanh activition function
        WX : dot(W, X)
        X : Feature Array
        W : Weight Array
    '''
    return 2 * Sigmoid(2*WX) - 1

def ReLu(WX):
    '''
        The rectified linear unit
        WX : dot(W, X)
        X : Feature Array
        W : Weight Array
    '''
    return max(0, WX)

def LeaklyReLu(WX, α=0.01):
    '''
        The leakly rectified linear unit
        WX : dot(W, X)
        X : Feature Array
        W : Weight Array
    '''
    return max(α*WX, WX)

def ELU(WX, α=0.01):
    '''
        ELU activation
        WX : dot(W, X)
        X : Feature Array
        W : Weight Array
    '''
    return max(α*(math.exp(-WX) - 1), 0)