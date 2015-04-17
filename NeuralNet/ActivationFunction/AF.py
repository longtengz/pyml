import math

def sigmoid(x, alpha=1):
    return 1 / (1 + math.exp(-x * alpha))

def sigmoidDiff(y):
    return (1-y)*y

def tan(x):
    return math.tanh(x)

def tanDiff(y):
    return 1 - y**2
