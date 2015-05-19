import sys
# TODO
# need to normalize this path for Windows users
sys.path.insert(0, '../../../NeuralNet')

from NeuralNet import NN
from ActivationFunction.AF import *

trainingPairs = list()
testPairs = list()


with open('../data/trainingDigits.data', 'r') as trainingDigitsFile:
    for line in list(trainingDigitsFile):
        # make outputValue as a vector, if the output is number n, then we make the (n+1)th element in vector as 1, and the rest elements are of value 0
        outputValue = [0] * 9
        outputValue.insert(int(line[0]), 1)

        inputValue = [int(x) for x in list(line[2:-1])]

        trainingPairs.append([inputValue, outputValue])


with open('../data/testDigits.data', 'r') as testDigitsFile:
    for line in list(testDigitsFile):
        outputValue = [0] * 9
        outputValue.insert(int(line[0]), 1)

        inputValue = [int(x) for x in list(line[2:-1])]

        testPairs.append([inputValue, outputValue])


print('start')

#digitsClassifier = NN([1024, 30, 10], sigmoid, sigmoidDiff)
#digitsClassifier = NN([1024, 30, 10], sigmoid, sigmoidDiff, '../data/digitsWeights-1024-500-10.data')

# stochastic gradient descent
digitsClassifier = NN([1024, 30, 10], sigmoid, sigmoidDiff, '../data/digitsWeights-sgd-1024-500-10.data')

digitsClassifier.train(trainingPairs, 10, 0.05, isSGD=True)

digitsClassifier.test(testPairs)

#digitsClassifier.saveWeightsToFile('../data/digitsWeights-1024-500-10.data')

# SGD 
digitsClassifier.saveWeightsToFile('../data/digitsWeights-sgd-1024-500-10.data')
