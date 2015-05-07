import random

import sys
sys.path.insert(0, '../../../NeuralNet')


from NeuralNet import NN
from ActivationFunction.AF import *


# wine data inputOutputPairs construct inputOutputPairs

inputOutputPairs = list()

wineDataFile = open('../data/wine.data', 'r')

for line in list(wineDataFile):
    if line[0] == '1':
        output = [1, 0, 0]
    elif line[0] == '2':
        output = [0, 1, 0]
    elif line[0] == '3':
        output = [0, 0, 1]

    inputValue = line[2:].split(',')
    
    inputValue = [float(ipt) for ipt in inputValue]

    # feature scaling
    inputValue[0] = inputValue[0] / 13.5
    inputValue[1] = inputValue[1] / 1.7
    inputValue[2] = inputValue[2] / 2.3
    inputValue[3] = inputValue[3] / 13
    inputValue[4] = inputValue[4] / 110
    inputValue[5] = inputValue[5] / 2.7
    inputValue[6] = inputValue[6] / 3
    inputValue[7] = inputValue[7] / .2
    inputValue[8] = inputValue[8] / 2
    inputValue[9] = inputValue[9] / 5
    inputValue[10] = inputValue[10] / 1 
    inputValue[11] = inputValue[11] / 3.5
    inputValue[12] = inputValue[12] / 1100

    inputOutputPairs.append([inputValue, output])

wineDataFile.close()

testPairs = list()

for x in range(20):
    testPairs.append(inputOutputPairs.pop(random.randrange(len(inputOutputPairs))))


""" start training """

"""wine data"""

wineClassifier = NN([13, 20, 20, 10, 3], sigmoid, sigmoidDiff, '../data/wineWeights-13-20-20-10-3.data')

#wineClassifier.train(inputOutputPairs, 1000, 0.05)

wineClassifier.test(testPairs)

#wineClassifier.saveWeightsToFile('../data/wineWeights-13-20-20-10-3.data')

"""xor data"""

#xorPairs = [
#    [[0, 0], [0]],
#    [[0, 1], [1]],
#    [[1, 0], [1]],
#    [[1, 1], [0]],
#]

#xor = NN([2, 2, 1], tan, tanDiff)
#xor = NN([2, 3, 3, 1], sigmoid, sigmoidDiff)

#xor.train(xorPairs, 5000, 0.1)

#xor.test(xorPairs)

