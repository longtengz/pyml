import sys
sys.path.insert(0, '../../../kNN')

from kNN import kNN

digitsDataPool = list()
digitsLabelPool = list()

testDigitsData = list()
testDigitsLabel = list()

trainingDataFile = '../data/reducedDigits.data'

with open(trainingDataFile, 'r') as digitsFile:
    for index, line in enumerate(list(digitsFile)):
            digitsLabelPool.append(int(line[0]))
            digitsDataPool.append([int(x) for x in line[2:-1]])


with open('../data/testDigits.data', 'r') as testDigitsFile:
    for index, line in enumerate(list(testDigitsFile)):
            testDigitsLabel.append(int(line[0]))
            testDigitsData.append([int(x) for x in line[2:-1]])



error = 0

for index, testData in enumerate(testDigitsData):
    kNN_result = kNN(testData, digitsDataPool, digitsLabelPool, 10)
    print('get', kNN_result, ', it is in fact', testDigitsLabel[index])
    if kNN_result != testDigitsLabel[index]:
        error += 1

print('Error rate: ', error / len(testDigitsLabel))
