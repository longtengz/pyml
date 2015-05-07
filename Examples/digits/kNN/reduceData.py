import sys
sys.path.insert(0, '../../../kNN')

from condensedNN import cNN

digitsDataPool = list()
digitsLabelPool = list()

trainingDataFile = '../data/trainingDigits.data'
reduceDataFile = '../data/reducedDigits.data'

with open(trainingDataFile, 'r') as digitsFile:
    for index, line in enumerate(list(digitsFile)):
            digitsLabelPool.append(int(line[0]))
            digitsDataPool.append([int(x) for x in line[2:-1]])


digitsDataPool, digitsLabelPool = cNN(digitsDataPool, digitsLabelPool)

with open(reduceDataFile, 'w+') as reduceDigitsFile:
    print(len(digitsDataPool))
    for index, digitsData in enumerate(digitsDataPool):
        reduceDigitsFile.write(str(digitsLabelPool[index]) + '-')
        reduceDigitsFile.write(''.join([str(x) for x in digitsData]))
        reduceDigitsFile.write('\n')
