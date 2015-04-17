# Condensed Nearest Neighbors
# the Hart algorithm
# used for Data Reduction, essentially add speed and save memory space used by kNN

# Given a set of training data X, cNN works iteratively:
# 1. Scan all data points of X, looking for a data point x whose nearest prototype from STORE has a different label than x
# 2. Remove x from X and add it to STORE
# 3. Repeat the scan until no more prototypes are added to STORE
#
# Use STORE instead of X for kNN

from kNN import kNN

def cNN(dataPool, labelPool):
    prototypePool = list()
    protoLabelPool = list()

    # put the first data points in the STORE
    prototypePool.append(dataPool.pop())
    protoLabelPool.append(labelPool.pop())

    for index, data in enumerate(dataPool):
        dataLabel = labelPool[index]

        nearestPrototypeLabel = kNN(data, prototypePool, protoLabelPool, 1)

        if nearestPrototypeLabel != dataLabel:
            prototypePool.append(dataPool[index])
            protoLabelPool.append(labelPool[index])

    return prototypePool, protoLabelPool
