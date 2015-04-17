import random 
import math

def rand(start, stop):
    return random.uniform(start, stop)


def makeMatrix(dimension, randValue=False, start=0, stop=1):
    matrix = list()

    for dim in dimension:
        if randValue == False:
            matrix.append([0 for x in range(dim)])
        if randValue == True:
            matrix.append([rand(start, stop) for x in range(dim)])

    return matrix


class NN():
    def __init__(self, arch, activationFunc, activationDiffFunc, weights=None):

        self.activationFunc = activationFunc
        self.activationDiffFunc = activationDiffFunc

        self.arch = arch

        self.weightsDimension = [(early+1) * self.arch[index+1] for index, early in enumerate(self.arch[:-1])]
        # add bias node's weights to the next layer(the next layer is without bias node)
        # bias node have no inputs and it only has output which is of value 1

        self.activationsDimension = [x+1 for x in self.arch[1:-1]]
        self.activationsDimension.append(self.arch[-1])
        # the output layer doesn't have a bias node, so it doesn't need to add one more activation

        if weights is None:
            self.weights = makeMatrix(self.weightsDimension, randValue=True, start=-1, stop=1)
            # start afresh, train examples to get new weights
        elif type(weights) is str:
            self.readWeightsFromFile(weights)
            # using saved weights
        else:
            raise TypeError('weights must be a string or None')

        self.derivatives = makeMatrix(self.weightsDimension)

    def forwardActivations(self, inputValue):
        self.activations = makeMatrix(self.activationsDimension)

        for index, weight in enumerate(self.weights):
            if index == 0:
                earlierNodes = [1]
                earlierNodes.extend(inputValue)
                # add bias Node in input layer
            else:
                earlierNodes = self.activations[index - 1]

            for n1 in range(self.activationsDimension[index]):
                if index != (len(self.arch) - 2) and n1 == 0:
                    self.activations[index][n1] = 1
                    # set the bias nodes' activation value to be 1
                else:
                    # for n2 in range(self.activationsDimension[index-1]):
                    for n2, earlierActivation in enumerate(earlierNodes):
                        if index == len(self.arch) - 2:
                            self.activations[index][n1] += earlierActivation * weight[n2 * self.arch[index+1] + n1]
                        else:
                            self.activations[index][n1] += earlierActivation * weight[n2 * self.arch[index+1] + n1 - 1]
                        # here we are using the model that the later nodes get weighted sum of earlier layers as input
                        # and the output of this node would be the sigmoid function of the input
                    self.activations[index][n1] = self.activationFunc(self.activations[index][n1])


    def backPropagate(self, inputValue, outputValue):
        deltas = makeMatrix(self.arch[1:])

        for index, dlts in enumerate(reversed(deltas)):
            for idx, dt in enumerate(dlts):
                if index == 0:
                    dlts[idx] = self.activations[-1][idx] - outputValue[idx]
                else:
                    for idex, earlier_dt in enumerate(deltas[-index]):
                        dlts[idx] += earlier_dt * self.weights[-index][(idx + 1) * self.arch[-index] + idex] * self.activationDiffFunc(self.activations[-index - 1][idx + 1])



        # compute derivatives using previously computed deltas
        for index, ders in enumerate(self.derivatives):
            for idx, der in enumerate(ders):
                if index == 0:
                    if idx < self.arch[1]:
                        ders[idx] += deltas[index][math.floor(idx % self.arch[1])] * 1
                    else:
                        ders[idx] += deltas[index][math.floor(idx % self.arch[1])] * inputValue[math.floor(idx / self.arch[1]) - 1]
                else:
                    ders[idx] += deltas[index][math.floor(idx % self.arch[index+1])] * self.activations[index-1][math.floor(idx / self.arch[index+1])]



    def updateWeights(self, learningRate, examplesNum):
        # using previously computed derivatives to update weights
        for index, weight in enumerate(self.weights):
            for idx, wt in enumerate(weight):
                weight[idx] -= learningRate * self.derivatives[index][idx] / examplesNum
                self.derivatives[index][idx] = 0
                # reset derivatives

    def train(self, trainingExamples, iterations, learningRate):
        examplesNum = len(trainingExamples)

        newError = 0
        oldError = 0

        for iteration in range(iterations):

            oldError = newError
            newError = 0.0

            for inputValue, outputValue in trainingExamples:
                self.forwardActivations(inputValue)
                # forward pass
                self.backPropagate(inputValue, outputValue)
                # backward pass
                for index, output in enumerate(outputValue):
                    newError += output * math.log(self.activations[-1][index]) + (1 - output) * math.log(1 - self.activations[-1][index])

            newError = newError / (-examplesNum)

            print(iteration, oldError - newError)

            if oldError - newError < 0 and iteration != 0:
                print('\n\n\nlearningRate is too big now')
                return

            self.updateWeights(learningRate, examplesNum)

    def test(self, inputOutputPairs):
        inputs, desiredOutputs = zip(*inputOutputPairs)

        inputs = list(inputs)
        desiredOutputs = list(desiredOutputs)

        outputs = self.classify(inputs)

        for output, desired in zip(outputs, desiredOutputs):
            print(output, desired)

    def classify(self, inputs):
        outputs = list()

        for ipt in inputs:
            self.forwardActivations(ipt)
            outputs.append(self.activations[-1])

        return outputs

    def saveWeightsToFile(self, fileName):
        weightsFile = open(fileName, 'w')

        for weight in self.weights:
            weightsFile.write(','.join([str(wt) for wt in weight]))
            weightsFile.write('\n')

        weightsFile.close()

    def readWeightsFromFile(self, fileName):
        weightsFile = open(fileName, 'r')

        self.weights = list()

        for index, line in enumerate(list(weightsFile)):
            savedWeights = line.split(',')
            if len(savedWeights) != self.weightsDimension[index]:
                print('weights dimension needs to be', self.weightsDimension)
                raise Exception("Wrong weights's dimension from weightsFile")
            else:
                savedWeights = [float(wt) for wt in savedWeights]
                self.weights.append(savedWeights)

        print(type(self.weights))

        weightsFile.close()
