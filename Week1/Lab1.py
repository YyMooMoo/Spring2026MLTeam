import mnist
import time
import random
import math
import numpy as np



def parseInputs():
    # Lists
    # Each image is a list of lists

    mndata = mnist.MNIST('samples')
    inputs, labels = mndata.load_training()

    labels = labels.tolist()
    expected = []
    for i in range(len(labels)):
        cur = [0 for j in range(10)]
        cur[labels[i]] = 1
        expected.append(cur)

    for i in range(len(inputs)):
        inputs[i].append(1)

    

    inputs = inputs[101:102]
    return inputs, expected





def transfer(value):
    if -value < 700:
        return 1 / (1 + math.e ** (-value))
    else:
        if value > 0: return 1
        return 0


def forwardProp(x, weights):
    for layer in range(len(x) - 1):
        nextLayer = []
        for node in x[layer + 1]: nextLayer.append(0.0)

        if layer != len(x) - 2:
            for i in range(len(x[layer])):
                for j in range(len(nextLayer)):
                    nextLayer[j] += float(x[layer][i] * weights[layer][i][j])
            
            for i in range(len(nextLayer)): nextLayer[i] = float(transfer(nextLayer[i]))

        else:
            nextLayer = [x[layer][i] * weights[layer][i][0] for i in range(len(weights[layer]))]

        for i in range(len(nextLayer)): x[layer + 1][i] = nextLayer[i]
    return x


def backProp(x, weights, expected, alpha):
    errors = []
    for i in x:
        layer = []
        for j in i:
            layer.append(0.0)
        errors.append(layer)

    gradients = []
    for i in weights:
        layer = []
        for j in i:
            nodeWeights = []
            for p in j:
                nodeWeights.append(0.0)
            layer.append(nodeWeights)
        gradients.append(layer)

    lastLayer = len(x) - 2
    for i in range(len(x[lastLayer])):
        errors[lastLayer][i] = weights[lastLayer][i][0] * (expected[i] - x[lastLayer + 1][i]) * x[lastLayer][i] * (1 - x[lastLayer][i])
        gradients[lastLayer][i][0] = (expected[i] - x[lastLayer + 1][i]) * alpha *  x[lastLayer][i]

    for layer in range(len(x) - 3, -1, -1):
        for i in range(len(x[layer])):
            sum = 0.0
            for j in range(len(errors[layer + 1])):
                nextError = errors[layer + 1][j]
                gradients[layer][i][j] = alpha * x[layer][i] * nextError
                sum += (weights[layer][i][j] * nextError)
            errors[layer][i] = sum * x[layer][i] * (1 - x[layer][i])

    weights = [[[weights[layer][i][j] + gradients[layer][i][j] for j in range(len(gradients[layer][i]))] for i in range(len(gradients[layer]))] for layer in range(len(gradients))]

    return weights


def runNeuralNet(inputs, expected):

    alpha = 0.01
    epochs = 50000
    error = 0.0

    layerSizes = [len(inputs[0]), 32, 10, 10]

    weights = []
    for i in range(len(layerSizes) - 2):
        layerWeights = []
        for j in range(layerSizes[i]):
            nodeWeights = []
            for k in range(layerSizes[i + 1]):
                weight = random.random()
                nodeWeights.append(weight)
            layerWeights.append(nodeWeights)
        weights.append(layerWeights)
    nodeWeights = []
    for i in range(layerSizes[-1]):
        weight = random.random()
        nodeWeights.append([weight])
    weights.append(nodeWeights)

    print(weights[-1])


    start = time.time()
    minError = 1000000
    correct = 0
    total = 0
    for epoch in range(epochs):
        print(epoch)
        sum = 0.0
        for i in range(len(inputs)):
            actual = expected[i].index(max(expected[i]))
            currInputs = inputs[i]
            x = []
            for j in range(len(layerSizes)):
                layer = []
                if j == 0:
                    layer = currInputs
                else:
                    for p in range(layerSizes[j]):
                        layer.append(0.0)
                x.append(layer)
            x = forwardProp(x, weights)
            weights = backProp(x, weights, expected[i], alpha)
            sum += abs(expected[i][actual] - x[-1][actual])
            if actual == x[-1].index(max(x[-1])):
                correct += 1
            total += 1
        error = 0.5 * (sum**2)
        #print(error)


        if epoch % 10 == 0:
            print(weights[-1])
            output(weights)
            print(correct, total)
            print(correct/total)
            print()
            correct = 0
            total = 0
                    

    output(weights)


def output(weights):
    file = open('weights.txt', 'w')
    for i in range(len(weights)):
        layer = weights[i]
        for length in range(len(layer[0])):
            line = ''
            for node in layer:
                line += str(node[length]) + ' '
            #print(line, end = '')
            file.write(line)
        if i < len(weights) - 1:
            file.write('\n')
        #print()
    file.close()
    print(time.time() - start)


start = time.time()
def main():
    inputs, expected = parseInputs()
    runNeuralNet(inputs, expected)





if __name__ == '__main__':
    main()

# Raymond Fu, pd 3, 2025

