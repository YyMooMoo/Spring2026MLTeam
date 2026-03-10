from math import log
import numpy as np
import matplotlib.pyplot as plt


# Activation Function and Derivative
def sigmoid(x): return 1 / (1 + np.e**(-1*x))
def sigmoidPrime(x): return np.e**(-1*x) / ((1 + np.e**(-1*x))**2) 


# Some useful conversion functions
def ListtoVector(new_list):
    return np.array(new_list, dtype=float).reshape(len(new_list), 1)

def VectortoList(new_vec):
    length = new_vec.size
    reshaped = new_vec.reshape(1, length)
    new_list = []
    for v in reshaped[0]:
        new_list.append(v)
    return new_list


# A function that sets up your initial network including randomized weights
#input is the form of a list of the number of neurons in each layer.
# e.g [784, x, y, 10] where x and y are the number of neurons in your hidden layers
def architecture(new_list):
    weights = list()
    biases = list()
    weights.append(None)
    biases.append(None)
    network_length = len(new_list)

    for c in range(network_length - 1):
        weight_matrix = 2 * np.random.rand(new_list[c + 1], new_list[c]) - 1
        bias_matrix = 2 * np.random.rand(new_list[c + 1], 1) - 1
        weights.append(weight_matrix)
        biases.append(bias_matrix)

    return weights, biases


# Take in the CSVs and vectorize the output
def read_file(file_name):
    toReturn = list()
    with open(file_name) as f:
        for line in f:
            image = line.strip().split(",")
            output = image.pop(0)

            # Normalize pixel values to [0,1]
            in_vec = ListtoVector([float(v) / 255.0 for v in image])

            out_vec = list()
            for c in range(10):
                if c == int(output):
                    out_vec.append(1)
                else:
                    out_vec.append(0)

            out_vec = ListtoVector(out_vec)
            toAppend = (in_vec, out_vec)
            toReturn.append(toAppend)

    return toReturn


# Feed forward of the network
def p_net(A_vec, weights, biases, inp):
    a = inp
    for i in range(1, len(weights)):
        z = np.dot(weights[i], a) + biases[i]
        a = A_vec(z)
    return a


# Forward pass that stores all z's and activations for backprop
def forward_full(weights, biases, inp):
    activations = [inp]
    zs = [None]
    a = inp

    for i in range(1, len(weights)):
        z = np.dot(weights[i], a) + biases[i]
        a = sigmoid(z)
        zs.append(z)
        activations.append(a)

    return zs, activations


# Backpropagation for one epoch
def one_epoch(training, weights, biases, lr=0.1):
    for inp, target in training:
        zs, activations = forward_full(weights, biases, inp)

        L = len(weights) - 1
        deltas = [None] * (L + 1)

        # Output layer delta
        deltas[L] = (activations[L] - target) * sigmoidPrime(zs[L])

        # Hidden layer deltas
        for l in range(L - 1, 0, -1):
            deltas[l] = np.dot(weights[l + 1].T, deltas[l + 1]) * sigmoidPrime(zs[l])

        # Update weights and biases
        for l in range(1, L + 1):
            weights[l] = weights[l] - lr * np.dot(deltas[l], activations[l - 1].T)
            biases[l] = biases[l] - lr * deltas[l]

    return weights, biases


# Accuracy function
def accuracy(data, weights, biases):
    correct = 0
    total = len(data)

    for inp, target in data:
        output = p_net(sigmoid, weights, biases, inp)
        predicted_label = np.argmax(output)
        true_label = np.argmax(target)

        if predicted_label == true_label:
            correct += 1

    return correct / total


if __name__ == "__main__":
    train_data = read_file("mnist_train.csv")
    test_data = read_file("mnist_test.csv")

    train_data = train_data[:5000]
    test_data = test_data[:5000]

    weights, biases = architecture([784, 64, 32, 10])

    epochs = 20
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        weights, biases = one_epoch(train_data, weights, biases, lr=0.1)

        train_acc = accuracy(train_data, weights, biases)
        test_acc = accuracy(test_data, weights, biases)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1}: Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")

    plt.plot(range(1, epochs + 1), train_accs, label="Train Accuracy")
    plt.plot(range(1, epochs + 1), test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.show()
