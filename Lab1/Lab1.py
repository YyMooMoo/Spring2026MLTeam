from math import log
import numpy as np

#Activation Function and Derivative
def sigmoid(x): return 1 / (1 + np.e**(-1*x))
def sigmoidPrime(x): return np.e**(-1*x) / ((1 + np.e**(-1*x))**2) 


#Some useful comversion functions
def ListtoVector(new_list):
    length = len(new_list)
    vec = np.arange(length)
    for i,v in enumerate(new_list):
        vec[i] = v
    return vec.reshape(length, 1)

def VectortoList(new_vec):
    length = (new_vec.size)
    reshaped = new_vec.reshape(1, length)
    new_list = list()
    for v in reshaped[0]:
        new_list.append(v)
    return new_list

#A function that setups your initial network including randomized weights, input is the form of a list of the number of neurons in each layer. e.g [784, x, y, 10] where x and y are the number of neurons in your hiddenn layers
def architecture(new_list):
    weights = list()
    biases = list()
    weights.append(None)
    biases.append(None)
    network_length = len(new_list)
    for c in range(network_length-1):
        weight_matrix = 2 * np.random.rand(new_list[c+1], new_list[c]) - 1
        bias_matrix = 2 * np.random.rand(new_list[c+1],1) - 1
        weights.append(weight_matrix)
        biases.append(bias_matrix)
    return weights, biases

#take in the CSVs and vectorize the output, would reccomend experimenting with to see exactly what happens
def read_file(file_name):
    toReturn = list()
    with open(file_name) as f:
        for line in f:
            image = line[0:len(line)-1].split(",")
            output = image.pop(0)
            in_vec = ListtoVector(image)
            out_vec = list()
            for c in range(10):
                if c == int(output):
                    out_vec.append(1)
                else:
                    out_vec.append(0)
            out_vec = ListtoVector(out_vec)
            toAppend = (in_vec,out_vec)
            toReturn.append(toAppend)
    return toReturn

#TODO A feed forward of the network where A_vec is the activation function, weights is a list of all the weight matrices, biases is a list of all the bias vectors, and inp is the input, return the output as a vector
def p_net(A_vec, weights, biases, inp):
    activation = inp

    for layer in range(1, len(weights)):
        z = np.matmul(weights[layer], activation) + biases[layer]
        activation = A_vec(z)

    return activation

#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases):
    learning_rate = 0.1

    for inp, expected in training:
        inp = inp.astype(float) / 255.0
        expected = expected.astype(float)
        activations = [inp]
        z_values = [None]
        activation = inp
        
        # forward pass
        for layer in range(1, len(weights)):
            z = np.matmul(weights[layer], activation) + biases[layer]
            z_values.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        deltas = [None] * len(weights)

        # output layer change
        deltas[-1] = (activations[-1] - expected) * sigmoidPrime(z_values[-1])

        # hidden layer changes
        for layer in range(len(weights) - 2, 0, -1):
            deltas[layer] = np.matmul(weights[layer + 1].T, deltas[layer + 1]) * sigmoidPrime(z_values[layer])

        # gradient descent update
        for layer in range(1, len(weights)):
            weights[layer] = weights[layer] - learning_rate * np.matmul(deltas[layer], activations[layer - 1].T)
            biases[layer] = biases[layer] - learning_rate * deltas[layer]

    return weights, biases

#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch
import matplotlib.pyplot as plt

def get_accuracy(data, weights, biases):
    correct = 0

    for inp, expected in data:
        inp = inp.astype(float)/255.0
        output = p_net(sigmoid, weights, biases, inp)
        predicted_label = np.argmax(output)
        actual_label = np.argmax(expected)
        if predicted_label == actual_label:
            correct += 1
    return correct/len(data)

weights, biases = architecture([784, 64, 32, 10])

training_data = read_file("mnist_train.csv")
test_data = read_file("mnist_test.csv")

epochs = 10
train_accs = []
test_accs = []

for epoch in range(epochs):
    weights, biases = one_epoch(training_data, weights, biases)
    train_accuracy = get_accuracy(training_data, weights, biases)
    test_accuracy = get_accuracy(test_data, weights, biases)
    train_accs.append(train_accuracy)
    test_accs.append(test_accuracy)
    print("Epoch", epoch + 1, "Train Accuracy:", train_accuracy, "Test Accuracy:", test_accuracy)

plt.plot(range(1, epochs + 1), train_accs, label="Train Accuracy")
plt.plot(range(1, epochs + 1), test_accs, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")
plt.legend()
plt.show()







