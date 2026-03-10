from math import log
import numpy as np
import matplotlib.pyplot as plt

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
    a = inp
    for i in range(1, len(weights)):
        z = weights[i] @ a + biases[i]
        a = A_vec(z)
    return a

#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases):
    learning_rate = 0.25
    for inp, target in training:
        activations = [inp]
        zs = []
        a = inp
        for i in range(1, len(weights)):
            z = weights[i] @ a + biases[i]
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        delta = (activations[-1] - target) * sigmoidPrime(zs[-1])
        biases[-1] -= learning_rate * delta
        weights[-1] -= learning_rate * delta @ activations[-2].T
        for l in range(2, len(weights)):
            z = zs[-l]
            sp = sigmoidPrime(z)
            delta = (weights[-l+1].T @ delta) * sp
            biases[-l] -= learning_rate * delta
            weights[-l] -= learning_rate * delta @ activations[-l-1].T
    return weights, biases

#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch

training = read_file('mnist_train.csv')
test = read_file('mnist_test.csv')

# Normalize inputs
for i in range(len(training)):
    training[i] = (training[i][0] / 255.0, training[i][1])
for i in range(len(test)):
    test[i] = (test[i][0] / 255.0, test[i][1])

network = [784, 30, 10]
weights, biases = architecture(network)

epochs = 10
train_accs = []
test_accs = []

for epoch in range(epochs):
    weights, biases = one_epoch(training, weights, biases)
    
    # Compute train accuracy
    correct = 0
    for inp, target in training:
        output = p_net(sigmoid, weights, biases, inp)
        if np.argmax(output) == np.argmax(target):
            correct += 1
    train_acc = correct / len(training)
    train_accs.append(train_acc)
    
    # Compute test accuracy
    correct = 0
    for inp, target in test:
        output = p_net(sigmoid, weights, biases, inp)
        if np.argmax(output) == np.argmax(target):
            correct += 1
    test_acc = correct / len(test)
    test_accs.append(test_acc)
    
    print(f'Epoch {epoch+1}: Train acc {train_acc:.4f}, Test acc {test_acc:.4f}')

plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy')
plt.plot(range(1, epochs+1), test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()






