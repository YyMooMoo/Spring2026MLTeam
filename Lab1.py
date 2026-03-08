import numpy as np
import matplotlib.pyplot as plt
import random

#Activation Function and Derivative
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x): 
    return np.exp(-x) / ((1 + np.exp(-x))**2)


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
        weight_matrix = (2 * np.random.rand(new_list[c+1], new_list[c]) - 1) / np.sqrt(new_list[c])
        bias_matrix = (2 * np.random.rand(new_list[c+1],1) - 1) / np.sqrt(new_list[c])
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
            #in_vec = ListtoVector(image)
            in_vec = np.array(image, dtype=float).reshape(784, 1) / 255.0
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

def eval(dataset, weights, biases):
    correct_predictions = 0
    for inp, label in dataset:
        a, _ = p_net(weights, biases, inp)
        prediction = np.argmax(a[-1])
        actual = np.argmax(label)
        if prediction == actual:
            correct_predictions += 1
    # Return accuracy as %
    return (correct_predictions / len(dataset)) * 100

def MSE(v1, v2):
    c=0
    for i in range(len(v1)):
        c+=(v1[i]-v2[i])**2
    return c/2

#TODO A feed forward of the network where A_vec is the activation function, weights is a list of all the weight matrices, biases is a list of all the bias vectors, and inp is the input, return the output as a vector
def p_net(weights: list[float], biases: list[float], inp: np.ndarray):
    a=[inp]
    z=[]
    for l in range(1,len(weights)):
        z.append(np.dot(weights[l],a[l-1])+biases[l])
        a.append(sigmoid(z[l-1]))
        
    return a, z

#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases, LR=0.1):
    for inp, label in training:
        output, z = p_net(weights, biases, inp)
        L = len(weights) - 1  # number of layers

        d = [None] * (L + 1)
        d[L] = (output[-1] - label) * sigmoidPrime(z[-1])
        for l in range(L - 1, 0, -1):
            d[l] = np.dot(weights[l + 1].T, d[l + 1]) * sigmoidPrime(z[l - 1])

        # Update weights and biases
        for l in range(1, L + 1):
            weights[l] -= LR * np.dot(d[l], output[l - 1].T)
            biases[l]  -= LR * d[l]

    return weights, biases

#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch
if __name__ == "__main__":
    training = read_file("mnist_train.csv")
    testing = read_file("mnist_test.csv")
    weights, biases = architecture([784, 400, 200, 10])
    
    epochs=10
    train_accs=[]
    test_accs=[]
    LR=0.5
    
    print(f"====== epochs: {epochs} | learning rate: {LR} ======")
    
    for epoch in range(epochs):
        weights, biases = one_epoch(training, weights, biases, LR)
        
        train_acc = eval(training, weights, biases)
        test_acc = eval(testing, weights, biases)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_accs, label="Training Accuracy", marker='o')
    plt.plot(range(1, epochs + 1), test_accs, label="Testing Accuracy", marker='s')
    
    plt.title("Neural Network Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.grid(True)
    plt.show()
    





