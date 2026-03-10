from math import log
import numpy as np
import random
import matplotlib.pyplot as plt

#Activation Function and Derivative
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
    s = sigmoid(x)
    return s * (1 - s)


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
    a_prev = inp
    caches = [None]
    

    for i in range (1, len(weights)):
        
        z = weights[i]@a_prev+biases[i]
        a = A_vec(z)
        
        cache = (weights[i], a_prev, biases[i], z, a)
        a_prev = a
        caches.append(cache)
        
        
    return a, caches

#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(training, weights, biases):
    a, caches, y, alpha = training
    da = a-y
    for i in range(len(weights)-1, 0, -1):
        W, a_prev, b, z, a = caches[i]

        dz=sigmoidPrime(z)*da
        da = W.T@dz
        
        weights[i] = weights[i] - alpha*(dz@a_prev.T)
        biases[i] = biases[i] - alpha*dz

    return weights, biases

def networkFull(epoch, alpha):
    data = read_file('mnist_train.csv')
    test = read_file('mnist_test.csv')
    nodes = [784, 400, 200, 10]
    weights, biases = architecture(nodes)

    accarr = []
    epocharr = []

    for k in range(epoch):
        random.shuffle(data)
        for i in range(len(data)):
            x = data[i][0]/255
            y = data[i][1]
            

            a, caches = p_net(sigmoid, weights, biases, x)
            training = a, caches, y, alpha
            weights, biases = one_epoch(training, weights, biases)
            
        corr = 0
        for j in range(len(data)):
            x = data[j][0]/255
            y = data[j][1]
            prediction, _ = p_net(sigmoid, weights, biases, x)
            if (np.argmax(prediction) == np.argmax(y)):
                corr = corr+1
        acc = corr/len(data)
        accarr.append(acc)
        epocharr.append(k)


    correct = 0
    
    for m in range(len(test)):
        x = test[m][0]/255
        y = test[m][1]


        prediction, _ = p_net(sigmoid, weights, biases, x)

        
        

        if (np.argmax(prediction) == np.argmax(y)):
            correct = correct+1

    accuracy = correct/len(test)

    print("Final test accuracy: ", accuracy)
    plt.plot(epocharr, accarr)
    plt.xlabel("Epoch Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs on Training")
    plt.show()

networkFull(10, 0.05)

    

        




#TODO Run your model over some number of epochs should be at least 10 and display a graph that shows train and test accuracy on each Epoch



