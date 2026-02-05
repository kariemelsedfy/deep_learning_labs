import numpy
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import math
#creating & plotting training data
# test with more spread out blobs via adding cluster_std=2 in the function
EPOCHS = 100
N, D = 500, 2
X, Y = make_blobs(n_samples=N, centers=2, n_features=D, random_state=1)
Y = 2*(Y - 0.5)

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c="b", s=50)
plt.gca().set_aspect('equal', adjustable='box')

#initial weights
w0 = 1
w1 = 1
w2 = 1



#Activation function
def activation_function(weighted_sum_of_inputs):
    if weighted_sum_of_inputs >= 0:
        return 1
    else:
        return -1


#calculates weights given 2 initial weights, bias, and X, Y data.
def calculate_weights(weight_0, weight_1, weight_2, x_data, classifications, EPOCHS):

    for epoch in range(EPOCHS):
        for n, i in enumerate(x_data):
            prediction = activation_function(weight_1 * i[0] + weight_2 * i[1] + weight_0)

            if prediction != classifications[n]:
                if classifications[n] == 1:
                    weight_1 = weight_1 + i[0]
                    weight_2 = weight_2 + i[1]
                    weight_0 = weight_0 + 1
                if classifications[n] == -1:
                    weight_1 = weight_1 - i[0]
                    weight_2 = weight_2 - i[1]
                    weight_0 = weight_0 - 1

    return weight_0, weight_1, weight_2

def plot_function(weight_0, weight_1, weight_2, label):
    x1 = numpy.linspace(-15, 5, 100)
    x2 = (-weight_1*x1 - weight_0)/weight_2
    plt.plot(x1, x2, label=label)


#before plot just for reference
plot_function(w0, w1, w2, "before algorithm")


#runs function to get new weights
w0, w1, w2 = calculate_weights(w0, w1, w2, X, Y, EPOCHS)


#Plot function with new weights
plot_function(w0, w1, w2, "after algorithm")


plt.legend()
plt.show()
