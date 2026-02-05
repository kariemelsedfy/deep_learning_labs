from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time

def make_blob_data():
    X, y = make_blobs(n_samples=400, centers=4, cluster_std=2, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)

    data = run_MLP(X_train, X_test, y_train, y_test, 10, 10, 1)
    X_train = data[1]
    y_train = data[3]

    plt.figure()
    #plot_decision_boundaries(X_train, y_train, MLPClassifier, hidden_layer_sizes=(10, 10))
    plt.title("Blob Training Data")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.show()

    return X_train, X_test, y_train, y_test

def make_moon_data():
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)

    data = run_MLP(X_train, X_test, y_train, y_test, 10, 10, 1)
    X_train = data[1]
    y_train = data[3]

    plt.figure()
    #plot_decision_boundaries(X_train, y_train, MLPClassifier, hidden_layer_sizes=(10, 10))
    plt.title("Moon Training Data")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.show()

    return X_train, X_test, y_train, y_test

def line_best_fit(x_data, y_data):
    y = []
    w1, w0 = np.polyfit(x_data, y_data, 1)
    for i in x_data:
        y.append((w1*i) + w0)
    r_squared = round(r2_score(y_data, y), 3)
    plt.figtext(0.1, 0.01, "R^2 = " + str(r_squared), fontsize=10)
    plt.plot(x_data, y)

# runs basic MLP,
def run_MLP(X_train, X_test, y_train, y_test, hidden_layers, random_state, momentum):
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=random_state, solver='sgd', momentum = momentum, max_iter=5000)
    clf.fit(X_train, y_train)
    np.set_printoptions(suppress=True, precision=2)
    end = time.time()
    runtime = ((start-end)*1000)
    # visualizer
    #print(clf.predict_proba(X_test[0:3, :]))
    #print(clf.predict(X_test[0:3, :]))
    #print(y_test[0:3])
    score = clf.score(X_test, y_test)
    return [clf, X_train, X_test, y_train, y_test, runtime, score]

# decision boundary function
def plot_decision_boundaries(X, y, model_class, **model_params):
    model = model_class(**model_params, random_state=10)
    model.fit(X, y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = .01 * np.mean([x_max - x_min, y_max - y_min])
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)

def plot_mom_vs_runtime(X_train, X_test, y_train, y_test, type):
    x = []
    y = []
    for i in np.arange(0, 1, 0.01):
        x.append(i)
        start = time.time()
        run_MLP(X_train, X_test, y_train, y_test, 10, 10, i)
        end = time.time()
        y.append((end-start)*1000)

    plt.figure()
    line_best_fit(x, y)
    plt.scatter(x, y)
    plt.title("Momentum vs. Runtime for " + type + " Data")
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Momentum Value")
    plt.show()

def plot_mom_vs_accuracy(X_train, X_test, y_train, y_test, type):
    x = []
    y = []
    for i in np.arange(0, 1, 0.01):
        x.append(i)
        data = run_MLP(X_train, X_test, y_train, y_test, 10, 10, i)
        y.append(data[0].score(data[1], data[3]))

    plt.figure()
    plt.scatter(x, y)
    plt.title("Momentum vs. Accuracy Score for " + type + " Data")
    plt.ylabel("Score (0–100%)")
    plt.xlabel("Momentum Value")
    plt.show()

def plot_layer_size_vs_runtime(X_train, X_test, y_train, y_test, type):
    x = []
    y = []
    for i in np.arange(1, 100, 1):
        x.append(i)
        start = time.time()
        run_MLP(X_train, X_test, y_train, y_test, i, 10, .9,)
        end = time.time()
        y.append((end - start) * 1000)

    plt.figure()
    line_best_fit(x, y)
    plt.scatter(x, y)
    plt.title("Hidden Layer Size vs. Runtime for " + type + " Data")
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Hidden Layer Size")
    plt.show()

#momentum set to .9
def plot_layer_size_vs_accuracy(X_train, X_test, y_train, y_test, type):
    x = []
    y = []
    for i in np.arange(1, 100, 1):
        x.append(i)
        data = run_MLP(X_train, X_test, y_train, y_test, i, 10, .9,)
        y.append(data[0].score(data[1], data[3]))

    plt.figure()
    plt.scatter(x, y)
    plt.title("Hidden Layer Size vs. Accuracy Score for " + type + " Data")
    plt.ylabel("Score (0–100%)")
    plt.xlabel("Hidden Layer Size")
    plt.show()

X_train_moon, X_test_moon, y_train_moon, y_test_moon = make_moon_data()
print(0)
X_train_blob, X_test_blob, y_train_blob, y_test_blob = make_blob_data()
print(1)
plot_mom_vs_runtime(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")
print(2)
plot_mom_vs_accuracy(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")
print(3)
plot_layer_size_vs_accuracy(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")
print(4)
plot_layer_size_vs_runtime(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")
print(6)
plot_mom_vs_runtime(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")
print(7)
plot_mom_vs_accuracy(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")
print(8)
plot_layer_size_vs_accuracy(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")
print(9)
plot_layer_size_vs_runtime(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")
print("done")
#plt.show()