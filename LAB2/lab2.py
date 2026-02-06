from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time
import os

def save_plot(filename=None):
    """Save the current plot to a file using its title as the filename"""
    if filename is None:
        # Get title from current axes
        title = plt.gca().get_title()
        if title:
            filename = title
        else:
            filename = "plot"
    
    # Sanitize filename - remove special characters
    filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in filename)
    filename = filename.replace(' ', '_') + '.png'
    
    plt.savefig(filename)
    plt.close()

def make_blob_data():
    # Generate blob dataset: 4 well-separated spherical clusters
    X, y = make_blobs(n_samples=400, centers=4, cluster_std=2, random_state=10)
    # Split into 60% training and 40% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)

    # Visualize the training data distribution
    plt.figure()
    #plot_decision_boundaries(X_train, y_train, MLPClassifier, hidden_layer_sizes=(10, 10))
    plt.title("Blob Training Data")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    save_plot()

    return X_train, X_test, y_train, y_test

def make_moon_data():
    # Generate moons dataset: two interleaving half circles (challenging for classifiers)
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    # Split into 60% training and 40% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)

    # Visualize the training data distribution
    plt.figure()
    #plot_decision_boundaries(X_train, y_train, MLPClassifier, hidden_layer_sizes=(10, 10))
    plt.title("Moon Training Data")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    save_plot()

    return X_train, X_test, y_train, y_test

def line_best_fit(x_data, y_data):
    # Calculate linear regression line (fit the data with a line)
    y = []
    w1, w0 = np.polyfit(x_data, y_data, 1)  # Get slope (w1) and intercept (w0)
    for i in x_data:
        y.append((w1*i) + w0)  # Compute the predicted y values from the line
    # Calculate how well the line fits (R-squared value)
    r_squared = round(r2_score(y_data, y), 3)
    plt.figtext(0.1, 0.01, "R^2 = " + str(r_squared), fontsize=10)  # Display R² score on plot
    plt.plot(x_data, y)  # Draw the regression line

# Trains and evaluates a neural network with customizable architecture
def run_MLP(X_train, X_test, y_train, y_test, hidden_layers_size=10, random_state=10,  momentum=.9, number_of_layers=1, batch_size=20):
    start = time.time()
    # Create MLP with specified number of hidden layers, each with same size
    clf = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers_size for _ in range(number_of_layers)), random_state=random_state, solver='sgd', momentum = momentum, batch_size=batch_size, max_iter=5000)
    clf.fit(X_train, y_train)  # Train the model
    np.set_printoptions(suppress=True, precision=2)
    end = time.time()
    # Calculate training time in milliseconds
    runtime = ((end-start)*1000)
    # Get accuracy score on test set (0.0 to 1.0)
    score = clf.score(X_test, y_test)
    # Returns list: [model, train_X, test_X, train_y, test_y, runtime_ms, accuracy_score]
    # Important: Index [5] = runtime, Index [6] = accuracy score
    return [clf, X_train, X_test, y_train, y_test, runtime, score]

# Visualizes how the trained model classifies the feature space (decision boundaries)
def plot_decision_boundaries(X, y, model_class, **model_params):
    model = model_class(**model_params, random_state=10)
    model.fit(X, y)
    # Set boundaries slightly beyond the actual data range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Create a fine grid step size (smaller = more detailed boundary)
    h = .01 * np.mean([x_max - x_min, y_max - y_min])
    # Create mesh grid covering the entire feature space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Get model predictions for every point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Draw filled contours showing classification regions
    plt.contourf(xx, yy, Z, alpha=0.3)
    # Overlay actual data points with their true class labels
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)

# Test how momentum affects training speed
def plot_mom_vs_runtime(X_train, X_test, y_train, y_test, type):
    x = []  # Momentum values
    y = []  # Runtime values
    # Try momentum from 0 to 0.99 in increments of 0.01
    for i in np.arange(0, 1, 0.01):
        x.append(i)
        data = run_MLP(X_train, X_test, y_train, y_test, momentum=i)
        y.append(data[5])  # Extract runtime (index 5)

    plt.figure()
    line_best_fit(x, y)
    plt.scatter(x, y)
    plt.title("Momentum vs. Runtime for " + type + " Data")
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Momentum Value")
    save_plot()

# Test how momentum affects model accuracy
def plot_mom_vs_accuracy(X_train, X_test, y_train, y_test, type):
    x = []  # Momentum values
    y = []  # Accuracy values
    for i in np.arange(0, 1, 0.01):
        x.append(i)
        data = run_MLP(X_train, X_test, y_train, y_test, momentum=i)
        y.append(data[6])  # Extract accuracy score (index 6)

    plt.figure()
    plt.scatter(x, y)
    plt.title("Momentum vs. Accuracy Score for " + type + " Data")
    plt.ylabel("Score (0–100%)")
    plt.xlabel("Momentum Value")
    save_plot()

# Test how the size of hidden layers affects training speed
def plot_layer_size_vs_runtime(X_train, X_test, y_train, y_test, type):
    x = []  # Hidden layer sizes
    y = []  # Runtime values
    # Try layer sizes from 1 to 99 neurons
    for i in np.arange(1, 100, 1):
        x.append(i)
        data = run_MLP(X_train, X_test, y_train, y_test, hidden_layers_size=i)
        y.append(data[5])  # Extract runtime (index 5)

    plt.figure()
    line_best_fit(x, y)
    plt.scatter(x, y)
    plt.title("Hidden Layer Size vs. Runtime for " + type + " Data")
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Hidden Layer Size")
    save_plot()

# Test how the size of hidden layers affects model accuracy (momentum fixed at 0.9)
def plot_layer_size_vs_accuracy(X_train, X_test, y_train, y_test, type):
    x = []  # Hidden layer sizes
    y = []  # Accuracy values
    for i in np.arange(1, 100, 1):
        x.append(i)
        # Parameters: hidden_size=i, random_state=10, momentum=.9
        data = run_MLP(X_train, X_test, y_train, y_test, i, 10, .9,)
        y.append(data[6])  # Extract accuracy score (index 6)

    plt.figure()
    plt.scatter(x, y)
    plt.title("Hidden Layer Size vs. Accuracy Score for " + type + " Data")
    plt.ylabel("Score (0–100%)")
    plt.xlabel("Hidden Layer Size")
    save_plot()

# Test how the number of hidden layers affects model accuracy
def plot_number_of_layers_vs_accuracy(X_train, X_test, y_train, y_test, type):
    x = []  # Number of hidden layers
    y = []  # Accuracy values
    # Try networks with 1 to 9 hidden layers
    for i in np.arange(1, 10, 1):
        x.append(i)
        data = run_MLP(X_train, X_test, y_train, y_test, number_of_layers=i)
        y.append(data[6])  # Extract accuracy score (index 6)

    plt.figure()
    plt.scatter(x, y)
    plt.title("Number of Layers vs. Accuracy Score for " + type + " Data")
    plt.ylabel("Score (0–100%)")
    plt.xlabel("Number of layers")
    save_plot()



# Test how the number of hidden layers affects training speed
def plot_number_of_layers_vs_runtime(X_train, X_test, y_train, y_test, type):
    x = []  # Number of hidden layers
    y = []  # Runtime values
    for i in np.arange(1, 10, 1):
        x.append(i)
        data = run_MLP(X_train, X_test, y_train, y_test, number_of_layers=i)
        y.append(data[5])  # Extract runtime (index 5)

    plt.figure()
    line_best_fit(x, y)
    plt.scatter(x, y)
    plt.title("Number of layers vs. Runtime for " + type + " Data")
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Number of layers")
    save_plot()


# Test how batch size affects model accuracy
def plot_batch_size_vs_accuracy(X_train, X_test, y_train, y_test, type):
    x = []  # Batch sizes
    y = []  # Accuracy values
    # Try batch sizes from 1 to training set size, in increments of 10
    for i in np.arange(1, len(X_train),10):
        x.append(i)
        data = run_MLP(X_train, X_test, y_train, y_test, batch_size=i)
        y.append(data[6])  # Extract accuracy score (index 6)

    plt.figure()
    plt.scatter(x, y)
    plt.title("Batch Size vs. Accuracy Score for " + type + " Data")
    plt.ylabel("Score (0–100%)")
    plt.xlabel("Batch Size")
    save_plot()



# Test how batch size affects training speed
def plot_batch_size_vs_runtime(X_train, X_test, y_train, y_test, type):
    x = []  # Batch sizes
    y = []  # Runtime values
    # Try batch sizes from 1 to training set size, in increments of 10
    for i in np.arange(1, len(X_train), 10):
        x.append(i)
        data = run_MLP(X_train, X_test, y_train, y_test, batch_size=i)
        y.append(data[5])  # Extract runtime (index 5)

    plt.figure()
    line_best_fit(x, y)
    plt.scatter(x, y)
    plt.title("Batch Size vs. Runtime for " + type + " Data")
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Batch Size")
    save_plot()

# Generate datasets with train/test split
X_train_moon, X_test_moon, y_train_moon, y_test_moon = make_moon_data()
X_train_blob, X_test_blob, y_train_blob, y_test_blob = make_blob_data()

# Run experiments on Blob dataset
# Test momentum parameter
plot_mom_vs_runtime(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")
plot_mom_vs_accuracy(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")

# Test hidden layer size
plot_layer_size_vs_accuracy(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")
plot_layer_size_vs_runtime(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")

# Test number of hidden layers
plot_number_of_layers_vs_accuracy(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")
plot_number_of_layers_vs_runtime(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")

# Test batch size
plot_batch_size_vs_accuracy(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")
plot_batch_size_vs_runtime(X_train_blob, X_test_blob, y_train_blob, y_test_blob, "Blob")

# Run same experiments on Moon dataset
# Test momentum parameter
plot_mom_vs_runtime(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")
plot_mom_vs_accuracy(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")

plot_layer_size_vs_accuracy(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")
plot_layer_size_vs_runtime(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")

plot_number_of_layers_vs_accuracy(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")
plot_number_of_layers_vs_runtime(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")

plot_batch_size_vs_accuracy(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")
plot_batch_size_vs_runtime(X_train_moon, X_test_moon, y_train_moon, y_test_moon, "Moon")