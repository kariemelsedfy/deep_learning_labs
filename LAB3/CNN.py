from FMNISTDatasetCNN import FMNISTDataset
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchsummary import summary
from torch.optim import Adam
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loads data, calling the FMNISTDatasetCNN class
fmnist_train = datasets.FashionMNIST('~/data/FMNIST', download=True, train=True)
fmnist_test = datasets.FashionMNIST('~/data/FMNIST', download=True, train=False)
x_train, y_train = fmnist_train.data, fmnist_train.targets
x_test, y_test = fmnist_test.data, fmnist_test.targets

# separates data into training and test data
train_dataset = FMNISTDataset(x_train, y_train)
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = FMNISTDataset(x_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)

# helper function for run_CNN, determines loss
def train_batch(x, y, model, opt, loss_fn):
    model.train()
    opt.zero_grad() # Flush memory
    batch_loss = loss_fn(model(x), y) # Compute loss
    batch_loss.backward() # Compute gradients
    opt.step() # Make a GD step
    return batch_loss.detach().cpu().numpy()

# helper function for run_CNN, helps determine accuracy
@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    argmaxes = prediction.argmax(dim=1)
    s = torch.sum((argmaxes == y).float())/len(y)
    return s.cpu().numpy()

# main function that takes in a sequential model and n_epochs, computes losses and prints test accuracies
def run_CNN(model, n_epochs):
    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-3) # presets
    losses, accuracies = [], []

    # iterates once for each epoch, computing and updating weights
    for epoch in range(n_epochs):
        print(f"Running epoch {epoch + 1} of {n_epochs}")
        epoch_losses, epoch_accuracies = [], []

        # where the updates actually occur
        for batch in train_dl:
            x, y = batch
            batch_loss = train_batch(x, y, model, opt, loss_fn) # helper function which updates the model
            epoch_losses.append(batch_loss)

        epoch_loss = np.mean(epoch_losses)

        # computes training accuracies
        for batch in train_dl:
            x, y = batch
            batch_acc = accuracy(x, y, model)
            epoch_accuracies.append(batch_acc)

        epoch_accuracy = np.mean(epoch_accuracies)
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

    # computes test accuracies
    epoch_accuracies = []
    for batch in test_dl:
        x, y = batch
        batch_acc = accuracy(x, y, model)
        epoch_accuracies.append(batch_acc)
    print(f"Test accuracy: {np.mean(epoch_accuracies)}")

    return losses, accuracies, np.mean(epoch_accuracies)

# loss and accuracy plots for MLP comparison, batch norm and dropout
def plot(losses, accuracies, n_epochs):
    plt.figure(figsize=(13, 3))
    plt.subplot(121)
    plt.title('Training Loss value over epochs')
    plt.plot(np.arange(n_epochs) + 1, losses)
    plt.subplot(122)
    plt.title('Testing Accuracy value over epochs')
    plt.plot(np.arange(n_epochs) + 1, accuracies)
    plt.show()

# basic plot for all other experiments
def plot_general(x, y, title, x_name, y_name):
    plt.figure()
    plt.title(title)
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.scatter(x, y)
    plt.show()

# helper function for comparing how weights increase with changing parameters
def compute_weights(params):
    weights = 0
    for tensor in params:
        weights += tensor.numel()
    return weights

# kernel size varying experiment
def vary_kernel():
    times = []
    performances = []
    range = [2, 3, 5, 7, 9]

    for i in range:
        start = time.time()
        weights = []
        kernels = i
        out_channels = 10
        linear_size = out_channels * ((28 - kernels + 1) // 2)**2 # computes input size for linear input

        # basic sequential for experimentation
        kernel_model = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=kernels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(int(linear_size), 10)
        ).to(device)

        weights.append(compute_weights(list(filter_model.parameters())))

        _, _, performance = run_CNN(kernel_model, 5) # run CNN
        performances.append(performance)

        end = time.time()
        times.append((end - start))

    # plots data
    plot_general(range, performances, "Kernel Size vs. Accuracy", "Kernel Size", "Accuracy")
    plot_general(range, times, "Kernel Size vs. Runtime", "Kernel Size", "Runtime (s)")
    plot_general(range, weights, "Kernel Size vs. Weights", "Kernel Size", "Weights")

def vary_filters():
    times = []
    weights = []
    performances = []
    range = [5, 10, 15, 20, 25]
    for i in range:
        start = time.time()

        kernels = 1
        out_channels = i
        linear_size = i * ((28 - kernels + 1) // 2)**2 # computes input size for linear input

        # basic sequential for experimentation
        filter_model = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=kernels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(int(linear_size), 10)
        ).to(device)

        weights.append(compute_weights(list(filter_model.parameters())))

        _, _, performance = run_CNN(filter_model, 5) # run CNN
        performances.append(performance)

        end = time.time()
        times.append((end - start))

    # plots data
    plot_general(range, performances, "FilterSize vs. Accuracy", "Filter Size", "Accuracy")
    plot_general(range, times, "Filter Size vs. Runtime", "Filter Size", "Runtime (s)")
    plot_general(range, weights, "Filter Size vs. Weights", "Filter Size", "Weights")

# choose 3 kernels and 15 out channels (previous experiments showed this maximises accuracy)
def run_batch_norm():
    kernels = 3
    out_channels = 15
    linear_size = out_channels * ((28 - kernels + 1) // 2)**2 # computes input size for linear input

    # basic sequential for experimentation
    kernel_model = nn.Sequential(
        nn.Conv2d(1, out_channels, kernel_size=kernels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(int(linear_size), 10)
    ).to(device)

    losses, accuracies, _ = run_CNN(kernel_model, 5) # run CNN

    # plots data
    plot(losses, accuracies, 5)

# choose p = 0.8 as this is what the article uses for input units
# choose 3 kernels and 15 out channels (previous experiments showed this maximises accuracy)
def run_dropout():
    kernels = 3
    out_channels = 15
    linear_size = out_channels * ((28 - kernels + 1) // 2)**2 # computes input size for linear input

    # basic sequential for experimentation
    kernel_model = nn.Sequential(
        nn.Conv2d(1, out_channels, kernel_size=kernels),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(p=0.8), # same as article
        nn.Flatten(),
        nn.Linear(int(linear_size), 10)
    ).to(device)

    losses, accuracies, _ = run_CNN(kernel_model, 5) # run CNN

    # plots data
    plot(losses, accuracies, 5)

def compare_to_MLP():
    # complex sequential to take CNN to it limit to compare to MLP
    model_test = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(3200, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    ).to(device)

    #summary(model_test, (1, 28, 28))  # Notice the new shape

    losses, accuracies, _ = run_CNN(model_test, 5) # run CNN

    # plots data
    plot(losses, accuracies, 5)

compare_to_MLP() # runs MLP comparisons experiment
vary_kernel() # runs kernel variation experiment
vary_filters() # runs filter variation experiment
run_batch_norm() # runs batch normalization experiment
run_dropout() # runs dropout experiment
