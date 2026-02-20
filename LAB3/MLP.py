from FMNISTDatasetMLP import FMNISTDataset
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchsummary import summary
from torch.optim import Adam
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = Path(__file__).resolve().parent

# loads data, calling the FMNISTDatasetMLP class
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

# loss and accuracy plots for MLP comparison, batch norm and dropout
def plot(losses, accuracies, n_epochs):
    plt.figure(figsize=(13, 3))
    plt.subplot(121)
    plt.title('Training Loss value over epochs')
    plt.plot(np.arange(n_epochs) + 1, losses)
    plt.subplot(122)
    plt.title('Testing Accuracy value over epochs')
    plt.plot(np.arange(n_epochs) + 1, accuracies)
    plt.savefig(RESULTS_DIR / "mlp_training_curves.png", bbox_inches='tight')
    plt.close()

# main function that takes in a sequential model and n_epochs, computes losses and prints test accuracies
def run_MLP():
    losses, accuracies = [], []
    n_epochs = 5

    model = nn.Sequential(
        nn.Linear(28 * 28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-3) # presets

    # iterates once for each epoch, computing and updating weights
    for epoch in range(n_epochs):
        print(f"Running epoch {epoch + 1} of {n_epochs}")
        epoch_losses, epoch_accuracies = [], []

        # where the updates actually occur
        for batch in train_dl:
            x, y = batch
            batch_loss = train_batch(x, y, model, opt, loss_fn)
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

    epoch_accuracies = []

    # computes test accuracies
    for batch in test_dl:
     x, y = batch
     batch_acc = accuracy(x, y, model)
     epoch_accuracies.append(batch_acc)
    test_accuracy = np.mean(epoch_accuracies)
    print(f"Test accuracy: {test_accuracy}")

    # plot data
    plot(losses, accuracies, n_epochs)
run_MLP() # runs MLP experiment to compare to CNN
