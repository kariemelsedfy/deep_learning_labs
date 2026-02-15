import torch
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#MLP data set loader that DOES flatten the image
class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.view(-1,28*28).float()/255 #flattened images
        self.x, self.y = x, y
    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)
    def __len__(self):
        return len(self.x)
