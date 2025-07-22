import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    
    def __init__(self):
        # Data Loading
        xy = np.loadtxt('G:\Deep-learning-using-pytorch\Dataset\wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        # first "slice -> : for rows" Second "slice -> for columns" After coma"," 
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_Samples = xy.shape[0]
        
    def __getitem__(self, index):
        # Dataset[0]
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.n_Samples

dataset = WineDataset()

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# dataiter = iter(dataloader)
# data = next(dataiter)
# features, labels = data
# print(f"The Features are {features}, The labels are {labels}")

# let's take dummy training loop
