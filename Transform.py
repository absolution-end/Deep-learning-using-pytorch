import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Tensor:
    def __call__(self, sample):
        input , target = sample
        return torch.from_numpy(input), torch.from_numpy(target)

class Winedataset(Dataset):
    
    def __init__(self, transform):
        xy = np.loadtxt('G:\Deep-learning-using-pytorch\Dataset\wine.csv', delimiter=",", dtype=np.float32 ,skiprows=1)
        self.x = xy[:,1:]
        self.y = xy[:,[0]]
        self.n_samples = xy.shape[0]
        self.transform = transform
        
    def __getitem__(self,index):
        sample= self.x[index] , self.y[index]
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        return self.n_samples
    
dataset = Winedataset(transform=Tensor())

firstd = dataset[0]
item, label = firstd
print(type(item) , type(label))