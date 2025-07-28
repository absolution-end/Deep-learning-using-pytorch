# MNIST-Image data
# DataLoader and Transformation
# Multilayer Neural Net and Activation function
# Loss & Optimizer
# Training Loop (Batch training)
# Model evaluation
# GPU support


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameter
input_size = 800
hidden_size = 100
num_epoch = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Dataset has PTLImages from range [0,1]
# we Transform them to tensor of normilazation range [-1,1]
transforms = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((.5,.5,.5),(.5,.5,.5))]) # (R,G,B)
