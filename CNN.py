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
num_epoch = 4
num_classes = 10
batch_size = 4
learning_rate = 0.001

# Dataset has PTLImages from range [0,1]
# we Transform them to tensor of normilazation range [-1,1]
transforms = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((.5,.5,.5),(.5,.5,.5))]) # (R,G,B)

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transforms)

test_dataset = torchvision.datasets.CIFAR10(root='/data', train=False,
                                            download=False,transform=transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog','horse','ship','truck')

# implement conv net
class ConvNet(nn.Module):
        def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
        def forward(self, x):
                pass
        
model = ConvNet().to(device)

# loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epoch):
        for i , (image, label) in enumerate(train_loader):
                
                # origin shape: [4,3,32,32] = 4,3,1024
                # input layer: 3 input channel, 6 output channel, 5 kernal channel
                image = image.to(device)
                label = label.to(device)
                
                # forward pass
                output = model(image)
                loss = criterion(output, label)
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (1+i) %2000 ==0:
                        print(f'epoch [{epoch+1}/{num_epoch}], step[{i+1}/{n_total_steps}], [loss: {loss.item():.4f}]')
                        
print("finish training")