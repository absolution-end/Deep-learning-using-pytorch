# MNIST
# DataLoader , Transformation 
# Multilayer Neural Net, activation function
# Loss and Otimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
import sys
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist2")
# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameter
input_size =784
hidden_size=100
num_epoch=2
num_classes=10
batch_size=100
learning_rate=0.01

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False)

example = iter(train_loader)
samples, labels = next(example)
print(samples.shape , labels.shape)

for i in range(6):
    # first argument 2 is number of rows and "3" number of columns and third argumnet (i+1) is the index of the current subplot
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('Mnist_images', img_grid)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
        
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out 

model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

# loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

writer.add_graph(model, samples.reshape(-1, 28*28))
# writer.close()
# sys.exit()
# training loop
n_total_steps = len(train_loader)
running_loss =0.0
correct_pred =0.0
for epoch in range(num_epoch):
    for i,(image, labels) in enumerate(train_loader):
        # here we flatten the image
        
        image = image.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward
        output = model(image)
        loss = criterion(output, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predictions = torch.max(output, 1)
        correct_pred += (predictions==labels).sum().item()
        
        if (i+1) %100 == 0:
            print(f"epoch {epoch +1}/ {num_epoch}, step {i+1}/{n_total_steps}, loss {loss.item():.4f}")
            writer.add_scalar('training loss', running_loss/100, epoch * n_total_steps +i)
            writer.add_scalar('accuracy', correct_pred/100, epoch * n_total_steps +i)
            running_loss = 0.0
            correct_pred = 0.0
# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in (test_loader):
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        
        # value and index
        _, predictions = torch.max(output,1) 
        n_samples += labels.shape[0]
        n_correct += (predictions==labels).sum().item()
        
    acc = 100.0  * n_correct/n_samples
    print(f'accuracy = {acc}')    
writer.close()