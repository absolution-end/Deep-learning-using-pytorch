# Data
# Hyperparameter
# dataloder and transformers
# graph if needed
# Multilayer Neural Network
# training loop (batch sizing)
# test data or model evaluation


import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transformrs
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameter
batch_size = 100
num_epoch = 2
hidden_size=10
num_class = 10
learning_rate = 0.001
input_size=784
train_dataset = torchvision.datasets.MNIST(root='./data'
                                           ,train=True, transform=transformrs.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transformrs.ToTensor())

train_load = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
test_load = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

example = iter(train_load)
samples, lables = next(example)
print(samples.shape, lables.shape)

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(samples[i][0])
plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_class)
        
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_class).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop 
n_steps = len(train_load)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_load):
        
        # for image flatten
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        
        # forward
        output = model(images)
        loss = criterion(output,labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) %100 ==0:
            print(f"epoch {epoch+1}/{num_epoch} steps {i+1}/{n_steps}, loss {loss.item():.4f}")
            
# test or validate score
with torch.no_grad():
    n_correct=0
    n_samples=0
    for images, labels in (test_load):
        # flatten image
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        
        # model
        output = model(images)
        
        # values and index
        _, predictions = torch.max(output,1)
        n_samples += labels.shape[0]
        n_correct += (predictions==labels).sum().item()
    
    acc = 100.0 * n_correct/n_samples
    print(f'accuracy={acc}')