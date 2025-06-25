# 1.) Design model (input, output and forward pass)
# 2.) Construct loss and optimizer
# 3.) Training loop
    #   - Forward pass
    #   - Backward pass
    #   - Update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0.) Load dataset
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0],1) # Reshape y to be a column vector

n_samples, n_features = x.shape

# 1.) Design model (input, output and forward pass)
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2.) Construct loss and optimizer
learining_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learining_rate)

# 3.) Training loop
n_iterations = 100

for epoch in range(n_iterations):
    # forward pass 
    y_pred = model(x)
    
    # loss
    l = criterion(y_pred, y)
    
    # backward pass
    l.backward()
    # update weights
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}/{n_iterations}, loss = {l.item():.4f}')

predict= model(x).detach().numpy()

plt.plot(x_numpy, y_numpy, 'ro', label='Original data')
plt.plot(x_numpy, predict, 'b',label='Fitted line')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using PyTorch')
plt.show()