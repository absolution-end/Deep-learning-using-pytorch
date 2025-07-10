# 1.) Design model(input, output and forward pass)
# 2.) Construct loss and optimizer
# 3.) Training loop
#     -Forward pass
#     -Backward pass
#     -Update weights
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0.) Prepare dataset
bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target

n_samples, n_features = x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Scale
sc = StandardScaler()
x_train =sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#  after scaling we need to convert the data into PyTorch tensors
x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# To change the shape we use "veiw()"
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
# 1.) model
# f = wx + b, in the end we will have a sigmoid activation function
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear =nn.Linear(input_size, 1)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegression(n_features)
# 2.) loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3.) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x_train)
    # compute Loss
    loss =criterion(y_pred, y_train)
    # Backward pass
    loss.backward()
    # optimizer step
    optimizer.step()
    
    # optimizer zero grad
    optimizer.zero_grad()
    
    if (epoch+1) %10 == 0:
        print(f"Epoch [{epoch+1} /{num_epochs}],loss = {loss.item():.4f}")

# Evaluate the model
with torch.no_grad():
    y_test_pred = model(x_test)
    y_test_pred_cls = y_test_pred.round()  # Convert probabilities to class labels (0 or 1)
    
    acc = (y_test_pred_cls == y_test.view(y_test.shape[0], 1)).sum().item() / y_test.shape[0]
    print(f'Accuracy: {acc * 100:.2f}%')