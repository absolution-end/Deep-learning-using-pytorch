# 1.) Design model (input, output and forward pass)
# 2.) construct loss and optimizer
# 3.) training loop
#     - forward pass
#     - backward pass
#     - update weights

# imports 
import torch 
import torch.nn as nn #Neural Network module

x= torch.tensor([1,2,3,4],dtype=float)
y= torch.tensor([2,4,6,8],dtype=float)

w= torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# forward pass
def forward(x):
    return w * x

print(f"prediction before training: {forward(5):.3f}")

# Training
learning_rate = 0.01
n_iterations = 100

loss = nn.MSELoss() #mean squared error loss
optimizer = torch.optim.SGD([w], lr = learning_rate)

for epoch in range(n_iterations):
    # forward pass
    y_pred = forward(x)
    # loss computation
    l = loss(y, y_pred) 
    
    # backward pass
    l.backward()  # computes the gradient of the loss w.r.t. w
    
    # update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad() # reset gradients to zero before the next iteration
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")
        
# Final prediction after training
print(f"Prediction after training: f(5) {forward(5):.3f}")