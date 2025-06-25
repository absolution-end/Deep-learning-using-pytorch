# Step 4: Gradients in PyTorch
# 1.) Design model (input, output and forward pass)
# 2.) Construct loss and optimizer
# 3.) Training loop
#     - Forward pass
#     - Backward pass
#     - Update weights

import torch
import torch.nn as nn

x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

n_samples , n_features = x.shape
input_size = n_features
output_size = n_features
print(f"{n_samples}, {n_features}")
x_test = torch.tensor([5], dtype=torch.float32)

model = nn.Linear(input_size, output_size)

print(f"prediction before training:{model(x_test).item()}")

# training
Learning_rate =0.01
n_iterations = 100

loss = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)

for epoch in range(n_iterations):
    # forward pass = prediction
    y_pred = model(x)
    
    #loss computation
    l = loss(y, y_pred)
    
    # backward pass = compute gradients
    l.backward()
    
    # update weights
    optimizer.step() # update the parameters using the gradients computed by the backward pass
    optimizer.zero_grad() # reset gradients to zero before the next iteration
    
    if epoch % 10 ==0:
        [w, b] = model.parameters()
        print(f"Epoch {epoch+1}: w = {w[0][0].item():.3f}, b = {b.item():.3f}, loss = {l:.8f}")
# Final prediction after training
print(f"Prediction after training: f(5) = {model(x_test).item():.3f}")
