import torch

#^y = w*x

#^y = 2*x``
x = torch.tensor([1,2,3,4],dtype=float)
y = torch.tensor([2,4,6,8],dtype=float)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction
def forward(x):
    return w*x

#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

#gradient
# MSE =1/N * (w*x - y)**2
# dj/dw =1/N *2*(w*x-y)*x

print(f"Prediction before training: {forward(5):.3f}")

#training 
learning_rate = 0.01
n_iterations = 20

for epoch in range(n_iterations):
    # prediction = forward pass
    y_pred = forward(x)

    #loss
    l = loss(y,y_pred)
    
    #gradient = backward pass
    l.backward()  # computes the gradient of the loss w.r.t. w
    
    #update weights
    with torch.no_grad(): # using no grad context to avoid tracking history
        w -= learning_rate * w.grad
    
    # zero gradients
    w.grad.zero_()   # reset gradients to zero before the next iteration
    if epoch % 1 == 0:
        print(f"Epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")
print(f"Prediction after training: f(5) {forward(5):.3f}")