import numpy as np

#^y = w*x

#^y = 2*x
x = np.array([1,2,3,4],dtype=float)
y = np.array([2,4,6,8],dtype=float)

w = 0.0 

#model prediction
def forward(x):
    return w*x

#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

#gradient
# MSE =1/N * (w*x - y)**2
# dj/dw =1/N *2*(w*x-y)*x

def gradient(x, y, y_predicted):
    return np.dot(2*x, (y_predicted - y)).mean()

print(f"Prediction before training: {forward(5):.3f}")

#training 
learning_rate = 0.01
n_iterations = 20

for epoch in range(n_iterations):
    # prediction = forward pass
    y_pred = forward(x)

    #loss
    l = loss(y,y_pred)
    
    #gradient
    dw = gradient(x, y, y_pred)
    
    #update weights
    
    w -= learning_rate * dw
    
    if epoch % 1 == 0:
        print(f"Epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")
print(f"Prediction after training: f(5) {forward(5):.3f}")

