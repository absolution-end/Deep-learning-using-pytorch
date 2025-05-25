import torch

# The requires_grad=True argument indicates that we want to track operations on this tensor
# and compute gradients with respect to it during backpropagation.

x= torch.randn(3, requires_grad=True)
print(x)

# Perform some operations on the tensor
y =  x+2
print(y)

z = y * y * 3
z = z.mean()
print(z)

z.backward()  # This computes the gradient of out with respect to x dz/dx
print(x.grad)  # This will print the gradient of out with respect to x


# to not use autograd, we can use the methods like :
x.requires_grad(False)
# or
x.detach()  # This will return a new tensor that does not require gradients 
# or
x = x.detach()  # This will also return a new tensor that does not require gradients
# or
with torch.no_grad():