import torch 

x = torch.empty((2, 3, 4), dtype=torch.float32)
print(x)
print("\n")


# If want to print random values 
x =torch.rand((2, 3, 4), dtype=torch.float32)
print(x,"\n")

x = torch.randn((2,2), dtype=torch.float32)
print(x[1,1].item())

# Tensor using numpy 
import numpy as np
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(b)  # This will show the updated values in b as well since they share memory
print(a) # This will show the updated values in a
"""
A tensor in PyTorch is a multi-dimensional array, similar to NumPy's ndarray,
and serves as the fundamental data structure for representing and manipulating data. 
Tensors can represent scalars (0-dimensional), vectors (1-dimensional), matrices (2-dimensional),
and higher-dimensional data.
PyTorch utilizes tensors to encode the inputs and outputs of a model, as well as the model's parameters. 
"""   