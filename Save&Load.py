import torch
import torch.nn as nn

# Lazy method

####Complete Model####
torch.save(model,PATH)

# class must be defined somewhere
model = torch.load(PATH)
model.eval()



#### STATE DICT ####
torch.save(model.state_dict(), PATH)

# model must be created again with the parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()