import torch

# Example 2D tensor
tensor = torch.tensor([[1.0, 2.0], [float('nan'), 4.0], [float('nan'), 6.0]])

tensor = torch.where(torch.isnan(tensor), torch.tensor(0.0), tensor) 


# nan_mask = torch.isnan(tensor)
# tensor[nan_mask] = 0

"""
std = torch.clamp(std, min=0.0)
std = torch.where(torch.isnan(std), torch.tensor(0.0), std) 
res = None
try:
    res = torch.normal(self.loc.expand(shape), std)
except RuntimeError as ex:
    print('error', ex, std)
"""

print(tensor) 