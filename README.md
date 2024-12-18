# TorchPad
A simple multipurpose padding/cropping utility function for Pytorch.

# Example Usecase:
```python
import torch
from torchpad import tpad as pad

inpt = torch.rand(2, 3, 10, 20, 30)
target = torch.rand(2, 3, 15, 25, 35)
resized = pad(inpt, target)
print(resized.shape)  # Output: (2, 3, 15, 25, 35)

inpt = torch.rand(2, 3, 10, 20, 30)
target = (15, 25, 35)
resized = pad(inpt, target)
print(resized.shape)  # Output: (2, 3, 15, 25, 35)

inpt = torch.rand(2, 3, 10, 20, 30)
target = 25
resized = pad(inpt, target)
print(resized.shape)  # Output: (2, 3, 25, 25, 25)

inpt = torch.rand(2, 3, 10, 20, 30)
target = (-1, 25, -1)  # Ignore 1st and 3rd dimensions
resized = pad(inpt, target)
print(resized.shape)  # Output: (2, 3, 10, 25, 30)

inpt = torch.rand(2, 3, 10, 20, 30)
target = (-1, -1, -1)
resized = pad(inpt, target)
print(resized.shape)  # Output: (2, 3, 10, 20, 30)

target = "invalid"
resized = pad(inpt, target)  # Raises ValueError
```
