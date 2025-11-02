import torch

def tensorized_and(x, y):
    # x and y are vectors representing boolean values
    return torch.min(x, y)

# Represent True as [1, 0] and False as [0, 1]
true = torch.tensor([1.0, 0.0])
false = torch.tensor([0.0, 1.0])

# Compute True AND False
result = tensorized_and(true, false)
print("True AND False:", result)