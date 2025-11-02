import torch

def differentiable_and(x, y):
    return x * y  # Approximation of AND operation using product

# Example tensors representing logical values
x = torch.tensor([1.0], requires_grad=True)  # True
y = torch.tensor([0.0], requires_grad=True)  # False

# Compute AND operation
result = differentiable_and(x, y)
result.backward()

print("Differentiable AND result:", result.item())
print("Gradient with respect to x:", x.grad.item())