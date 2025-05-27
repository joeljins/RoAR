import torch

def g(x, x_0, theta, lamb):
    return 1 - (1 / (1 + torch.exp(-theta * x))) + lamb * torch.abs(x_0 - x)

def gradient(x, x_0, theta, lamb):
    sigmoid_derivative = -theta * torch.exp(-theta * x) / (1 + torch.exp(-theta * x))**2
    abs_derivative = -lamb * torch.sign(x_0 - x)
    return sigmoid_derivative + abs_derivative

x = torch.tensor(0.0, requires_grad=False)
alpha = 0.1
epoch =  1000

epsilon = 1e-4
for i in range(epoch):
    grad = gradient(x, torch.tensor(0.0), 1.0, 0.1)
    if torch.abs(grad) < epsilon:
        print(f"Stopping at epoch {i+1} due to small gradient: {grad.item():.6f}")
        break
    x = x - alpha * grad
    print(f"Epoch {i+1}: x = {x.item():.4f}, gradient = {grad.item():.4f}")

