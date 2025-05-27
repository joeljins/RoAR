import torch

# Parameters
x_0 = torch.tensor([1.0])
theta = 5.0
lamb = 0.1

def g(x, x_0, theta, lamb):
    return 1 - (1 / (1 + torch.exp(-theta * x))) + lamb * torch.abs(x_0 - x)

# Step 1: Initialize x with requires_grad=True to track gradients
x = torch.tensor([0.0], requires_grad=True)
learning_rate = 0.1

# Step 2: Run gradient descent
for step in range(100):
    # Step 3: Compute loss
    loss = g(x, x_0, theta, lamb)

    # Step 4: Backward pass
    loss.backward()

    # Step 5: Gradient descent update
    with torch.no_grad():
        x -= learning_rate * x.grad

    # Step 6: Zero gradients
    x.grad.zero_()

    # Print progress
    if step % 10 == 0:
        print(f"Step {step}: x = {x.item():.4f}, loss = {loss.item():.4f}")
