import torch.nn as nn
import torch
import numpy as np

import matplotlib.pyplot as plt

N = 10
x = np.linspace(start=1, stop=10, num=N)
e = np.random.normal(0, 10, N)
y = x*x + e
x = torch.from_numpy(x).reshape(-1, 1).float()
y = torch.from_numpy(y).reshape(-1, 1).float()

h1 = 5
h2 = 3
w1: torch.Tensor = torch.tensor(np.random.normal(0, 1, (1, h1)), requires_grad=True)
w2: torch.Tensor = torch.tensor(np.random.normal(0, 1, (h1, h2)), requires_grad=True)
w3: torch.Tensor = torch.tensor(np.random.normal(0, 1, (h2, 1)), requires_grad=True)

def model(x):
    x = torch.matmul(x, w1.float())
    x = torch.sigmoid(x)
    x = torch.matmul(x, w2.float())
    x = torch.sigmoid(x)
    x = torch.matmul(x, w3.float())
    return x


eta = 1e-5
loss_fn = nn.MSELoss()



losses = []
EPOCH = 100000
print("|------------------------------------------------------------------------------|")
print(" ", end='', flush=True)
for t in range(EPOCH):
    # Forward pass
    yhat = model(x.float())

    # Measure the loss
    loss: torch.Tensor = loss_fn(y, model(x))
    loss.backward()
    losses.append(loss.detach().numpy())



    # Backprop
    for p in [w1, w2, w3]:
        with torch.no_grad():
            temp = p - p.grad * eta
            p.copy_(temp)

    if t in np.floor(np.linspace(start=0, stop=EPOCH, num=78)):
        print("#", end='', flush=True)


plt.plot(losses)
plt.show()


plt.scatter(x, y )
yhat = model(x).detach().numpy()
plt.plot(x, yhat, color="red", linewidth = 9)
plt.show()
