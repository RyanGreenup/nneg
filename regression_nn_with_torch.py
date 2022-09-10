import torch.nn as nn
import torch
import numpy as np

import matplotlib.pyplot as plt

x = np.linspace(start=1, stop=10, num=30)
y = np.sin(x)
x = torch.from_numpy(x).reshape(-1, 1).float()
y = torch.from_numpy(y).reshape(-1, 1).float()

h1 = 100
h2 = 100
## Define a model
model = nn.Sequential(
    nn.Linear(in_features=1, out_features=h1),
    nn.Sigmoid(),
    nn.Linear(h1, h2),
    nn.Sigmoid(),
    nn.Linear(h2, 1),
)


loss_fn = nn.MSELoss()
eta = 1e-3
opt = torch.optim.RMSprop(model.parameters(), lr = eta)



losses = []
EPOCH = 10000
print("|------------------------------------------------------------------------------|")
print(" ", end='', flush=True)
for t in range(EPOCH):
    # Forward pass
    yhat = model(x.float())

    # Measure the loss
    loss = loss_fn(yhat, y)

    if t in np.floor(np.linspace(start=0, stop=EPOCH, num=78)):
        print("#", end='', flush=True)


    # Backprop
    opt.zero_grad()
    loss.backward()
    opt.step()


plt.plot(x, y )
yhat = model(x).detach().numpy()
plt.plot(x, yhat)
plt.show()
