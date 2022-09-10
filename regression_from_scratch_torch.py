import torch.nn as nn
import torch
import numpy as np

import matplotlib.pyplot as plt

N = 50
x = np.linspace(start=1, stop=10, num=N)
e = np.random.normal(0, 10, N)
y = x*x + e
x = torch.from_numpy(x).reshape(-1, 1).float()
y = torch.from_numpy(y).reshape(-1, 1).float()

h1 = 2**5
h2 = 1
w1 = torch.from_numpy(np.random.normal(h1, N)).reshape(-1, 1).float()
w1.shape
3+3

w1 = torch.from_numpy(np.random.normal(h2)).reshape(-1, 1).float()
def model(x, y):
    out =


## Define a model
model = nn.Sequential(
    nn.Linear(in_features=1, out_features=h1),
)
model.append(
        nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(h1, h2)))
#model.append(
#        nn.Sequential(
#            nn.Sigmoid(),
#            nn.Linear(h2, 1)))

loss_fn = nn.MSELoss()
eta = 1e-2
opt = torch.optim.SGD(model.parameters(), lr = eta, momentum=0)



losses = []
EPOCH = 1000
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


plt.scatter(x, y )
yhat = model(x).detach().numpy()
plt.plot(x, yhat, color="red", linewidth = 9)
plt.show()
