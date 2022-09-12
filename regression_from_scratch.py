# Imports
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

# Create Data
N = 30
x = np.linspace(start=1, stop=10, num=N)
e = np.random.normal(0, 10, N)
y = x

x = x.reshape(-1,1)
y = y.reshape(-1, 1)

# Generate Weights
h1 = 1
h2 = 16
w1: np.ndarray = np.random.normal(0, 1, (1, h1))
w2: np.ndarray = np.random.normal(0, 1, (h1, h2))
w3: np.ndarray = np.random.normal(0, 1, (h2, h1))
w1_grad = [0]
w2_grad = [0]
w3_grad = [0]

def matmul(x: np.ndarray, w: np.ndarray):
    return np.matmul(x, w), x

def sigmoid(x):
    grad = -np.exp(x)/(np.exp(x) + 1)**2
    return 1/(1+np.exp(x)), grad

def model(x):
    # First Layer
    grad = 1
    x, dw = matmul(x, w1)
    grad *= dw
    return x, grad

plt.scatter(x, y )
yhat, _ = model(x)
plt.plot(x, yhat, color="red", linewidth = 9)
plt.show()


def my_loss(x, y):
    yhat, _ = model(x)
    l = np.sum((yhat-y)**2)
    g = 2*(yhat-y)
    return l, g

eta = 5e-6
losses = []
EPOCH = 100000

print("|------------------------------------------------------------------------------|")
print(" ", end='', flush=True)
for t in range(EPOCH):
# Forward Pass
    yhat, dw = model(x)
# Loss
    l, dy    = my_loss(x, y)
    losses.append(l)
# Backward
    grad = np.matmul(dy.transpose(), dw)
# Backprop
    w1 -= grad * eta
    if t in np.floor(np.linspace(start=0, stop=EPOCH, num=78)):
        print("#", end='', flush=True)


plt.plot(losses)
plt.show()


plt.scatter(x, y )
yhat, _ = model(x)
plt.plot(x, yhat, color="red", linewidth = 9)
plt.show()



