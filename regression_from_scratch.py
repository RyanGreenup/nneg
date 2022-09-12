# Imports
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

# Create Data
N = 30
x = np.linspace(start=1, stop=10, num=N)
e = np.random.normal(0, 10, N)
y = x *x + e

x = x.reshape(-1,1)
y = y.reshape(-1, 1)

# Generate Weights
h1 = 32
h2 = 16
w1: np.ndarray = np.random.normal(0, 1, (1, h1))
w2: np.ndarray = np.random.normal(0, 1, (h1, h2))
w3: np.ndarray = np.random.normal(0, 1, (h2, h1))
w1_grad = [0]
w2_grad = [0]
w3_grad = [0]

def matmul(x: np.ndarray, w: np.ndarray):
    return np.matmul(x, w), x.transpose()

def sigmoid(x):
    grad = -np.exp(x)/(np.exp(x) + 1)**2
    return 1/(1+np.exp(x)), grad

def model(x):
    # First Layer
    grad = 1
    x, dw = matmul(x, w1)
    grad *= dw
    x, dw = sigmoid(x)
    grad = np.matmul(grad, dw)
    w1_grad = grad
    # Second Layer
    grad = 1
    x, dw = matmul(x, w2)
    grad *= dw
    x, dw = sigmoid(x)
    grad = np.matmul(grad, dw)
    w2_grad = grad
    # Third Layer
    grad = 1
    x, dw = matmul(x, w3)
    grad *= dw
    w3_grad = grad
    return x, [w1_grad, w2_grad, w3_grad]

plt.scatter(x, y )
yhat = model(x)[0]
plt.plot(x, yhat, color="red", linewidth = 9)
plt.show()


def my_loss(x):
    l = np.sum((model(x)-x)**2)
    g = 2*np.sum((model(x)-x))
    return l, g

eta = 5e-6
losses = []
EPOCH = 10000
print("|------------------------------------------------------------------------------|")
print(" ", end='', flush=True)
for t in range(EPOCH):
# Forward pass
    yhat, grad = model(x)

# Measure the loss
    loss, dy = my_loss(x)
    # loss.backward()
    grad *= dy
    losses.append(loss)

    # Backprop
    w1 -= grad[0]
    w2 -= grad[1]
    w3 -= grad[2]

    if t in np.floor(np.linspace(start=0, stop=EPOCH, num=78)):
        print("#", end='', flush=True)


plt.plot(losses)
plt.show()


plt.scatter(x, y )
yhat = model(x)
plt.plot(x, yhat, color="red", linewidth = 9)
plt.show()



