# Imports
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

class Data:

    """Data for fitting Neural Network"""

    def __init__(self, sigma: float = 10, N: int = 30):
        """Initialise the data"""
        self.x = np.linspace(start=1, stop=10, num=N)
        e = np.random.normal(0, sigma, N)
        self.y = self.x**2 + e

        self.x = self.x.reshape(-1,1)
        self.y = self.y.reshape(-1, 1)
    # TODO add test training split


class Vec():

    """Any shape of values, a wrapper over numpy"""

    def __init__(self, values, shape):
        """Initialise a vec"""
        self.x = np.array(values).reshape(shape)
        self.shape = shape

class Weight():
    """A vector that also tracks the gradient"""

    def __init__(self, shape: tuple):
        """Initialise Random Weights"""
        self.x: np.ndarray = np.random.random(shape)
        self.shape = shape
        self.grad = np.ones(shape)

    def matmul(self, B: np.ndarray):
        self.grad = self.grad @ B.transpose()
        return self.x @ B

    def sigmoid(self):
        self.grad = -np.exp(self.x)/(np.exp(self.x) + 1)**2
        return 1/(1+np.exp(self.x)), grad

# Create Data
d = Data(N=5)

# Generate Weights
h1 = 1
h2 = 1

w1 = Weight( (1, h1))
w2 = Weight((h1, h2))
w3 = Weight((h2, 1))

w1.x.shape
d.x.transpose().shape

w1.matmul(d.x.transpose())
w1.x @ d.x.transpose()

def model(x):
    # TODO add bias
    y = w1.matmul(x.transpose())
    return y

plt.scatter(d.x, d.y )
yhat = model(d.x)
plt.plot(d.x, yhat, color="red", linewidth = 9)
plt.show()


def my_loss(x, y):
    yhat, _ = model(x)
    l = np.sum((yhat-y)**2)
    g = 2*(yhat-y)
    return l, g

eta = 1e-6
losses = []
EPOCH = 1000

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



