# Imports
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

# TODO Add bias
# TODO Add neurons
# TODO Add 3 layers

# Create Data
N = 30
x = np.linspace(start=1, stop=10, num=N)
e = np.random.normal(0, 10, N)
y = x

x = x.reshape(-1,1)
y = y.reshape(-1, 1)

# Generate Weights
h1 = 1
h2 = 1

w1: np.ndarray = np.random.random((1, h1))
w2: np.ndarray = np.random.random((h1, h2))
w3: np.ndarray = np.random.random((h2, 1))

w1_grad = [0]
w2_grad = [0]
w3_grad = [0]

a="------------------->"
if w1[0][0] < 0 and w2[0][0] < 0:
    print(a+"Big Bump!")
elif w1[0][0] < 0:
    print(a+"no Bump (yetish)!")
elif w2[0][0] < 0:
    print(a+"Small Bump")
else:
    print(a+"Perfection")


# def matmul(x: np.ndarray, w: np.ndarray):
#    return np.matmul(x, w), x

def matmul(x: np.ndarray, w: np.ndarray):
    y = np.matmul(x, w)
    dw = x
    return y, dw

def sigmoid(x):
    grad = -np.exp(x)/(np.exp(x) + 1)**2
    y = 1/(1+np.exp(x))
    return y, grad

def model(x):
    # First Layer
    grad = np.array([1,1]) # np.ones((30,2))
    grad2 = 1
    x, dw = matmul(x, w1)
    grad *= dw
    x, dx = sigmoid(x)
    grad *= dx
    # Second Layer
    x, dw = matmul(x, w2)
    grad2 *= dw
    return x, [grad, grad2]

plt.scatter(x, y )
yhat, _ = model(x)
plt.plot(x, yhat, color="red", linewidth = 9)
plt.show()


def my_loss(x, y):
    yhat, _ = model(x)
    l = np.sum((yhat-y)**2)
    g = 2*(yhat-y)
    return l, g

eta = 1e-04
losses = []
EPOCH = 10000

print("|------------------------------------------------------------------------------|")
print(" ", end='', flush=True)
for t in range(EPOCH):
# Forward Pass
    yhat, dw = model(x)
# Loss
    l, dy    = my_loss(x, y)
    losses.append(l)
# Backward
    grad = np.matmul(dy.transpose(), dw[0])
    grad2 = np.matmul(dy.transpose(), dw[1])
# Backprop
    w1 -= grad * eta
    w2 -= grad2 * eta
    if t in np.floor(np.linspace(start=0, stop=EPOCH, num=78)):
        print("#", end='', flush=True)

plt.plot(losses)
plt.show()


plt.scatter(x, y )
yhat, _ = model(x)
plt.plot(x, yhat, color="red", linewidth = 9)
plt.show()

# print(w1)
# print(w2)


import numpy as np
mything: np.ndarray = np.array([1,3,4,5,32,3,2,2,])
np.sum(mything)
type(mything)
