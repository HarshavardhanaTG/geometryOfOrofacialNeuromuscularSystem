"""Codes are COPIED from 
Brooks, Daniel, Olivier Schwander, Frédéric Barbaresco, Jean-Yves Schneider, and Matthieu Cord. 
Riemannian batch normalization for SPD neural networks. 
Advances in Neural Information Processing Systems 32 (2019)

Codes are based on the formulations in 
Zhiwu Huang and Luc Van Gool. 
A riemannian network for spd matrix learning. 
In Proceedings of  the AAAI conference on artificial intelligence, volume 31, 2017. 
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim
from . import functional

class StiefelOptim():

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for W in self.parameters:
            dirTan = projTanXStiefel(W.grad.data, W.data)
            WNew = expXStiefel(-self.lr * dirTan.data, W.data)
            W.data = WNew

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.data.zero_()


def projTanXStiefel(x, X):
    return x - X.matmul(x.transpose(1, 2)).matmul(X)

def expXStiefel(x, X):
    a = X + x
    Q = torch.zeros_like(a)
    for i in range(a.shape[0]):
        q, _= gramSchmidt(a[i])
        Q[i] = q
    return Q

def gramSchmidt(V):
    n, N = V.shape
    W = torch.zeros_like(V)
    R = torch.zeros((N, N)).double().to(V.device)
    W[:, 0] = V[:, 0]/torch.norm(V[:, 0])
    R[0, 0] = W[:, 0].dot(V[:, 0])
    for i in range(1, N):
        proj = torch.zeros(n).double().to(V.device)
        for j in range(i):
            proj = proj + V[:, i].dot(W[:, j]) * W[:, j]
            R[j, i] = W[:, j].dot(V[:, i])
        if(isclose(torch.norm(V[:, i] - proj), torch.DoubleTensor([0]).to(V.device))):
            W[:, i] = V[:, i]/torch.norm(V[:, i])
        else:
            W[:, i] = (V[:, i] - proj)/torch.norm(V[:, i] - proj)
        R[i, i] = W[:, i].dot(V[:, i])
    return W, R

def isclose(a, b, rtol = 1e-05, atol = 1e-08):
    return ((a - b).abs() <= (atol + rtol * b.abs())).all()


class MixOptimizer():

    def __init__(self, parameters, optimizer = torch.optim.SGD, lr = 1e-2, *args, **kwargs):
        parameters = list(parameters)
        parameters = [param for param in parameters if param.requires_grad]
        self.lr = lr
        self.stiefelParameters = [param for param in parameters if param.__class__.__name__ == 'StiefelParameter']
        
        self.otherParameters = [param for param in parameters if param.__class__.__name__ == 'Parameter']

        self.stiefelOptim = StiefelOptim(self.stiefelParameters, self.lr)
        self.optim = optimizer(self.otherParameters, lr, *args, **kwargs)

    def step(self):
        self.optim.step()
        self.stiefelOptim.step()

    def zero_grad(self):
        self.optim.zero_grad()
        self.stiefelOptim.zero_grad()