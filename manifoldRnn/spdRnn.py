"""Neural network architecture for multivariate sEMG timeseries learning.

Parts of the code are COPIED from 
Seungwoo Jeong, Wonjun Ko, Ahmad Wisnu Mulyadi, and Heung-Il Suk. 
Deep efficient continuous manifold learning for time series modeling. 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torchdiffeq import odeint as odeint

from . import spdNN

class spdRnnNet(nn.Module):
    def __init__(self, classifications):
        super(spdRnnNet, self).__init__()
        self.classifications = classifications
        self.CNN = spdNet()
        self.RNN = rnnNet(self.classifications)

    def forward(self, x):
        b, s, c, c = x.shape
        x = self.CNN(x)
        x = self.RNN(x)
        return x

class rnnNet(nn.Module):
    def __init__(self, classifications, ode = True, device = 'cuda:0'):
        super(rnnNet, self).__init__()
        self.latents = 20
        self.diagUnits = self.latents
        self.lowTriag = self.latents * (self.latents - 1) // 2
        self.nLayers = 1
        self.factor = 3
        self.classifications = classifications

        self.odefunc = ODEFunc(nInputs = (self.diagUnits)  + (self.lowTriag // self.factor), nLayers = self.nLayers, nUnits = (self.diagUnits) + (self.lowTriag // self.factor))
        self.rgruD = RGRUCell(self.latents, self.diagUnits, True)
        self.rgruL = RGRUCell(self.lowTriag, self.lowTriag // self.factor, False)

        self.odefuncRe = ODEFunc(nInputs = (self.diagUnits)  + (self.lowTriag // self.factor), nLayers = self.nLayers, nUnits = (self.diagUnits) + (self.lowTriag // self.factor))
        self.rgruDRe = RGRUCell(self.latents, self.diagUnits, True)
        self.rgruLRe = RGRUCell(self.lowTriag, self.lowTriag // self.factor, False)

        self.softplus = nn.Softplus()
        self.cls = nn.Sequential(nn.Linear(self.diagUnits + (self.lowTriag // self.factor), self.classifications))

        self.ode = ode
        self.device = device


    def forward(self, x):
        b, s, _ , _ = x.shape
        xD, xL = self.cholDe(x)

        hD = torch.ones(x.shape[0], self.diagUnits, device = self.device)
        hL = torch.zeros(x.shape[0], self.lowTriag // self.factor, device = self.device)

        hDRe = torch.ones(x.shape[0], self.diagUnits, device = self.device)
        hLRe = torch.zeros(x.shape[0], self.lowTriag // self.factor, device = self.device)

        times = torch.from_numpy(np.arange(s + 1)).float().to(self.device)

        out = []
        outRe = []

        for i in range(x.shape[1]):
            if self.ode == True:
                hp = odeint(self.odefunc, torch.cat((hD.log(), hL), dim = 1), times[i:i + 2], rtol = 1e-4, atol = 1e-5, method = 'euler')[1]
                hD = hp[:, :self.diagUnits].tanh().exp()
                hL = hp[:, self.diagUnits:]

                hpRe = odeint(self.odefuncRe, torch.cat((hDRe.log(), hLRe), dim = 1), times[i:i + 2], rtol = 1e-4, atol = 1e-5, method = 'euler')[1]
                hDRe = hpRe[:, :self.diagUnits].tanh().exp()
                hLRe = hpRe[:, self.diagUnits:]

            hD = self.rgruD(xD[:, i, :], hD)
            hL = self.rgruL(xL[:, i, :], hL)
            out.append(torch.cat((hD.log(), hL), dim = 1))

            hDRe = self.rgruDRe(xD[:, x.shape[1] - i - 1, :], hDRe)
            hLRe = self.rgruLRe(xL[:, x.shape[1] - i - 1, :], hLRe)
            outRe.append(torch.cat((hDRe.log(), hLRe), dim = 1))
        h = torch.stack(out).mean(0)
        hRe = torch.stack(outRe).mean(0)
        h = h + hRe
        return self.cls(h)

    def cholDe(self, x):
        b, s, n, n = x.shape
        x = x.reshape(-1, n, n)
        L = torch.linalg.cholesky(x)
        d = x.new_zeros(b * s, n)
        l = x.new_zeros(b * s, n * (n - 1) // 2)
        for i in range(b * s):
            d[i] = L[i].diag()
            l[i] = torch.cat([L[i][j: j + 1, :j] for j in range(1, n)], dim = 1)[0]
        return d.reshape(b, s, -1), l.reshape(b, s, -1)

class RGRUCell(nn.Module):

    def __init__(self, inputSize, hiddenSize, diag = True):
        super(RGRUCell, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.diag = diag
        if diag:
            layer = PosLinear
            self.nonlinear = nn.Softplus()
        else:
            layer = nn.Linear
            self.nonlinear = nn.Tanh()
        self.x2h = layer(inputSize, 3 * hiddenSize, bias = False)
        self.h2h = layer(hiddenSize, 3 * hiddenSize, bias = False)
        self.bias = nn.Parameter(torch.rand(3 * hiddenSize))
        self.resetParameters()

    def resetParameters(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gateX = self.x2h(x)
        gateH = self.h2h(hidden)

        gateX = gateX.squeeze()
        gateH = gateH.squeeze()

        iR, iI, iN = gateX.chunk(3, 1)
        hR, hI, hN = gateH.chunk(3, 1)
        bR, bI, bN = self.bias.chunk(3, 0)

        if self.diag:
            resetgate = (bR.abs() * (iR.log() + hR.log()).exp()).sigmoid()
            inputgate = (bI.abs() * (iI.log() + hI.log()).exp()).sigmoid()
            newgate = self.nonlinear((bN.abs() * (iN.log() + (resetgate * hN).log()).exp()))
            hy = (newgate.log() * (inputgate) + (1 - inputgate) * hidden.log()).exp()
        else:
            resetgate = (iR + hR + bR).sigmoid()
            inputgate = (iI + hI + bI).sigmoid()
            newgate = self.nonlinear(iN + (resetgate * hN) + bN)
            hy = (1 - inputgate) * hidden + inputgate * newgate

        return hy

class PosLinear(nn.Module):
    def __init__(self, inDim, outDim, bias = False):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((inDim, outDim)))

    def forward(self, x):
        return torch.matmul(x, torch.abs(self.weight))


class ODEFunc(nn.Module):
    def __init__(self, nInputs, nLayers, nUnits):
        super(ODEFunc, self).__init__()
        self.gradientNet = odefunc(nInputs, nLayers, nUnits)

    def forward(self, tLocal, y, backwards = False):
        grad = self.getOdeGradientNN(tLocal, y)
        if backwards:
            grad = -grad
        return grad

    def getOdeGradientNN(self, tLocal, y):
        return self.gradientNet(y)

    def sampleNextPointFromPrior(self, tLocal, y):
        return self.getOdeGradientNN(tLocal, y)

class odefunc(nn.Module):
    def __init__(self, nInputs, nLayers, nUnits):
        super(odefunc, self).__init__()
        self.Layers = nn.ModuleList()
        self.Layers.append(nn.Linear(nInputs, nUnits))
        for i in range(nLayers):
            self.Layers.append(nn.Sequential(nn.Tanh(), nn.Linear(nUnits, nUnits // 2)))
            self.Layers.append(nn.Dropout(0.1))
            self.Layers.append(nn.Sequential(nn.Tanh(), nn.Linear(nUnits // 2, nUnits)))
        self.Layers.append(nn.Tanh())
        self.Layers.append(nn.Dropout(0.1))
        self.Layers.append(nn.Linear(nUnits, nInputs))

    def forward(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x

class spdNet(nn.Module):
    def __init__(self):
        super(spdNet, self).__init__()
        
        self.bimap1 = spdNN.BiMap(1, 22, 22)
        self.bimap2 = spdNN.BiMap(1, 22, 20)
        
    
    def forward(self, x):
        b, s, c, _ = x.shape
        x = x.reshape(b * s, c, c)
        x = x.unsqueeze(1)

        x = self.bimap1(x)
        x = self.bimap2(x)

        x = x.squeeze()
        return x.reshape(b, s, 20, 20)