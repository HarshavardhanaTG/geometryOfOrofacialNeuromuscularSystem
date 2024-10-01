"""Neural network architecture for multivariate sEMG timeseries learning."""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from . import spdNN

class manifoldNet(nn.Module):
    def __init__(self, classifications):
        super(manifoldNet, self).__init__()
        
        self.re = spdNN.ReEig()
        self.bimap1 = spdNN.BiMap(1, 22, 22)
        self.bimap2 = spdNN.BiMap(1, 22, 20)
        self.bimap3 = spdNN.BiMap(1, 20, 16)

        self.logeig = spdNN.LogEig()

        self.linear = nn.Linear(256, classifications)

    def forward(self, x):
        
        x = self.re(self.bimap1(x))
        x = self.re(self.bimap2(x))
        x = self.re(self.bimap3(x))
        xVec = self.logeig(x).view(x.shape[0], -1)
        x = self.linear(xVec)
        return x


class learnSPDMatrices(nn.Module):
    def __init__(self, classifications):
        super(learnSPDMatrices, self).__init__()
        
        self.snn = manifoldNet(classifications)

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.snn(x)
        return x
