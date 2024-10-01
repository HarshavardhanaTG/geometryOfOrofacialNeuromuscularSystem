"""Codes are COPIED from 
Brooks, Daniel, Olivier Schwander, Frédéric Barbaresco, Jean-Yves Schneider, and Matthieu Cord. 
Riemannian batch normalization for SPD neural networks. 
Advances in Neural Information Processing Systems 32 (2019)

Codes are based on the formulations in 
Zhiwu Huang and Luc Van Gool. 
A riemannian network for spd matrix learning. 
In Proceedings of  the AAAI conference on artificial intelligence, volume 31, 2017. 
"""


import torch
import torch.nn as nn
from torch.autograd import Function as F
from . import functional

dtype = torch.float32
device = torch.device('cpu')

class BiMap(nn.Module):

    def __init__(self, hi, ni, no):
        super(BiMap, self).__init__()
        self._W = functional.StiefelParameter(torch.empty(hi, ni, no, dtype = dtype, device = device))
        self._hi = hi
        self._ni = ni
        self._no = no
        functional.initBimapParameter(self._W)

    def forward(self, X):
        return functional.bimapChannels(X, self._W)
    
class LogEig(nn.Module):

    def forward(self, P):
        return functional.LogEig.apply(P)
    
class ReEig(nn.Module):
    
    def forward(self, P):
        return functional.ReEig.apply(P)