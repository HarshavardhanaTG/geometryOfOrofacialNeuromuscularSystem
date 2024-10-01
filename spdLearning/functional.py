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
from torch.autograd import Function as F

class StiefelParameter(nn.Parameter):
    pass

def initBimapParameter(W):
    hi, ni, no = W.shape
    for j in range(hi):
        v = torch.empty(ni, ni, dtype = W.dtype, device = W.device).uniform_(0, 1)
        vv = torch.svd(v.matmul(v.t()))[0][:, :no]
        W.data[j] = vv


def bimap(X, W):
    return W.t().matmul(X).matmul(W)

def bimapChannels(X, W):
    batchSize, channelsIn, nIn, _ = X.shape
    channelsOut, _, nOut = W.shape
    P = torch.zeros(batchSize, channelsOut, nOut, nOut, dtype = X.dtype, device = X.device)
    for co in range(channelsOut):
        P[:, co, :, :] = bimap(X[:, co, :, :], W[co, :, :])
    return P


def modeigForward(P, op, eigMode = 'svd', param = None):
    
    batchSize, channels , n, n = P.shape 
    U, S = torch.zeros_like(P, device = P.device), torch.zeros(batchSize, channels, n, dtype = P.dtype, device = P.device)
    for i in range(batchSize):
        for j in range(channels):
            if(eigMode == 'eig'):
                s, U[i, j] = torch.eig(P[i, j], True)
                S[i, j] = s[:, 0]
            elif(eigMode == 'svd'):
                U[i, j], S[i, j], _ = torch.svd(P[i, j])
    SFn = op.fn(S, param)
    X = U.matmul(BatchDiag(SFn)).matmul(U.transpose(2, 3))
    return X, U, S, SFn

def modeigBackward(dx, U, S, SFn, op, param = None):
    
    SFnDeriv = BatchDiag(op.fnDeriv(S, param))
    SS = S[..., None].repeat(1, 1, 1, S.shape[-1])
    SSFn = SFn[..., None].repeat(1, 1, 1, SFn.shape[-1])
    L = (SSFn - SSFn.transpose(2, 3))/(SS - SS.transpose(2, 3))
    L[L == -np.inf] = 0
    L[L == np.inf] = 0
    L[torch.isnan(L)] = 0
    L = L + SFnDeriv
    dp = L * (U.transpose(2, 3).matmul(dx).matmul(U))
    dp = U.matmul(dp).matmul(U.transpose(2, 3))
    return dp

class LogEig(F):
    
    @staticmethod
    def forward(ctx, P):
        X, U, S, SFn = modeigForward(P, LogOp)
        ctx.save_for_backward(U, S, SFn)
        return X
    @staticmethod
    def backward(ctx, dx):
        U, S, SFn = ctx.saved_variables
        return modeigBackward(dx, U, S, SFn, LogOp)

class ReEig(F):
   
    @staticmethod
    def forward(ctx, P):
        X, U, S, SFn = modeigForward(P, ReOp)
        ctx.save_for_backward(U, S, SFn)
        return X
    @staticmethod
    def backward(ctx, dx):
        
        U, S, SFn = ctx.saved_variables
        return modeigBackward(dx, U, S, SFn, ReOp)
    

class LogOp():

    @staticmethod
    def fn(S, param = None):
        return torch.log(S)
    @staticmethod
    def fnDeriv(S, param = None):
        return 1/S

class ReOp():
    
    _threshold = 1e-3
    @classmethod
    def fn(cls, S, param = None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)
    @classmethod
    def fnDeriv(cls, S, param = None):
        return (S > cls._threshold).float()

def BatchDiag(P):
    batchSize, channels, n = P.shape
    Q = torch.zeros(batchSize, channels, n, n, dtype = P.dtype, device = P.device)
    for i in range(batchSize):
        for j in range(channels):
            Q[i, j] = P[i, j].diag()
    return Q
