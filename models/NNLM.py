# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: NNLM.py
@time: 2019/7/17 14:10

Pytorch 实现的 NNLM(Neural Network Language Model) - Predict Next Word

"""
import torch
import torch.nn as nn

class NNLM(nn.Module):
    def __init__(self, n_class, m, n_step, n_hidden):
        super(NNLM, self).__init__()
        # 定义一些参数
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.dtype = torch.FloatTensor
        self.n_class = n_class
        self.m = m

        self.C = nn.Embedding(n_class, m)
        # 初始化工作这里就做了
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(self.dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(self.dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(self.dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(self.dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(self.dtype))

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, self.n_step * self.m) # [batch_size, n_step * n_class]
        tanh = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U) # [batch_size, n_class]
        return output
