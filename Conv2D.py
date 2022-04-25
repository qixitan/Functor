# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: Conv2D.py
# @Time: 2022/4/24 20:36

import torch
import torch.nn as nn
import numpy as np


class Conv2D(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            kernel_size: (int, int),
            stride: int = 1,
            padding: int = 0,
    ):
        super(Conv2D, self).__init__()
        self.weight = torch.stack(
            [nn.Parameter(torch.randn(size=(inplanes,  kernel_size[0], kernel_size[1]))) for _ in range(planes)], 0
        )
        self.bias = nn.Parameter(torch.zeros(inplanes, 1))
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        return torch.stack([sum(self.corr2d(x, k) + b for x, k, b in zip(X, K, self.bias)) for K in self.weight], 0)

    def corr2d(self, X, K):
        h, w = K.shape
        y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i, j] = (X[i:i + h, j:j + w] * K).sum()
        return y


def Conv2D_np(X, inplanes: int, planes: int, kernel_size: int):
    assert X.shape[0] == inplanes
    kernel = np.stack([np.random.randn(inplanes, kernel_size, kernel_size) for _ in range(planes)], 0)
    bias = np.random.randn(inplanes, 1)
    return np.stack([sum(corr2d(x, k) + b for x, k, b in zip(X, K, bias)) for K in kernel], 0)


def corr2d(X, K):
        h,w = K.shape
        y = np.zeros(shape=(X.shape[0]-h+1, X.shape[1]-w+1))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i][j] = (X[i:i+h, j:j+w]).sum()
        return y


def test1():
    inplanes, planes, kernel_size = 3, 4, (3, 3)
    X = torch.randn(inplanes, 32, 32)
    net = Conv2D(inplanes=inplanes, planes=planes, kernel_size=kernel_size)
    y = net(X)
    # print(y.shape)


def test2():
    inplanes, planes, kernel_size = 3, 4, 3
    X = np.random.randn(inplanes, 32, 32)
    y = Conv2D_np(X, inplanes, planes, kernel_size)
    # print(y.shape)


def main():
    import time
    t1 = time.time()
    test1()
    t2 = time.time()
    test2()
    t3 = time.time()
    print("pytorch 用时: {}; numpy 用时：{}".format(t2-t1, t3-t2))  


if __name__ == '__main__':
    main()
    pass
