import torch
import json
from scipy.optimize import linear_sum_assignment
import numpy as np
import random
import torch.optim as optim
from torch.autograd import Function
import matplotlib.pyplot as plt

manualSeed = 14030
beta1 = 0.5


class RowSinkLayer(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = x.clone()
        for i in range(x.size(0)):
            output[i] = row_norm_sink(output[i])
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        q = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        for i in range(q.size(0)):
            for j in range(q.size(1)):
                for k in range(q.size(2)):
                    derivative = 0
                    for y in range(q.size(1)):
                        curr = q[i][y][k]
                        a = 0
                        col_sum = torch.sum(q[i][:,k])

                        if y == j:
                            a = 1 / col_sum

                        b = grad_output[i][y][k] / (col_sum * col_sum)
                        derivative += curr * (a - b)

                    grad_input[i][j][k] = derivative
    
        return grad_input


class ColSinkLayer(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = x.clone()
        for i in range(x.size(0)):
            output[i] = col_norm_sink(output[i])
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        q = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        for i in range(q.size(0)):
            for j in range(q.size(1)):
                for k in range(q.size(2)):
                    derivative = 0
                    for x in range(q.size(2)):
                        curr = q[i][j][x]
                        a = 0
                        col_sum = torch.sum(q[i][x])

                        if x == k:
                            a = 1 / col_sum

                        b = grad_output[i][j][x] / (col_sum * col_sum)
                        derivative += curr * (a - b)

                    grad_input[i][j][k] = derivative
        
        return grad_input

class SinkLayer(torch.nn.Module):
    def __init__(self, iters):
        super(SinkLayer, self).__init__()
        self.iters = iters
        self.col_sink = ColSinkLayer.apply
        self.row_sink = RowSinkLayer.apply  

    def forward(self, x):
        output = x.clone() + 1E-3
        for i in range(self.iters):
            output = self.col_sink(self.row_sink(output))

        return output