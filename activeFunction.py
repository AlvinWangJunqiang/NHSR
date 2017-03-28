#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/3/26 20:29
# @Author  : ConanCui
# @Site    : 
# @File    : activeFunction.py
# @Software: PyCharm Community Edition
import numpy as np


def tanh(x):
    temp = np.exp(2*x)
    return( temp - 1)/( temp + 1)

def dtanh(x):
    return 1 - tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return ( 1 - sigmoid(x))*sigmoid(x)

class activationFunction:
    def __init__(self,gama ,beta ,type = 'linear' ):
        self.gama = gama
        self.beta = beta
        self.type = type
    # def fun(self, x):
    #     return (x)
    # def derivative(self, x):
    #     return 1
    def fun(self, x ):
        if self.type == 'linear':
            return (x)
        if self.type == 'tanh':
            return self.beta * tanh(self.gama*x)
        if self.type == 'sigmoid':
            return self.beta * sigmoid(self.gama*x)

    def derivative(self, x):
        if self.type == 'linear':
            return (1)
        if self.type == 'tanh':
            return self.beta * self.gama * dtanh(self.gama*x)
        if self.type == 'sigmoid':
            return self.beta * self.gama * dsigmoid(self.gama*x)












