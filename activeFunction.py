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

